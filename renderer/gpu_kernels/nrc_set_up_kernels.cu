#define PURE_CUDA
#include "renderer_kernel_common.h"

CUDA_DEVICE_KERNEL void prepareNRC(
    uint32_t offsetToSelectUnbiasedPath,
    bool isNewSequence) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex >= plp.s->maxNumTrainingSuffixes)
        return;

    uint32_t prevBufIdx = (plp.f->nrcBufferIndex + 1) % 2;
    uint32_t bufIdx = plp.f->nrcBufferIndex;
    if (linearIndex == 0) {
        // JP: �O��Ő������ꂽ�P���f�[�^���Ɋ�Â��ăg���[�j���O�����_�����O�T�C�Y�𒲐�����B
        // EN: Adjust training rendering size based on the number of training data generated
        //     in the previous frame.
        int2 newTrainImageSize;
        if (isNewSequence) {
            newTrainImageSize = plp.s->imageSize / 4;
        }
        else {
            uint32_t prevNumTrainingData = *(plp.s->numTrainingData[prevBufIdx]);
            float r = std::sqrt(static_cast<float>(prevNumTrainingData) / numTrainingDataPerFrame);
            int2 curTrainImageSize = *(plp.s->trainImageSize[prevBufIdx]);
            newTrainImageSize = make_int2(min(
                make_float2(curTrainImageSize.x / r, curTrainImageSize.y / r),
                make_float2(plp.s->imageSize)));
        }
        *(plp.s->trainImageSize[bufIdx]) = newTrainImageSize;

        *(plp.s->numTrainingData[bufIdx]) = 0;
        plp.f->offsetToSelectUnbiasedPath = offsetToSelectUnbiasedPath;

        *(plp.s->targetMinMax[bufIdx][0]) = RGBSpectrumAsOrderedInt(RGBSpectrum::Infinity()); // min
        *(plp.s->targetMinMax[bufIdx][1]) = RGBSpectrumAsOrderedInt(-RGBSpectrum::Infinity()); // max
        *(plp.s->targetAvg[bufIdx]) = RGBSpectrum::Zero();
    }

    TrainingSuffixTerminalInfo terminalInfo;
    terminalInfo.prevVertexDataIndex = invalidVertexDataIndex;
    terminalInfo.hasQuery = false;
    terminalInfo.pathLength = 0;
    plp.s->trainSuffixTerminalInfoBuffer[linearIndex] = terminalInfo;
}

CUDA_DEVICE_KERNEL void propagateRadianceValues() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex >= plp.s->maxNumTrainingSuffixes)
        return;

    const TrainingSuffixTerminalInfo &terminalInfo = plp.s->trainSuffixTerminalInfoBuffer[linearIndex];
    if (terminalInfo.prevVertexDataIndex == invalidVertexDataIndex)
        return;

    RGBSpectrum contribution = RGBSpectrum::Zero();
    if (terminalInfo.hasQuery) {
        RGBSpectrum inferredValue = plp.s->inferredRadianceBuffer[linearIndex];
        Assert(inferredValue.allFinite(),
               "Invalid inferred radiance value (%g, %g, %g)",
               rgbprint(inferredValue));
        contribution = max(inferredValue, RGBSpectrum::Zero());
        if (plp.f->radianceScale > 0)
            contribution /= plp.f->radianceScale;

        const RadianceQuery &terminalQuery = plp.s->inferenceRadianceQueryBuffer[linearIndex];
        contribution *= (terminalQuery.diffuseReflectance + terminalQuery.specularReflectance);
    }

    // JP: �eTraining Vertex�̃��[�J���X���[�v�b�g���悶�Ȃ���ċA�I�Ƀl�b�g���[�N����^����ꂽ�P�x��
    //     �`�d�����邱�ƂŌP���f�[�^������������B
    // EN: Recursively propagate the radiance value from the network while multiplying a local throughput
    //     at each training vertex to complete training data.
    uint32_t lastTrainDataIndex = terminalInfo.prevVertexDataIndex;
    while (lastTrainDataIndex != invalidVertexDataIndex) {
        const TrainingVertexInfo &vertexInfo = plp.s->trainVertexInfoBuffer[lastTrainDataIndex];
        RGBSpectrum &targetValue = plp.s->trainTargetBuffer[0][lastTrainDataIndex];
        RGBSpectrum indirectCont = vertexInfo.localThroughput * contribution;
        contribution = targetValue + indirectCont;

        const RadianceQuery &query = plp.s->trainRadianceQueryBuffer[0][lastTrainDataIndex];
        RGBSpectrum refFactor = query.diffuseReflectance + query.specularReflectance;
        targetValue = safeDivide(contribution, refFactor);

        lastTrainDataIndex = vertexInfo.prevVertexDataIndex;
    }
}

CUDA_DEVICE_KERNEL void shuffleTrainingData() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t bufIdx = plp.f->nrcBufferIndex;

    uint32_t numTrainingData = *(plp.s->numTrainingData[bufIdx]);
    if (numTrainingData > 0) {
        LinearCongruentialGenerator &shuffler = plp.s->dataShufflerBuffer[linearIndex];
        static_assert((numTrainingDataPerFrame & (numTrainingDataPerFrame - 1)) == 0,
                      "The number of traing data is assumed to be the power of 2 here.");
        uint32_t dstIdx = shuffler.next() % numTrainingDataPerFrame;

        // JP: �p�X�g���[�T�[�����������T���v����������Ȃ��Ƃ��̓��b�v�A���E���h����B
        // EN: Wrap around for the case where the path tracer generates too few samples.
        uint32_t srcIdx = linearIndex % numTrainingData;
        RadianceQuery query = plp.s->trainRadianceQueryBuffer[0][srcIdx];
        RGBSpectrum targetValue = plp.s->trainTargetBuffer[0][srcIdx];
        TrainingVertexInfo vertInfo;
        if constexpr (debugTrainingDataShuffle)
            vertInfo = plp.s->trainVertexInfoBuffer[srcIdx];
        else
            (void)vertInfo;

        if (!query.isValid()) {
            printf("p: (%g, %g, %g), n: (%g, %g), v: (%g, %g), "
                   "r: %g, d: (%g, %g, %g), s: (%g, %g, %g)\n",
                   vec3print(query.position),
                   query.normal_phi, query.normal_theta,
                   query.vOut_phi, query.vOut_theta,
                   query.roughness,
                   rgbprint(query.diffuseReflectance),
                   rgbprint(query.specularReflectance));
            query.position = Point3D::Zero();
            query.normal_phi = 0.0f;
            query.normal_theta = 0.0f;
            query.vOut_phi = 0.0f;
            query.vOut_theta = 0.0f;
            query.roughness = 0.0f;
            query.diffuseReflectance = query.specularReflectance = RGBSpectrum::Zero();
        }
        if (!targetValue.allNonNegativeFinite()) {
            printf("tgt: (%g, %g, %g)\n", rgbprint(targetValue));
            targetValue = RGBSpectrum::Zero();
        }

        CUDA_SHARED_MEM uint32_t sm_pool[3 * sizeof(RGBSpectrum) / sizeof(uint32_t)];
        auto &sm_minRadiance = reinterpret_cast<RGBSpectrumAsOrderedInt &>(sm_pool[0]);
        auto &sm_maxRadiance = reinterpret_cast<RGBSpectrumAsOrderedInt &>(sm_pool[3]);
        auto &sm_avgRadiance = reinterpret_cast<RGBSpectrum &>(sm_pool[6]);
        if (threadIdx.x == 0) {
            sm_minRadiance = RGBSpectrumAsOrderedInt(RGBSpectrum::Infinity());
            sm_maxRadiance = RGBSpectrumAsOrderedInt(-RGBSpectrum::Infinity());
            sm_avgRadiance = RGBSpectrum::Zero();
        }
        __syncthreads();
        atomicMin_RGBSpectrum_block(&sm_minRadiance, targetValue);
        atomicMax_RGBSpectrum_block(&sm_maxRadiance, targetValue);
        atomicAdd_RGBSpectrum_block(&sm_avgRadiance, targetValue / numTrainingDataPerFrame);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicMin_RGBSpectrum(plp.s->targetMinMax[bufIdx][0], sm_minRadiance);
            atomicMax_RGBSpectrum(plp.s->targetMinMax[bufIdx][1], sm_maxRadiance);
            atomicAdd_RGBSpectrum(plp.s->targetAvg[bufIdx], sm_avgRadiance);
        }

        // JP: ���X�֐��̌v�Z�ɂ���[�����Z��h�����߂̃C�v�V�������x�z�I�ɂȂ�Ȃ��悤�A
        //     �l�b�g���[�N�ɓ��͂���l�̃X�P�[���𒲐�����K�v������B
        // EN: Adjusting the scale of the input values to the network is required so that
        //     the epsilon to avoid division by zero in the loss function calculation does not dominate.
        if (plp.f->radianceScale > 0)
            targetValue *= plp.f->radianceScale;
        targetValue = min(targetValue, RGBSpectrum(1e+6f));

        plp.s->trainRadianceQueryBuffer[1][dstIdx] = query;
        plp.s->trainTargetBuffer[1][dstIdx] = targetValue;
        if constexpr (debugTrainingDataShuffle)
            plp.s->shuffledTrainVertexInfoBuffer[dstIdx] = vertInfo;
    }
    else {
        RadianceQuery query = {};
        plp.s->trainRadianceQueryBuffer[1][linearIndex] = query;
        plp.s->trainTargetBuffer[1][linearIndex] = RGBSpectrum::Zero();
        if constexpr (debugTrainingDataShuffle) {
            TrainingVertexInfo vertInfo = {};
            plp.s->shuffledTrainVertexInfoBuffer[linearIndex] = vertInfo;
        }
    }
}

CUDA_DEVICE_KERNEL void accumulateInferredRadianceValues() {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t numPixels = plp.s->imageSize.x * plp.s->imageSize.y;
    if (linearIndex >= numPixels)
        return;

    const TerminalInfo &terminalInfo = plp.s->inferenceTerminalInfoBuffer[linearIndex];

    // JP: Rendering Path�͂��̎��_�ł͏I�[�O�̊e���_�ɂ����銮�S�Ȑ���l(���ڌ� + �Ԑڌ�)�̗ݐϒl��
    //     �I�[�_�ɂ����钼�ڌ��̐���l��ݐς��Ă���B�l�b�g���[�N���琄�肳�ꂽ�I�[�_�ɂ�����Ԑڌ�
    //     ���X���[�v�b�g���悶�ėݐς��邱�ƂŃs�N�Z��������������B
    //     �p�X�����V�A�����[���b�g�ŏI�������ꍇ�△�����ɔ�񂾏ꍇ�̓l�b�g���[�N�̐���͎g���Ȃ��B
    // EN: Each rendering path have accumulated complete estimates (direct + indirect light) at vertices
    //     preceded to the terminal and a direct light estimate at the terminal so far.
    //     Accumulate the predicted indirect light from the network multiplied by a throughput to
    //     complete a pixel.
    //     Network prediction is not used in the case where the path ended with Russian roulette or traced
    //     to infinity.
    uint2 pixelIndex = make_uint2(linearIndex % plp.s->imageSize.x,
                                  linearIndex / plp.s->imageSize.x);
    RGBSpectrum directCont = plp.s->perFrameContributionBuffer.read(pixelIndex);
    RGBSpectrum radiance = RGBSpectrum::Zero();
    if (terminalInfo.hasQuery) {
        RGBSpectrum inferredValue = plp.s->inferredRadianceBuffer[linearIndex];
        Assert(inferredValue.allFinite(),
               "Invalid inferred radiance value (%g, %g, %g)",
               rgbprint(inferredValue));
        radiance = max(inferredValue, RGBSpectrum::Zero());
        if (plp.f->radianceScale > 0)
            radiance /= plp.f->radianceScale;

        const RadianceQuery &terminalQuery = plp.s->inferenceRadianceQueryBuffer[linearIndex];
        radiance *= (terminalQuery.diffuseReflectance + terminalQuery.specularReflectance);
    }
    RGBSpectrum indirectCont = terminalInfo.throughput * radiance;
    RGBSpectrum contribution = directCont + indirectCont;

    RGBSpectrum prevResult = RGBSpectrum::Zero();
    if (plp.f->numAccumFrames > 0)
        prevResult = plp.s->accumBuffer.read(pixelIndex);
    float curFrameWeight = 1.0f / (plp.f->numAccumFrames + 1);
    RGBSpectrum result = (1 - curFrameWeight) * prevResult + curFrameWeight * contribution;
    plp.s->accumBuffer.write(pixelIndex, result);
}
