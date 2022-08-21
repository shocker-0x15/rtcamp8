#include "renderer_kernel_common.h"

static constexpr bool debugVisualizeBaseColor = false;



CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRaySignature::set(&visibility);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void createRadianceQuery(
    const Point3D &positionInWorld, const Normal3D &normalInWorld, const Vector3D &scatteredDirInWorld,
    float roughness, const RGBSpectrum &diffuseReflectance, const RGBSpectrum &specularReflectance,
    RadianceQuery* query) {
    float phi, theta;
    query->position = plp.f->worldDimInfo.aabb.calcLocalCoordinates(positionInWorld);
    normalInWorld.toPolarYUp(&phi, &theta);
    query->normal_phi = phi;
    query->normal_theta = theta;
    scatteredDirInWorld.toPolarYUp(&phi, &theta);
    query->vOut_phi = phi;
    query->vOut_theta = theta;
    query->roughness = roughness;
    query->diffuseReflectance = diffuseReflectance;
    query->specularReflectance = specularReflectance;
}

template <bool withNRC, bool generateTrainingData>
CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex());
    uint32_t nrcBufIdx = plp.f->nrcBufferIndex;
    int2 imageSize = generateTrainingData ?
        *(plp.s->trainImageSize[nrcBufIdx]) : plp.s->imageSize;

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    Point3D rayOrigin;
    Vector3D rayDirection;
    float dirPDensity;
    {
        const PerspectiveCamera &camera = plp.f->camera;
        float px = (launchIndex.x + rng.getFloat0cTo1o()) / imageSize.x;
        float py = (launchIndex.y + rng.getFloat0cTo1o()) / imageSize.y;
        float vh = 2 * std::tan(camera.fovY * 0.5f);
        float vw = camera.aspect * vh;
        float sensorArea = vw * vh; // normalized

        rayOrigin = camera.position;
        Vector3D localRayDir = Vector3D(vw * (-0.5f + px), vh * (0.5f - py), -1).normalize();
        rayDirection = normalize(camera.orientation.toMatrix3x3() * localRayDir);
        dirPDensity = 1 / (pow3(std::fabs(localRayDir.z)) * sensorArea);
    }

    bool isUnbiasedTrainingPath;
    uint32_t prevTrainDataIndex;
    RGBSpectrum prevLocalThroughput;
    if constexpr (generateTrainingData) {
        const uint2 tileSize = make_uint2(4, 4);
        const uint32_t numPixelsInTile = tileSize.x * tileSize.y;
        uint2 localIndexInTile = launchIndex % tileSize;
        uint32_t tileLocalLinearIndex = localIndexInTile.y * tileSize.x + localIndexInTile.x;
        isUnbiasedTrainingPath =
            (tileLocalLinearIndex + plp.f->offsetToSelectUnbiasedPath) % numPixelsInTile == 0;

        prevTrainDataIndex = invalidVertexDataIndex;
    }
    else {
        (void)isUnbiasedTrainingPath;
        (void)prevTrainDataIndex;
        (void)prevLocalThroughput;
    }

    RGBSpectrum contribution = RGBSpectrum::Zero();
    RGBSpectrum throughput = RGBSpectrum::One() / dirPDensity;
    float initImportance = throughput.luminance();
    uint32_t pathLength = 0;
    float primaryPathSpread;
    float curSqrtPathSpread = 0.0f;
    while (true) {
        ++pathLength;

        uint32_t instSlot = 0xFFFFFFFF;
        uint32_t geomInstSlot = 0xFFFFFFFF;
        uint32_t primIndex = 0xFFFFFFFF;
        float b1 = 0.0f, b2 = 0.0f;
        float hitDist = 1e+10f;
        optixu::trace<ClosestRaySignature>(
            plp.f->travHandle,
            rayOrigin.toNativeType(), rayDirection.toNativeType(), 0.0f, 1e+10f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            PathTracingRayType::Closest, maxNumRayTypes, PathTracingRayType::Closest,
            instSlot, geomInstSlot, primIndex, b1, b2, hitDist);

        bool volEventHappens = false;
        if (plp.s->densityGrid) {
            const nvdb::FloatGrid* densityGrid = plp.s->densityGrid;
            const float densityCoeff = plp.f->densityCoeff;
            const float majorant = densityCoeff * plp.s->majorant;
            const nvdb::DefaultReadAccessor<float> &acc = densityGrid->getAccessor();
            const auto sampler = nvdb::createSampler<1, nvdb::DefaultReadAccessor<float>, false>(acc);
            auto map = densityGrid->map();

            nvdb::Ray<float> nvdbRay(
                nvdb::Vec3f(rayOrigin.x, rayOrigin.y, rayOrigin.z),
                nvdb::Vec3f(rayDirection.x, rayDirection.y, rayDirection.z),
                0.0f, hitDist);
            if (nvdbRay.clip(plp.s->densityGridBBox)) {
                float fpDist = std::fmax(0.0f, nvdbRay.t0());
                while (true) {
                    fpDist += -std::log(1.0f - rng.getFloat0cTo1o()) / majorant;
                    if (fpDist > nvdbRay.t1())
                        break;
                    nvdb::Vec3f evalP = nvdbRay(fpDist);
                    nvdb::Vec3f fIdx = densityGrid->worldToIndexF(evalP);
                    float density = densityCoeff * sampler(fIdx);
                    if (rng.getFloat0cTo1o() < density / majorant) {
                        volEventHappens = true;
                        hitDist = fpDist;
                        break;
                    }
                }
            }
        }

        // JP: 何にもヒットしなかった、環境光源にレイが飛んだ場合。
        // EN: 
        if (instSlot == 0xFFFFFFFF && !volEventHappens) {
            if (plp.f->enableEnvironmentalLight) {
                float hypAreaPDensity;
                RGBSpectrum radiance = plp.f->envLight.evaluate(rayDirection, &hypAreaPDensity);
                float misWeight = 1.0f;
                if (pathLength > 1) {
                    float lightPDensity =
                        (plp.f->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                        hypAreaPDensity;
                    float bsdfPDensity = dirPDensity;
                    misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                }
                Assert(radiance.allNonNegativeFinite() && rtc8::isfinite(misWeight),
                       "EnvMap Hit (l: %u): (%g, %g, %g), misW: %g",
                       pathLength, rgbprint(radiance), misWeight);
                RGBSpectrum implicitCont = radiance * misWeight;
                if constexpr (!generateTrainingData)
                    contribution += throughput * implicitCont;
                // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
                // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
                //     to the target value.
                if (prevTrainDataIndex != invalidVertexDataIndex)
                    plp.s->trainTargetBuffer[0][prevTrainDataIndex] += prevLocalThroughput * implicitCont;
            }

            uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
            if constexpr (generateTrainingData) {
                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = prevTrainDataIndex;
                terminalInfo.hasQuery = false;
                terminalInfo.pathLength = pathLength;
                terminalInfo.isUnbiasedPath = isUnbiasedTrainingPath;
                plp.s->trainSuffixTerminalInfoBuffer[linearIndex] = terminalInfo;
            }
            else {
                if constexpr (withNRC) {
                    TerminalInfo terminalInfo;
                    terminalInfo.throughput = throughput;
                    terminalInfo.hasQuery = false;
                    terminalInfo.pathLength = pathLength;
                    plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
                }
            }
            break;
        }

        InteractionPoint interPt;
        uint32_t surfMatSlot = 0xFFFFFFFF;
        if (volEventHappens) {
            interPt.position = rayOrigin + hitDist * rayDirection;
            interPt.shadingFrame = ReferenceFrame(-rayDirection);
            interPt.inMedium = true;
        }
        else {
            const Instance &inst = plp.f->instances[instSlot];
            const GeometryInstance &geomInst = plp.s->geometryInstances[geomInstSlot];
            computeSurfacePoint(inst, geomInst, primIndex, b1, b2, &interPt);

            surfMatSlot = geomInst.surfMatSlot;

            if (geomInst.normal) {
                Normal3D modLocalNormal = geomInst.readModifiedNormal(
                    geomInst.normal, geomInst.normalDimInfo, interPt.asSurf.texCoord);
                applyBumpMapping(modLocalNormal, &interPt.shadingFrame);
            }
        }

        Vector3D vOut(-rayDirection);
        Vector3D vOutLocal = interPt.toLocal(vOut);

        if constexpr (withNRC) {
            float dist2 = squaredDistance(rayOrigin, interPt.position);
            float absDotVN = interPt.calcAbsDot(rayDirection);
            if (pathLength == 1)
                primaryPathSpread = dist2 / (4 * pi_v<float> * absDotVN);
            else
                curSqrtPathSpread += std::sqrt(dist2 / (dirPDensity * absDotVN));
            Assert(isfinite(curSqrtPathSpread),
                   "Invalid path spread: %g (d2: %g, p: %g, dot: %g",
                   curSqrtPathSpread, dist2, dirPDensity, absDotVN);
        }
        else {
            (void)primaryPathSpread;
            (void)curSqrtPathSpread;
        }

        // Implicit light sampling
        if (!volEventHappens) {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            if (vOutLocal.z > 0 && surfMat.emittance) {
                float4 texValue = tex2DLod<float4>(
                    surfMat.emittance, interPt.asSurf.texCoord.u, interPt.asSurf.texCoord.v, 0.0f);
                RGBSpectrum emittance(texValue.x, texValue.y, texValue.z);
                float misWeight = 1.0f;
                if (pathLength > 1) {
                    float dist2 = squaredDistance(rayOrigin, interPt.position);
                    float lightPDensity = interPt.asSurf.hypAreaPDensity * dist2 / vOutLocal.z;
                    float bsdfPDensity = dirPDensity;
                    misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                }
                // Assume a diffuse emitter.
                RGBSpectrum implicitCont = emittance * (misWeight / pi_v<float>);
                if constexpr (!generateTrainingData)
                    contribution += throughput * implicitCont;

                if constexpr (generateTrainingData) {
                    // JP: 1つ前の頂点に対する直接照明(Implicit)によるScattered Radianceをターゲット値に加算。
                    // EN: Accumulate scattered radiance at the previous vertex by direct lighting (implicit)
                    //     to the target value.
                    if (prevTrainDataIndex != invalidVertexDataIndex)
                        plp.s->trainTargetBuffer[0][prevTrainDataIndex] += prevLocalThroughput * implicitCont;
                }
            }
        }

        // Russian roulette
        /*
        withNRC genData Exit Condition
              0       0     RR (l > 1)
              0       1     RR (l > 2)
              1       0     RR (l > 1) or cache
              1       1     RR (l > 2) or cache (biased)
              1       1     RR (l > 2)          (unbiased)
        */
        float continueProb = 1.0f;
        bool performRR = true;
        if constexpr (generateTrainingData)
            performRR = pathLength > 2;
        bool terminatedByRR = false;
        if (performRR) {
            continueProb = std::fmin(throughput.luminance() / initImportance, 1.0f);
            if (rng.getFloat0cTo1o() >= continueProb)
                terminatedByRR = true;
        }
        // JP: キャッシュがある場合はRRで終わるときもキャッシュクエリーを発行して終わる。
        if constexpr (!withNRC) {
            if (terminatedByRR)
                break;
        }

        BSDF bsdf;
        BSDFQuery bsdfQuery;
        if (volEventHappens) {
            bsdf.setup(plp.f->scatteringAlbedo);
            bsdfQuery = BSDFQuery();
        }
        else {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            bsdf.setup(surfMat, interPt.asSurf.texCoord);
            bsdfQuery = BSDFQuery(vOutLocal, interPt.toLocal(interPt.asSurf.geometricNormal));
        }

        if constexpr (withNRC) {
            bool endsWidthCache = terminatedByRR;
            if constexpr (generateTrainingData) {
                if (!isUnbiasedTrainingPath)
                    endsWidthCache |= pow2(curSqrtPathSpread) > 2 * pathTerminationFactor * primaryPathSpread;
            }
            else {
                endsWidthCache |= pow2(curSqrtPathSpread) > pathTerminationFactor * primaryPathSpread;
            }

            if (endsWidthCache) {
                uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;

                float roughness;
                RGBSpectrum diffuseReflectance, specularReflectance;
                bsdf.getSurfaceParameters(
                    &diffuseReflectance, &specularReflectance, &roughness);

                // JP: Radianceクエリーのための情報を記録する。
                // EN: Store information for radiance query.
                RadianceQuery radQuery;
                createRadianceQuery(
                    interPt.position, interPt.shadingFrame.normal, vOut,
                    roughness, diffuseReflectance, specularReflectance,
                    &radQuery);

                plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;

                if constexpr (generateTrainingData) {
                    // JP: 直前のTraining VertexへのリンクとともにTraining Suffixを終了させる。
                    // EN: Finish the training suffix with the link to the previous training vertex.
                    TrainingSuffixTerminalInfo terminalInfo;
                    terminalInfo.prevVertexDataIndex = prevTrainDataIndex;
                    terminalInfo.hasQuery = true;
                    terminalInfo.pathLength = pathLength;
                    terminalInfo.isUnbiasedPath = isUnbiasedTrainingPath;
                    plp.s->trainSuffixTerminalInfoBuffer[linearIndex] = terminalInfo;
                }
                else {
                    TerminalInfo terminalInfo;
                    terminalInfo.throughput = throughput;
                    terminalInfo.hasQuery = true;
                    terminalInfo.pathLength = pathLength;
                    plp.s->inferenceTerminalInfoBuffer[linearIndex] = terminalInfo;
                }

                break;
            }
        }

        throughput /= continueProb;
        if constexpr (debugVisualizeBaseColor) {
            contribution += throughput * bsdf.evaluateDHReflectanceEstimate(bsdfQuery);
            break;
        }
        RGBSpectrum neeCont = performNextEventEstimation(interPt, bsdf, bsdfQuery, rng);
        if constexpr (!generateTrainingData)
            contribution += throughput * neeCont;

        BSDFSample bsdfSample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult bsdfResult;
        RGBSpectrum fsValue = bsdf.sampleF(bsdfQuery, bsdfSample, &bsdfResult);
        if (bsdfResult.dirPDensity == 0.0f)
            break; // sampling failed.
        rayDirection = interPt.fromLocal(bsdfResult.dirLocal);
        float dotSGN = interPt.calcDot(rayDirection);
        RGBSpectrum localThroughput = fsValue * (std::fabs(dotSGN) / bsdfResult.dirPDensity);
        throughput *= localThroughput;
        Assert(
            localThroughput.allNonNegativeFinite(),
            "tp: (%g, %g, %g), dotSGN: %g, dirP: %g",
            rgbprint(localThroughput), dotSGN, bsdfResult.dirPDensity);

        rayOrigin = interPt.calcOffsetRayOrigin(dotSGN > 0.0f);
        dirPDensity = bsdfResult.dirPDensity;

        if constexpr (generateTrainingData) {
            uint32_t trainDataIndex = atomicAdd(plp.s->numTrainingData[nrcBufIdx], 1u);

            float roughness;
            RGBSpectrum diffuseReflectance, specularReflectance;
            bsdf.getSurfaceParameters(
                &diffuseReflectance, &specularReflectance, &roughness);

            RadianceQuery radQuery;
            createRadianceQuery(
                interPt.position, interPt.shadingFrame.normal, vOut,
                roughness, diffuseReflectance, specularReflectance,
                &radQuery);

            if (trainDataIndex < trainBufferSize) {
                plp.s->trainRadianceQueryBuffer[0][trainDataIndex] = radQuery;

                // JP: ローカルスループットと前のTraining Vertexへのリンクを記録。
                // EN: Record the local throughput and the link to the previous training vertex.
                TrainingVertexInfo vertInfo;
                vertInfo.localThroughput = localThroughput;
                vertInfo.prevVertexDataIndex = prevTrainDataIndex;
                vertInfo.pathLength = pathLength;
                vertInfo.isUnbiasedPath = isUnbiasedTrainingPath;
                plp.s->trainVertexInfoBuffer[trainDataIndex] = vertInfo;

                // JP: 現在の頂点に対する直接照明(NEE)によるScattered Radianceでターゲット値を初期化。
                // EN: Initialize a target value by scattered radiance at the current vertex
                //     by direct lighting (NEE).
                plp.s->trainTargetBuffer[0][trainDataIndex] = neeCont;
                //if (!neeCont.allFinite())
                //    printf("NEE: (%g, %g, %g)\n",
                //           rgbprint(neeCont));

                prevLocalThroughput = localThroughput;
                prevTrainDataIndex = trainDataIndex;
            }
            // JP: 訓練データがバッファーを溢れた場合は強制的にTraining Suffixを終了させる。
            // EN: Forcefully end the training suffix if the training data buffer become full.
            else {
                uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
                plp.s->inferenceRadianceQueryBuffer[linearIndex] = radQuery;

                TrainingSuffixTerminalInfo terminalInfo;
                terminalInfo.prevVertexDataIndex = prevTrainDataIndex;
                terminalInfo.hasQuery = true;
                terminalInfo.pathLength = pathLength;
                terminalInfo.isUnbiasedPath = isUnbiasedTrainingPath;
                plp.s->trainSuffixTerminalInfoBuffer[linearIndex] = terminalInfo;

                break;
            }
        }
    }

    plp.s->rngBuffer.write(launchIndex, rng);

    if constexpr (!generateTrainingData) {
        if (!contribution.allNonNegativeFinite()) {
            printf("Store Cont.: %4u, %4u: (%g, %g, %g)\n",
                   launchIndex.x, launchIndex.y, rgbprint(contribution));
            return;
        }

        if constexpr (withNRC) {
            plp.s->perFrameContributionBuffer.write(launchIndex, contribution);
        }
        else {
            RGBSpectrum prevResult = RGBSpectrum::Zero();
            if (plp.f->numAccumFrames > 0)
                prevResult = plp.s->accumBuffer.read(launchIndex);
            float curFrameWeight = 1.0f / (plp.f->numAccumFrames + 1);
            RGBSpectrum result = (1 - curFrameWeight) * prevResult + curFrameWeight * contribution;
            plp.s->accumBuffer.write(launchIndex, result);
        }
    }
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTrace)() {
    pathTrace_generic<false, false>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(generateTrainingData)() {
    pathTrace_generic<true, true>();
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTraceWithNRC)() {
    pathTrace_generic<true, false>();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTrace)() {
    uint32_t instSlot = optixGetInstanceId();
    auto sbtr = HitGroupSBTRecordData::get();
    auto hp = HitPointParameter::get();
    float dist = optixGetRayTmax();

    ClosestRaySignature::set(&instSlot, &sbtr.geomInstSlot, &hp.primIndex, &hp.b1, &hp.b2, &dist);
}
