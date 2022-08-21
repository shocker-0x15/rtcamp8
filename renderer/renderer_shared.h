#pragma once

#include "../common/common_renderer_types.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/SampleFromVoxels.h>

namespace rtc8::shared {

static constexpr float probToSampleEnvLight = 0.25f;
static constexpr float pathTerminationFactor = 0.01f;
static constexpr uint32_t numTrainingDataPerFrame = 1 << 16;
static constexpr uint32_t trainBufferSize = 2 * numTrainingDataPerFrame;
static constexpr bool debugTrainingDataShuffle = false;



struct PathTracingRayType {
    enum Value {
        Closest,
        Visibility,
        NumTypes
    } value;

    CUDA_DEVICE_FUNCTION constexpr PathTracingRayType(Value v = Closest) : value(v) {}

    CUDA_DEVICE_FUNCTION operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

constexpr uint32_t maxNumRayTypes = 2;



struct LightSample {
    RGBSpectrum emittance;
    Point3D position;
    Normal3D normal;
    uint32_t atInfinity : 1;
};




struct RadianceQuery {
    Point3D position;
    float normal_phi;
    float normal_theta;
    float vOut_phi;
    float vOut_theta;
    float roughness;
    RGBSpectrum diffuseReflectance;
    RGBSpectrum specularReflectance;

    CUDA_DEVICE_FUNCTION bool isValid() const {
        return
            position.allFinite() &&
            isfinite(normal_phi) && isfinite(normal_theta) &&
            isfinite(vOut_phi) && isfinite(vOut_theta) &&
            isfinite(roughness) &&
            diffuseReflectance.allFinite() &&
            specularReflectance.allFinite();
    }
};

struct TerminalInfo {
    RGBSpectrum throughput;
    uint32_t hasQuery : 1;
    // for stats/debug
    uint32_t pathLength : 8;
};

static constexpr uint32_t invalidVertexDataIndex = 0x003FFFFF;

struct TrainingVertexInfo {
    RGBSpectrum localThroughput;
    uint32_t prevVertexDataIndex : 22;
    // for stats/debug
    uint32_t pathLength : 8;
    uint32_t isUnbiasedPath : 1;
};

struct TrainingSuffixTerminalInfo {
    uint32_t prevVertexDataIndex : 22;
    uint32_t hasQuery : 1;
    // for stats/debug
    uint32_t pathLength : 8;
    uint32_t isUnbiasedPath : 1;
};

class LinearCongruentialGenerator {
    static constexpr uint32_t a = 1103515245;
    static constexpr uint32_t c = 12345;
    static constexpr uint32_t m = 1u << 31;
    uint32_t m_state;

public:
    LinearCongruentialGenerator() : m_state(0) {}

    CUDA_COMMON_FUNCTION void setState(uint32_t seed) {
        m_state = seed;
    }

    CUDA_COMMON_FUNCTION uint32_t next() {
        m_state = ((m_state * a) + c) % m;
        return m_state;
    }
};



struct StaticPipelineLaunchParameters {
    BSDFProcedureSet* bsdfProcedureSets;
    SurfaceMaterial* surfaceMaterials;
    GeometryInstance* geometryInstances;
    GeometryGroup* geometryGroups;

    int2 imageSize;
    optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;
    optixu::NativeBlockBuffer2D<RGBSpectrum> accumBuffer;

    nanovdb::FloatGrid* densityGrid;
    nanovdb::BBox<nanovdb::Vec3f> densityGridBBox;
    float majorant;

    uint32_t maxNumTrainingSuffixes;
    uint32_t* numTrainingData[2];
    int2* trainImageSize[2];
    RGBSpectrumAsOrderedInt* targetMinMax[2][2];
    RGBSpectrum* targetAvg[2];
    RadianceQuery* inferenceRadianceQueryBuffer; // image size or #(training suffix)
    TerminalInfo* inferenceTerminalInfoBuffer; // image size
    TrainingSuffixTerminalInfo* trainSuffixTerminalInfoBuffer; // #(training suffix)
    RGBSpectrum* inferredRadianceBuffer; // image size or #(training suffix)
    optixu::NativeBlockBuffer2D<RGBSpectrum> perFrameContributionBuffer; // image size
    RadianceQuery* trainRadianceQueryBuffer[2]; // #(training vertex)
    RGBSpectrum* trainTargetBuffer[2]; // #(training vertex)
    TrainingVertexInfo* trainVertexInfoBuffer; // #(training vertex)
    TrainingVertexInfo* shuffledTrainVertexInfoBuffer; // #(training vertex), only for debug
    LinearCongruentialGenerator* dataShufflerBuffer; // numTrainingDataPerFrame
};

struct PerFramePipelineLaunchParameters {
    Instance* instances;

    EnvironmentalLight envLight;
    WorldDimInfo worldDimInfo;
    LightDistribution lightInstDist;

    optixu::NativeBlockBuffer2D<RGBSpectrum> outputBuffer;

    int2 mousePosition;

    uint32_t offsetToSelectUnbiasedPath;
    uint32_t nrcBufferIndex;
    float densityCoeff;
    float scatteringAlbedo;
    float scatteringForwardness;
    float radianceScale;

    OptixTraversableHandle travHandle;
    PerspectiveCamera camera;
    uint32_t numAccumFrames;
    uint32_t enableDebugPrint : 1;
    uint32_t enableEnvironmentalLight : 1;
};

struct PipelineLaunchParameters {
    StaticPipelineLaunchParameters* s;
    PerFramePipelineLaunchParameters* f;
};



using ClosestRaySignature = optixu::PayloadSignature<
    uint32_t, uint32_t, uint32_t, float, float, float>;
using VisibilityRaySignature = optixu::PayloadSignature<float>;

} // namespace rtc8::shared



#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM rtc8::shared::PipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS rtc8::shared::PipelineLaunchParameters plp;
#endif

namespace rtc8::device {

CUDA_DEVICE_FUNCTION CUDA_INLINE int2 getMousePosition() {
    return plp.f->mousePosition;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled() {
    return plp.f->enableDebugPrint;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE const shared::BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot) {
    return plp.s->bsdfProcedureSets[slot];
}

CUDA_DEVICE_FUNCTION CUDA_INLINE const float &getScatteringForwardness() {
    return plp.f->scatteringForwardness;
}

} // namespace rtc8::device

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)