#pragma once

#include "../common/common_renderer_types.h"

namespace rtc8::shared {

static constexpr float probToSampleEnvLight = 0.25f;



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



struct StaticPipelineLaunchParameters {
    BSDFProcedureSet* bsdfProcedureSets;
    SurfaceMaterial* surfaceMaterials;
    GeometryInstance* geometryInstances;
    GeometryGroup* geometryGroups;

    int2 imageSize;
    optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;
    optixu::NativeBlockBuffer2D<RGBSpectrum> accumBuffer;
};

struct PerFramePipelineLaunchParameters {
    Instance* instances;

    EnvironmentalLight envLight;
    Point3D worldCenter;
    float worldRadius;
    float envLightPowerCoeff;
    float envLightRotation;
    LightDistribution lightInstDist;

    optixu::NativeBlockBuffer2D<RGBSpectrum> outputBuffer;

    int2 mousePosition;

    OptixTraversableHandle travHandle;
    PerspectiveCamera camera;
    uint32_t numAccumFrames;
    uint32_t enableDebugPrint : 1;
};

struct PipelineLaunchParameters {
    const StaticPipelineLaunchParameters* s;
    const PerFramePipelineLaunchParameters* f;
};



using ClosestRaySignature = optixu::PayloadSignature<
    uint32_t, uint32_t, uint32_t, float, float>;
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

} // namespace rtc8::device

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)