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


struct PerspectiveCamera {
    float aspect;
    float fovY;
    Point3D position;
    Matrix3x3 orientation;

    CUDA_COMMON_FUNCTION float2 calcScreenPosition(const Point3D &posInWorld) const {
        Matrix3x3 invOri = invert(orientation);
        Vector3D posInView = invOri * (posInWorld - position);
        float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
        float h = 2 * std::tan(fovY / 2);
        float w = aspect * h;
        return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                            1 - (posAtZ1.y + 0.5f * h) / h);
    }
};



struct LightSample {
    RGBSpectrum emittance;
    Point3D position;
    Normal3D normal;
    unsigned int atInfinity : 1;
};



struct StaticPipelineLaunchParameters {
    GeometryInstance* geometryInstances;
    BSDFProcedureSet* bsdfProcedureSets;
    SurfaceMaterial* surfaceMaterials;

    int2 imageSize;
    optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;
};

struct PerFramePipelineLaunchParameters {
    Instance* instances;

    EnvironmentalLight envLight;
    Point3D worldCenter;
    float worldRadius;
    float envLightPowerCoeff;
    float envLightRotation;
    LightDistribution lightInstDist;

    OptixTraversableHandle travHandle;
    PerspectiveCamera camera;
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

CUDA_DEVICE_FUNCTION CUDA_INLINE const BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot) {
    return plp.s->bsdfProcedureSets[slot];
}

} // namespace rtc8::device

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)