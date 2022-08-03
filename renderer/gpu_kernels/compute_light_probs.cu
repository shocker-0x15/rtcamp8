#define PURE_CUDA
#include "../../common/common_renderer_types.h"

using namespace rtc8;
using namespace rtc8::shared;
using namespace rtc8::device;

CUDA_DEVICE_FUNCTION CUDA_INLINE float computeTriangleImportance(
    GeometryInstance* geomInst, uint32_t triIndex,
    const SurfaceMaterial* materialBuffer) {
    const SurfaceMaterial &mat = materialBuffer[geomInst->surfMatSlot];
    const Triangle &tri = geomInst->triangles[triIndex];
    const Vertex (&v)[3] = {
        geomInst->vertices[tri.indices[0]],
        geomInst->vertices[tri.indices[1]],
        geomInst->vertices[tri.indices[2]]
    };

    Normal3D normal = cross(v[1].position - v[0].position, v[2].position - v[0].position);
    float area = 0.5f * normal.length();

    const auto sample = []
    (CUtexObject texture, const TexCoord2D &texCoord) {
        float4 texValue = tex2DLod<float4>(texture, texCoord.u, texCoord.v, 0);
        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    };

    // TODO: もっと正確な、少なくとも保守的な推定の実装。テクスチャー空間中の面積に応じてMIPレベルを選択する？
    RGBSpectrum emittanceEstimate = RGBSpectrum::Zero();
    emittanceEstimate += sample(mat.emittance, v[0].texCoord);
    emittanceEstimate += sample(mat.emittance, v[1].texCoord);
    emittanceEstimate += sample(mat.emittance, v[2].texCoord);
    emittanceEstimate /= 3;

    float importance = emittanceEstimate.luminance() * area;
    Assert(rtc8::isfinite(importance), "imp: %g, area", importance, area);
    //printf("area: %g, %p, %g\n", area, mat.emittance, importance);
    return importance;
}

CUDA_DEVICE_KERNEL void computeTriangleProbBuffer(
    GeometryInstance* geomInst, uint32_t numTriangles,
    const SurfaceMaterial* materialBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        geomInst->emitterPrimDist.setNumValues(numTriangles);
    if (linearIndex < numTriangles) {
        float importance = computeTriangleImportance(geomInst, linearIndex, materialBuffer);
        geomInst->emitterPrimDist.setWeightAt(linearIndex, importance);
    }
}



CUDA_DEVICE_FUNCTION CUDA_INLINE float computeGeomInstImportance(
    GeometryGroup* geomGroup,
    const GeometryInstance* geometryInstanceBuffer, uint32_t geomInstIndex) {
    uint32_t slot = geomGroup->geomInstSlots[geomInstIndex];
    const GeometryInstance &geomInst = geometryInstanceBuffer[slot];
    float importance = geomInst.emitterPrimDist.integral();
    return importance;
}

CUDA_DEVICE_KERNEL void computeGeomInstProbBuffer(
    GeometryGroup* geomGroup, uint32_t numGeomInsts,
    const GeometryInstance* geometryInstanceBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        geomGroup->lightGeomInstDist.setNumValues(numGeomInsts);
    if (linearIndex < numGeomInsts) {
        float importance = computeGeomInstImportance(geomGroup, geometryInstanceBuffer, linearIndex);
        geomGroup->lightGeomInstDist.setWeightAt(linearIndex, importance);
    }
}



// TODO: instSlot?
CUDA_DEVICE_FUNCTION CUDA_INLINE float computeInstImportance(
    const Instance &inst, const GeometryGroup* geomGroupBuffer) {
    Vector3D scale;
    inst.transform.mat.decompose(&scale, nullptr, nullptr);
    float uniformScale = scale.x;
    const GeometryGroup &geomGroup = geomGroupBuffer[inst.geomGroupSlot];
    float importance = pow2(uniformScale) * geomGroup.lightGeomInstDist.integral();
    //printf("scale: %g, groupSlot: %u, imp: %g\n", uniformScale, inst.geomGroupSlot, importance);
    return importance;
}

CUDA_DEVICE_KERNEL void computeInstProbBuffer(
    DiscreteDistribution1D* lightInstDist, uint32_t numInsts,
    const GeometryGroup* geomGroupBuffer, const Instance* instanceBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        lightInstDist->setNumValues(numInsts);
    if (linearIndex < numInsts) {
        const Instance &inst = instanceBuffer[linearIndex];
        float importance = computeInstImportance(inst, geomGroupBuffer);
        lightInstDist->setWeightAt(linearIndex, importance);
    }
}



CUDA_DEVICE_KERNEL void finalizeDiscreteDistribution1D(
    DiscreteDistribution1D* lightInstDist) {
    if (threadIdx.x == 0)
        lightInstDist->finalize();
}
