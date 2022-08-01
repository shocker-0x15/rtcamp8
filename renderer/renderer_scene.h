#pragma once

#include "renderer_shared.h"
#include "../common/common_host.h"

namespace rtc8 {

constexpr cudau::BufferType bufferType = cudau::BufferType::Device;

template <typename T>
using Ref = std::shared_ptr<T>;

template <typename T>
using TypedBufferRef = std::shared_ptr<cudau::TypedBuffer<T>>;



enum class PathTracingEntryPoint {
    pathTrace,
};

struct GPUEnvironment {
    CUcontext cuContext;
    optixu::Context optixContext;

    cudau::TypedBuffer<shared::BSDFProcedureSet> bsdfProcedureSetBuffer;
    std::vector<shared::BSDFProcedureSet> mappedBsdfProcedureSets;

    template <typename EntryPointType>
    struct Pipeline {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::ProgramGroup> entryPoints;
        std::unordered_map<std::string, optixu::ProgramGroup> programs;
        std::vector<optixu::ProgramGroup> callablePrograms;
        cudau::Buffer sbt;
        cudau::Buffer hitGroupSbt;

        void setEntryPoint(EntryPointType et) {
            optixPipeline.setRayGenerationProgram(entryPoints.at(et));
        }
    };

    Pipeline<PathTracingEntryPoint> pathTracing;

    optixu::Material optixDefaultMaterial;

    void initialize() {
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        optixContext = optixu::Context::create(cuContext/*, 4, DEBUG_SELECT(true, false)*/);

        optixDefaultMaterial = optixContext.createMaterial();
        optixu::Module emptyModule;

        {
            Pipeline<PathTracingEntryPoint> &pipeline = pathTracing;
            optixu::Pipeline &p = pipeline.optixPipeline;
            optixu::Module &m = pipeline.optixModule;
            p = optixContext.createPipeline();

            p.setPipelineOptions(
                std::max({
                    shared::ClosestRaySignature::numDwords,
                    shared::VisibilityRaySignature::numDwords
                            }),
                optixu::calcSumDwords<float2>(),
                "plp", sizeof(shared::PipelineLaunchParameters),
                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

            m = p.createModuleFromPTXString(
                readTxtFile(getExecutableDirectory() / "renderer/ptxes/optix_pathtracing_kernels.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            pipeline.entryPoints[PathTracingEntryPoint::pathTrace] =
                p.createRayGenProgram(m, RT_RG_NAME_STR("pathTrace"));

            optixu::ProgramGroup chPathTrace = p.createHitProgramGroupForTriangleIS(
                m, RT_CH_NAME_STR("pathTrace"),
                emptyModule, nullptr);
            pipeline.programs[RT_CH_NAME_STR("pathTrace")] = chPathTrace;

            optixu::ProgramGroup ahVisibility = p.createHitProgramGroupForTriangleIS(
                emptyModule, nullptr,
                m, RT_AH_NAME_STR("visibility"));
            pipeline.programs[RT_AH_NAME_STR("visibility")] = ahVisibility;

            optixu::ProgramGroup emptyMiss = p.createMissProgram(emptyModule, nullptr);
            pipeline.programs["emptyMiss"] = emptyMiss;

            p.setNumMissRayTypes(shared::PathTracingRayType::NumTypes);
            p.setMissProgram(shared::PathTracingRayType::Closest, emptyMiss);
            p.setMissProgram(shared::PathTracingRayType::Visibility, emptyMiss);

            p.setNumCallablePrograms(NumCallablePrograms);
            pipeline.callablePrograms.resize(NumCallablePrograms);
            for (int i = 0; i < NumCallablePrograms; ++i) {
                optixu::ProgramGroup program = p.createCallableProgramGroup(
                    m, callableProgramEntryPoints[i],
                    emptyModule, nullptr);
                pipeline.callablePrograms[i] = program;
                p.setCallableProgram(i, program);
            }

            p.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            optixDefaultMaterial.setHitGroup(shared::PathTracingRayType::Closest, chPathTrace);
            optixDefaultMaterial.setHitGroup(shared::PathTracingRayType::Visibility, ahVisibility);

            size_t sbtSize;
            p.generateShaderBindingTableLayout(&sbtSize);
            pipeline.sbt.initialize(cuContext, bufferType, sbtSize, 1);
            pipeline.sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());
        }
    }

    void finalize() {

    }

    void setupDeviceData() {
        bsdfProcedureSetBuffer.initialize(cuContext, bufferType, mappedBsdfProcedureSets);
    }

    uint32_t registerBSDFProcedureSet(const shared::BSDFProcedureSet &procSet) {
        uint32_t index = static_cast<uint32_t>(mappedBsdfProcedureSets.size());
        mappedBsdfProcedureSets.resize(index + 1);
        mappedBsdfProcedureSets[index] = procSet;
        return index;
    }
};

extern GPUEnvironment g_gpuEnv;



class Texture2D;
class SurfaceMaterial;
class Geometry;
class GeometryGroup;
class Instance;



class Scene {
    static constexpr uint32_t maxNumMaterials = 1 << 10;
    static constexpr uint32_t maxNumGeometryInstances = 1 << 16;
    static constexpr uint32_t maxNumInstances = 1 << 16;

    optixu::Scene m_optixScene;

    SlotFinder m_surfaceMaterialSlotFinder;
    SlotFinder m_geometryInstanceSlotFinder;
    SlotFinder m_instanceSlotFinder;
    cudau::TypedBuffer<shared::SurfaceMaterial> m_surfaceMaterialBuffer;
    cudau::TypedBuffer<shared::GeometryInstance> m_geometryInstanceBuffer;
    cudau::TypedBuffer<shared::Instance> m_instanceBuffer;
    std::unordered_map<uint32_t, Ref<SurfaceMaterial>> m_surfaceMaterialSlotOwners;
    std::unordered_map<uint32_t, Ref<Geometry>> m_geometryInstanceSlotOwners;
    std::unordered_map<uint32_t, Ref<Instance>> m_instanceSlotOwners;

public:
    void initialize() {
        m_optixScene = g_gpuEnv.optixContext.createScene();

        m_surfaceMaterialSlotFinder.initialize(maxNumMaterials);
        m_surfaceMaterialBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumMaterials);
        m_geometryInstanceSlotFinder.initialize(maxNumGeometryInstances);
        m_geometryInstanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumGeometryInstances);
        m_instanceSlotFinder.initialize(maxNumInstances);
        m_instanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumInstances);
    }

    void finalize() {
        m_instanceBuffer.finalize();
        m_instanceSlotFinder.finalize();
        m_geometryInstanceBuffer.finalize();
        m_geometryInstanceSlotFinder.finalize();
        m_surfaceMaterialBuffer.finalize();
        m_surfaceMaterialSlotFinder.finalize();
    }

    void allocateSurfaceMaterial(const Ref<SurfaceMaterial> &surfMat);

    void allocateGeometryInstance(const Ref<Geometry> &geom);

    void allocateGeometryAccelerationStructure(const Ref<GeometryGroup> &geomGroup);

    void allocateInstance(const Ref<Instance> &inst);
};

extern Scene g_scene;



class Texture2D {
    Ref<cudau::Array> m_texArray;
    cudau::TextureSampler m_sampler;
    CUtexObject m_texObject;
    uint32_t m_dirty : 1;

public:
    Texture2D(const Ref<cudau::Array> &texArray) :
        m_texArray(texArray), m_texObject(0), m_dirty(true) {
        m_sampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
        m_sampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
        m_sampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
        m_sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    }
    ~Texture2D() {
        if (m_texObject)
            /*CUDADRV_CHECK(*/cuTexObjectDestroy(m_texObject)/*)*/;
    }

    void setReadMode(cudau::TextureReadMode readMode) {
        m_sampler.setReadMode(readMode);
        m_dirty = true;
    }

    CUtexObject getDeviceTexture() {
        if (m_dirty) {
            if (m_texObject)
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
            m_texObject = m_sampler.createTextureObject(*m_texArray);
            m_dirty = false;
        }
        return m_texObject;
    }

    shared::TexDimInfo getDimInfo() const {
        shared::TexDimInfo dimInfo = {};
        uint32_t w = m_texArray->getWidth();
        uint32_t h = m_texArray->getHeight();
        bool wIsPowerOfTwo = (w & (w - 1)) == 0;
        bool hIsPowerOfTwo = (h & (h - 1)) == 0;
        dimInfo.dimX = w;
        dimInfo.dimY = h;
        dimInfo.isNonPowerOfTwo = !wIsPowerOfTwo || !hIsPowerOfTwo;
        dimInfo.isBCTexture = m_texArray->isBCTexture();
        dimInfo.isLeftHanded = false;
        return dimInfo;
    }
};



class SurfaceMaterial {
    uint32_t m_slot;
    Ref<Texture2D> m_emittance;

public:
    SurfaceMaterial() : m_slot(0xFFFFFFFF) {}
    ~SurfaceMaterial() {}

    void associateScene(uint32_t slot) {
        m_slot = slot;
    }

    void setEmittance(const Ref<Texture2D> &emittance) {
        m_emittance = emittance;
    }

    bool isEmissive() const {
        return m_emittance != nullptr;
    }

    uint32_t getSlot() const {
        return m_slot;
    }

    virtual void setupDeviceType(shared::SurfaceMaterial* deviceData) const {
        if (m_emittance)
            deviceData->emittance = m_emittance->getDeviceTexture();
    }
};



class SimplePBRSurfaceMaterial : public SurfaceMaterial {
    static uint32_t s_procSetSlot;

    Ref<Texture2D> m_baseColor_opacity;
    Ref<Texture2D> m_occlusion_roughness_metallic;

public:
    static void setBSDFProcedureSet() {
        shared::BSDFProcedureSet procSet;
        procSet.setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;
        procSet.getSurfaceParameters = CallableProgram_DichromaticBRDF_getSurfaceParameters;
        procSet.sampleF = CallableProgram_DichromaticBRDF_sampleF;
        procSet.evaluateF = CallableProgram_DichromaticBRDF_evaluateF;
        procSet.evaluatePDF = CallableProgram_DichromaticBRDF_evaluatePDF;
        procSet.evaluateDHReflectanceEstimate = CallableProgram_DichromaticBRDF_evaluateDHReflectanceEstimate;
        s_procSetSlot = g_gpuEnv.registerBSDFProcedureSet(procSet);
    }

    SimplePBRSurfaceMaterial() {}

    void set(
        const Ref<Texture2D> &baseColor_opacity,
        const Ref<Texture2D> &occlusion_roughness_metallic) {
        m_baseColor_opacity = baseColor_opacity;
        m_occlusion_roughness_metallic = occlusion_roughness_metallic;
    }

    void setupDeviceType(shared::SurfaceMaterial* deviceData) const override {
        SurfaceMaterial::setupDeviceType(deviceData);
        auto &body = *reinterpret_cast<shared::SimplePBRSurfaceMaterial*>(deviceData->body);
        body.baseColor_opacity = m_baseColor_opacity->getDeviceTexture();
        body.baseColor_opacity_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        body.occlusion_roughness_metallic = m_occlusion_roughness_metallic->getDeviceTexture();
        body.occlusion_roughness_metallic_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        deviceData->bsdfProcSetSlot = s_procSetSlot;
        deviceData->setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;
    }
};



enum class BumpMapTextureType {
    NormalMap = 0,
    NormalMap_BC,
    NormalMap_BC_2ch,
    HeightMap,
    HeightMap_BC,
};



struct VertexBuffer {
    std::vector<shared::Vertex> onHost;
    cudau::TypedBuffer<shared::Vertex> onDevice;
};



class Geometry {
protected:
    optixu::GeometryInstance m_optixGeomInst;
    uint32_t m_slot;
    BoundingBox3D m_aabb;

public:
    Geometry() : m_slot(0xFFFFFFFF) {}
    virtual ~Geometry() {}

    virtual void associateScene(uint32_t slot, optixu::GeometryInstance optixGeomInst) {
        m_slot = slot;
        m_optixGeomInst = optixGeomInst;
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }

    optixu::GeometryInstance getOptixGeometryInstance() const {
        return m_optixGeomInst;
    }

    virtual void setupDeviceType(shared::GeometryInstance* deviceData) const {}
};



class TriangleMeshGeometry : public Geometry {
    Ref<VertexBuffer> m_vertices;
    cudau::TypedBuffer<shared::Triangle> m_triangles;
    Ref<Texture2D> m_normalMap;
    Ref<SurfaceMaterial> m_surfMat;
    BumpMapTextureType m_bumpMapType;

public:
    TriangleMeshGeometry() : m_bumpMapType(BumpMapTextureType::NormalMap_BC_2ch) {}
    ~TriangleMeshGeometry() {
        m_triangles.finalize();
    }

    void set(
        const Ref<VertexBuffer> &vertices,
        const std::vector<shared::Triangle> &triangles,
        const Ref<Texture2D> &normalMap, BumpMapTextureType bumpMapType,
        const Ref<SurfaceMaterial> &surfMat) {
        m_vertices = vertices;
        m_triangles.initialize(g_gpuEnv.cuContext, bufferType, triangles);
        m_normalMap = normalMap;
        m_bumpMapType = bumpMapType;
        m_surfMat = surfMat;
        m_aabb = BoundingBox3D();
        for (int triIdx = 0; triIdx < triangles.size(); ++triIdx) {
            const shared::Triangle &tri = triangles[triIdx];
            const shared::Vertex(&vs)[3] = {
                vertices->onHost[tri.indices[0]],
                vertices->onHost[tri.indices[1]],
                vertices->onHost[tri.indices[2]],
            };
            m_aabb
                .unify(vs[0].position)
                .unify(vs[1].position)
                .unify(vs[2].position);
        }

        m_optixGeomInst.setVertexBuffer(m_vertices->onDevice);
        m_optixGeomInst.setTriangleBuffer(m_triangles);
    }

    void setupDeviceType(shared::GeometryInstance* deviceData) const override {
        Geometry::setupDeviceType(deviceData);
        deviceData->vertices = m_vertices->onDevice.getDevicePointer();
        deviceData->triangles = m_triangles.getDevicePointer();
        deviceData->normal = m_normalMap->getDeviceTexture();
        deviceData->normalDimInfo = m_normalMap->getDimInfo();
        if (m_bumpMapType == BumpMapTextureType::NormalMap ||
            m_bumpMapType == BumpMapTextureType::NormalMap_BC)
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap;
        else if (m_bumpMapType == BumpMapTextureType::NormalMap_BC_2ch)
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap2ch;
        else
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromHeightMap;
        deviceData->surfMatSlot = m_surfMat->getSlot();
    }
};



class GeometryGroup {
    optixu::GeometryAccelerationStructure m_optixGas;
    std::set<Ref<Geometry>> m_geometries;
    BoundingBox3D m_aabb;

    cudau::Buffer m_optixAsScratchMem;
    cudau::Buffer m_optixAsMem;
    uint32_t m_dirty : 1;
public:
    GeometryGroup() : m_dirty(false) {}

    void associateScene(optixu::GeometryAccelerationStructure optixGas) {
        m_optixGas = optixGas;
    }

    void set(const std::set<Ref<Geometry>> &geoms) {
        m_dirty = true;
        m_aabb = BoundingBox3D();
        for (const Ref<Geometry> &geom : geoms) {
            m_geometries.insert(geom);
            m_optixGas.addChild(geom->getOptixGeometryInstance());
            m_aabb.unify(geom->getAABB());
        }
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }
};



struct KeyInstanceState {
    float timePoint;
    float scale;
    Quaternion orientation;
    Point3D position;
};



class Instance {
    optixu::Instance m_optixInst;
    uint32_t m_slot;
    std::vector<KeyInstanceState> m_keyStates;
    Ref<GeometryGroup> m_geomGroup;
    Matrix4x4 m_staticTransform;

public:
    Instance() : m_slot(0xFFFFFFFF) {}

    void associateScene(uint32_t slot, optixu::Instance optixInst) {
        m_slot = slot;
        m_optixInst = optixInst;
    }

    void setGeometryGroup(const Ref<GeometryGroup> &geomGroup, const Matrix4x4 &staticTransform) {
        m_geomGroup = geomGroup;
        m_staticTransform = staticTransform;
    }

    void setKeyStates(const std::vector<KeyInstanceState> &states) {
        m_keyStates = states;
    }
};



struct KeyCameraState {
    float timePoint;
    Point3D position;
    Point3D lookAt;
    Vector3D up;
    float fovY;
};



class Camera {
    std::vector<KeyCameraState> m_keyStates;

public:
    Camera(const std::vector<KeyCameraState> &keyStates) :
        m_keyStates(keyStates) {}
};



struct ActiveCameraInfo {
    float timePoint;
    std::string name;
};

struct RenderConfigs {
    uint32_t imageWidth;
    uint32_t imageHeight;
    float timeBegin;
    float timeEnd;
    uint32_t fps;

    std::map<std::string, Ref<Camera>> cameras;
    std::vector<ActiveCameraInfo> activeCameraInfos;

    std::vector<Ref<SurfaceMaterial>> surfaceMaterials;
    std::vector<Ref<Geometry>> geometries;
    std::vector<Ref<GeometryGroup>> geometryGroups;
    std::vector<Ref<Instance>> instances;
};

void loadScene(const std::filesystem::path &filePath, RenderConfigs* renderConfigs);

} // namespace rtc8