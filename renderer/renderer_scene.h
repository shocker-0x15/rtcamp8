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

    CUmodule postProcessKernelsModule;
    CUdeviceptr plpForPostProcessKernelsModule;
    cudau::Kernel applyToneMap;

    cudau::TypedBuffer<shared::BSDFProcedureSet> bsdfProcedureSetBuffer;
    std::vector<shared::BSDFProcedureSet> mappedBsdfProcedureSets;

    optixu::Material optixDefaultMaterial;

    template <typename EntryPointType>
    struct Pipeline {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::ProgramGroup> entryPoints;
        std::unordered_map<std::string, optixu::ProgramGroup> programs;
        std::vector<optixu::ProgramGroup> callablePrograms;
        cudau::Buffer sbt;

        void setEntryPoint(EntryPointType et) {
            optixPipeline.setRayGenerationProgram(entryPoints.at(et));
        }
    };

    Pipeline<PathTracingEntryPoint> pathTracing;

    void initialize() {
        const std::filesystem::path exeDir = getExecutableDirectory();

        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        optixContext = optixu::Context::create(cuContext/*, 4, DEBUG_SELECT(true, false)*/);



        size_t symbolSize;

        CUDADRV_CHECK(cuModuleLoad(
            &postProcessKernelsModule,
            (exeDir / "renderer/ptxes/post_process_kernels.ptx").string().c_str()));
        applyToneMap.set(postProcessKernelsModule, "applyToneMap", cudau::dim3(8, 8), 0);
        CUDADRV_CHECK(cuModuleGetGlobal(
            &plpForPostProcessKernelsModule, &symbolSize, postProcessKernelsModule, "plp"));



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
    std::unordered_set<Ref<GeometryGroup>> m_geometryGroups;
    std::unordered_map<uint32_t, Ref<Instance>> m_instanceSlotOwners;

    optixu::InstanceAccelerationStructure m_optixIas;
    cudau::Buffer m_optixAsScratchMem;
    cudau::Buffer m_optixAsMem;
    cudau::TypedBuffer<OptixInstance> m_optixInstanceBuffer;

    cudau::Buffer m_hitGroupSbt;

public:
    void initialize() {
        m_optixScene = g_gpuEnv.optixContext.createScene();

        m_surfaceMaterialSlotFinder.initialize(maxNumMaterials);
        m_surfaceMaterialBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumMaterials);
        m_geometryInstanceSlotFinder.initialize(maxNumGeometryInstances);
        m_geometryInstanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumGeometryInstances);
        m_instanceSlotFinder.initialize(maxNumInstances);
        m_instanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumInstances);

        m_optixIas = m_optixScene.createInstanceAccelerationStructure();
        m_optixIas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    }

    void finalize() {

    }



    void allocateSurfaceMaterial(const Ref<SurfaceMaterial> &surfMat);

    void allocateGeometryInstance(const Ref<Geometry> &geom);

    void allocateGeometryAccelerationStructure(const Ref<GeometryGroup> &geomGroup);

    void allocateInstance(const Ref<Instance> &inst);



    shared::SurfaceMaterial* getSurfaceMaterialsOnDevice() const {
        return m_surfaceMaterialBuffer.getDevicePointer();
    }

    shared::GeometryInstance* getGeometryInstancesOnDevice() const {
        return m_geometryInstanceBuffer.getDevicePointer();
    }

    shared::Instance* getInstancesOnDevice() const {
        return m_instanceBuffer.getDevicePointer();
    }



    OptixTraversableHandle buildASs(CUstream stream, float timePoint);
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
protected:
    uint32_t m_slot;
    Ref<Texture2D> m_emittance;
    uint32_t m_dirty : 1;

public:
    SurfaceMaterial() : m_slot(0xFFFFFFFF), m_dirty(false) {}
    ~SurfaceMaterial() {}

    void associateScene(uint32_t slot) {
        m_slot = slot;
        m_dirty = true;
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

    virtual void setUpDeviceType(shared::SurfaceMaterial* deviceData)  {
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

        m_dirty = true;
    }

    void setUpDeviceType(shared::SurfaceMaterial* deviceData) override {
        if (!m_dirty)
            return;

        SurfaceMaterial::setUpDeviceType(deviceData);
        auto &body = *reinterpret_cast<shared::SimplePBRSurfaceMaterial*>(deviceData->body);
        body.baseColor_opacity = m_baseColor_opacity->getDeviceTexture();
        body.baseColor_opacity_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        body.occlusion_roughness_metallic = m_occlusion_roughness_metallic->getDeviceTexture();
        body.occlusion_roughness_metallic_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        deviceData->bsdfProcSetSlot = s_procSetSlot;
        deviceData->setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;

        m_dirty = false;
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
    uint32_t m_dirty : 1;

public:
    Geometry() : m_slot(0xFFFFFFFF), m_dirty(false) {}
    virtual ~Geometry() {}

    virtual void associateScene(uint32_t slot, optixu::GeometryInstance optixGeomInst) {
        m_slot = slot;
        m_optixGeomInst = optixGeomInst;
        m_dirty = true;
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }

    optixu::GeometryInstance getOptixGeometryInstance() const {
        return m_optixGeomInst;
    }

    uint32_t getSlot() const {
        return m_slot;
    }

    virtual void setUpDeviceType(shared::GeometryInstance* deviceData) {}
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

        m_dirty = true;
    }

    void setUpDeviceType(shared::GeometryInstance* deviceData) override {
        if (!m_dirty)
            return;

        Geometry::setUpDeviceType(deviceData);
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
        m_dirty = false;
    }
};



class GeometryGroup {
    optixu::GeometryAccelerationStructure m_optixGas;
    std::set<Ref<Geometry>> m_geometries;
    BoundingBox3D m_aabb;

    cudau::Buffer m_optixAsScratchMem;
    cudau::Buffer m_optixAsMem;
    cudau::TypedBuffer<uint32_t> m_geomInstSlotBuffer;
    LightDistribution m_lightGeomInstDist;
    uint32_t m_dirty : 1;

public:
    GeometryGroup() : m_dirty(false) {}

    void associateScene(optixu::GeometryAccelerationStructure optixGas) {
        m_optixGas = optixGas;
        m_dirty = true;
    }

    void set(const std::set<Ref<Geometry>> &geoms) {
        m_aabb = BoundingBox3D();
        std::vector<uint32_t> geomInstSlots(geoms.size());
        uint32_t nextIndex = 0;
        for (const Ref<Geometry> &geom : geoms) {
            m_geometries.insert(geom);
            m_optixGas.addChild(geom->getOptixGeometryInstance());
            m_aabb.unify(geom->getAABB());
            geomInstSlots[nextIndex] = geom->getSlot();
            ++nextIndex;
        }
        m_geomInstSlotBuffer.initialize(g_gpuEnv.cuContext, bufferType, geomInstSlots);
        m_lightGeomInstDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, geomInstSlots.size());
        m_dirty = true;
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }

    optixu::GeometryAccelerationStructure getOptixGAS() const {
        return m_optixGas;
    }

    void buildAS(CUstream stream) {
        if (!m_dirty)
            return;

        OptixAccelBufferSizes sizes;
        m_optixGas.prepareForBuild(&sizes);

        if (!m_optixAsMem.isInitialized())
            m_optixAsMem.initialize(g_gpuEnv.cuContext, bufferType, sizes.outputSizeInBytes, 1);
        else if (m_optixAsMem.sizeInBytes() < sizes.outputSizeInBytes)
            m_optixAsMem.resize(sizes.outputSizeInBytes, 1);

        if (!m_optixAsScratchMem.isInitialized())
            m_optixAsScratchMem.initialize(g_gpuEnv.cuContext, bufferType, sizes.tempSizeInBytes, 1);
        else if (m_optixAsScratchMem.sizeInBytes() < sizes.tempSizeInBytes)
            m_optixAsScratchMem.resize(sizes.tempSizeInBytes, 1);

        m_optixGas.rebuild(stream, m_optixAsMem, m_optixAsScratchMem);

        m_dirty = false;
    }

    void setUpDeviceData(shared::Instance* deviceData) {
        deviceData->geomInstSlots = m_geomInstSlotBuffer.getDevicePointer();
        m_lightGeomInstDist.getDeviceType(&deviceData->lightGeomInstDist);
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
    float m_lastTimePoint;
    uint32_t m_dirty : 1;

public:
    Instance() : m_slot(0xFFFFFFFF), m_lastTimePoint(NAN), m_dirty(false) {}

    void associateScene(uint32_t slot, optixu::Instance optixInst) {
        m_slot = slot;
        m_optixInst = optixInst;
        m_dirty = true;
    }

    void setGeometryGroup(const Ref<GeometryGroup> &geomGroup, const Matrix4x4 &staticTransform) {
        m_optixInst.setChild(geomGroup->getOptixGAS());
        m_geomGroup = geomGroup;
        m_staticTransform = staticTransform;
        m_dirty = true;
    }

    void setKeyStates(const std::vector<KeyInstanceState> &states) {
        m_keyStates = states;
        m_dirty = true;
    }

    void setUpDeviceType(shared::Instance* deviceData, float timePoint) {
        uint32_t numStates = static_cast<uint32_t>(m_keyStates.size());
        int idx = 0;
        for (int d = nextPowerOf2(numStates) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= numStates)
                continue;
            const KeyInstanceState &keyState = m_keyStates[idx + d];
            if (keyState.timePoint <= timePoint)
                idx += d;
        }
        KeyInstanceState states[2];
        states[0] = m_keyStates[idx];
        states[1] = m_keyStates[std::min(static_cast<uint32_t>(idx) + 1, numStates - 1)];
        float t = safeDivide(timePoint - states[0].timePoint, states[1].timePoint - states[0].timePoint);
        float scale = lerp(states[0].scale, states[1].scale, t);
        Quaternion orientation = slerp(states[0].orientation, states[1].orientation, t);
        Point3D position = lerp(states[0].position, states[1].position, t);
        Matrix4x4 transform =
            translate4x4(position) * orientation.toMatrix4x4() * scale4x4(scale) *
            m_staticTransform;

        deviceData->transform = transform;
        deviceData->prevTransform = transform; // TODO?
        deviceData->uniformScale = scale;
        m_geomGroup->setUpDeviceData(deviceData);

        const float xfmArray[12] = {
            transform.m00, transform.m01, transform.m02, transform.m03,
            transform.m10, transform.m11, transform.m12, transform.m13,
            transform.m20, transform.m21, transform.m22, transform.m23,
        };
        m_optixInst.setTransform(xfmArray);
    }
};



struct KeyCameraState {
    float timePoint;
    Point3D position;
    Point3D positionLookAt;
    Vector3D up;
    float fovY;
};



class Camera {
    std::vector<KeyCameraState> m_keyStates;

public:
    Camera(const std::vector<KeyCameraState> &keyStates) :
        m_keyStates(keyStates) {}

    void setUpDeviceType(shared::PerspectiveCamera* deviceData, float timePoint) {
        uint32_t numStates = static_cast<uint32_t>(m_keyStates.size());
        int idx = 0;
        for (int d = nextPowerOf2(numStates) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= numStates)
                continue;
            const KeyCameraState &keyState = m_keyStates[idx + d];
            if (keyState.timePoint <= timePoint)
                idx += d;
        }
        KeyCameraState states[2];
        states[0] = m_keyStates[idx];
        states[1] = m_keyStates[std::min(static_cast<uint32_t>(idx) + 1, numStates - 1)];
        float t = safeDivide(timePoint - states[0].timePoint, states[1].timePoint - states[0].timePoint);
        Point3D position = lerp(states[0].position, states[1].position, t);
        Quaternion ori0 = qLookAt(states[0].positionLookAt - states[0].position, states[0].up);
        Quaternion ori1 = qLookAt(states[1].positionLookAt - states[1].position, states[1].up);
        Quaternion orientation = slerp(ori0, ori1, t);
        float fovY = lerp(states[0].fovY, states[1].fovY, t);
        deviceData->position = position;
        deviceData->orientation = conjugate(orientation)/* * qRotateY(pi_v<float>)*/;
        deviceData->fovY = fovY * pi_v<float> / 180;
    }
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