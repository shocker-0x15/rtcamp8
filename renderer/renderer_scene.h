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



class SurfaceMaterial;



class SceneMemory {
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

public:
    void initialize() {
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

    uint32_t allocateSurfaceMaterial() {
        uint32_t slot = m_surfaceMaterialSlotFinder.getFirstAvailableSlot();
        Assert(slot != 0xFFFFFFFF, "failed to allocate a SurfaceMaterial.");
        return slot;
    }

    uint32_t allocateGeometryInstance() {
        uint32_t slot = m_geometryInstanceSlotFinder.getFirstAvailableSlot();
        Assert(slot != 0xFFFFFFFF, "failed to allocate a GeometryInstance.");
        return slot;
    }

    uint32_t allocateInstance() {
        uint32_t slot = m_instanceSlotFinder.getFirstAvailableSlot();
        Assert(slot != 0xFFFFFFFF, "failed to allocate an Instance.");
        return slot;
    }
};

extern SceneMemory g_sceneMem;



class SurfaceMaterial {
    uint32_t m_slot;

public:
    SurfaceMaterial();
    ~SurfaceMaterial() {}

    virtual void setupDeviceType(shared::SurfaceMaterial* deviceData) const = 0;
};



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



class SimplePBRSurfaceMaterial : SurfaceMaterial {
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

    SimplePBRSurfaceMaterial(
        const Ref<Texture2D> &baseColor_opacity,
        const Ref<Texture2D> &occlusion_roughness_metallic) :
        m_baseColor_opacity(baseColor_opacity),
        m_occlusion_roughness_metallic(occlusion_roughness_metallic) {}

    void setupDeviceType(shared::SurfaceMaterial* deviceData) const override {
        auto &body = *reinterpret_cast<shared::SimplePBRSurfaceMaterial*>(deviceData->body);
        body.baseColor_opacity = m_baseColor_opacity->getDeviceTexture();
        body.baseColor_opacity_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        body.occlusion_roughness_metallic = m_occlusion_roughness_metallic->getDeviceTexture();
        body.occlusion_roughness_metallic_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        deviceData->bsdfProcSetSlot = s_procSetSlot;
        deviceData->setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;
    }
};



class Geometry {
    uint32_t m_slot;

public:
    Geometry();
    ~Geometry();
};



class TriangleMesh : public Geometry {
    TypedBufferRef<shared::Vertex> m_vertices;
    TypedBufferRef<shared::Triangle> m_triangles;
    Ref<Texture2D> m_normalMap;
    Ref<SurfaceMaterial> m_material;

public:
    TriangleMesh(
        const TypedBufferRef<shared::Vertex> &vertices,
        const TypedBufferRef<shared::Triangle> &triangles,
        const Ref<Texture2D> &normalMap,
        const Ref<SurfaceMaterial> &material) :
        m_vertices(vertices), m_triangles(triangles), m_normalMap(normalMap),
        m_material(material) {}
};



class GeometryGroup {
    std::vector<Ref<Geometry>> m_geometries;

    cudau::Buffer m_asScratchMem;
    cudau::Buffer m_asMem;
    uint32_t m_dirty : 1;
public:
    GeometryGroup() :
        m_dirty(true) {}
};



struct KeyInstanceState {
    float timePoint;
    float scale;
    Quaternion orientation;
    Point3D position;
};



class Instance {
    uint32_t m_slot;
    std::vector<KeyInstanceState> m_keyStates;

public:
    Instance();
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
    Camera() {}
};



void loadScene(const std::filesystem::path &filePath);

} // namespace rtc8