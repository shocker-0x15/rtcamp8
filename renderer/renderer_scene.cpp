#include "renderer_scene.h"
#include <regex>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "../common/dds_loader.h"
#include "../ext/stb_image.h"

namespace rtc8 {

GPUEnvironment g_gpuEnv;
Scene g_scene;



void Scene::allocateSurfaceMaterial(const Ref<SurfaceMaterial> &surfMat) {
    uint32_t slot = m_surfaceMaterialSlotFinder.getFirstAvailableSlot();
    Assert(slot != 0xFFFFFFFF, "failed to allocate a SurfaceMaterial.");
    m_surfaceMaterialSlotFinder.setInUse(slot);
    m_surfaceMaterialSlotOwners[slot] = surfMat;
    surfMat->associateScene(slot);
}

void Scene::allocateGeometryInstance(const Ref<Geometry> &geom) {
    uint32_t slot = m_geometryInstanceSlotFinder.getFirstAvailableSlot();
    Assert(slot != 0xFFFFFFFF, "failed to allocate a GeometryInstance.");
    m_geometryInstanceSlotFinder.setInUse(slot);
    m_geometryInstanceSlotOwners[slot] = geom;
    optixu::GeometryInstance optixGeomInst = m_optixScene.createGeometryInstance();
    optixGeomInst.setUserData(slot);
    optixGeomInst.setNumMaterials(1, optixu::BufferView());
    optixGeomInst.setMaterial(0, 0, g_gpuEnv.optixDefaultMaterial);
    geom->associateScene(slot, optixGeomInst);
}

void Scene::allocateGeometryAccelerationStructure(const Ref<GeometryGroup> &geomGroup) {
    optixu::GeometryAccelerationStructure optixGas = m_optixScene.createGeometryAccelerationStructure();
    optixGas.setNumMaterialSets(1);
    optixGas.setNumRayTypes(0, shared::maxNumRayTypes);
    geomGroup->associateScene(optixGas);
}

void Scene::allocateInstance(const Ref<Instance> &inst) {
    uint32_t slot = m_instanceSlotFinder.getFirstAvailableSlot();
    Assert(slot != 0xFFFFFFFF, "failed to allocate an Instance.");
    m_instanceSlotFinder.setInUse(slot);
    m_instanceSlotOwners[slot] = inst;
    optixu::Instance optixInst = m_optixScene.createInstance();
    optixInst.setID(slot);
    inst->associateScene(slot, optixInst);
}



uint32_t SimplePBRSurfaceMaterial::s_procSetSlot;



static void translate(
    dds::Format ddsFormat,
    cudau::ArrayElementType* cudaType, bool* needsDegamma, bool* isHDR) {
    *needsDegamma = false;
    *isHDR = false;
    switch (ddsFormat) {
    case dds::Format::BC1_UNorm:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        break;
    case dds::Format::BC1_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC1_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC2_UNorm:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        break;
    case dds::Format::BC2_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC2_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC3_UNorm:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        break;
    case dds::Format::BC3_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC3_UNorm;
        *needsDegamma = true;
        break;
    case dds::Format::BC4_UNorm:
        *cudaType = cudau::ArrayElementType::BC4_UNorm;
        break;
    case dds::Format::BC4_SNorm:
        *cudaType = cudau::ArrayElementType::BC4_SNorm;
        break;
    case dds::Format::BC5_UNorm:
        *cudaType = cudau::ArrayElementType::BC5_UNorm;
        break;
    case dds::Format::BC5_SNorm:
        *cudaType = cudau::ArrayElementType::BC5_SNorm;
        break;
    case dds::Format::BC6H_UF16:
        *cudaType = cudau::ArrayElementType::BC6H_UF16;
        *isHDR = true;
        break;
    case dds::Format::BC6H_SF16:
        *cudaType = cudau::ArrayElementType::BC6H_SF16;
        *isHDR = true;
        break;
    case dds::Format::BC7_UNorm:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        break;
    case dds::Format::BC7_UNorm_sRGB:
        *cudaType = cudau::ArrayElementType::BC7_UNorm;
        *needsDegamma = true;
        break;
    default:
        break;
    }
};

static BumpMapTextureType getBumpMapType(cudau::ArrayElementType elemType) {
    if (elemType == cudau::ArrayElementType::BC1_UNorm ||
        elemType == cudau::ArrayElementType::BC2_UNorm ||
        elemType == cudau::ArrayElementType::BC3_UNorm ||
        elemType == cudau::ArrayElementType::BC7_UNorm)
        return BumpMapTextureType::NormalMap_BC;
    else if (elemType == cudau::ArrayElementType::BC4_SNorm ||
             elemType == cudau::ArrayElementType::BC4_UNorm)
        return BumpMapTextureType::HeightMap_BC;
    else if (elemType == cudau::ArrayElementType::BC5_UNorm)
        return BumpMapTextureType::NormalMap_BC_2ch;
    else
        Assert_NotImplemented();
    return BumpMapTextureType::NormalMap;
}

static BumpMapTextureType getBumpMapType(GLenum glFormat) {
    if (glFormat == 0x8CF1 ||
        glFormat == 0x83F2 ||
        glFormat == 0x83F3 ||
        glFormat == 0x838C)
        return BumpMapTextureType::NormalMap_BC;
    else if (glFormat == 0x8DBB ||
             glFormat == 0x8DBC)
        return BumpMapTextureType::HeightMap_BC;
    else if (glFormat == 0x8DBD)
        return BumpMapTextureType::NormalMap_BC_2ch;
    else
        Assert_NotImplemented();
    return BumpMapTextureType::NormalMap;
}

struct TextureCacheKey {
    std::filesystem::path filePath;
    CUcontext cuContext;

    bool operator<(const TextureCacheKey &rKey) const {
        if (filePath < rKey.filePath)
            return true;
        else if (filePath > rKey.filePath)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx1ImmTextureCacheKey {
    float immValue;
    CUcontext cuContext;

    bool operator<(const Fx1ImmTextureCacheKey &rKey) const {
        if (immValue < rKey.immValue)
            return true;
        else if (immValue > rKey.immValue)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx3ImmTextureCacheKey {
    float3 immValue;
    CUcontext cuContext;

    bool operator<(const Fx3ImmTextureCacheKey &rKey) const {
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct Fx4ImmTextureCacheKey {
    float4 immValue;
    CUcontext cuContext;

    bool operator<(const Fx4ImmTextureCacheKey &rKey) const {
        if (immValue.w < rKey.immValue.w)
            return true;
        else if (immValue.w > rKey.immValue.w)
            return false;
        if (immValue.z < rKey.immValue.z)
            return true;
        else if (immValue.z > rKey.immValue.z)
            return false;
        if (immValue.y < rKey.immValue.y)
            return true;
        else if (immValue.y > rKey.immValue.y)
            return false;
        if (immValue.x < rKey.immValue.x)
            return true;
        else if (immValue.x > rKey.immValue.x)
            return false;
        if (cuContext < rKey.cuContext)
            return true;
        else if (cuContext > rKey.cuContext)
            return false;
        return false;
    }
};

struct TextureCacheValue {
    Ref<cudau::Array> texture;
    bool needsDegamma;
    bool isHDR;
    BumpMapTextureType bumpMapType;
};

static std::map<TextureCacheKey, TextureCacheValue> s_textureCache;
static std::map<Fx1ImmTextureCacheKey, TextureCacheValue> s_Fx1ImmTextureCache;
static std::map<Fx3ImmTextureCacheKey, TextureCacheValue> s_Fx3ImmTextureCache;
static std::map<Fx4ImmTextureCacheKey, TextureCacheValue> s_Fx4ImmTextureCache;

static void createFx1ImmTexture(
    float immValue,
    bool isNormalized,
    Ref<cudau::Array>* texture) {
    Fx1ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = g_gpuEnv.cuContext;
    if (s_Fx1ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx1ImmTextureCache.at(cacheKey);
        *texture = value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.texture = std::make_shared<cudau::Array>();
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint8_t data = std::min<uint32_t>(255 * immValue, 255);
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 1);
    }
    else {
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write(&immValue, 1);
    }

    s_Fx1ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = s_Fx1ImmTextureCache.at(cacheKey).texture;
}

static void createFx3ImmTexture(
    const float3 &immValue,
    bool isNormalized,
    Ref<cudau::Array>* texture) {
    Fx3ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = g_gpuEnv.cuContext;
    if (s_Fx3ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx3ImmTextureCache.at(cacheKey);
        *texture = value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.texture = std::make_shared<cudau::Array>();
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         255 << 24);
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, 1.0f
        };
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write(data, 4);
    }

    s_Fx3ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = s_Fx3ImmTextureCache.at(cacheKey).texture;
}

static void createFx4ImmTexture(
    const float4 &immValue,
    bool isNormalized,
    Ref<cudau::Array>* texture) {
    Fx4ImmTextureCacheKey cacheKey;
    cacheKey.immValue = immValue;
    cacheKey.cuContext = g_gpuEnv.cuContext;
    if (s_Fx4ImmTextureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_Fx4ImmTextureCache.at(cacheKey);
        *texture = value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.texture = std::make_shared<cudau::Array>();
    cacheValue.isHDR = !isNormalized;
    if (isNormalized) {
        uint32_t data = ((std::min<uint32_t>(255 * immValue.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immValue.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immValue.z, 255) << 16) |
                         (std::min<uint32_t>(255 * immValue.w, 255) << 24));
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        float data[4] = {
            immValue.x, immValue.y, immValue.z, immValue.w
        };
        cacheValue.texture->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture->write(data, 4);
    }

    s_Fx4ImmTextureCache[cacheKey] = std::move(cacheValue);

    *texture = s_Fx4ImmTextureCache.at(cacheKey).texture;
}

static bool loadTexture(
    const std::filesystem::path &filePath, const float4 &fallbackValue,
    Ref<cudau::Array>* texture,
    bool* needsDegamma,
    bool* isHDR = nullptr) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = g_gpuEnv.cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = value.texture;
        *needsDegamma = value.needsDegamma;
        if (isHDR)
            *isHDR = value.isHDR;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    cacheValue.texture = std::make_shared<cudau::Array>();
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma, &cacheValue.isHDR);
            cacheValue.texture->initialize2D(
                g_gpuEnv.cuContext, elemType, 1,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, mipCount);
            for (int32_t mipLevel = 0; mipLevel < std::max(mipCount - 2, 1); ++mipLevel)
                cacheValue.texture->write<uint8_t>(
                    imageData[mipLevel], static_cast<uint32_t>(sizes[mipLevel]), mipLevel);
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(
            filePath.string().c_str(), &width, &height, &n, 4);
        if (linearImageData) {
            cacheValue.texture->initialize2D(
                g_gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            cacheValue.texture->write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
            cacheValue.needsDegamma = true;
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);

        *texture = s_textureCache.at(cacheKey).texture;
        *needsDegamma = s_textureCache.at(cacheKey).needsDegamma;
        if (isHDR)
            *isHDR = s_textureCache.at(cacheKey).isHDR;
    }
    else {
        createFx4ImmTexture(fallbackValue, true, texture);
        cacheValue.needsDegamma = true;
        cacheValue.isHDR = false;
    }

    return success;
}

static bool loadNormalTexture(
    const std::filesystem::path &filePath,
    Ref<cudau::Array>* texture,
    BumpMapTextureType* bumpMapType) {
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = g_gpuEnv.cuContext;
    if (s_textureCache.count(cacheKey)) {
        const TextureCacheValue &value = s_textureCache.at(cacheKey);
        *texture = value.texture;
        *bumpMapType = value.bumpMapType;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue;
    cacheValue.texture = std::make_shared<cudau::Array>();
    if (filePath.extension() == ".dds" ||
        filePath.extension() == ".DDS") {
        int32_t width, height, mipCount;
        dds::Format ddsFormat;
        size_t* sizes;
        uint8_t** imageData = dds::load(filePath.string().c_str(),
                                        &width, &height, &mipCount, &sizes, &ddsFormat);
        if (imageData) {
            bool isHDR;
            cudau::ArrayElementType elemType;
            translate(ddsFormat, &elemType, &cacheValue.needsDegamma, &isHDR);
            cacheValue.bumpMapType = getBumpMapType(elemType);
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap_BC ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture->initialize2D(
                g_gpuEnv.cuContext, elemType, 1,
                cudau::ArraySurface::Disable,
                textureGather,
                width, height, mipCount);
            for (int32_t mipLevel = 0; mipLevel < std::max(mipCount - 2, 1); ++mipLevel)
                cacheValue.texture->write<uint8_t>(
                    imageData[mipLevel], static_cast<uint32_t>(sizes[mipLevel]), mipLevel);
            dds::free(imageData, mipCount, sizes);
        }
        else {
            success = false;
        }
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filePath.string().c_str(),
                                             &width, &height, &n, 4);
        std::string filename = filePath.filename().string();
        if (n > 1 &&
            filename != "spnza_bricks_a_bump.png") // Dedicated fix for crytek sponza model.
            cacheValue.bumpMapType = BumpMapTextureType::NormalMap;
        else
            cacheValue.bumpMapType = BumpMapTextureType::HeightMap;
        if (linearImageData) {
            auto textureGather = cacheValue.bumpMapType == BumpMapTextureType::HeightMap ?
                cudau::ArrayTextureGather::Enable :
                cudau::ArrayTextureGather::Disable;
            cacheValue.texture->initialize2D(
                g_gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, textureGather,
                width, height, 1);
            cacheValue.texture->write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
        }
        else {
            success = false;
        }
    }

    if (success) {
        s_textureCache[cacheKey] = std::move(cacheValue);
        *texture = s_textureCache.at(cacheKey).texture;
        *bumpMapType = s_textureCache.at(cacheKey).bumpMapType;
    }
    else {
        createFx3ImmTexture(float3(0.5f, 0.5f, 1.0f), true, texture);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }

    return success;
}

static Ref<cudau::Array> createNormalTexture(
    const std::filesystem::path &normalPath, BumpMapTextureType* bumpMapType) {
    Ref<cudau::Array> arrayNormal;
    if (normalPath.empty()) {
        createFx3ImmTexture(float3(0.5f, 0.5f, 1.0f), true, &arrayNormal);
        *bumpMapType = BumpMapTextureType::NormalMap;
    }
    else {
        hpprintf("  Reading: %s ... ", normalPath.string().c_str());
        if (loadNormalTexture(normalPath, &arrayNormal, bumpMapType))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    return arrayNormal;
}

static Ref<cudau::Array> createEmittanceTexture(
    const std::filesystem::path &emittancePath, const RGBSpectrum &immEmittance,
    bool* needsDegamma, bool* isHDR) {
    Ref<cudau::Array> arrayEmittance;
    *needsDegamma = false;
    *isHDR = false;
    float3 f3_immEmittance(
        immEmittance.r,
        immEmittance.g,
        immEmittance.b);
    if (emittancePath.empty()) {
        if (immEmittance != RGBSpectrum::Zero()) {
            createFx3ImmTexture(f3_immEmittance, false, &arrayEmittance);
            *isHDR = true;
        }
    }
    else {
        hpprintf("  Reading: %s ... ", emittancePath.string().c_str());
        if (loadTexture(emittancePath, float4(f3_immEmittance, 1.0f),
                        &arrayEmittance, needsDegamma, isHDR))
            hpprintf("done.\n");
        else
            hpprintf("failed.\n");
    }
    return arrayEmittance;
}



static Ref<SurfaceMaterial> createSimplePBRMaterial(
    const std::filesystem::path &baseColor_opacityPath, const RGBSpectrum &immBaseColor, float immOpacity,
    const std::filesystem::path &occlusion_roughness_metallicPath,
    const float3 &immOcclusion_roughness_metallic) {

    bool needsDegamma = false;


    Ref<cudau::Array> arrayBaseColor_opacity;
    float4 immBaseColor_opacity(immBaseColor.r, immBaseColor.g, immBaseColor.b, immOpacity);
    if (!baseColor_opacityPath.empty()) {
        hpprintf("  Reading: %s ... ", baseColor_opacityPath.string().c_str());
        bool done = loadTexture(
            baseColor_opacityPath, immBaseColor_opacity,
            &arrayBaseColor_opacity, &needsDegamma);
        hpprintf(done ? "done.\n" : "failed.\n");
    }
    if (!arrayBaseColor_opacity) {
        createFx4ImmTexture(immBaseColor_opacity, true, &arrayBaseColor_opacity);
        needsDegamma = true;
    }

    auto texBaseColor_opacity = std::make_shared<Texture2D>(arrayBaseColor_opacity);
    if (needsDegamma)
        texBaseColor_opacity->setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    Ref<cudau::Array> arrayOcclusion_roughness_metallic;
    if (!occlusion_roughness_metallicPath.empty()) {
        hpprintf("  Reading: %s ... ", occlusion_roughness_metallicPath.string().c_str());
        bool done = loadTexture(
            occlusion_roughness_metallicPath, float4(immOcclusion_roughness_metallic, 0.0f),
            &arrayOcclusion_roughness_metallic, &needsDegamma);
        hpprintf(done ? "done.\n" : "failed.\n");
    }
    if (!arrayOcclusion_roughness_metallic) {
        createFx3ImmTexture(immOcclusion_roughness_metallic, true, &arrayOcclusion_roughness_metallic);
    }

    auto texOcclusion_roughness_metallic = std::make_shared<Texture2D>(arrayOcclusion_roughness_metallic);
    texOcclusion_roughness_metallic->setReadMode(cudau::TextureReadMode::NormalizedFloat);

    auto ret = std::make_shared<SimplePBRSurfaceMaterial>();
    g_scene.allocateSurfaceMaterial(ret);

    ret->set(texBaseColor_opacity, texOcclusion_roughness_metallic);

    return ret;
}



struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

static void computeFlattenedNodes(const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
                                  std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(Vector4D(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
                                 Vector4D(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
                                 Vector4D(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
                                 Vector4D(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * transpose(curXfm);
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    if (curNode->mNumMeshes > 0) {
        std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
        flattenedNodes.push_back(flattenedNode);
    }

    for (uint32_t cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

struct NormalMapInfo {
    Ref<Texture2D> texture;
    BumpMapTextureType bumpMapType;
};

struct GeometryGroupInstance {
    Ref<GeometryGroup> geomGroup;
    Matrix4x4 transform;
};

static void loadTriangleMesh(
    const std::filesystem::path &filePath, const Matrix4x4 &preTransform,
    std::vector<GeometryGroupInstance>* geomGroupInsts) {
    hpprintf("Reading: %s ... ", filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* aiscene = importer.ReadFile(
        filePath.string(),
        aiProcess_Triangulate
        | aiProcess_GenNormals
        | aiProcess_CalcTangentSpace
        /*| aiProcess_FlipUVs*/);
    if (!aiscene) {
        hpprintf("failed to load %s.\n", filePath.string().c_str());
        return;
    }
    hpprintf("done.\n");

    std::filesystem::path dirPath = filePath;
    dirPath.remove_filename();



    std::vector<Ref<SurfaceMaterial>> surfMats;
    std::vector<NormalMapInfo> normalMaps;
    for (uint32_t matIdx = 0; matIdx < aiscene->mNumMaterials; ++matIdx) {
        std::filesystem::path emittancePath;
        RGBSpectrum immEmittance(0.0f);

        const aiMaterial* aiMat = aiscene->mMaterials[matIdx];
        aiString strValue;
        float color[3];

        std::string matName;
        if (aiMat->Get(AI_MATKEY_NAME, strValue) == aiReturn_SUCCESS)
            matName = strValue.C_Str();
        hpprintf("%s:\n", matName.c_str());

        std::filesystem::path diffuseColorPath;
        RGBSpectrum immDiffuseColor;
        std::filesystem::path specularColorPath;
        RGBSpectrum immSpecularColor;
        float immSmoothness;
        if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
            diffuseColorPath = dirPath / strValue.C_Str();
        }
        else {
            if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = 0.0f;
                color[1] = 0.0f;
                color[2] = 0.0f;
            }
            immDiffuseColor = RGBSpectrum(color[0], color[1], color[2]);
        }

        if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
            specularColorPath = dirPath / strValue.C_Str();
        }
        else {
            if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = 0.0f;
                color[1] = 0.0f;
                color[2] = 0.0f;
            }
            immSpecularColor = RGBSpectrum(color[0], color[1], color[2]);
        }

        if (aiMat->Get(AI_MATKEY_SHININESS, &immSmoothness, nullptr) != aiReturn_SUCCESS)
            immSmoothness = 0.0f;
        immSmoothness = std::sqrt(immSmoothness);
        immSmoothness = immSmoothness / 11.0f/*30.0f*/;

        std::filesystem::path normalPath;
        if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_TEXTURE_NORMALS(0), strValue) == aiReturn_SUCCESS)
            normalPath = dirPath / strValue.C_Str();

        if (aiMat->Get(AI_MATKEY_TEXTURE_EMISSIVE(0), strValue) == aiReturn_SUCCESS)
            emittancePath = dirPath / strValue.C_Str();
        else if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS)
            immEmittance = RGBSpectrum(color[0], color[1], color[2]);

        // JP: diffuseテクスチャーとしてベースカラー + 不透明度
        //     specularテクスチャーとしてオクルージョン、ラフネス、メタリック
        //     が格納されていると仮定している。
        // EN: We assume diffuse texture as base color + opacity,
        //     specular texture as occlusion, roughness, metallic.
        Ref<SurfaceMaterial> surfMat = createSimplePBRMaterial(
            diffuseColorPath, immDiffuseColor, 1.0f,
            specularColorPath, float3(immSpecularColor.r, immSpecularColor.g, immSpecularColor.b));

        bool needsDegamma;
        bool isHDR;
        Ref<cudau::Array> emittanceArray =
            createEmittanceTexture(emittancePath, immEmittance, &needsDegamma, &isHDR);
        Ref<Texture2D> emittanceTex;
        if (emittanceArray) {
            emittanceTex = std::make_shared<Texture2D>(emittanceArray);
            if (needsDegamma)
                emittanceTex->setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);
            else if (isHDR)
                emittanceTex->setReadMode(cudau::TextureReadMode::ElementType);
            else
                emittanceTex->setReadMode(cudau::TextureReadMode::NormalizedFloat);
        }
        surfMat->setEmittance(emittanceTex);

        NormalMapInfo normalMapInfo;
        Ref<cudau::Array> normalArray = createNormalTexture(normalPath, &normalMapInfo.bumpMapType);
        normalMapInfo.texture = std::make_shared<Texture2D>(normalArray);
        normalMapInfo.texture->setReadMode(cudau::TextureReadMode::NormalizedFloat);

        surfMats.push_back(surfMat);
        normalMaps.push_back(normalMapInfo);
    }



    std::vector<Ref<Geometry>> geometries;
    for (uint32_t meshIdx = 0; meshIdx < aiscene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = aiscene->mMeshes[meshIdx];

        std::vector<shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];

            Normal3D normal(ain.x, ain.y, ain.z);
            normal.normalize();

            Vector3D tangent;
            if (aiMesh->mTangents) {
                aiVector3D aitc0Dir = aiMesh->mTangents[vIdx];
                tangent = Vector3D(aitc0Dir.x, aitc0Dir.y, aitc0Dir.z);
            }
            if (!aiMesh->mTangents || !tangent.allFinite()) {
                Vector3D bitangent;
                normal.makeCoordinateSystem(&tangent, &bitangent);
            }

            const aiVector3D ait = aiMesh->mTextureCoords[0] ?
                aiMesh->mTextureCoords[0][vIdx] :
                aiVector3D(0.0f, 0.0f, 0.0f);

            shared::Vertex v;
            v.position = Point3D(aip.x, aip.y, aip.z);
            v.normal = normal;
            v.tangent = tangent;
            v.texCoord = TexCoord2D(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            shared::Triangle tri;
            tri.indices[0] = aif.mIndices[0];
            tri.indices[1] = aif.mIndices[1];
            tri.indices[2] = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        auto vertexBuffer = std::make_shared<VertexBuffer>();
        vertexBuffer->onHost = std::move(vertices);
        vertexBuffer->onDevice.initialize(g_gpuEnv.cuContext, bufferType, vertexBuffer->onHost);

        const NormalMapInfo &normalMap = normalMaps[aiMesh->mMaterialIndex];
        const Ref<SurfaceMaterial> &surfMat = surfMats[aiMesh->mMaterialIndex];

        auto geom = std::make_shared<TriangleMeshGeometry>();
        g_scene.allocateGeometryInstance(geom);

        geom->set(vertexBuffer, triangles, normalMap.texture, normalMap.bumpMapType, surfMat);

        geometries.push_back(geom);
    }



    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(aiscene, preTransform, aiscene->mRootNode, flattenedNodes);
    //for (int i = 0; i < flattenedNodes.size(); ++i) {
    //    const Matrix4x4 &mat = flattenedNodes[i].transform;
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m00, mat.m01, mat.m02, mat.m03);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m10, mat.m11, mat.m12, mat.m13);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m20, mat.m21, mat.m22, mat.m23);
    //    hpprintf("%8.5f, %8.5f, %8.5f, %8.5f\n", mat.m30, mat.m31, mat.m32, mat.m33);
    //    hpprintf("\n");
    //}

    std::map<std::set<Ref<Geometry>>, Ref<GeometryGroup>> geomGroupMap;
    std::vector<Ref<GeometryGroup>> geometryGroups;
    for (int nodeIdx = 0; nodeIdx < flattenedNodes.size(); ++nodeIdx) {
        const FlattenedNode &node = flattenedNodes[nodeIdx];
        if (node.meshIndices.size() == 0)
            continue;

        std::set<Ref<Geometry>> srcGeoms;
        for (int i = 0; i < node.meshIndices.size(); ++i)
            srcGeoms.insert(geometries[node.meshIndices[i]]);
        Ref<GeometryGroup> geomGroup;
        if (geomGroupMap.count(srcGeoms) > 0) {
            geomGroup = geomGroupMap.at(srcGeoms);
        }
        else {
            geomGroup = std::make_shared<GeometryGroup>();
            g_scene.allocateGeometryAccelerationStructure(geomGroup);

            geomGroup->set(srcGeoms);

            geometryGroups.push_back(geomGroup);
        }

        GeometryGroupInstance geomGroupInst;
        geomGroupInst.geomGroup = geomGroup;
        geomGroupInst.transform = node.transform;
        geomGroupInsts->push_back(geomGroupInst);
    }
}

void createRectangle(
    float width, float depth,
    const RGBSpectrum &reflectance,
    const std::filesystem::path &emittancePath,
    const RGBSpectrum &immEmittance,
    const Matrix4x4 &transform,
    GeometryGroupInstance* geomGroupInst) {
    Ref<SurfaceMaterial> surfMat = createSimplePBRMaterial(
        "", reflectance, 1.0f,
        "", float3(0.0f, 0.5f, 0.0f));

    bool needsDegamma;
    bool isHDR;
    Ref<cudau::Array> emittanceArray =
        createEmittanceTexture(emittancePath, immEmittance, &needsDegamma, &isHDR);
    Ref<Texture2D> emittanceTex;
    if (emittanceArray) {
        emittanceTex = std::make_shared<Texture2D>(emittanceArray);
        if (needsDegamma)
            emittanceTex->setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);
        else if (isHDR)
            emittanceTex->setReadMode(cudau::TextureReadMode::ElementType);
        else
            emittanceTex->setReadMode(cudau::TextureReadMode::NormalizedFloat);
    }
    surfMat->setEmittance(emittanceTex);

    float hw = 0.5f * width;
    float hd = 0.5f * depth;
    std::vector<shared::Vertex> vertices = {
        {Point3D(-hw, 0.0f, -hd), Normal3D(0, -1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f)},
        {Point3D(hw, 0.0f, -hd), Normal3D(0, -1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f)},
        {Point3D(hw, 0.0f, hd), Normal3D(0, -1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f)},
        {Point3D(-hw, 0.0f, hd), Normal3D(0, -1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f)},
    };
    std::vector<shared::Triangle> triangles = {
        shared::Triangle{0, 1, 2},
        shared::Triangle{0, 2, 3},
    };

    NormalMapInfo normalMapInfo;
    Ref<cudau::Array> normalArray = createNormalTexture("", &normalMapInfo.bumpMapType);
    normalMapInfo.texture = std::make_shared<Texture2D>(normalArray);
    normalMapInfo.texture->setReadMode(cudau::TextureReadMode::NormalizedFloat);

    auto vertexBuffer = std::make_shared<VertexBuffer>();
    vertexBuffer->onHost = std::move(vertices);
    vertexBuffer->onDevice.initialize(g_gpuEnv.cuContext, bufferType, vertexBuffer->onHost);

    auto geom = std::make_shared<TriangleMeshGeometry>();
    g_scene.allocateGeometryInstance(geom);

    geom->set(vertexBuffer, triangles, normalMapInfo.texture, normalMapInfo.bumpMapType, surfMat);

    auto geomGroup = std::make_shared<GeometryGroup>();
    g_scene.allocateGeometryAccelerationStructure(geomGroup);

    geomGroup->set({ geom });

    geomGroupInst->geomGroup = geomGroup;
    geomGroupInst->transform = transform;
}



template <typename... Types>
void throwRuntimeErrorAtLine(bool expr, uint32_t line, const char* fmt, const Types &... args) {
    if (!expr) {
        std::string mFmt = std::string("l.%u: ") + fmt;
        _throwRuntimeError(mFmt.c_str(), line, args...);
    }
}

static const char* reInteger = R"(([+-]?\d+))";
static const char* reReal = R"(([+-]?(?:\d+\.?\d*|\.\d+)))";
static const char* reString = R"((\S+?))";
static const char* reQuotedPath = R"*("(.+?)")*";

static std::regex makeRegex(const std::vector<std::string> &tokens) {
    const char* reSpace = R"([ \t]+?)";
    std::string pattern = tokens[0];
    for (uint32_t i = 1; i < tokens.size(); ++i)
        pattern += reSpace + tokens[i];
    pattern += "$";
    return std::move(std::regex(pattern));
}

static std::smatch testRegex(
    const std::regex &re, const char* cmd, uint32_t lineIndex, const std::string &line) {
    std::smatch m;
    throwRuntimeErrorAtLine(
        std::regex_search(line, m, re), lineIndex + 1,
        "failed to parse \"%s\" command: %s",
        cmd, line.c_str());
    return std::move(m);
}

struct MeshInfo {
    struct File {
        std::filesystem::path path;
        float scale;
    };
    struct Rectangle {
        float dimX;
        float dimZ;
        RGBSpectrum emittance;
    };

    std::variant<
        File,
        Rectangle
    > body;
};

struct InstanceInfo {
    std::string meshName;
    Point3D nextKeyPosition;
    float nextKeyScale;
    Quaternion nextKeyOrientation;
    std::vector<KeyInstanceState> keyStates;
};

struct CameraInfo {
    Point3D nextKeyPosition;
    Point3D nextKeyLookAt;
    Vector3D nextKeyUp;
    float nextKeyFovY;
    std::vector<KeyCameraState> keyStates;
};

struct SceneLoadingContext {
    uint32_t lineIndex;
    std::string line;

    uint32_t imageWidth;
    uint32_t imageHeight;
    float timeBegin;
    float timeEnd;
    uint32_t fps;
    std::map<std::string, MeshInfo> meshInfos;
    std::map<std::string, InstanceInfo> instInfos;
    std::map<std::string, CameraInfo> camInfos;
    std::vector<ActiveCameraInfo> activeCamInfos;
};

using lineFunc = std::function<void(SceneLoadingContext &context)>;
std::map<std::string, lineFunc> processors = {
    {
        "image",
        [](SceneLoadingContext &context) {
            static const char* cmd = "image";
            static const std::regex re = makeRegex({cmd, reInteger, reInteger});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            int32_t w = std::stoi(m[1].str().c_str());
            int32_t h = std::stoi(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                w > 0 && h > 0, context.lineIndex + 1,
                "Invalid image size: %d x %d", w, h);

            context.imageWidth = static_cast<uint32_t>(w);
            context.imageHeight = static_cast<uint32_t>(h);
        }
    },
    {
        "time",
        [](SceneLoadingContext &context) {
            static const char* cmd = "time";
            static const std::regex re = makeRegex({cmd, reReal, reReal, reInteger});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            float timeBegin = std::stof(m[1].str().c_str());
            float timeEnd = std::stof(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                timeBegin >= 0.0f && timeBegin <= timeEnd, context.lineIndex + 1,
                "Invalid timeBegin/End: %f - %f", timeBegin, timeEnd);

            context.timeBegin = timeBegin;
            context.timeEnd = timeEnd;
            context.fps = static_cast<uint32_t>(std::stoi(m[3].str().c_str()));
        }
    },
    {
        "mesh",
        [](SceneLoadingContext &context) {
            static const char* cmd = "mesh";
            static const std::regex re = makeRegex({cmd, reQuotedPath, reReal, reString});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string meshName = m[3].str();
            throwRuntimeErrorAtLine(
                !context.meshInfos.contains(meshName), context.lineIndex + 1,
                "Mesh %s has been already created.",
                meshName.c_str());

            std::filesystem::path meshFilePath = m[1].str();
            throwRuntimeErrorAtLine(
                std::filesystem::exists(meshFilePath), context.lineIndex + 1,
                "Mesh %s does not exist.", meshFilePath.string().c_str());
            float scale = std::stof(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                scale > 0.0f, context.lineIndex + 1,
                "Invalid mesh scale: %f", scale);

            MeshInfo::File meshInfo;
            meshInfo.path = meshFilePath;
            meshInfo.scale = scale;
            context.meshInfos[meshName].body = meshInfo;
        }
    },
    {
        "rect",
        [](SceneLoadingContext &context) {
            static const char* cmd = "rect";
            static const std::regex re = makeRegex({cmd, reReal, reReal, reString});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string meshName = m[3].str();
            throwRuntimeErrorAtLine(
                !context.meshInfos.contains(meshName), context.lineIndex + 1,
                "Mesh %s has been already created.",
                meshName.c_str());

            float dimX = std::stof(m[1].str().c_str());
            float dimZ = std::stof(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                dimX > 0.0f && dimZ > 0.0f, context.lineIndex + 1,
                "Invalid rectangle size: %f x %f", dimX, dimZ);

            MeshInfo::Rectangle rectInfo;
            rectInfo.dimX = dimX;
            rectInfo.dimZ = dimZ;
            rectInfo.emittance = RGBSpectrum::Zero();
            context.meshInfos[meshName].body = rectInfo;
        }
    },
    {
        "emittance",
        [](SceneLoadingContext &context) {
            static const char* cmd = "emittance";
            static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string meshName = m[1].str();
            throwRuntimeErrorAtLine(
                context.meshInfos.contains(meshName), context.lineIndex + 1,
                "Mesh %s does not exist.",
                meshName.c_str());
            MeshInfo &meshInfo = context.meshInfos.at(meshName);
            throwRuntimeErrorAtLine(
                std::holds_alternative<MeshInfo::Rectangle>(meshInfo.body), context.lineIndex + 1,
                "Emittance cannot be assigned to this mesh.",
                meshName.c_str());

            RGBSpectrum emittance(
                std::stof(m[2].str().c_str()),
                std::stof(m[3].str().c_str()),
                std::stof(m[4].str().c_str()));
            throwRuntimeErrorAtLine(
                emittance.allNonNegativeFinite(), context.lineIndex + 1,
                "Invalid emittance: (%f, %f, %f)", emittance.r, emittance.g, emittance.b);

            auto &rectInfo = std::get<MeshInfo::Rectangle>(meshInfo.body);
            rectInfo.emittance = emittance;
        }
    },
    {
        "inst",
        [](SceneLoadingContext &context) {
            static const char* cmd = "inst";
            static const std::regex re = makeRegex({cmd, reString, reString});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string meshName = m[1].str();
            throwRuntimeErrorAtLine(
                context.meshInfos.contains(meshName), context.lineIndex + 1,
                "Mesh %s does not exist.",
                meshName.c_str());
            std::string instName = m[2].str();
            throwRuntimeErrorAtLine(
                !context.instInfos.contains(instName), context.lineIndex + 1,
                "Instance %s has been already created.",
                instName.c_str());
            throwRuntimeErrorAtLine(
                !context.camInfos.contains(instName), context.lineIndex + 1,
                "Camera with the same name %s exists.",
                instName.c_str());

            InstanceInfo instInfo;
            instInfo.meshName = meshName;
            instInfo.nextKeyScale = 1.0f;
            instInfo.nextKeyOrientation = Quaternion::Identity();
            instInfo.nextKeyPosition = Point3D::Zero();
            context.instInfos[instName] = instInfo;
        }
    },
    {
        "camera",
        [](SceneLoadingContext &context) {
            static const char* cmd = "camera";
            static const std::regex re = makeRegex({cmd, reString});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string camName = m[1].str();
            throwRuntimeErrorAtLine(
                !context.camInfos.contains(camName), context.lineIndex + 1,
                "Camera %s has been already created.",
                camName.c_str());
            throwRuntimeErrorAtLine(
                !context.instInfos.contains(camName), context.lineIndex + 1,
                "Instance with the same name %s exists.",
                camName.c_str());

            if (context.camInfos.empty()) {
                ActiveCameraInfo activeCamInfo;
                activeCamInfo.timePoint = 0.0f;
                context.activeCamInfos.push_back(activeCamInfo);
            }

            CameraInfo camInfo;
            camInfo.nextKeyPosition = Point3D(0, 1, 5);
            camInfo.nextKeyLookAt = Point3D(0, 0, 0);
            camInfo.nextKeyUp = Vector3D(0, 1, 0);
            camInfo.nextKeyFovY = 60 * pi_v<float> / 180;
            context.camInfos[camName] = camInfo;
        }
    },
    {
        "scale",
        [](SceneLoadingContext &context) {
            static const char* cmd = "scale";
            static const std::regex re = makeRegex({cmd, reString, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string instName = m[1].str();
            throwRuntimeErrorAtLine(
                context.instInfos.contains(instName), context.lineIndex + 1,
                "Instance %s does not exist.",
                instName.c_str());

            context.instInfos.at(instName).nextKeyScale = std::stof(m[2].str().c_str());
        }
    },
    {
        "rotate",
        [](SceneLoadingContext &context) {
            static const char* cmd = "rotate";
            static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string instName = m[1].str();
            throwRuntimeErrorAtLine(
                context.instInfos.contains(instName), context.lineIndex + 1,
                "Instance %s does not exist.",
                instName.c_str());

            context.instInfos.at(instName).nextKeyOrientation = qRotate(
                std::stof(m[5].str().c_str()) * pi_v<float> / 180,
                std::stof(m[2].str().c_str()),
                std::stof(m[3].str().c_str()),
                std::stof(m[4].str().c_str()));
        }
    },
    {
        "trans",
        [](SceneLoadingContext &context) {
            static const char* cmd = "trans";
            static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string tgtName = m[1].str();
            bool isInst = context.instInfos.contains(tgtName);
            bool isCam = context.camInfos.contains(tgtName);
            throwRuntimeErrorAtLine(
                isInst || isCam, context.lineIndex + 1,
                "Instance/Camera %s does not exist.",
                tgtName.c_str());

            Point3D position(
                std::stof(m[2].str().c_str()),
                std::stof(m[3].str().c_str()),
                std::stof(m[4].str().c_str()));

            if (isInst)
                context.instInfos.at(tgtName).nextKeyPosition = position;
            else
                context.camInfos.at(tgtName).nextKeyPosition = position;
        }
    },
    {
        "fovy",
        [](SceneLoadingContext &context) {
            static const char* cmd = "fovy";
            static const std::regex re = makeRegex({cmd, reString, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string camName = m[1].str();
            throwRuntimeErrorAtLine(
                context.camInfos.contains(camName), context.lineIndex + 1,
                "Camera %s does not exist.",
                camName.c_str());

            float fovY = std::stof(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                fovY > 0.0f, context.lineIndex + 1,
                "Invalid fovY: %f", fovY);

            context.camInfos.at(camName).nextKeyFovY = fovY;
        }
    },
    {
        "lookat",
        [](SceneLoadingContext &context) {
            static const char* cmd = "lookat";
            static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string camName = m[1].str();
            throwRuntimeErrorAtLine(
                context.camInfos.contains(camName), context.lineIndex + 1,
                "Camera %s does not exist.",
                camName.c_str());

            context.camInfos.at(camName).nextKeyLookAt = Point3D(
                std::stof(m[2].str().c_str()),
                std::stof(m[3].str().c_str()),
                std::stof(m[4].str().c_str()));
        }
    },
    {
        "addkey",
        [](SceneLoadingContext &context) {
            static const char* cmd = "addkey";
            static const std::regex re = makeRegex({cmd, reString, reReal, reString, reString});
            std::smatch m = testRegex(re, cmd, context.lineIndex, context.line);

            std::string tgtName = m[1].str();
            bool isInst = context.instInfos.contains(tgtName);
            bool isCam = context.camInfos.contains(tgtName);
            throwRuntimeErrorAtLine(
                isInst || isCam, context.lineIndex + 1,
                "Instance/Camera %s does not exist.",
                tgtName.c_str());

            float timePoint = std::stof(m[2].str().c_str());
            throwRuntimeErrorAtLine(
                timePoint >= 0.0f, context.lineIndex + 1,
                "Invalid timePoint: %f", timePoint);

            if (isInst) {
                InstanceInfo &instInfo = context.instInfos.at(tgtName);
                KeyInstanceState state;
                state.timePoint = timePoint;
                state.scale = instInfo.nextKeyScale;
                state.orientation = instInfo.nextKeyOrientation;
                state.position = instInfo.nextKeyPosition;
                instInfo.keyStates.push_back(state);
            }
            else {
                CameraInfo &camInfo = context.camInfos.at(tgtName);
                KeyCameraState state;
                state.timePoint = timePoint;
                state.position = camInfo.nextKeyPosition;
                state.lookAt = camInfo.nextKeyLookAt;
                state.up = camInfo.nextKeyUp;
                state.fovY = camInfo.nextKeyFovY;
                camInfo.keyStates.push_back(state);
            }
        }
    },
};



void loadScene(const std::filesystem::path &sceneFilePath, RenderConfigs* renderConfigs) {
    throwRuntimeError(std::filesystem::exists(sceneFilePath), "Scene file does not exist!");

    std::vector<std::string> lines;
    {
        const auto trim = []
        (const std::string &str,
         const std::string &whitespace = " \t") -> std::string {
             const auto strBegin = str.find_first_not_of(whitespace);
             if (strBegin == std::string::npos)
                 return ""; // no content

             const auto strEnd = str.find_last_not_of(whitespace);
             const auto strRange = strEnd - strBegin + 1;

             return str.substr(strBegin, strRange);
        };

        std::stringstream ss(readTxtFile(sceneFilePath));
        std::string line;
        while (std::getline(ss, line, '\n')) {
            if (!line.empty() && line.back() == '\r')
                line = line.substr(0, line.size() - 1);
            line = trim(line);
            //if (line.empty())
            //    continue;
            lines.push_back(line);
        }
    }

    SceneLoadingContext context;
    context.lineIndex = -1;
    for (const std::string &line : lines) {
        ++context.lineIndex;
        if (line.empty())
            continue;
        size_t nextSpacePos = line.find_first_of(" \t");
        std::string firstToken = nextSpacePos != std::string::npos ?
            line.substr(0, nextSpacePos) : line;
        throwRuntimeErrorAtLine(
            processors.contains(firstToken), context.lineIndex + 1,
            "Unknown token found: \"%s\"",
            firstToken.c_str());
        context.line = line;
        processors.at(firstToken)(context);
    }

    renderConfigs->imageWidth = context.imageWidth;
    renderConfigs->imageHeight = context.imageHeight;
    renderConfigs->timeBegin = context.timeBegin;
    renderConfigs->timeEnd = context.timeEnd;
    renderConfigs->fps = context.fps;

    // JP: カメラを生成する。
    for (auto &it : context.camInfos) {
        const std::string &camName = it.first;
        CameraInfo &camInfo = it.second;
        std::vector<KeyCameraState> states = std::move(camInfo.keyStates);
        std::sort(
            states.begin(), states.end(),
            [](const KeyCameraState &a, const KeyCameraState &b) {
                return a.timePoint < b.timePoint;
            });
        auto camera = std::make_shared<Camera>(std::move(states));
        renderConfigs->cameras[camName] = camera;
    }

    // JP: アクティブカメラの切替情報を得る。
    std::sort(
        context.activeCamInfos.begin(), context.activeCamInfos.end(),
        [](const ActiveCameraInfo &a, const ActiveCameraInfo &b) {
            return a.timePoint < b.timePoint;
        });
    renderConfigs->activeCameraInfos = std::move(context.activeCamInfos);

    // JP: メッシュデータを読み込み、テクスチャーやマテリアル、ジオメトリとジオメトリグループ
    //     を生成する。1つのメッシュデータからは複数のジオメトリグループ(とトランスフォームの組)
    //     が作られる。
    std::unordered_map<std::string, std::vector<GeometryGroupInstance>> meshNameToGeomGroupInsts;
    for (auto &it : context.meshInfos) {
        const std::string &meshName = it.first;
        const MeshInfo &meshInfo = it.second;
        std::vector<GeometryGroupInstance> geomGroupInsts;
        if (std::holds_alternative<MeshInfo::File>(meshInfo.body)) {
            const MeshInfo::File &fileInfo = std::get<MeshInfo::File>(meshInfo.body);
            loadTriangleMesh(fileInfo.path, scale4x4<float>(fileInfo.scale), &geomGroupInsts);
        }
        else if (std::holds_alternative<MeshInfo::Rectangle>(meshInfo.body)) {
            const MeshInfo::Rectangle &rectInfo = std::get<MeshInfo::Rectangle>(meshInfo.body);
            GeometryGroupInstance geomGroupInst;
            createRectangle(
                rectInfo.dimX, rectInfo.dimZ, RGBSpectrum(0.9f), "", rectInfo.emittance,
                Matrix4x4::Identity(), &geomGroupInst);
            geomGroupInsts.push_back(geomGroupInst);
        }
        meshNameToGeomGroupInsts[meshName] = std::move(geomGroupInsts);
    }

    // JP: インスタンスを生成する。
    for (auto &it : context.instInfos) {
        const std::string &instName = it.first;
        InstanceInfo &instInfo = it.second;
        std::vector<KeyInstanceState> states = std::move(instInfo.keyStates);
        std::sort(
            states.begin(), states.end(),
            [](const KeyInstanceState &a, const KeyInstanceState &b) {
                return a.timePoint < b.timePoint;
            });
        const std::vector<GeometryGroupInstance> &geomGroupInsts =
            meshNameToGeomGroupInsts.at(instInfo.meshName);
        for (const GeometryGroupInstance &geomGroupInst : geomGroupInsts) {
            auto inst = std::make_shared<Instance>();
            g_scene.allocateInstance(inst);
            inst->setGeometryGroup(geomGroupInst.geomGroup, geomGroupInst.transform);
            inst->setKeyStates(states);
        }
    }
}

}
