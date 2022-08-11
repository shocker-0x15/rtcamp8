#include "renderer_kernel_common.h"

static constexpr bool debugVisualizeBaseColor = false;

CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTrace)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex());

    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    Point3D rayOrigin;
    Vector3D rayDirection;
    float dirPDensity;
    {
        const PerspectiveCamera &camera = plp.f->camera;
        float px = (launchIndex.x + rng.getFloat0cTo1o()) / plp.s->imageSize.x;
        float py = (launchIndex.y + rng.getFloat0cTo1o()) / plp.s->imageSize.y;
        float vh = 2 * std::tan(camera.fovY * 0.5f);
        float vw = camera.aspect * vh;
        float sensorArea = vw * vh; // normalized

        rayOrigin = camera.position;
        Vector3D localRayDir = Vector3D(vw * (-0.5f + px), vh * (0.5f - py), -1).normalize();
        rayDirection = normalize(camera.orientation.toMatrix3x3() * localRayDir);
        dirPDensity = 1 / (pow3(std::fabs(localRayDir.z)) * sensorArea);
    }

    RGBSpectrum contribution = RGBSpectrum::Zero();
    RGBSpectrum throughput = RGBSpectrum::One() / dirPDensity;
    float initImportance = throughput.luminance();
    uint32_t pathLength = 0;
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
            const float densityCoeff = plp.s->densityCoeff;
            const float majorant = plp.s->majorant;
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
                contribution += throughput * radiance * misWeight;
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

        if (!volEventHappens) {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            // Implicit light sampling
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
                contribution += throughput * emittance * (misWeight / pi_v<float>);
            }
        }

        // Russian roulette
        float continueProb = std::fmin(throughput.luminance() / initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            break;
        throughput /= continueProb;

        constexpr float scatteringAlbedo = 0.99f;
        BSDF bsdf;
        BSDFQuery bsdfQuery;
        if (volEventHappens) {
            bsdf.setup(scatteringAlbedo);
            bsdfQuery = BSDFQuery();
        }
        else {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            bsdf.setup(surfMat, interPt.asSurf.texCoord);
            bsdfQuery = BSDFQuery(vOutLocal, interPt.toLocal(interPt.asSurf.geometricNormal));
        }

        if constexpr (debugVisualizeBaseColor) {
            contribution += throughput * bsdf.evaluateDHReflectanceEstimate(bsdfQuery);
            break;
        }

        contribution += throughput * performNextEventEstimation(interPt, bsdf, bsdfQuery, rng);

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
    }

    plp.s->rngBuffer.write(launchIndex, rng);

    if (!contribution.allNonNegativeFinite()) {
        printf("Store Cont.: %4u, %4u: (%g, %g, %g)\n",
               launchIndex.x, launchIndex.y, rgbprint(contribution));
        return;
    }

    RGBSpectrum prevResult = RGBSpectrum::Zero();
    if (plp.f->numAccumFrames > 0)
        prevResult = plp.s->accumBuffer.read(launchIndex);
    float curFrameWeight = 1.0f / (plp.f->numAccumFrames + 1);
    RGBSpectrum result = (1 - curFrameWeight) * prevResult + curFrameWeight * contribution;
    plp.s->accumBuffer.write(launchIndex, result);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTrace)() {
    uint32_t instSlot = optixGetInstanceId();
    auto sbtr = HitGroupSBTRecordData::get();
    auto hp = HitPointParameter::get();
    float dist = optixGetRayTmax();

    ClosestRaySignature::set(&instSlot, &sbtr.geomInstSlot, &hp.primIndex, &hp.b1, &hp.b2, &dist);
}
