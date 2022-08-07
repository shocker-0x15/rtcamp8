#define PURE_CUDA
#include "../renderer_shared.h"

using namespace rtc8;
using namespace rtc8::shared;
using namespace rtc8::device;

CUDA_CONSTANT_MEM ArHosekSkyModelState states[ArHosekSkyModelCMFSet::numBands];
CUDA_CONSTANT_MEM ArHosekSkyModelCMFSet cmfSet;
CUDA_CONSTANT_MEM float* solarDatasets;
CUDA_CONSTANT_MEM float* limbDarkeningDatasets;

// ----------------------------------------------------------------
// JP: 以下の関数はext/hosek_wilkie_2013のコードをCUDA用に移植したもの。

CUDA_DEVICE_FUNCTION CUDA_INLINE float ArHosekSkyModel_GetRadianceInternal(
    ArHosekSkyModelConfiguration configuration,
    float theta,
    float gamma) {
    const float expM = std::exp(configuration[4] * gamma);
    const float rayM = std::cos(gamma) * std::cos(gamma);
    const float mieM = (1.0f + std::cos(gamma) * std::cos(gamma)) / std::pow((1.0f + configuration[8] * configuration[8] - 2.0f * configuration[8] * cos(gamma)), 1.5f);
    const float zenith = std::sqrt(std::cos(theta));

    return (1.0f + configuration[0] * std::exp(configuration[1] / (std::cos(theta) + 0.01f))) *
        (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float arhosekskymodel_radiance(
    ArHosekSkyModelState* state,
    float theta,
    float gamma,
    float wavelength) {
    int low_wl = (wavelength - 320.0f) / 40.0f;

    if (low_wl < 0 || low_wl >= 11)
        return 0.0f;

    float interp = std::fmod((wavelength - 320.0f) / 40.0f, 1.0f);

    float val_low =
        ArHosekSkyModel_GetRadianceInternal(
            state->configs[low_wl],
            theta,
            gamma)
        * state->radiances[low_wl]
        * state->emission_correction_factor_sky[low_wl];

    if (interp < 1e-6f)
        return val_low;

    float result = (1.0f - interp) * val_low;

    if (low_wl + 1 < 11) {
        result +=
            interp
            * ArHosekSkyModel_GetRadianceInternal(
                state->configs[low_wl + 1],
                theta,
                gamma)
            * state->radiances[low_wl + 1]
            * state->emission_correction_factor_sky[low_wl + 1];
    }

    return result;
}

static constexpr int pieces = 45;
static constexpr int order = 4;

CUDA_DEVICE_FUNCTION CUDA_INLINE float arhosekskymodel_sr_internal(
    ArHosekSkyModelState* state,
    int turbidity,
    int wl,
    float elevation) {
    int pos = (int)(std::pow(2.0f * elevation / pi_v<float>, 1.0f / 3.0f) * pieces); // floor

    if (pos > 44)
        pos = 44;

    const float break_x =
        std::pow(((float)pos / (float)pieces), 3.0f) * (pi_v<float> * 0.5f);

    const float* coefs =
        &solarDatasets[1800 * wl + (order * pieces * turbidity + order * (pos + 1) - 1)];

    float res = 0.0f;
    const float x = elevation - break_x;
    float x_exp = 1.0f;

    for (int i = 0; i < order; ++i) {
        res += x_exp * *coefs--;
        x_exp *= x;
    }

    return res * state->emission_correction_factor_sun[wl];
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float arhosekskymodel_solar_radiance_internal2(
    ArHosekSkyModelState* state,
    float wavelength,
    float elevation,
    float gamma) {
    //assert(
    //    wavelength >= 320.0f
    //    && wavelength <= 720.0f
    //    && state->turbidity >= 1.0f
    //    && state->turbidity <= 10.0f
    //);

    int turb_low = (int)state->turbidity - 1;
    float turb_frac = state->turbidity - (float)(turb_low + 1);

    if (turb_low == 9) {
        turb_low = 8;
        turb_frac = 1.0f;
    }

    int wl_low = (int)((wavelength - 320.0f) / 40.0f);
    float wl_frac = std::fmod(wavelength, 40.0f) / 40.0f;

    if (wl_low == 10) {
        wl_low = 9;
        wl_frac = 1.0f;
    }

    float direct_radiance =
        (1.0f - turb_frac)
        * ((1.0f - wl_frac)
           * arhosekskymodel_sr_internal(
               state,
               turb_low,
               wl_low,
               elevation)
           + wl_frac
           * arhosekskymodel_sr_internal(
               state,
               turb_low,
               wl_low + 1,
               elevation)
           )
        + turb_frac
        * ((1.0f - wl_frac)
           * arhosekskymodel_sr_internal(
               state,
               turb_low + 1,
               wl_low,
               elevation)
           + wl_frac
           * arhosekskymodel_sr_internal(
               state,
               turb_low + 1,
               wl_low + 1,
               elevation)
           );

    float ldCoefficient[6];

    for (int i = 0; i < 6; i++)
        ldCoefficient[i] =
        (1.0f - wl_frac) * limbDarkeningDatasets[6 * wl_low + i]
        + wl_frac * limbDarkeningDatasets[6 * (wl_low + 1) + i];

    // sun distance to diameter ratio, squared

    const float sol_rad_sin = std::sin(state->solar_radius);
    const float ar2 = 1 / (sol_rad_sin * sol_rad_sin);
    const float singamma = std::sin(gamma);
    float sc2 = 1.0f - ar2 * singamma * singamma;
    if (sc2 < 0.0f) sc2 = 0.0f;
    float sampleCosine = std::sqrt(sc2);

    //   The following will be improved in future versions of the model:
    //   here, we directly use fitted 5th order polynomials provided by the
    //   astronomical community for the limb darkening effect. Astronomers need
    //   such accurate fittings for their predictions. However, this sort of
    //   accuracy is not really needed for CG purposes, so an approximated
    //   dataset based on quadratic polynomials will be provided in a future
    //   release.

    float darkeningFactor =
        ldCoefficient[0]
        + ldCoefficient[1] * sampleCosine
        + ldCoefficient[2] * std::pow(sampleCosine, 2.0f)
        + ldCoefficient[3] * std::pow(sampleCosine, 3.0f)
        + ldCoefficient[4] * std::pow(sampleCosine, 4.0f)
        + ldCoefficient[5] * std::pow(sampleCosine, 5.0f);

    direct_radiance *= darkeningFactor;

    return direct_radiance;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float arhosekskymodel_solar_radiance(
    ArHosekSkyModelState* state,
    float theta,
    float gamma,
    float wavelength) {
    float direct_radiance =
        arhosekskymodel_solar_radiance_internal2(
            state,
            wavelength,
            ((pi_v<float> / 2.0f) - theta),
            gamma);

    float inscattered_radiance =
        arhosekskymodel_radiance(
            state,
            theta,
            gamma,
            wavelength);

    return  direct_radiance + inscattered_radiance;
}

// ----------------------------------------------------------------



CUDA_DEVICE_KERNEL void generateArHosekSkyEnvironmentalTexture(
    optixu::NativeBlockBuffer2D<RGBSpectrum> dstTex, uint2 imageSize, Vector3D sunDirection) {
    uint2 pix = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                           blockDim.y * blockIdx.y + threadIdx.y);
    if (pix.x >= imageSize.x || pix.y > imageSize.y)
        return;

    float theta = pi_v<float> * (pix.y + 0.5f) / imageSize.y;
    RGBSpectrum rgb = RGBSpectrum::Zero();
    if (theta < 0.5f * pi_v<float>) {
        float phi = 2 * pi_v<float> *(pix.x + 0.5f) / imageSize.x;
        Vector3D viewVec = Vector3D::fromPolarYUp(phi, theta);
        float gamma = std::acos(clamp(dot(viewVec, sunDirection), -1.0f, 1.0f));
        float xyz[3] = { 0, 0, 0 };
        for (int bandIdx = 0; bandIdx < ArHosekSkyModelCMFSet::numBands; ++bandIdx) {
            ArHosekSkyModelState &state = states[bandIdx];
            float centerWavelength = cmfSet.centerWavelengths[bandIdx];
            float value;
            if (gamma < state.solar_radius)
                value = arhosekskymodel_solar_radiance(&state, theta, gamma, centerWavelength);
            else
                value = arhosekskymodel_radiance(&state, theta, gamma, centerWavelength);
            xyz[0] += cmfSet.xs[bandIdx] * value;
            xyz[1] += cmfSet.ys[bandIdx] * value;
            xyz[2] += cmfSet.zs[bandIdx] * value;
        }
        xyz[0] /= cmfSet.integralCmf;
        xyz[1] /= cmfSet.integralCmf;
        xyz[2] /= cmfSet.integralCmf;
        rgb = RGBSpectrum::fromXYZ(xyz);
    }
    dstTex.write(pix, rgb);
}
