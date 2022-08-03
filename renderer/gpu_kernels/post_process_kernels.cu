#define PURE_CUDA
#include "../renderer_shared.h"

using namespace rtc8;
using namespace rtc8::shared;
using namespace rtc8::device;



CUDA_DEVICE_FUNCTION CUDA_INLINE RGBSpectrum applySimpleToneMap(
    const RGBSpectrum &input, float brightness) {
    RGBSpectrum color = input;
    float lum = input.luminance();
    if (lum > 0.0f) {
        float lumT = 1 - std::exp(-brightness * lum);
        color *= lumT / lum;
    }
    return color;
}

CUDA_DEVICE_KERNEL void applyToneMap() {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    RGBSpectrum accumResult = plp.s->accumBuffer.read(launchIndex);
    RGBSpectrum output = applySimpleToneMap(accumResult, /*plp.f->brighness*/1.0f);

    plp.f->outputBuffer.write(launchIndex, output);
}