#pragma once

#include "../common/basic_types.h"

namespace rtc8::shared {

struct PixelFeature {
    RGBSpectrum noisyColor;
    RGBSpectrum albedo;
    Normal3D normal;
    //float dx, dy;
};

struct TrainingItem {
    static constexpr uint32_t size = 7;
    PixelFeature neighbors[pow2(size)];
};

} // namespace rtc8::shared
