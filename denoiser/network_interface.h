#pragma once

#include <cuda.h>
#include <cstdint>
#include <filesystem>

namespace rtc8 {

// JP: サンプルプログラム全体をnvcc経由でコンパイルしないといけない状況を避けるため、
//     pimplイディオムによってtiny-cuda-nnをcpp側に隔離する。
// EN: Isolate the tiny-cuda-nn into the cpp side by pimpl idiom to avoid the situation where
//     the entire sample program needs to be compiled via nvcc.
class Denoiser {
    class Priv;
    Priv* m = nullptr;

public:
    Denoiser();
    ~Denoiser();

    void initialize(uint32_t mlpWidth, uint32_t numHiddenLayers, float learningRate);
    void finalize();

    void infer(CUstream stream, float* inputData, uint32_t numData, float* predictionData);
    void train(CUstream stream, float* inputData, float* targetData, uint32_t numData,
               float* lossOnCPU = nullptr);

    void serialize(const std::filesystem::path &filepath) const;
    void deserialize(const std::filesystem::path &filepath);
};

} // namespace rtc8
