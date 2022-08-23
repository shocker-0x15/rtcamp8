#include "network_interface.h"
#include "../common/common_shared.h"
#include "denoiser_shared.h"

#include <cuda_runtime.h>

#if !defined(TCNN_MIN_GPU_ARCH)
#   define TCNN_MIN_GPU_ARCH 86
#endif
#include <tiny-cuda-nn/config.h>
#include <memory>
#include <fstream>

namespace rtc8 {

using namespace tcnn;
using precision_t = network_precision_t;

constexpr static uint32_t numInputDims = sizeof(shared::TrainingItem) / sizeof(float);
// denoised color: 3
constexpr static uint32_t numOutputDims = 3;

class Denoiser::Priv {
    std::shared_ptr<Loss<precision_t>> loss;
    std::shared_ptr<Optimizer<precision_t>> optimizer;
    std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;

    std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

public:
    friend class Denoiser;

    Priv() {}
};



Denoiser::Denoiser() {
    m = new Priv();
}

Denoiser::~Denoiser() {
    delete m;
}

void Denoiser::initialize(uint32_t mlpWidth, uint32_t numHiddenLayers, float learningRate) {
    json config = {
        {"loss", {
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            {"otype", "Adam"},
            {"learning_rate", learningRate},
            {"beta1", 0.9f},
            {"beta2", 0.99f},
            {"l2_reg", 1e-6f},
        }},
        {"network", {
            {"otype", "CutlassMLP"},
            {"n_neurons", mlpWidth},
            {"n_hidden_layers", numHiddenLayers},
            {"activation", "ReLU"},
            {"output_activation", "None"},
        }},
        {"encoding", {
            {"otype", "Identity"}
        }}
    };

    m->loss.reset(create_loss<precision_t>(config.value("loss", json::object())));
    m->optimizer.reset(create_optimizer<precision_t>(config.value("optimizer", json::object())));
    m->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        numInputDims, numOutputDims,
        config.value("encoding", json::object()),
        config.value("network", json::object()));

    m->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
        m->network, m->optimizer, m->loss);
}

void Denoiser::finalize() {
    free_all_gpu_memory_arenas();
    m->trainer = nullptr;
    m->network = nullptr;
    m->optimizer = nullptr;
    m->loss = nullptr;
}

void Denoiser::infer(
    CUstream stream, float* inputData, uint32_t numData, float* predictionData) {
    Assert((numData & 0x7F) == 0, "numData must be a multiple of 128.");
    GPUMatrix<float> inputs(inputData, numInputDims, numData);
    GPUMatrix<float> predictions(predictionData, numOutputDims, numData);
    m->network->inference(stream, inputs, predictions);
}

void Denoiser::train(
    CUstream stream, float* inputData, float* targetData, uint32_t numData, float* lossOnCPU) {
    Assert((numData & 0x7F) == 0, "numData must be a multiple of 128.");
    GPUMatrix<float> inputs(inputData, numInputDims, numData);
    GPUMatrix<float> targets(targetData, numOutputDims, numData);
    auto context = m->trainer->training_step(stream, inputs, targets);
    if (lossOnCPU)
        *lossOnCPU = m->trainer->loss(stream, *context);
}

void Denoiser::serialize(const std::filesystem::path &filepath) const {
    json data = m->trainer->serialize(false);
    std::ofstream f(filepath, std::ios::out | std::ios::binary);
    json::to_msgpack(data, f);
}

void Denoiser::deserialize(const std::filesystem::path &filepath) {
    std::ifstream f(filepath, std::ios::in | std::ios::binary);
    json data = json::from_msgpack(f);
    m->trainer->deserialize(data);
}

} // namespace rtc8
