#include "denoiser_shared.h"
#include "../common/common_host.h"

#include "../common/stopwatch.h"

#include "network_interface.h"
#include <json/json.hpp>
#include "tinyexr.h"

namespace rtc8 {

using json = nlohmann::json;

enum class AppMode {
    Training = 0,
    Inference
};

static AppMode g_appMode = AppMode::Inference;
static std::filesystem::path g_exeDir = getExecutableDirectory();
static std::filesystem::path g_datasetConfigPath;
static std::filesystem::path g_datasetPath;
static std::filesystem::path g_networkDataPath = g_exeDir / "data";

static void parseCommandline(int32_t argc, const char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];

        if (strncmp(arg, "-", 1) != 0)
            continue;

        if (strncmp(arg, "-train", 7) == 0) {
            if (i + 2 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_appMode = AppMode::Training;
            g_datasetConfigPath = argv[i + 1];
            g_datasetPath = argv[i + 2];
            if (!std::filesystem::exists(g_datasetConfigPath) ||
                std::filesystem::is_directory(g_datasetConfigPath)) {
                hpprintf("%s does not exist or is a directory.", g_datasetConfigPath.string().c_str());
                exit(EXIT_FAILURE);
            }
            if (!std::filesystem::exists(g_datasetPath) ||
                !std::filesystem::is_directory(g_datasetPath)) {
                hpprintf("%s does not exist or is not a directory.", g_datasetPath.string().c_str());
                exit(EXIT_FAILURE);
            }
            i += 2;
        }
        else if (strncmp(arg, "-data", 6) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_networkDataPath = argv[i + 1];
            if (!std::filesystem::exists(g_networkDataPath) ||
                std::filesystem::is_directory(g_networkDataPath)) {
                hpprintf("%s does not exist or is a directory.", g_networkDataPath.string().c_str());
                exit(EXIT_FAILURE);
            }
            i += 1;
        }
        else {
            printf("Unknown option.\n");
            exit(EXIT_FAILURE);
        }
    }
}

int32_t mainFunc(int32_t argc, const char* argv[]) {
	parseCommandline(argc, argv);

	CUcontext cuContext;
	CUDADRV_CHECK(cuInit(0));
	CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
	CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

	CUstream cuStream;
	CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    Denoiser denoiser;
    denoiser.initialize(512, 6, 1e-2f);

    if (g_appMode == AppMode::Training) {
        std::ifstream f(g_datasetConfigPath);
        json scenes = json::parse(f, nullptr, true, /*skip_comments=*/true);

        const uint32_t numScenes = scenes.size();

        //for (int sceneIdx = 0; sceneIdx < scenes.size(); ++sceneIdx) {
        //    printf("%s\n", scenes[sceneIdx].value("name", "").c_str());
        //}

        auto const mapPrimarySampleToDiscrete = [](
            float u01, uint32_t numValues, float* uRemapped = nullptr) {
            uint32_t idx = min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
            if (uRemapped)
                *uRemapped = u01 * numValues - idx;
            return idx;
        };

        constexpr uint32_t numEpochs = 100;
        constexpr uint32_t numItemsPerEpoch = 65536;
        constexpr float colorScale = 0.1f;

        cudau::TypedBuffer<shared::TrainingItem> trainingItemBuffer;
        cudau::TypedBuffer<RGBSpectrum> targetColorBuffer;
        trainingItemBuffer.initialize(cuContext, cudau::BufferType::Device, numItemsPerEpoch);
        targetColorBuffer.initialize(cuContext, cudau::BufferType::Device, numItemsPerEpoch);

        std::mt19937 rng(123124913);
        std::uniform_real_distribution<float> u01;
        const uint32_t numItemsPerScene = (numItemsPerEpoch + numScenes - 1) / numScenes;
        for (int epochIdx = 0; epochIdx < numEpochs; ++epochIdx) {
            uint32_t nextItemOffset = 0;
            uint32_t numRemainingItems = numItemsPerEpoch;
            std::vector<shared::TrainingItem> trainingItems(numItemsPerEpoch);
            std::vector<RGBSpectrum> targetColors(numItemsPerEpoch);
            for (int sceneIdx = 0; sceneIdx < numScenes; ++sceneIdx) {
                std::string sceneName = scenes[(sceneIdx + epochIdx) % numScenes].value("name", "");
                std::filesystem::path noisyImgPath = g_datasetPath / (sceneName + "_noisy.exr");
                std::filesystem::path refImgPath = g_datasetPath / (sceneName + "_ref.exr");

                int32_t width, height;
                float* noisyImgData;
                float* refImgData;
                int32_t exrRet;
                const char* exrErrorMessage;

                const char* noisy_chs[] = {
                    "R", "G", "B",
                    "Albedo.R", "Albedo.G", "Albedo.B",
                    "Nsx", "Nsy", "Nsz",
                };
                constexpr uint32_t numNoisyChs = lengthof(noisy_chs);

                const char* ref_chs[] = {
                    "R", "G", "B",
                };
                constexpr uint32_t numRefChs = lengthof(ref_chs);

                exrRet = LoadEXRWithChannels(
                    &noisyImgData, &width, &height,
                    noisyImgPath.string().c_str(),
                    noisy_chs, numNoisyChs, &exrErrorMessage);
                Assert(exrRet == TINYEXR_SUCCESS, "failed to read a layer: %s", exrErrorMessage);

                exrRet = LoadEXRWithChannels(
                    &refImgData, &width, &height,
                    refImgPath.string().c_str(),
                    ref_chs, numRefChs, &exrErrorMessage);
                Assert(exrRet == TINYEXR_SUCCESS, "failed to read a layer: %s", exrErrorMessage);

                bool debugOutput = false;
                if (debugOutput) {
                    float* debugImg = new float[width * height * 3];
                    SDRImageSaverConfig saveConfig = {};
                    saveConfig.alphaForOverride = -1;
                    saveConfig.applyToneMap = true;
                    saveConfig.apply_sRGB_gammaCorrection = true;
                    saveConfig.brightnessScale = 1.0f;
                    saveConfig.flipY = false;

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = y * width + x;
                            debugImg[3 * idx + 0] = noisyImgData[numNoisyChs * idx + 0];
                            debugImg[3 * idx + 1] = noisyImgData[numNoisyChs * idx + 1];
                            debugImg[3 * idx + 2] = noisyImgData[numNoisyChs * idx + 2];
                        }
                    }
                    saveImage("dbg_noisyBeauty.png", width, height, 3, debugImg, saveConfig);

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = y * width + x;
                            debugImg[3 * idx + 0] = refImgData[numRefChs * idx + 0];
                            debugImg[3 * idx + 1] = refImgData[numRefChs * idx + 1];
                            debugImg[3 * idx + 2] = refImgData[numRefChs * idx + 2];
                        }
                    }
                    saveImage("dbg_refBeauty.png", width, height, 3, debugImg, saveConfig);

                    saveConfig.applyToneMap = false;
                    saveConfig.apply_sRGB_gammaCorrection = false;

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = y * width + x;
                            debugImg[3 * idx + 0] = noisyImgData[numNoisyChs * idx + 3];
                            debugImg[3 * idx + 1] = noisyImgData[numNoisyChs * idx + 4];
                            debugImg[3 * idx + 2] = noisyImgData[numNoisyChs * idx + 5];
                        }
                    }
                    saveImage("dbg_albedo.png", width, height, 3, debugImg, saveConfig);

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = y * width + x;
                            debugImg[3 * idx + 0] = 0.5f * noisyImgData[numNoisyChs * idx + 6] + 0.5f;
                            debugImg[3 * idx + 1] = 0.5f * noisyImgData[numNoisyChs * idx + 7] + 0.5f;
                            debugImg[3 * idx + 2] = 0.5f * noisyImgData[numNoisyChs * idx + 8] + 0.5f;
                        }
                    }
                    saveImage("dbg_shading_normal.png", width, height, 3, debugImg, saveConfig);

                    delete[] debugImg;
                }

                if (epochIdx == 0) {
                    RGBSpectrum maxNoisyBeauty = -RGBSpectrum::Infinity();
                    RGBSpectrum maxRefBeauty = -RGBSpectrum::Infinity();
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = y * width + x;

                            bool allValid = true;
                            for (int ch = 0; ch < numNoisyChs; ++ch) {
                                if (!std::isfinite(noisyImgData[numNoisyChs * idx + ch]))
                                    allValid = false;
                            }
                            if (!allValid) {
                                hpprintf("Noisy %4u, %4u: ", x, y);
                                for (int ch = 0; ch < numNoisyChs; ++ch)
                                    hpprintf("%g, ", noisyImgData[numNoisyChs * idx + ch]);
                                hpprintf("\n");
                            }
                            maxNoisyBeauty = max(maxNoisyBeauty, RGBSpectrum(
                                noisyImgData[numNoisyChs * idx + 0],
                                noisyImgData[numNoisyChs * idx + 1],
                                noisyImgData[numNoisyChs * idx + 2]));

                            allValid = true;
                            for (int ch = 0; ch < numRefChs; ++ch) {
                                if (!std::isfinite(refImgData[numRefChs * idx + ch]))
                                    allValid = false;
                            }
                            if (!allValid) {
                                hpprintf("Ref %4u, %4u: ", x, y);
                                for (int ch = 0; ch < numRefChs; ++ch)
                                    hpprintf("%g, ", refImgData[numRefChs * idx + ch]);
                                hpprintf("\n");
                            }
                            maxRefBeauty = max(maxRefBeauty, RGBSpectrum(
                                refImgData[numRefChs * idx + 0],
                                refImgData[numRefChs * idx + 1],
                                refImgData[numRefChs * idx + 2]));
                        }
                    }

                    hpprintf("Scene %3u: max (noisy): %g, %g, %g, max (ref): %g, %g, %g\n",
                             sceneIdx,
                             rgbprint(maxNoisyBeauty),
                             rgbprint(maxRefBeauty));
                }

                uint32_t numItems = std::min(numItemsPerScene, numRemainingItems);
                for (int itemIdx = 0; itemIdx < numItems; ++itemIdx) {
                    constexpr int32_t halfSize = shared::TrainingItem::size / 2;
                    int cx =
                        halfSize +
                        mapPrimarySampleToDiscrete(u01(rng), width - 2 * halfSize);
                    int cy =
                        halfSize +
                        mapPrimarySampleToDiscrete(u01(rng), height - 2 * halfSize);

                    shared::TrainingItem item;
                    for (int offy = -halfSize; offy <= halfSize; ++offy) {
                        for (int offx = -halfSize; offx <= halfSize; ++offx) {
                            shared::PixelFeature feature;
                            uint32_t nbIdx = (cy + offy) * width + (cx + offx);
                            Assert(nbIdx < width * height, "Neighbor index is OOB.");
                            feature.noisyColor = colorScale * RGBSpectrum(
                                noisyImgData[numNoisyChs * nbIdx + 0],
                                noisyImgData[numNoisyChs * nbIdx + 1],
                                noisyImgData[numNoisyChs * nbIdx + 2]);
                            feature.albedo = RGBSpectrum(
                                noisyImgData[numNoisyChs * nbIdx + 3],
                                noisyImgData[numNoisyChs * nbIdx + 4],
                                noisyImgData[numNoisyChs * nbIdx + 5]);
                            feature.normal = Normal3D(
                                noisyImgData[numNoisyChs * nbIdx + 6],
                                noisyImgData[numNoisyChs * nbIdx + 7],
                                noisyImgData[numNoisyChs * nbIdx + 8]);

                            uint32_t idxInItem =
                                (offy + halfSize) * shared::TrainingItem::size +
                                (offx + halfSize);
                            item.neighbors[idxInItem] = feature;
                        }
                    }

                    uint32_t idx = cy * width + cx;
                    trainingItems[nextItemOffset + itemIdx] = item;
                    targetColors[nextItemOffset + itemIdx] = colorScale * RGBSpectrum(
                        refImgData[numRefChs * idx + 0],
                        refImgData[numRefChs * idx + 1],
                        refImgData[numRefChs * idx + 2]);
                }
                nextItemOffset += numItems;
                numRemainingItems -= numItems;

                free(refImgData);
                free(noisyImgData);
            } // loop for scenes

            std::mt19937 shuffleEngine;
            shuffleEngine.seed(151312311);
            std::shuffle(trainingItems.begin(), trainingItems.end(), shuffleEngine);
            shuffleEngine.seed(151312311);
            std::shuffle(targetColors.begin(), targetColors.end(), shuffleEngine);

            trainingItemBuffer.write(trainingItems);
            targetColorBuffer.write(targetColors);

            float loss;
            denoiser.train(
                cuStream,
                reinterpret_cast<float*>(trainingItemBuffer.getDevicePointer()),
                reinterpret_cast<float*>(targetColorBuffer.getDevicePointer()),
                numItemsPerEpoch,
                &loss);
            CUDADRV_CHECK(cuStreamSynchronize(cuStream));
            hpprintf("Epoch %03u: Loss: %.3f\n", epochIdx, loss);

            {
                std::string sceneName = scenes[0].value("name", "");
                std::filesystem::path noisyImgPath = g_datasetPath / (sceneName + "_noisy.exr");
                std::filesystem::path refImgPath = g_datasetPath / (sceneName + "_ref.exr");

                int32_t width, height;
                float* noisyImgData;
                float* refImgData;
                int32_t exrRet;
                const char* exrErrorMessage;

                const char* noisy_chs[] = {
                    "R", "G", "B",
                    "Albedo.R", "Albedo.G", "Albedo.B",
                    "Nsx", "Nsy", "Nsz",
                };
                constexpr uint32_t numNoisyChs = lengthof(noisy_chs);

                exrRet = LoadEXRWithChannels(
                    &noisyImgData, &width, &height,
                    noisyImgPath.string().c_str(),
                    noisy_chs, numNoisyChs, &exrErrorMessage);
                Assert(exrRet == TINYEXR_SUCCESS, "failed to read a layer: %s", exrErrorMessage);

                uint32_t numElementsPadded = (width * height + 127) / 128 * 128;
                cudau::TypedBuffer<shared::TrainingItem> inputFeatureBuffer;
                inputFeatureBuffer.initialize(
                    cuContext, cudau::BufferType::Device,
                    numElementsPadded);
                cudau::TypedBuffer<RGBSpectrum> outputBuffer;
                outputBuffer.initialize(
                    cuContext, cudau::BufferType::Device,
                    numElementsPadded);

                shared::TrainingItem* inputFeatures = inputFeatureBuffer.map();
                for (int cy = 0; cy < height; ++cy) {
                    for (int cx = 0; cx < width; ++cx) {
                        constexpr int32_t halfSize = shared::TrainingItem::size / 2;

                        shared::TrainingItem item;
                        for (int offy = -halfSize; offy <= halfSize; ++offy) {
                            int nbY = cy + offy;
                            if (nbY < 0)
                                nbY = -nbY;
                            else if (nbY >= height)
                                nbY = height - 1 - (nbY - height + 1);
                            for (int offx = -halfSize; offx <= halfSize; ++offx) {
                                int nbX = cx + offx;
                                if (nbX < 0)
                                    nbX = -nbX;
                                else if (nbX >= width)
                                    nbX = width - 1 - (nbX - width + 1);

                                shared::PixelFeature feature;
                                uint32_t nbIdx = nbY * width + nbX;
                                Assert(nbIdx < width * height, "Neighbor index is OOB.");
                                feature.noisyColor = colorScale * RGBSpectrum(
                                    noisyImgData[numNoisyChs * nbIdx + 0],
                                    noisyImgData[numNoisyChs * nbIdx + 1],
                                    noisyImgData[numNoisyChs * nbIdx + 2]);
                                feature.albedo = RGBSpectrum(
                                    noisyImgData[numNoisyChs * nbIdx + 3],
                                    noisyImgData[numNoisyChs * nbIdx + 4],
                                    noisyImgData[numNoisyChs * nbIdx + 5]);
                                feature.normal = Normal3D(
                                    noisyImgData[numNoisyChs * nbIdx + 6],
                                    noisyImgData[numNoisyChs * nbIdx + 7],
                                    noisyImgData[numNoisyChs * nbIdx + 8]);

                                uint32_t idxInItem =
                                    (offy + halfSize) * shared::TrainingItem::size +
                                    (offx + halfSize);
                                item.neighbors[idxInItem] = feature;
                            }
                        }

                        inputFeatureBuffer[cy * width + cx] = item;
                    }
                }
                inputFeatureBuffer.unmap();

                denoiser.infer(
                    cuStream,
                    reinterpret_cast<float*>(inputFeatureBuffer.getDevicePointer()),
                    numElementsPadded,
                    reinterpret_cast<float*>(outputBuffer.getDevicePointer()));

                char filename[256];
                sprintf_s(filename, "test_output_%3u.exr", epochIdx);
                float* outputs = reinterpret_cast<float*>(outputBuffer.map());
                saveImageHDR(
                    filename, width, height, 3, 1.0f,
                    outputs, false);
                outputBuffer.unmap();

                outputBuffer.finalize();
                inputFeatureBuffer.finalize();

                free(noisyImgData);
            }
        } // loop for epochs

        targetColorBuffer.finalize();
        trainingItemBuffer.finalize();
    }
    else {

    }

    denoiser.finalize();

    return 0;
}

}



int32_t main(int32_t argc, const char* argv[]) {
	try {
		return rtc8::mainFunc(argc, argv);
	}
	catch (const std::exception &ex) {
		hpprintf("Error: %s\n", ex.what());
		return -1;
	}
}