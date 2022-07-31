#include "renderer_scene.h"
#include <regex>

namespace rtc8 {

GPUEnvironment g_gpuEnv;
SceneMemory g_sceneMem;

SurfaceMaterial::SurfaceMaterial() {
    m_slot = g_sceneMem.allocateSurfaceMaterial();
}



uint32_t SimplePBRSurfaceMaterial::s_procSetSlot;



Geometry::Geometry() {
    m_slot = g_sceneMem.allocateGeometryInstance();
}



Instance::Instance() {
    m_slot = g_sceneMem.allocateInstance();
}



template <typename... Types>
void throwRuntimeErrorAtLine(bool expr, uint32_t line, const char* fmt, const Types &... args) {
    if (!expr) {
        std::string mFmt = std::string("l.%u: ") + fmt;
        _throwRuntimeError(mFmt.c_str(), line, args...);
    }
}

void loadScene(const std::filesystem::path &sceneFilePath) {
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
    uint32_t imageWidth;
    uint32_t imageHeight;
    float timeBegin;
    float timeEnd;
    uint32_t fps;
    std::map<std::string, MeshInfo> meshInfos;
    std::map<std::string, InstanceInfo> instInfos;
    std::map<std::string, CameraInfo> camInfos;
    
    uint32_t lineIndex = -1;

    const char* reInteger = R"(([+-]?\d+))";
    const char* reReal = R"(([+-]?(?:\d+\.?\d*|\.\d+)))";
    const char* reString = R"((\S+?))";
    const char* reQuotedPath = R"*("(.+?)")*";

    const auto makeRegex = [](const std::vector<std::string> &tokens) {
        const char* reSpace = R"([ \t]+?)";
        std::string pattern = tokens[0];
        for (uint32_t i = 1; i < tokens.size(); ++i)
            pattern += reSpace + tokens[i];
        pattern += "$";
        return std::move(std::regex(pattern));
    };
    const auto testRegex = [&](const std::regex &re, const char* cmd, const std::string &line) {
        std::smatch m;
        throwRuntimeErrorAtLine(
            std::regex_search(line, m, re), lineIndex + 1,
            "failed to parse \"%s\" command: %s",
            cmd, line.c_str());
        return std::move(m);
    };

    using lineFunc = std::function<void(const std::string &line)>;
    std::map<std::string, lineFunc> processors = {
        {
            "image",
            [&](const std::string &line) {
                static const char* cmd = "image";
                static const std::regex re = makeRegex({cmd, reInteger, reInteger});
                std::smatch m = testRegex(re, cmd, line);

                imageWidth = static_cast<uint32_t>(std::stoi(m[1].str().c_str()));
                imageHeight = static_cast<uint32_t>(std::stoi(m[2].str().c_str()));
            }
        },
        {
            "time",
            [&](const std::string &line) {
                static const char* cmd = "time";
                static const std::regex re = makeRegex({cmd, reReal, reReal, reInteger});
                std::smatch m = testRegex(re, cmd, line);

                timeBegin = std::stof(m[1].str().c_str());
                timeEnd = std::stof(m[2].str().c_str());
                fps = static_cast<uint32_t>(std::stoi(m[3].str().c_str()));
            }
        },
        {
            "mesh",
            [&](const std::string &line) {
                static const char* cmd = "mesh";
                static const std::regex re = makeRegex({cmd, reQuotedPath, reReal, reString});
                std::smatch m = testRegex(re, cmd, line);

                std::string meshName = m[3].str();
                throwRuntimeErrorAtLine(
                    !meshInfos.contains(meshName), lineIndex + 1,
                    "Mesh %s has been already created.",
                    meshName.c_str());

                MeshInfo::File meshInfo;
                meshInfo.path = m[1].str();
                meshInfo.scale = std::stof(m[2].str().c_str());
                meshInfos[meshName].body = meshInfo;
            }
        },
        {
            "rect",
            [&](const std::string &line) {
                static const char* cmd = "rect";
                static const std::regex re = makeRegex({cmd, reReal, reReal, reString});
                std::smatch m = testRegex(re, cmd, line);

                std::string meshName = m[3].str();
                throwRuntimeErrorAtLine(
                    !meshInfos.contains(meshName), lineIndex + 1,
                    "Mesh %s has been already created.",
                    meshName.c_str());

                MeshInfo::Rectangle rectInfo;
                rectInfo.dimX = std::stof(m[1].str().c_str());
                rectInfo.dimZ = std::stof(m[2].str().c_str());
                rectInfo.emittance = RGBSpectrum::Zero();
                meshInfos[meshName].body = rectInfo;
            }
        },
        {
            "emittance",
            [&](const std::string &line) {
                static const char* cmd = "emittance";
                static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string meshName = m[1].str();
                throwRuntimeErrorAtLine(
                    meshInfos.contains(meshName), lineIndex + 1,
                    "Mesh %s does not exist.",
                    meshName.c_str());
                MeshInfo &meshInfo = meshInfos.at(meshName);
                throwRuntimeErrorAtLine(
                    std::holds_alternative<MeshInfo::Rectangle>(meshInfo.body), lineIndex + 1,
                    "Emittance cannot be assigned to this mesh.",
                    meshName.c_str());

                auto &rectInfo = std::get<MeshInfo::Rectangle>(meshInfo.body);
                rectInfo.emittance = RGBSpectrum(
                    std::stof(m[2].str().c_str()),
                    std::stof(m[3].str().c_str()),
                    std::stof(m[4].str().c_str()));
            }
        },
        {
            "inst",
            [&](const std::string &line) {
                static const char* cmd = "inst";
                static const std::regex re = makeRegex({cmd, reString, reString});
                std::smatch m = testRegex(re, cmd, line);

                std::string meshName = m[1].str();
                throwRuntimeErrorAtLine(
                    meshInfos.contains(meshName), lineIndex + 1,
                    "Mesh %s does not exist.",
                    meshName.c_str());
                std::string instName = m[2].str();
                throwRuntimeErrorAtLine(
                    !instInfos.contains(instName), lineIndex + 1,
                    "Instance %s has been already created.",
                    instName.c_str());
                throwRuntimeErrorAtLine(
                    !camInfos.contains(instName), lineIndex + 1,
                    "Camera with the same name %s exists.",
                    instName.c_str());

                InstanceInfo instInfo;
                instInfo.meshName = meshName;
                instInfo.nextKeyScale = 1.0f;
                instInfo.nextKeyOrientation = Quaternion::Identity();
                instInfo.nextKeyPosition = Point3D::Zero();
                instInfos[instName] = instInfo;
            }
        },
        {
            "camera",
            [&](const std::string &line) {
                static const char* cmd = "camera";
                static const std::regex re = makeRegex({cmd, reString});
                std::smatch m = testRegex(re, cmd, line);

                std::string camName = m[1].str();
                throwRuntimeErrorAtLine(
                    !camInfos.contains(camName), lineIndex + 1,
                    "Camera %s has been already created.",
                    camName.c_str());
                throwRuntimeErrorAtLine(
                    !instInfos.contains(camName), lineIndex + 1,
                    "Instance with the same name %s exists.",
                    camName.c_str());

                CameraInfo camInfo;
                camInfo.nextKeyPosition = Point3D(0, 1, 5);
                camInfo.nextKeyLookAt = Point3D(0, 0, 0);
                camInfo.nextKeyUp = Vector3D(0, 1, 0);
                camInfo.nextKeyFovY = 60 * pi_v<float> / 180;
                camInfos[camName] = camInfo;
            }
        },
        {
            "scale",
            [&](const std::string &line) {
                static const char* cmd = "scale";
                static const std::regex re = makeRegex({cmd, reString, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string instName = m[1].str();
                throwRuntimeErrorAtLine(
                    instInfos.contains(instName), lineIndex + 1,
                    "Instance %s does not exist.",
                    instName.c_str());

                instInfos.at(instName).nextKeyScale = std::stof(m[2].str().c_str());
            }
        },
        {
            "rotate",
            [&](const std::string &line) {
                static const char* cmd = "rotate";
                static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string instName = m[1].str();
                throwRuntimeErrorAtLine(
                    instInfos.contains(instName), lineIndex + 1,
                    "Instance %s does not exist.",
                    instName.c_str());

                instInfos.at(instName).nextKeyOrientation = qRotate(
                    std::stof(m[5].str().c_str()),
                    std::stof(m[2].str().c_str()),
                    std::stof(m[3].str().c_str()),
                    std::stof(m[4].str().c_str()));
            }
        },
        {
            "trans",
            [&](const std::string &line) {
                static const char* cmd = "trans";
                static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string tgtName = m[1].str();
                bool isInst = instInfos.contains(tgtName);
                bool isCam = camInfos.contains(tgtName);
                throwRuntimeErrorAtLine(
                    isInst || isCam, lineIndex + 1,
                    "Instance/Camera %s does not exist.",
                    tgtName.c_str());

                if (isInst) {
                    instInfos.at(tgtName).nextKeyPosition = Point3D(
                        std::stof(m[2].str().c_str()),
                        std::stof(m[3].str().c_str()),
                        std::stof(m[4].str().c_str()));
                }
                else {
                    camInfos.at(tgtName).nextKeyPosition = Point3D(
                        std::stof(m[2].str().c_str()),
                        std::stof(m[3].str().c_str()),
                        std::stof(m[4].str().c_str()));
                }
            }
        },
        {
            "fovy",
            [&](const std::string &line) {
                static const char* cmd = "fovy";
                static const std::regex re = makeRegex({cmd, reString, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string camName = m[1].str();
                throwRuntimeErrorAtLine(
                    camInfos.contains(camName), lineIndex + 1,
                    "Camera %s does not exist.",
                    camName.c_str());

                camInfos.at(camName).nextKeyFovY = std::stof(m[2].str().c_str());
            }
        },
        {
            "lookat",
            [&](const std::string &line) {
                static const char* cmd = "lookat";
                static const std::regex re = makeRegex({cmd, reString, reReal, reReal, reReal});
                std::smatch m = testRegex(re, cmd, line);

                std::string camName = m[1].str();
                throwRuntimeErrorAtLine(
                    camInfos.contains(camName), lineIndex + 1,
                    "Camera %s does not exist.",
                    camName.c_str());

                camInfos.at(camName).nextKeyLookAt = Point3D(
                    std::stof(m[2].str().c_str()),
                    std::stof(m[3].str().c_str()),
                    std::stof(m[4].str().c_str()));
            }
        },
        {
            "addkey",
            [&](const std::string &line) {
                static const char* cmd = "addkey";
                static const std::regex re = makeRegex({cmd, reString, reReal, reString, reString});
                std::smatch m = testRegex(re, cmd, line);

                std::string tgtName = m[1].str();
                bool isInst = instInfos.contains(tgtName);
                bool isCam = camInfos.contains(tgtName);
                throwRuntimeErrorAtLine(
                    isInst || isCam, lineIndex + 1,
                    "Instance/Camera %s does not exist.",
                    tgtName.c_str());

                if (isInst) {
                    InstanceInfo &instInfo = instInfos.at(tgtName);
                    KeyInstanceState state;
                    state.timePoint = std::stof(m[2].str().c_str());
                    state.scale = instInfo.nextKeyScale;
                    state.orientation = instInfo.nextKeyOrientation;
                    state.position = instInfo.nextKeyPosition;
                    instInfo.keyStates.push_back(state);
                }
                else {
                    CameraInfo &camInfo = camInfos.at(tgtName);
                    KeyCameraState state;
                    state.timePoint = std::stof(m[2].str().c_str());
                    state.position = camInfo.nextKeyPosition;
                    state.lookAt = camInfo.nextKeyLookAt;
                    state.up = camInfo.nextKeyUp;
                    state.fovY = camInfo.nextKeyFovY;
                    camInfo.keyStates.push_back(state);
                }
            }
        },
    };

    for (const std::string &line : lines) {
        ++lineIndex;
        if (line.empty())
            continue;
        size_t nextSpacePos = line.find_first_of(" \t");
        std::string firstToken = nextSpacePos != std::string::npos ?
            line.substr(0, nextSpacePos) : line;
        throwRuntimeErrorAtLine(
            processors.contains(firstToken), lineIndex + 1,
            "Unknown token found: \"%s\"",
            firstToken.c_str());
        processors.at(firstToken)(line);
    }

    printf("");
}

}
