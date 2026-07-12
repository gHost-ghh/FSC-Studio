#pragma once

#include "fsc/vision/ModelPaths.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fsc::vision {

struct OnnxSessionInfo {
    std::filesystem::path modelPath;
    RuntimeMode requestedMode = RuntimeMode::Auto;
    std::string provider;
    struct ValueInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string elementType;
    };
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
};

#ifdef FSC_ENABLE_ONNX
OnnxSessionInfo inspectOnnxModel(const std::filesystem::path& modelPath, RuntimeMode mode);
#else
inline OnnxSessionInfo inspectOnnxModel(const std::filesystem::path& modelPath, RuntimeMode mode) {
    return {modelPath, mode, "not built with ONNX Runtime", {}, {}};
}
#endif

} // namespace fsc::vision
