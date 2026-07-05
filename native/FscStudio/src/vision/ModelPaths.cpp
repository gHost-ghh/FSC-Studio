#include "fsc/vision/ModelPaths.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace fsc::vision {
namespace {

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

InsightFaceModelPaths InsightFaceModelPaths::fromBuffaloL(std::filesystem::path rootDirectory) {
    if (rootDirectory.filename() != "buffalo_l") {
        rootDirectory /= "buffalo_l";
    }
    return {
        rootDirectory,
        rootDirectory / "det_10g.onnx",
        rootDirectory / "w600k_r50.onnx",
        rootDirectory / "2d106det.onnx",
        rootDirectory / "1k3d68.onnx",
        rootDirectory / "genderage.onnx",
    };
}

std::vector<std::filesystem::path> InsightFaceModelPaths::missingFiles() const {
    const std::vector<std::filesystem::path> all{
        detectionModelPath,
        recognitionModelPath,
        landmark2dModelPath,
        landmark3dModelPath,
        genderAgeModelPath,
    };
    std::vector<std::filesystem::path> missing;
    for (const auto& path : all) {
        if (!std::filesystem::exists(path)) {
            missing.push_back(path);
        }
    }
    return missing;
}

std::string toString(RuntimeMode mode) {
    switch (mode) {
    case RuntimeMode::Cpu:
        return "cpu";
    case RuntimeMode::DirectMl:
        return "directml";
    case RuntimeMode::Auto:
    default:
        return "auto";
    }
}

RuntimeMode parseRuntimeMode(std::string value) {
    value = lowerAscii(std::move(value));
    if (value == "cpu") {
        return RuntimeMode::Cpu;
    }
    if (value == "directml" || value == "dml" || value == "gpu") {
        return RuntimeMode::DirectMl;
    }
    return RuntimeMode::Auto;
}

} // namespace fsc::vision
