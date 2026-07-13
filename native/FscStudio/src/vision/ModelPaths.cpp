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

InsightFaceModelPaths InsightFaceModelPaths::optimizedFor(RuntimeMode mode) const {
    if (mode != RuntimeMode::QnnNpu) {
        return *this;
    }

    const auto qnnRoot = rootDirectory / "qnn_htp";
    return {
        qnnRoot,
        qnnRoot / detectionModelPath.filename(),
        qnnRoot / recognitionModelPath.filename(),
        qnnRoot / landmark2dModelPath.filename(),
        qnnRoot / landmark3dModelPath.filename(),
        qnnRoot / genderAgeModelPath.filename(),
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
    case RuntimeMode::Cuda:
        return "cuda";
    case RuntimeMode::QnnNpu:
        return "qnn-npu";
    case RuntimeMode::QnnGpu:
        return "qnn-gpu";
    case RuntimeMode::Auto:
    default:
        return "auto";
    }
}

std::string executionProviderName(RuntimeMode mode) {
    switch (mode) {
    case RuntimeMode::Cpu:
        return "CPUExecutionProvider";
    case RuntimeMode::DirectMl:
        return "DmlExecutionProvider";
    case RuntimeMode::Cuda:
        return "CUDAExecutionProvider";
    case RuntimeMode::QnnNpu:
        return "QNNExecutionProvider (HTP/NPU)";
    case RuntimeMode::QnnGpu:
        return "QNNExecutionProvider (Adreno GPU)";
    case RuntimeMode::Auto:
    default:
        return "Auto";
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
    if (value == "cuda" || value == "nvidia") {
        return RuntimeMode::Cuda;
    }
    if (value == "qnn-npu" || value == "qnn_htp" || value == "npu" || value == "htp") {
        return RuntimeMode::QnnNpu;
    }
    if (value == "qnn-gpu" || value == "qnn_gpu" || value == "adreno") {
        return RuntimeMode::QnnGpu;
    }
    return RuntimeMode::Auto;
}

} // namespace fsc::vision
