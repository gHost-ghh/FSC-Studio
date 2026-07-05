#include "fsc/vision/OnnxRuntimeSession.hpp"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>

namespace fsc::vision {

namespace {

std::wstring widePath(const std::filesystem::path& path) {
    return path.wstring();
}

std::string providerName(RuntimeMode mode) {
    switch (mode) {
    case RuntimeMode::DirectMl:
        return "DmlExecutionProvider";
    case RuntimeMode::Cpu:
        return "CPUExecutionProvider";
    case RuntimeMode::Auto:
    default:
        return "Auto";
    }
}

std::string elementTypeName(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "float64";
    default:
        return "other";
    }
}

OnnxSessionInfo::ValueInfo valueInfo(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    size_t index,
    bool input) {
    auto name = input
        ? session.GetInputNameAllocated(index, allocator)
        : session.GetOutputNameAllocated(index, allocator);
    auto typeInfo = input
        ? session.GetInputTypeInfo(index)
        : session.GetOutputTypeInfo(index);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    return {
        name.get(),
        tensorInfo.GetShape(),
        elementTypeName(tensorInfo.GetElementType()),
    };
}

} // namespace

OnnxSessionInfo inspectOnnxModel(const std::filesystem::path& modelPath, RuntimeMode mode) {
    if (!std::filesystem::exists(modelPath)) {
        throw std::runtime_error("ONNX model file does not exist: " + modelPath.string());
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FSC Studio Native");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // DirectML provider wiring is added in the next checkpoint after the exact
    // ONNX Runtime distribution is installed and verified. CPU load parity comes first.
    Ort::Session session(env, widePath(modelPath).c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxSessionInfo info;
    info.modelPath = modelPath;
    info.requestedMode = mode;
    info.provider = providerName(mode);

    const size_t inputCount = session.GetInputCount();
    for (size_t index = 0; index < inputCount; ++index) {
        info.inputs.push_back(valueInfo(session, allocator, index, true));
    }
    const size_t outputCount = session.GetOutputCount();
    for (size_t index = 0; index < outputCount; ++index) {
        info.outputs.push_back(valueInfo(session, allocator, index, false));
    }
    return info;
}

} // namespace fsc::vision
