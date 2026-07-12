#include "fsc/vision/OnnxRuntimeSession.hpp"

#include "fsc/core/PathEncoding.hpp"

#include <onnxruntime_cxx_api.h>
#ifdef FSC_ONNXRUNTIME_HAS_DML
#include <dml_provider_factory.h>
#endif

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

Ort::SessionOptions cpuSessionOptions() {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

Ort::SessionOptions dmlSessionOptions() {
#ifdef FSC_ONNXRUNTIME_HAS_DML
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.DisableMemPattern();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    const OrtDmlApi* dmlApi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi)));
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(options, 0));
    return options;
#else
    throw std::runtime_error("This ONNX Runtime build does not include DirectML provider headers.");
#endif
}

Ort::SessionOptions sessionOptionsFor(RuntimeMode mode, std::string& actualProvider) {
    if (mode == RuntimeMode::DirectMl) {
        actualProvider = "DmlExecutionProvider";
        return dmlSessionOptions();
    }
    if (mode == RuntimeMode::Auto) {
        try {
            actualProvider = "DmlExecutionProvider";
            return dmlSessionOptions();
        } catch (const std::exception& ex) {
            actualProvider = "CPUExecutionProvider (Auto fallback: " + std::string(ex.what()) + ")";
            return cpuSessionOptions();
        }
    }
    actualProvider = "CPUExecutionProvider";
    return cpuSessionOptions();
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
        throw std::runtime_error("ONNX model file does not exist: " + fsc::core::pathToUtf8(modelPath));
    }

    if (mode == RuntimeMode::Auto) {
        try {
            auto info = inspectOnnxModel(modelPath, RuntimeMode::DirectMl);
            info.requestedMode = RuntimeMode::Auto;
            return info;
        } catch (const std::exception& exception) {
            auto info = inspectOnnxModel(modelPath, RuntimeMode::Cpu);
            info.requestedMode = RuntimeMode::Auto;
            info.provider += " (Auto fallback: " + std::string(exception.what()) + ')';
            return info;
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FSC Studio Native");
    std::string actualProvider;
    auto options = sessionOptionsFor(mode, actualProvider);
    Ort::Session session(env, widePath(modelPath).c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxSessionInfo info;
    info.modelPath = modelPath;
    info.requestedMode = mode;
    info.provider = actualProvider.empty() ? providerName(mode) : actualProvider;

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
