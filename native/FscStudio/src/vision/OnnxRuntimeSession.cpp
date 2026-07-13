#include "fsc/vision/OnnxRuntimeSession.hpp"

#include "fsc/core/PathEncoding.hpp"
#include "fsc/vision/RuntimeProvider.hpp"

#include <onnxruntime_cxx_api.h>

#include <sstream>
#include <stdexcept>

namespace fsc::vision {

namespace {

std::wstring widePath(const std::filesystem::path& path) {
    return path.wstring();
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
        std::ostringstream failures;
        for (const auto candidate : detail::automaticRuntimeCandidates()) {
            try {
                auto info = inspectOnnxModel(modelPath, candidate);
                info.requestedMode = RuntimeMode::Auto;
                if (!failures.str().empty()) {
                    info.provider += " (Auto fallback after " + failures.str() + ')';
                }
                return info;
            } catch (const std::exception& exception) {
                if (failures.tellp() > 0) {
                    failures << "; ";
                }
                failures << toString(candidate) << ": " << exception.what();
            }
        }
        throw std::runtime_error("No ONNX Runtime execution provider could create the model session: " + failures.str());
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FSC Studio Native");
    auto options = detail::sessionOptionsFor(mode);
    Ort::Session session(env, widePath(modelPath).c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxSessionInfo info;
    info.modelPath = modelPath;
    info.requestedMode = mode;
    info.provider = detail::executionProviderName(mode);

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
