#include "fsc/vision/RuntimeProvider.hpp"

#ifdef FSC_ONNXRUNTIME_HAS_DML
#include <dml_provider_factory.h>
#endif

#include <stdexcept>
#include <unordered_map>

namespace fsc::vision::detail {
namespace {

Ort::SessionOptions cpuSessionOptions() {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

Ort::SessionOptions directMlSessionOptions() {
#ifdef FSC_ONNXRUNTIME_HAS_DML
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.DisableMemPattern();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    const OrtDmlApi* dmlApi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
        "DML",
        ORT_API_VERSION,
        reinterpret_cast<const void**>(&dmlApi)));
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(options, 0));
    return options;
#else
    throw std::runtime_error("This ONNX Runtime build does not include DirectML.");
#endif
}

Ort::SessionOptions cudaSessionOptions() {
#ifdef FSC_ONNXRUNTIME_HAS_CUDA
    auto options = cpuSessionOptions();
    Ort::CUDAProviderOptions cudaOptions;
    cudaOptions.Update({
        {"device_id", "0"},
        {"cudnn_conv_algo_search", "HEURISTIC"},
        {"do_copy_in_default_stream", "1"},
        {"use_tf32", "1"},
    });
    options.AppendExecutionProvider_CUDA_V2(*cudaOptions);
    return options;
#else
    throw std::runtime_error("This ONNX Runtime build does not include CUDA.");
#endif
}

Ort::SessionOptions qnnSessionOptions(bool npu) {
#ifdef FSC_ONNXRUNTIME_HAS_QNN
    auto options = cpuSessionOptions();
    if (npu) {
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
    std::unordered_map<std::string, std::string> providerOptions{
        {"backend_path", npu ? "QnnHtp.dll" : "QnnGpu.dll"},
    };
    if (npu) {
        providerOptions.emplace("htp_performance_mode", "burst");
    }
    options.AppendExecutionProvider("QNN", providerOptions);
    return options;
#else
    (void)npu;
    throw std::runtime_error("This ONNX Runtime build does not include QNN.");
#endif
}

} // namespace

Ort::SessionOptions sessionOptionsFor(RuntimeMode mode) {
    switch (mode) {
    case RuntimeMode::Cpu:
        return cpuSessionOptions();
    case RuntimeMode::DirectMl:
        return directMlSessionOptions();
    case RuntimeMode::Cuda:
        return cudaSessionOptions();
    case RuntimeMode::QnnNpu:
        return qnnSessionOptions(true);
    case RuntimeMode::QnnGpu:
        return qnnSessionOptions(false);
    case RuntimeMode::Auto:
    default:
        throw std::invalid_argument("Auto runtime mode must be resolved before creating session options.");
    }
}

std::vector<RuntimeMode> automaticRuntimeCandidates() {
    std::vector<RuntimeMode> modes;
#if defined(_M_ARM64) || defined(__aarch64__)
#ifdef FSC_ONNXRUNTIME_HAS_QNN
    modes.push_back(RuntimeMode::QnnNpu);
    modes.push_back(RuntimeMode::QnnGpu);
#endif
#ifdef FSC_ONNXRUNTIME_HAS_DML
    modes.push_back(RuntimeMode::DirectMl);
#endif
#else
#ifdef FSC_ONNXRUNTIME_HAS_CUDA
    modes.push_back(RuntimeMode::Cuda);
#endif
#ifdef FSC_ONNXRUNTIME_HAS_DML
    modes.push_back(RuntimeMode::DirectMl);
#endif
#endif
    modes.push_back(RuntimeMode::Cpu);
    return modes;
}

std::string executionProviderName(RuntimeMode mode) {
    return fsc::vision::executionProviderName(mode);
}

bool runtimeModeCompiled(RuntimeMode mode) noexcept {
    switch (mode) {
    case RuntimeMode::Auto:
    case RuntimeMode::Cpu:
        return true;
    case RuntimeMode::DirectMl:
#ifdef FSC_ONNXRUNTIME_HAS_DML
        return true;
#else
        return false;
#endif
    case RuntimeMode::Cuda:
#ifdef FSC_ONNXRUNTIME_HAS_CUDA
        return true;
#else
        return false;
#endif
    case RuntimeMode::QnnNpu:
    case RuntimeMode::QnnGpu:
#ifdef FSC_ONNXRUNTIME_HAS_QNN
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

std::vector<std::string> availableExecutionProviders() {
    return Ort::GetAvailableProviders();
}

} // namespace fsc::vision::detail
