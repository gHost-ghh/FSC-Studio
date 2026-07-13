#pragma once

#include "fsc/vision/ModelPaths.hpp"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace fsc::vision::detail {

[[nodiscard]] Ort::SessionOptions sessionOptionsFor(RuntimeMode mode);
[[nodiscard]] std::vector<RuntimeMode> automaticRuntimeCandidates();
[[nodiscard]] std::string executionProviderName(RuntimeMode mode);
[[nodiscard]] bool runtimeModeCompiled(RuntimeMode mode) noexcept;
[[nodiscard]] std::vector<std::string> availableExecutionProviders();

} // namespace fsc::vision::detail
