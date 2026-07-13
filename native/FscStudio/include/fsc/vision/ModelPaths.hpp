#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fsc::vision {

enum class RuntimeMode {
    Auto,
    Cpu,
    DirectMl,
    Cuda,
    QnnNpu,
    QnnGpu
};

struct InsightFaceModelPaths {
    std::filesystem::path rootDirectory;
    std::filesystem::path detectionModelPath;
    std::filesystem::path recognitionModelPath;
    std::filesystem::path landmark2dModelPath;
    std::filesystem::path landmark3dModelPath;
    std::filesystem::path genderAgeModelPath;

    static InsightFaceModelPaths fromBuffaloL(std::filesystem::path rootDirectory);
    [[nodiscard]] InsightFaceModelPaths optimizedFor(RuntimeMode mode) const;
    [[nodiscard]] std::vector<std::filesystem::path> missingFiles() const;
};

std::string toString(RuntimeMode mode);
std::string executionProviderName(RuntimeMode mode);
RuntimeMode parseRuntimeMode(std::string value);

} // namespace fsc::vision
