#pragma once

#include "fsc/vision/Image.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fsc::legacy {

struct LegacyDtbImage {
    std::string fileName;
    fsc::vision::RgbImage image;
};

struct LegacyConversionOptions {
    fsc::vision::InsightFaceModelPaths models;
    fsc::vision::RuntimeMode runtimeMode = fsc::vision::RuntimeMode::Auto;
    float detectionThreshold = 0.50f;
    int limit = 0;
    std::function<void(const std::string& message, int current, int total)> progress;
};

struct LegacyConversionSummary {
    std::filesystem::path sourcePath;
    std::filesystem::path outputPath;
    std::filesystem::path imageDirectory;
    int rowsTotal = 0;
    int facesSaved = 0;
    int skippedRows = 0;
    std::vector<std::string> messages;
};

// Reads only the trusted legacy FSC tuple layout. The parser never invokes
// Python globals or executes pickle reducers; unsupported payloads are rejected.
std::vector<LegacyDtbImage> loadLegacyDtbImages(const std::filesystem::path& sourcePath);

LegacyConversionSummary convertLegacyDtb(
    const std::filesystem::path& sourcePath,
    std::filesystem::path outputPath,
    const LegacyConversionOptions& options = {});

} // namespace fsc::legacy
