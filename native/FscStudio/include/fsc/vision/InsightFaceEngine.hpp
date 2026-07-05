#pragma once

#include "fsc/vision/FaceGeometry.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <memory>
#include <vector>

namespace fsc::vision {

struct AnalyzedFace {
    Detection detection;
    std::vector<float> embedding;
};

class InsightFaceEngine {
public:
    InsightFaceEngine(InsightFaceModelPaths models, RuntimeMode mode = RuntimeMode::Cpu);
    ~InsightFaceEngine();

    InsightFaceEngine(const InsightFaceEngine&) = delete;
    InsightFaceEngine& operator=(const InsightFaceEngine&) = delete;
    InsightFaceEngine(InsightFaceEngine&&) noexcept;
    InsightFaceEngine& operator=(InsightFaceEngine&&) noexcept;

    [[nodiscard]] std::vector<Detection> detect(
        const RgbImage& image,
        float detectionThreshold = 0.55f,
        int maxFaces = 10) const;

    [[nodiscard]] std::vector<float> extractEmbedding(
        const RgbImage& image,
        const std::array<Point2f, 5>& keypoints) const;

    [[nodiscard]] std::vector<AnalyzedFace> analyze(
        const RgbImage& image,
        float detectionThreshold = 0.55f,
        int maxFaces = 10) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fsc::vision
