#pragma once

#include "fsc/vision/Image.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace fsc::mesh {

inline constexpr std::size_t kMediaPipeFaceMeshPointCount = 478;

struct MediaPipeFaceLandmarkerOptions {
    std::filesystem::path modelAssetPath;
    fsc::vision::RuntimeMode runtimeMode = fsc::vision::RuntimeMode::Auto;
    float minFacePresenceConfidence = 0.5f;
    float cropScale = 1.5f;
};

// Runs the MediaPipe 478-point landmark network through ONNX Runtime. Supplying
// the database face box and five detection keypoints avoids a second full-image
// detector pass and keeps multi-face records associated with the correct face.
class MediaPipeFaceLandmarker {
public:
    explicit MediaPipeFaceLandmarker(MediaPipeFaceLandmarkerOptions options = {});
    ~MediaPipeFaceLandmarker();

    MediaPipeFaceLandmarker(const MediaPipeFaceLandmarker&) = delete;
    MediaPipeFaceLandmarker& operator=(const MediaPipeFaceLandmarker&) = delete;
    MediaPipeFaceLandmarker(MediaPipeFaceLandmarker&&) noexcept;
    MediaPipeFaceLandmarker& operator=(MediaPipeFaceLandmarker&&) noexcept;

    [[nodiscard]] std::vector<std::vector<double>> detect(
        const fsc::vision::RgbImage& image,
        const std::vector<double>& faceBox,
        const std::vector<std::vector<double>>& keypoints = {}) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::filesystem::path defaultMediaPipeFaceLandmarkerModelPath();

[[nodiscard]] bool isMediaPipeFaceMesh(const std::vector<std::vector<double>>& points) noexcept;

// Mirrors fsc_studio_services._best_matching_mesh_for_record: use bounding-box
// IoU when possible, otherwise select the largest detected face.
[[nodiscard]] std::vector<std::vector<double>> selectBestMediaPipeFaceMesh(
    const std::vector<std::vector<std::vector<double>>>& meshes,
    const std::vector<double>& recordBbox);

} // namespace fsc::mesh
