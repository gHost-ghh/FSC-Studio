#pragma once

#include "fsc/vision/Image.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace fsc::mesh {

inline constexpr std::size_t kMediaPipeFaceMeshPointCount = 478;

struct MediaPipeFaceLandmarkerOptions {
    std::filesystem::path modelAssetPath;
    std::filesystem::path runtimeLibraryPath;
    int maxFaces = 10;
    float minFaceDetectionConfidence = 0.5f;
    float minFacePresenceConfidence = 0.5f;
};

// Uses MediaPipe's native C Task API. It is the same face-landmarker task and
// coordinate convention used by the Python application, without a Python runtime.
class MediaPipeFaceLandmarker {
public:
    explicit MediaPipeFaceLandmarker(MediaPipeFaceLandmarkerOptions options = {});
    ~MediaPipeFaceLandmarker();

    MediaPipeFaceLandmarker(const MediaPipeFaceLandmarker&) = delete;
    MediaPipeFaceLandmarker& operator=(const MediaPipeFaceLandmarker&) = delete;
    MediaPipeFaceLandmarker(MediaPipeFaceLandmarker&&) noexcept;
    MediaPipeFaceLandmarker& operator=(MediaPipeFaceLandmarker&&) noexcept;

    [[nodiscard]] std::vector<std::vector<std::vector<double>>> detect(
        const fsc::vision::RgbImage& image) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::filesystem::path defaultMediaPipeFaceLandmarkerModelPath();
[[nodiscard]] std::filesystem::path defaultMediaPipeRuntimeLibraryPath();

[[nodiscard]] bool isMediaPipeFaceMesh(const std::vector<std::vector<double>>& points) noexcept;

// Mirrors fsc_studio_services._best_matching_mesh_for_record: use bounding-box
// IoU when possible, otherwise select the largest detected face.
[[nodiscard]] std::vector<std::vector<double>> selectBestMediaPipeFaceMesh(
    const std::vector<std::vector<std::vector<double>>>& meshes,
    const std::vector<double>& recordBbox);

} // namespace fsc::mesh
