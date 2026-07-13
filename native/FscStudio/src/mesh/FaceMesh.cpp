#include "fsc/mesh/FaceMesh.hpp"

#include "fsc/core/PathEncoding.hpp"

#ifdef FSC_ENABLE_ONNX
#include "fsc/vision/RuntimeProvider.hpp"
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fsc::mesh {
namespace {

constexpr int kMeshInputSize = 256;
constexpr float kDefaultCropScale = 1.5f;

std::string utf8Path(const std::filesystem::path& path) {
    return fsc::core::pathToUtf8(path);
}

std::filesystem::path processDirectory() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (written == 0) {
            return {};
        }
        if (written < buffer.size() - 1) {
            buffer.resize(written);
            return std::filesystem::path(buffer).parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    return {};
#endif
}

std::filesystem::path environmentPath(const char* name) {
    if (const char* value = std::getenv(name); value != nullptr && *value != '\0') {
        return std::filesystem::u8path(value);
    }
    return {};
}

std::filesystem::path existingPath(const std::filesystem::path& path) {
    return !path.empty() && std::filesystem::is_regular_file(path) ? path : std::filesystem::path{};
}

double bboxArea(const std::vector<double>& bbox) {
    if (bbox.size() < 4) {
        return 0.0;
    }
    return std::max(0.0, bbox[2] - bbox[0]) * std::max(0.0, bbox[3] - bbox[1]);
}

std::vector<double> bboxFromMesh(const std::vector<std::vector<double>>& mesh) {
    if (mesh.empty()) {
        return {};
    }
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& point : mesh) {
        if (point.size() < 2 || !std::isfinite(point[0]) || !std::isfinite(point[1])) {
            continue;
        }
        minX = std::min(minX, point[0]);
        minY = std::min(minY, point[1]);
        maxX = std::max(maxX, point[0]);
        maxY = std::max(maxY, point[1]);
    }
    if (!std::isfinite(minX) || !std::isfinite(minY) || !std::isfinite(maxX) || !std::isfinite(maxY)) {
        return {};
    }
    return {minX, minY, maxX, maxY};
}

double bboxIou(const std::vector<double>& first, const std::vector<double>& second) {
    if (first.size() < 4 || second.size() < 4) {
        return 0.0;
    }
    const double left = std::max(first[0], second[0]);
    const double top = std::max(first[1], second[1]);
    const double right = std::min(first[2], second[2]);
    const double bottom = std::min(first[3], second[3]);
    const double intersection = std::max(0.0, right - left) * std::max(0.0, bottom - top);
    const double denominator = bboxArea(first) + bboxArea(second) - intersection;
    return denominator > 0.0 ? intersection / denominator : 0.0;
}

struct FaceCropTransform {
    double centerX = 0.0;
    double centerY = 0.0;
    double side = 0.0;
    double cosine = 1.0;
    double sine = 0.0;
};

FaceCropTransform faceCropTransform(
    const std::vector<double>& faceBox,
    const std::vector<std::vector<double>>& keypoints,
    float cropScale) {
    if (faceBox.size() < 4 || !std::all_of(faceBox.begin(), faceBox.begin() + 4, [](double value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("Dense Mesh requires a valid face bounding box.");
    }
    const double width = faceBox[2] - faceBox[0];
    const double height = faceBox[3] - faceBox[1];
    if (width <= 1.0 || height <= 1.0) {
        throw std::runtime_error("Dense Mesh face bounding box is too small.");
    }

    FaceCropTransform transform;
    transform.centerX = (faceBox[0] + faceBox[2]) * 0.5;
    transform.centerY = (faceBox[1] + faceBox[3]) * 0.5;
    transform.side = std::max(width, height) * std::clamp(cropScale, 1.0f, 3.0f);

    if (keypoints.size() >= 2 && keypoints[0].size() >= 2 && keypoints[1].size() >= 2) {
        const double dx = keypoints[1][0] - keypoints[0][0];
        const double dy = keypoints[1][1] - keypoints[0][1];
        if (std::isfinite(dx) && std::isfinite(dy) && std::hypot(dx, dy) > 1.0) {
            const double angle = std::atan2(dy, dx);
            transform.cosine = std::cos(angle);
            transform.sine = std::sin(angle);
        }
    }
    return transform;
}

float sampleWithZeroBorder(const fsc::vision::RgbImage& image, double x, double y, int channel) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double dx = x - x0;
    const double dy = y - y0;
    const auto at = [&](int px, int py) -> double {
        if (px < 0 || py < 0 || px >= image.width || py >= image.height) {
            return 0.0;
        }
        return image.pixels[static_cast<std::size_t>((py * image.width + px) * 3 + channel)];
    };
    const double top = at(x0, y0) * (1.0 - dx) + at(x1, y0) * dx;
    const double bottom = at(x0, y1) * (1.0 - dx) + at(x1, y1) * dx;
    return static_cast<float>((top * (1.0 - dy) + bottom * dy) / 255.0);
}

std::vector<float> makeInputTensor(
    const fsc::vision::RgbImage& image,
    const FaceCropTransform& transform) {
    std::vector<float> input(static_cast<std::size_t>(kMeshInputSize * kMeshInputSize * 3));
    const double scale = transform.side / kMeshInputSize;
    for (int y = 0; y < kMeshInputSize; ++y) {
        for (int x = 0; x < kMeshInputSize; ++x) {
            const double cropX = (x + 0.5 - kMeshInputSize * 0.5) * scale;
            const double cropY = (y + 0.5 - kMeshInputSize * 0.5) * scale;
            const double sourceX = transform.centerX + transform.cosine * cropX - transform.sine * cropY;
            const double sourceY = transform.centerY + transform.sine * cropX + transform.cosine * cropY;
            const std::size_t offset = static_cast<std::size_t>((y * kMeshInputSize + x) * 3);
            input[offset] = sampleWithZeroBorder(image, sourceX, sourceY, 0);
            input[offset + 1] = sampleWithZeroBorder(image, sourceX, sourceY, 1);
            input[offset + 2] = sampleWithZeroBorder(image, sourceX, sourceY, 2);
        }
    }
    return input;
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const double expValue = std::exp(-value);
        return 1.0 / (1.0 + expValue);
    }
    const double expValue = std::exp(value);
    return expValue / (1.0 + expValue);
}

} // namespace

class MediaPipeFaceLandmarker::Impl {
public:
    explicit Impl(MediaPipeFaceLandmarkerOptions options)
        : minFacePresenceConfidence_(std::clamp(options.minFacePresenceConfidence, 0.0f, 1.0f)),
          cropScale_(options.cropScale > 0.0f ? options.cropScale : kDefaultCropScale) {
#ifdef FSC_ENABLE_ONNX
        modelPath_ = options.modelAssetPath.empty()
            ? defaultMediaPipeFaceLandmarkerModelPath()
            : std::move(options.modelAssetPath);
        if (!std::filesystem::is_regular_file(modelPath_)) {
            throw std::runtime_error("MediaPipe face-landmark ONNX model was not found: " + utf8Path(modelPath_));
        }

        std::vector<fsc::vision::RuntimeMode> candidates;
        if (options.runtimeMode == fsc::vision::RuntimeMode::Auto) {
            candidates = fsc::vision::detail::automaticRuntimeCandidates();
            candidates.erase(
                std::remove(candidates.begin(), candidates.end(), fsc::vision::RuntimeMode::QnnNpu),
                candidates.end());
        } else if (options.runtimeMode == fsc::vision::RuntimeMode::QnnNpu) {
            // This floating-point display model is better suited to Adreno GPU;
            // the quantized InsightFace models continue to use HTP/NPU.
            if (fsc::vision::detail::runtimeModeCompiled(fsc::vision::RuntimeMode::QnnGpu)) {
                candidates.push_back(fsc::vision::RuntimeMode::QnnGpu);
            }
            candidates.push_back(fsc::vision::RuntimeMode::Cpu);
        } else {
            candidates.push_back(options.runtimeMode);
        }

        std::ostringstream failures;
        for (const auto candidate : candidates) {
            try {
                auto sessionOptions = fsc::vision::detail::sessionOptionsFor(candidate);
                session_ = std::make_unique<Ort::Session>(environment_, modelPath_.wstring().c_str(), sessionOptions);
                runtimeMode_ = candidate;
                break;
            } catch (const std::exception& exception) {
                if (failures.tellp() > 0) {
                    failures << "; ";
                }
                failures << fsc::vision::toString(candidate) << ": " << exception.what();
            }
        }
        if (!session_) {
            throw std::runtime_error("No ONNX Runtime provider could load the dense face mesh model: " + failures.str());
        }

        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1) {
            throw std::runtime_error("Dense face mesh model must have exactly one input tensor.");
        }
        inputName_ = session_->GetInputNameAllocated(0, allocator).get();
        for (std::size_t index = 0; index < session_->GetOutputCount(); ++index) {
            const auto name = session_->GetOutputNameAllocated(index, allocator);
            const std::string value = name.get();
            if (value == "Identity") {
                landmarksOutputName_ = value;
            } else if (value == "Identity_1") {
                presenceOutputName_ = value;
            }
        }
        if (landmarksOutputName_.empty() || presenceOutputName_.empty()) {
            throw std::runtime_error("Dense face mesh model is missing the expected landmark or presence output.");
        }
#else
        (void)options;
        throw std::runtime_error("Dense Mesh generation requires an ONNX-enabled FSC Studio build.");
#endif
    }

    std::vector<std::vector<double>> detect(
        const fsc::vision::RgbImage& image,
        const std::vector<double>& faceBox,
        const std::vector<std::vector<double>>& keypoints) const {
#ifdef FSC_ENABLE_ONNX
        if (image.empty()) {
            throw std::runtime_error("Cannot detect a face mesh in an empty image.");
        }
        const auto transform = faceCropTransform(faceBox, keypoints, cropScale_);
        auto input = makeInputTensor(image, transform);
        const std::array<int64_t, 4> inputShape{1, kMeshInputSize, kMeshInputSize, 3};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());
        const std::array<const char*, 1> inputNames{inputName_.c_str()};
        const std::array<const char*, 2> outputNames{
            landmarksOutputName_.c_str(),
            presenceOutputName_.c_str(),
        };
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            inputNames.size(),
            outputNames.data(),
            outputNames.size());
        if (outputs.size() != 2 || !outputs[0].IsTensor() || !outputs[1].IsTensor()) {
            throw std::runtime_error("Dense face mesh model returned invalid outputs.");
        }
        const auto landmarkInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const auto presenceInfo = outputs[1].GetTensorTypeAndShapeInfo();
        if (landmarkInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            landmarkInfo.GetElementCount() != kMediaPipeFaceMeshPointCount * 3 ||
            presenceInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            presenceInfo.GetElementCount() < 1) {
            throw std::runtime_error("Dense face mesh model returned incompatible tensor shapes.");
        }

        const float* rawPoints = outputs[0].GetTensorData<float>();
        const double presence = sigmoid(outputs[1].GetTensorData<float>()[0]);
        if (presence < minFacePresenceConfidence_) {
            throw std::runtime_error(
                "Dense face mesh confidence is below the configured threshold (" +
                std::to_string(presence) + ").");
        }

        const double scale = transform.side / kMeshInputSize;
        std::vector<std::vector<double>> mesh;
        mesh.reserve(kMediaPipeFaceMeshPointCount);
        for (std::size_t index = 0; index < kMediaPipeFaceMeshPointCount; ++index) {
            const double cropX = (rawPoints[index * 3] - kMeshInputSize * 0.5) * scale;
            const double cropY = (rawPoints[index * 3 + 1] - kMeshInputSize * 0.5) * scale;
            mesh.push_back({
                transform.centerX + transform.cosine * cropX - transform.sine * cropY,
                transform.centerY + transform.sine * cropX + transform.cosine * cropY,
                rawPoints[index * 3 + 2] * scale,
            });
        }
        return mesh;
#else
        (void)image;
        (void)faceBox;
        (void)keypoints;
        throw std::runtime_error("Dense Mesh generation requires an ONNX-enabled FSC Studio build.");
#endif
    }

private:
    float minFacePresenceConfidence_ = 0.5f;
    float cropScale_ = kDefaultCropScale;
#ifdef FSC_ENABLE_ONNX
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "FSC Studio Dense Mesh"};
    std::filesystem::path modelPath_;
    std::unique_ptr<Ort::Session> session_;
    fsc::vision::RuntimeMode runtimeMode_ = fsc::vision::RuntimeMode::Cpu;
    std::string inputName_;
    std::string landmarksOutputName_;
    std::string presenceOutputName_;
#endif
};

MediaPipeFaceLandmarker::MediaPipeFaceLandmarker(MediaPipeFaceLandmarkerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

MediaPipeFaceLandmarker::~MediaPipeFaceLandmarker() = default;
MediaPipeFaceLandmarker::MediaPipeFaceLandmarker(MediaPipeFaceLandmarker&&) noexcept = default;
MediaPipeFaceLandmarker& MediaPipeFaceLandmarker::operator=(MediaPipeFaceLandmarker&&) noexcept = default;

std::vector<std::vector<double>> MediaPipeFaceLandmarker::detect(
    const fsc::vision::RgbImage& image,
    const std::vector<double>& faceBox,
    const std::vector<std::vector<double>>& keypoints) const {
    return impl_->detect(image, faceBox, keypoints);
}

std::filesystem::path defaultMediaPipeFaceLandmarkerModelPath() {
    for (const char* name : {"FSC_FACE_MESH_MODEL", "FSC_MEDIAPIPE_FACE_LANDMARKER_MODEL"}) {
        if (const auto configured = existingPath(environmentPath(name)); !configured.empty()) {
            return configured;
        }
    }
    if (const auto packaged = existingPath(
            processDirectory() / "models" / "mediapipe" / "face_landmarks_detector.onnx");
        !packaged.empty()) {
        return packaged;
    }
    if (const auto workingDirectory = existingPath(
            std::filesystem::current_path() / "model" / "mediapipe" / "face_landmarks_detector.onnx");
        !workingDirectory.empty()) {
        return workingDirectory;
    }
    return {};
}

bool isMediaPipeFaceMesh(const std::vector<std::vector<double>>& points) noexcept {
    if (points.size() != kMediaPipeFaceMeshPointCount) {
        return false;
    }
    for (const auto& point : points) {
        if (point.size() != 3 || !std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<double>> selectBestMediaPipeFaceMesh(
    const std::vector<std::vector<std::vector<double>>>& meshes,
    const std::vector<double>& recordBbox) {
    if (meshes.empty()) {
        throw std::runtime_error("MediaPipe did not detect a face mesh in the source image.");
    }
    const std::vector<std::vector<double>>* selected = &meshes.front();
    if (meshes.size() > 1 && recordBbox.size() >= 4) {
        double bestScore = -1.0;
        for (const auto& mesh : meshes) {
            const double score = bboxIou(recordBbox, bboxFromMesh(mesh));
            if (score > bestScore) {
                bestScore = score;
                selected = &mesh;
            }
        }
        if (bestScore <= 0.0) {
            selected = nullptr;
        }
    }
    if (selected == nullptr) {
        selected = &*std::max_element(meshes.begin(), meshes.end(), [](const auto& left, const auto& right) {
            return bboxArea(bboxFromMesh(left)) < bboxArea(bboxFromMesh(right));
        });
    }
    if (!isMediaPipeFaceMesh(*selected)) {
        throw std::runtime_error(
            "MediaPipe returned " + std::to_string(selected->size()) +
            " landmarks; FSC requires the 478-point Face Landmarker result.");
    }
    return *selected;
}

} // namespace fsc::mesh
