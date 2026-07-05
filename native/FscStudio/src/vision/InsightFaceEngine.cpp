#include "fsc/vision/InsightFaceEngine.hpp"

#include "fsc/core/VectorMath.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace fsc::vision {
namespace {

constexpr int DetectorInputSize = 640;
constexpr int RecognitionInputSize = 112;

std::wstring widePath(const std::filesystem::path& path) {
    return path.wstring();
}

std::vector<std::string> outputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    std::vector<std::string> names;
    names.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<const char*> namePointers(const std::vector<std::string>& names) {
    std::vector<const char*> pointers;
    pointers.reserve(names.size());
    for (const auto& name : names) {
        pointers.push_back(name.c_str());
    }
    return pointers;
}

std::vector<float> detectorTensor(const RgbImage& image, float& scale) {
    const auto canvas = letterboxToSquare(image, DetectorInputSize, scale);
    std::vector<float> tensor(static_cast<size_t>(1 * 3 * DetectorInputSize * DetectorInputSize));
    const int planeSize = DetectorInputSize * DetectorInputSize;
    for (int y = 0; y < DetectorInputSize; ++y) {
        for (int x = 0; x < DetectorInputSize; ++x) {
            const size_t pixel = static_cast<size_t>((y * DetectorInputSize + x) * 3);
            const int index = y * DetectorInputSize + x;
            tensor[static_cast<size_t>(0 * planeSize + index)] = (static_cast<float>(canvas.pixels[pixel]) - 127.5f) / 128.0f;
            tensor[static_cast<size_t>(1 * planeSize + index)] = (static_cast<float>(canvas.pixels[pixel + 1]) - 127.5f) / 128.0f;
            tensor[static_cast<size_t>(2 * planeSize + index)] = (static_cast<float>(canvas.pixels[pixel + 2]) - 127.5f) / 128.0f;
        }
    }
    return tensor;
}

RgbImage alignFace(const RgbImage& image, const std::array<Point2f, 5>& keypoints) {
    const auto transform = estimateSimilarityTransform(keypoints, arcFace112ReferencePoints());
    const float determinant = transform.a * transform.a + transform.b * transform.b;
    if (determinant <= 1e-12f) {
        throw std::runtime_error("Invalid face alignment transform.");
    }

    RgbImage output;
    output.width = RecognitionInputSize;
    output.height = RecognitionInputSize;
    output.pixels.resize(static_cast<size_t>(RecognitionInputSize * RecognitionInputSize * 3));
    for (int y = 0; y < RecognitionInputSize; ++y) {
        for (int x = 0; x < RecognitionInputSize; ++x) {
            const float dx = static_cast<float>(x) - transform.tx;
            const float dy = static_cast<float>(y) - transform.ty;
            const float srcX = (transform.a * dx + transform.b * dy) / determinant;
            const float srcY = (-transform.b * dx + transform.a * dy) / determinant;
            const size_t offset = static_cast<size_t>((y * RecognitionInputSize + x) * 3);
            for (int channel = 0; channel < 3; ++channel) {
                output.pixels[offset + static_cast<size_t>(channel)] =
                    sampleBilinearChannel(image, srcX, srcY, channel);
            }
        }
    }
    return output;
}

std::vector<float> recognitionTensor(const RgbImage& aligned) {
    if (aligned.width != RecognitionInputSize || aligned.height != RecognitionInputSize) {
        throw std::runtime_error("Aligned face must be 112x112.");
    }
    std::vector<float> tensor(static_cast<size_t>(1 * 3 * RecognitionInputSize * RecognitionInputSize));
    const int planeSize = RecognitionInputSize * RecognitionInputSize;
    for (int y = 0; y < RecognitionInputSize; ++y) {
        for (int x = 0; x < RecognitionInputSize; ++x) {
            const size_t pixel = static_cast<size_t>((y * RecognitionInputSize + x) * 3);
            const int index = y * RecognitionInputSize + x;
            tensor[static_cast<size_t>(0 * planeSize + index)] = (static_cast<float>(aligned.pixels[pixel]) - 127.5f) / 127.5f;
            tensor[static_cast<size_t>(1 * planeSize + index)] = (static_cast<float>(aligned.pixels[pixel + 1]) - 127.5f) / 127.5f;
            tensor[static_cast<size_t>(2 * planeSize + index)] = (static_cast<float>(aligned.pixels[pixel + 2]) - 127.5f) / 127.5f;
        }
    }
    return tensor;
}

float tensorAt(const Ort::Value& value, int row, int col) {
    const auto* data = value.GetTensorData<float>();
    const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || row < 0 || col < 0 || row >= shape[0] || col >= shape[1]) {
        throw std::runtime_error("Unexpected SCRFD output tensor shape.");
    }
    return data[static_cast<size_t>(row * shape[1] + col)];
}

float clamp(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

} // namespace

class InsightFaceEngine::Impl {
public:
    Impl(InsightFaceModelPaths models, RuntimeMode mode)
        : models_(std::move(models)),
          mode_(mode),
          env_(ORT_LOGGING_LEVEL_WARNING, "FSC Studio Native InsightFace") {
        const auto missing = models_.missingFiles();
        if (!missing.empty()) {
            throw std::runtime_error("Missing InsightFace model: " + missing.front().string());
        }

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        detector_ = std::make_unique<Ort::Session>(env_, widePath(models_.detectionModelPath).c_str(), options);
        recognizer_ = std::make_unique<Ort::Session>(env_, widePath(models_.recognitionModelPath).c_str(), options);
        detectorOutputNames_ = outputNames(*detector_, allocator_);
        recognizerOutputNames_ = outputNames(*recognizer_, allocator_);
    }

    std::vector<Detection> detect(const RgbImage& image, float threshold, int maxFaces) const {
        if (image.empty()) {
            throw std::runtime_error("Cannot detect faces in an empty RGB image.");
        }

        float scale = 1.0f;
        auto input = detectorTensor(image, scale);
        std::array<int64_t, 4> inputShape{1, 3, DetectorInputSize, DetectorInputSize};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputName = "input.1";
        const auto outputNamePointers = namePointers(detectorOutputNames_);
        auto outputs = detector_->Run(
            Ort::RunOptions{nullptr},
            &inputName,
            &inputValue,
            1,
            outputNamePointers.data(),
            outputNamePointers.size());

        const std::array<int, 3> strides{8, 16, 32};
        std::vector<Detection> detections;
        for (size_t strideIndex = 0; strideIndex < strides.size(); ++strideIndex) {
            const int stride = strides[strideIndex];
            const int featureWidth = DetectorInputSize / stride;
            const int featureHeight = DetectorInputSize / stride;
            constexpr int anchorCount = 2;
            const int total = featureWidth * featureHeight * anchorCount;
            const auto& scores = outputs[strideIndex];
            const auto& boxes = outputs[strideIndex + 3];
            const auto& landmarks = outputs[strideIndex + 6];
            for (int index = 0; index < total; ++index) {
                const float score = tensorAt(scores, index, 0);
                if (score < threshold) {
                    continue;
                }
                const int anchor = index / anchorCount;
                const float anchorX = static_cast<float>((anchor % featureWidth) * stride);
                const float anchorY = static_cast<float>((anchor / featureWidth) * stride);
                const float left = tensorAt(boxes, index, 0) * stride;
                const float top = tensorAt(boxes, index, 1) * stride;
                const float right = tensorAt(boxes, index, 2) * stride;
                const float bottom = tensorAt(boxes, index, 3) * stride;

                Detection detection;
                detection.score = score;
                detection.box = {
                    clamp((anchorX - left) / scale, 0.0f, static_cast<float>(image.width - 1)),
                    clamp((anchorY - top) / scale, 0.0f, static_cast<float>(image.height - 1)),
                    clamp((anchorX + right) / scale, 0.0f, static_cast<float>(image.width - 1)),
                    clamp((anchorY + bottom) / scale, 0.0f, static_cast<float>(image.height - 1)),
                };
                for (int point = 0; point < 5; ++point) {
                    detection.keypoints[static_cast<size_t>(point)] = {
                        clamp((anchorX + tensorAt(landmarks, index, point * 2) * stride) / scale, 0.0f, static_cast<float>(image.width - 1)),
                        clamp((anchorY + tensorAt(landmarks, index, point * 2 + 1) * stride) / scale, 0.0f, static_cast<float>(image.height - 1)),
                    };
                }
                detections.push_back(std::move(detection));
            }
        }

        return nonMaximumSuppression(std::move(detections), 0.4f, maxFaces);
    }

    std::vector<float> extractEmbedding(const RgbImage& image, const std::array<Point2f, 5>& keypoints) const {
        auto aligned = alignFace(image, keypoints);
        auto input = recognitionTensor(aligned);
        std::array<int64_t, 4> inputShape{1, 3, RecognitionInputSize, RecognitionInputSize};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputName = "input.1";
        const auto outputNamePointers = namePointers(recognizerOutputNames_);
        auto outputs = recognizer_->Run(
            Ort::RunOptions{nullptr},
            &inputName,
            &inputValue,
            1,
            outputNamePointers.data(),
            outputNamePointers.size());
        const auto* data = outputs.front().GetTensorData<float>();
        const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty() || shape.back() != 512) {
            throw std::runtime_error("Unexpected ArcFace embedding output shape.");
        }
        std::vector<float> embedding(data, data + 512);
        return fsc::core::normalize(embedding);
    }

private:
    InsightFaceModelPaths models_;
    RuntimeMode mode_;
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> detector_;
    std::unique_ptr<Ort::Session> recognizer_;
    std::vector<std::string> detectorOutputNames_;
    std::vector<std::string> recognizerOutputNames_;
};

InsightFaceEngine::InsightFaceEngine(InsightFaceModelPaths models, RuntimeMode mode)
    : impl_(std::make_unique<Impl>(std::move(models), mode)) {}

InsightFaceEngine::~InsightFaceEngine() = default;
InsightFaceEngine::InsightFaceEngine(InsightFaceEngine&&) noexcept = default;
InsightFaceEngine& InsightFaceEngine::operator=(InsightFaceEngine&&) noexcept = default;

std::vector<Detection> InsightFaceEngine::detect(const RgbImage& image, float detectionThreshold, int maxFaces) const {
    return impl_->detect(image, detectionThreshold, maxFaces);
}

std::vector<float> InsightFaceEngine::extractEmbedding(
    const RgbImage& image,
    const std::array<Point2f, 5>& keypoints) const {
    return impl_->extractEmbedding(image, keypoints);
}

std::vector<AnalyzedFace> InsightFaceEngine::analyze(const RgbImage& image, float detectionThreshold, int maxFaces) const {
    std::vector<AnalyzedFace> faces;
    for (auto& detection : detect(image, detectionThreshold, maxFaces)) {
        faces.push_back({detection, extractEmbedding(image, detection.keypoints)});
    }
    return faces;
}

} // namespace fsc::vision
