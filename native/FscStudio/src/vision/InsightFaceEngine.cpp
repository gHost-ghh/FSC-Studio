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
constexpr int LandmarkInputSize = 192;

struct LandmarkCropTransform {
    float scale = 1.0f;
    float tx = 0.0f;
    float ty = 0.0f;
};

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

std::uint8_t sampleBilinearChannelWithZeroBorder(const RgbImage& image, float x, float y, int channel) {
    if (image.empty() || channel < 0 || channel >= 3) {
        return 0;
    }
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) || y > static_cast<float>(image.height - 1)) {
        return 0;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    const auto at = [&](int px, int py) -> float {
        return static_cast<float>(image.pixels[static_cast<size_t>((py * image.width + px) * 3 + channel)]);
    };
    const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * dx;
    const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * dx;
    return static_cast<std::uint8_t>(std::clamp(top + (bottom - top) * dy, 0.0f, 255.0f));
}

RgbImage cropForLandmarks(const RgbImage& image, Rectf box, LandmarkCropTransform& transform) {
    const float width = std::max(1.0f, box.x2 - box.x1);
    const float height = std::max(1.0f, box.y2 - box.y1);
    const float centerX = (box.x1 + box.x2) * 0.5f;
    const float centerY = (box.y1 + box.y2) * 0.5f;
    transform.scale = LandmarkInputSize / (std::max(width, height) * 1.5f);
    transform.tx = LandmarkInputSize * 0.5f - centerX * transform.scale;
    transform.ty = LandmarkInputSize * 0.5f - centerY * transform.scale;

    RgbImage output;
    output.width = LandmarkInputSize;
    output.height = LandmarkInputSize;
    output.pixels.resize(static_cast<size_t>(LandmarkInputSize * LandmarkInputSize * 3));
    for (int y = 0; y < LandmarkInputSize; ++y) {
        for (int x = 0; x < LandmarkInputSize; ++x) {
            const float srcX = (static_cast<float>(x) - transform.tx) / transform.scale;
            const float srcY = (static_cast<float>(y) - transform.ty) / transform.scale;
            const size_t offset = static_cast<size_t>((y * LandmarkInputSize + x) * 3);
            for (int channel = 0; channel < 3; ++channel) {
                output.pixels[offset + static_cast<size_t>(channel)] =
                    sampleBilinearChannelWithZeroBorder(image, srcX, srcY, channel);
            }
        }
    }
    return output;
}

std::vector<float> landmarkTensor(const RgbImage& crop) {
    if (crop.width != LandmarkInputSize || crop.height != LandmarkInputSize) {
        throw std::runtime_error("Landmark crop must be 192x192.");
    }
    std::vector<float> tensor(static_cast<size_t>(1 * 3 * LandmarkInputSize * LandmarkInputSize));
    const int planeSize = LandmarkInputSize * LandmarkInputSize;
    for (int y = 0; y < LandmarkInputSize; ++y) {
        for (int x = 0; x < LandmarkInputSize; ++x) {
            const size_t pixel = static_cast<size_t>((y * LandmarkInputSize + x) * 3);
            const int index = y * LandmarkInputSize + x;
            tensor[static_cast<size_t>(0 * planeSize + index)] = static_cast<float>(crop.pixels[pixel]);
            tensor[static_cast<size_t>(1 * planeSize + index)] = static_cast<float>(crop.pixels[pixel + 1]);
            tensor[static_cast<size_t>(2 * planeSize + index)] = static_cast<float>(crop.pixels[pixel + 2]);
        }
    }
    return tensor;
}

std::vector<float> tensorValues(const Ort::Value& value) {
    const auto* data = value.GetTensorData<float>();
    const auto info = value.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    return {data, data + count};
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

double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

int reflect101(int value, int size) {
    if (size <= 1) {
        return 0;
    }
    while (value < 0 || value >= size) {
        if (value < 0) {
            value = -value;
        } else {
            value = 2 * size - 2 - value;
        }
    }
    return value;
}

AnalyzedFace scoreFaceQuality(AnalyzedFace face, const RgbImage& image) {
    const int x1 = static_cast<int>(std::clamp(std::round(face.detection.box.x1), 0.0f, static_cast<float>(image.width - 1)));
    const int y1 = static_cast<int>(std::clamp(std::round(face.detection.box.y1), 0.0f, static_cast<float>(image.height - 1)));
    const int x2 = static_cast<int>(std::clamp(std::round(face.detection.box.x2), 0.0f, static_cast<float>(image.width)));
    const int y2 = static_cast<int>(std::clamp(std::round(face.detection.box.y2), 0.0f, static_cast<float>(image.height)));
    const int faceWidth = std::max(0, x2 - x1);
    const int faceHeight = std::max(0, y2 - y1);
    const double imageArea = std::max(1, image.width * image.height);
    face.qualityAreaRatio = (faceWidth * faceHeight) / imageArea;
    if (faceWidth <= 0 || faceHeight <= 0) {
        return face;
    }

    std::vector<double> gray(static_cast<size_t>(faceWidth * faceHeight));
    double sum = 0.0;
    for (int y = 0; y < faceHeight; ++y) {
        for (int x = 0; x < faceWidth; ++x) {
            const size_t pixel = static_cast<size_t>(((y1 + y) * image.width + (x1 + x)) * 3);
            const double red = image.pixels[pixel];
            const double green = image.pixels[pixel + 1];
            const double blue = image.pixels[pixel + 2];
            const double value = std::round(0.299 * red + 0.587 * green + 0.114 * blue);
            gray[static_cast<size_t>(y * faceWidth + x)] = value;
            sum += value;
        }
    }

    const double count = static_cast<double>(gray.size());
    face.qualityBrightness = sum / count;
    double contrastAccumulator = 0.0;
    for (const double value : gray) {
        const double delta = value - face.qualityBrightness;
        contrastAccumulator += delta * delta;
    }
    face.qualityContrast = std::sqrt(contrastAccumulator / count);

    double laplacianSum = 0.0;
    double laplacianSquaredSum = 0.0;
    const auto grayAt = [&](int x, int y) -> double {
        x = reflect101(x, faceWidth);
        y = reflect101(y, faceHeight);
        return gray[static_cast<size_t>(y * faceWidth + x)];
    };
    for (int y = 0; y < faceHeight; ++y) {
        for (int x = 0; x < faceWidth; ++x) {
            const double value =
                grayAt(x - 1, y) +
                grayAt(x + 1, y) +
                grayAt(x, y - 1) +
                grayAt(x, y + 1) -
                4.0 * grayAt(x, y);
            laplacianSum += value;
            laplacianSquaredSum += value * value;
        }
    }
    const double laplacianMean = laplacianSum / count;
    face.qualitySharpness = std::max(0.0, laplacianSquaredSum / count - laplacianMean * laplacianMean);

    const double detComponent = clamp01((static_cast<double>(face.detection.score) - 0.45) / 0.5);
    const double areaComponent = clamp01(face.qualityAreaRatio / 0.08);
    const double sharpnessComponent = clamp01(face.qualitySharpness / 160.0);
    const double brightnessComponent = clamp01(1.0 - std::abs(face.qualityBrightness - 128.0) / 128.0);
    const double contrastComponent = clamp01(face.qualityContrast / 64.0);
    const double score =
        0.35 * detComponent +
        0.20 * areaComponent +
        0.25 * sharpnessComponent +
        0.10 * brightnessComponent +
        0.10 * contrastComponent;
    face.qualityScore = std::round(clamp01(score) * 1000000.0) / 1000000.0;
    return face;
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
        landmark2d_ = std::make_unique<Ort::Session>(env_, widePath(models_.landmark2dModelPath).c_str(), options);
        landmark3d_ = std::make_unique<Ort::Session>(env_, widePath(models_.landmark3dModelPath).c_str(), options);
        detectorOutputNames_ = outputNames(*detector_, allocator_);
        recognizerOutputNames_ = outputNames(*recognizer_, allocator_);
        landmark2dOutputNames_ = outputNames(*landmark2d_, allocator_);
        landmark3dOutputNames_ = outputNames(*landmark3d_, allocator_);
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

    std::vector<Point2f> extractLandmarks2d(const RgbImage& image, Rectf box) const {
        LandmarkCropTransform transform;
        auto crop = cropForLandmarks(image, box, transform);
        auto input = landmarkTensor(crop);
        std::array<int64_t, 4> inputShape{1, 3, LandmarkInputSize, LandmarkInputSize};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputName = "data";
        const auto outputNamePointers = namePointers(landmark2dOutputNames_);
        auto outputs = landmark2d_->Run(
            Ort::RunOptions{nullptr},
            &inputName,
            &inputValue,
            1,
            outputNamePointers.data(),
            outputNamePointers.size());
        const auto values = tensorValues(outputs.front());
        if (values.size() < 106 * 2 || values.size() % 2 != 0) {
            throw std::runtime_error("Unexpected 2D landmark output shape.");
        }

        const size_t pointCount = values.size() / 2;
        const size_t start = pointCount > 106 ? pointCount - 106 : 0;
        std::vector<Point2f> landmarks;
        landmarks.reserve(106);
        for (size_t i = start; i < pointCount; ++i) {
            const float cropX = (values[i * 2] + 1.0f) * (LandmarkInputSize * 0.5f);
            const float cropY = (values[i * 2 + 1] + 1.0f) * (LandmarkInputSize * 0.5f);
            landmarks.push_back({
                (cropX - transform.tx) / transform.scale,
                (cropY - transform.ty) / transform.scale,
            });
        }
        return landmarks;
    }

    std::vector<Point3f> extractLandmarks3d(const RgbImage& image, Rectf box) const {
        LandmarkCropTransform transform;
        auto crop = cropForLandmarks(image, box, transform);
        auto input = landmarkTensor(crop);
        std::array<int64_t, 4> inputShape{1, 3, LandmarkInputSize, LandmarkInputSize};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputName = "data";
        const auto outputNamePointers = namePointers(landmark3dOutputNames_);
        auto outputs = landmark3d_->Run(
            Ort::RunOptions{nullptr},
            &inputName,
            &inputValue,
            1,
            outputNamePointers.data(),
            outputNamePointers.size());
        const auto values = tensorValues(outputs.front());
        if (values.size() < 68 * 3 || values.size() % 3 != 0) {
            throw std::runtime_error("Unexpected 3D landmark output shape.");
        }

        const size_t pointCount = values.size() / 3;
        const size_t start = pointCount > 68 ? pointCount - 68 : 0;
        std::vector<Point3f> landmarks;
        landmarks.reserve(68);
        for (size_t i = start; i < pointCount; ++i) {
            const float cropX = (values[i * 3] + 1.0f) * (LandmarkInputSize * 0.5f);
            const float cropY = (values[i * 3 + 1] + 1.0f) * (LandmarkInputSize * 0.5f);
            const float cropZ = values[i * 3 + 2] * (LandmarkInputSize * 0.5f);
            landmarks.push_back({
                (cropX - transform.tx) / transform.scale,
                (cropY - transform.ty) / transform.scale,
                cropZ / transform.scale,
            });
        }
        return landmarks;
    }

private:
    InsightFaceModelPaths models_;
    RuntimeMode mode_;
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> detector_;
    std::unique_ptr<Ort::Session> recognizer_;
    std::unique_ptr<Ort::Session> landmark2d_;
    std::unique_ptr<Ort::Session> landmark3d_;
    std::vector<std::string> detectorOutputNames_;
    std::vector<std::string> recognizerOutputNames_;
    std::vector<std::string> landmark2dOutputNames_;
    std::vector<std::string> landmark3dOutputNames_;
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

std::vector<Point2f> InsightFaceEngine::extractLandmarks2d(const RgbImage& image, Rectf box) const {
    return impl_->extractLandmarks2d(image, box);
}

std::vector<Point3f> InsightFaceEngine::extractLandmarks3d(const RgbImage& image, Rectf box) const {
    return impl_->extractLandmarks3d(image, box);
}

std::vector<AnalyzedFace> InsightFaceEngine::analyze(const RgbImage& image, float detectionThreshold, int maxFaces) const {
    std::vector<AnalyzedFace> faces;
    for (auto& detection : detect(image, detectionThreshold, maxFaces)) {
        faces.push_back(scoreFaceQuality({
            detection,
            extractEmbedding(image, detection.keypoints),
            extractLandmarks2d(image, detection.box),
            extractLandmarks3d(image, detection.box),
        }, image));
    }
    return faces;
}

} // namespace fsc::vision
