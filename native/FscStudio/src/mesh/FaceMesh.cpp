#include "fsc/mesh/FaceMesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fsc::mesh {
namespace {

constexpr int kMpImageFormatSrgb = 1;
constexpr int kMpRunningModeImage = 1;
constexpr int kMpHostEnvironmentPythonCompatible = 3;
constexpr int kMpHostSystemWindows = 3;

struct BaseOptionsC {
    const char* modelAssetBuffer = nullptr;
    unsigned int modelAssetBufferCount = 0;
    const char* modelAssetPath = nullptr;
    int delegate = 0;
    int hostEnvironment = 0;
    int hostSystem = 0;
    const char* hostVersion = nullptr;
    const char* caBundlePath = nullptr;
};

struct NormalizedLandmarkC {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    bool hasVisibility = false;
    float visibility = 0.0f;
    bool hasPresence = false;
    float presence = 0.0f;
    const char* name = nullptr;
};

struct NormalizedLandmarksC {
    const NormalizedLandmarkC* landmarks = nullptr;
    unsigned int landmarksCount = 0;
};

struct CategoriesC;
struct MatrixC;
struct FaceLandmarkerResultC {
    const NormalizedLandmarksC* faceLandmarks = nullptr;
    unsigned int faceLandmarksCount = 0;
    const CategoriesC* faceBlendshapes = nullptr;
    unsigned int faceBlendshapesCount = 0;
    const MatrixC* facialTransformationMatrixes = nullptr;
    unsigned int facialTransformationMatrixesCount = 0;
};

using FaceLandmarkerResultCallback = void (*)(int, const FaceLandmarkerResultC*, void*, long long);

struct FaceLandmarkerOptionsC {
    BaseOptionsC baseOptions{};
    int runningMode = kMpRunningModeImage;
    int numFaces = 10;
    float minFaceDetectionConfidence = 0.5f;
    float minFacePresenceConfidence = 0.5f;
    float minTrackingConfidence = 0.5f;
    bool outputFaceBlendshapes = false;
    bool outputFacialTransformationMatrixes = false;
    FaceLandmarkerResultCallback resultCallback = nullptr;
};

static_assert(sizeof(BaseOptionsC) == 56);
static_assert(sizeof(NormalizedLandmarkC) == 40);
static_assert(sizeof(NormalizedLandmarksC) == 16);
static_assert(sizeof(FaceLandmarkerResultC) == 48);
static_assert(sizeof(FaceLandmarkerOptionsC) == 88);

using ErrorFreeFn = void (*)(void*);
using ImageCreateFromUint8DataFn = int (*)(int, int, int, const unsigned char*, int, void**, char**);
using ImageFreeFn = void (*)(void*);
using FaceLandmarkerCreateFn = int (*)(const FaceLandmarkerOptionsC*, void**, char**);
using FaceLandmarkerDetectImageFn = int (*)(void*, void*, const void*, FaceLandmarkerResultC*, char**);
using FaceLandmarkerCloseResultFn = void (*)(FaceLandmarkerResultC*);
using FaceLandmarkerCloseFn = int (*)(void*, char**);

std::string utf8Path(const std::filesystem::path& path) {
#if defined(_WIN32)
    const auto value = path.u8string();
    return {value.begin(), value.end()};
#else
    return path.string();
#endif
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

} // namespace

class MediaPipeFaceLandmarker::Impl {
public:
    explicit Impl(MediaPipeFaceLandmarkerOptions options) {
#ifdef _WIN32
        const auto modelPath = options.modelAssetPath.empty()
            ? defaultMediaPipeFaceLandmarkerModelPath()
            : options.modelAssetPath;
        const auto runtimePath = options.runtimeLibraryPath.empty()
            ? defaultMediaPipeRuntimeLibraryPath()
            : options.runtimeLibraryPath;
        if (!std::filesystem::is_regular_file(modelPath)) {
            throw std::runtime_error("MediaPipe face-landmarker task was not found: " + utf8Path(modelPath));
        }
        if (!std::filesystem::is_regular_file(runtimePath)) {
            throw std::runtime_error("MediaPipe native runtime was not found: " + utf8Path(runtimePath));
        }

        library = LoadLibraryW(runtimePath.wstring().c_str());
        if (library == nullptr) {
            throw std::runtime_error("Failed to load MediaPipe native runtime: " + utf8Path(runtimePath));
        }
        try {
            errorFree = load<ErrorFreeFn>("MpErrorFree");
            imageCreate = load<ImageCreateFromUint8DataFn>("MpImageCreateFromUint8Data");
            imageFree = load<ImageFreeFn>("MpImageFree");
            faceCreate = load<FaceLandmarkerCreateFn>("MpFaceLandmarkerCreate");
            faceDetect = load<FaceLandmarkerDetectImageFn>("MpFaceLandmarkerDetectImage");
            faceCloseResult = load<FaceLandmarkerCloseResultFn>("MpFaceLandmarkerCloseResult");
            faceClose = load<FaceLandmarkerCloseFn>("MpFaceLandmarkerClose");

            modelPathUtf8 = utf8Path(modelPath);
            FaceLandmarkerOptionsC nativeOptions{};
            nativeOptions.baseOptions.modelAssetPath = modelPathUtf8.c_str();
            nativeOptions.baseOptions.delegate = 0;
            nativeOptions.baseOptions.hostEnvironment = kMpHostEnvironmentPythonCompatible;
            nativeOptions.baseOptions.hostSystem = kMpHostSystemWindows;
            nativeOptions.baseOptions.hostVersion = "native";
            nativeOptions.runningMode = kMpRunningModeImage;
            nativeOptions.numFaces = std::clamp(options.maxFaces, 1, 100);
            nativeOptions.minFaceDetectionConfidence = options.minFaceDetectionConfidence;
            nativeOptions.minFacePresenceConfidence = options.minFacePresenceConfidence;
            nativeOptions.minTrackingConfidence = 0.5f;
            check(faceCreate(&nativeOptions, &handle, &lastError), "create MediaPipe FaceLandmarker");
        } catch (...) {
            release();
            throw;
        }
#else
        (void)options;
        throw std::runtime_error("MediaPipe FaceLandmarker is currently implemented for Windows.");
#endif
    }

    ~Impl() { release(); }

    std::vector<std::vector<std::vector<double>>> detect(const fsc::vision::RgbImage& image) const {
#ifdef _WIN32
        if (image.empty()) {
            throw std::runtime_error("Cannot detect a face mesh in an empty image.");
        }
        if (image.pixels.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Image is too large for the MediaPipe face-landmarker API.");
        }

        void* nativeImage = nullptr;
        char* imageError = nullptr;
        check(imageCreate(
                kMpImageFormatSrgb,
                image.width,
                image.height,
                image.pixels.data(),
                static_cast<int>(image.pixels.size()),
                &nativeImage,
                &imageError),
            "create MediaPipe image",
            imageError);

        FaceLandmarkerResultC result{};
        char* detectError = nullptr;
        try {
            check(faceDetect(handle, nativeImage, nullptr, &result, &detectError), "detect MediaPipe face mesh", detectError);
            std::vector<std::vector<std::vector<double>>> meshes;
            meshes.reserve(result.faceLandmarksCount);
            for (unsigned int faceIndex = 0; faceIndex < result.faceLandmarksCount; ++faceIndex) {
                const auto& normalized = result.faceLandmarks[faceIndex];
                std::vector<std::vector<double>> mesh;
                mesh.reserve(normalized.landmarksCount);
                for (unsigned int pointIndex = 0; pointIndex < normalized.landmarksCount; ++pointIndex) {
                    const auto& point = normalized.landmarks[pointIndex];
                    mesh.push_back({
                        static_cast<double>(point.x) * image.width,
                        static_cast<double>(point.y) * image.height,
                        static_cast<double>(point.z) * image.width,
                    });
                }
                if (!mesh.empty()) {
                    meshes.push_back(std::move(mesh));
                }
            }
            faceCloseResult(&result);
            result = {};
            imageFree(nativeImage);
            return meshes;
        } catch (...) {
            if (result.faceLandmarks != nullptr) {
                faceCloseResult(&result);
            }
            if (nativeImage != nullptr) {
                imageFree(nativeImage);
            }
            throw;
        }
#else
        (void)image;
        throw std::runtime_error("MediaPipe FaceLandmarker is currently implemented for Windows.");
#endif
    }

private:
#ifdef _WIN32
    template <typename Function>
    Function load(const char* name) {
        const auto address = GetProcAddress(library, name);
        if (address == nullptr) {
            throw std::runtime_error(std::string("MediaPipe runtime is missing required function: ") + name);
        }
        return reinterpret_cast<Function>(address);
    }

    void check(int status, const char* action, char* suppliedError = nullptr) const {
        char* error = suppliedError != nullptr ? suppliedError : lastError;
        if (status == 0) {
            if (error != nullptr) {
                errorFree(error);
            }
            if (suppliedError == nullptr) {
                lastError = nullptr;
            }
            return;
        }
        const std::string message = error != nullptr ? error : "unknown MediaPipe error";
        if (error != nullptr) {
            errorFree(error);
        }
        if (suppliedError == nullptr) {
            lastError = nullptr;
        }
        throw std::runtime_error(std::string("Failed to ") + action + ": " + message);
    }

    void release() noexcept {
        if (handle != nullptr && faceClose != nullptr) {
            char* ignored = nullptr;
            faceClose(handle, &ignored);
            if (ignored != nullptr && errorFree != nullptr) {
                errorFree(ignored);
            }
            handle = nullptr;
        }
        if (library != nullptr) {
            FreeLibrary(library);
            library = nullptr;
        }
    }

    HMODULE library = nullptr;
    void* handle = nullptr;
    ErrorFreeFn errorFree = nullptr;
    ImageCreateFromUint8DataFn imageCreate = nullptr;
    ImageFreeFn imageFree = nullptr;
    FaceLandmarkerCreateFn faceCreate = nullptr;
    FaceLandmarkerDetectImageFn faceDetect = nullptr;
    FaceLandmarkerCloseResultFn faceCloseResult = nullptr;
    FaceLandmarkerCloseFn faceClose = nullptr;
    mutable char* lastError = nullptr;
    std::string modelPathUtf8;
#else
    void release() noexcept {}
#endif
};

MediaPipeFaceLandmarker::MediaPipeFaceLandmarker(MediaPipeFaceLandmarkerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

MediaPipeFaceLandmarker::~MediaPipeFaceLandmarker() = default;
MediaPipeFaceLandmarker::MediaPipeFaceLandmarker(MediaPipeFaceLandmarker&&) noexcept = default;
MediaPipeFaceLandmarker& MediaPipeFaceLandmarker::operator=(MediaPipeFaceLandmarker&&) noexcept = default;

std::vector<std::vector<std::vector<double>>> MediaPipeFaceLandmarker::detect(const fsc::vision::RgbImage& image) const {
    return impl_->detect(image);
}

std::filesystem::path defaultMediaPipeFaceLandmarkerModelPath() {
    if (const auto configured = existingPath(environmentPath("FSC_MEDIAPIPE_FACE_LANDMARKER_MODEL")); !configured.empty()) {
        return configured;
    }
    if (const auto packaged = existingPath(processDirectory() / "models" / "mediapipe" / "face_landmarker.task"); !packaged.empty()) {
        return packaged;
    }
    if (const auto workingDirectory = existingPath(std::filesystem::current_path() / "models" / "mediapipe" / "face_landmarker.task"); !workingDirectory.empty()) {
        return workingDirectory;
    }
    return {};
}

std::filesystem::path defaultMediaPipeRuntimeLibraryPath() {
    if (const auto configured = existingPath(environmentPath("FSC_MEDIAPIPE_RUNTIME_LIBRARY")); !configured.empty()) {
        return configured;
    }
    if (const auto packaged = existingPath(processDirectory() / "libmediapipe.dll"); !packaged.empty()) {
        return packaged;
    }
    return existingPath(std::filesystem::current_path() / "libmediapipe.dll");
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
