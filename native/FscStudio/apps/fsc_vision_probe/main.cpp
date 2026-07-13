#include "fsc/core/PathEncoding.hpp"
#include "fsc/vision/FaceGeometry.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"
#include "fsc/vision/OnnxRuntimeSession.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace fsc::vision;

namespace {

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  fsc_vision_probe models <model_root>\n"
        << "  fsc_vision_probe onnx <model_path> [auto|cpu|directml|cuda|qnn-npu|qnn-gpu]\n"
        << "  fsc_vision_probe detect <model_root> <image_path> [threshold] [embedding_output.txt]\n"
        << "  fsc_vision_probe benchmark <model_root> <image_path> [mode] [iterations] [warmup]\n"
        << "  fsc_vision_probe decode <image_path> <output.ppm>\n"
        << "  fsc_vision_probe align\n";
}

std::string shapeText(const std::vector<int64_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            text += ",";
        }
        text += std::to_string(shape[i]);
    }
    text += "]";
    return text;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 2;
    }

    const std::string command = argv[1];
    if (command == "decode") {
        if (argc < 4) {
            printUsage();
            return 2;
        }
        const auto image = loadImageRgb(fsc::core::pathFromUtf8(argv[2]));
        std::ofstream output(fsc::core::pathFromUtf8(argv[3]), std::ios::binary | std::ios::trunc);
        if (!output) {
            std::cerr << "Failed to create decoded PPM output.\n";
            return 1;
        }
        output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
        output.write(
            reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
        return output ? 0 : 1;
    }

    if (command == "onnx") {
        if (argc < 3) {
            printUsage();
            return 2;
        }
        const auto mode = argc >= 4 ? parseRuntimeMode(argv[3]) : RuntimeMode::Auto;
        const auto info = inspectOnnxModel(argv[2], mode);
        std::cout << "model=" << info.modelPath.string() << "\n";
        std::cout << "requested_mode=" << toString(info.requestedMode) << "\n";
        std::cout << "provider=" << info.provider << "\n";
        for (const auto& item : info.inputs) {
            std::cout << "input=" << item.name << "\t" << item.elementType << "\t" << shapeText(item.shape) << "\n";
        }
        for (const auto& item : info.outputs) {
            std::cout << "output=" << item.name << "\t" << item.elementType << "\t" << shapeText(item.shape) << "\n";
        }
        return info.provider == "not built with ONNX Runtime" ? 1 : 0;
    }

    if (command == "detect") {
#ifdef FSC_ENABLE_ONNX
        if (argc < 4) {
            printUsage();
            return 2;
        }
        const float threshold = argc >= 5 ? std::stof(argv[4]) : 0.55f;
        const std::string embeddingOutputPath = argc >= 6 ? argv[5] : "";
        const auto models = InsightFaceModelPaths::fromBuffaloL(argv[2]);
        const auto image = loadImageRgb(argv[3]);
        InsightFaceEngine engine(models, RuntimeMode::Cpu);
        const auto faces = engine.analyze(image, threshold, 10);
        std::ofstream embeddingOutput;
        if (!embeddingOutputPath.empty()) {
            embeddingOutput.open(embeddingOutputPath, std::ios::out | std::ios::trunc);
            if (!embeddingOutput) {
                std::cerr << "Failed to open embedding output: " << embeddingOutputPath << "\n";
                return 1;
            }
            embeddingOutput << std::fixed << std::setprecision(9);
        }
        std::cout << "faces=" << faces.size() << "\n";
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            std::cout
                << "face=" << i
                << "\tscore=" << std::fixed << std::setprecision(4) << face.detection.score
                << "\tbox=[" << face.detection.box.x1 << "," << face.detection.box.y1 << ","
                << face.detection.box.x2 << "," << face.detection.box.y2 << "]"
                << "\tembedding_dim=" << face.embedding.size();
            if (!face.embedding.empty()) {
                double norm = 0.0;
                for (const float value : face.embedding) {
                    norm += static_cast<double>(value) * static_cast<double>(value);
                }
                std::cout << "\tembedding_norm=" << std::sqrt(norm);
            }
            std::cout
                << "\tlandmarks2d=" << face.landmarks2d.size()
                << "\tlandmarks3d=" << face.landmarks3d.size()
                << "\tquality=" << face.qualityScore;
            if (!face.landmarks2d.empty()) {
                const auto& point = face.landmarks2d.front();
                std::cout << "\tfirst2d=[" << point.x << "," << point.y << "]";
            }
            if (!face.landmarks3d.empty()) {
                const auto& point = face.landmarks3d.front();
                std::cout << "\tfirst3d=[" << point.x << "," << point.y << "," << point.z << "]";
            }
            std::cout << "\n";
            if (embeddingOutput) {
                for (size_t j = 0; j < face.embedding.size(); ++j) {
                    if (j > 0) {
                        embeddingOutput << ",";
                    }
                    embeddingOutput << face.embedding[j];
                }
                embeddingOutput << "\n";
            }
        }
        return 0;
#else
        std::cerr << "detect requires FSC_ENABLE_ONNX=ON.\n";
        return 1;
#endif
    }

    if (command == "benchmark") {
#ifdef FSC_ENABLE_ONNX
        if (argc < 4) {
            printUsage();
            return 2;
        }
        const auto mode = argc >= 5 ? parseRuntimeMode(argv[4]) : RuntimeMode::Auto;
        const int iterations = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 10;
        const int warmup = argc >= 7 ? std::max(0, std::atoi(argv[6])) : 2;
        const auto models = InsightFaceModelPaths::fromBuffaloL(fsc::core::pathFromUtf8(argv[2]));
        const auto image = loadImageRgb(fsc::core::pathFromUtf8(argv[3]));

        const auto initializeStart = std::chrono::steady_clock::now();
        InsightFaceEngine engine(models, mode);
        const auto initializeEnd = std::chrono::steady_clock::now();
        for (int index = 0; index < warmup; ++index) {
            (void)engine.analyze(image, 0.50f, 10);
        }

        std::vector<double> durations;
        durations.reserve(static_cast<size_t>(iterations));
        size_t totalFaces = 0;
        for (int index = 0; index < iterations; ++index) {
            const auto start = std::chrono::steady_clock::now();
            const auto faces = engine.analyze(image, 0.50f, 10);
            const auto end = std::chrono::steady_clock::now();
            totalFaces += faces.size();
            durations.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
        std::sort(durations.begin(), durations.end());
        const size_t p95Index = static_cast<size_t>(
            std::ceil(0.95 * static_cast<double>(durations.size())) - 1.0);
        const double mean = sum / static_cast<double>(durations.size());
        std::cout
            << std::fixed << std::setprecision(3)
            << "requested_mode=" << toString(mode) << "\n"
            << "provider=" << executionProviderName(engine.actualRuntimeMode()) << "\n"
            << "image=" << image.width << "x" << image.height << "\n"
            << "initialization_ms="
            << std::chrono::duration<double, std::milli>(initializeEnd - initializeStart).count() << "\n"
            << "warmup=" << warmup << "\n"
            << "iterations=" << iterations << "\n"
            << "mean_faces=" << static_cast<double>(totalFaces) / static_cast<double>(iterations) << "\n"
            << "mean_ms=" << mean << "\n"
            << "median_ms=" << durations[durations.size() / 2] << "\n"
            << "p95_ms=" << durations[std::min(p95Index, durations.size() - 1)] << "\n"
            << "fps=" << (mean > 0.0 ? 1000.0 / mean : 0.0) << "\n";
        return 0;
#else
        std::cerr << "benchmark requires FSC_ENABLE_ONNX=ON.\n";
        return 1;
#endif
    }

    if (command == "models") {
        if (argc < 3) {
            printUsage();
            return 2;
        }
        const auto paths = InsightFaceModelPaths::fromBuffaloL(argv[2]);
        std::cout << "root=" << paths.rootDirectory.string() << "\n";
        std::cout << "detector=" << paths.detectionModelPath.string() << "\n";
        std::cout << "recognizer=" << paths.recognitionModelPath.string() << "\n";
        std::cout << "landmark2d=" << paths.landmark2dModelPath.string() << "\n";
        std::cout << "landmark3d=" << paths.landmark3dModelPath.string() << "\n";
        std::cout << "genderage=" << paths.genderAgeModelPath.string() << "\n";
        const auto missing = paths.missingFiles();
        if (missing.empty()) {
            std::cout << "status=ready\n";
            return 0;
        }
        std::cout << "status=missing\n";
        for (const auto& path : missing) {
            std::cout << "missing=" << path.string() << "\n";
        }
        return 1;
    }

    if (command == "align") {
        const auto reference = arcFace112ReferencePoints();
        const auto transform = estimateSimilarityTransform(reference, reference);
        std::cout
            << std::fixed << std::setprecision(6)
            << "a=" << transform.a << "\n"
            << "b=" << transform.b << "\n"
            << "tx=" << transform.tx << "\n"
            << "ty=" << transform.ty << "\n";
        return 0;
    }

    printUsage();
    return 2;
}
