#include "fsc/core/Database.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>

using namespace fsc::core;

namespace {

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  fsc_native_probe <database.fscdb> stats\n"
        << "  fsc_native_probe <database.fscdb> search <face_id> [top_k]\n"
        << "  fsc_native_probe <database.fscdb> identify <face_id> [strict|balanced|broad]\n"
        << "  fsc_native_probe <database.fscdb> image-search <model_root> <image.ppm> [top_k] [threshold] [strict|balanced|broad]\n";
}

IdentityMode parseMode(const std::string& value) {
    if (value == "balanced") {
        return IdentityMode::Balanced;
    }
    if (value == "broad") {
        return IdentityMode::Broad;
    }
    return IdentityMode::Strict;
}

void printIdentityResult(const IdentityResult& result) {
    std::cout << "decision=" << result.decision << "\n";
    std::cout << "message=" << result.message << "\n";
    for (const auto& candidate : result.candidates) {
        std::cout
            << candidate.profile.personName << "\t"
            << std::fixed << std::setprecision(4) << candidate.score << "\t"
            << "margin=" << candidate.margin << "\t"
            << "confidence=" << std::setprecision(1) << candidate.confidence * 100.0 << "%\t"
            << "evidence=" << candidate.evidenceFaceId << "\n";
    }
}

void printSearchHits(const std::vector<SearchHit>& hits, int64_t excludeFaceId = 0) {
    for (const auto& hit : hits) {
        if (excludeFaceId != 0 && hit.record.id == excludeFaceId) {
            continue;
        }
        std::cout
            << hit.record.id << "\t"
            << std::fixed << std::setprecision(4) << hit.cosine << "\t"
            << std::setprecision(1) << hit.similarityPercent() << "%\t"
            << hit.record.fileName << "\t"
            << hit.record.personName << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage();
        return 2;
    }

    try {
        Database database(argv[1]);
        const std::string command = argv[2];
        if (command == "stats") {
            const auto stats = database.statistics();
            std::cout << "format_version=" << stats.formatVersion << "\n";
            std::cout << "metric=" << stats.metric << "\n";
            std::cout << "model=" << stats.modelName << "\n";
            std::cout << "faces=" << stats.faceCount << "\n";
            std::cout << "people=" << stats.peopleCount << "\n";
            std::cout << "review=" << stats.reviewCount << "\n";
            std::cout << "average_quality=" << std::fixed << std::setprecision(4) << stats.averageQuality << "\n";
            return 0;
        }

        if (command == "search") {
            if (argc < 4) {
                printUsage();
                return 2;
            }
            const int64_t faceId = std::strtoll(argv[3], nullptr, 10);
            const int topK = argc >= 5 ? std::atoi(argv[4]) : 10;
            const auto query = database.loadFace(faceId);
            if (!query.has_value()) {
                std::cerr << "Face id not found: " << faceId << "\n";
                return 1;
            }
            const auto hits = searchFaces(database.loadFaces(false), query->embedding, topK, -1.0, false);
            printSearchHits(hits, faceId);
            return 0;
        }

        if (command == "identify") {
            if (argc < 4) {
                printUsage();
                return 2;
            }
            const int64_t faceId = std::strtoll(argv[3], nullptr, 10);
            const auto mode = argc >= 5 ? parseMode(argv[4]) : IdentityMode::Strict;
            const auto query = database.loadFace(faceId);
            if (!query.has_value()) {
                std::cerr << "Face id not found: " << faceId << "\n";
                return 1;
            }
            const auto result = identifyPerson(database.loadIdentityProfiles(), query->embedding, mode, 5);
            printIdentityResult(result);
            return 0;
        }

        if (command == "image-search") {
#ifdef FSC_ENABLE_ONNX
            if (argc < 5) {
                printUsage();
                return 2;
            }
            const int topK = argc >= 6 ? std::atoi(argv[5]) : 10;
            const float threshold = argc >= 7 ? std::stof(argv[6]) : 0.55f;
            const auto mode = argc >= 8 ? parseMode(argv[7]) : IdentityMode::Strict;
            const auto models = fsc::vision::InsightFaceModelPaths::fromBuffaloL(argv[3]);
            const auto image = fsc::vision::loadPpmRgb(argv[4]);
            fsc::vision::InsightFaceEngine engine(models, fsc::vision::RuntimeMode::Cpu);
            const auto faces = engine.analyze(image, threshold, 10);
            const auto records = database.loadFaces(false);
            const auto profiles = database.loadIdentityProfiles();

            std::cout << "faces=" << faces.size() << "\n";
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto& face = faces[i];
                double embeddingNorm = 0.0;
                for (const float value : face.embedding) {
                    embeddingNorm += static_cast<double>(value) * static_cast<double>(value);
                }
                std::cout
                    << "query_face=" << i
                    << "\tscore=" << std::fixed << std::setprecision(4) << face.detection.score
                    << "\tbox=[" << face.detection.box.x1 << "," << face.detection.box.y1 << ","
                    << face.detection.box.x2 << "," << face.detection.box.y2 << "]"
                    << "\tembedding_dim=" << face.embedding.size()
                    << "\tembedding_norm=" << std::sqrt(embeddingNorm) << "\n";

                std::cout << "identity:\n";
                printIdentityResult(identifyPerson(profiles, face.embedding, mode, 5));
                std::cout << "hits:\n";
                printSearchHits(searchFaces(records, face.embedding, topK, -1.0, false));
            }
            return 0;
#else
            std::cerr << "image-search requires FSC_ENABLE_ONNX=ON.\n";
            return 1;
#endif
        }

        printUsage();
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
