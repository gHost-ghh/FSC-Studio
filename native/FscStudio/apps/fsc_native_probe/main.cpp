#include "fsc/core/Database.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace fsc::core;

namespace {

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  fsc_native_probe <database.fscdb> stats\n"
        << "  fsc_native_probe <database.fscdb> search <face_id> [top_k]\n"
        << "  fsc_native_probe <database.fscdb> identify <face_id> [strict|balanced|broad]\n";
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
            for (const auto& hit : hits) {
                if (hit.record.id == faceId) {
                    continue;
                }
                std::cout
                    << hit.record.id << "\t"
                    << std::fixed << std::setprecision(4) << hit.cosine << "\t"
                    << std::setprecision(1) << hit.similarityPercent() << "%\t"
                    << hit.record.fileName << "\t"
                    << hit.record.personName << "\n";
            }
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
            return 0;
        }

        printUsage();
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
