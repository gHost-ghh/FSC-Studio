#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/PathEncoding.hpp"
#include "fsc/core/Search.hpp"
#ifdef FSC_ENABLE_ONNX
#include "fsc/legacy/LegacyDtb.hpp"
#endif
#include "fsc/mesh/FaceMesh.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>

using namespace fsc::core;

namespace {

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  fsc_native_probe <database.fscdb> create-db [replace|no-replace]\n"
        << "  fsc_native_probe <database.fscdb> stats\n"
        << "  fsc_native_probe <database.fscdb> people\n"
        << "  fsc_native_probe <database.fscdb> add-person <name> [notes]\n"
        << "  fsc_native_probe <database.fscdb> assign-person <face_id> <person_id>\n"
        << "  fsc_native_probe <database.fscdb> update-review <face_id> <state> <ignored:0|1> [notes]\n"
        << "  fsc_native_probe <database.fscdb> search <face_id> [top_k]\n"
        << "  fsc_native_probe <database.fscdb> identify <face_id> [strict|balanced|broad]\n"
        << "  fsc_native_probe <database.fscdb> train-profiles [min_quality] [max_exemplars]\n"
        << "  fsc_native_probe <database.fscdb> build-mesh <face_id> [face_landmarks_detector.onnx]\n"
        << "  fsc_native_probe <database.fscdb> repair-invalid-meshes [face_landmarks_detector.onnx]\n"
        << "  fsc_native_probe <database.fscdb> import-image <model_root> <image_path> [threshold] [person_id]\n"
        << "  fsc_native_probe <database.fscdb> image-search <model_root> <image_path> [top_k] [threshold] [strict|balanced|broad]\n"
        << "  fsc_native_probe <output.fscdb> inspect-legacy-dtb <source.dtb>\n"
        << "  fsc_native_probe <output.fscdb> convert-legacy-dtb <source.dtb> <model_root> [auto|cpu|directml|cuda|qnn-npu|qnn-gpu] [limit]\n";
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

std::vector<std::vector<double>> keypointRows(const std::array<fsc::vision::Point2f, 5>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point2f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point3f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y, point.z});
    }
    return rows;
}

std::string qualityJson(const fsc::vision::AnalyzedFace& face) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6)
           << "{\"det_score\":" << face.detection.score
           << ",\"area_ratio\":" << face.qualityAreaRatio
           << ",\"sharpness\":" << face.qualitySharpness
           << ",\"brightness\":" << face.qualityBrightness
           << ",\"contrast\":" << face.qualityContrast
           << ",\"native\":true}";
    return stream.str();
}

FaceInsertRecord insertRecordFromFace(
    const std::filesystem::path& imagePath,
    const fsc::vision::AnalyzedFace& face,
    int64_t personId,
    const std::string& imageHash,
    bool duplicate) {
    FaceInsertRecord record;
    record.fileName = pathToUtf8(imagePath.filename());
    record.sourcePath = pathToUtf8(imagePath);
    record.embedding = face.embedding;
    record.embeddingDim = static_cast<int>(face.embedding.size());
    record.bbox = {
        face.detection.box.x1,
        face.detection.box.y1,
        face.detection.box.x2,
        face.detection.box.y2,
    };
    record.keypoints = keypointRows(face.detection.keypoints);
    record.landmarks2d = landmarkRows(face.landmarks2d);
    record.landmarks3d = landmarkRows(face.landmarks3d);
    record.detectionScore = face.detection.score;
    record.qualityScore = face.qualityScore;
    record.qualityJson = qualityJson(face);
    record.imageHash = imageHash;
    record.personId = personId;
    if (duplicate) {
        record.reviewState = "duplicate";
        record.notes = "Same source image hash already exists in this database.";
    }
    return record;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage();
        return 2;
    }

    try {
        const std::filesystem::path databasePath = argv[1];
        const std::string command = argv[2];
#ifdef FSC_ENABLE_ONNX
        if (command == "inspect-legacy-dtb") {
            if (argc < 4) {
                printUsage();
                return 2;
            }
            const auto rows = fsc::legacy::loadLegacyDtbImages(argv[3]);
            std::cout << "rows=" << rows.size() << "\n";
            for (size_t index = 0; index < rows.size(); ++index) {
                const auto& row = rows[index];
                std::cout << (index + 1) << "\t" << row.fileName << "\t"
                          << row.image.width << 'x' << row.image.height << "\n";
            }
            return 0;
        }
        if (command == "convert-legacy-dtb") {
            if (argc < 5) {
                printUsage();
                return 2;
            }
            fsc::legacy::LegacyConversionOptions options;
            options.models = fsc::vision::InsightFaceModelPaths::fromBuffaloL(argv[4]);
            options.runtimeMode = argc >= 6 ? fsc::vision::parseRuntimeMode(argv[5]) : fsc::vision::RuntimeMode::Auto;
            options.limit = argc >= 7 ? std::max(0, std::atoi(argv[6])) : 0;
            options.progress = [](const std::string& message, int current, int total) {
                std::cout << '[' << current << '/' << total << "] " << message << "\n";
            };
            const auto summary = fsc::legacy::convertLegacyDtb(argv[3], databasePath, options);
            std::cout << "output=" << pathToUtf8(summary.outputPath) << "\n";
            std::cout << "image_directory=" << pathToUtf8(summary.imageDirectory) << "\n";
            std::cout << "saved=" << summary.facesSaved << "\n";
            std::cout << "skipped=" << summary.skippedRows << "\n";
            std::cout << "total=" << summary.rowsTotal << "\n";
            return 0;
        }
#endif
        if (command == "create-db") {
            const bool replace = argc < 4 || std::string(argv[3]) != "no-replace";
            Database::createEmpty(databasePath, replace);
            Database created(databasePath);
            const auto stats = created.statistics();
            std::cout << "created=" << pathToUtf8(databasePath) << "\n";
            std::cout << "format_version=" << stats.formatVersion << "\n";
            std::cout << "metric=" << stats.metric << "\n";
            std::cout << "model=" << stats.modelName << "\n";
            return 0;
        }

        Database database(databasePath);
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

        if (command == "people") {
            const auto people = database.loadPeople();
            std::cout << "people=" << people.size() << "\n";
            for (const auto& person : people) {
                std::cout
                    << person.id << "\t"
                    << person.name << "\t"
                    << "faces=" << person.faceCount << "\t"
                    << "avg_quality=" << std::fixed << std::setprecision(4) << person.averageQuality << "\t"
                    << "identity=" << person.identityStatus << "\t"
                    << "samples=" << person.identitySampleCount << "\t"
                    << "exemplars=" << person.identityExemplarCount << "\n";
            }
            return 0;
        }

        if (command == "add-person") {
            if (argc < 4) {
                printUsage();
                return 2;
            }
            const std::string notes = argc >= 5 ? argv[4] : "";
            const auto id = database.upsertPerson(argv[3], notes);
            std::cout << "person_id=" << id << "\n";
            return 0;
        }

        if (command == "assign-person") {
            if (argc < 5) {
                printUsage();
                return 2;
            }
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto personId = std::strtoll(argv[4], nullptr, 10);
            database.assignFaceToPerson(faceId, personId);
            std::cout << "assigned_face=" << faceId << "\n";
            std::cout << "person_id=" << personId << "\n";
            return 0;
        }

        if (command == "update-review") {
            if (argc < 6) {
                printUsage();
                return 2;
            }
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const std::string state = argv[4];
            const bool ignored = std::atoi(argv[5]) != 0;
            const std::string notes = argc >= 7 ? argv[6] : "";
            database.updateFaceReview(faceId, state, ignored, notes);
            std::cout << "updated_face=" << faceId << "\n";
            std::cout << "review_state=" << state << "\n";
            std::cout << "ignored=" << (ignored ? "true" : "false") << "\n";
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

        if (command == "train-profiles") {
            IdentityTrainingOptions options;
            options.minQuality = argc >= 4 ? std::stod(argv[3]) : options.minQuality;
            options.maxExemplars = argc >= 5 ? std::atoi(argv[4]) : options.maxExemplars;
            const auto summary = database.rebuildIdentityProfiles(options);
            std::cout << "profiles_built=" << summary.profilesBuilt << "\n";
            std::cout << "weak_profiles=" << summary.weakProfiles << "\n";
            std::cout << "skipped_people=" << summary.skippedPeople << "\n";
            std::cout << "samples_used=" << summary.samplesUsed << "\n";
            for (const auto& message : summary.messages) {
                std::cout << "message=" << message << "\n";
            }
            return 0;
        }

        if (command == "build-mesh") {
            if (argc < 4) {
                printUsage();
                return 2;
            }
            const int64_t faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value()) {
                std::cerr << "Face id not found: " << faceId << "\n";
                return 1;
            }
            if (face->sourcePath.empty() || !std::filesystem::is_regular_file(face->sourcePath)) {
                throw std::runtime_error("The face source image is unavailable for MediaPipe dense-mesh analysis.");
            }
            fsc::mesh::MediaPipeFaceLandmarkerOptions options;
            if (argc >= 5) {
                options.modelAssetPath = argv[4];
            }
            fsc::mesh::MediaPipeFaceLandmarker landmarker(std::move(options));
            const auto mesh = landmarker.detect(
                fsc::vision::loadImageRgb(face->sourcePath),
                face->bbox,
                face->keypoints);
            database.updateFaceMesh3d(faceId, mesh);
            std::cout << "updated_face=" << faceId << "\n";
            std::cout << "mesh_points=" << mesh.size() << "\n";
            return 0;
        }

        if (command == "repair-invalid-meshes") {
            fsc::mesh::MediaPipeFaceLandmarkerOptions options;
            if (argc >= 4) {
                options.modelAssetPath = argv[3];
            }
            fsc::mesh::MediaPipeFaceLandmarker landmarker(std::move(options));
            int scanned = 0;
            int retained = 0;
            int repaired = 0;
            int cleared = 0;
            int unavailable = 0;
            int failed = 0;
            for (const auto& summary : database.loadFaces(true)) {
                const auto face = database.loadFace(summary.id);
                if (!face.has_value()) {
                    continue;
                }
                ++scanned;
                if (fsc::mesh::isMediaPipeFaceMesh(face->faceMesh3d)) {
                    ++retained;
                    continue;
                }
                if (face->faceMesh3d.empty()) {
                    ++retained;
                    continue;
                }
                if (!face->faceMesh3d.empty()) {
                    database.clearFaceMesh3d(face->id);
                    ++cleared;
                }
                try {
                    if (face->sourcePath.empty() || !std::filesystem::is_regular_file(face->sourcePath)) {
                        throw std::runtime_error("source image is unavailable");
                    }
                    const auto mesh = landmarker.detect(
                        fsc::vision::loadImageRgb(face->sourcePath),
                        face->bbox,
                        face->keypoints);
                    database.updateFaceMesh3d(face->id, mesh);
                    ++repaired;
                } catch (const std::exception& ex) {
                    const std::string message = ex.what();
                    if (message.starts_with("MediaPipe did not detect a face mesh")) {
                        ++unavailable;
                        std::cerr << "face=" << face->id << " cleared: " << message << "\n";
                    } else {
                        ++failed;
                        std::cerr << "face=" << face->id << " failed: " << message << "\n";
                    }
                }
            }
            std::cout << "scanned=" << scanned << "\n";
            std::cout << "retained=" << retained << "\n";
            std::cout << "repaired=" << repaired << "\n";
            std::cout << "cleared=" << cleared << "\n";
            std::cout << "unavailable=" << unavailable << "\n";
            std::cout << "failed=" << failed << "\n";
            return failed == 0 ? 0 : 1;
        }

        if (command == "import-image") {
#ifdef FSC_ENABLE_ONNX
            if (argc < 5) {
                printUsage();
                return 2;
            }
            const float threshold = argc >= 6 ? std::stof(argv[5]) : 0.55f;
            const int64_t personId = argc >= 7 ? std::strtoll(argv[6], nullptr, 10) : 0;
            const auto models = fsc::vision::InsightFaceModelPaths::fromBuffaloL(argv[3]);
            const std::filesystem::path imagePath = argv[4];
            const auto imageHash = sha256File(imagePath);
            const bool duplicate = database.imageHashExists(imageHash);
            const auto image = fsc::vision::loadImageRgb(imagePath);
            fsc::vision::InsightFaceEngine engine(models, fsc::vision::RuntimeMode::Cpu);
            const auto faces = engine.analyze(image, threshold, 10);
            std::cout << "faces=" << faces.size() << "\n";
            std::cout << "image_hash=" << imageHash << "\n";
            std::cout << "duplicate=" << (duplicate ? "true" : "false") << "\n";
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto id = database.insertFace(insertRecordFromFace(imagePath, faces[i], personId, imageHash, duplicate));
                std::cout
                    << "inserted=" << id
                    << "\tquery_face=" << i
                    << "\tscore=" << std::fixed << std::setprecision(4) << faces[i].detection.score
                    << "\tquality=" << faces[i].qualityScore
                    << "\tlandmarks2d=" << faces[i].landmarks2d.size()
                    << "\tlandmarks3d=" << faces[i].landmarks3d.size() << "\n";
            }
            return 0;
#else
            std::cerr << "import-image requires FSC_ENABLE_ONNX=ON.\n";
            return 1;
#endif
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
            const auto image = fsc::vision::loadImageRgb(argv[4]);
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
                    << "\tembedding_norm=" << std::sqrt(embeddingNorm)
                    << "\tlandmarks2d=" << face.landmarks2d.size()
                    << "\tlandmarks3d=" << face.landmarks3d.size()
                    << "\tquality=" << face.qualityScore << "\n";

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
