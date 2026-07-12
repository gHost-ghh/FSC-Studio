#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/PathEncoding.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/core/VectorMath.hpp"
#include "fsc/mesh/FaceMesh.hpp"
#include "fsc/vision/FaceGeometry.hpp"
#include "fsc/vision/ModelPaths.hpp"
#ifdef FSC_ENABLE_ONNX
#include "fsc/legacy/LegacyDtb.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

using namespace fsc::core;

namespace {

void vectorMathNormalizes() {
    const std::vector<float> value{3.0f, 4.0f};
    const auto normalized = normalize(value);
    assert(std::abs(norm(normalized) - 1.0) < 1e-6);
}

void searchOrdersByCosine() {
    FaceRecord a;
    a.id = 1;
    a.embedding = normalize(std::vector<float>{1.0f, 0.0f});
    FaceRecord b;
    b.id = 2;
    b.embedding = normalize(std::vector<float>{0.7f, 0.7f});
    FaceRecord c;
    c.id = 3;
    c.embedding = normalize(std::vector<float>{-1.0f, 0.0f});
    const auto hits = searchFaces({c, b, a}, std::vector<float>{1.0f, 0.0f}, 3);
    assert(hits.size() == 3);
    assert(hits[0].record.id == 1);
    assert(hits[1].record.id == 2);
    assert(hits[2].record.id == 3);
}

void identityWeakProfilesRequireReview() {
    IdentityProfile profile;
    profile.personId = 7;
    profile.personName = "Example";
    profile.sampleCount = 2;
    profile.prototypeCount = 1;
    profile.embeddingDim = 2;
    profile.centroid = normalize(std::vector<float>{1.0f, 0.0f});
    profile.exemplars = {profile.centroid};
    profile.exemplarWeights = {1.0f};
    profile.status = "weak";
    profile.strict = {0.70, 0.60, 0.03};

    const auto result = identifyPerson({profile}, std::vector<float>{1.0f, 0.0f});
    assert(result.decision == "review");
    assert(!result.candidates.empty());
    assert(result.candidates.front().profile.personName == "Example");
}

void visionSimilarityTransformMapsReferencePoints() {
    const auto destination = fsc::vision::arcFace112ReferencePoints();
    auto source = destination;
    for (auto& point : source) {
        point.x = point.x * 1.25f + 12.0f;
        point.y = point.y * 1.25f - 7.0f;
    }

    const auto transform = fsc::vision::estimateSimilarityTransform(source, destination);
    for (size_t i = 0; i < source.size(); ++i) {
        const auto mapped = fsc::vision::applyTransform(transform, source[i]);
        assert(std::abs(mapped.x - destination[i].x) < 0.01f);
        assert(std::abs(mapped.y - destination[i].y) < 0.01f);
    }
}

void visionNmsKeepsBestBoxes() {
    fsc::vision::Detection first;
    first.box = {0, 0, 100, 100};
    first.score = 0.95f;
    fsc::vision::Detection overlap;
    overlap.box = {5, 5, 105, 105};
    overlap.score = 0.90f;
    fsc::vision::Detection separate;
    separate.box = {220, 220, 260, 260};
    separate.score = 0.75f;

    const auto keep = fsc::vision::nonMaximumSuppression({overlap, separate, first}, 0.40f, 10);
    assert(keep.size() == 2);
    assert(std::abs(keep[0].score - 0.95f) < 1e-6f);
    assert(std::abs(keep[1].score - 0.75f) < 1e-6f);
}

void modelPathResolutionUsesBuffaloRoot() {
    const auto paths = fsc::vision::InsightFaceModelPaths::fromBuffaloL("model/insightface/models");
    assert(paths.rootDirectory.filename() == "buffalo_l");
    assert(paths.detectionModelPath.filename() == "det_10g.onnx");
    assert(fsc::vision::parseRuntimeMode("dml") == fsc::vision::RuntimeMode::DirectMl);
}

void mediaPipeMeshValidationRejectsSyntheticFallbacks() {
    std::vector<std::vector<double>> mesh(fsc::mesh::kMediaPipeFaceMeshPointCount, {10.0, 20.0, -3.0});
    assert(fsc::mesh::isMediaPipeFaceMesh(mesh));
    mesh.push_back({1.0, 2.0, 3.0});
    assert(!fsc::mesh::isMediaPipeFaceMesh(mesh));
    mesh.resize(fsc::mesh::kMediaPipeFaceMeshPointCount);
    mesh[0][2] = std::numeric_limits<double>::infinity();
    assert(!fsc::mesh::isMediaPipeFaceMesh(mesh));
}

void databasePersonActionsRoundTrip() {
    const auto path = std::filesystem::temp_directory_path() /
        ("fsc_core_people_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".fscdb");
    Database::createEmpty(path, true);
    {
        Database database(path);

        FaceInsertRecord record;
        record.fileName = "person-test.jpg";
        record.sourcePath = "person-test.jpg";
        record.embedding = normalize(std::vector<float>{1.0f, 0.0f, 0.0f});
        record.embeddingDim = static_cast<int>(record.embedding.size());
        record.bbox = {1.0, 2.0, 12.0, 22.0};
        record.detectionScore = 0.9;
        record.qualityScore = 0.8;
        record.imageHash = "person-action-smoke";
        const auto faceId = database.insertFace(record);
        record.fileName = "person-test-duplicate.jpg";
        record.sourcePath = "person-test-duplicate.jpg";
        const auto duplicateFaceId = database.insertFace(record);
        auto batchA = record;
        batchA.fileName = "batch-a.jpg";
        batchA.sourcePath = "batch-a.jpg";
        batchA.imageHash = "batch-a";
        auto batchB = batchA;
        batchB.fileName = "batch-b.jpg";
        batchB.sourcePath = "batch-b.jpg";
        batchB.imageHash = "batch-b";
        const auto batchIds = database.insertFaces({batchA, batchB});
        assert(batchIds.size() == 2);
        assert(batchIds[0] > duplicateFaceId && batchIds[1] > batchIds[0]);
        database.setFaceTags(faceId, "alpha, beta");
        database.setFaceTags(duplicateFaceId, "alpha");

        const auto stats = database.statistics();
        assert(stats.faceCount == 4);
        assert(stats.tagCount == 2);
        assert(stats.duplicateImageGroupCount == 1);
        assert(std::abs(stats.averageQuality - 0.8) < 1e-9);
        assert(std::abs(stats.minimumQuality - 0.8) < 1e-9);
        assert(std::abs(stats.maximumQuality - 0.8) < 1e-9);
        const auto tags = database.loadTagSummaries();
        assert(tags.size() == 2);
        assert(tags.front().name == "alpha" && tags.front().faceCount == 2);

        const auto sourceId = database.upsertPerson("Source", "source notes");
        const auto targetId = database.upsertPerson("Target", "target notes");
        database.assignFaceToPerson(faceId, sourceId);
        database.updateFaceReview(batchIds.front(), "open", true, "keep this note");
        const int batchAssigned = database.assignFacesToPerson(
            {batchIds.front(), batchIds.back(), batchIds.front()},
            "Batch Person",
            "batch-tag, alpha",
            true);
        assert(batchAssigned == 2);
        const auto peopleAfterBatchAssignment = database.loadPeople();
        const auto batchPerson = std::find_if(peopleAfterBatchAssignment.begin(), peopleAfterBatchAssignment.end(), [](const auto& person) {
            return person.name == "Batch Person";
        });
        assert(batchPerson != peopleAfterBatchAssignment.end() && batchPerson->faceCount == 2);
        const auto refreshedBatchFace = database.loadFace(batchIds.front());
        assert(refreshedBatchFace.has_value());
        assert(refreshedBatchFace->personName == "Batch Person");
        assert(refreshedBatchFace->reviewState == "reviewed");
        assert(!refreshedBatchFace->ignored);
        assert(refreshedBatchFace->notes == "keep this note");
        assert(refreshedBatchFace->tagText.find("batch-tag") != std::string::npos);
        database.renamePerson(sourceId, "Renamed", "renamed notes");

        const auto people = database.loadPeople();
        const auto renamed = std::find_if(people.begin(), people.end(), [sourceId](const auto& person) {
            return person.id == sourceId && person.name == "Renamed" && person.notes == "renamed notes" &&
                person.faceCount == 1 && person.representativeFaceId > 0;
        });
        assert(renamed != people.end());

        const int moved = database.mergePeople(sourceId, targetId);
        assert(moved == 1);
        auto members = database.loadFacesForPerson(targetId, true);
        assert(members.size() == 1);
        assert(members.front().id == faceId);

        const int cleared = database.clearPersonAssignment(targetId, true);
        assert(cleared == 1);
        const auto face = database.loadFace(faceId);
        assert(face.has_value());
        assert(face->personId == 0);
        const auto preview = database.loadFacePreview(faceId);
        assert(preview.has_value());
        assert(preview->sourcePath == "person-test.jpg");
        assert(preview->bbox.size() == 4);
        assert(preview->embedding.empty());
    }
    std::filesystem::remove(path);
    std::filesystem::remove(pathWithSuffix(path, "-wal"));
    std::filesystem::remove(pathWithSuffix(path, "-shm"));
}

void databaseUnicodePathsRoundTrip() {
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() /
        pathFromUtf8("fsc_\xE4\xBA\xBA\xE8\x84\xB8_\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E_\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4_" + suffix);
    const auto databasePath = root /
        pathFromUtf8("\xE6\x95\xB0\xE6\x8D\xAE\xE5\xBA\x93_\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E_\xEC\x96\xBC\xEA\xB5\xB4.fscdb");
    const auto backupPath = root /
        pathFromUtf8("\xE5\xA4\x87\xE4\xBB\xBD_\xE4\xBA\xBA\xE8\x84\xB8_\xEB\xB0\xB1\xEC\x97\x85.fscdb");
    const auto imagePath = root /
        pathFromUtf8("\xE7\x85\xA7\xE7\x89\x87_\xE9\xA1\x94_\xEC\x96\xBC\xEA\xB5\xB4.jpg");

    assert(pathFromUtf8(pathToUtf8(databasePath)) == databasePath);
    std::filesystem::create_directories(root);
    {
        std::ofstream image(imagePath, std::ios::binary);
        image << "unicode-path-hash-smoke";
    }
    assert(!sha256File(imagePath).empty());
    Database::createEmpty(databasePath, true);
    {
        Database database(databasePath);
        FaceInsertRecord record;
        record.fileName = pathToUtf8(imagePath.filename());
        record.sourcePath = pathToUtf8(imagePath);
        record.embedding = normalize(std::vector<float>{0.0f, 1.0f, 0.0f});
        record.embeddingDim = static_cast<int>(record.embedding.size());
        record.bbox = {2.0, 3.0, 20.0, 24.0};
        record.detectionScore = 0.91;
        record.qualityScore = 0.82;
        record.imageHash = "unicode-path-smoke";
        const auto faceId = database.insertFace(record);
        const auto personId = database.upsertPerson("\xE6\x9D\x8E\xE9\x9B\xB7_\xE5\xB1\xB1\xE7\x94\xB0_\xEA\xB9\x80");
        database.assignFaceToPerson(faceId, personId);
        const auto result = database.backupTo(backupPath);
        assert(result.ok);
        assert(result.outputPath == pathToUtf8(std::filesystem::absolute(backupPath)));
    }
    {
        Database backup(backupPath);
        const auto faces = backup.loadFaces(true);
        assert(faces.size() == 1);
        assert(faces.front().fileName == pathToUtf8(imagePath.filename()));
        assert(faces.front().sourcePath == pathToUtf8(imagePath));
        assert(!faces.front().personName.empty());
        assert(backup.checkIntegrity().ok);
    }
    std::filesystem::remove_all(root);
}

#ifdef FSC_ENABLE_ONNX
void appendByte(std::vector<std::uint8_t>& output, std::uint8_t value) {
    output.push_back(value);
}

void appendShortUnicode(std::vector<std::uint8_t>& output, std::string_view text) {
    assert(text.size() <= 255);
    appendByte(output, 0x8c);
    appendByte(output, static_cast<std::uint8_t>(text.size()));
    output.insert(output.end(), text.begin(), text.end());
}

void appendShortBytes(std::vector<std::uint8_t>& output, const std::vector<std::uint8_t>& bytes) {
    assert(bytes.size() <= 255);
    appendByte(output, 'C');
    appendByte(output, static_cast<std::uint8_t>(bytes.size()));
    output.insert(output.end(), bytes.begin(), bytes.end());
}

void appendGlobal(std::vector<std::uint8_t>& output, std::string_view module, std::string_view name) {
    appendByte(output, 'c');
    output.insert(output.end(), module.begin(), module.end());
    appendByte(output, '\n');
    output.insert(output.end(), name.begin(), name.end());
    appendByte(output, '\n');
}

void appendBinInt(std::vector<std::uint8_t>& output, int32_t value) {
    appendByte(output, 'J');
    const auto raw = static_cast<uint32_t>(value);
    for (int offset = 0; offset < 4; ++offset) {
        appendByte(output, static_cast<std::uint8_t>((raw >> (offset * 8U)) & 0xffU));
    }
}

void appendBinFloat(std::vector<std::uint8_t>& output, double value) {
    appendByte(output, 'G');
    const auto raw = std::bit_cast<uint64_t>(value);
    for (int offset = 7; offset >= 0; --offset) {
        appendByte(output, static_cast<std::uint8_t>((raw >> (offset * 8U)) & 0xffU));
    }
}

std::vector<std::uint8_t> trustedLegacyDtbFixture() {
    std::vector<std::uint8_t> output;
    appendByte(output, 0x80); appendByte(output, 4); // PROTO 4
    appendByte(output, ']');                         // root list
    appendByte(output, '(');                         // row tuple mark
    appendByte(output, 'N');                         // legacy 68-point placeholder
    appendGlobal(output, "_dlib_pybind11", "vector");
    appendByte(output, ')'); appendByte(output, 0x81); // EMPTY_TUPLE, NEWOBJ
    appendShortBytes(output, {0x81, 0x01, 0x07});
    appendByte(output, 0x85); appendByte(output, 'b'); // TUPLE1, BUILD
    appendBinFloat(output, 1.0);

    appendGlobal(output, "numpy.core.multiarray", "_reconstruct");
    appendGlobal(output, "numpy", "ndarray");
    appendByte(output, 'K'); appendByte(output, 0);
    appendByte(output, 0x85);
    appendShortBytes(output, {'b'});
    appendByte(output, 0x87); appendByte(output, 'R');

    appendByte(output, '(');                         // ndarray state mark
    appendByte(output, 'K'); appendByte(output, 1);   // version
    appendByte(output, 'K'); appendByte(output, 2);
    appendByte(output, 'K'); appendByte(output, 3);
    appendByte(output, 'K'); appendByte(output, 3);
    appendByte(output, 0x87);                         // shape tuple
    appendGlobal(output, "numpy", "dtype");
    appendShortUnicode(output, "u1");
    appendByte(output, 0x89); appendByte(output, 0x88); appendByte(output, 0x87); appendByte(output, 'R');
    appendByte(output, '(');                          // dtype state mark
    appendByte(output, 'K'); appendByte(output, 3);
    appendShortUnicode(output, "|");
    appendByte(output, 'N'); appendByte(output, 'N'); appendByte(output, 'N');
    appendBinInt(output, -1); appendBinInt(output, -1);
    appendByte(output, 'K'); appendByte(output, 0);
    appendByte(output, 't'); appendByte(output, 'b'); // dtype BUILD
    appendByte(output, 0x89);                         // C order
    std::vector<std::uint8_t> image(18);
    for (std::uint8_t index = 0; index < image.size(); ++index) image[index] = index;
    appendShortBytes(output, image);
    appendByte(output, 't'); appendByte(output, 'b'); // ndarray BUILD
    appendShortUnicode(output, "legacy_face.jpg");
    appendByte(output, 't'); appendByte(output, 'a'); // row tuple, append
    appendByte(output, '.');
    return output;
}

void legacyDtbReaderLoadsTrustedEmbeddedImage() {
    const auto path = std::filesystem::temp_directory_path() /
        ("fsc_legacy_fixture_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dtb");
    const auto payload = trustedLegacyDtbFixture();
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    const auto rows = fsc::legacy::loadLegacyDtbImages(path);
    assert(rows.size() == 1);
    assert(rows.front().fileName == "legacy_face.jpg");
    assert(rows.front().image.width == 3);
    assert(rows.front().image.height == 2);
    assert(rows.front().image.pixels.size() == 18);
    assert(rows.front().image.pixels.front() == 0);
    assert(rows.front().image.pixels.back() == 17);
    std::filesystem::remove(path);
}
#endif

} // namespace

int main() {
    vectorMathNormalizes();
    searchOrdersByCosine();
    identityWeakProfilesRequireReview();
    visionSimilarityTransformMapsReferencePoints();
    visionNmsKeepsBestBoxes();
    modelPathResolutionUsesBuffaloRoot();
    mediaPipeMeshValidationRejectsSyntheticFallbacks();
    databasePersonActionsRoundTrip();
    databaseUnicodePathsRoundTrip();
#ifdef FSC_ENABLE_ONNX
    legacyDtbReaderLoadsTrustedEmbeddedImage();
#endif
    std::cout << "fsc_core_tests passed\n";
    return 0;
}
