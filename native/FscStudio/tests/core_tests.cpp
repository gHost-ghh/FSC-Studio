#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Database.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/core/VectorMath.hpp"
#include "fsc/mesh/FaceMesh.hpp"
#include "fsc/vision/FaceGeometry.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

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
        database.setFaceTags(faceId, "alpha, beta");
        database.setFaceTags(duplicateFaceId, "alpha");

        const auto stats = database.statistics();
        assert(stats.faceCount == 2);
        assert(stats.tagCount == 2);
        assert(stats.duplicateImageGroupCount == 1);
        const auto tags = database.loadTagSummaries();
        assert(tags.size() == 2);
        assert(tags.front().name == "alpha" && tags.front().faceCount == 2);

        const auto sourceId = database.upsertPerson("Source", "source notes");
        const auto targetId = database.upsertPerson("Target", "target notes");
        database.assignFaceToPerson(faceId, sourceId);
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
    std::filesystem::remove(path.string() + "-wal");
    std::filesystem::remove(path.string() + "-shm");
}

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
    std::cout << "fsc_core_tests passed\n";
    return 0;
}
