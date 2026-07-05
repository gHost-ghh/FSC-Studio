#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/core/VectorMath.hpp"

#include <cassert>
#include <cmath>
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

} // namespace

int main() {
    vectorMathNormalizes();
    searchOrdersByCosine();
    identityWeakProfilesRequireReview();
    std::cout << "fsc_core_tests passed\n";
    return 0;
}
