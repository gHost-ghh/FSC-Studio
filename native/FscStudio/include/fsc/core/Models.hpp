#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fsc::core {

struct DatabaseStatistics {
    int64_t faceCount = 0;
    int64_t peopleCount = 0;
    int64_t reviewCount = 0;
    double averageQuality = 0.0;
    std::string formatVersion;
    std::string metric;
    std::string modelName;
};

struct FaceRecord {
    int64_t id = 0;
    std::string fileName;
    std::string sourcePath;
    std::vector<float> embedding;
    int embeddingDim = 0;
    double detectionScore = 0.0;
    double qualityScore = 0.0;
    int64_t personId = 0;
    std::string personName;
    bool ignored = false;
    std::string reviewState;
    std::string notes;
    std::string createdAt;
};

struct PersonSummary {
    int64_t id = 0;
    std::string name;
    int64_t faceCount = 0;
    int64_t ignoredCount = 0;
    int64_t reviewCount = 0;
    double averageQuality = 0.0;
    std::string identityStatus;
    int identitySampleCount = 0;
    int identityExemplarCount = 0;
    double identityAcceptThreshold = 0.0;
    std::string identityHealth;
};

struct SearchHit {
    FaceRecord record;
    double cosine = 0.0;

    [[nodiscard]] double similarityPercent() const {
        return (cosine + 1.0) * 50.0;
    }
};

enum class IdentityMode {
    Strict,
    Balanced,
    Broad
};

struct IdentityThresholds {
    double accept = 0.72;
    double review = 0.62;
    double margin = 0.035;
};

struct IdentityProfile {
    int64_t personId = 0;
    std::string personName;
    int sampleCount = 0;
    int prototypeCount = 0;
    int embeddingDim = 0;
    std::vector<float> centroid;
    std::vector<std::vector<float>> prototypes;
    std::vector<std::vector<float>> exemplars;
    std::vector<float> exemplarWeights;
    std::vector<int64_t> exemplarFaceIds;
    std::vector<int64_t> hardNegativeFaceIds;
    std::vector<std::vector<float>> hardNegativeEmbeddings;
    IdentityThresholds strict;
    IdentityThresholds balanced;
    IdentityThresholds broad;
    double acceptThreshold = 0.0;
    double reviewThreshold = 0.0;
    double meanSimilarity = 0.0;
    double minSimilarity = 0.0;
    double maxSimilarity = 0.0;
    double qualityMean = 0.0;
    std::vector<int64_t> evidenceFaceIds;
    std::string status;
    std::string health;
    std::string strategyVersion;
    std::string scoringModelVersion;
    std::string updatedAt;
};

struct IdentityCandidate {
    IdentityProfile profile;
    double score = 0.0;
    double margin = 0.0;
    double confidence = 0.0;
    int bestExemplarIndex = -1;
    int64_t evidenceFaceId = 0;
};

struct IdentityResult {
    std::string decision = "unknown";
    std::vector<IdentityCandidate> candidates;
    std::string message;
};

} // namespace fsc::core
