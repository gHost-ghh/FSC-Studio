#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fsc::core {

struct DatabaseStatistics {
    int64_t faceCount = 0;
    int64_t peopleCount = 0;
    int64_t tagCount = 0;
    int64_t reviewCount = 0;
    int64_t ignoredCount = 0;
    int64_t duplicateImageGroupCount = 0;
    double averageQuality = 0.0;
    double minimumQuality = 0.0;
    double maximumQuality = 0.0;
    std::string formatVersion;
    std::string metric;
    std::string modelName;
};

struct MaintenanceResult {
    std::string action;
    bool ok = false;
    std::string message;
    std::string outputPath;
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
    std::string tagText;
    bool ignored = false;
    std::string reviewState;
    std::string notes;
    std::string createdAt;
    int duplicateCount = 0;
    std::vector<double> bbox;
    std::vector<std::vector<double>> landmarks2d;
    std::vector<std::vector<double>> landmarks3d;
    std::vector<std::vector<double>> faceMesh3d;
};

struct FaceInsertRecord {
    std::string fileName;
    std::string sourcePath;
    std::vector<float> embedding;
    int embeddingDim = 0;
    std::vector<double> bbox;
    std::vector<std::vector<double>> keypoints;
    std::vector<std::vector<double>> landmarks2d;
    std::vector<std::vector<double>> landmarks3d;
    double detectionScore = 0.0;
    double qualityScore = 0.0;
    std::string qualityJson;
    std::string imageHash;
    int64_t personId = 0;
    std::string reviewState = "open";
    std::string notes;
};

struct PersonSummary {
    int64_t id = 0;
    std::string name;
    std::string notes;
    int64_t faceCount = 0;
    int64_t ignoredCount = 0;
    int64_t reviewCount = 0;
    double averageQuality = 0.0;
    int64_t representativeFaceId = 0;
    std::string identityStatus;
    int identitySampleCount = 0;
    int identityExemplarCount = 0;
    double identityAcceptThreshold = 0.0;
    std::string identityScoringModelVersion;
    std::string identityHealth;
};

struct TagSummary {
    std::string name;
    int64_t faceCount = 0;
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

struct IdentityTrainingOptions {
    double minQuality = 0.35;
    int maxExemplars = 12;
    std::vector<int64_t> personIds;
};

struct IdentityTrainingSummary {
    int profilesBuilt = 0;
    int weakProfiles = 0;
    int skippedPeople = 0;
    int samplesUsed = 0;
    std::vector<std::string> messages;
};

} // namespace fsc::core
