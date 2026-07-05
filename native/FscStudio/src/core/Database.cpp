#include "fsc/core/Database.hpp"

#include "fsc/core/VectorMath.hpp"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fsc::core {
namespace {

class Statement {
public:
    Statement(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            throw std::runtime_error(sqlite3_errmsg(db));
        }
    }

    ~Statement() {
        sqlite3_finalize(stmt_);
    }

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    [[nodiscard]] sqlite3_stmt* get() const noexcept { return stmt_; }

private:
    sqlite3_stmt* stmt_ = nullptr;
};

std::string textColumn(sqlite3_stmt* stmt, int column) {
    const auto* text = sqlite3_column_text(stmt, column);
    return text == nullptr ? std::string{} : reinterpret_cast<const char*>(text);
}

std::vector<float> floatBlob(sqlite3_stmt* stmt, int column, int expectedDim = 0) {
    const auto* blob = sqlite3_column_blob(stmt, column);
    const int bytes = sqlite3_column_bytes(stmt, column);
    if (blob == nullptr || bytes <= 0 || bytes % static_cast<int>(sizeof(float)) != 0) {
        return {};
    }
    const int count = bytes / static_cast<int>(sizeof(float));
    if (expectedDim > 0 && count != expectedDim && count % expectedDim != 0) {
        return {};
    }
    std::vector<float> values(static_cast<size_t>(count));
    std::memcpy(values.data(), blob, static_cast<size_t>(bytes));
    return values;
}

std::vector<int64_t> intListFromJson(const std::string& value) {
    std::vector<int64_t> output;
    if (value.empty()) {
        return output;
    }
    try {
        const auto json = nlohmann::json::parse(value);
        if (!json.is_array()) {
            return output;
        }
        for (const auto& item : json) {
            output.push_back(item.get<int64_t>());
        }
    } catch (...) {
        output.clear();
    }
    return output;
}

IdentityThresholds thresholdsFor(const nlohmann::json& root, const std::string& mode, IdentityThresholds defaults) {
    if (!root.is_object() || !root.contains(mode) || !root.at(mode).is_object()) {
        return defaults;
    }
    const auto& item = root.at(mode);
    defaults.accept = item.value("accept", defaults.accept);
    defaults.review = item.value("review", defaults.review);
    defaults.margin = item.value("margin", defaults.margin);
    return defaults;
}

std::vector<std::vector<float>> rowsFromFlat(std::vector<float> flat, int rowCount, int dim) {
    std::vector<std::vector<float>> rows;
    if (rowCount <= 0 || dim <= 0 || flat.size() != static_cast<size_t>(rowCount * dim)) {
        return rows;
    }
    rows.reserve(static_cast<size_t>(rowCount));
    for (int row = 0; row < rowCount; ++row) {
        const auto begin = flat.begin() + row * dim;
        rows.emplace_back(begin, begin + dim);
        rows.back() = normalize(rows.back());
    }
    return rows;
}

} // namespace

Database::Database(std::filesystem::path path) : path_(std::move(path)) {
    const int code = sqlite3_open_v2(path_.string().c_str(), &db_, SQLITE_OPEN_READWRITE, nullptr);
    if (code != SQLITE_OK) {
        const std::string message = db_ == nullptr ? "Failed to open SQLite database." : sqlite3_errmsg(db_);
        if (db_ != nullptr) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error(message);
    }
}

Database::~Database() {
    if (db_ != nullptr) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

Database::Database(Database&& other) noexcept : path_(std::move(other.path_)), db_(other.db_) {
    other.db_ = nullptr;
}

Database& Database::operator=(Database&& other) noexcept {
    if (this != &other) {
        if (db_ != nullptr) {
            sqlite3_close(db_);
        }
        path_ = std::move(other.path_);
        db_ = other.db_;
        other.db_ = nullptr;
    }
    return *this;
}

std::map<std::string, std::string> Database::metadata() const {
    Statement statement(db_, "SELECT key, value FROM metadata ORDER BY key");
    std::map<std::string, std::string> output;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        output[textColumn(statement.get(), 0)] = textColumn(statement.get(), 1);
    }
    return output;
}

DatabaseStatistics Database::statistics() const {
    auto meta = metadata();
    DatabaseStatistics stats;
    stats.formatVersion = meta["format_version"];
    stats.metric = meta["metric"];
    stats.modelName = meta["model_name"];

    {
        Statement statement(db_, "SELECT COUNT(*) FROM faces");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.faceCount = sqlite3_column_int64(statement.get(), 0);
        }
    }
    {
        Statement statement(db_, "SELECT COUNT(*) FROM persons");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.peopleCount = sqlite3_column_int64(statement.get(), 0);
        }
    }
    {
        Statement statement(db_, "SELECT COUNT(*) FROM faces WHERE ignored = 0 AND review_state != 'reviewed'");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.reviewCount = sqlite3_column_int64(statement.get(), 0);
        }
    }
    {
        Statement statement(db_, "SELECT COALESCE(AVG(quality_score), 0.0) FROM faces");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.averageQuality = sqlite3_column_double(statement.get(), 0);
        }
    }
    return stats;
}

std::vector<FaceRecord> Database::loadFaces(bool includeIgnored, int limit) const {
    const char* sql =
        "SELECT f.id, f.file_name, COALESCE(f.source_path, ''), f.embedding_blob, f.embedding_dim, "
        "COALESCE(f.det_score, 0), COALESCE(f.quality_score, 0), COALESCE(f.person_id, 0), "
        "COALESCE(p.name, ''), f.ignored, COALESCE(f.review_state, 'open'), COALESCE(f.notes, ''), "
        "COALESCE(f.created_at, '') "
        "FROM faces f LEFT JOIN persons p ON p.id = f.person_id "
        "WHERE (?1 != 0 OR f.ignored = 0) "
        "ORDER BY f.id DESC "
        "LIMIT CASE WHEN ?2 > 0 THEN ?2 ELSE -1 END";
    Statement statement(db_, sql);
    sqlite3_bind_int(statement.get(), 1, includeIgnored ? 1 : 0);
    sqlite3_bind_int(statement.get(), 2, limit);

    std::vector<FaceRecord> records;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        FaceRecord record;
        record.id = sqlite3_column_int64(statement.get(), 0);
        record.fileName = textColumn(statement.get(), 1);
        record.sourcePath = textColumn(statement.get(), 2);
        record.embeddingDim = sqlite3_column_int(statement.get(), 4);
        record.embedding = normalize(floatBlob(statement.get(), 3, record.embeddingDim));
        record.detectionScore = sqlite3_column_double(statement.get(), 5);
        record.qualityScore = sqlite3_column_double(statement.get(), 6);
        record.personId = sqlite3_column_int64(statement.get(), 7);
        record.personName = textColumn(statement.get(), 8);
        record.ignored = sqlite3_column_int(statement.get(), 9) != 0;
        record.reviewState = textColumn(statement.get(), 10);
        record.notes = textColumn(statement.get(), 11);
        record.createdAt = textColumn(statement.get(), 12);
        if (!record.embedding.empty()) {
            records.push_back(std::move(record));
        }
    }
    return records;
}

std::optional<FaceRecord> Database::loadFace(int64_t faceId) const {
    const char* sql =
        "SELECT f.id, f.file_name, COALESCE(f.source_path, ''), f.embedding_blob, f.embedding_dim, "
        "COALESCE(f.det_score, 0), COALESCE(f.quality_score, 0), COALESCE(f.person_id, 0), "
        "COALESCE(p.name, ''), f.ignored, COALESCE(f.review_state, 'open'), COALESCE(f.notes, ''), "
        "COALESCE(f.created_at, '') "
        "FROM faces f LEFT JOIN persons p ON p.id = f.person_id WHERE f.id = ?1";
    Statement statement(db_, sql);
    sqlite3_bind_int64(statement.get(), 1, faceId);
    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    FaceRecord record;
    record.id = sqlite3_column_int64(statement.get(), 0);
    record.fileName = textColumn(statement.get(), 1);
    record.sourcePath = textColumn(statement.get(), 2);
    record.embeddingDim = sqlite3_column_int(statement.get(), 4);
    record.embedding = normalize(floatBlob(statement.get(), 3, record.embeddingDim));
    record.detectionScore = sqlite3_column_double(statement.get(), 5);
    record.qualityScore = sqlite3_column_double(statement.get(), 6);
    record.personId = sqlite3_column_int64(statement.get(), 7);
    record.personName = textColumn(statement.get(), 8);
    record.ignored = sqlite3_column_int(statement.get(), 9) != 0;
    record.reviewState = textColumn(statement.get(), 10);
    record.notes = textColumn(statement.get(), 11);
    record.createdAt = textColumn(statement.get(), 12);
    return record;
}

std::vector<PersonSummary> Database::loadPeople() const {
    const char* sql =
        "SELECT p.id, p.name, COUNT(f.id) AS face_count, "
        "SUM(CASE WHEN f.ignored != 0 THEN 1 ELSE 0 END) AS ignored_count, "
        "SUM(CASE WHEN f.ignored = 0 AND f.review_state != 'reviewed' THEN 1 ELSE 0 END) AS review_count, "
        "COALESCE(AVG(f.quality_score), 0), COALESCE(profile.status, ''), "
        "COALESCE(profile.sample_count, 0), COALESCE(profile.prototype_count, 0), "
        "COALESCE(profile.accept_threshold, 0), COALESCE(profile.calibration_json, '{}') "
        "FROM persons p "
        "LEFT JOIN faces f ON f.person_id = p.id "
        "LEFT JOIN person_identity_profiles profile ON profile.person_id = p.id "
        "GROUP BY p.id, p.name, profile.status, profile.sample_count, profile.prototype_count, "
        "profile.accept_threshold, profile.calibration_json "
        "ORDER BY p.name COLLATE NOCASE";
    Statement statement(db_, sql);
    std::vector<PersonSummary> people;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        PersonSummary person;
        person.id = sqlite3_column_int64(statement.get(), 0);
        person.name = textColumn(statement.get(), 1);
        person.faceCount = sqlite3_column_int64(statement.get(), 2);
        person.ignoredCount = sqlite3_column_int64(statement.get(), 3);
        person.reviewCount = sqlite3_column_int64(statement.get(), 4);
        person.averageQuality = sqlite3_column_double(statement.get(), 5);
        person.identityStatus = textColumn(statement.get(), 6);
        person.identitySampleCount = sqlite3_column_int(statement.get(), 7);
        person.identityExemplarCount = sqlite3_column_int(statement.get(), 8);
        person.identityAcceptThreshold = sqlite3_column_double(statement.get(), 9);
        try {
            const auto calibration = nlohmann::json::parse(textColumn(statement.get(), 10));
            person.identityHealth = calibration.value("health", "");
        } catch (...) {
            person.identityHealth.clear();
        }
        people.push_back(std::move(person));
    }
    return people;
}

std::vector<IdentityProfile> Database::loadIdentityProfiles() const {
    const char* sql =
        "SELECT profile.person_id, p.name, profile.sample_count, profile.prototype_count, "
        "profile.embedding_dim, profile.centroid_blob, profile.prototypes_blob, profile.exemplar_blob, "
        "profile.exemplar_face_ids_json, profile.exemplar_weights_blob, profile.hard_negative_face_ids_json, "
        "profile.thresholds_json, profile.calibration_json, profile.strategy_version, "
        "profile.scoring_model_version, profile.accept_threshold, profile.review_threshold, "
        "profile.mean_similarity, profile.min_similarity, profile.max_similarity, profile.quality_mean, "
        "profile.evidence_face_ids_json, profile.status, profile.updated_at "
        "FROM person_identity_profiles profile JOIN persons p ON p.id = profile.person_id "
        "ORDER BY p.name COLLATE NOCASE";
    Statement statement(db_, sql);
    std::vector<IdentityProfile> profiles;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        IdentityProfile profile;
        profile.personId = sqlite3_column_int64(statement.get(), 0);
        profile.personName = textColumn(statement.get(), 1);
        profile.sampleCount = sqlite3_column_int(statement.get(), 2);
        profile.prototypeCount = sqlite3_column_int(statement.get(), 3);
        profile.embeddingDim = sqlite3_column_int(statement.get(), 4);
        profile.centroid = normalize(floatBlob(statement.get(), 5, profile.embeddingDim));
        profile.prototypes = rowsFromFlat(floatBlob(statement.get(), 6, profile.embeddingDim), profile.prototypeCount, profile.embeddingDim);

        auto exemplarFlat = floatBlob(statement.get(), 7, profile.embeddingDim);
        const int exemplarCount = profile.embeddingDim > 0
            ? static_cast<int>(exemplarFlat.size() / static_cast<size_t>(profile.embeddingDim))
            : 0;
        profile.exemplars = rowsFromFlat(std::move(exemplarFlat), exemplarCount, profile.embeddingDim);
        if (profile.exemplars.empty()) {
            profile.exemplars = profile.prototypes;
        }

        profile.exemplarFaceIds = intListFromJson(textColumn(statement.get(), 8));
        profile.exemplarWeights = floatBlob(statement.get(), 9, 0);
        if (profile.exemplarWeights.size() != profile.exemplars.size()) {
            profile.exemplarWeights.assign(profile.exemplars.size(), 1.0f);
        }

        profile.hardNegativeFaceIds = intListFromJson(textColumn(statement.get(), 10));
        try {
            const auto thresholds = nlohmann::json::parse(textColumn(statement.get(), 11));
            profile.strict = thresholdsFor(thresholds, "strict", {0.72, 0.62, 0.035});
            profile.balanced = thresholdsFor(thresholds, "balanced", {0.68, 0.58, 0.025});
            profile.broad = thresholdsFor(thresholds, "broad", {0.62, 0.52, 0.015});
        } catch (...) {
            profile.strict = {0.72, 0.62, 0.035};
            profile.balanced = {0.68, 0.58, 0.025};
            profile.broad = {0.62, 0.52, 0.015};
        }
        try {
            const auto calibration = nlohmann::json::parse(textColumn(statement.get(), 12));
            profile.health = calibration.value("health", "");
        } catch (...) {
            profile.health.clear();
        }
        profile.strategyVersion = textColumn(statement.get(), 13);
        profile.scoringModelVersion = textColumn(statement.get(), 14);
        profile.acceptThreshold = sqlite3_column_double(statement.get(), 15);
        profile.reviewThreshold = sqlite3_column_double(statement.get(), 16);
        profile.meanSimilarity = sqlite3_column_double(statement.get(), 17);
        profile.minSimilarity = sqlite3_column_double(statement.get(), 18);
        profile.maxSimilarity = sqlite3_column_double(statement.get(), 19);
        profile.qualityMean = sqlite3_column_double(statement.get(), 20);
        profile.evidenceFaceIds = intListFromJson(textColumn(statement.get(), 21));
        profile.status = textColumn(statement.get(), 22);
        profile.updatedAt = textColumn(statement.get(), 23);
        profiles.push_back(std::move(profile));
    }

    for (auto& profile : profiles) {
        if (profile.hardNegativeFaceIds.empty()) {
            continue;
        }
        std::string sqlHard = "SELECT id, embedding_blob, embedding_dim FROM faces WHERE id IN (";
        for (size_t i = 0; i < profile.hardNegativeFaceIds.size(); ++i) {
            sqlHard += i == 0 ? "?" : ",?";
        }
        sqlHard += ")";
        Statement hardStatement(db_, sqlHard.c_str());
        for (size_t i = 0; i < profile.hardNegativeFaceIds.size(); ++i) {
            sqlite3_bind_int64(hardStatement.get(), static_cast<int>(i + 1), profile.hardNegativeFaceIds[i]);
        }
        while (sqlite3_step(hardStatement.get()) == SQLITE_ROW) {
            const int dim = sqlite3_column_int(hardStatement.get(), 2);
            auto embedding = normalize(floatBlob(hardStatement.get(), 1, dim));
            if (dim == profile.embeddingDim && !embedding.empty()) {
                profile.hardNegativeEmbeddings.push_back(std::move(embedding));
            }
        }
    }
    return profiles;
}

} // namespace fsc::core
