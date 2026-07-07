#include "fsc/core/Database.hpp"

#include "fsc/core/VectorMath.hpp"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
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

std::string trimText(std::string value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string lowerText(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> parseTagText(const std::string& tagText) {
    std::string normalized = tagText;
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    std::replace(normalized.begin(), normalized.end(), '|', ',');
    std::vector<std::string> tags;
    std::set<std::string> seen;
    std::stringstream stream(normalized);
    std::string part;
    while (std::getline(stream, part, ',')) {
        auto tag = trimText(part);
        const auto key = lowerText(tag);
        if (!tag.empty() && seen.insert(key).second) {
            tags.push_back(std::move(tag));
        }
    }
    return tags;
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

std::vector<double> doubleListFromJson(const std::string& value) {
    std::vector<double> output;
    if (value.empty()) {
        return output;
    }
    try {
        const auto root = nlohmann::json::parse(value);
        if (!root.is_array()) {
            return output;
        }
        output.reserve(root.size());
        for (const auto& item : root) {
            if (item.is_number()) {
                output.push_back(item.get<double>());
            }
        }
    } catch (...) {
        output.clear();
    }
    return output;
}

std::vector<std::vector<double>> pointRowsFromJson(const std::string& value) {
    std::vector<std::vector<double>> rows;
    if (value.empty()) {
        return rows;
    }
    try {
        const auto root = nlohmann::json::parse(value);
        if (!root.is_array()) {
            return rows;
        }
        rows.reserve(root.size());
        for (const auto& item : root) {
            if (!item.is_array() || item.size() < 2) {
                continue;
            }
            std::vector<double> row;
            row.reserve(std::min<size_t>(item.size(), 4));
            for (const auto& valueItem : item) {
                if (valueItem.is_number()) {
                    row.push_back(valueItem.get<double>());
                }
            }
            if (row.size() >= 2) {
                rows.push_back(std::move(row));
            }
        }
    } catch (...) {
        rows.clear();
    }
    return rows;
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

struct TrainingFace {
    int64_t id = 0;
    int64_t personId = 0;
    std::string personName;
    int embeddingDim = 0;
    std::vector<float> embedding;
    double quality = 0.0;
};

void execSql(sqlite3* db, const char* sql) {
    char* error = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &error) != SQLITE_OK) {
        std::string message = error == nullptr ? "SQLite command failed." : error;
        sqlite3_free(error);
        throw std::runtime_error(message);
    }
}

std::string placeholders(size_t count) {
    std::string output;
    for (size_t i = 0; i < count; ++i) {
        output += i == 0 ? "?" : ",?";
    }
    return output;
}

std::vector<float> weightedCentroid(const std::vector<std::vector<float>>& matrix, const std::vector<double>& qualities) {
    if (matrix.empty()) {
        return {};
    }
    std::vector<float> centroid(matrix.front().size(), 0.0f);
    double total = 0.0;
    for (size_t row = 0; row < matrix.size(); ++row) {
        const double weight = std::clamp(qualities[row], 0.10, 1.0);
        total += weight;
        for (size_t dim = 0; dim < centroid.size(); ++dim) {
            centroid[dim] += static_cast<float>(matrix[row][dim] * weight);
        }
    }
    if (total > 1e-8) {
        for (auto& value : centroid) {
            value = static_cast<float>(value / total);
        }
    }
    return normalize(centroid);
}

std::vector<double> consistencyScores(const std::vector<std::vector<float>>& matrix) {
    std::vector<double> consistency(matrix.size(), 1.0);
    if (matrix.size() <= 1) {
        return consistency;
    }
    for (size_t row = 0; row < matrix.size(); ++row) {
        double total = 0.0;
        int count = 0;
        for (size_t other = 0; other < matrix.size(); ++other) {
            if (row == other) {
                continue;
            }
            total += dot(matrix[row], matrix[other]);
            ++count;
        }
        consistency[row] = count > 0 ? total / count : 1.0;
    }
    return consistency;
}

std::vector<int> selectExemplarPositions(
    const std::vector<std::vector<float>>& matrix,
    const std::vector<double>& qualities,
    int maxExemplars) {
    if (matrix.empty()) {
        return {};
    }
    const auto consistency = consistencyScores(matrix);
    std::vector<int> order(matrix.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        const double leftScore = std::clamp(qualities[static_cast<size_t>(left)], 0.0, 1.0) * 0.62 + consistency[static_cast<size_t>(left)] * 0.38;
        const double rightScore = std::clamp(qualities[static_cast<size_t>(right)], 0.0, 1.0) * 0.62 + consistency[static_cast<size_t>(right)] * 0.38;
        if (std::abs(leftScore - rightScore) > 1e-12) {
            return leftScore > rightScore;
        }
        return left < right;
    });

    const int limit = std::max(1, maxExemplars);
    std::vector<int> selected;
    for (int index : order) {
        if (selected.empty()) {
            selected.push_back(index);
        } else {
            double maxSimilarity = -1.0;
            for (int existing : selected) {
                maxSimilarity = std::max(maxSimilarity, dot(matrix[static_cast<size_t>(existing)], matrix[static_cast<size_t>(index)]));
            }
            if (maxSimilarity < 0.985 || selected.size() < std::min<size_t>(3, matrix.size())) {
                selected.push_back(index);
            }
        }
        if (selected.size() >= static_cast<size_t>(limit)) {
            break;
        }
    }
    return selected;
}

std::vector<float> exemplarWeights(
    const std::vector<std::vector<float>>& matrix,
    const std::vector<double>& qualities,
    const std::vector<int>& positions) {
    if (positions.empty()) {
        return {};
    }
    const auto consistency = consistencyScores(matrix);
    std::vector<float> weights;
    weights.reserve(positions.size());
    for (int position : positions) {
        const size_t index = static_cast<size_t>(position);
        const double quality = std::clamp(qualities[index], 0.10, 1.0);
        const double centrality = std::clamp((consistency[index] + 1.0) * 0.5, 0.10, 1.0);
        weights.push_back(static_cast<float>(std::max(0.05, quality * 0.65 + centrality * 0.35)));
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total > 1e-8) {
        for (auto& weight : weights) {
            weight = static_cast<float>(weight / total);
        }
    }
    return weights;
}

std::vector<float> flattenRows(const std::vector<std::vector<float>>& rows) {
    std::vector<float> output;
    size_t total = 0;
    for (const auto& row : rows) {
        total += row.size();
    }
    output.reserve(total);
    for (const auto& row : rows) {
        output.insert(output.end(), row.begin(), row.end());
    }
    return output;
}

void bindFloatVector(sqlite3_stmt* stmt, int index, const std::vector<float>& values) {
    sqlite3_bind_blob(
        stmt,
        index,
        values.data(),
        static_cast<int>(values.size() * sizeof(float)),
        SQLITE_TRANSIENT);
}

nlohmann::json thresholdsJson(const IdentityProfile& profile) {
    return {
        {"strict", {{"accept", profile.strict.accept}, {"review", profile.strict.review}, {"margin", profile.strict.margin}}},
        {"balanced", {{"accept", profile.balanced.accept}, {"review", profile.balanced.review}, {"margin", profile.balanced.margin}}},
        {"broad", {{"accept", profile.broad.accept}, {"review", profile.broad.review}, {"margin", profile.broad.margin}}},
    };
}

double ridgeScore(const std::vector<std::vector<float>>& exemplars, const std::vector<float>& query) {
    if (exemplars.empty()) {
        return 0.0;
    }
    if (exemplars.size() == 1) {
        return dot(exemplars.front(), query);
    }

    const size_t n = exemplars.size();
    std::vector<std::vector<double>> gram(n, std::vector<double>(n, 0.0));
    std::vector<double> rhs(n, 0.0);
    for (size_t row = 0; row < n; ++row) {
        rhs[row] = dot(exemplars[row], query);
        for (size_t col = 0; col < n; ++col) {
            gram[row][col] = dot(exemplars[row], exemplars[col]);
        }
        gram[row][row] += 0.035;
    }

    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(gram[row][col]) > std::abs(gram[pivot][col])) {
                pivot = row;
            }
        }
        std::swap(gram[col], gram[pivot]);
        std::swap(rhs[col], rhs[pivot]);
        const double divisor = std::abs(gram[col][col]) < 1e-9 ? 1e-9 : gram[col][col];
        for (size_t j = col; j < n; ++j) {
            gram[col][j] /= divisor;
        }
        rhs[col] /= divisor;
        for (size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = gram[row][col];
            for (size_t j = col; j < n; ++j) {
                gram[row][j] -= factor * gram[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    for (auto& weight : rhs) {
        weight = std::max(0.0, weight);
    }
    const double total = std::accumulate(rhs.begin(), rhs.end(), 0.0);
    if (total <= 1e-8) {
        double best = -1.0;
        for (const auto& exemplar : exemplars) {
            best = std::max(best, dot(exemplar, query));
        }
        return best;
    }

    std::vector<float> reconstruction(query.size(), 0.0f);
    for (size_t row = 0; row < exemplars.size(); ++row) {
        for (size_t dim = 0; dim < reconstruction.size(); ++dim) {
            reconstruction[dim] += static_cast<float>((rhs[row] / total) * exemplars[row][dim]);
        }
    }
    reconstruction = normalize(reconstruction);
    return dot(reconstruction, query);
}

double scoreProfileFallback(
    const std::vector<float>& centroid,
    const std::vector<std::vector<float>>& exemplars,
    const std::vector<float>& weights,
    const std::vector<float>& query) {
    if (query.empty()) {
        return 0.0;
    }
    std::vector<double> scores;
    scores.reserve(exemplars.size());
    for (const auto& exemplar : exemplars) {
        scores.push_back(exemplar.size() == query.size() ? dot(exemplar, query) : -1.0);
    }
    const double centroidScore = centroid.size() == query.size() ? dot(centroid, query) : 0.0;
    const double best = scores.empty() ? centroidScore : *std::max_element(scores.begin(), scores.end());
    auto sorted = scores;
    std::sort(sorted.begin(), sorted.end());
    const size_t topCount = std::min<size_t>(3, sorted.size());
    const double topMean = topCount == 0
        ? best
        : std::accumulate(sorted.end() - static_cast<std::ptrdiff_t>(topCount), sorted.end(), 0.0) / static_cast<double>(topCount);
    double weightScore = best;
    if (weights.size() == scores.size() && !scores.empty()) {
        weightScore = 0.0;
        for (size_t index = 0; index < scores.size(); ++index) {
            weightScore += scores[index] * weights[index];
        }
    }
    const double reconstruction = ridgeScore(exemplars, query);
    return best * 0.32 + topMean * 0.20 + centroidScore * 0.20 + reconstruction * 0.20 + weightScore * 0.08;
}

std::vector<double> upperPairwiseScores(const std::vector<std::vector<float>>& matrix) {
    std::vector<double> scores;
    if (matrix.size() < 2) {
        return scores;
    }
    for (size_t row = 0; row < matrix.size(); ++row) {
        for (size_t col = row + 1; col < matrix.size(); ++col) {
            scores.push_back(dot(matrix[row], matrix[col]));
        }
    }
    return scores;
}

double percentile(std::vector<double> values, double percent) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * percent / 100.0;
    const auto low = static_cast<size_t>(std::floor(position));
    const auto high = static_cast<size_t>(std::ceil(position));
    if (low == high) {
        return values[low];
    }
    const double fraction = position - static_cast<double>(low);
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

std::string utcNowText() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &time);
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

void createSchema(sqlite3* db) {
    execSql(db, R"sql(
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            source_path TEXT,
            embedding_blob BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            bbox_json TEXT,
            kps_json TEXT,
            landmarks_json TEXT,
            landmarks3d_json TEXT,
            face_mesh3d_json TEXT,
            det_score REAL,
            quality_score REAL NOT NULL DEFAULT 0,
            quality_json TEXT,
            preview_png BLOB,
            person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
            image_hash TEXT,
            ignored INTEGER NOT NULL DEFAULT 0,
            review_state TEXT NOT NULL DEFAULT 'open',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE face_tags (
            face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY(face_id, tag_id)
        );

        CREATE TABLE person_identity_profiles (
            person_id INTEGER PRIMARY KEY REFERENCES persons(id) ON DELETE CASCADE,
            sample_count INTEGER NOT NULL,
            prototype_count INTEGER NOT NULL,
            embedding_dim INTEGER NOT NULL,
            centroid_blob BLOB NOT NULL,
            prototypes_blob BLOB NOT NULL,
            exemplar_blob BLOB,
            exemplar_face_ids_json TEXT NOT NULL DEFAULT '[]',
            exemplar_weights_blob BLOB,
            hard_negative_face_ids_json TEXT NOT NULL DEFAULT '[]',
            thresholds_json TEXT NOT NULL DEFAULT '{}',
            calibration_json TEXT NOT NULL DEFAULT '{}',
            strategy_version TEXT NOT NULL DEFAULT 'gallery_v2',
            scoring_model_version TEXT NOT NULL DEFAULT 'numpy_gallery_v1',
            accept_threshold REAL NOT NULL,
            review_threshold REAL NOT NULL,
            mean_similarity REAL NOT NULL,
            min_similarity REAL NOT NULL,
            max_similarity REAL NOT NULL,
            quality_mean REAL NOT NULL,
            evidence_face_ids_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'weak',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_faces_file_name ON faces(file_name);
        CREATE INDEX idx_faces_quality_score ON faces(quality_score);
        CREATE INDEX idx_faces_person_id ON faces(person_id);
        CREATE INDEX idx_faces_image_hash ON faces(image_hash);
        CREATE INDEX idx_faces_ignored ON faces(ignored);
        CREATE INDEX idx_faces_review_state ON faces(review_state);
        CREATE INDEX idx_face_tags_tag_id ON face_tags(tag_id);
    )sql");
}

void writeMetadata(sqlite3* db, const std::string& key, const std::string& value) {
    Statement statement(db, "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)");
    sqlite3_bind_text(statement.get(), 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement.get(), 2, value.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db));
    }
}

IdentityProfile buildIdentityProfileFromFaces(
    int64_t personId,
    const std::string& personName,
    const std::vector<TrainingFace>& faces,
    const std::vector<TrainingFace>& negatives,
    int maxExemplars) {
    IdentityProfile profile;
    profile.personId = personId;
    profile.personName = personName;
    if (faces.empty()) {
        return profile;
    }

    const int dim = faces.front().embeddingDim;
    std::vector<std::vector<float>> matrix;
    std::vector<int64_t> faceIds;
    std::vector<double> qualities;
    for (const auto& face : faces) {
        if (face.embeddingDim == dim && face.embedding.size() == static_cast<size_t>(dim)) {
            matrix.push_back(normalize(face.embedding));
            faceIds.push_back(face.id);
            qualities.push_back(face.quality);
        }
    }
    if (matrix.empty()) {
        return profile;
    }

    profile.sampleCount = static_cast<int>(matrix.size());
    profile.embeddingDim = dim;
    profile.centroid = weightedCentroid(matrix, qualities);

    const auto selected = selectExemplarPositions(matrix, qualities, maxExemplars);
    profile.exemplarWeights = exemplarWeights(matrix, qualities, selected);
    for (int position : selected) {
        profile.exemplars.push_back(matrix[static_cast<size_t>(position)]);
        profile.exemplarFaceIds.push_back(faceIds[static_cast<size_t>(position)]);
    }
    profile.prototypes = profile.exemplars;
    profile.prototypeCount = static_cast<int>(profile.exemplars.size());
    profile.evidenceFaceIds = profile.exemplarFaceIds;

    const auto pairwise = upperPairwiseScores(matrix);
    if (pairwise.empty()) {
        profile.meanSimilarity = 1.0;
        profile.minSimilarity = 1.0;
        profile.maxSimilarity = 1.0;
    } else {
        profile.meanSimilarity = std::accumulate(pairwise.begin(), pairwise.end(), 0.0) / static_cast<double>(pairwise.size());
        profile.minSimilarity = *std::min_element(pairwise.begin(), pairwise.end());
        profile.maxSimilarity = *std::max_element(pairwise.begin(), pairwise.end());
    }
    const double mean = profile.meanSimilarity;
    double variance = 0.0;
    for (const double value : pairwise) {
        variance += (value - mean) * (value - mean);
    }
    const double scoreStd = pairwise.empty() ? 0.0 : std::sqrt(variance / static_cast<double>(pairwise.size()));
    profile.qualityMean = std::accumulate(qualities.begin(), qualities.end(), 0.0) / static_cast<double>(qualities.size());

    std::vector<std::vector<float>> negativeMatrix;
    std::vector<int64_t> negativeFaceIds;
    for (const auto& face : negatives) {
        if (face.embeddingDim == dim && face.embedding.size() == static_cast<size_t>(dim)) {
            negativeMatrix.push_back(normalize(face.embedding));
            negativeFaceIds.push_back(face.id);
        }
    }

    std::vector<double> negativeScores;
    negativeScores.reserve(negativeMatrix.size());
    for (const auto& negative : negativeMatrix) {
        negativeScores.push_back(scoreProfileFallback(profile.centroid, profile.exemplars, profile.exemplarWeights, negative));
    }
    std::vector<int> negativeOrder(negativeScores.size());
    std::iota(negativeOrder.begin(), negativeOrder.end(), 0);
    std::sort(negativeOrder.begin(), negativeOrder.end(), [&](int left, int right) {
        if (std::abs(negativeScores[static_cast<size_t>(left)] - negativeScores[static_cast<size_t>(right)]) > 1e-12) {
            return negativeScores[static_cast<size_t>(left)] > negativeScores[static_cast<size_t>(right)];
        }
        return left < right;
    });
    for (int index : negativeOrder) {
        if (profile.hardNegativeFaceIds.size() >= 12) {
            break;
        }
        profile.hardNegativeFaceIds.push_back(negativeFaceIds[static_cast<size_t>(index)]);
    }

    std::vector<double> positiveScores;
    if (matrix.size() >= 2) {
        for (size_t holdout = 0; holdout < matrix.size(); ++holdout) {
            std::vector<std::vector<float>> support;
            std::vector<double> supportQualities;
            for (size_t row = 0; row < matrix.size(); ++row) {
                if (row == holdout) {
                    continue;
                }
                support.push_back(matrix[row]);
                supportQualities.push_back(qualities[row]);
            }
            const auto supportCentroid = weightedCentroid(support, supportQualities);
            const auto supportPositions = selectExemplarPositions(
                support,
                supportQualities,
                std::min(12, std::max<int>(1, static_cast<int>(support.size()))));
            std::vector<std::vector<float>> supportExemplars;
            for (int position : supportPositions) {
                supportExemplars.push_back(support[static_cast<size_t>(position)]);
            }
            const auto supportWeights = exemplarWeights(support, supportQualities, supportPositions);
            positiveScores.push_back(scoreProfileFallback(supportCentroid, supportExemplars, supportWeights, matrix[holdout]));
        }
    }

    const double positiveMean = positiveScores.empty()
        ? profile.meanSimilarity - std::max(0.04, scoreStd * 1.2)
        : std::accumulate(positiveScores.begin(), positiveScores.end(), 0.0) / static_cast<double>(positiveScores.size());
    const double posP05 = positiveScores.empty() ? positiveMean : percentile(positiveScores, 5.0);
    const double posP10 = positiveScores.empty() ? positiveMean : percentile(positiveScores, 10.0);
    const double posP20 = positiveScores.empty() ? positiveMean : percentile(positiveScores, 20.0);
    const double negP90 = negativeScores.empty() ? 0.0 : percentile(negativeScores, 90.0);
    const double negP95 = negativeScores.empty() ? 0.0 : percentile(negativeScores, 95.0);
    const double negP99 = negativeScores.empty() ? 0.0 : percentile(negativeScores, 99.0);
    const double negMax = negativeScores.empty() ? 0.0 : *std::max_element(negativeScores.begin(), negativeScores.end());

    double strictAccept = 0.92;
    double balancedAccept = 0.90;
    double broadAccept = 0.88;
    if (profile.sampleCount >= 3) {
        strictAccept = std::clamp(std::max(negP99 + 0.045, posP20 - 0.025), 0.48, 0.94);
        balancedAccept = std::clamp(std::max(negP95 + 0.030, posP10 - 0.045), 0.43, strictAccept);
        broadAccept = std::clamp(std::max(negP90 + 0.015, posP05 - 0.070), 0.38, balancedAccept);
    }
    profile.strict = {strictAccept, std::clamp(strictAccept - 0.100, 0.35, strictAccept - 0.020), 0.055};
    profile.balanced = {balancedAccept, std::clamp(balancedAccept - 0.115, 0.32, balancedAccept - 0.020), 0.040};
    profile.broad = {broadAccept, std::clamp(broadAccept - 0.130, 0.28, broadAccept - 0.020), 0.025};
    profile.acceptThreshold = profile.strict.accept;
    profile.reviewThreshold = profile.strict.review;
    profile.status = profile.sampleCount >= 3 ? "strong" : "weak";
    if (profile.status == "weak") {
        profile.health = "weak: add at least 3 confirmed faces";
    } else if (profile.sampleCount < 5) {
        profile.health = "fair: add more angles";
    } else if (!positiveScores.empty() && !negativeScores.empty()) {
        const double gap = percentile(positiveScores, 10.0) - percentile(negativeScores, 95.0);
        profile.health = gap < 0.03 ? "risky: close hard negatives" : (gap < 0.08 ? "fair: narrow margin" : "healthy");
    } else {
        profile.health = "healthy";
    }
    profile.strategyVersion = "gallery_v2";
    profile.scoringModelVersion = "numpy_gallery_v1";
    profile.updatedAt = utcNowText();
    return profile;
}


} // namespace

void Database::createEmpty(std::filesystem::path path, bool replace) {
    if (path.extension() == ".dtb") {
        throw std::runtime_error("Legacy .dtb files cannot be overwritten with the native .fscdb schema.");
    }
    if (!path.has_extension()) {
        path += ".fscdb";
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (replace) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(path.string() + "-wal", ignored);
        std::filesystem::remove(path.string() + "-shm", ignored);
    } else if (std::filesystem::exists(path)) {
        throw std::runtime_error("Database already exists: " + path.string());
    }

    sqlite3* db = nullptr;
    const int code = sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (code != SQLITE_OK) {
        const std::string message = db == nullptr ? "Failed to create SQLite database." : sqlite3_errmsg(db);
        if (db != nullptr) {
            sqlite3_close(db);
        }
        throw std::runtime_error(message);
    }

    try {
        execSql(db, "PRAGMA journal_mode=WAL");
        execSql(db, "PRAGMA foreign_keys=ON");
        createSchema(db);
        writeMetadata(db, "format_version", "8");
        writeMetadata(db, "metric", "cosine_normed_embedding");
        writeMetadata(db, "created_at", utcNowText());
        writeMetadata(db, "model_name", "buffalo_l");
        writeMetadata(db, "application", "FSC Studio Native");
        sqlite3_close(db);
    } catch (...) {
        sqlite3_close(db);
        throw;
    }
}

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
        Statement statement(db_, "SELECT COUNT(*) FROM tags");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.tagCount = sqlite3_column_int64(statement.get(), 0);
        }
    }
    {
        Statement statement(db_, "SELECT COUNT(*) FROM faces WHERE ignored = 0 AND review_state != 'reviewed'");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.reviewCount = sqlite3_column_int64(statement.get(), 0);
        }
    }
    {
        Statement statement(db_, "SELECT COUNT(*) FROM faces WHERE ignored != 0");
        if (sqlite3_step(statement.get()) == SQLITE_ROW) {
            stats.ignoredCount = sqlite3_column_int64(statement.get(), 0);
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
        "COALESCE(f.created_at, ''), "
        "COALESCE((SELECT GROUP_CONCAT(name, ', ') FROM ("
        "SELECT t.name AS name FROM face_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.face_id = f.id ORDER BY t.name COLLATE NOCASE"
        ")), '') "
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
        record.tagText = textColumn(statement.get(), 13);
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
        "COALESCE(f.created_at, ''), COALESCE(f.bbox_json, ''), COALESCE(f.landmarks_json, ''), "
        "COALESCE(f.landmarks3d_json, ''), COALESCE(f.face_mesh3d_json, ''), "
        "COALESCE((SELECT GROUP_CONCAT(name, ', ') FROM ("
        "SELECT t.name AS name FROM face_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.face_id = f.id ORDER BY t.name COLLATE NOCASE"
        ")), '') "
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
    record.bbox = doubleListFromJson(textColumn(statement.get(), 13));
    record.landmarks2d = pointRowsFromJson(textColumn(statement.get(), 14));
    record.landmarks3d = pointRowsFromJson(textColumn(statement.get(), 15));
    record.faceMesh3d = pointRowsFromJson(textColumn(statement.get(), 16));
    record.tagText = textColumn(statement.get(), 17);
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

std::vector<std::string> Database::loadTags() const {
    Statement statement(db_, "SELECT name FROM tags ORDER BY name COLLATE NOCASE");
    std::vector<std::string> tags;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        tags.push_back(textColumn(statement.get(), 0));
    }
    return tags;
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

bool Database::imageHashExists(const std::string& imageHash) const {
    if (imageHash.empty()) {
        return false;
    }
    Statement statement(db_, "SELECT 1 FROM faces WHERE image_hash = ? LIMIT 1");
    sqlite3_bind_text(statement.get(), 1, imageHash.c_str(), -1, SQLITE_TRANSIENT);
    return sqlite3_step(statement.get()) == SQLITE_ROW;
}

MaintenanceResult Database::checkIntegrity() const {
    Statement statement(db_, "PRAGMA integrity_check");
    std::vector<std::string> messages;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        messages.push_back(textColumn(statement.get(), 0));
    }
    MaintenanceResult result;
    result.action = "integrity_check";
    result.ok = messages.size() == 1 && messages.front() == "ok";
    if (messages.empty()) {
        result.message = "No integrity result returned.";
    } else {
        std::ostringstream stream;
        const size_t limit = std::min<size_t>(messages.size(), 20);
        for (size_t index = 0; index < limit; ++index) {
            if (index > 0) {
                stream << "; ";
            }
            stream << messages[index];
        }
        if (messages.size() > limit) {
            stream << "; ... " << (messages.size() - limit) << " more";
        }
        result.message = stream.str();
    }
    return result;
}

MaintenanceResult Database::backupTo(const std::filesystem::path& outputPath) const {
    auto output = std::filesystem::absolute(outputPath);
    const auto source = std::filesystem::absolute(path_);
    if (output == source) {
        throw std::runtime_error("Backup path must differ from the source database.");
    }
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }

    sqlite3* destination = nullptr;
    const int openCode = sqlite3_open_v2(output.string().c_str(), &destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (openCode != SQLITE_OK) {
        std::string message = destination == nullptr ? "Failed to open backup database." : sqlite3_errmsg(destination);
        if (destination != nullptr) {
            sqlite3_close(destination);
        }
        throw std::runtime_error(message);
    }

    sqlite3_backup* backup = sqlite3_backup_init(destination, "main", db_, "main");
    if (backup == nullptr) {
        std::string message = sqlite3_errmsg(destination);
        sqlite3_close(destination);
        throw std::runtime_error(message);
    }
    const int stepCode = sqlite3_backup_step(backup, -1);
    const int finishCode = sqlite3_backup_finish(backup);
    if (finishCode != SQLITE_OK || (stepCode != SQLITE_DONE && stepCode != SQLITE_OK)) {
        std::string message = sqlite3_errmsg(destination);
        sqlite3_close(destination);
        throw std::runtime_error(message);
    }
    sqlite3_close(destination);

    MaintenanceResult result;
    result.action = "backup";
    result.ok = true;
    result.outputPath = output.string();
    result.message = "Backup created: " + result.outputPath;
    return result;
}

MaintenanceResult Database::checkpointWal(bool truncate) {
    Statement statement(db_, truncate ? "PRAGMA wal_checkpoint(TRUNCATE)" : "PRAGMA wal_checkpoint(PASSIVE)");
    MaintenanceResult result;
    result.action = "wal_checkpoint";
    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        result.ok = false;
        result.message = "No checkpoint result returned.";
        return result;
    }
    const int busy = sqlite3_column_int(statement.get(), 0);
    const int log = sqlite3_column_int(statement.get(), 1);
    const int checkpointed = sqlite3_column_int(statement.get(), 2);
    result.ok = true;
    result.message = "busy=" + std::to_string(busy) + " log=" + std::to_string(log) + " checkpointed=" + std::to_string(checkpointed);
    return result;
}

MaintenanceResult Database::vacuum() {
    execSql(db_, "VACUUM");
    return {"vacuum", true, "VACUUM completed.", {}};
}

int64_t Database::upsertPerson(const std::string& name, const std::string& notes) {
    if (name.empty()) {
        throw std::runtime_error("Person name cannot be empty.");
    }
    {
        Statement statement(db_, "INSERT OR IGNORE INTO persons(name, notes) VALUES (?, ?)");
        sqlite3_bind_text(statement.get(), 1, name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(statement.get(), 2, notes.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(statement.get()) != SQLITE_DONE) {
            throw std::runtime_error(sqlite3_errmsg(db_));
        }
    }
    if (!notes.empty()) {
        Statement update(db_, "UPDATE persons SET notes = ? WHERE name = ?");
        sqlite3_bind_text(update.get(), 1, notes.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(update.get(), 2, name.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(update.get()) != SQLITE_DONE) {
            throw std::runtime_error(sqlite3_errmsg(db_));
        }
    }
    Statement select(db_, "SELECT id FROM persons WHERE name = ?");
    sqlite3_bind_text(select.get(), 1, name.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(select.get()) != SQLITE_ROW) {
        throw std::runtime_error("Failed to read inserted person id.");
    }
    return sqlite3_column_int64(select.get(), 0);
}

void Database::assignFaceToPerson(int64_t faceId, int64_t personId) {
    Statement statement(db_, "UPDATE faces SET person_id = ? WHERE id = ?");
    if (personId > 0) {
        sqlite3_bind_int64(statement.get(), 1, personId);
    } else {
        sqlite3_bind_null(statement.get(), 1);
    }
    sqlite3_bind_int64(statement.get(), 2, faceId);
    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_));
    }
    if (sqlite3_changes(db_) == 0) {
        throw std::runtime_error("Face id not found: " + std::to_string(faceId));
    }
}

void Database::setFaceTags(int64_t faceId, const std::string& tagText, bool append) {
    {
        Statement check(db_, "SELECT 1 FROM faces WHERE id = ? LIMIT 1");
        sqlite3_bind_int64(check.get(), 1, faceId);
        if (sqlite3_step(check.get()) != SQLITE_ROW) {
            throw std::runtime_error("Face id not found: " + std::to_string(faceId));
        }
    }

    const auto tags = parseTagText(tagText);
    execSql(db_, "BEGIN IMMEDIATE");
    try {
        if (!append) {
            {
                Statement clear(db_, "DELETE FROM face_tags WHERE face_id = ?");
                sqlite3_bind_int64(clear.get(), 1, faceId);
                if (sqlite3_step(clear.get()) != SQLITE_DONE) {
                    throw std::runtime_error(sqlite3_errmsg(db_));
                }
            }
        }
        for (const auto& tag : tags) {
            {
                Statement insertTag(db_, "INSERT OR IGNORE INTO tags(name) VALUES (?)");
                sqlite3_bind_text(insertTag.get(), 1, tag.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(insertTag.get()) != SQLITE_DONE) {
                    throw std::runtime_error(sqlite3_errmsg(db_));
                }
            }
            int64_t tagId = 0;
            {
                Statement selectTag(db_, "SELECT id FROM tags WHERE name = ?");
                sqlite3_bind_text(selectTag.get(), 1, tag.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(selectTag.get()) != SQLITE_ROW) {
                    throw std::runtime_error("Failed to read tag id.");
                }
                tagId = sqlite3_column_int64(selectTag.get(), 0);
            }
            Statement link(db_, "INSERT OR IGNORE INTO face_tags(face_id, tag_id) VALUES (?, ?)");
            sqlite3_bind_int64(link.get(), 1, faceId);
            sqlite3_bind_int64(link.get(), 2, tagId);
            if (sqlite3_step(link.get()) != SQLITE_DONE) {
                throw std::runtime_error(sqlite3_errmsg(db_));
            }
        }
        execSql(db_, "COMMIT");
    } catch (...) {
        execSql(db_, "ROLLBACK");
        throw;
    }
}

void Database::updateFaceReview(int64_t faceId, const std::string& reviewState, bool ignored, const std::string& notes) {
    if (reviewState.empty()) {
        throw std::runtime_error("Review state cannot be empty.");
    }
    Statement statement(db_, "UPDATE faces SET review_state = ?, ignored = ?, notes = ? WHERE id = ?");
    sqlite3_bind_text(statement.get(), 1, reviewState.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(statement.get(), 2, ignored ? 1 : 0);
    sqlite3_bind_text(statement.get(), 3, notes.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement.get(), 4, faceId);
    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_));
    }
    if (sqlite3_changes(db_) == 0) {
        throw std::runtime_error("Face id not found: " + std::to_string(faceId));
    }
}

void Database::updateFaceMesh3d(int64_t faceId, const std::vector<std::vector<double>>& faceMesh3d) {
    if (faceMesh3d.empty()) {
        throw std::runtime_error("Face mesh cannot be empty.");
    }
    const auto meshJson = nlohmann::json(faceMesh3d).dump();
    Statement statement(db_, "UPDATE faces SET face_mesh3d_json = ? WHERE id = ?");
    sqlite3_bind_text(statement.get(), 1, meshJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement.get(), 2, faceId);
    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_));
    }
    if (sqlite3_changes(db_) == 0) {
        throw std::runtime_error("Face id not found: " + std::to_string(faceId));
    }
}

int64_t Database::insertFace(const FaceInsertRecord& record) {
    const char* sql =
        "INSERT INTO faces ("
        "file_name, source_path, embedding_blob, embedding_dim, bbox_json, kps_json, "
        "landmarks_json, landmarks3d_json, face_mesh3d_json, det_score, quality_score, "
        "quality_json, preview_png, person_id, image_hash, ignored, review_state, notes"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, 0, ?, ?)";
    Statement statement(db_, sql);

    const auto embedding = normalize(record.embedding);
    if (embedding.empty()) {
        throw std::runtime_error("Cannot insert a face without an embedding.");
    }
    const int embeddingDim = record.embeddingDim > 0 ? record.embeddingDim : static_cast<int>(embedding.size());
    if (embeddingDim != static_cast<int>(embedding.size())) {
        throw std::runtime_error("Face embedding dimension does not match the embedding size.");
    }

    const auto bboxJson = nlohmann::json(record.bbox).dump();
    const auto keypointsJson = nlohmann::json(record.keypoints).dump();
    const auto landmarks2dJson = nlohmann::json(record.landmarks2d).dump();
    const auto landmarks3dJson = nlohmann::json(record.landmarks3d).dump();
    const std::string qualityJson = record.qualityJson.empty() ? "{}" : record.qualityJson;
    const std::string reviewState = record.reviewState.empty() ? "open" : record.reviewState;

    sqlite3_bind_text(statement.get(), 1, record.fileName.c_str(), -1, SQLITE_TRANSIENT);
    if (record.sourcePath.empty()) {
        sqlite3_bind_null(statement.get(), 2);
    } else {
        sqlite3_bind_text(statement.get(), 2, record.sourcePath.c_str(), -1, SQLITE_TRANSIENT);
    }
    bindFloatVector(statement.get(), 3, embedding);
    sqlite3_bind_int(statement.get(), 4, embeddingDim);
    sqlite3_bind_text(statement.get(), 5, bboxJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement.get(), 6, keypointsJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement.get(), 7, landmarks2dJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement.get(), 8, landmarks3dJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(statement.get(), 9, record.detectionScore);
    sqlite3_bind_double(statement.get(), 10, record.qualityScore);
    sqlite3_bind_text(statement.get(), 11, qualityJson.c_str(), -1, SQLITE_TRANSIENT);
    if (record.personId > 0) {
        sqlite3_bind_int64(statement.get(), 12, record.personId);
    } else {
        sqlite3_bind_null(statement.get(), 12);
    }
    if (record.imageHash.empty()) {
        sqlite3_bind_null(statement.get(), 13);
    } else {
        sqlite3_bind_text(statement.get(), 13, record.imageHash.c_str(), -1, SQLITE_TRANSIENT);
    }
    sqlite3_bind_text(statement.get(), 14, reviewState.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement.get(), 15, record.notes.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_));
    }
    return sqlite3_last_insert_rowid(db_);
}

IdentityTrainingSummary Database::rebuildIdentityProfiles(const IdentityTrainingOptions& options) {
    IdentityTrainingSummary summary;
    const bool targeted = !options.personIds.empty();

    std::string peopleSql = "SELECT id, name FROM persons";
    if (targeted) {
        peopleSql += " WHERE id IN (" + placeholders(options.personIds.size()) + ")";
    }
    peopleSql += " ORDER BY name COLLATE NOCASE";
    Statement peopleStatement(db_, peopleSql.c_str());
    for (size_t i = 0; i < options.personIds.size(); ++i) {
        sqlite3_bind_int64(peopleStatement.get(), static_cast<int>(i + 1), options.personIds[i]);
    }
    std::vector<std::pair<int64_t, std::string>> people;
    while (sqlite3_step(peopleStatement.get()) == SQLITE_ROW) {
        people.emplace_back(sqlite3_column_int64(peopleStatement.get(), 0), textColumn(peopleStatement.get(), 1));
    }
    if (people.empty()) {
        summary.messages.push_back(targeted ? "No selected people are available for training." : "No named people are available for training.");
        return summary;
    }

    std::string facesSql =
        "SELECT f.id, f.person_id, p.name, f.embedding_blob, f.embedding_dim, COALESCE(f.quality_score, 0) "
        "FROM faces f JOIN persons p ON p.id = f.person_id "
        "WHERE f.person_id IS NOT NULL AND f.ignored = 0 AND f.quality_score >= ?";
    if (targeted) {
        facesSql += " AND f.person_id IN (" + placeholders(options.personIds.size()) + ")";
    }
    facesSql += " ORDER BY p.name COLLATE NOCASE, f.quality_score DESC, f.id ASC";
    Statement facesStatement(db_, facesSql.c_str());
    sqlite3_bind_double(facesStatement.get(), 1, options.minQuality);
    for (size_t i = 0; i < options.personIds.size(); ++i) {
        sqlite3_bind_int64(facesStatement.get(), static_cast<int>(i + 2), options.personIds[i]);
    }

    std::vector<TrainingFace> allFaces;
    while (sqlite3_step(facesStatement.get()) == SQLITE_ROW) {
        TrainingFace face;
        face.id = sqlite3_column_int64(facesStatement.get(), 0);
        face.personId = sqlite3_column_int64(facesStatement.get(), 1);
        face.personName = textColumn(facesStatement.get(), 2);
        face.embeddingDim = sqlite3_column_int(facesStatement.get(), 4);
        face.embedding = normalize(floatBlob(facesStatement.get(), 3, face.embeddingDim));
        face.quality = sqlite3_column_double(facesStatement.get(), 5);
        if (!face.embedding.empty()) {
            allFaces.push_back(std::move(face));
        }
    }

    execSql(db_, "BEGIN IMMEDIATE");
    try {
        if (targeted) {
            std::string deleteSql = "DELETE FROM person_identity_profiles WHERE person_id IN (" + placeholders(options.personIds.size()) + ")";
            Statement deleteStatement(db_, deleteSql.c_str());
            for (size_t i = 0; i < options.personIds.size(); ++i) {
                sqlite3_bind_int64(deleteStatement.get(), static_cast<int>(i + 1), options.personIds[i]);
            }
            if (sqlite3_step(deleteStatement.get()) != SQLITE_DONE) {
                throw std::runtime_error(sqlite3_errmsg(db_));
            }
        } else {
            execSql(db_, "DELETE FROM person_identity_profiles");
        }

        const char* saveSql =
            "INSERT OR REPLACE INTO person_identity_profiles ("
            "person_id, sample_count, prototype_count, embedding_dim, centroid_blob, prototypes_blob, "
            "exemplar_blob, exemplar_face_ids_json, exemplar_weights_blob, hard_negative_face_ids_json, "
            "thresholds_json, calibration_json, strategy_version, scoring_model_version, accept_threshold, "
            "review_threshold, mean_similarity, min_similarity, max_similarity, quality_mean, "
            "evidence_face_ids_json, status, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

        for (const auto& [personId, personName] : people) {
            std::vector<TrainingFace> faces;
            std::vector<TrainingFace> negatives;
            for (const auto& face : allFaces) {
                if (face.personId == personId) {
                    faces.push_back(face);
                } else {
                    negatives.push_back(face);
                }
            }
            if (faces.empty()) {
                ++summary.skippedPeople;
                std::ostringstream message;
                message << personName << ": skipped, no usable assigned faces above quality " << std::fixed << std::setprecision(2) << options.minQuality << ".";
                summary.messages.push_back(message.str());
                continue;
            }

            auto profile = buildIdentityProfileFromFaces(personId, personName, faces, negatives, options.maxExemplars);
            if (profile.centroid.empty() || profile.exemplars.empty()) {
                ++summary.skippedPeople;
                summary.messages.push_back(personName + ": skipped, embeddings could not be normalized.");
                continue;
            }

            const auto prototypes = flattenRows(profile.prototypes);
            const auto exemplars = flattenRows(profile.exemplars);
            const auto thresholds = thresholdsJson(profile).dump();
            const auto calibration = nlohmann::json{
                {"positive_count", std::max(0, profile.sampleCount - 1)},
                {"negative_count", negatives.size()},
                {"positive_mean", profile.meanSimilarity},
                {"negative_max", nullptr},
                {"strict_gap", nullptr},
                {"health", profile.health},
            }.dump();
            const auto exemplarFaceIds = nlohmann::json(profile.exemplarFaceIds).dump();
            const auto hardNegativeFaceIds = nlohmann::json(profile.hardNegativeFaceIds).dump();
            const auto evidenceFaceIds = nlohmann::json(profile.evidenceFaceIds).dump();

            Statement saveStatement(db_, saveSql);
            sqlite3_bind_int64(saveStatement.get(), 1, profile.personId);
            sqlite3_bind_int(saveStatement.get(), 2, profile.sampleCount);
            sqlite3_bind_int(saveStatement.get(), 3, profile.prototypeCount);
            sqlite3_bind_int(saveStatement.get(), 4, profile.embeddingDim);
            bindFloatVector(saveStatement.get(), 5, profile.centroid);
            bindFloatVector(saveStatement.get(), 6, prototypes);
            bindFloatVector(saveStatement.get(), 7, exemplars);
            sqlite3_bind_text(saveStatement.get(), 8, exemplarFaceIds.c_str(), -1, SQLITE_TRANSIENT);
            bindFloatVector(saveStatement.get(), 9, profile.exemplarWeights);
            sqlite3_bind_text(saveStatement.get(), 10, hardNegativeFaceIds.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 11, thresholds.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 12, calibration.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 13, profile.strategyVersion.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 14, profile.scoringModelVersion.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(saveStatement.get(), 15, profile.acceptThreshold);
            sqlite3_bind_double(saveStatement.get(), 16, profile.reviewThreshold);
            sqlite3_bind_double(saveStatement.get(), 17, profile.meanSimilarity);
            sqlite3_bind_double(saveStatement.get(), 18, profile.minSimilarity);
            sqlite3_bind_double(saveStatement.get(), 19, profile.maxSimilarity);
            sqlite3_bind_double(saveStatement.get(), 20, profile.qualityMean);
            sqlite3_bind_text(saveStatement.get(), 21, evidenceFaceIds.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 22, profile.status.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(saveStatement.get(), 23, profile.updatedAt.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(saveStatement.get()) != SQLITE_DONE) {
                throw std::runtime_error(sqlite3_errmsg(db_));
            }

            ++summary.profilesBuilt;
            summary.samplesUsed += profile.sampleCount;
            if (profile.status == "weak") {
                ++summary.weakProfiles;
            }
            std::ostringstream message;
            message << personName << ": " << profile.sampleCount << " sample(s), "
                    << profile.prototypeCount << " exemplar(s), " << profile.status
                    << ", health " << profile.health
                    << ", strict accept " << std::fixed << std::setprecision(3) << profile.acceptThreshold << ".";
            summary.messages.push_back(message.str());
        }
        execSql(db_, "COMMIT");
    } catch (...) {
        execSql(db_, "ROLLBACK");
        throw;
    }

    return summary;
}

} // namespace fsc::core
