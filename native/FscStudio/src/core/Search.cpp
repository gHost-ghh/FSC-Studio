#include "fsc/core/Search.hpp"

#include "fsc/core/VectorMath.hpp"

#include <algorithm>

namespace fsc::core {

std::vector<SearchHit> searchFaces(
    const std::vector<FaceRecord>& records,
    std::span<const float> queryEmbedding,
    int topK,
    double threshold,
    bool includeIgnored) {
    const auto query = normalize(queryEmbedding);
    std::vector<SearchHit> hits;
    for (const auto& record : records) {
        if (!includeIgnored && record.ignored) {
            continue;
        }
        if (record.embedding.empty() || record.embedding.size() != query.size()) {
            continue;
        }
        const double score = dot(record.embedding, query);
        if (score >= threshold) {
            hits.push_back({record, score});
        }
    }
    std::sort(hits.begin(), hits.end(), [](const SearchHit& left, const SearchHit& right) {
        return left.cosine > right.cosine;
    });
    if (topK > 0 && hits.size() > static_cast<size_t>(topK)) {
        hits.resize(static_cast<size_t>(topK));
    }
    return hits;
}

} // namespace fsc::core
