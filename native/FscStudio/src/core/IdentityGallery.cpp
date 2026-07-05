#include "fsc/core/IdentityGallery.hpp"

#include "fsc/core/VectorMath.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fsc::core {
namespace {

double clamp(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

const IdentityThresholds& thresholdsFor(const IdentityProfile& profile, IdentityMode mode) {
    switch (mode) {
    case IdentityMode::Balanced:
        return profile.balanced;
    case IdentityMode::Broad:
        return profile.broad;
    case IdentityMode::Strict:
    default:
        return profile.strict;
    }
}

std::vector<double> solveLinear(std::vector<std::vector<double>> a, std::vector<double> b) {
    const size_t n = b.size();
    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        const double divisor = std::abs(a[col][col]) < 1e-9 ? 1e-9 : a[col][col];
        for (size_t j = col; j < n; ++j) {
            a[col][j] /= divisor;
        }
        b[col] /= divisor;
        for (size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[row][col];
            for (size_t j = col; j < n; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    return b;
}

double ridgeReconstructionScore(const std::vector<std::vector<float>>& exemplars, std::span<const float> query) {
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

    auto weights = solveLinear(std::move(gram), std::move(rhs));
    for (double& weight : weights) {
        weight = std::max(0.0, weight);
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
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
            reconstruction[dim] += static_cast<float>((weights[row] / total) * exemplars[row][dim]);
        }
    }
    reconstruction = normalize(reconstruction);
    return dot(reconstruction, query);
}

double hardNegativePenalty(
    const std::vector<std::vector<float>>& hardNegatives,
    std::span<const float> query,
    double profileScore) {
    if (hardNegatives.empty()) {
        return 0.0;
    }
    double hardBest = -1.0;
    for (const auto& negative : hardNegatives) {
        if (negative.size() == query.size()) {
            hardBest = std::max(hardBest, dot(negative, query));
        }
    }
    return clamp((hardBest - profileScore + 0.035) * 0.60, 0.0, 0.12);
}

struct ScoreParts {
    double score = 0.0;
    int bestIndex = -1;
};

ScoreParts scoreProfile(const IdentityProfile& profile, std::span<const float> query) {
    const auto& exemplars = profile.exemplars.empty() ? profile.prototypes : profile.exemplars;
    std::vector<double> exemplarScores;
    exemplarScores.reserve(exemplars.size());
    for (const auto& exemplar : exemplars) {
        exemplarScores.push_back(exemplar.size() == query.size() ? dot(exemplar, query) : -1.0);
    }

    int bestIndex = 0;
    if (!exemplarScores.empty()) {
        bestIndex = static_cast<int>(std::distance(
            exemplarScores.begin(),
            std::max_element(exemplarScores.begin(), exemplarScores.end())));
    }

    const double centroidScore = profile.centroid.size() == query.size() ? dot(profile.centroid, query) : 0.0;
    const double best = exemplarScores.empty() ? centroidScore : exemplarScores[static_cast<size_t>(bestIndex)];

    auto sortedScores = exemplarScores;
    std::sort(sortedScores.begin(), sortedScores.end());
    const size_t topCount = std::min<size_t>(3, sortedScores.size());
    double topMean = best;
    if (topCount > 0) {
        topMean = std::accumulate(sortedScores.end() - static_cast<std::ptrdiff_t>(topCount), sortedScores.end(), 0.0)
            / static_cast<double>(topCount);
    }

    double weightScore = best;
    if (profile.exemplarWeights.size() == exemplarScores.size() && !exemplarScores.empty()) {
        weightScore = 0.0;
        for (size_t i = 0; i < exemplarScores.size(); ++i) {
            weightScore += exemplarScores[i] * static_cast<double>(profile.exemplarWeights[i]);
        }
    }

    const double reconstruction = ridgeReconstructionScore(exemplars, query);
    const double penalty = hardNegativePenalty(profile.hardNegativeEmbeddings, query, std::max(best, centroidScore));
    const double fallback =
        best * 0.32 +
        topMean * 0.20 +
        centroidScore * 0.20 +
        reconstruction * 0.20 +
        weightScore * 0.08 -
        penalty;

    return {fallback, bestIndex};
}

double confidence(const IdentityProfile& profile, double score, double margin, IdentityMode mode) {
    const auto& thresholds = thresholdsFor(profile, mode);
    const double span = std::max(0.04, thresholds.accept - thresholds.review);
    const double scoreConfidence = clamp((score - thresholds.review) / span, 0.0, 1.0);
    const double marginConfidence = clamp(margin / std::max(thresholds.margin * 2.0, 1e-6), 0.0, 1.0);
    if (profile.status == "weak") {
        return std::min(0.69, scoreConfidence * 0.75 + marginConfidence * 0.15);
    }
    return clamp(scoreConfidence * 0.68 + marginConfidence * 0.32, 0.0, 1.0);
}

} // namespace

const char* modeName(IdentityMode mode) {
    switch (mode) {
    case IdentityMode::Balanced:
        return "balanced";
    case IdentityMode::Broad:
        return "broad";
    case IdentityMode::Strict:
    default:
        return "strict";
    }
}

IdentityResult identifyPerson(
    const std::vector<IdentityProfile>& profiles,
    std::span<const float> queryEmbedding,
    IdentityMode mode,
    int topK) {
    if (profiles.empty()) {
        return {"unknown", {}, "No trained identity profiles."};
    }

    const auto query = normalize(queryEmbedding);
    if (query.empty()) {
        return {"unknown", {}, "Query embedding is empty."};
    }

    std::vector<IdentityCandidate> scored;
    for (const auto& profile : profiles) {
        if (profile.embeddingDim != static_cast<int>(query.size())) {
            continue;
        }
        const auto parts = scoreProfile(profile, query);
        IdentityCandidate candidate;
        candidate.profile = profile;
        candidate.score = parts.score;
        candidate.bestExemplarIndex = parts.bestIndex;
        if (parts.bestIndex >= 0 && static_cast<size_t>(parts.bestIndex) < profile.exemplarFaceIds.size()) {
            candidate.evidenceFaceId = profile.exemplarFaceIds[static_cast<size_t>(parts.bestIndex)];
        } else if (!profile.evidenceFaceIds.empty()) {
            candidate.evidenceFaceId = profile.evidenceFaceIds.front();
        }
        scored.push_back(std::move(candidate));
    }

    std::sort(scored.begin(), scored.end(), [](const IdentityCandidate& left, const IdentityCandidate& right) {
        return left.score > right.score;
    });
    if (scored.empty()) {
        return {"unknown", {}, "No identity profile matches the query embedding dimensions."};
    }

    if (topK <= 0) {
        topK = 1;
    }
    if (scored.size() > static_cast<size_t>(topK)) {
        scored.resize(static_cast<size_t>(topK));
    }
    for (size_t i = 0; i < scored.size(); ++i) {
        const double nextScore = i + 1 < scored.size()
            ? scored[i + 1].score
            : thresholdsFor(scored[i].profile, mode).review;
        scored[i].margin = scored[i].score - nextScore;
        scored[i].confidence = confidence(scored[i].profile, scored[i].score, scored[i].margin, mode);
    }

    const auto& best = scored.front();
    const auto& thresholds = thresholdsFor(best.profile, mode);
    if (best.score < thresholds.review) {
        return {"unknown", scored, "No person passes the review threshold."};
    }
    if (best.profile.status == "weak") {
        return {"review", scored, "Best profile is weak; manual confirmation is required."};
    }
    if (best.score >= thresholds.accept && best.margin >= thresholds.margin) {
        return {"confirmed", scored, std::string("Identity confirmed by ") + modeName(mode) + " gallery profile."};
    }
    return {"review", scored, "Identity is plausible but needs review."};
}

} // namespace fsc::core
