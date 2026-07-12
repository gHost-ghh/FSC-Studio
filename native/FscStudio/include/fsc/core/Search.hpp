#pragma once

#include "fsc/core/Models.hpp"

#include <span>
#include <vector>

namespace fsc::core {

std::vector<SearchHit> searchFaces(
    const std::vector<FaceRecord>& records,
    std::span<const float> queryEmbedding,
    int topK = 30,
    double threshold = -1.0,
    bool includeIgnored = false);

} // namespace fsc::core
