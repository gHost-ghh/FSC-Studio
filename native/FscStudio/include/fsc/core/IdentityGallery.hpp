#pragma once

#include "fsc/core/Models.hpp"

#include <span>
#include <vector>

namespace fsc::core {

IdentityResult identifyPerson(
    const std::vector<IdentityProfile>& profiles,
    std::span<const float> queryEmbedding,
    IdentityMode mode = IdentityMode::Strict,
    int topK = 5);

const char* modeName(IdentityMode mode);

} // namespace fsc::core
