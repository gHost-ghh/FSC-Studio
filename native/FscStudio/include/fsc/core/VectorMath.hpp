#pragma once

#include <span>
#include <vector>

namespace fsc::core {

double dot(std::span<const float> a, std::span<const float> b);
double norm(std::span<const float> values);
std::vector<float> normalize(std::span<const float> values);
double cosineSimilarity(std::span<const float> a, std::span<const float> b);

} // namespace fsc::core
