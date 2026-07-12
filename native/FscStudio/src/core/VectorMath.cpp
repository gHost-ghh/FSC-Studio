#include "fsc/core/VectorMath.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fsc::core {

double dot(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match.");
    }
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        value += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return value;
}

double norm(std::span<const float> values) {
    double total = 0.0;
    for (const float value : values) {
        total += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::sqrt(total);
}

std::vector<float> normalize(std::span<const float> values) {
    std::vector<float> output(values.begin(), values.end());
    const double length = norm(values);
    if (length <= 1e-12) {
        std::fill(output.begin(), output.end(), 0.0f);
        return output;
    }
    for (float& value : output) {
        value = static_cast<float>(static_cast<double>(value) / length);
    }
    return output;
}

double cosineSimilarity(std::span<const float> a, std::span<const float> b) {
    return dot(normalize(a), normalize(b));
}

} // namespace fsc::core
