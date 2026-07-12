#include "fsc/core/PathEncoding.hpp"

#include <string>

namespace fsc::core {

std::string pathToUtf8(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return {
        reinterpret_cast<const char*>(encoded.data()),
        encoded.size()};
}

std::filesystem::path pathFromUtf8(std::string_view value) {
#if defined(__cpp_char8_t)
    std::u8string encoded;
    encoded.assign(
        reinterpret_cast<const char8_t*>(value.data()),
        reinterpret_cast<const char8_t*>(value.data() + value.size()));
    return std::filesystem::path(encoded);
#else
    return std::filesystem::u8path(value.begin(), value.end());
#endif
}

std::filesystem::path pathWithSuffix(
    const std::filesystem::path& path,
    std::string_view suffix) {
    auto output = path;
    output += pathFromUtf8(suffix).native();
    return output;
}

} // namespace fsc::core
