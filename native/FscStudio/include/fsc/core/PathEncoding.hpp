#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace fsc::core {

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path pathFromUtf8(std::string_view value);
[[nodiscard]] std::filesystem::path pathWithSuffix(
    const std::filesystem::path& path,
    std::string_view suffix);

} // namespace fsc::core
