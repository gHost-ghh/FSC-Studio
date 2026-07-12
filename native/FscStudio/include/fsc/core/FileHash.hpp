#pragma once

#include <filesystem>
#include <string>

namespace fsc::core {

std::string sha256File(const std::filesystem::path& path);

} // namespace fsc::core
