#pragma once

#include "fsc/core/Models.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace fsc::core {

class Database {
public:
    explicit Database(std::filesystem::path path);
    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&&) noexcept;
    Database& operator=(Database&&) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::map<std::string, std::string> metadata() const;
    [[nodiscard]] DatabaseStatistics statistics() const;
    [[nodiscard]] std::vector<FaceRecord> loadFaces(bool includeIgnored = true, int limit = 0) const;
    [[nodiscard]] std::optional<FaceRecord> loadFace(int64_t faceId) const;
    [[nodiscard]] std::vector<PersonSummary> loadPeople() const;
    [[nodiscard]] std::vector<IdentityProfile> loadIdentityProfiles() const;

private:
    std::filesystem::path path_;
    sqlite3* db_ = nullptr;
};

} // namespace fsc::core
