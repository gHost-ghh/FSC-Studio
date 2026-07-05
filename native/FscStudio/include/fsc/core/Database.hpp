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

    static void createEmpty(std::filesystem::path path, bool replace = true);

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
    [[nodiscard]] bool imageHashExists(const std::string& imageHash) const;
    int64_t upsertPerson(const std::string& name, const std::string& notes = {});
    void assignFaceToPerson(int64_t faceId, int64_t personId);
    int64_t insertFace(const FaceInsertRecord& record);
    IdentityTrainingSummary rebuildIdentityProfiles(const IdentityTrainingOptions& options = {});

private:
    std::filesystem::path path_;
    sqlite3* db_ = nullptr;
};

} // namespace fsc::core
