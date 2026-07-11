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
    [[nodiscard]] std::vector<FaceRecord> loadFacesForPerson(int64_t personId, bool includeIgnored = true) const;
    [[nodiscard]] std::optional<FaceRecord> loadFace(int64_t faceId) const;
    [[nodiscard]] std::optional<FaceRecord> loadFacePreview(int64_t faceId) const;
    [[nodiscard]] std::vector<PersonSummary> loadPeople() const;
    [[nodiscard]] std::vector<std::string> loadTags() const;
    [[nodiscard]] std::vector<TagSummary> loadTagSummaries(int limit = 0) const;
    [[nodiscard]] std::vector<IdentityProfile> loadIdentityProfiles() const;
    [[nodiscard]] bool imageHashExists(const std::string& imageHash) const;
    [[nodiscard]] MaintenanceResult checkIntegrity() const;
    [[nodiscard]] MaintenanceResult backupTo(const std::filesystem::path& outputPath) const;
    [[nodiscard]] MaintenanceResult checkpointWal(bool truncate = true);
    [[nodiscard]] MaintenanceResult vacuum();
    int64_t upsertPerson(const std::string& name, const std::string& notes = {});
    void renamePerson(int64_t personId, const std::string& name, const std::string& notes = {});
    int mergePeople(int64_t sourcePersonId, int64_t targetPersonId);
    int clearPersonAssignment(int64_t personId, bool deletePerson = true);
    void assignFaceToPerson(int64_t faceId, int64_t personId);
    void setFaceTags(int64_t faceId, const std::string& tagText, bool append = false);
    void updateFaceReview(int64_t faceId, const std::string& reviewState, bool ignored, const std::string& notes = {});
    void updateFaceMesh3d(int64_t faceId, const std::vector<std::vector<double>>& faceMesh3d);
    void clearFaceMesh3d(int64_t faceId);
    int64_t insertFace(const FaceInsertRecord& record);
    std::vector<int64_t> insertFaces(const std::vector<FaceInsertRecord>& records);
    IdentityTrainingSummary rebuildIdentityProfiles(const IdentityTrainingOptions& options = {});

private:
    std::filesystem::path path_;
    sqlite3* db_ = nullptr;
};

} // namespace fsc::core
