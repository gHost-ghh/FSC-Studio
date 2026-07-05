#include "fsc/core/FileHash.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

namespace fsc::core {
namespace {

#ifdef _WIN32
void throwIfFailed(NTSTATUS status, const char* message) {
    if (status < 0) {
        throw std::runtime_error(message);
    }
}

struct AlgorithmHandle {
    ~AlgorithmHandle() {
        if (handle != nullptr) {
            BCryptCloseAlgorithmProvider(handle, 0);
        }
    }

    BCRYPT_ALG_HANDLE handle = nullptr;
};

struct HashHandle {
    ~HashHandle() {
        if (handle != nullptr) {
            BCryptDestroyHash(handle);
        }
    }

    BCRYPT_HASH_HANDLE handle = nullptr;
};

std::string hexLower(const std::vector<unsigned char>& bytes) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : bytes) {
        stream << std::setw(2) << static_cast<int>(byte);
    }
    return stream.str();
}
#endif

} // namespace

std::string sha256File(const std::filesystem::path& path) {
#ifdef _WIN32
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for SHA-256: " + path.string());
    }

    AlgorithmHandle algorithm;
    throwIfFailed(
        BCryptOpenAlgorithmProvider(&algorithm.handle, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
        "Failed to open SHA-256 provider.");

    DWORD objectLength = 0;
    DWORD bytesWritten = 0;
    throwIfFailed(
        BCryptGetProperty(
            algorithm.handle,
            BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&objectLength),
            sizeof(objectLength),
            &bytesWritten,
            0),
        "Failed to read SHA-256 object length.");

    DWORD hashLength = 0;
    throwIfFailed(
        BCryptGetProperty(
            algorithm.handle,
            BCRYPT_HASH_LENGTH,
            reinterpret_cast<PUCHAR>(&hashLength),
            sizeof(hashLength),
            &bytesWritten,
            0),
        "Failed to read SHA-256 hash length.");

    std::vector<unsigned char> objectBuffer(objectLength);
    HashHandle hash;
    throwIfFailed(
        BCryptCreateHash(
            algorithm.handle,
            &hash.handle,
            objectBuffer.data(),
            static_cast<ULONG>(objectBuffer.size()),
            nullptr,
            0,
            0),
        "Failed to create SHA-256 hash.");

    std::vector<char> buffer(1024 * 1024);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = file.gcount();
        if (count > 0) {
            throwIfFailed(
                BCryptHashData(
                    hash.handle,
                    reinterpret_cast<PUCHAR>(buffer.data()),
                    static_cast<ULONG>(count),
                    0),
                "Failed to update SHA-256 hash.");
        }
    }

    std::vector<unsigned char> digest(hashLength);
    throwIfFailed(
        BCryptFinishHash(hash.handle, digest.data(), static_cast<ULONG>(digest.size()), 0),
        "Failed to finish SHA-256 hash.");
    return hexLower(digest);
#else
    (void)path;
    throw std::runtime_error("SHA-256 file hashing is only implemented for Windows builds.");
#endif
}

} // namespace fsc::core
