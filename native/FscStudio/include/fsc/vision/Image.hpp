#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace fsc::vision {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool empty() const noexcept {
        return width <= 0 || height <= 0 || pixels.size() != static_cast<size_t>(width * height * 3);
    }
};

RgbImage loadPpmRgb(const std::filesystem::path& path);
RgbImage loadImageRgb(const std::filesystem::path& path);
RgbImage resizeBilinear(const RgbImage& image, int width, int height);
RgbImage letterboxToSquare(const RgbImage& image, int size, float& scale);
std::uint8_t sampleBilinearChannel(const RgbImage& image, float x, float y, int channel);

} // namespace fsc::vision
