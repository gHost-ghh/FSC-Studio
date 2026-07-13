#include "fsc/vision/Image.hpp"

#include "fsc/core/PathEncoding.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <combaseapi.h>
#include <wincodec.h>
#include <wrl/client.h>
#endif

namespace fsc::vision {
namespace {

std::string nextToken(std::istream& stream) {
    std::string token;
    char ch = 0;
    while (stream.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            continue;
        }
        if (ch == '#') {
            std::string ignored;
            std::getline(stream, ignored);
            continue;
        }
        token.push_back(ch);
        break;
    }
    while (stream.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        token.push_back(ch);
    }
    return token;
}

std::uint8_t clampByte(float value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 255.0f));
}

#ifdef _WIN32
void throwIfFailed(HRESULT result, const char* message) {
    if (FAILED(result)) {
        throw std::runtime_error(message);
    }
}

struct ComInitializer {
    ComInitializer() {
        result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(result) && result != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("Failed to initialize COM for Windows image loading.");
        }
    }

    ~ComInitializer() {
        if (SUCCEEDED(result)) {
            CoUninitialize();
        }
    }

    HRESULT result = S_OK;
};
#endif

} // namespace

RgbImage loadPpmRgb(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open PPM image: " + fsc::core::pathToUtf8(path));
    }

    const auto magic = nextToken(stream);
    if (magic != "P6") {
        throw std::runtime_error("Only binary P6 PPM images are supported by the native probe.");
    }
    const int width = std::stoi(nextToken(stream));
    const int height = std::stoi(nextToken(stream));
    const int maxValue = std::stoi(nextToken(stream));
    if (width <= 0 || height <= 0 || maxValue != 255) {
        throw std::runtime_error("Unsupported PPM dimensions or max value.");
    }

    RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width * height * 3));
    stream.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (stream.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("PPM image ended before all pixels were read.");
    }
    return image;
}

RgbImage loadImageRgb(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    std::string lowerExtension;
    lowerExtension.reserve(extension.size());
    for (const char ch : extension) {
        lowerExtension.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lowerExtension == ".ppm") {
        return loadPpmRgb(path);
    }

#ifdef _WIN32
    ComInitializer com;
    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    throwIfFailed(
        CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(factory.GetAddressOf())),
        "Failed to create Windows Imaging Component factory.");

    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    throwIfFailed(
        factory->CreateDecoderFromFilename(
            path.wstring().c_str(),
            nullptr,
            GENERIC_READ,
            WICDecodeMetadataCacheOnDemand,
            decoder.GetAddressOf()),
        "Failed to open image through Windows Imaging Component.");

    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    throwIfFailed(decoder->GetFrame(0, frame.GetAddressOf()), "Failed to read first image frame.");

    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    throwIfFailed(factory->CreateFormatConverter(converter.GetAddressOf()), "Failed to create WIC format converter.");
    throwIfFailed(
        converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom),
        "Failed to convert image to RGB pixels.");

    UINT width = 0;
    UINT height = 0;
    throwIfFailed(converter->GetSize(&width, &height), "Failed to read image dimensions.");
    if (width == 0 || height == 0 || width > static_cast<UINT>(std::numeric_limits<int>::max()) || height > static_cast<UINT>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Unsupported image dimensions.");
    }

    RgbImage image;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    const UINT stride = width * 3;
    image.pixels.resize(static_cast<size_t>(stride) * height);
    throwIfFailed(
        converter->CopyPixels(nullptr, stride, static_cast<UINT>(image.pixels.size()), image.pixels.data()),
        "Failed to copy image pixels.");
    return image;
#else
    throw std::runtime_error("Only PPM image loading is available on this platform.");
#endif
}

RgbImage resizeBilinear(const RgbImage& image, int width, int height) {
    if (image.empty()) {
        throw std::runtime_error("Cannot resize an empty RGB image.");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Resize target dimensions must be positive.");
    }

    RgbImage output;
    output.width = width;
    output.height = height;
    output.pixels.resize(static_cast<size_t>(width * height * 3));

    const float scaleX = image.width / static_cast<float>(width);
    const float scaleY = image.height / static_cast<float>(height);
    for (int y = 0; y < height; ++y) {
        const float srcY = (y + 0.5f) * scaleY - 0.5f;
        for (int x = 0; x < width; ++x) {
            const float srcX = (x + 0.5f) * scaleX - 0.5f;
            const size_t offset = static_cast<size_t>((y * width + x) * 3);
            for (int channel = 0; channel < 3; ++channel) {
                output.pixels[offset + static_cast<size_t>(channel)] =
                    sampleBilinearChannel(image, srcX, srcY, channel);
            }
        }
    }
    return output;
}

RgbImage letterboxToSquare(const RgbImage& image, int size, float& scale) {
    if (image.empty()) {
        throw std::runtime_error("Cannot letterbox an empty RGB image.");
    }
    if (size <= 0) {
        throw std::runtime_error("Letterbox size must be positive.");
    }

    const float imageRatio = static_cast<float>(image.height) / image.width;
    int resizedWidth = 0;
    int resizedHeight = 0;
    if (imageRatio > 1.0f) {
        resizedHeight = size;
        resizedWidth = std::max(1, static_cast<int>(size / imageRatio));
        scale = static_cast<float>(resizedHeight) / image.height;
    } else {
        resizedWidth = size;
        resizedHeight = std::max(1, static_cast<int>(size * imageRatio));
        scale = static_cast<float>(resizedWidth) / image.width;
    }
    const auto resized = resizeBilinear(image, resizedWidth, resizedHeight);

    RgbImage output;
    output.width = size;
    output.height = size;
    output.pixels.assign(static_cast<size_t>(size * size * 3), 0);
    for (int y = 0; y < resized.height; ++y) {
        const auto* source = &resized.pixels[static_cast<size_t>(y * resized.width * 3)];
        auto* target = &output.pixels[static_cast<size_t>(y * size * 3)];
        std::copy(source, source + resized.width * 3, target);
    }
    return output;
}

std::uint8_t sampleBilinearChannel(const RgbImage& image, float x, float y, int channel) {
    if (image.empty() || channel < 0 || channel >= 3) {
        return 0;
    }
    x = std::clamp(x, 0.0f, static_cast<float>(image.width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(image.height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    const auto at = [&](int px, int py) -> float {
        return static_cast<float>(image.pixels[static_cast<size_t>((py * image.width + px) * 3 + channel)]);
    };
    const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * dx;
    const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * dx;
    return clampByte(top + (bottom - top) * dy);
}

} // namespace fsc::vision
