#include "fsc/legacy/LegacyDtb.hpp"

#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/PathEncoding.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace fsc::legacy {
namespace {

constexpr size_t kMaxPickleBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kMaxContainerItems = 2ULL * 1024ULL * 1024ULL;
constexpr size_t kMaxImagePixels = 120ULL * 1024ULL * 1024ULL;

enum class ValueKind {
    Mark,
    None,
    Boolean,
    Integer,
    Floating,
    String,
    Bytes,
    List,
    Tuple,
    Dictionary,
    Global,
    Object,
    Dtype,
    NumpyArray,
};

struct NumpyArrayData {
    std::vector<int64_t> shape;
    std::string dtype;
    bool fortranOrder = false;
    std::vector<std::uint8_t> bytes;
};

struct PickleValue;
using ValuePtr = std::shared_ptr<PickleValue>;

struct PickleValue {
    ValueKind kind = ValueKind::None;
    bool boolean = false;
    int64_t integer = 0;
    double floating = 0.0;
    std::string text;
    std::vector<std::uint8_t> bytes;
    std::vector<ValuePtr> items;
    std::vector<std::pair<ValuePtr, ValuePtr>> dictionary;
    std::string globalModule;
    std::string globalName;
    std::optional<NumpyArrayData> array;
};

ValuePtr makeValue(ValueKind kind) {
    auto value = std::make_shared<PickleValue>();
    value->kind = kind;
    return value;
}

class Reader {
public:
    explicit Reader(std::vector<std::uint8_t> data) : data_(std::move(data)) {}

    [[nodiscard]] bool atEnd() const noexcept { return position_ >= data_.size(); }

    std::uint8_t readByte() {
        require(1);
        return data_[position_++];
    }

    uint16_t readU16() {
        require(2);
        const auto value = static_cast<uint16_t>(data_[position_]) |
            static_cast<uint16_t>(data_[position_ + 1]) << 8U;
        position_ += 2;
        return value;
    }

    uint32_t readU32() {
        require(4);
        uint32_t value = 0;
        for (int offset = 0; offset < 4; ++offset) {
            value |= static_cast<uint32_t>(data_[position_ + static_cast<size_t>(offset)]) << (offset * 8U);
        }
        position_ += 4;
        return value;
    }

    uint64_t readU64() {
        require(8);
        uint64_t value = 0;
        for (int offset = 0; offset < 8; ++offset) {
            value |= static_cast<uint64_t>(data_[position_ + static_cast<size_t>(offset)]) << (offset * 8U);
        }
        position_ += 8;
        return value;
    }

    std::vector<std::uint8_t> readBytes(size_t count) {
        require(count);
        std::vector<std::uint8_t> value(
            data_.begin() + static_cast<std::ptrdiff_t>(position_),
            data_.begin() + static_cast<std::ptrdiff_t>(position_ + count));
        position_ += count;
        return value;
    }

    std::string readString(size_t count) {
        const auto bytes = readBytes(count);
        return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    }

    std::string readLine() {
        const size_t start = position_;
        while (!atEnd() && data_[position_] != '\n') {
            ++position_;
        }
        if (atEnd()) {
            throw std::runtime_error("Legacy DTB pickle ended inside a line opcode.");
        }
        const std::string value(
            reinterpret_cast<const char*>(data_.data() + start),
            position_ - start);
        ++position_;
        return value;
    }

private:
    void require(size_t count) const {
        if (count > data_.size() - position_) {
            throw std::runtime_error("Legacy DTB pickle ended unexpectedly.");
        }
    }

    std::vector<std::uint8_t> data_;
    size_t position_ = 0;
};

bool isSequence(const ValuePtr& value) {
    return value != nullptr && (value->kind == ValueKind::List || value->kind == ValueKind::Tuple);
}

const std::vector<ValuePtr>& sequenceItems(const ValuePtr& value, const char* context) {
    if (!isSequence(value)) {
        throw std::runtime_error(std::string("Legacy DTB expected ") + context + " to be a sequence.");
    }
    return value->items;
}

int64_t asInteger(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::Integer) {
        throw std::runtime_error(std::string("Legacy DTB expected ") + context + " to be an integer.");
    }
    return value->integer;
}

bool asBoolean(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::Boolean) {
        throw std::runtime_error(std::string("Legacy DTB expected ") + context + " to be a boolean.");
    }
    return value->boolean;
}

std::string asString(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::String) {
        throw std::runtime_error(std::string("Legacy DTB expected ") + context + " to be text.");
    }
    return value->text;
}

const std::vector<std::uint8_t>& asBytes(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::Bytes) {
        throw std::runtime_error(std::string("Legacy DTB expected ") + context + " to be binary data.");
    }
    return value->bytes;
}

bool isNumpyReconstruct(const ValuePtr& callable) {
    return callable != nullptr && callable->kind == ValueKind::Global &&
        (callable->globalModule == "numpy.core.multiarray" || callable->globalModule == "numpy._core.multiarray") &&
        callable->globalName == "_reconstruct";
}

bool isNumpyDtype(const ValuePtr& callable) {
    return callable != nullptr && callable->kind == ValueKind::Global &&
        callable->globalModule == "numpy" && callable->globalName == "dtype";
}

class RestrictedPickleParser {
public:
    explicit RestrictedPickleParser(std::vector<std::uint8_t> data) : reader_(std::move(data)) {}

    ValuePtr parse() {
        while (!reader_.atEnd()) {
            const auto opcode = reader_.readByte();
            switch (opcode) {
            case 0x80: // PROTO
                if (reader_.readByte() > 5) {
                    throw std::runtime_error("Legacy DTB uses an unsupported pickle protocol.");
                }
                break;
            case 0x95: { // FRAME
                const auto frameSize = reader_.readU64();
                if (frameSize > kMaxPickleBytes) {
                    throw std::runtime_error("Legacy DTB pickle frame exceeds the safety limit.");
                }
                break;
            }
            case '.': // STOP
                if (stack_.empty()) {
                    throw std::runtime_error("Legacy DTB pickle stopped with an empty stack.");
                }
                return stack_.back();
            case '(': // MARK
                push(makeValue(ValueKind::Mark));
                break;
            case 'N':
                push(makeValue(ValueKind::None));
                break;
            case 0x88:
            case 0x89: {
                auto value = makeValue(ValueKind::Boolean);
                value->boolean = opcode == 0x88;
                push(std::move(value));
                break;
            }
            case 'I':
            case 'L': {
                const auto text = reader_.readLine();
                auto value = makeValue(ValueKind::Integer);
                if (text == "01") {
                    value->kind = ValueKind::Boolean;
                    value->boolean = true;
                } else if (text == "00") {
                    value->kind = ValueKind::Boolean;
                    value->boolean = false;
                } else {
                    const auto numeric = text.ends_with('L') ? text.substr(0, text.size() - 1) : text;
                    value->integer = std::stoll(numeric);
                }
                push(std::move(value));
                break;
            }
            case 'J': {
                auto value = makeValue(ValueKind::Integer);
                value->integer = static_cast<int32_t>(reader_.readU32());
                push(std::move(value));
                break;
            }
            case 'K': {
                auto value = makeValue(ValueKind::Integer);
                value->integer = reader_.readByte();
                push(std::move(value));
                break;
            }
            case 'M': {
                auto value = makeValue(ValueKind::Integer);
                value->integer = reader_.readU16();
                push(std::move(value));
                break;
            }
            case 0x8a:
                pushSignedLong(reader_.readByte());
                break;
            case 0x8b:
                pushSignedLong(reader_.readU32());
                break;
            case 'G': {
                const uint64_t bits = byteswap(reader_.readU64());
                auto value = makeValue(ValueKind::Floating);
                value->floating = std::bit_cast<double>(bits);
                push(std::move(value));
                break;
            }
            case 'F': {
                auto value = makeValue(ValueKind::Floating);
                value->floating = std::stod(reader_.readLine());
                push(std::move(value));
                break;
            }
            case 0x8c:
                pushString(reader_.readString(reader_.readByte()));
                break;
            case 'X':
                pushString(reader_.readString(reader_.readU32()));
                break;
            case 0x8d:
                pushString(reader_.readString(checkedSize(reader_.readU64(), "Unicode string")));
                break;
            case 'V':
                pushString(reader_.readLine());
                break;
            case 'S':
                pushString(readQuotedString(reader_.readLine()));
                break;
            case 'C':
                pushBytes(reader_.readBytes(reader_.readByte()));
                break;
            case 'B':
                pushBytes(reader_.readBytes(reader_.readU32()));
                break;
            case 0x8e:
                pushBytes(reader_.readBytes(checkedSize(reader_.readU64(), "binary data")));
                break;
            case 'U':
                pushString(reader_.readString(reader_.readByte()));
                break;
            case 'T':
                pushString(reader_.readString(reader_.readU32()));
                break;
            case ']':
                push(makeValue(ValueKind::List));
                break;
            case 'l': {
                auto value = makeValue(ValueKind::List);
                value->items = popMarkedItems();
                push(std::move(value));
                break;
            }
            case ')':
                push(makeValue(ValueKind::Tuple));
                break;
            case 't': {
                auto value = makeValue(ValueKind::Tuple);
                value->items = popMarkedItems();
                push(std::move(value));
                break;
            }
            case 0x85:
                makeFixedTuple(1);
                break;
            case 0x86:
                makeFixedTuple(2);
                break;
            case 0x87:
                makeFixedTuple(3);
                break;
            case '}':
                push(makeValue(ValueKind::Dictionary));
                break;
            case 'd': {
                auto value = makeValue(ValueKind::Dictionary);
                setDictionaryItems(*value, popMarkedItems());
                push(std::move(value));
                break;
            }
            case 'a': {
                const auto item = pop();
                auto list = peek("APPEND");
                if (list->kind != ValueKind::List) {
                    throw std::runtime_error("Legacy DTB pickle APPEND target is not a list.");
                }
                ensureContainerSize(list->items.size() + 1);
                list->items.push_back(item);
                break;
            }
            case 'e': {
                const auto items = popMarkedItems();
                auto list = peek("APPENDS");
                if (list->kind != ValueKind::List) {
                    throw std::runtime_error("Legacy DTB pickle APPENDS target is not a list.");
                }
                ensureContainerSize(list->items.size() + items.size());
                list->items.insert(list->items.end(), items.begin(), items.end());
                break;
            }
            case 's': {
                const auto value = pop();
                const auto key = pop();
                auto dictionary = peek("SETITEM");
                if (dictionary->kind != ValueKind::Dictionary) {
                    throw std::runtime_error("Legacy DTB pickle SETITEM target is not a dictionary.");
                }
                ensureContainerSize(dictionary->dictionary.size() + 1);
                dictionary->dictionary.emplace_back(key, value);
                break;
            }
            case 'u': {
                const auto items = popMarkedItems();
                auto dictionary = peek("SETITEMS");
                if (dictionary->kind != ValueKind::Dictionary) {
                    throw std::runtime_error("Legacy DTB pickle SETITEMS target is not a dictionary.");
                }
                setDictionaryItems(*dictionary, items);
                break;
            }
            case 'c': {
                const auto module = reader_.readLine();
                const auto name = reader_.readLine();
                pushGlobal(module, name);
                break;
            }
            case 0x93: { // STACK_GLOBAL
                const auto name = asString(pop(), "STACK_GLOBAL name");
                const auto module = asString(pop(), "STACK_GLOBAL module");
                pushGlobal(module, name);
                break;
            }
            case 'R':
                reduce();
                break;
            case 0x81:
                newObject();
                break;
            case 'b':
                build();
                break;
            case 0x94:
                memo_.push_back(peek("MEMOIZE"));
                break;
            case 'q':
                setMemo(reader_.readByte(), peek("BINPUT"));
                break;
            case 'r':
                setMemo(reader_.readU32(), peek("LONG_BINPUT"));
                break;
            case 'h':
                push(getMemo(reader_.readByte()));
                break;
            case 'j':
                push(getMemo(reader_.readU32()));
                break;
            case 'g':
                push(getMemo(static_cast<size_t>(std::stoull(reader_.readLine()))));
                break;
            case 'p':
                setMemo(static_cast<size_t>(std::stoull(reader_.readLine())), peek("PUT"));
                break;
            case '0':
                (void)pop();
                break;
            case '1':
                (void)popMarkedItems();
                break;
            case '2':
                push(peek("DUP"));
                break;
            default:
                throw std::runtime_error(
                    "Legacy DTB pickle uses an unsupported opcode 0x" + hexByte(opcode) + ".");
            }
        }
        throw std::runtime_error("Legacy DTB pickle ended without a STOP opcode.");
    }

private:
    static uint64_t byteswap(uint64_t value) {
        return ((value & 0x00000000000000FFULL) << 56U) |
            ((value & 0x000000000000FF00ULL) << 40U) |
            ((value & 0x0000000000FF0000ULL) << 24U) |
            ((value & 0x00000000FF000000ULL) << 8U) |
            ((value & 0x000000FF00000000ULL) >> 8U) |
            ((value & 0x0000FF0000000000ULL) >> 24U) |
            ((value & 0x00FF000000000000ULL) >> 40U) |
            ((value & 0xFF00000000000000ULL) >> 56U);
    }

    static size_t checkedSize(uint64_t value, const char* context) {
        if (value > kMaxPickleBytes || value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error(std::string("Legacy DTB ") + context + " exceeds the safety limit.");
        }
        return static_cast<size_t>(value);
    }

    static std::string hexByte(uint8_t value) {
        std::ostringstream stream;
        stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
        return stream.str();
    }

    static std::string readQuotedString(const std::string& value) {
        if (value.size() >= 2 && (value.front() == '\'' || value.front() == '"') && value.back() == value.front()) {
            return value.substr(1, value.size() - 2);
        }
        return value;
    }

    void push(ValuePtr value) {
        ensureContainerSize(stack_.size() + 1);
        stack_.push_back(std::move(value));
    }

    ValuePtr pop() {
        if (stack_.empty()) {
            throw std::runtime_error("Legacy DTB pickle stack underflow.");
        }
        auto value = stack_.back();
        stack_.pop_back();
        return value;
    }

    ValuePtr peek(const char* context) const {
        if (stack_.empty()) {
            throw std::runtime_error(std::string("Legacy DTB pickle stack underflow in ") + context + ".");
        }
        return stack_.back();
    }

    std::vector<ValuePtr> popMarkedItems() {
        for (size_t position = stack_.size(); position > 0; --position) {
            if (stack_[position - 1]->kind != ValueKind::Mark) {
                continue;
            }
            std::vector<ValuePtr> values(
                stack_.begin() + static_cast<std::ptrdiff_t>(position),
                stack_.end());
            stack_.erase(stack_.begin() + static_cast<std::ptrdiff_t>(position - 1), stack_.end());
            ensureContainerSize(values.size());
            return values;
        }
        throw std::runtime_error("Legacy DTB pickle is missing a MARK opcode.");
    }

    void makeFixedTuple(size_t count) {
        if (stack_.size() < count) {
            throw std::runtime_error("Legacy DTB pickle tuple stack underflow.");
        }
        auto value = makeValue(ValueKind::Tuple);
        value->items.assign(stack_.end() - static_cast<std::ptrdiff_t>(count), stack_.end());
        stack_.erase(stack_.end() - static_cast<std::ptrdiff_t>(count), stack_.end());
        push(std::move(value));
    }

    static void ensureContainerSize(size_t size) {
        if (size > kMaxContainerItems) {
            throw std::runtime_error("Legacy DTB pickle exceeds the container safety limit.");
        }
    }

    static void setDictionaryItems(PickleValue& dictionary, const std::vector<ValuePtr>& items) {
        if (items.size() % 2 != 0) {
            throw std::runtime_error("Legacy DTB pickle dictionary has an incomplete key/value pair.");
        }
        ensureContainerSize(dictionary.dictionary.size() + items.size() / 2);
        for (size_t index = 0; index < items.size(); index += 2) {
            dictionary.dictionary.emplace_back(items[index], items[index + 1]);
        }
    }

    void pushSignedLong(size_t count) {
        if (count == 0) {
            auto value = makeValue(ValueKind::Integer);
            push(std::move(value));
            return;
        }
        if (count > sizeof(int64_t)) {
            throw std::runtime_error("Legacy DTB integer exceeds native conversion limits.");
        }
        const auto bytes = reader_.readBytes(count);
        uint64_t raw = 0;
        for (size_t index = 0; index < bytes.size(); ++index) {
            raw |= static_cast<uint64_t>(bytes[index]) << (index * 8U);
        }
        if (count < sizeof(int64_t) && (bytes.back() & 0x80U) != 0U) {
            raw |= (~uint64_t{0}) << (count * 8U);
        }
        auto value = makeValue(ValueKind::Integer);
        value->integer = static_cast<int64_t>(raw);
        push(std::move(value));
    }

    void pushString(std::string text) {
        auto value = makeValue(ValueKind::String);
        value->text = std::move(text);
        push(std::move(value));
    }

    void pushBytes(std::vector<std::uint8_t> bytes) {
        auto value = makeValue(ValueKind::Bytes);
        value->bytes = std::move(bytes);
        push(std::move(value));
    }

    void pushGlobal(std::string module, std::string name) {
        auto value = makeValue(ValueKind::Global);
        value->globalModule = std::move(module);
        value->globalName = std::move(name);
        push(std::move(value));
    }

    void reduce() {
        const auto arguments = pop();
        const auto callable = pop();
        if (!isSequence(arguments)) {
            throw std::runtime_error("Legacy DTB pickle REDUCE arguments are not a tuple.");
        }
        if (isNumpyReconstruct(callable)) {
            auto array = makeValue(ValueKind::NumpyArray);
            array->array = NumpyArrayData{};
            push(std::move(array));
            return;
        }
        if (isNumpyDtype(callable)) {
            const auto& values = sequenceItems(arguments, "numpy dtype arguments");
            if (values.empty()) {
                throw std::runtime_error("Legacy DTB numpy dtype has no descriptor.");
            }
            auto dtype = makeValue(ValueKind::Dtype);
            dtype->text = asString(values.front(), "numpy dtype descriptor");
            push(std::move(dtype));
            return;
        }
        auto object = makeValue(ValueKind::Object);
        object->items = {callable, arguments};
        push(std::move(object));
    }

    void newObject() {
        const auto arguments = pop();
        const auto cls = pop();
        if (!isSequence(arguments)) {
            throw std::runtime_error("Legacy DTB pickle NEWOBJ arguments are not a tuple.");
        }
        auto object = makeValue(ValueKind::Object);
        object->items = {cls, arguments};
        push(std::move(object));
    }

    void build() {
        const auto state = pop();
        const auto instance = peek("BUILD");
        if (instance->kind == ValueKind::NumpyArray) {
            setNumpyArrayState(*instance, state);
        }
    }

    static void setNumpyArrayState(PickleValue& array, const ValuePtr& state) {
        const auto& stateItems = sequenceItems(state, "numpy array state");
        if (stateItems.size() < 5) {
            throw std::runtime_error("Legacy DTB numpy array state is incomplete.");
        }
        const auto& shapeItems = sequenceItems(stateItems[1], "numpy array shape");
        if (shapeItems.size() != 3) {
            throw std::runtime_error("Legacy DTB image does not have three dimensions.");
        }
        NumpyArrayData data;
        data.shape.reserve(shapeItems.size());
        size_t elementCount = 1;
        for (const auto& item : shapeItems) {
            const auto dimension = asInteger(item, "numpy image dimension");
            if (dimension <= 0 || static_cast<uint64_t>(dimension) > kMaxImagePixels) {
                throw std::runtime_error("Legacy DTB image dimensions are invalid.");
            }
            const auto size = static_cast<size_t>(dimension);
            if (size > kMaxImagePixels / elementCount) {
                throw std::runtime_error("Legacy DTB image exceeds the pixel safety limit.");
            }
            elementCount *= size;
            data.shape.push_back(dimension);
        }
        if (stateItems[2] == nullptr || stateItems[2]->kind != ValueKind::Dtype) {
            throw std::runtime_error("Legacy DTB image dtype is unavailable.");
        }
        data.dtype = stateItems[2]->text;
        data.fortranOrder = asBoolean(stateItems[3], "numpy array order");
        data.bytes = asBytes(stateItems[4], "numpy image payload");
        if (data.bytes.size() != elementCount) {
            throw std::runtime_error("Legacy DTB image payload size does not match its dimensions.");
        }
        array.array = std::move(data);
    }

    void setMemo(size_t index, ValuePtr value) {
        if (index > kMaxContainerItems) {
            throw std::runtime_error("Legacy DTB pickle memo exceeds the safety limit.");
        }
        if (memo_.size() <= index) {
            memo_.resize(index + 1);
        }
        memo_[index] = std::move(value);
    }

    ValuePtr getMemo(size_t index) const {
        if (index >= memo_.size() || memo_[index] == nullptr) {
            throw std::runtime_error("Legacy DTB pickle references an unknown memo entry.");
        }
        return memo_[index];
    }

    Reader reader_;
    std::vector<ValuePtr> stack_;
    std::vector<ValuePtr> memo_;
};

std::vector<std::uint8_t> readFile(const std::filesystem::path& sourcePath) {
    if (!std::filesystem::is_regular_file(sourcePath)) {
        throw std::runtime_error("Legacy DTB source file does not exist: " + fsc::core::pathToUtf8(sourcePath));
    }
    const auto size = std::filesystem::file_size(sourcePath);
    if (size == 0 || size > kMaxPickleBytes) {
        throw std::runtime_error("Legacy DTB source file is empty or exceeds the 1 GiB safety limit.");
    }
    std::ifstream file(sourcePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open legacy DTB source file: " + fsc::core::pathToUtf8(sourcePath));
    }
    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (file.gcount() != static_cast<std::streamsize>(data.size())) {
        throw std::runtime_error("Unable to read the full legacy DTB source file.");
    }
    return data;
}

fsc::vision::RgbImage imageFromArray(const PickleValue& value) {
    if (value.kind != ValueKind::NumpyArray || !value.array.has_value()) {
        throw std::runtime_error("Legacy DTB row image is not a NumPy array.");
    }
    const auto& array = *value.array;
    if (array.dtype != "u1" && array.dtype != "|u1" && array.dtype != "uint8") {
        throw std::runtime_error("Legacy DTB image dtype is not uint8.");
    }
    if (array.fortranOrder) {
        throw std::runtime_error("Legacy DTB image uses unsupported Fortran memory order.");
    }
    if (array.shape.size() != 3 || array.shape[2] != 3) {
        throw std::runtime_error("Legacy DTB image is not an RGB array.");
    }
    fsc::vision::RgbImage image;
    image.height = static_cast<int>(array.shape[0]);
    image.width = static_cast<int>(array.shape[1]);
    image.pixels = array.bytes;
    if (image.empty()) {
        throw std::runtime_error("Legacy DTB image has an invalid RGB payload.");
    }
    return image;
}

std::filesystem::path normalizedOutputPath(std::filesystem::path outputPath) {
    if (outputPath.empty()) {
        throw std::runtime_error("A converted .fscdb output path is required.");
    }
    if (outputPath.extension().empty()) {
        outputPath += ".fscdb";
    } else if (outputPath.extension() != ".fscdb") {
        outputPath.replace_extension(".fscdb");
    }
    return outputPath;
}

void writePpm(const std::filesystem::path& outputPath, const fsc::vision::RgbImage& image) {
    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to create converted legacy image: " + fsc::core::pathToUtf8(outputPath));
    }
    file << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (!file) {
        throw std::runtime_error("Unable to write converted legacy image: " + fsc::core::pathToUtf8(outputPath));
    }
}

std::vector<std::vector<double>> keypointRows(const std::array<fsc::vision::Point2f, 5>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point2f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point3f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y, point.z});
    }
    return rows;
}

std::string qualityJson(const fsc::vision::AnalyzedFace& face) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6)
           << "{\"det_score\":" << face.detection.score
           << ",\"area_ratio\":" << face.qualityAreaRatio
           << ",\"sharpness\":" << face.qualitySharpness
           << ",\"brightness\":" << face.qualityBrightness
           << ",\"contrast\":" << face.qualityContrast
           << ",\"native\":true,\"converted_from_legacy\":true}";
    return stream.str();
}

fsc::core::FaceInsertRecord recordFromLegacyFace(
    const std::filesystem::path& extractedImagePath,
    const std::string& originalFileName,
    const fsc::vision::AnalyzedFace& face,
    const std::string& imageHash,
    bool duplicate) {
    fsc::core::FaceInsertRecord record;
    record.fileName = originalFileName;
    record.sourcePath = fsc::core::pathToUtf8(extractedImagePath);
    record.embedding = face.embedding;
    record.embeddingDim = static_cast<int>(face.embedding.size());
    record.bbox = {
        face.detection.box.x1,
        face.detection.box.y1,
        face.detection.box.x2,
        face.detection.box.y2,
    };
    record.keypoints = keypointRows(face.detection.keypoints);
    record.landmarks2d = landmarkRows(face.landmarks2d);
    record.landmarks3d = landmarkRows(face.landmarks3d);
    record.detectionScore = face.detection.score;
    record.qualityScore = face.qualityScore;
    record.qualityJson = qualityJson(face);
    record.imageHash = imageHash;
    if (duplicate) {
        record.reviewState = "duplicate";
        record.notes = "Same embedded legacy image hash already exists in this database.";
    }
    return record;
}

std::string lowerExtension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

} // namespace

std::vector<LegacyDtbImage> loadLegacyDtbImages(const std::filesystem::path& sourcePath) {
    if (lowerExtension(sourcePath) != ".dtb") {
        throw std::runtime_error("Legacy conversion expects a .dtb source file.");
    }
    const auto root = RestrictedPickleParser(readFile(sourcePath)).parse();
    const auto& rows = sequenceItems(root, "legacy DTB root");
    if (rows.size() > kMaxContainerItems) {
        throw std::runtime_error("Legacy DTB contains too many rows.");
    }

    std::vector<LegacyDtbImage> output;
    output.reserve(rows.size());
    for (size_t index = 0; index < rows.size(); ++index) {
        const auto& fields = sequenceItems(rows[index], "legacy DTB row");
        if (fields.size() < 5) {
            throw std::runtime_error("Legacy DTB row " + std::to_string(index + 1) + " is incomplete.");
        }
        LegacyDtbImage image;
        image.image = imageFromArray(*fields[3]);
        image.fileName = asString(fields[4], "legacy DTB file name");
        if (image.fileName.empty()) {
            image.fileName = "legacy_row_" + std::to_string(index + 1) + ".ppm";
        }
        output.push_back(std::move(image));
    }
    return output;
}

LegacyConversionSummary convertLegacyDtb(
    const std::filesystem::path& sourcePath,
    std::filesystem::path outputPath,
    const LegacyConversionOptions& options) {
    const auto images = loadLegacyDtbImages(sourcePath);
    outputPath = normalizedOutputPath(std::move(outputPath));
    std::error_code equivalenceError;
    if (std::filesystem::equivalent(sourcePath, outputPath, equivalenceError) && !equivalenceError) {
        throw std::runtime_error("Legacy DTB output path must differ from the source path.");
    }
    if (options.models.missingFiles().size() > 0) {
        throw std::runtime_error("InsightFace model files are unavailable for legacy conversion.");
    }

    LegacyConversionSummary summary;
    summary.sourcePath = sourcePath;
    summary.outputPath = outputPath;
    summary.imageDirectory = fsc::core::pathWithSuffix(
        outputPath.parent_path() / outputPath.stem(),
        "_legacy_images");
    summary.rowsTotal = options.limit > 0 ? std::min<int>(options.limit, static_cast<int>(images.size())) : static_cast<int>(images.size());
    if (summary.rowsTotal == 0) {
        throw std::runtime_error("Legacy DTB contains no image rows to convert.");
    }

    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }
    std::error_code ignored;
    std::filesystem::remove_all(summary.imageDirectory, ignored);
    std::filesystem::create_directories(summary.imageDirectory);
    fsc::core::Database::createEmpty(outputPath, true);
    fsc::core::Database database(outputPath);
    fsc::vision::InsightFaceEngine engine(options.models, options.runtimeMode);

    const auto report = [&](const std::string& message, int current) {
        summary.messages.push_back(message);
        if (options.progress) {
            options.progress(message, current, summary.rowsTotal);
        }
    };

    for (int index = 0; index < summary.rowsTotal; ++index) {
        const auto& legacyImage = images[static_cast<size_t>(index)];
        const int current = index + 1;
        try {
            const auto faces = engine.analyze(legacyImage.image, options.detectionThreshold, 10);
            if (faces.size() != 1) {
                ++summary.skippedRows;
                report(legacyImage.fileName + ": skipped, detected " + std::to_string(faces.size()) + " faces", current);
                continue;
            }
            const auto imagePath = summary.imageDirectory /
                ("legacy_" + std::to_string(current) + ".ppm");
            writePpm(imagePath, legacyImage.image);
            const auto imageHash = fsc::core::sha256File(imagePath);
            const bool duplicate = database.imageHashExists(imageHash);
            database.insertFace(recordFromLegacyFace(imagePath, legacyImage.fileName, faces.front(), imageHash, duplicate));
            ++summary.facesSaved;
            report(legacyImage.fileName + ": saved", current);
        } catch (const std::exception& exception) {
            ++summary.skippedRows;
            report("row " + std::to_string(current) + ": skipped (" + exception.what() + ')', current);
        }
    }
    return summary;
}

} // namespace fsc::legacy
