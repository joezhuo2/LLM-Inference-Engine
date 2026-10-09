#include "engine/loader/safetensors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine {
namespace {
constexpr uint64_t MAX_HEADER_SIZE = 100'000'000;

[[noreturn]] void throw_error(const std::string& message) {
    throw std::runtime_error("safetensors parser error: " + message);
}

[[noreturn]] void throw_tensor_error(std::string_view tensor_name, const std::string& message) {
    throw std::runtime_error("safetensors error for tensor " + std::string(tensor_name) + ": " +
                             message);
}

size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case DType::BF16:
        case DType::F16:
            return 2;
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::I64:
            return 8;
        default:
            throw_error("unsupported enum DType");
    }
}
}  // namespace

SafetensorsHeader parse_safetensors_header(std::span<const std::byte> file_bytes) {
    const size_t total_file_size = file_bytes.size();

    if (total_file_size < 8) {
        throw_error("file size is smaller than the minimum 8 byte header prefix");
    }

    uint64_t header_length = 0;
    for (size_t i = 0; i < 8; ++i) {
        header_length |= static_cast<uint64_t>(static_cast<uint8_t>(file_bytes[i])) << (8 * i);
    }

    if (header_length > MAX_HEADER_SIZE) {
        throw_error("Header length N (" + std::to_string(header_length) +
                    " bytes) exceeds maximum allowed bytes of 100,000,000 bytes");
    }

    const uint64_t remaining_bytes = static_cast<uint64_t>(total_file_size - 8);
    if (header_length > remaining_bytes) {
        throw_error("Header length N (" + std::to_string(header_length) +
                    " bytes) exceeds remaining file size (" + std::to_string(remaining_bytes) +
                    " bytes)");
    }

    const uint64_t data_offset = 8 + header_length;

    const char* json_data_ptr = reinterpret_cast<const char*>(file_bytes.data() + 8);
    const std::string_view json_view(json_data_ptr, static_cast<size_t>(header_length));

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(json_view);
    } catch (const nlohmann::json::exception& e) {
        throw_error("invalid json header: " + std::string(e.what()));
    }

    if (!root.is_object()) {
        throw_error("top level json value must be an object");
    }

    if (root.contains("__metadata__")) {
        const auto& metadata = root["__metadata__"];
        if (metadata.is_object()) {
            for (auto it = metadata.begin(); it != metadata.end(); ++it) {
                if (!it.value().is_string()) {
                    throw_error("metadata key '" + it.key() + "' must have a string value");
                }
            }
        } else if (!metadata.is_null()) {
            throw_error("metadata must be null or an object");
        }
    }

    SafetensorsHeader header;
    header.data_offset = data_offset;

    auto is_strict_integer = [](const nlohmann::json& val) -> bool {
        return val.is_number_integer() || val.is_number_unsigned();
    };

    auto parse_dtype_string = [](std::string_view tensor_name,
                                 const std::string& dtype_str) -> DType {
        if (dtype_str == "BF16") return DType::BF16;
        if (dtype_str == "F16") return DType::F16;
        if (dtype_str == "F32") return DType::F32;
        if (dtype_str == "I32") return DType::I32;
        if (dtype_str == "I64") return DType::I64;

        throw_tensor_error(tensor_name, "Unsupported or invalid dtype '" + dtype_str + "'");
    };

    struct Range {
        uint64_t begin;
        uint64_t end;
        std::string name;
    };

    std::vector<Range> ranges;
    ranges.reserve(root.size());

    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string& tensor_name = it.key();

        if (tensor_name == "__metadata__") {
            continue;
        }

        const auto& tensor_val = it.value();

        if (!tensor_val.is_object()) {
            throw_tensor_error(tensor_name, "value must be a json object");
        }

        if (!tensor_val.contains("dtype") || !tensor_val.contains("shape") ||
            !tensor_val.contains("data_offsets")) {
            throw_tensor_error(tensor_name, "missing required field dtype, shape, or data offsets");
        }

        const auto& dtype_val = tensor_val["dtype"];
        if (!dtype_val.is_string()) {
            throw_tensor_error(tensor_name, "dtype must be a string");
        }
        DType dtype = parse_dtype_string(tensor_name, dtype_val.get<std::string>());

        const auto& shape_val = tensor_val["shape"];
        if (!shape_val.is_array()) {
            throw_tensor_error(tensor_name, "shape must be an array");
        }

        std::vector<int64_t> shape;
        shape.reserve(shape_val.size());
        for (const auto& dim_val : shape_val) {
            if (!is_strict_integer(dim_val)) {
                throw_tensor_error(tensor_name, "shape elements must be strict integers");
            }

            int64_t dim = dim_val.get<int64_t>();
            if (dim < 0) {
                throw_tensor_error(tensor_name, "shape elements cannot be negative");
            }

            shape.push_back(dim);
        }

        const auto& offsets_val = tensor_val["data_offsets"];
        if (!offsets_val.is_array() || offsets_val.size() != 2) {
            throw_tensor_error(tensor_name, "data offsets must be an array of 2 integers");
        }

        if (!offsets_val[0].is_number_unsigned() || !offsets_val[1].is_number_unsigned()) {
            throw_tensor_error(tensor_name, "data offset values must be non-negative integers");
        }

        const uint64_t begin_offset = offsets_val[0].get<uint64_t>();
        const uint64_t end_offset = offsets_val[1].get<uint64_t>();

        if (begin_offset > end_offset) {
            throw_tensor_error(tensor_name, "begin offset (" + std::to_string(begin_offset) +
                                                ") is greater than end offset (" +
                                                std::to_string(end_offset) + ")");
        }

        uint64_t total_elements = 1;
        const size_t dtype_size = get_dtype_size(dtype);

        for (const int64_t dim : shape) {
            if (dim == 0) {
                total_elements = 0;
                break;
            }
            if (total_elements >
                std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
                throw_tensor_error(tensor_name, "Shape dimensions overflow 64-bit integer limit");
            }
            total_elements *= static_cast<uint64_t>(dim);
        }

        if (total_elements > std::numeric_limits<uint64_t>::max() / dtype_size) {
            throw_tensor_error(tensor_name, "Total byte size overflows 64-bit integer limit");
        }
        const uint64_t expected_bytes = total_elements * static_cast<uint64_t>(dtype_size);
        const uint64_t declared_bytes = end_offset - begin_offset;

        if (declared_bytes != expected_bytes) {
            throw_tensor_error(tensor_name, "Declared data_offsets byte range (" +
                                                std::to_string(declared_bytes) +
                                                ") does not match calculated shape byte size (" +
                                                std::to_string(expected_bytes) + ")");
        }

        TensorEntry entry;
        entry.dtype = dtype;
        entry.shape = std::move(shape);
        entry.offset = data_offset + begin_offset;
        entry.bytes = declared_bytes;

        header.tensors[tensor_name] = entry;
        ranges.push_back(Range{begin_offset, end_offset, tensor_name});
    }

    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
        if (a.begin != b.begin) {
            return a.begin < b.begin;
        }
        return a.end < b.end;
    });

    const uint64_t payload_size = static_cast<uint64_t>(total_file_size) - data_offset;

    if (ranges.empty()) {
        if (payload_size != 0) {
            throw_error("Data section has " + std::to_string(payload_size) +
                        " bytes but contains no tensors");
        }
    } else {
        if (ranges[0].begin != 0) {
            throw_error("First tensor range does not begin at relative offset 0 (begins at " +
                        std::to_string(ranges[0].begin) + ")");
        }

        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].begin != ranges[i - 1].end) {
                if (ranges[i].begin < ranges[i - 1].end) {
                    throw_tensor_error(ranges[i].name, "Overlaps with previous tensor range");
                } else {
                    throw_tensor_error(ranges[i].name, "Gap detected before this tensor range");
                }
            }
        }

        if (ranges.back().end != payload_size) {
            throw_error("Tensor ranges do not tile the data section exactly (last range ends at " +
                        std::to_string(ranges.back().end) + ", expected " +
                        std::to_string(payload_size) + ")");
        }
    }

    return header;
}
}  // namespace engine
