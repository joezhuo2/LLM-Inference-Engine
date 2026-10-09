#pragma once

#include <vector>

#include "engine/loader/checksum.h"
#include "engine/loader/weight_layout.h"

namespace engine {

std::vector<TensorChecksum> checksum_device(const WeightLayout& layout, const void* device);

}  // namespace engine
