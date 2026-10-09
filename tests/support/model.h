#pragma once

#include <memory>

#include "engine/runtime/weights.h"
#include "support/paths.h"

namespace engine::test {

inline const DeviceWeights& tinyllama() {
    static const auto weights = std::make_unique<DeviceWeights>(load_weights(model_dir(), nullptr));
    return *weights;
}

}  // namespace engine::test
