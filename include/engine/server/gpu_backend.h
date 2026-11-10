#pragma once

#include <memory>
#include <string>

#include "engine/runtime/weights.h"
#include "engine/server/backend.h"

namespace engine::server {

struct GpuBackendOptions {
    int max_batch = 16;
    bool graphs = false;
    int graph_steps = 1;
    int token_budget = 0;
};

std::unique_ptr<Backend> make_gpu_backend(const DeviceWeights& weights,
                                          const GpuBackendOptions& options);

std::string gpu_name();

}  // namespace engine::server
