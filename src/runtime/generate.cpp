#include "engine/runtime/generate.h"

#include <stdexcept>

#include "engine/runtime/device_buffer.h"
#include "engine/sampling/greedy.h"
#include "kernels/cuda_check.h"

namespace engine {

std::vector<int32_t> generate_greedy(ModelRunner& runner, std::span<const int32_t> prompt,
                                     int max_new_tokens, int32_t eos_id,
                                     const TokenCallback& on_token) {
    if (max_new_tokens < 1)
        throw std::invalid_argument("generate_greedy: max_new_tokens must be positive");
    if (int64_t(prompt.size()) + max_new_tokens - 1 > runner.max_context())
        throw std::invalid_argument(
            "generate_greedy: the prompt plus max_new_tokens does not fit in the context");

    DeviceBuffer next(sizeof(int32_t));
    const Tensor next_id(next.data(), DType::I32, {1});
    std::vector<int32_t> out;
    runner.reset();
    Tensor logits = runner.forward(prompt);
    while (true) {
        greedy(next_id, logits, runner.stream());
        int32_t token = 0;
        CUDA_CHECK(cudaMemcpyAsync(&token, next.data(), sizeof(token), cudaMemcpyDeviceToHost,
                                   runner.stream()));
        CUDA_CHECK(cudaStreamSynchronize(runner.stream()));
        out.push_back(token);
        if (on_token) on_token(token);
        if (token == eos_id || int(out.size()) == max_new_tokens) break;
        logits = runner.forward(std::span(&token, 1));
    }
    return out;
}

}  // namespace engine
