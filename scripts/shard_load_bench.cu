#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"
#include "engine/loader/tensor_parallel.h"
#include "engine/loader/weight_layout.h"
#include "engine/model/config.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/weights.h"
#include "kernels/cuda_check.h"

using namespace engine;

namespace {

using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Plan>
void row(const std::string& label, std::span<const std::byte> file, cudaStream_t stream, int runs,
         Plan&& plan) {
    std::vector<double> plan_ms(runs), upload_ms(runs);
    WeightLayout layout;
    for (int i = 0; i < runs; ++i) {
        const auto start = Clock::now();
        layout = plan();
        plan_ms[i] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    DeviceBuffer device(layout.total_bytes);
    upload_weights(file, layout, device.data(), stream);
    for (int i = 0; i < runs; ++i)
        upload_ms[i] = upload_weights(file, layout, device.data(), stream).seconds * 1e3;
    const double ms = median(upload_ms);
    std::printf("| %s | %zu | %.1f | %.2f | %.1f | %.1f to %.1f | %.2f |\n", label.c_str(),
                layout.copies.size(), double(layout.total_bytes) / (1 << 20), median(plan_ms), ms,
                *std::min_element(upload_ms.begin(), upload_ms.end()),
                *std::max_element(upload_ms.begin(), upload_ms.end()),
                double(layout.total_bytes) / 1e9 / (ms / 1e3));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "models/tinyllama";
    const int runs = argc > 2 ? std::atoi(argv[2]) : 7;
    const ModelConfig config = load_config(model + "/config.json");
    const MappedFile mapped(model + "/model.safetensors");
    const SafetensorsHeader header = parse_safetensors_header(mapped.bytes());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::printf(
        "| layout | copies | MiB on the device | plan ms | upload ms | min to max | GB/s |\n| --- "
        "| --- | --- | --- | --- | --- | --- |\n");
    row("full model", mapped.bytes(), stream, runs,
        [&] { return plan_weight_layout(config, header); });
    for (const int ranks : {1, 2, 4}) {
        for (int rank = 0; rank < ranks; ++rank) {
            row("rank " + std::to_string(rank) + " of " + std::to_string(ranks), mapped.bytes(),
                stream, runs, [&] { return plan_shard_layout(config, header, rank, ranks); });
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
