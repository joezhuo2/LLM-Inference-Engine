#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "commands.h"
#include "engine/loader/checksum.h"
#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"

#if ENGINE_HAS_CUDA
#include "engine/runtime/checksum.h"
#include "engine/runtime/weights.h"
#endif

namespace engine::app {

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path out;
    bool from_file = false;
};

std::string shape_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

std::vector<TensorChecksum> from_file(const std::filesystem::path& model) {
    const MappedFile file(model / "model.safetensors");
    return checksum_file(file.bytes(), parse_safetensors_header(file.bytes()));
}

#if ENGINE_HAS_CUDA
std::vector<TensorChecksum> from_device(const std::filesystem::path& model) {
    const DeviceWeights w = load_weights(model, nullptr);
    std::printf("loaded %.2f GB in %.3f s (%.2f GB/s)\n", double(w.stats.bytes) / 1e9,
                w.stats.seconds, w.stats.gb_per_s());
    return checksum_device(w.layout, w.buffer.data());
}
#endif

void run(const Options& o) {
#if ENGINE_HAS_CUDA
    const bool device = !o.from_file;
    const auto sums = device ? from_device(o.model) : from_file(o.model);
#else
    const bool device = false;
    const auto sums = from_file(o.model);
#endif
    for (const TensorChecksum& c : sums) {
        std::printf("%-52s %-5s %-14s %+.10e %.10e\n", c.name.c_str(),
                    std::string(dtype_name(c.dtype)).c_str(), shape_string(c.shape).c_str(), c.sum,
                    c.abs_sum);
    }
    std::printf("%zu tensors, checksummed from the %s\n", sums.size(),
                device ? "device weights" : "file");
    if (!o.out.empty()) {
        nlohmann::json j = checksums_to_json(sums);
        j["source"] = device ? "device" : "file";
        std::ofstream(o.out) << j.dump(2) << '\n';
    }
}

}  // namespace

void add_checksum_command(CLI::App& app) {
    auto opts = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand("checksum", "Print an FP64 checksum of every weight tensor");
    cmd->add_option("--model", opts->model,
                    "Model directory with config.json and model.safetensors")
        ->required()
        ->check(CLI::ExistingDirectory);
    cmd->add_option("--out", opts->out, "Also write the checksums as JSON to this file");
    cmd->add_flag("--from-file", opts->from_file,
                  "Checksum the file on the host instead of the uploaded device weights "
                  "(always the case in CPU-only builds)");
    cmd->callback([opts] { run(*opts); });
}

}  // namespace engine::app
