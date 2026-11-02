#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "commands.h"
#include "engine/server/chat_api.h"
#include "engine/server/cors.h"

#if ENGINE_HAS_CUDA
#include "engine/core/version.h"
#include "engine/runtime/weights.h"
#include "engine/server/gpu_backend.h"
#include "engine/server/http_server.h"
#include "engine/server/serving_loop.h"
#include "engine/tokenizer/tokenizer.h"
#endif

namespace engine::app {

namespace {

struct Options {
    std::filesystem::path model;
    std::string host = "127.0.0.1";
    int port = 8000;
    int max_batch = 16;
    bool graphs = false;
    std::vector<std::string> cors_origins = server::kDefaultCorsOrigins;
    server::SamplingDefaults defaults{};
};

#if ENGINE_HAS_CUDA
std::string model_name(const std::filesystem::path& model) {
    return model.filename().empty() ? model.parent_path().filename().string()
                                    : model.filename().string();
}

bool is_loopback(const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

std::string joined(const std::vector<std::string>& items) {
    std::string out;
    for (const std::string& item : items) out += (out.empty() ? "" : ", ") + item;
    return out;
}

server::HttpOptions http_options(const Options& o, const ModelConfig& config) {
    nlohmann::ordered_json meta = {{"engine_version", std::string(version())},
                                   {"gpu", server::gpu_name()},
                                   {"model", model_name(o.model)},
                                   {"max_batch", o.max_batch},
                                   {"graphs", o.graphs}};
    return {.cors_origins = o.cors_origins,
            .defaults = o.defaults,
            .model = model_name(o.model),
            .meta = std::move(meta),
            .max_context = config.max_positions};
}

void run(const Options& o) {
    validate(SamplingParams{o.defaults.temperature, 0, o.defaults.top_k, o.defaults.top_p});
    for (const std::string& origin : o.cors_origins) server::validate_origin(origin);
    if (!is_loopback(o.host))
        std::fprintf(stderr,
                     "warning: listening on %s; the server has no authentication, so anyone who "
                     "can reach this address can use the GPU\n",
                     o.host.c_str());

    const Tokenizer tokenizer(o.model / "tokenizer.model");
    const DeviceWeights w = load_weights(o.model, nullptr);
    const auto backend =
        server::make_gpu_backend(w, {.max_batch = o.max_batch, .graphs = o.graphs});
    server::ServingLoop loop(
        *backend, [&](std::span<const int32_t> ids) { return tokenizer.decode(ids); },
        w.config.eos_id);
    server::HttpServer http(
        loop, [&](const std::string& text) { return tokenizer.encode(text, false); },
        http_options(o, w.config));
    const int port = http.bind(o.host, o.port);

    std::exception_ptr failure;
    std::jthread engine([&](std::stop_token stop) {
        try {
            loop.run(stop);
        } catch (...) {
            failure = std::current_exception();
            http.stop();
        }
    });
    std::fprintf(stderr, "serving %s on http://%s:%d with %d KV blocks; CORS origins: %s\n",
                 model_name(o.model).c_str(), o.host.c_str(), port, backend->num_blocks(),
                 joined(o.cors_origins).c_str());
    http.listen();
    engine.request_stop();
    engine.join();
    if (failure) std::rethrow_exception(failure);
}
#else
void run(const Options&) {
    throw std::runtime_error("serve needs a CUDA build");
}
#endif

}  // namespace

void add_serve_command(CLI::App& app) {
    auto opts = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand(
        "serve", "Serve the OpenAI-compatible chat API, metrics and the step stream over HTTP");
    cmd->add_option("--model", opts->model,
                    "Model directory with config.json, model.safetensors and tokenizer.model")
        ->required()
        ->check(CLI::ExistingDirectory);
    cmd->add_option("--host", opts->host,
                    "Address to listen on; keep the loopback default, the server has no "
                    "authentication")
        ->capture_default_str();
    cmd->add_option("--port", opts->port, "Port to listen on")
        ->capture_default_str()
        ->check(CLI::Range(1, 65535));
    cmd->add_option("--max-batch", opts->max_batch, "Most sequences decoded together")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    cmd->add_flag("--graphs", opts->graphs,
                  "Replay CUDA Graphs of the decode step (one per power-of-two batch size up to "
                  "--max-batch)");
    cmd->add_option("--cors-origin", opts->cors_origins,
                    "Origin allowed to call the server from a browser, such as "
                    "https://example.vercel.app; repeat for several (default: "
                    "http://localhost:3000)");
    cmd->add_option("--temperature", opts->defaults.temperature,
                    "Default temperature when a request sets none; 0 decodes greedily")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-k", opts->defaults.top_k,
                    "Default top-k when a request sets none; 0 is off")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-p", opts->defaults.top_p,
                    "Default top-p when a request sets none; 1 is off")
        ->capture_default_str()
        ->check(CLI::Range(0.0f, 1.0f));
    cmd->add_option("--max-tokens", opts->defaults.max_tokens,
                    "Default max_tokens when a request sets none, clamped to the context left")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    cmd->callback([opts] { run(*opts); });
}

}  // namespace engine::app
