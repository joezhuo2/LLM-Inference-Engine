#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "commands.h"
#include "engine/bench/csv.h"
#include "engine/bench/metrics.h"
#include "engine/bench/workload.h"
#include "engine/core/version.h"
#include "engine/model/config.h"

#if ENGINE_HAS_CUDA
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "engine/runtime/bench.h"
#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/runtime/weights.h"
#include "engine/sched/scheduler.h"
#endif

namespace engine::app {

namespace {

struct Options {
    std::filesystem::path model;
    int num_requests = 64;
    std::string prompt_len = "128";
    std::string output_len = "256";
    double request_rate = 0.0;
    int max_batch = 0;
    bool graphs = false;
    int warmup = 3;
    uint64_t seed = 0;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool stop_at_eos = false;
    bool profile = false;
    std::filesystem::path out;
    std::filesystem::path summary;
};

#if ENGINE_HAS_CUDA
std::string format_range(LengthRange r) {
    return r.min == r.max ? std::to_string(r.min)
                          : std::to_string(r.min) + ":" + std::to_string(r.max);
}

struct OwnedStream {
    cudaStream_t stream = nullptr;

    OwnedStream() {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("cannot create a CUDA stream");
    }
    ~OwnedStream() { cudaStreamDestroy(stream); }
    OwnedStream(const OwnedStream&) = delete;
    OwnedStream& operator=(const OwnedStream&) = delete;
};

std::vector<int64_t> graph_buckets(int max_batch) {
    std::vector<int64_t> buckets;
    for (int64_t b = 1; b < max_batch; b *= 2) buckets.push_back(b);
    buckets.push_back(max_batch);
    return buckets;
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::ofstream open_csv(const std::filesystem::path& path) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write " + path.string());
    return file;
}

std::string gpu_name() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return "unknown";
    return prop.name;
}

void run(const Options& o) {
    if (std::string_view(ENGINE_BUILD_TYPE) != "Release")
        std::fprintf(stderr, "warning: this is a %s build; benchmark the release preset\n",
                     ENGINE_BUILD_TYPE);
    const ModelConfig config = load_config(o.model / "config.json");
    WorkloadConfig wc{.num_requests = o.num_requests,
                      .prompt_len = parse_length_range(o.prompt_len),
                      .output_len = parse_length_range(o.output_len),
                      .vocab = config.vocab,
                      .bos_id = config.bos_id,
                      .eos_id = config.eos_id,
                      .request_rate = o.request_rate,
                      .seed = o.seed,
                      .params = {o.temperature, o.seed, o.top_k, o.top_p}};
    Workload workload = make_workload(wc);
    const int64_t context = int64_t(wc.prompt_len.max) + wc.output_len.max - 1;
    if (context > config.max_positions)
        throw std::invalid_argument("the longest prompt plus output needs " +
                                    std::to_string(context) + " tokens of context; the model has " +
                                    std::to_string(config.max_positions));
    const auto summary_path =
        o.summary.empty() ? o.out.parent_path() /
                                (o.out.stem().string() + "_summary" + o.out.extension().string())
                          : o.summary;
    std::ofstream out = open_csv(o.out);
    std::ofstream summary_out = open_csv(summary_path);

    const DeviceWeights w = load_weights(o.model, nullptr);
    const int32_t eos_id = o.stop_at_eos ? w.config.eos_id : -1;
    const bool batched = o.max_batch > 0;
    if (o.graphs && !batched)
        throw std::invalid_argument("--graphs needs --max-batch of at least 1");

    int num_blocks = 0;
    ServeStats stats;
    std::optional<ModelRunner> one_runner;
    std::optional<OwnedStream> owned_stream;
    std::optional<KvCache> cache;
    std::optional<PagedRunner> paged_runner;
    std::optional<Sampler> sampler;
    std::optional<DecodeGraphs> graphs;
    SchedulerConfig sc;
    if (batched) {
        num_blocks = kv_cache_blocks(w.config, kKvBlockSize, device_free_bytes());
        const int64_t max_prefill = std::max<int64_t>(wc.prompt_len.max, 2048);
        sc = {.num_blocks = num_blocks,
              .block_size = kKvBlockSize,
              .max_batch = o.max_batch,
              .max_prefill_tokens = int(max_prefill),
              .max_context = int(context),
              .eos_id = eos_id};
        if (o.graphs) owned_stream.emplace();
        cache.emplace(w.config, num_blocks, kKvBlockSize);
        paged_runner.emplace(w.config, w.weights, *cache, max_prefill, o.max_batch, context,
                             o.graphs ? owned_stream->stream : nullptr);
        sampler.emplace(o.max_batch);
        if (o.graphs) graphs.emplace(*paged_runner, graph_buckets(o.max_batch));
    } else {
        one_runner.emplace(w.config, w.weights, wc.prompt_len.max, context, nullptr);
    }
    const auto drive = [&](Workload& load, Clock::time_point start) {
        if (!batched) {
            run_one_at_a_time(*one_runner, load, eos_id, start);
            return ServeStats{};
        }
        Scheduler scheduler(sc);
        return run_batched(scheduler, *paged_runner, *sampler, load, start,
                           graphs ? &*graphs : nullptr);
    };
    if (o.warmup > 0) {
        WorkloadConfig warm = wc;
        warm.num_requests = o.warmup;
        warm.request_rate = 0.0;
        warm.seed = o.seed + 1;
        Workload warmup = make_workload(warm);
        drive(warmup, Clock::now());
    }

    if (o.profile) cudaProfilerStart();
    const auto start = Clock::now();
    stats = drive(workload, start);
    if (o.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStop();
    }
    Clock::time_point end = start;
    std::vector<RequestMetrics> metrics;
    int64_t preemptions = 0;
    for (const auto& r : workload.requests) {
        metrics.push_back(request_metrics(r, start));
        preemptions += r.preemptions;
        end = std::max(end, r.finished);
    }
    const Summary s = summarize(metrics, std::chrono::duration<double>(end - start).count());
    const uint64_t weight_bytes =
        w.stats.bytes - (w.config.tie_embeddings ? 0 : w.layout.embed.bytes);
    const double bandwidth = double(weight_bytes) / s.tpot_s.p50 / 1e9;

    write_request_csv(out, metrics);
    CsvFields fields = {
        {"version", std::string(version())},
        {"date", utc_now()},
        {"gpu", gpu_name()},
        {"build", ENGINE_BUILD_TYPE},
        {"model", o.model.filename().empty() ? o.model.parent_path().filename().string()
                                             : o.model.filename().string()},
        {"driver", !batched   ? "one_at_a_time"
                   : o.graphs ? "batched_graphs"
                              : "batched"},
        {"max_batch", std::to_string(batched ? o.max_batch : 1)},
        {"num_requests", std::to_string(o.num_requests)},
        {"prompt_len", format_range(wc.prompt_len)},
        {"output_len", format_range(wc.output_len)},
        {"request_rate", csv_number(o.request_rate)},
        {"warmup", std::to_string(o.warmup)},
        {"seed", std::to_string(o.seed)},
        {"temperature", csv_number(o.temperature)},
        {"top_k", std::to_string(o.top_k)},
        {"top_p", csv_number(o.top_p)},
        {"stop_at_eos", o.stop_at_eos ? "1" : "0"},
    };
    for (auto& field : summary_fields(s)) fields.push_back(std::move(field));
    if (batched) {
        fields.emplace_back("kv_blocks", std::to_string(num_blocks));
        fields.emplace_back("peak_kv_blocks", std::to_string(num_blocks - stats.min_free_blocks));
        fields.emplace_back("steps", std::to_string(stats.steps));
        fields.emplace_back("prefill_steps", std::to_string(stats.prefill_steps));
        fields.emplace_back("preemptions", std::to_string(preemptions));
    } else {
        fields.emplace_back("decode_weight_bytes", std::to_string(weight_bytes));
        fields.emplace_back("bandwidth_gb_s", csv_number(bandwidth));
    }
    write_csv_row(summary_out, fields);
    out.close();
    summary_out.close();
    if (!out || !summary_out) throw std::runtime_error("writing the CSV files failed");

    std::fprintf(stderr,
                 "%lld requests, %lld output tokens in %.2f s: %.1f tokens/s\n"
                 "TTFT p50 %.1f ms, p90 %.1f ms; TPOT p50 %.2f ms (%.1f tokens/s per request)\n",
                 static_cast<long long>(s.requests), static_cast<long long>(s.output_tokens),
                 s.wall_s, s.output_throughput, s.ttft_s.p50 * 1e3, s.ttft_s.p90 * 1e3,
                 s.tpot_s.p50 * 1e3, 1.0 / s.tpot_s.p50);
    if (batched)
        std::fprintf(stderr, "peak KV blocks %d of %d, %lld preemptions\n",
                     num_blocks - stats.min_free_blocks, num_blocks,
                     static_cast<long long>(preemptions));
    else
        std::fprintf(stderr, "%.1f GB/s of weights\n", bandwidth);
    std::fprintf(stderr, "wrote %s and %s\n", o.out.c_str(), summary_path.c_str());
}
#else
void run(const Options&) {
    throw std::runtime_error("bench needs a CUDA build");
}
#endif

}  // namespace

void add_bench_command(CLI::App& app) {
    auto opts = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand(
        "bench", "Serve a synthetic workload and write per-request and summary CSVs");
    cmd->add_option("--model", opts->model,
                    "Model directory with config.json, model.safetensors and tokenizer.model")
        ->required()
        ->check(CLI::ExistingDirectory);
    cmd->add_option("--max-batch", opts->max_batch,
                    "Serve through the paged runner with at most this many concurrent sequences; "
                    "0 runs requests one at a time on the contiguous runner")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_flag("--graphs", opts->graphs,
                  "Replay CUDA Graphs of the decode step (one per power-of-two batch size up to "
                  "--max-batch); needs --max-batch");
    cmd->add_option("--num-requests", opts->num_requests, "Requests to time")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    cmd->add_option("--prompt-len", opts->prompt_len,
                    "Prompt tokens, BOS included: N, or MIN:MAX drawn uniformly per request")
        ->capture_default_str();
    cmd->add_option("--output-len", opts->output_len,
                    "Tokens to generate: N, or MIN:MAX drawn uniformly per request")
        ->capture_default_str();
    cmd->add_option("--request-rate", opts->request_rate,
                    "Poisson arrivals per second; 0 submits every request at once")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--warmup", opts->warmup, "Untimed requests to run first")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--seed", opts->seed,
                    "Seed for the prompts, lengths and arrivals; request i samples with seed + i")
        ->capture_default_str();
    cmd->add_option("--temperature", opts->temperature,
                    "Sample from softmax(logits / temperature); 0 decodes greedily")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-k", opts->top_k,
                    "When sampling, keep the k most likely tokens; 0 is off")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    cmd->add_option("--top-p", opts->top_p,
                    "When sampling, keep the most likely tokens whose probability reaches p; 1 is "
                    "off")
        ->capture_default_str()
        ->check(CLI::Range(0.0f, 1.0f));
    cmd->add_flag("--stop-at-eos", opts->stop_at_eos,
                  "End a request at EOS instead of always generating its full output length");
    cmd->add_flag("--profile", opts->profile,
                  "Call cudaProfilerStart and cudaProfilerStop around the timed run, for nsys "
                  "--capture-range=cudaProfilerApi and ncu --profile-from-start off");
    cmd->add_option("--out", opts->out, "Per-request CSV to write, for example results/b1.csv")
        ->required();
    cmd->add_option("--summary", opts->summary,
                    "Summary CSV to write (default: the --out name with _summary added)");
    cmd->callback([opts] { run(*opts); });
}

}  // namespace engine::app
