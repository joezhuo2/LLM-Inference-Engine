#include "engine/server/http_server.h"

#include <httplib.h>

#include <chrono>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "engine/tokenizer/chat.h"

namespace engine::server {

namespace {

using Json = nlohmann::ordered_json;
using std::chrono::milliseconds;

constexpr auto kStreamPoll = milliseconds(1000);
constexpr auto kWaitPoll = milliseconds(250);
constexpr size_t kMaxBody = 1 << 20;
constexpr size_t kMinThreads = 16;
constexpr size_t kMaxThreads = 128;

Json latency_json(const std::optional<Percentiles>& p) {
    if (!p) return nullptr;
    return {{"p50", p->p50}, {"p99", p->p99}};
}

void send_json(httplib::Response& res, int status, const Json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

uint64_t random_seed() {
    std::random_device entropy;
    return uint64_t(entropy()) << 32 | entropy();
}

void set_event_stream_headers(httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
}

struct StreamState {
    ServingLoop* loop;
    Ticket ticket;
    Completion completion;
    bool started = false;
    bool done = false;
};

std::string next_stream_text(StreamState& s, httplib::DataSink& sink) {
    if (!s.started) {
        s.started = true;
        return sse_data(role_chunk(s.completion).dump());
    }
    while (sink.is_writable()) {
        const auto event = s.ticket.events->pop(kStreamPoll);
        if (!event) {
            if (s.ticket.events->closed()) break;
            continue;
        }
        if (!event->finish)
            return sse_data(content_chunk(s.completion, event->text, event->step).dump());
        s.done = true;
        return sse_data(finish_chunk(s.completion, *event->finish, event->step).dump()) +
               sse_data("[DONE]");
    }
    return {};
}

}  // namespace

Json metrics_json(const MetricsSnapshot& m, const Json& meta, int block_size) {
    const Counters& c = m.counters;
    return {{"version", 1},
            {"time_s", m.time_s},
            {"meta", meta},
            {"requests",
             {{"waiting", c.waiting},
              {"running", c.running},
              {"finished_total", c.finished_total},
              {"preemptions_total", c.preemptions_total}}},
            {"kv",
             {{"blocks_total", m.blocks_total},
              {"blocks_free", c.blocks_free},
              {"block_size", block_size}}},
            {"tokens", {{"output_total", c.output_total}, {"output_per_s", m.output_per_s}}},
            {"latency", {{"ttft_s", latency_json(m.ttft_s)}, {"tpot_s", latency_json(m.tpot_s)}}},
            {"steps_total", c.steps_total}};
}

struct HttpServer::Impl {
    ServingLoop& loop;
    Encode encode;
    HttpOptions options;
    CorsPolicy cors;
    httplib::Server http;

    Impl(ServingLoop& l, Encode e, HttpOptions o)
        : loop(l), encode(std::move(e)), options(std::move(o)), cors(options.cors_origins) {
        http.new_task_queue = [] { return new httplib::ThreadPool(kMinThreads, kMaxThreads); };
        http.set_payload_max_length(kMaxBody);
        http.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            return route_cors(req, res);
        });
        http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, {{"version", 1}, {"status", "ok"}});
        });
        http.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, metrics_json(loop.metrics(), options.meta, options.block_size));
        });
        http.Get("/v1/steps",
                 [this](const httplib::Request&, httplib::Response& res) { stream_steps(res); });
        http.Post("/v1/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) { chat(req, res); });
    }

    httplib::Server::HandlerResponse route_cors(const httplib::Request& req,
                                                httplib::Response& res) {
        const std::string origin = req.get_header_value("Origin");
        if (req.method != "OPTIONS") {
            if (origin.empty()) return httplib::Server::HandlerResponse::Unhandled;
            for (const auto& [name, value] : cors.response_headers(origin))
                res.set_header(name, value);
            if (cors.allows(origin)) return httplib::Server::HandlerResponse::Unhandled;
            send_json(res, 403,
                      error_json("origin " + origin + " is not allowed; see --cors-origin"));
            return httplib::Server::HandlerResponse::Handled;
        }
        const bool private_network =
            req.get_header_value("Access-Control-Request-Private-Network") == "true";
        for (const auto& [name, value] : cors.preflight_headers(
                 origin, req.get_header_value("Access-Control-Request-Headers"), private_network))
            res.set_header(name, value);
        res.status = cors.allows(origin) ? 204 : 403;
        return httplib::Server::HandlerResponse::Handled;
    }

    void stream_steps(httplib::Response& res) {
        set_event_stream_headers(res);
        auto subscription = loop.steps().subscribe();
        res.set_chunked_content_provider(
            "text/event-stream", [subscription](size_t, httplib::DataSink& sink) {
                const auto step = subscription->pop(kStreamPoll);
                if (!step && subscription->closed()) {
                    sink.done();
                    return true;
                }
                const std::string text = step ? sse_event("step", *step) : ": keepalive\n\n";
                return sink.write(text.data(), text.size());
            });
    }

    void chat(const httplib::Request& req, httplib::Response& res) {
        try {
            const ChatRequest request = parse_chat_request(req.body, options.defaults);
            Submission submission{.prompt = encode(chat_prompt(request.messages, true)),
                                  .params = request.params,
                                  .stop = request.stop};
            const int prompt_tokens = int(submission.prompt.size());
            submission.max_tokens =
                resolve_max_tokens(request.max_tokens, options.defaults.max_tokens, prompt_tokens,
                                   options.max_context);
            submission.params.seed = request.seed.value_or(random_seed());
            const Ticket ticket = loop.submit(std::move(submission));
            const Completion completion{ticket.id, int64_t(std::time(nullptr)), options.model};
            if (request.stream)
                stream_chat(res, ticket, completion);
            else
                wait_chat(req, res, ticket, completion, prompt_tokens);
        } catch (const ApiError& e) {
            send_json(res, e.status(), error_json(e.what()));
        }
    }

    void stream_chat(httplib::Response& res, const Ticket& ticket, const Completion& completion) {
        set_event_stream_headers(res);
        auto state = std::make_shared<StreamState>(StreamState{&loop, ticket, completion});
        res.set_chunked_content_provider(
            "text/event-stream",
            [state](size_t, httplib::DataSink& sink) {
                if (state->done) {
                    sink.done();
                    return true;
                }
                const std::string text = next_stream_text(*state, sink);
                if (text.empty()) return false;
                return sink.write(text.data(), text.size());
            },
            [state](bool) {
                if (!state->done) state->loop->cancel(state->ticket.id);
            });
    }

    void wait_chat(const httplib::Request& req, httplib::Response& res, const Ticket& ticket,
                   const Completion& completion, int prompt_tokens) {
        std::string text;
        while (true) {
            const auto event = ticket.events->pop(kWaitPoll);
            if (!event) {
                if (ticket.events->closed()) break;
                if (req.is_connection_closed()) {
                    loop.cancel(ticket.id);
                    return;
                }
                continue;
            }
            text += event->text;
            if (!event->finish) continue;
            send_json(res, 200,
                      completion_json(completion, text, *event->finish, event->step, prompt_tokens,
                                      event->completion_tokens));
            return;
        }
        send_json(res, 500, error_json("the engine dropped the request"));
    }
};

HttpServer::HttpServer(ServingLoop& loop, Encode encode, HttpOptions options)
    : impl_(std::make_unique<Impl>(loop, std::move(encode), std::move(options))) {}

HttpServer::~HttpServer() = default;

int HttpServer::bind(const std::string& host, int port) {
    const bool bound = port == 0 ? (port = impl_->http.bind_to_any_port(host)) > 0
                                 : impl_->http.bind_to_port(host, port);
    if (!bound) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    return port;
}

void HttpServer::listen() {
    impl_->http.listen_after_bind();
}

void HttpServer::stop() {
    impl_->http.stop();
}

}  // namespace engine::server
