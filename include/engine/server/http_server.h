#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "engine/kv/cache_size.h"
#include "engine/server/chat_api.h"
#include "engine/server/cors.h"
#include "engine/server/serving_loop.h"

namespace engine::server {

using Encode = std::function<std::vector<int32_t>(const std::string& text)>;

struct HttpOptions {
    std::vector<std::string> cors_origins = kDefaultCorsOrigins;
    SamplingDefaults defaults{};
    std::string model = "tinyllama";
    nlohmann::ordered_json meta = nlohmann::ordered_json::object();
    int max_context = 2048;
    int block_size = kKvBlockSize;
};

nlohmann::ordered_json metrics_json(const MetricsSnapshot& m, const nlohmann::ordered_json& meta,
                                    int block_size);

class HttpServer {
public:
    HttpServer(ServingLoop& loop, Encode encode, HttpOptions options);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    int bind(const std::string& host, int port);
    void listen();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::server
