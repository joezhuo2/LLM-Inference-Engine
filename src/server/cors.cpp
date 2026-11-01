#include "engine/server/cors.h"

#include <algorithm>
#include <stdexcept>

namespace engine::server {

void validate_origin(const std::string& origin) {
    const auto scheme_end = origin.find("://");
    const std::string scheme = origin.substr(0, scheme_end);
    const bool bad = scheme_end == std::string::npos || (scheme != "http" && scheme != "https") ||
                     origin.size() == scheme_end + 3 ||
                     origin.find('/', scheme_end + 3) != std::string::npos;
    if (bad)
        throw std::invalid_argument("CORS origin '" + origin +
                                    "' must look like https://host or http://host:port, with no "
                                    "path or trailing slash");
}

CorsPolicy::CorsPolicy(std::vector<std::string> origins) : origins_(std::move(origins)) {
    for (const std::string& origin : origins_) validate_origin(origin);
}

bool CorsPolicy::allows(const std::string& origin) const {
    return std::find(origins_.begin(), origins_.end(), origin) != origins_.end();
}

HeaderList CorsPolicy::response_headers(const std::string& origin) const {
    if (!allows(origin)) return {{"Vary", "Origin"}};
    return {{"Access-Control-Allow-Origin", origin}, {"Vary", "Origin"}};
}

HeaderList CorsPolicy::preflight_headers(const std::string& origin,
                                         const std::string& request_headers,
                                         bool private_network) const {
    if (!allows(origin)) return {{"Vary", "Origin"}};
    HeaderList headers = response_headers(origin);
    headers.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    headers.emplace_back("Access-Control-Allow-Headers",
                         request_headers.empty() ? "Content-Type" : request_headers);
    headers.emplace_back("Access-Control-Max-Age", "600");
    if (private_network) headers.emplace_back("Access-Control-Allow-Private-Network", "true");
    return headers;
}

}  // namespace engine::server
