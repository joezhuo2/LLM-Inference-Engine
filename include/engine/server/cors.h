#pragma once

#include <string>
#include <utility>
#include <vector>

namespace engine::server {

using HeaderList = std::vector<std::pair<std::string, std::string>>;

inline const std::vector<std::string> kDefaultCorsOrigins = {"http://localhost:3000"};

void validate_origin(const std::string& origin);

class CorsPolicy {
public:
    explicit CorsPolicy(std::vector<std::string> origins);

    bool allows(const std::string& origin) const;
    HeaderList response_headers(const std::string& origin) const;
    HeaderList preflight_headers(const std::string& origin, const std::string& request_headers,
                                 bool private_network) const;

    const std::vector<std::string>& origins() const { return origins_; }

private:
    std::vector<std::string> origins_;
};

}  // namespace engine::server
