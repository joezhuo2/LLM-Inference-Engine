#pragma once

#include <string>
#include <vector>

namespace engine::server {

inline const std::vector<std::string> kLoopbackHosts = {"localhost", "127.0.0.1", "[::1]"};

std::string host_name(const std::string& host_header);
bool host_allowed(const std::string& host_header, const std::vector<std::string>& allowed);

}  // namespace engine::server
