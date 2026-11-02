#include "engine/server/host_check.h"

#include <algorithm>
#include <cctype>

namespace engine::server {

namespace {

bool valid_port(const std::string& port) {
    return !port.empty() && std::all_of(port.begin(), port.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::string lowered(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return text;
}

}  // namespace

std::string host_name(const std::string& host_header) {
    const bool bracketed = !host_header.empty() && host_header.front() == '[';
    const auto name_end = bracketed ? host_header.find(']') : host_header.find(':');
    if (name_end == std::string::npos) return bracketed ? "" : lowered(host_header);
    const auto port_colon = bracketed ? name_end + 1 : name_end;
    if (port_colon < host_header.size()) {
        if (host_header[port_colon] != ':' || !valid_port(host_header.substr(port_colon + 1)))
            return "";
    }
    return lowered(host_header.substr(0, port_colon));
}

bool host_allowed(const std::string& host_header, const std::vector<std::string>& allowed) {
    const std::string name = host_name(host_header);
    return !name.empty() && std::find(allowed.begin(), allowed.end(), name) != allowed.end();
}

}  // namespace engine::server
