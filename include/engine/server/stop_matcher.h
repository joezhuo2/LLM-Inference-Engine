#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::server {

class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    struct Result {
        std::string text;
        bool stopped = false;
    };

    Result push(std::string_view piece);
    std::string flush();

private:
    std::vector<std::string> stops_;
    std::string held_;

    size_t partial_match() const;
};

}  // namespace engine::server
