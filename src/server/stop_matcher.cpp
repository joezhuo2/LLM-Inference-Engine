#include "engine/server/stop_matcher.h"

#include <algorithm>
#include <utility>

namespace engine::server {

StopMatcher::Result StopMatcher::push(std::string_view piece) {
    held_ += piece;
    size_t first = std::string::npos;
    for (const std::string& stop : stops_) first = std::min(first, held_.find(stop));
    if (first != std::string::npos) {
        Result r{held_.substr(0, first), true};
        held_.clear();
        return r;
    }
    const size_t keep = partial_match();
    Result r{held_.substr(0, held_.size() - keep), false};
    held_.erase(0, held_.size() - keep);
    return r;
}

std::string StopMatcher::flush() {
    return std::exchange(held_, {});
}

size_t StopMatcher::partial_match() const {
    size_t longest = 0;
    for (const std::string& stop : stops_) {
        for (size_t n = std::min(stop.size() - 1, held_.size()); n > longest; --n) {
            if (std::string_view(held_).ends_with(std::string_view(stop).substr(0, n))) {
                longest = n;
                break;
            }
        }
    }
    return longest;
}

}  // namespace engine::server
