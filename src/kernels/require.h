#pragma once

#include <stdexcept>
#include <string>

namespace engine {

inline void require(bool ok, const char* op, const char* what) {
    if (!ok) throw std::invalid_argument(std::string(op) + ": " + what);
}

}  // namespace engine
