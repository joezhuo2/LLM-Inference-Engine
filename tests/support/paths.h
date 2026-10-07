#pragma once

#include <cstdlib>
#include <filesystem>

namespace engine::test {

inline std::filesystem::path model_dir() {
    if (const char* env = std::getenv("ENGINE_MODEL_DIR")) return env;
    return ENGINE_MODEL_DIR;
}

}  // namespace engine::test
