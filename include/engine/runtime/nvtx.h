#pragma once

#include <nvtx3/nvToolsExt.h>

#include <utility>

namespace engine {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

template <typename F>
void ranged(const char* name, F&& body) {
    const NvtxRange range(name);
    std::forward<F>(body)();
}

}  // namespace engine
