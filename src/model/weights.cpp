#include "engine/model/weights.h"

#include <algorithm>
#include <cstddef>

namespace engine {

namespace {

Tensor view(const Region& r, void* base) {
    Tensor t;
    t.data = static_cast<std::byte*>(base) + r.offset;
    t.dtype = DType::BF16;
    t.ndim = int(r.shape.size());
    std::copy(r.shape.begin(), r.shape.end(), t.shape.begin());
    return t;
}

}  // namespace

ModelWeights bind_weights(const WeightLayout& layout, void* base) {
    ModelWeights w;
    w.embed = view(layout.embed, base);
    w.final_norm = view(layout.final_norm, base);
    w.lm_head = view(layout.lm_head, base);
    for (const LayerLayout& l : layout.layers) {
        w.layers.push_back({view(l.ln1, base), view(l.wqkv, base), view(l.wo, base),
                            view(l.ln2, base), view(l.wgu, base), view(l.wdown, base)});
    }
    return w;
}

}  // namespace engine
