#pragma once

#include <cuda_fp8.h>

#include <cstdint>

namespace engine::test {

inline uint8_t to_fp8(float f) {
    return __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
}

inline float from_fp8(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}

inline bool fp8_is_nan(uint8_t b) {
    return (b & 0x7F) == 0x7F;
}

}  // namespace engine::test
