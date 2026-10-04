#pragma once

// Same type as cudaStream_t, declared without CUDA headers so public interfaces stay plain C++.
struct CUstream_st;

namespace engine {

using Stream = CUstream_st*;

}  // namespace engine
