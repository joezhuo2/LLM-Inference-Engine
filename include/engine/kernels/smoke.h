#pragma once

namespace engine::kernels {

// Adds two n-element vectors on the GPU and returns the max abs error vs the CPU result.
float smoke_vector_add(int n);

}  // namespace engine::kernels
