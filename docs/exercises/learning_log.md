# Foundational
### Core Ideas
- **SIMT** => single instruction, multiple threads
- **kernel** => function that can be run by multiple GPU threads at the same time with different sets of data
- **memory overflow** => program tries to write more memory than physically available, can lead to crasahes, out of bounds errors, or launch fails
- **thread** => thing that performs its assigned set of instructions with its own data
- **block** => array of threads, have shared memory, live on the same SM
    - **Streaming Multiprocessor (SM)** => 
- **grid** => array of blocks
- **warp** => physical instruction executor with size of 32 threads that share the same instruction set with different data. 
    - **warp divergence** => conditional logic, warp starts by executing the instruction set for one group while disabling other groups, then the next, and so on

### Core C++
- **`&` (lvalue reference)** => reference to memory location of variable
- **`&&` (rvalue reference)** => reference to a temporary object (to take the object's resources)
- **lvalue** => object with an identity (stays in memory)
- **rvalue** => temporary value with no identity (temporary)
- **`std::move`** => casts an lvalue to an rvalue
- **`*` (in expression)** => dereference - access to value inside a memory location
- **`T*` (in constructor)** => pointer to memory address
- **move semantics (eg. move constructor/assignment)** => setting the pointers of one object of anothers and then setting that object's pointers to `nullptr`
- **move constructor (eg. `Buffer(Buffer&&)`)** => constructs a new object by stealing the pointers of another
- **move assignment (eg. `operator=(Buffer&&)`)** => must first delete existing pointers to prevent memory leaks, and check if the move target is itself 
- **destructor (eg. `~Buffer()`)** => called when object is no longer in scope or destroyed, used for cleanup
- **copy constructor (eg. `Buffer(const Buffer&)`)** => prevents copy constructors from being called (eg. `Myclass a; Myclass b = a;` => b (brand new) is initialized as a copy of a) since enabling it would allow 2 objects to have the same memory pointer => leads to `double free` error when they are released 
- **copy assignment (eg. `operator=(const Buffer&)`)** => same as copy constructor, but can also cause memory leaks (assigning value of one to original without releasing the original pointer) (eg. `Myclass a; b = a;` => b's values are updated to a's values)
- **`= delete`** => declared/exists, compiler is banned from calling this method
- **`= default`** => use default implementation 
- **`size_t`** => unsigned int that can match the system architecture (can be 32 bit or 64 bit), no overflow or negative numbers
- **`nullptr`** => pointer type, null pointer, used to clear confusion by setting pointers to `0` or `NULL`, which are `int`s
- **const** => for variables: immutability, for methods: object state is immutable 
- **`sizeof`** => returns (as a `std::size_t`) the size of the parameter in bytes
- **`sizeof(float)`** => memory size occupied by a singular precision number (usually 4 bytes)
- **`#pragma once` / header files** => 
- **`class` access specifiers (`private` / `public`)** => 
- **default member initializer (`void* p{nullptr}`)** => 
- **fixed-width integers (`uint8_t`, `uint64_t`)** => 
- **`template <typename T>`** => 
- **lambda and capture `[&]`** => 
- **macro (`#define`) and the `do { } while (0)` idiom** => 
- **scope resolution `::` (eg. `::open`, `std::`)** => 
- **pointer arithmetic** => arithemetic using pointers (+, -, comparison), changes/compares where the pointer is pointing, not the data at the pointer locationo
- **stack** => uses LIFO, CPU auto manage, ultra fast, small, can get stack overflow, things disappear after function finishes
- **heap** => ask for specific amount of memory, computer finds location with that much free memory and returns the pointe, slower, massive size, long-duration, stays until you explicitly delete it

# Part 0 - Foundations
## Buffer (Task 1)
### Core Ideas
- **RAII** => coding practice where resources are allocated in the constructor and set free in the destructor
- **ownership** => which entity is responsible for managing a resource's lifecycle (creation, access, cleanup)
- **leak** => allocating resource to a location, but then losing access to that pointer without freeing it, causing it to never be usable until the program exits
- **double free** => releases the same resource allocation twice, or 2 objects both think they own the pointer and both remove it 
- **use after free** => trying to access ptr after it has been released (through free, delete, delete[], etc.)

### special member functions
- **constructor (and why `explicit`)** => 
- **member initializer list** => 
- **rule of five (layer 1: just what the five are and why they come together)** => 
- **default constructor (`Buffer() = default`)** => 
- **implicit conversion** => 
- **temporary object (eg. `a = Buffer(128)`)** => 

### Values and references
- **moved-from state** => 
- **self-move-assignment (`b = std::move(b)`)** => 
- **std::exchange** => `a = std::exchange(b, newVal)` updates `b` to `newVal` and sets `a` to the old value of `b`, can take the the buffer of another object and then update its pointer to null, all in one call
- **`this` pointer** => 
- **`this != &other` self-assignment check** => 
- **const reference parameters (`const Buffer&`)** => 
- **returning `*this` from `operator=`** => 
- **const overloading (`data()` vs `data() const`)** => 
- **why noexcept moves matter for `std::vector` reallocation** => 

### Memory
- **`void*`** => 
- **`malloc`** => finds free space for requested memory size and returns the pointer to the memory location
- **`std::free`** => 
- **`std::memset`** => 
- **`std::bad_alloc`** => 
- **`noexcept`** => 
- **`std::vector<T>`** => basically a dynamic array
- **`malloc(0)` and zero-size allocation** => 
- **dangling pointer** => 

### Testing
- **ASSERT vs EXPECT (GoogleTest)** => both mark tests as failed, assert quits the current test, expect continues the test, assert is used for when continuing the test would be pointless/crash
- **`static_assert` + `std::is_copy_constructible_v` / `is_move_constructible_v`** => 
- **`std::declval`** => 
- **AddressSanitizer** => 
- **`TEST` macro and test suites** => 
- **`EXPECT_EQ` / `EXPECT_NE` / `EXPECT_TRUE`** => expect equals (2 vals), expect not equals (2 vals), expect true (bool)
- **`decltype` + `std::is_same_v`** => 
- **`std::is_nothrow_move_constructible_v`** => 
- **`gtest_discover_tests` / CTest** => 
- **`-fsanitize=address` flags** => 
- **compile-time vs runtime checks** => 

## Safe Tensor Reader (Task 2)
### Core Ideas
- **file descriptor** => 
- **POSIX** => 
- **RAII applied to OS resources (FDCloser / MemoryUnmapper)** => 
- **untrusted input validation (corrupt or malicious headers)** => 
- **integer overflow in bounds checks (`a + b > n` vs `b > n - a`)** => 
- **zero-copy** => 
- **unsigned underflow (`file_size - 8`)** => 
- **local struct as RAII guard** => 

### Opening and inspecting files
- **`open`** => opens a file and gives a file descriptor
- **`O_RDONLY`** => flag that lets open files be read from only, write fails
- **`close`** => closes the file descriptor (file is left untouched)
- **`struct stat` (sys/stat.h)** => 
- **`fstat(fd, &sb)`** => 
- **`st_size`, `off_t` -> `size_t` conversion** => 
- **`errno`** => 
- **`std::strerror(errnum)`** => 
- **`-1` as the syscall failure convention** => 
- **`c_str()`** => 

### Memory mapping
- **`mmap`** => 
- **`PROT_READ` / prot flags** => 
- **`MAP_SHARED` vs MAP_PRIVATE** => 
- **`MAP_FAILED`** => 
- **`munmap`** => 
- **virtual memory and lazy page faults (why mapping 2 GB is instant)** => 
- **page cache** => 
- **page size and alignment (`sysconf(_SC_PAGESIZE)`)** => 

### Bytes and types
- **little endian unsigned int** => 
- **std::endian + static_assert** => 
- **why `sizeof(uint64_t)` instead of 8** => 
- **`std::memcpy`** => 
- **strict aliasing (why memcpy instead of `*(uint64_t*)ptr`)** => 
- **`static_cast`** => 
- **`reinterpret_cast`** => 
- **byte buffer (`const uint8_t*`)** => 
- **unaligned access** => 

### Safetensors format
- **layout (8-byte length, JSON header, tensor data)** => 
- **header space padding (8-byte alignment)** => 
- **data_offsets are relative to the data section** => 
- **`__metadata__`** => 
- **BF16 vs FP16 vs FP32** => 
- **shape -> byte size (numel * dtype size)** => 
- **keys sorted as strings (`layers.10` before `layers.2`)** => 
- **grouped-query attention (why k_proj/v_proj are 256 wide)** => 

### Errors and program entry
- **`throw` / `try` / `catch`** => 
- **`std::runtime_error`** => 
- **uncaught exception -> std::terminate (the "Aborted (core dumped)")** => 
- **argc / argv** => 
- **`std::cerr` vs `std::cout`** => 
- **stack unwinding (why guards run on `throw`)** => 
- **`what()`** => 
- **exit codes (`return 1`)** => 

### Build
- **`cmake_minimum_required`, `project`** => 
- **`CMAKE_CXX_STANDARD` / `CMAKE_CXX_STANDARD_REQUIRED`** => 
- **`add_executable`** => 
- **`target_compile_options`** => 
- **`-Wall` `-Wextra` `-Wpedantic`** => 
- **out-of-source builds (`cmake -B build`, `cmake --build build`)** => 
- **header-only code and `inline` (why `static inline` in a class is redundant)** => 
- **`add_subdirectory`** => 
- **`target_link_libraries`** => 
- **`FetchContent`** => 
- **`CMakePresets.json` (configure/build presets)** => 
- **`CMAKE_BUILD_TYPE` (Debug vs Release)** => 

## Bandwith Test (Task 3/CUDA Task 1)
### Execution model
- **`__global__`** => 
- **`<<<...>>>` (execution configuration operator)** => 
- **`blockIdx` / `blockIdx.x`** => 
- **`blockDim.x`** => 
- **`threadIdx`** => 
- **thread global index calculation** =>
- **global thread index (`blockIdx.x * blockDim.x + threadIdx.x`)** => 
- **bounds check (`if (i < n)`)** => 
- **ceiling division for grid size** => 
- **asynchronous kernel launch** => 
- **`cudaDeviceSynchronize()`** => 
- **threads per block / blocks per grid (launch config)** => 
- **default stream (in-order execution)** => 

### Memory
- **memory hierarchy** => 
- **memory coalescing** => 
- **host vs device memory** => 
- **`cudaMalloc()`** => allocates free high speed memory to CPU host
- **`cudaMallocManaged()`** => 
- **managed memory page migration** => 
- **PCIe bottleneck** => 
- **`cudaMemcpy()`** => 
- **`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`** => 
- **`cudaMemset()`** => 
- **`cudaFree()`** => frees high speed memory allocated to CPU host
- **treating `*a`, `*b`, `*c` as arrays but they are defined as float pointers** => 
- **`std::fill`** => 
- **`delete[]`** => 
- **`cudaMalloc(&d, bytes)` taking `void**`** => 
- **host pointer vs device pointer** => 
- **`1 << 26`** => 
- **GB vs GiB** => 
- **memory bus width and memory clock (theoretical peak)** => 

### Timing and measurement
- **`cudaEvent_t`** => 
- **`cudaEventCreate()`** => 
- **`cudaEventRecord()`** => 
- **`cudaEventElapsedTime()`** => 
- **`cudaEventDestroy()`** => 
- **warmup launch** => 
- **`3.0 * bytes`** => 2 bytes used for reading b[i] and a[i] and one for writing to c[i]
- **`/1000.0`, `/1e9`** => turn ms to s, byte to gigabytes
- **memory-bound** => kernel speed limited by how fast data moves from/to memory
- **compute-bound** => lernel speed limited by how fast each core can do calculations
- **effective bandwidth** => measured rate that a real run achieved 
- **theoretical peak** => hardware limit calculated based on specs
- **power limit / clock throttling** => 
- **`a << b`** => 
- **`cudaEventSynchronize()`** => 
- **`cudaGetLastError()`** => 
- **`cudaError_t` / `cudaSuccess` / `cudaGetErrorString()`** => 
- **`CUDA_CHECK` macro** => 

### Verification and tooling
- **correctness verification (`h[i] != 3.0f`)** => 
- **`nvcc`** => 
- **`-arch=native` / `sm_120`** => 
- **`compute-sanitizer`** => 
- **`ncu` (Nsight Compute)** => 
- **Speed of Light (ncu section)** => 
- **DRAM Throughput % vs Compute (SM) Throughput %** => 
- **ncu clock control (locked base clocks)** => 
- **`-lineinfo`** => 
- **`.ncu-rep` reports** => 

## Three way Summation (Task 4/CUDA Task 2)
### Core Ideas
- **`reductions`** => turning large collections of values into single value
- **`naive atomics`** => unoptimized code that has many threads call the same function concurrently
- **`shared memory`** => user managed on-chip SRAM that is shared across threads in the same block
- **`warp shuffle`** => allows threads within the same warp to share data with each other without reading or writing
- **`warp reduction`** => computing result using all threads of a warp using warp shuffle
- **warp / block / grid reduction hierarchy** => 
- **atomic contention (serialization on one address)** => 
- **floating point non-associativity (summation order)** => 
- **one element per thread vs grid-stride loop** => 
- **vectorized loads (`float4`)** => 

### Atomics and shared memory
- **`atomicAdd(address, val)`** => uses one uninterupted action instead of 3 (RMW) and guarantees thread safety and correct answer by queuing concurrent calls
- **`__shared__`** => 
- **dynamic shared memory** => 
- **`__syncthreads()`** => 
- **binary tree reduction loop** => 
- **`>>=` (bitwise right shift), same as `a = >> a`** => shifts all bits to the right and removes the LSB, same as division for unsigned integers, but NOT for signed integers
- **`__device__`** => 
- **static vs dynamic shared memory (`static __shared__`)** => 
- **`__syncthreads()` barrier** => 
- **race condition** => 

### Warp shuffle
- **`__shfl_down_sync(mask, val, offset)`** => 
- **`0xffffffff` mask** => 
- **lane (`threadIdx.x % 32`) and warp ID** => 
- **`warpReduceSum` helper pattern** => 

### Host code and measurement
- **`cudaMemcpyHostToDevice`** => 
- **`free()`** => 
- **`constexpr`** => 
- **median of 5 launches** =>
- **`template <typename Launch>` + lambda for `bench()`** => 
- **`std::sort` for the median** => 
- **`#include` of `.cu` files** =>

## PyTorch TinyLlama Forward Pass (Task 5/Transformer Task 1)
### Core Ideas
- **reference oracle** => 
- **decoder-only transformer** => 
- **decoder layer (pre-norm residual block)** => 
- **residual connection** => 
- **hidden state** => 
- **logits** => 
- **greedy decoding (`argmax`)** => 
- **next-token prediction (why only `logits[0, -1, :]` is used)** => 
- **autoregressive** => 
- **tokenizer (BOS token)** => 
- **no KV cache yet (full sequence recomputed per forward)** => 

### Model shapes
- **`hidden_size` (2048)** => 
- **`num_attention_heads` (32) and `head_dim` (64)** => 
- **`num_key_value_heads` (4)** => 
- **grouped-query attention (GQA)** => 
- **GQA group size (32 / 4 = 8, query head `h` uses KV head `h // 8`)** => 
- **`intermediate_size` (5632)** => 
- **`vocab_size` (32000)** => 
- **`num_hidden_layers` (22)** => 
- **tied vs untied embeddings (`lm_head` is separate)** => 
- **tensor shape `[batch, seq, heads, head_dim]` vs `[batch, heads, seq, head_dim]`** => 

### Normalization
- **RMSNorm** => 
- **RMSNorm vs LayerNorm (no mean subtraction, no bias)** => 
- **`eps` (1e-5)** => 
- **learned scale weight (`input_layernorm`, `post_attention_layernorm`, `model.norm`)** => 
- **`torch.rsqrt`** => 
- **why the norm statistics are computed in FP32** => 
- **cast to BF16 before multiplying by the weight** => 
- **pre-norm vs post-norm** => 

### Attention
- **query / key / value projections (`q_proj`, `k_proj`, `v_proj`)** => 
- **`nn.Linear` weight layout `[out, in]` and `x @ W.T`** => 
- **`.view()` to split heads** => 
- **`.transpose(1, 2)`** => 
- **`.contiguous()`** => 
- **`repeat_interleave` (expanding KV heads for GQA)** => 
- **scaled dot-product attention** => 
- **scaling by `sqrt(head_dim)` (divide by 8.0)** => 
- **`q @ k.transpose(-2, -1)`** => 
- **causal mask** => 
- **`torch.triu(diagonal=1)`** => 
- **`-inf` mask and softmax** => 
- **softmax in FP32 then cast back to BF16** => 
- **attention output projection (`o_proj`)** => 
- **merging heads back to `hidden_size`** => 

### RoPE
- **rotary position embedding (RoPE)** => 
- **`inv_freq` (`1 / theta^(2i/d)`)** => 
- **`rope_theta` (10000)** => 
- **`torch.outer(t, inv_freq)`** => 
- **cos/sin cache (precomputed per position)** => 
- **`rotate_half` pairing (element `i` with `i + 32`)** => 
- **`torch.cat((freqs, freqs), dim=-1)`** => 
- **why RoPE is applied to Q and K but not V** => 
- **cos/sin broadcast with `unsqueeze(0).unsqueeze(0)`** => 
- **relative position encoding through rotation** => 

### MLP
- **SwiGLU** => 
- **gate / up / down projections** => 
- **SiLU (`F.silu`)** => 
- **elementwise gating (`silu(gate) * up`)** => 
- **MLP expansion and contraction (2048 -> 5632 -> 2048)** => 

### Numerics
- **BF16 (8 exponent bits, 7 mantissa bits)** => 
- **BF16 vs FP16 vs FP32 range and precision** => 
- **FP32 accumulation inside matmul** => 
- **rounding order effects (why results match only within tolerance)** => 
- **`.float()` and `.to(dtype)` casts** => 
- **`torch.bfloat16`** => 

### Weights and PyTorch
- **safetensors `load_file`** => 
- **state dict key names (`model.layers.{i}.self_attn.q_proj.weight`)** => 
- **moving tensors to the GPU (`.to(device=...)`)** => 
- **`torch.no_grad()`** => 
- **embedding lookup by indexing (`weights[...][tokens]`)** => 
- **`AutoTokenizer`** => 
- **`tokenizer.decode`** => 
- **`torch.argmax`** => 
- **`.item()`** => 
- **`dtype` and `device` of masks and caches** =>

### KV cache
- **KV (Key-Value) cache** => AI makes notes about every word it has already generated to prevent recomputing all tokens from scratch by storing it (usually in GPU VRAM)
- **why K and V are cached but Q is not** => K (read from) and V (meaning) need to be used in the future, and Q (questions about past tokens) are not needed by future tokens
- **prefill** => initial calculation of all KV tensors for all prompt tokens and storing into KV cache
- **decode** => phase when AI generates one token at a time and adding new KV to the cache
- **prefill vs decode (matrix x matrix vs matrix x vector)** => 
- **static preallocated cache vs `torch.cat` growing cache** => 
- **cache shape `[layers, batch, kv_heads, max_len, head_dim]`** => 
- **cache size per token (2 x layers x kv_heads x head_dim x 2 bytes)** => 
- **in-place slice assignment (`kc[i, :, :, start:end] = k`)** => 
- **`start` offset (absolute position of the first new token)** => 
- **slicing the valid cache region (`[:end]`)** => 
- **RoPE applied before caching (keys are stored already rotated)** => 
- **RoPE cos/sin sliced by absolute position (`cos[:, :, start:end]`)** => 
- **why the cache removes the O(n^2) recompute per token** => 
- **rectangular score matrix `[T, end]` (new queries vs all cached keys)** => 
- **causal mask offset (`triu(start + 1)`)** => 
- **decode mask is empty (`T = 1` sees every cached key)** => 
- **GQA shrinks the cache 8x (4 KV heads instead of 32)** => 
- **`last_only` (lm_head on the last token only)** => 
- **why `lm_head` is skipped for prompt positions during generation** => 
- **why zeros in the unused cache region never get read** => 

### Generation loop
- **generation loop** => 
- **greedy decoding with a cache (prefill once, then one token per step)** => 
- **`pos` counter (position of the next token)** => 
- **feeding the sampled token back as the next input** => 
- **`torch.tensor([[tok]])` (batch 1, seq 1 input)** => 
- **EOS token (`tokenizer.eos_token_id`)** => 
- **stop conditions (EOS or max new tokens)** => 
- **max context length (`MAX_LEN` = 2048)** => 
- **`.item()` GPU sync per token** => 

### Verification against HF
- **oracle comparison** => 
- **golden test** => 
- **`output_hidden_states=True`** => 
- **`hidden_states` alignment (embedding + 22 layers, last entry is post-norm)** => 
- **relative error (`||a - b|| / ||b||`)** => 
- **mean absolute logit error** => 
- **why tolerances instead of bit-exact equality** => 
- **error growth across layers** => 
- **teacher forcing** => 
- **teacher-forced argmax match rate** => 
- **free-running greedy match** => 
- **first divergence index** => 
- **near-tie logits (why argmax can flip on tiny error)** => 
- **`hf.generate(do_sample=False)`** => 
- **`attention_mask`** => 
- **`.eval()`** => 
- **`from_pretrained` with `torch_dtype`** => 
- **PASS/FAIL thresholds from the plan (1e-2, 5e-2, 0.05, 99%, 90%)** => 
- **debugging method (first layer that diverges, then first op)** =>

# Part 1 - Engine (M1 Foundations)
## Build Scaffolding (Branch 1)
### CMake and dependencies
- **`FetchContent` with `FIND_PACKAGE_ARGS`** => 
- **FetchContent name vs package name** => 
- **CLI11 subcommands** => 
- **`target_link_libraries` `PUBLIC` vs `PRIVATE`** => 

### CI and tooling
- **ctest labels and `-L` / `-LE`** => 
- **`todo_joe` label** => 
- **GitHub Actions job / matrix** => 
- **`python -m py_compile`** => 
- **clangd `CompilationDatabase`** => 
- **`ReflowComments: false`** => 

### Python
- **pinned requirements (`==`)** => 
- **`--extra-index-url` and local versions (`2.11.0+cu128`)** => 

## Tensor and Buffers (Branch 2)
### C++
- **`enum class X : uint8_t`** => 
- **`constexpr` function** => 
- **switch case fallthrough labels** => 
- **owning vs non-owning (view) types** => 
- **`std::initializer_list` constructor** => 
- **`static constexpr` member** => 
- **forward declaration / incomplete type** => 
- **type alias (`using Stream = CUstream_st*`)** => 
- **`std::is_same_v` in a `static_assert`** => 
- **`std::bit_cast<float>(uint32_t)`** => 
- **destruction order of statics** => 

### CUDA error handling
- **`cublasStatus_t` / `cublasGetStatusName`** => 
- **`NDEBUG` (Debug vs Release)** => 
- **asynchronous errors** => 
- **`cudaGetLastError`** => 
- **stream capture (`cudaStreamIsCapturing`)** => 
- **capture modes (`cudaStreamCaptureModeThreadLocal`)** => 

### Memory
- **NaN bit patterns** => 
- **`cudaMemset`** => 
- **`cudaFree(nullptr)`** => 
- **pageable vs pinned (page-locked) memory** => 
- **`cudaMallocHost` / `cudaFreeHost`** => 
- **`cudaMemcpyAsync` from pageable memory** => 
- **`cudaPointerGetAttributes`** => 

### Testing
- **death tests (`EXPECT_DEATH(stmt, regex)`)** => 
- **`GTEST_SKIP()`** => 
- **`compute-sanitizer --leak-check full`** => 

## cuBLAS GEMM (Branch 3)
### Numerics
- **round-to-nearest-even (float to BF16)** => 
- **the `+ 0x7FFF + lsb` rounding trick** => 
- **`0x1p-8f` hex float literals** => 

### cuBLAS setup
- **`cublasHandle_t` (`cublasContext*`)** => 
- **`cublasSetStream`** => 
- **`cublasSetWorkspace` and why a fixed workspace matters for CUDA Graphs** => 
- **`cublasSetStream` resets the workspace (call order)** => 
- **`cublasGetStream`** => 

### Row-major GEMM through cuBLAS
- **column-major vs row-major storage** => 
- **a row-major matrix read as column-major is its transpose** => 
- **the row-major trick (`Y^T = W X^T`)** => 
- **`CUBLAS_OP_T` / `CUBLAS_OP_N`** => 
- **`m`, `n`, `k` in `cublasGemmEx` (out, B, in)** => 
- **leading dimension (`lda`, `ldb`, `ldc`)** => 
- **`cublasGemmEx` (mixed-precision GEMM)** => 
- **`CUDA_R_16BF`** => 
- **`CUBLAS_COMPUTE_32F` (FP32 accumulation)** => 
- **`alpha` / `beta` and why they are `float` here** => 
- **`CUBLAS_GEMM_DEFAULT` (algorithm selection)** => 
- **Tensor Cores** => 
- **GEMM vs GEMV (B = 1)** => 

### Testing
- **test fixtures (`TEST_F`, `SetUp`, `TearDown`)** => 
- **parameterized tests (`TEST_P`, `INSTANTIATE_TEST_SUITE_P`)** => 
- **CPU reference on BF16-rounded inputs** => 
- **mixed absolute and relative tolerance** => 
- **`ASSERT_NEAR`** => 

## CI Hardening (build/ci-timeouts)
- **`timeout-minutes` (and GitHub's 6-hour default)** => 
- **workflow-level `env:` in GitHub Actions** => 
- **`apt-get -o Acquire::Retries` / `Acquire::http::Timeout`** => 
- **package mirrors (why the CUDA container's apt worked while the runner's hung)** => 
- **pinning the clang-format major version (why 17 and 18 can disagree)** => 
- **`cmd || (fallback)` in a shell step** => 
## Model Config (Branch 4)
### Parsing
- **`nlohmann::json::parse`** => 
- **`json::at` vs `json::value` (required vs optional fields)** => 
- **`json::get<T>()`** => 
- **`json::parse_error` / `json::type_error`** => 
- **rethrowing with context (wrapping a library exception)** => 
- **`[[noreturn]]`** => 
- **function templates (`template <typename T> T field(...)`)** => 
- **`std::string_view` parameters** => 
- **`std::filesystem::path`** => 
- **reading a whole file (`std::ifstream` + `rdbuf()`)** => 

### Config semantics
- **`rope_scaling` (and why the engine rejects it)** => 
- **`hidden_act` (`silu`)** => 
- **`attention_bias` / `mlp_bias`** => 
- **`pretraining_tp`** => 
- **derived widths (`kv_dim`, `qkv_dim`, `group_size`)** => 
- **fail fast on unsupported configs** => 

### Testing
- **raw string literals (`R"(...)"`)** => 
- **asserting on the exception message, not just the type** => 
- **`target_compile_definitions` (passing a path into C++ as a macro)** => 
- **`std::getenv` overrides** => 
- **skipping tests that need local data (`GTEST_SKIP` when the model is missing)** => 

## Safetensors Loader Stub (Branch 5)
### Mapping files
- **`MAP_PRIVATE` vs `MAP_SHARED` for a read-only mapping** => 
- **closing the file descriptor right after `mmap`** => 
- **`O_CLOEXEC`** => 
- **why `mmap` of a zero-length file fails** => 
- **`std::span<const std::byte>`** => 
- **`std::byte` vs `uint8_t`** => 
- **`std::system_error` and `std::generic_category()`** => 
- **`const_cast` (and why `munmap` needs it here)** => 
- **`static std::atomic<int>` counter for unique temp names** => 
- **`std::filesystem::temp_directory_path`** => 

### Safetensors parsing (your Part C)
- **little-endian `uint64` length prefix** => 
- **the 100,000,000-byte header limit** => 
- **byte ranges that tile the data section (no gaps, no overlaps, full coverage)** => 
- **overflow-safe size math (shape product times dtype size)** => 
- **JSON integers vs `1.0`** => 
- **UTF-8 validation of the header** => 
- **duplicate JSON keys (last one wins)** => 
- **`__metadata__` (null or string map)** => 
- **`std::map` ordering of string keys (`layers.10` before `layers.2`)** => 
- **`std::logic_error` vs `std::runtime_error` (why the stub cannot pass the rejection tests)** => 

### Testing
- **`todo_joe` workflow (`ctest -L todo_joe`, then move the file to `tests/unit/`)** => 
- **`std::as_bytes`** => 
- **building malformed inputs byte by byte** => 
- **`-Wdangling-reference` (a reference into a temporary's member)** => 
- **most vexing parse (`T(f())` read as a declaration; `T{f()}` fixes it)** => 
- **GCC vs Clang strictness (why macOS CI caught what Ubuntu did not)** => 

## Weight Upload (Branch 6)
### Streams and ordering
- **`cudaMemset` is asynchronous with respect to the host** => 
- **legacy default stream vs `cudaStreamNonBlocking` streams (implicit synchronization)** => 
- **`cudaDeviceSynchronize`** => 

### Weight layout
- **one allocation for all weights vs one per tensor** => 
- **256-byte alignment and rounding up (`(x + a - 1) / a * a`)** => 
- **fusing QKV and gate/up by stacking rows (why row-major makes it a byte concatenation)** => 
- **why fusion turns 5 GEMMs per layer into 2** => 
- **copy list sorted by source offset (sequential reads of an mmap)** => 
- **rejecting unexpected tensors (fail fast)** => 
- **`std::set::contains` / `std::sort` with a lambda comparator** => 

### Testing
- **fake checkpoints (testing the planner without the real file)** => 
- **invariant tests (aligned, ordered, disjoint, fully covered)** => 
- **`-Wdangling-else` with gtest macros (always brace)** => 
- **pointer arithmetic on `std::byte*` (byte offsets into one allocation)** => 
- **`SCOPED_TRACE` in loops** => 

### Double-buffered upload
- **double buffering (overlapping host copies with PCIe transfers)** => 
- **staging through pinned memory instead of copying from the mmap directly** => 
- **`cudaEvent_t` as a "this buffer is free again" signal** => 
- **`cudaEventCreateWithFlags(..., cudaEventDisableTiming)`** => 
- **`cudaEventRecord` / `cudaEventSynchronize`** => 
- **synchronizing on an event that was never recorded (returns immediately)** => 
- **`cudaEventSynchronize` vs `cudaStreamSynchronize`** => 
- **chunking a copy (`std::min` of the chunk and the remaining bytes)** => 
- **`std::chrono::steady_clock` and `duration<double>`** => 
- **effective GB/s (bytes / seconds / 1e9)** => 
- **page faults on first touch of an mmap (disk reads hidden inside `memcpy`)** => 

### Loading end to end
- **composing small pieces (config, map, parse, plan, upload, bind)** => 
- **why the mmap can be released once the upload is synchronized** => 
- **moving a struct that owns a `DeviceBuffer` (views stay valid because the pointer does not change)** => 
- **CMake list escaping (why `LABELS "todo_joe;gpu"` lost a label through `gtest_discover_tests`)** => 

## Local Notes (docs/ignore-claude-notes)
- **`.gitignore` directory patterns (`docs/claude/`)** => 
- **ignored files are per working tree (not shared between `git worktree` checkouts)** => 

## Safetensors Parser (loader/safetensors-parser)
- **nlohmann `is_number_integer()` vs `is_number_unsigned()` (non-negative integers parse as unsigned)** => 
- **`get<int64_t>()` on a value above INT64_MAX** => 
- **signed-to-unsigned conversion wraparound (`static_cast<uint64_t>(-1)`)** => 
- **validating in one pass instead of re-walking the input** => 
- **`git mv` (keeping file history across a move)** => 
- **CTest labels and `ctest -LE` (excluding tests by label)** => 

## Checksums (Branch 7)
### Checksum math
- **why a checksum sums in FP64 (and also sums absolute values)** => 
- **comparing sums relative to the abs-sum, not to the sum (cancellation near zero)** => 
- **decoding little-endian byte pairs instead of casting to `uint16_t*` (alignment)** => 
- **BF16 to float by shifting 16 bits left** => 
- **`std::span::subspan`** => 
- **`nlohmann/json_fwd.hpp` (forward declarations to keep headers light)** => 
- **JSON round-trip of doubles (17 significant digits)** => 
- **verifying an upload by reading it back (round-trip testing)** => 
- **why identical values summed in the same order give bit-identical results** => 

### CLI
- **CLI11 options, flags, `->required()` and `->check(CLI::ExistingDirectory)`** => 
- **sharing parsed options with a callback (`std::make_shared<Options>` captured by the lambda)** => 
- **catching exceptions at the top of `main` (`app.parse` vs `CLI11_PARSE`)** => 
- **`$<BOOL:...>` generator expression and `#if ENGINE_HAS_CUDA`** => 
- **the legacy default stream (`nullptr` as a `cudaStream_t`)** => 

### Python reference
- **`safe_open(..., framework="pt")` and `get_slice(name).get_dtype()`** => 
- **`tensor.to(torch.float64)` before summing** => 
- **`argparse` with `type=Path`** => 
- **why PyTorch's pairwise summation and a sequential C++ loop still agree to about 1e-16 here** => 

### Golden tests
- **golden files (generated reference data, gitignored, regenerated by a script)** => 
- **tolerance as a fraction of the abs-sum (6+ significant digits)** => 
- **skipping when local data is missing vs failing** => 
- **proving a test can fail (tampering with the golden on purpose)** => 

## Docs (Branch 8)
### Ownership
- **authorship vs review (writing code yourself vs reviewing code someone else wrote)** => 
- **git author identity (why `git log` cannot show who wrote a commit here)** => 
### README
- **reproducing results from a clean clone** => 
- **dangling symlink (a tracked link to an absolute path on one machine)** => 
### Project overview
- **arithmetic intensity (FLOPs per byte read)** => 
- **bandwidth floor on decode latency (bytes read per token / bandwidth)** => 

## Repo Housekeeping (Branch 9)
### Git
- **`git rm --cached` (untracking a file without deleting it locally)** => 
- **why pulling a commit that untracks a file deletes it from other checkouts** => 
- **why compiled binaries stay out of git (build outputs vs sources)** => 
