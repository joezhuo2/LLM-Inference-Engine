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
- **`cudaMalloc()`** => allocates free high speed GPU memory
- **`cudaMallocManaged()`** => 
- **managed memory page migration** => 
- **PCIe bottleneck** => 
- **`cudaMemcpy()`** => 
- **`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`** => 
- **`cudaMemset()`** => 
- **`cudaFree()`** => frees high speed GPU memory
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

## Tokenizer (Branch 10)
### Dependencies
- **pkg-config and `.pc` files (how C/C++ libraries advertise include and link flags)** => 
- **`pkg_check_modules(... IMPORTED_TARGET)`** => 
- **`URL_HASH SHA256=...` (pinning a download)** => 
- **`EXCLUDE_FROM_ALL` (only build what you link)** => 
- **directory-scoped `add_compile_options` and why third-party code should not get your warning flags** => 
### Tokenization
- **SentencePiece BPE (pieces, merges, scores)** => 
- **the `▁` word-boundary marker and the dummy prefix** => 
- **byte fallback (`<0x0A>` and other byte pieces)** => 
- **special tokens (`<s>` BOS, `</s>` EOS, `<unk>`) and why their text maps to one id** => 
- **HF `legacy: false` (no prefix after a special token)** => 
- **pimpl-style forward declaration with `std::unique_ptr` (keeping a dependency's header private)** => 
- **why the destructor must be defined in the .cpp when a member is `unique_ptr` to an incomplete type** => 
### Chat template
- **chat templates (Jinja in `tokenizer_config.json`) and the Zephyr format** => 
- **generation prompt (`add_generation_prompt`)** => 
- **why `apply_chat_template` adds no BOS here** => 
### Golden tests for the tokenizer
- **fuzzing against a reference implementation (seeded random inputs)** => 
### Reference dumps
- **teacher forcing (feeding the reference tokens instead of your own)** => 
- **PyTorch forward hooks (`register_forward_hook`, capturing a module's input or output)** => 
- **eager vs SDPA attention in HF** => 
- **greedy decoding** => 
- **BF16 near-ties (why two correct implementations can pick different argmax tokens)** => 
- **self-consistency checks on golden data (checking the reference before trusting it)** => 

## Naive Oracle Kernels (Branch 11)
### Embedding
- **`__nv_bfloat16` (`cuda_bf16.h`)** => 
- **one block per row with a block-stride loop (`for (i = threadIdx.x; i < n; i += blockDim.x)`)** => 
- **why a zero-size grid is an invalid launch configuration** => 
- **oracle kernel (a simple, obviously correct kernel used to check fast ones)** => 
### Testing kernels
- **distance in ulps between two BF16 values (ordering sign-magnitude bit patterns)** => 
### RMSNorm
- **block-level sum with a shared-memory tree reduction (`stride = blockDim / 2; stride > 0; stride /= 2`)** => 
- **why the reduction loop needs `__syncthreads()` after every step, outside the `if`** => 
- **`rsqrtf` (CUDA's reciprocal square root)** => 
- **matching HF's intermediate rounding (normalize in FP32, round to BF16, then multiply by the weight)** => 
- **aliasing `out` and `x` safely (each thread reads element i before writing element i)** => 
- **why FP32 summation order changes the result (non-associative floating-point addition)** => 
- **a tolerance derived from the algorithm (one flipped rounding, then a multiply, is at most 2 ulps)** => 
- **mutation testing (breaking the kernel on purpose to check the tolerance still fails)** => 
### Residual add
- **64-bit element index (`int64_t(blockIdx.x) * blockDim.x`) and when 32 bits overflow** => 
- **BF16 + BF16 in PyTorch (computed in FP32, rounded once)** => 
### RoPE kernel
- **in-place update of a fused QKV row (Q heads and K heads are one contiguous range)** => 
- **passing a small array to a kernel by value (a struct parameter instead of a device buffer)** => 
- **why `powf` on the GPU and on the CPU can differ in the last bit** => 
- **catastrophic cancellation (a 1-ulp input change becoming many ulps of a small difference)** => 
- **why angle errors grow with position (`pos * inv_freq`)** => 
### SwiGLU kernel
- **fused gate and up output (`[T, 2 * ff]`, gate in the first half of each row)** => 
- **`expf` vs the fast `__expf` intrinsic (and why the oracle uses the accurate one)** => 
- **mapping a flat index to row and column (`t = idx / ff`, `j = idx % ff`)** => 

## Naive Attention (Branch 12)
### Contiguous KV cache
- **`cudaMemcpy2DAsync` (copying a strided column block: pitch vs width vs height)** => 
- **contiguous KV cache layout (`[max_seq, kv_heads * head_dim]` per layer, row = position)** => 
### Naive attention kernel
- **one block per (token, head) with a 2D grid (`blockIdx.x`, `blockIdx.y`)** => 
- **sizing dynamic shared memory at launch (the third `<<<>>>` parameter and the 48 KB default limit)** => 
- **the order of roundings in HF's eager attention (scores to BF16, scale, FP32 softmax back to BF16, then P @ V)** => 
- **subtracting the max before `exp` (numerically stable softmax)** => 
- **`fmaf` and FMA contraction (why the GPU's `a * b + c` and the CPU's can round differently)** => 
- **one attention call for prefill and decode (query `t` at position `start + t` sees cache rows `0..start + t`)** => 
### Attention against HF
- **why a different dot-product order flips BF16 scores, and why `exp` amplifies a flip in a large score** => 
- **ulps vs absolute error near zero (why a huge ulp count can be a tiny difference)** => 
- **scaling a tolerance by the inputs (`|error| <= c * max |v|` for a weighted average of V)** => 
### CI container images
- **container registries (Docker Hub vs NVIDIA's nvcr.io) and the image reference `registry/namespace/repo:tag`** => 
- **anonymous pull rate limits on shared CI runners (why a job can fail before building anything)** => 

## Forward Pass (Branch 13)
### Greedy decoding
- **argmax reduction carrying (value, index) pairs, and why ties must go to the lowest index to match `torch.argmax`** => 
### Model runner
- **preallocating every activation buffer for `max_tokens` rows (no allocation inside the forward pass)** => 
- **the residual stream updated in place (`x = x + attn_out`, `x = x + mlp_out`)** => 
- **`cudaMemcpyAsync` from pageable host memory (staged through a pinned buffer; when the host copy may be freed)** => 
- **inspecting intermediate tensors with a callback (`std::function` hook per layer)** => 
- **cuBLAS algorithm selection by shape (why a fused QKV GEMM and separate K and V GEMMs round differently)** => 
- **why a one-row decode GEMM and a T-row prefill GEMM give slightly different results** => 
- **relative L2 error per layer as a hidden-state tolerance** => 
### Golden harness
- **averaging an error over every position vs averaging per-prompt averages (why short prompts weigh less)** => 
### Free-running generation
- **free-running vs teacher-forced evaluation (why one early flip changes every later token)** => 
- **comparing at the first divergence only (after it the two sequences condition on different tokens)** => 
- **a near-tie measured in BF16 ulps of the reference's own logits** => 
### Streaming text
- **incremental detokenization (decode everything, emit the new suffix)** => 
- **byte-fallback tokens and why a partial UTF-8 character decodes as U+FFFD** => 
- **range-for over a member of a temporary (`for (x : f().member())` dangles before C++23)** => 
### engine generate
- **time to first token vs decode tokens per second** => 
- **stdout vs stderr buffering (why output order can interleave, and `fflush`)** => 

## Sampling (Branch 14)
### Shared argmax
- **`__shared__` arrays declared inside a `__device__` function (one copy per block, shared by every call in that block)** => 
- **passing a lambda to a templated `__device__` function (inlined at compile time, no function pointer)** => 
### Temperature sampling
- **temperature scaling (`softmax(logits / T)`: T < 1 sharpens, T > 1 flattens, T -> 0 is greedy)** => 
- **the Gumbel distribution and `g = -log(-log(u))`** => 
- **the Gumbel-max trick (`argmax(logits / T + g)` is a sample from `softmax(logits / T)`)** => 
- **counter-based RNG (Philox4x32-10): random numbers as a function of (key, counter), no state to carry between steps** => 
- **`curand_init(seed, subsequence, offset, &state)` and the cuRAND device API** => 
- **an open interval (0, 1) for `u` (why `curand_uniform`'s (0, 1] would give `-log(0)`)** => 
- **float vs double precision of `-log(u)` near u = 1 (the Gumbel tail that decides a 32,000-way argmax)** => 
- **seeded determinism (the same seed and step give the same token)** => 
- **chi-square goodness-of-fit test, degrees of freedom and the critical value** => 
- **pooling rare outcomes into one bucket (expected count too small for chi-square)** => 
### Sampled generation
- **the token's position in the answer as the RNG step (why a sampled answer is reproducible from its seed)** => 
- **`!(x >= 0)` to reject NaN along with negatives (every comparison with NaN is false)** => 
### Sampling flags
- **`std::random_device` as a source of a fresh seed (and printing it so the run can be repeated)** => 
- **`std::optional` for an option that may be absent (`value_or`)** => 
### M2 criterion
- **a success criterion that the reference itself cannot meet (why to check self-consistency of the reference first)** => 

## Paged KV Cache (Branch 16)
### Cache sizing
- **bytes per KV block (`2 x layers x kv_heads x head_dim x block_size x 2 bytes`: K and V, every layer, BF16)** => 
- **sizing the KV cache from the memory left after the weights (instead of a fixed number of tokens)** => 
- **headroom (why only 90% of the free memory goes to the cache)** => 
- **narrowing `int64_t` to `int` and clamping with `std::min` before the cast** => 
### KV cache memory
- **`cudaMemGetInfo` (free and total device memory, and why free memory is only a snapshot)** => 
- **per-layer cache layout `[num_blocks, kv_heads, block_size, head_dim]` (why one (block, head) tile is contiguous)** => 
- **one big allocation carved into views vs many small `cudaMalloc` calls** => 
- **poisoning fresh memory with NaN (`0xFF` bytes) to catch reads of slots nothing wrote** => 
### write_kv
- **slot mapping (the block manager decides on the CPU, the kernel only scatters)** => 
- **scatter vs gather memory access** => 
- **padding rows in a batch and the `-1` slot** => 
- **why a kernel cannot validate device-side indices cheaply (a check would need a copy back to the host)** => 
- **launching with an empty grid (`cudaErrorInvalidConfiguration`)** => 
### Testing kernels
- **guard bytes around a buffer to catch out-of-bounds writes** => 
- **comparing the whole buffer, not just the expected cells (a write to a wrong place)** => 
- **mutation testing (checking that a deliberately broken implementation fails the tests)** => 
- **`std::abort` vs throwing in a stub (each ctest test runs in its own process)** => 
## KV Block Manager (Branch 15)
### Paged KV cache
- **paged KV cache (fixed-size blocks instead of one contiguous buffer per sequence)** => 
- **internal vs external fragmentation (why reserving the maximum context per sequence wastes memory)** => 
- **block table (logical block index to physical block id, like a page table)** => 
- **physical slot of a token (`table[pos / block_size] * block_size + pos % block_size`)** => 
- **reserved null block (a safe target for padded batch entries)** => 
- **free list as a stack of block ids** => 
### Interface and errors
- **`using` type alias (`using SeqId = int64_t`)** => 
- **precondition and `std::invalid_argument` vs `std::logic_error` (why a stub's exception must not satisfy a rejection test)** => 
- **strong exception guarantee (a failed call leaves the object unchanged)** => 
- **reference invalidation (why `table()`'s reference is only valid until the next non-const call)** => 
### Tests
- **class invariants checked after every operation** => 
- **shadow model (a simple model of the expected state the test keeps alongside the real object)** => 
- **randomized testing with a fixed seed (`std::mt19937`, reproducible failures)** => 
- **unspecified evaluation order of function arguments (why `EXPECT_EQ(f(), g())` must not depend on f running first)** => 
- **defaulted `operator==` (C++20) for comparing whole snapshots** => 
## Paged Decode Attention (Branch 17)
### Paged attention
- **gathering K and V through a block table at read time (the scatter in `write_kv` is the other half)** => 
- **context length (`context_lens[b]`, the tokens in the cache including the current one)** => 
- **one query row per sequence in decode vs many in prefill** => 
- **prefill as one decode row per prompt token (rows sharing one block table, context `pos + 1`)** => 
- **two-pass softmax (scores stored, then max, sum and weights) vs one pass** => 
- **online softmax (running max, running sum and a rescaled accumulator)** => 
- **masking past the context and why `p = 0` does not cancel a NaN (`0 * NaN = NaN`)** => 
- **`exp` overflow in FP32 (above about 88.7)** => 
### Testing attention
- **error bound relative to `sum p_j |v_j|` (why an absolute tolerance is wrong for long contexts)** => 
- **floating-point non-associativity (why summation order changes the bits)** => 
- **batch invariance (a row's output must not depend on the other rows in the batch)** => 
- **exact scores from small integers (testing overflow without rounding noise)** => 
## Batched Sampling (Branch 18)
### Sampling parameters
- **aggregate initialization (`{1.0f, 7}` fills fields in declaration order, so new fields go at the end)** => 
- **validating once at the boundary instead of in every kernel call** => 
### Per-row sampling
- **per-row parameters as a device array of structs (one small upload per step)** => 
- **a trivially copyable struct shared by host and device code (same layout on both sides)** => 
- **stream ordering (why one device buffer can be overwritten every step by `cudaMemcpyAsync` on the same stream)** => 
- **block-uniform branch vs warp divergence (every thread of a block takes the same side of `temperature == 0`)** => 
- **noise keyed by (seed, step, token) instead of the batch row (the same request draws the same tokens in any batch)** => 
- **`std::span<const T>` parameters and `size_bytes()`** => 
### Top-k by threshold search
- **top-k sampling (keep the k most likely tokens, renormalize, draw)** => 
- **order of HF's logits warpers (temperature, then top-k, then top-p)** => 
- **selecting by a threshold instead of sorting (the k-th largest value as a binary search over counts)** => 
- **order-preserving integer key of a float (flip the sign bit of positives, every bit of negatives)** => 
- **`-0.0f` vs `+0.0f` (equal as floats, different bits; `x + 0.0f` folds them)** => 
- **`__float_as_uint` (reinterpreting bits without conversion)** => 
- **`__shfl_xor_sync` butterfly reduction (every lane ends with the same total)** => 
- **block-wide sum reused in a loop (why a `__syncthreads()` before writing the shared array)** => 
- **block-uniform loop control (why every thread must see the same total before deciding)** => 
- **upper midpoint `hi - (hi - lo) / 2` (avoids overflow and an endless loop when `lo = mid`)** => 
- **equivalent mutant (a change that cannot alter any observable result)** => 
### Top-p by threshold search
- **top-p (nucleus) sampling (the smallest set of top tokens whose probability reaches p)** => 
- **HF's top-p rule (`cumsum(sorted ascending) <= 1 - p` removed) as "mass strictly above < p"** => 
- **why the threshold search needs a monotone function (float sums of non-negative terms never decrease as terms are added)** => 
- **max subtraction before `expf` (softmax without overflow)** => 
- **ties at the cut: a threshold on values cannot split a group of equal logits** => 
- **stable vs unstable sort (why HF keeps different tied tokens on CPU and CUDA)** => 
- **BF16 logit ties (one ulp is 0.0625 between 8 and 16)** => 
- **a generic block reduction with an operator (`block_reduce(v, op)` for sum and max)** => 
- **`fmaxf` (and why max is commutative, so the butterfly agrees in every lane)** => 
- **chi-square test at a setting near a cut (why the tests assert a margin first)** => 
### Command line
- **CLI11 validators (`CLI::Range`, `CLI::NonNegativeNumber`) vs the library's own validation** => 
- **failing fast before expensive work (checking parameters before loading 2.2 GB of weights)** => 
- **FP64 throughput on GeForce GPUs (why 64,000 double logs per token show up in nsys)** => 
## Scheduler (Branch 19)
### Requests
- **request lifecycle (waiting, running, finished)** => 
- **`enum class` (scoped enumeration)** => 
- **designated initializers (C++20, `Request{.id = 1, .prompt = {...}}`, fields in declaration order)** => 
- **default member initializers (`std::vector<int32_t> prompt{};`) and GCC's `-Wmissing-field-initializers`** => 
- **`std::chrono::steady_clock::time_point` (a monotonic timestamp, never adjusted like the wall clock)** => 
- **TTFT, TPOT and end-to-end latency from per-request timestamps** => 
- **why the last sampled token never needs a KV slot (`prompt + max_new_tokens - 1` fits the context)** => 
### Continuous batching
- **continuous (iteration-level) batching vs static batching** => 
- **prefill step vs decode step (why the MVP never mixes them in one forward pass)** => 
- **admission control by free KV blocks and a prefill token budget** => 
- **first come, first served and head-of-line blocking (why admission never skips ahead to a smaller request)** => 
- **starvation (a request that could wait forever) and how a FIFO queue prevents it** => 
- **recompute-style preemption (free the blocks, prefill prompt plus output again later) vs swapping to host memory** => 
- **why preempt the most recently admitted sequence (it has the least work to redo)** => 
- **liveness: why a cache that holds one `max_context` sequence guarantees the scheduler never stalls** => 
- **a scheduler that owns its block manager and exposes it read-only (`const BlockManager&`)** => 
