# Foundational
### Core Ideas
- **kernel** => function that can be run by multiple GPU threads at the same time with different sets of data
- **thread** => thing that performs its assigned set of instructions with its own data
- **memory overflow** =>
- **block** => 
- **grid** => 
- **warp** => 

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

### Values and references
- **moved-from state** => 
- **self-move-assignment (`b = std::move(b)`)** => 
- **std::exchange** => `a = std::exchange(b, newVal)` updates `b` to `newVal` and sets `a` to the old value of `b`, can take the the buffer of another object and then update its pointer to null, all in one call

### Memory
- **`void*`** => 
- **`malloc`** => finds free space for requested memory size and returns the pointer to the memory location
- **`std::free`** => 
- **`std::memset`** => 
- **`std::bad_alloc`** => 
- **`noexcept`** => 
- **`std::vector<T>`** => basically a dynamic array

### Testing
- **ASSERT vs EXPECT (GoogleTest)** => both mark tests as failed, assert quits the current test, expect continues the test, assert is used for when continuing the test would be pointless/crash
- **`static_assert` + `std::is_copy_constructible_v` / `is_move_constructible_v`** => 
- **`std::declval`** => 
- **AddressSanitizer** => 

## Safe Tensor Reader (Task 2)
### Core Ideas
- **file descriptor** => 
- **POSIX** => 
- **RAII applied to OS resources (FDCloser / MemoryUnmapper)** => 
- **untrusted input validation (corrupt or malicious headers)** => 
- **integer overflow in bounds checks (`a + b > n` vs `b > n - a`)** => 

### Opening and inspecting files
- **`open`** => opens a file and gives a file descriptor
- **`O_RDONLY`** => flag that lets open files be read from only, write fails
- **`close`** => closes the file descriptor (file is left untouched)
- **`struct stat` (sys/stat.h)** => 
- **`fstat(fd, &sb)`** => 
- **`st_size`, `off_t` -> `size_t` conversion** => 
- **`errno`** => 
- **`std::strerror(errnum)`** => 

### Memory mapping
- **mmap** => 
- **PROT_READ / prot flags** => 
- **MAP_SHARED vs MAP_PRIVATE** => 
- **MAP_FAILED** => 
- **munmap** => 
- **virtual memory and lazy page faults (why mapping 2 GB is instant)** => 
- **page cache** => 
- **page size and alignment (`sysconf(_SC_PAGESIZE)`)** => 

### Bytes and types
- **little endian unsigned int** => 
- **std::endian + static_assert** => 
- **why `sizeof(uint64_t)` instead of 8** => 
- **std::memcpy** => 
- **strict aliasing (why memcpy instead of `*(uint64_t*)ptr`)** => 
- **static_cast** => 
- **reinterpret_cast** => 

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
- **throw / try / catch** => 
- **std::runtime_error** => 
- **uncaught exception -> std::terminate (the "Aborted (core dumped)")** => 
- **argc / argv** => 
- **std::cerr vs std::cout** => 

### Build
- **`cmake_minimum_required`, `project`** => 
- **`CMAKE_CXX_STANDARD` / `CMAKE_CXX_STANDARD_REQUIRED`** => 
- **`add_executable`** => 
- **`target_compile_options`** => 
- **-Wall -Wextra -Wpedantic** => 
- **out-of-source builds (`cmake -B build`, `cmake --build build`)** => 
- **header-only code and `inline` (why `static inline` in a class is redundant)** =>

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

### Memory
- **memory hierarchy** => 
- **memory coalescing** => 
- **host vs device memory** => 
- **`cudaMalloc()`** => 
- **`cudaMallocManaged()`** => 
- **managed memory page migration** => 
- **PCIe bottleneck** => 
- **`cudaMemcpy()`** => 
- **`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`** => 
- **`cudaMemset()`** => 
- **`cudaFree()`** => 
- **treating `*a`, `*b`, `*c` as arrays but they are defined as float pointers** => 
- **`std::fill`** => 
- **`delete[]`** => 

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

### Verification and tooling
- **correctness verification (`h[i] != 3.0f`)** => 
- **`nvcc`** => 
- **`-arch=native` / `sm_120`** => 
- **`compute-sanitizer`** => 
- **`ncu` (Nsight Compute)** => 

### Three way Summation (Task 4/CUDA Task 2)
- **`naive atomics`**
- **`shared memory`**
- **`warp shuffle`**
- **`warp reduction`**
- **`reductions`**
- **`atomicAdd(address, val)`**
- **`__shared__`**
- **`__syncthreads()`**
- **binary tree reduction loop**
- **`>>=` (bitwise right shift)**
- **`__shfl_down_sync(mask, val, offset)`**
- **`0xffffffff` mask**
- **`cudaMemcpyHostToDevice`**
- **`free()`**
- **`constexpr`**