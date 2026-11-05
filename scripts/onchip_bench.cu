#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

struct Stamp {
    long long cycles;
    unsigned long long ns;
};

struct Span {
    Stamp start, end;
    double cycles() const { return double(end.cycles - start.cycles); }
    double ns() const { return double(end.ns - start.ns); }
    double mhz() const { return 1e3 * cycles() / ns(); }
};

__device__ Stamp stamp() {
    Stamp s;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s.ns));
    s.cycles = clock64();
    return s;
}

__device__ bool first_thread() {
    return blockIdx.x == 0 && threadIdx.x == 0;
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

void row(const char* metric, const char* unit, double value, double mhz) {
    if (mhz > 0)
        printf("%s,%s,%.2f,%.0f\n", metric, unit, value, mhz);
    else
        printf("%s,%s,%.2f,\n", metric, unit, value);
}

__global__ void chase_global(const unsigned long long* head, int steps, unsigned long long* out,
                             Span* span) {
    const unsigned long long* p = head;
    const Stamp start = stamp();
#pragma unroll 32
    for (int i = 0; i < steps; ++i) p = reinterpret_cast<const unsigned long long*>(__ldcg(p));
    const Stamp end = stamp();
    out[0] = reinterpret_cast<unsigned long long>(p);
    span->start = start;
    span->end = end;
}

template <int N>
__global__ void chase_shared(const unsigned* next, int steps, unsigned* out, Span* span) {
    __shared__ unsigned s[N];
    for (int i = 0; i < N; ++i) s[i] = next[i] * 4;
    const char* base = reinterpret_cast<const char*>(s);
    unsigned j = 0;
    const Stamp start = stamp();
#pragma unroll 32
    for (int i = 0; i < steps; ++i) j = *reinterpret_cast<const unsigned*>(base + j);
    const Stamp end = stamp();
    out[0] = j;
    span->start = start;
    span->end = end;
}

__global__ void fma_chain(float a, float b, int steps, float* out, Span* span) {
    float x = a;
    const Stamp start = stamp();
#pragma unroll 128
    for (int i = 0; i < steps; ++i) x = fmaf(x, a, b);
    const Stamp end = stamp();
    out[0] = x;
    span->start = start;
    span->end = end;
}

__global__ void read_l2(const float4* p, int n, int reps, float* out, Span* span) {
    const Stamp start = stamp();
    const int stride = gridDim.x * blockDim.x;
    float4 acc = {0, 0, 0, 0};
    for (int r = 0; r < reps; ++r) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n; i += 4 * stride) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) v[k] = __ldcg(p + i + k * stride);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                acc.x += v[k].x;
                acc.y += v[k].y;
                acc.z += v[k].z;
                acc.w += v[k].w;
            }
        }
    }
    if (acc.x + acc.y + acc.z + acc.w == -1.0f) out[0] = acc.x;
    const Stamp end = stamp();
    if (first_thread()) {
        span->start = start;
        span->end = end;
    }
}

constexpr int kSmemThreads = 256;
constexpr int kSmemLoads = 4;

template <typename T>
__device__ void add(T& acc, const T& v) {
    acc += v;
}

template <>
__device__ void add<float4>(float4& acc, const float4& v) {
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
}

template <typename T>
__device__ float total(const T& acc) {
    return acc;
}

template <>
__device__ float total<float4>(const float4& acc) {
    return acc.x + acc.y + acc.z + acc.w;
}

template <typename T>
__global__ void read_shared(int reps, int chunk, float* out, Span* span) {
    constexpr int kChunk = kSmemThreads * kSmemLoads;
    __shared__ T s[2 * kChunk];
    for (int i = threadIdx.x; i < int(sizeof(s) / 4); i += kSmemThreads)
        reinterpret_cast<float*>(s)[i] = float(i);
    __syncthreads();
    const Stamp start = stamp();
    T acc[kSmemLoads] = {};
    for (int r = 0; r < reps; ++r) {
        const T* p = s + ((r * chunk) & (2 * kChunk - 1)) + threadIdx.x;
#pragma unroll
        for (int k = 0; k < kSmemLoads; ++k) add(acc[k], p[k * kSmemThreads]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kSmemLoads; ++k) sum += total(acc[k]);
    if (sum == -1.0f) out[0] = sum;
    const Stamp end = stamp();
    if (first_thread()) {
        span->start = start;
        span->end = end;
    }
}

std::vector<unsigned> cycle(int nodes, int spacing) {
    std::vector<unsigned> order(nodes);
    for (int i = 0; i < nodes; ++i) order[i] = i;
    std::mt19937 gen(0);
    for (int i = nodes - 1; i > 0; --i)
        std::swap(order[i], order[std::uniform_int_distribution<int>(0, i - 1)(gen)]);
    std::vector<unsigned> next(size_t(nodes) * spacing, 0);
    for (int i = 0; i < nodes; ++i)
        next[size_t(order[i]) * spacing] = order[(i + 1) % nodes] * spacing;
    return next;
}

struct Device {
    unsigned* out;
    Span* span;
    Span read() const {
        check(cudaDeviceSynchronize(), "kernel");
        Span s;
        check(cudaMemcpy(&s, span, sizeof s, cudaMemcpyDeviceToHost), "copy span");
        return s;
    }
};

void global_latency(const Device& d, const char* metric, size_t bytes, int steps, bool warm) {
    const int spacing = 16;
    const int nodes = int(bytes / (spacing * 8));
    const std::vector<unsigned> next = cycle(nodes, spacing);
    unsigned long long* buf;
    check(cudaMalloc(&buf, next.size() * 8), "cudaMalloc");
    std::vector<unsigned long long> addr(next.size());
    for (size_t i = 0; i < next.size(); ++i)
        addr[i] = reinterpret_cast<unsigned long long>(buf + next[i]);
    check(cudaMemcpy(buf, addr.data(), addr.size() * 8, cudaMemcpyHostToDevice), "copy chain");
    auto* out = reinterpret_cast<unsigned long long*>(d.out);
    chase_global<<<1, 1>>>(buf, warm ? nodes : 0, out, d.span);
    chase_global<<<1, 1>>>(buf, steps, out, d.span);
    const Span s = d.read();
    row(metric, "cycles", s.cycles() / steps, s.mhz());
    row(metric, "ns", s.ns() / steps, s.mhz());
    cudaFree(buf);
}

void shared_latency(const Device& d) {
    constexpr int n = 4096;
    const std::vector<unsigned> next = cycle(n, 1);
    unsigned* buf;
    check(cudaMalloc(&buf, n * 4), "cudaMalloc");
    check(cudaMemcpy(buf, next.data(), n * 4, cudaMemcpyHostToDevice), "copy chain");
    const int steps = 1 << 20;
    chase_shared<n><<<1, 1>>>(buf, steps, d.out, d.span);
    chase_shared<n><<<1, 1>>>(buf, steps, d.out, d.span);
    const Span s = d.read();
    row("smem_latency", "cycles", s.cycles() / steps, s.mhz());
    row("smem_latency", "ns", s.ns() / steps, s.mhz());
    cudaFree(buf);
}

void fma_latency(const Device& d) {
    const int steps = 1 << 22;
    float* out = reinterpret_cast<float*>(d.out);
    fma_chain<<<1, 1>>>(0.999f, 0.001f, steps, out, d.span);
    fma_chain<<<1, 1>>>(0.999f, 0.001f, steps, out, d.span);
    const Span s = d.read();
    row("fma_latency", "cycles", s.cycles() / steps, s.mhz());
    row("fma_latency", "ns", s.ns() / steps, s.mhz());
}

template <typename Kernel, typename... Args>
std::pair<float, Span> timed(const Device& d, Kernel kernel, int blocks, int threads,
                             Args... args) {
    kernel<<<blocks, threads>>>(args..., reinterpret_cast<float*>(d.out), d.span);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(args..., reinterpret_cast<float*>(d.out), d.span);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms, d.read()};
}

void bandwidth_rows(const char* metric, double bytes, float ms, const Span& s, int sms,
                    bool per_sm) {
    const double gbs = bytes / (ms * 1e-3) / 1e9;
    row(metric, "GB/s", gbs, s.mhz());
    if (per_sm) row(metric, "GB/s/SM", gbs / sms, s.mhz());
    row(metric, "B/clk/SM", gbs * 1e3 / s.mhz() / sms, s.mhz());
}

void l2_bandwidth(const Device& d, int sms) {
    const int threads = 256;
    int per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, read_l2, threads, 0), "occupancy");
    const int blocks = per_sm * sms;
    const int step = 4 * blocks * threads;
    const int n = (8 << 20) / 16 / step * step;
    const int reps = 200;
    float4* buf;
    check(cudaMalloc(&buf, size_t(n) * 16), "cudaMalloc");
    check(cudaMemset(buf, 0, size_t(n) * 16), "cudaMemset");
    const auto [ms, s] =
        timed(d, read_l2, blocks, threads, static_cast<const float4*>(buf), n, reps);
    bandwidth_rows("l2_bandwidth", double(n) * 16 * reps, ms, s, sms, false);
    cudaFree(buf);
}

template <typename T>
void shared_bandwidth(const Device& d, const char* metric, int sms) {
    int per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, read_shared<T>, kSmemThreads, 0),
          "occupancy");
    const int blocks = per_sm * sms;
    const int reps = 1 << 15;
    const auto [ms, s] =
        timed(d, read_shared<T>, blocks, kSmemThreads, reps, kSmemThreads * kSmemLoads);
    bandwidth_rows(metric, double(blocks) * kSmemThreads * kSmemLoads * sizeof(T) * reps, ms, s,
                   sms, true);
}

int main() {
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    Device d;
    check(cudaMalloc(&d.out, 64), "cudaMalloc");
    check(cudaMalloc(&d.span, sizeof(Span)), "cudaMalloc");
    printf("metric,unit,value,sm_mhz\n");
    row("register_file", "KB/SM", prop.regsPerMultiprocessor * 4.0 / 1024, 0);
    row("register_file", "KB", prop.regsPerMultiprocessor * 4.0 / 1024 * prop.multiProcessorCount,
        0);
    row("smem_size", "KB/SM", prop.sharedMemPerMultiprocessor / 1024.0, 0);
    row("l2_size", "KB", prop.l2CacheSize / 1024.0, 0);
    shared_bandwidth<float>(d, "smem_bandwidth_32bit", prop.multiProcessorCount);
    shared_bandwidth<float4>(d, "smem_bandwidth_128bit", prop.multiProcessorCount);
    l2_bandwidth(d, prop.multiProcessorCount);
    fma_latency(d);
    shared_latency(d);
    global_latency(d, "l2_latency", 4 << 20, 1 << 18, true);
    global_latency(d, "dram_latency", 256 << 20, 1 << 16, false);
    check(cudaGetLastError(), "launch");
    cudaFree(d.out);
    cudaFree(d.span);
}
