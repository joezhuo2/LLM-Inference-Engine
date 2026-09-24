## First profile: vadd (Sep 24, 2026)

Kernel: vadd, n = 2^26 floats, grid 262144 x 256, 3 arrays x 256 MB = ~805 MB moved per launch
Tool: ncu <2026.2.1.0> (build 38283040) (public-release), --section SpeedOfLight, default clock control (locked base clocks)
Conditions: plugged in, <Balanced> G-Helper mode

DRAM Throughput: 95.20%
Compute (SM) Throughput: 24.14%
Duration: 2.39 ms
SM Frequency: 1.68 GHz (locked)
DRAM Frequency: 10.86 GHz
Achieved bandwidth: ~337 GB/s (805 MB / 2.39 ms)
Implied peak at these clocks: ~354 GB/s

Takeaway: memory bound (1 FLOP per 12 bytes); not the project's peak denominator, since ncu locks clocks. Use bandwidth_test.cu (unlocked, cudaEvent timing) for that.