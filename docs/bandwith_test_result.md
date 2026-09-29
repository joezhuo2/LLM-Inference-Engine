## Bandwidth Test Results
- **setup** => RTX 5060 Laptop, 55W/5W boost, 2^26 floats (268 MB per array), `cudaMalloc`
- **result** => 323.63 GB/s effective, ~84% of ~384 GB/s theoretical peak, output verified correct
- **managed memory** => ~11.4 GB/s with `cudaMallocManaged`, because pages migrated over PCIe during the timed launch