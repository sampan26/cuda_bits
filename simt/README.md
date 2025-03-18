# CUDA GEMM - Optimized Matrix Multiplication with CUDA Cores

## Performance Results

| Kernel | Time (ms)  | % cublas |
|----------------|--------------|---------|
| **CuBLAS** | 5.82246  | 100% |
| **triton ref** | 7.67590 | 75.86% |
| **naive** | 58.74278 | 9.91% |
| **SMEM Caching** | 46.35136 | 12.55% |
| **1D Blocktiling** | 41.86112 | 13.91% |
| **2D Blocktiling** | 10.33114 | 56.32% |
| **Warptiling** | 10.16422 | 57.30% |
| **Vectorized(a)** | 7.12704 | 81.72% |
| **Vectorized(b)** | 6.82086 | 85.34% |

## Resources

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA Matrix Multiplication - An Overview](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) and [GitHub Repository](https://github.com/leimao/CUDA-GEMM-Optimization/)
- [NVIDIA CUTLASS: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)

## Lessons Learned

### 1. Hierarchical Tiling
- **Block-level tiling:** Utilizes shared memory as a cache.
- **Warp-level tiling:** Improves data reuse within warps.
- **Thread-level tiling:** Leverages register memory to reduce memory latency.

### 2. Vectorized Memory Access
- **Observation:** Although 128-byte memory transactions are used, vectorized access can still boost performance by reducing the instruction count and enabling better pipelining.

### 3. Reducing Memory Address Computation
- Instead of recalculating addresses in every iteration, incrementing addresses reduces computational overhead.

### 4. Avoiding Bank Conflicts
- **Padding shared memory:** Prevents bank conflicts.
- **Swizzled layout:** Optimizes memory access patterns. *(TODO: Implementation details pending)*

### 5. Double Buffering
- Uses twice the shared memory to overlap data loading and computation, eliminating some `__syncthreads()` calls and improving performance through better interleaving. *(TODO: Further optimization work required)*

