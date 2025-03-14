# CUDA GEMM - Optimized Matrix Multiplication with CUDA Cores

## Overview

This project explores efficient matrix multiplication using CUDA cores, applying various optimization techniques to improve performance on an NVIDIA A6000 GPU.

## Resources

For deeper insights into CUDA matrix multiplication optimizations, refer to the following resources:

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA Matrix Multiplication - An Overview](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) and [GitHub Repository](https://github.com/leimao/CUDA-GEMM-Optimization/)
- [NVIDIA CUTLASS: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)

## Lessons Learned

### 1. Hierarchical Tiling
- Optimizing memory access at different levels:
  - **Block-level tiling**: Uses shared memory as a cache.
  - **Warp-level tiling**: Improves data reuse within warps.
  - **Thread-level tiling**: Leverages register memory for reduced memory latency.

### 2. Vectorized Memory Access
- **Observation**: Even though 128-byte memory transactions are already used, vectorized access seems to improve performance.
- **Possible Reason**: Reduced instruction count enables better instruction pipelining.

### 3. Reducing Memory Address Computation
- Instead of recalculating addresses in every iteration, **increment addresses** to reduce computational overhead.

### 4. Avoiding Bank Conflicts
- **Padding shared memory** to prevent conflicts (**TODO**).
- **Swizzled layout** to optimize memory access patterns (**TODO**).

### 5. Double Buffering
- Uses **twice the shared memory** to overlap data loading and computation.
- Eliminates the need for `__syncthreads()` after each computation phase, improving performance through better interleaving (**TODO**).

---

This README provides a structured and professional explanation of your project while maintaining technical depth. 🚀
