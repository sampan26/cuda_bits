# CUDA GEMM - Optimized Matrix Multiplication with CUDA Cores

## Overview

This project explores efficient matrix multiplication using CUDA cores, applying various optimization techniques to improve performance on an NVIDIA A6000 GPU. Our implementations range from a naïve approach to highly optimized kernels, demonstrating the impact of tiling, vectorized memory access, and register reuse on performance.

## Performance Results

The table below summarizes the kernel performance for a 4096×4096 matrix multiplication. Each kernel's runtime (in milliseconds) is converted to TFLOPS, where 1 TFLOPS equals 1,000 GFLOPS. The TFLOPS values are calculated based on the total floating-point operations (FLOPs) required for the operation, which for this matrix size is approximately 137.44 billion FLOPs.

| Kernel | Time (ms) | TFLOPS | % cublas |
|----------------|-----------|--------------|---------|
| **CuBLAS** | 5.82246 | 23.61 TFLOPS | 100% |
| **triton ref** | 7.67590 | 17.91 TFLOPS | 75.86% |
| **v1** | 58.74278 | 2.34 TFLOPS | 9.91% |
| **v2** | 46.35136 | 2.964 TFLOPS | 12.55% |
| **v3** | 41.86112 | 3.284 TFLOPS | 13.91% |
| **v4** | 10.33114 | 13.297 TFLOPS | 56.32% |
| **v5** | 10.16422 | 13.528 TFLOPS | 57.30% |
| **v6a** | 7.12704 | 19.293 TFLOPS | 81.72% |
| **v6b** | 6.82086 | 20.148 TFLOPS | 85.34% |

### How the Numbers Were Calculated

1. **Total Operations:**  
   For a 4096×4096 matrix multiplication, the operation count is:  
   \[
   2 \times 4096^3 \approx 137{,}438{,}953{,}472 \text{ FLOPs}.
   \]

2. **GFLOPS Calculation:**  
   The GFLOPS is computed by dividing the total FLOPs by the kernel runtime (converted to seconds) and then by \(10^9\). For example, for CuBLAS:  
   \[
   \text{GFLOPS} = \frac{137{,}438{,}953{,}472}{(5.82246/1000) \times 10^9} \approx 23{,}610 \text{ GFLOPS}.
   \]

3. **TFLOPS Conversion:**  
   Simply divide the GFLOPS value by 1,000. For CuBLAS, \(23{,}610 \div 1{,}000 \approx 23.61 \text{ TFLOPS}\).

## Resources

For deeper insights into CUDA matrix multiplication optimizations, refer to the following resources:

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

