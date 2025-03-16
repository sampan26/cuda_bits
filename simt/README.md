# cuda_gemm

# Matrix Multiplication - CUDA Cores

Resources:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
- https://siboehm.com/articles/22/CUDA-MMM
- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/ and https://github.com/leimao/CUDA-GEMM-Optimization/
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

All experiments run on A6000
Lessons learned:

- Hierarchical tiling: block-level (shared memory cache), warp-level, and thread-level (register cache).
- Vectorized memory access: not sure why it helps as we already use 128-byte memory transaction. Perhaps it reduces number of instructions and thus better pipelining?
- Reduce memory address computations: increment memory address after each iteration instead of calculating from the base address every time.
- Avoid bank conflicts:
  - Padding shared memory (TODO).
  - Swizzled layout (TODO).
- Double buffering: use double amount of shared memory, but don't need to `__syncthreads()` after computation code -> better data loading and computation interleaving (TODO).
