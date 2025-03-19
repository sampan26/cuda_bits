# CUDA GEMM 

This folder contains two projects exploring different approaches to optimizing matrix multiplication on NVIDIA GPUs:

## 1. CUDA GEMM with CUDA Cores
- Implements matrix multiplication using traditional CUDA cores
- Otimization techniques including warp tiling, vectorized memory access, etc
- Achieves up to 85% of cuBLAS performance on an NVIDIA A6000 GPU

## 2. Matrix Multiplication with Tensor Cores
- Implements matrix multiplication using NVIDIA Tensor Cores
- Utilizes inline PTX and shared memory optimization
- Achieves up to 95% of cuBLAS performance on an NVIDIA A6000 GPU

Both projects include performance benchmarks, optimization strategies, and references to relevant resources.