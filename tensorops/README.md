# Matrix Multiplication with Tensor Cores

## Resources
- [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) and [matmul-playground](https://github.com/alexarmbr/matmul-playground)
- [NVIDIA CUDA Parallel Thread Execution (PTX) Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA CUDA Inline PTX Assembly Documentation](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [NVIDIA CUTLASS Library](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch) (see `copy_smxx.hpp` and `mma_smxx.hpp`)

## Performance Comparison
For M = N = K = 4096, BF16 A row-major x B column-major, RTX A6000, compiled with `-O3 --use_fast_math`:

| Kernel name | Duration (ms) | TFLOPS | % of CuBLAS |
|-------------|---------------|--------|-------------|
| CuBLAS (via PyTorch) `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` | 1.16 | 118.48 TFLOPS | 100.00% |
| v1a (block+warp tiling, `mma.m16n8k8`) | 1.88 | 73.11 TFLOPS | 92.93% |
| v1b (`m16n8k6` with padded A shared memory) | 1.82 | 75.52 TFLOPS | 95.83% |

## Lessons Learned

### Inline PTX
- Components: instruction, outputs, inputs, constraints

### `ldmatrix` Instruction
- A warp loads 8x8 tiles of 16-bit from shared memory to registers
- Each thread holds 2 elements (32 threads in a warp)
- Thread-to-element mapping aligns with `mma` instructions
- Can load 1x, 2x, or 4x of 8x8 16-bit tiles
- More generically:
  - Loads 8 rows of 8 consecutive 16-bit elements
  - Each 8-element row can be anywhere in shared memory (with 16-byte alignment)
  - Collectively loads 8x 16-byte words
- Requires converting generic address to shared address using `ctva` instruction

### `mma` Instruction
- Typical shapes for FP16/BF16: `m16n8k8` and `m16n8k16`
- Each thread must hold specific elements in the tile (achieved via `ldmatrix`)
- For 16x8 tile: use `ldmatrix.x2`
- For 16x16 tile: use `ldmatrix.x4`
- Accumulation results held in register memory across threads in a warp
- Can write results directly from registers to global memory

### Shared Memory Optimization

#### Bank Conflicts
- 32 banks, each holds 4 bytes
- Each row of `ldmatrix` tile resides in 4 banks (16 bytes)
- No bank conflicts when shared memory width = 8x 16-bit elements (16 bytes)
- Larger widths typically used for vectorized global memory reads

#### Solutions

##### 1. Padded Shared Memory
- Classic but wasteful approach
- Requires 8-element padding to ensure 16-byte alignment for `ldmatrix`
