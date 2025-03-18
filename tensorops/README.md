# Matrix Multiplication with Tensor Cores

## Performance Results
For M = N = K = 4096, BF16 A row-major x B column-major, RTX A6000, compiled with `-O3 --use_fast_math`:

| Kernel name | Duration (ms) | % of CuBLAS |
|-------------|---------------|-------------|
| CuBLAS `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` | 1.16 | 100.00% |
| v1a (block+warp tiling, `mma.m16n8k8`) | 1.88 | 92.93% |
| v1b (`m16n8k6` with padded A shared memory) | 1.82 | 95.83% |


## Resources

- [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) and [matmul-playground](https://github.com/alexarmbr/matmul-playground)
- [NVIDIA CUDA Parallel Thread Execution (PTX) Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA CUDA Inline PTX Assembly Documentation](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [NVIDIA CUTLASS Library](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch) (see `copy_smxx.hpp` and `mma_smxx.hpp`)

# Lessons Learned

## 1. Inline PTX
- **Components:** Instruction, outputs, inputs, constraints
- **Usage:** Enables direct access to hardware-specific instructions for optimized performance

## 2. `ldmatrix` Instruction
- **Basic operation:** Allows a warp to load 8×8 tiles of 16-bit elements from shared memory to registers
- **Distribution:** Each thread holds 2 elements (across 32 threads in a warp)
- **Thread mapping:** Arranged to align perfectly with subsequent `mma` instructions
- **Loading capacity:** Supports 1×, 2×, or 4× of 8×8 16-bit tiles
- **Memory constraints:**
  - Loads 8 rows of 8 consecutive 16-bit elements
  - Each 8-element row can be positioned anywhere in shared memory (with 16-byte alignment)
  - Collectively loads 8× 16-byte words
- **Address conversion:** Requires `ctva` instruction to convert generic address to shared address

## 3. `mma` Instruction
- **Common dimensions:** Typically uses `m16n8k8` and `m16n8k16` shapes for FP16/BF16
- **Register arrangement:** Requires specific element positioning within threads (facilitated by `ldmatrix`)
- **Tile handling:**
  - 16×8 tile: Use `ldmatrix.x2`
  - 16×16 tile: Use `ldmatrix.x4`
- **Accumulation:** Results stored in register memory distributed across threads in a warp
- **Output flexibility:** Can write results directly from registers to global memory

## 4. Shared Memory Optimization

### Bank Conflicts
- **Structure:** 32 banks, each storing 4 bytes
- **Tile storage:** Each row of `ldmatrix` tile occupies 4 banks (16 bytes)
- **Optimal configuration:** No bank conflicts when shared memory width equals 8× 16-bit elements (16 bytes)
- **Practical usage:** Larger widths typically employed for vectorized global memory reads

### Solutions

#### Padded Shared Memory
- **Approach:** Classic but memory-inefficient solution
- **Implementation:** Requires 8-element padding to ensure proper 16-byte alignment for `ldmatrix`