# Matrix Multiplication with Tensor Cores

## Performance Results
For M = N = K = 4096, BF16 A row-major x B column-major, RTX A6000, compiled with `-O3 --use_fast_math`:

| Kernel name | Duration (ms) | % of CuBLAS |
|-------------|---------------|-------------|
| CuBLAS `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` | 155.65 | 100.00% |
| v1 (`m16n8k6` with padded A shared memory) | 239.62 | 64.95% |
| v2 (async loads) | 207.87 | 74.89% |
| v3 (double buffering shared memory loads) | 202.75 | 76.78% |
| v4 (pipelining/double buffering register loads) | 207.36 | 75.07% |


## Resources

- [gau-nerst](https://github.com/gau-nernst/learn-cuda/tree/main/02b_matmul_tensorop)
