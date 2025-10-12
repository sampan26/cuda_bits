
## matmul-bf16
- Implements half percision general matrix multiplication
    - first time i used inline PTX and shared memory optimization
    - Achieves up to 95% of cuBLAS performance on an NVIDIA A6000 GPU

## softmax
- Implements softmax in cuda in fp32, outperforming torch with 135% of its performance

## flash_attention
- Implements Flash Attention 1 for fp32, outperforming torch implementations
- Check the blog post I wrote about it [pan-samuel.github.io]
    - don't use tf32 tensor cores so greatly underperforms triton
