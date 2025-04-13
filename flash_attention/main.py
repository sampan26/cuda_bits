import torch
import math
import torch.utils.cpp_extension
from triton.testing import do_bench

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    "module",
    sources=["flashattention_v1.cu", "flash_attention.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v", "-allow-unsupported-compiler"],
    verbose=True,
)

def manual_attention_masking(q, k, v):
    B, nh, T, head_dim = q.size()
    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(q.device)
    attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    y = attn @ v
    return y

B, n_heads = 32, 6
T, head_dim = 1024, 128

dtype = torch.float32
device = 'cuda'

q = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)
k = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)
v = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)

output_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
output_v1 = module.flashattn_v1(q, k, v)

torch.testing.assert_close(output_v1, output_ref,rtol=5, atol=2)


print("Manual Attention Masking:", benchmark(manual_attention_masking, q, k, v))
print("Pytorch Flash attention:", benchmark(torch.nn.functional.scaled_dot_product_attention, q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True))
print("v1:", benchmark(module.flashattn_v1, q, k, v))

