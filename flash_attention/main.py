import torch
import math
import torch.utils.cpp_extension
from triton.testing import do_bench

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    "module",
    sources=["flashattention_v1.cu", "flashattention_v2.cu", "flashattention_v3.cu", "flashattention_v4.cu",  "flash_attention.cpp"],
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

B, n_heads = 64, 12
T, head_dim = 1024, 64

dtype = torch.float32
device = 'cuda'

q = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)
k = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)
v = torch.randn(B, n_heads, T, head_dim, dtype=dtype, device=device, requires_grad=False)

output_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
output_v1 = module.flashattn_v1(q, k, v)
output_v2 = module.flashattn_v2(q, k, v)
output_v3 = module.flashattn_v3(q, k, v)
output_v4 = module.flashattn_v4(q, k, v)


torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3, output_ref)



print("Manual Attention Masking:", benchmark(manual_attention_masking, q, k, v))
print("Pytorch Flash attention:", benchmark(torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=True))
print("v1:", benchmark(module.flashattn_v1, q, k, v))
print("v2:", benchmark(module.flashattn_v2, q, k, v))
print("v3:", benchmark(module.flashattn_v3, q, k, v))
print("v4:", benchmark(module.flashattn_v4, q, k, v))


