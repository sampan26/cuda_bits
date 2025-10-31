import torch
import math
import torch.utils.cpp_extension
from triton.testing import do_bench


def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    "module",
    sources=["01-bf16-gemm.cu", "gemm.cpp"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",  # Blackwell architecture
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "--use_fast_math",
        # "--ptxas-options=-v",
        # "-lineinfo",
        "-allow-unsupported-compiler"
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=True,
)

M, N, K = 8192, 8192, 8192

dtype = torch.bfloat16
device = 'cuda'

A = torch.randn(M, K, dtype=dtype, device=device, requires_grad=False)
B = torch.randn(N, K, dtype=dtype, device=device, requires_grad=False)

output_ref = torch.matmul(A, B.T)
# output_v1 = module.matmul_v1(A, B)

# torch.testing.assert_close(output_v1, output_ref)

print("Pytorch Matmul:", benchmark(torch.matmul, A, B.T))
# print("v1:", benchmark(module.flashattn_v1, A, B))
