import torch
import torch.utils.cpp_extension
from triton.testing import do_bench


def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")


module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul.cu", "matmul.cpp"],
    extra_cuda_cflags=["-O3","-arch=sm_80", "--use_fast_math", "-allow-unsupported-compiler", "--expt-relaxed-constexpr"],
    #verbose=True,
)

# for large n, there will be a larger deviation, since sum of many small elements are not accurate
# input1 = torch.randn(2048, 4096).bfloat16().cuda()
# input2 = torch.randn(8196, 4096).bfloat16().cuda().T
input1 = torch.randn(512, 1024).bfloat16().cuda()
input2 = torch.randn(2048, 1024).bfloat16().cuda().T

output_ref = torch.matmul(input1, input2)
output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)

torch.testing.assert_close(output_ref, output_v1, atol=1e-4, rtol=0.1)
torch.testing.assert_close(output_ref, output_v1)

print("CuBLAS:", benchmark(torch.matmul, input1, input2))
print("v1:", benchmark(module.matmul_v1, input1, input2))
print("v2:", benchmark(module.matmul_v2, input1, input2))

