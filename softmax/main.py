import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    "module",
    sources=["softmax.cu", "softmax.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math",  "-allow-unsupported-compiler"],
    verbose=True,
)

input = torch.randn(1024, 32768).cuda()

output_ref = torch.softmax(input, dim=-1)
output_v1 = module.softmax_v1(input)

torch.testing.assert_close(output_v1, output_ref)

print("CuBLAS:", benchmark(torch.softmax, input, dim=-1))
print("v1:", benchmark(module.softmax_v1, input))

