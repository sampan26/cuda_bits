import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    "module",
    sources=["matmul_v1.cu", "matmul_v2.cu", "matmul.cpp"],
    extra_cuda_cflags=[
        "-O3",
        #"--use_fast_math",
        "--ptxas-options=-v",
        "-gencode=arch=compute_90,code=sm_90",
    ],
    extra_ldflags=[
        "-lcuda",
        "-L/ib-scratch/chenguang03/scratch/pan.samuel/misc/cuda_install/targets/x86_64-linux/lib/stubs"
    ],
    verbose=True,
)



input1 = torch.randn(4096, 4096, device='cuda').bfloat16()
input2 = torch.randn(4096, 4096, device='cuda').bfloat16()

output_ref = torch.matmul(input1, input2)
#output_v1 = module.matmul_v1(input1, input2)
output_v2 = module.matmul_v2(input1, input2)
#torch.testing.assert_close(output_ref, output_v1, rtol=1e-2, atol=1e-3)

print("CuBLAS:", benchmark(torch.matmul, input1, input2))
#print("v1:", benchmark(module.matmul_v1, input1, input2))
print("v2:", benchmark(module.matmul_v2, input1, input2))
