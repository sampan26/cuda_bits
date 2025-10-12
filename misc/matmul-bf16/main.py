import torch
import torch.utils.cpp_extension
from triton.testing import do_bench
import time
import statistics
import numpy as np

def warm_up_gpu():
    """Warm up the GPU to ensure it's in a consistent power state"""
    dummy = torch.ones(8192, 8192, device='cuda')
    dummy = dummy @ dummy
    torch.cuda.synchronize()
    del dummy
    torch.cuda.empty_cache()

def reset_gpu_state():
    """Reset GPU state between benchmarks"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(0.1)  # Short cooldown between runs

def benchmark_with_stats(f, args, repeats=10, **kwargs):
    """Run multiple benchmarks and gather statistics"""
    results = []
    for _ in range(repeats):
        reset_gpu_state()
        result = do_bench(lambda: f(*args, **kwargs), return_mode="median")
        results.append(result)
    
    return {
        'median': statistics.median(results),
        'mean': statistics.mean(results),
        'min': min(results),
        'max': max(results),
        'std_dev': statistics.stdev(results) if len(results) > 1 else 0,
        'raw_results': results
    }

def run_benchmarks():
    module = torch.utils.cpp_extension.load(
        "module",
        sources=["matmul.cu", "matmul.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v", "-allow-unsupported-compiler"],
        verbose=True,
    )

    print("Warming up GPU...")
    warm_up_gpu()
    
    print("Setting up tensors...")
    input1 = torch.randn(4096, 4096, device='cuda').bfloat16()
    input2 = torch.randn(4096, 4096, device='cuda').bfloat16().T
    
    # First verify correctness
    output_ref = torch.matmul(input1, input2)
    
    # Create a dictionary of functions to benchmark
    kernels = {
        "CuBLAS": torch.matmul,
        "v1a": module.matmul_v1a,
        "v1b": module.matmul_v1b,
        "v2a": module.matmul_v2a,
        "v2b": module.matmul_v2b,
        "v3a": module.matmul_v3a,
        "v3b": module.matmul_v3b,
        "v4a": module.matmul_v4a,
        "v4b": module.matmul_v4b,
    }
    
    # Verify all kernels
    print("Verifying kernel correctness...")
    for name, kernel in kernels.items():
        if name != "CuBLAS":  # Skip reference implementation
            output = kernel(input1, input2)
            try:
                torch.testing.assert_close(output, output_ref)
                print(f"{name}: Correctness verified ✓")
            except Exception as e:
                print(f"{name}: Failed verification ✗")
                print(f"Error: {e}")
    
    # Benchmark all kernels
    print("\nRunning benchmarks (this may take a while)...")
    results = {}
    
    # Always benchmark in the same order
    for name, kernel in kernels.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_with_stats(kernel, (input1, input2))
    
    # Print results in a table
    print("\nResults (milliseconds):")
    print(f"{'Kernel':<10} {'Median':<10} {'Mean':<10} {'Min':<10} {'Max':<10} {'StdDev':<10}")
    print("-" * 60)
    
    for name, stats in results.items():
        print(f"{name:<10} {stats['median']*1000:<10.4f} {stats['mean']*1000:<10.4f} "
              f"{stats['min']*1000:<10.4f} {stats['max']*1000:<10.4f} {stats['std_dev']*1000:<10.4f}")
    
    # Calculate coefficients of variation (relative standard deviations)
    print("\nCoefficient of Variation (lower is more consistent):")
    for name, stats in results.items():
        cv = (stats['std_dev'] / stats['mean']) * 100 if stats['mean'] > 0 else float('inf')
        print(f"{name:<10} {cv:.2f}%")

if __name__ == "__main__":
    run_benchmarks()