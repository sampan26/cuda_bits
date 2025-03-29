#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ uint32_t cvta_shared(void const *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

template <int num> __device__ void ldmatrix(uint32_t reg[num], uint32_t addr);
template <> __device__ void ldmatrix<1>(uint32_t reg[1], uint32_t addr) {
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
    : "=r"(reg[0])  // output
    : "r"(addr)     // input
  );
}
template <> __device__ void ldmatrix<2>(uint32_t reg[2], uint32_t addr) {
  asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
    : "=r"(reg[0]), "=r"(reg[1])
    : "r"(addr)
  );
}
template <> __device__ void ldmatrix<4>(uint32_t reg[4], uint32_t addr) {
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
    : "r"(addr)
  );
}

template <int M, int N, int K, typename T> __device__ void mma(uint32_t A[], uint32_t B[], float acc[]);
template <> __device__ void mma<16, 8, 8, half>(uint32_t A[], uint32_t B[], float acc[]) {
  asm volatile (
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "  // D
    "{%4, %5}, "          // A
    "{%6}, "              // B
    "{%7, %8, %9, %10};"  // C
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
    : "r"(A[0]), "r"(A[1]),
      "r"(B[0]),
      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
  );
}
template <> __device__ void mma<16, 8, 8, nv_bfloat16>(uint32_t A[], uint32_t B[], float acc[]) {
  asm volatile (
    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, "  // D
    "{%4, %5}, "          // A
    "{%6}, "              // B
    "{%7, %8, %9, %10};"  // C
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
    : "r"(A[0]), "r"(A[1]),
      "r"(B[0]),
      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
  );
}
template <> __device__ void mma<16, 8, 16, half>(uint32_t A[], uint32_t B[], float acc[]) {
  asm volatile (
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "    // D
    "{%4, %5, %6, %7}, "    // A
    "{%8, %9}, "            // B
    "{%10, %11, %12, %13};" // C
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
  );
}
template <> __device__ void mma<16, 8, 16, nv_bfloat16>(uint32_t A[], uint32_t B[], float acc[]) {
  asm volatile (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, "    // D
    "{%4, %5, %6, %7}, "    // A
    "{%8, %9}, "            // B
    "{%10, %11, %12, %13};" // C
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])
  );
}

__device__ __forceinline__ void cp_async_cg(uint32_t dst_smem_addr, const void* src_gmem_ptr) {
    uint64_t src_gmem_addr = reinterpret_cast<uint64_t>(src_gmem_ptr);
    // Using L2::128B hint like the reference code (requires CUDA 11.4+)
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_addr), "l"(src_gmem_addr));
}

// Use cp.async.ca (cache all) as an alternative if needed.
__device__ __forceinline__ void cp_async_ca(uint32_t dst_smem_addr, const void* src_gmem_ptr) {
    uint64_t src_gmem_addr = reinterpret_cast<uint64_t>(src_gmem_ptr);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_addr), "l"(src_gmem_addr));
}

// Commit the group of pending cp.async operations.
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait for N groups of cp.async operations to complete.
// Typically N=0 waits for the most recently committed group.
// The reference uses N=0.
__device__ __forceinline__ void cp_async_wait_group() {
     asm volatile("cp.async.wait_group 0;\n" ::);
    // If targeting Ampere+, consider cp.async.wait_all or cp.async.arrive / cp.async.mbarrier.arrive
    // for potentially finer-grained synchronization, but wait_group(0) is standard.
}

// Wait for *all* pending cp.async operations (useful at the end if needed).
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}