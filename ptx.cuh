#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/pipeline>  

__device__ static __forceinline__ void init_barriers(uint64_t* bar, int count) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_addr), "r"(count)
    );
}

__device__ static __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_addr), "r"(bytes)
    );
}

__device__ static __forceinline__ void wait(uint64_t& bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

// tma helpers
template <int SWIZZLE_ELEMENTS>
__device__ static inline void tma_load(nv_bfloat16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 "
        " [%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx/SWIZZLE_ELEMENTS)
        : "memory"
    );
}


// tcgen05 helpers
__device__ static __forceinline__ void tmem_alloc(uint32_t* dst_addr, int n_cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;\n"
        :: "l"(dst_addr), "r"(n_cols)
    );
}


__device__ __forceinline__ void tcgen05_mma(uint32_t tm_addr, uint32_t i_desc, uint64_t a_desc, uint64_t b_desc) {
    asm volatile(
        "{.reg .pred P1;                                              \t\n"
        "setp.eq.u32 P1, 1, %4;                                       \t\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P1;  \t\n}"
        :: "r"(tm_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "n"(/*acc=*/0)
    );
}

__device__ __forceinline__ void tcgen05_commit_group(uint64_t *bar) {
    uint32_t mbar_ptr = static_cast<uint32_t >(__cvta_generic_to_shared(bar));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(mbar_ptr)
    );
}
