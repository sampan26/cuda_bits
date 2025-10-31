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

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
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
__device__ static __forceinline__ void tmem_alloc(uint64_t& dst_addr, int n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;\n"
        :: "l"((uint64_t)&dst_addr), "r"(n_cols)
    );
}

template <bool Transpose>
__device__ inline uint64_t make_smem_desc(nv_bfloat16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 
        (((addr & 0x3FFFF) >> 4) << 0)   |  //encoded start address
        (0x0L << 16)                     |
        Transpose ? ((((512L * N/16) & 0x3'FFFF) >> 4) << 16) : (((256L & 0x3FFFF) >> 4) << 32)  |
        (0b001L << 46)                   |
        (0b000L << 49)                   |
        (0b0L << 52)                     | 
        (0b0000'0000L << 53)             | // SBZ
        (0x6L << 61);                               // 32B swizzling mode

}

__device__ __forceinline__ void tcgen05_mma(uint32_t tm_addr, nv_bfloat16* sA, nv_bfloat16* sB) {
    constexpr uint32_t idesc = 
        (0b00 << 0)      | // dense mma
        (0b0 << 2)       | // no sparsity
        (0b0 << 3)       | // no saturation
        (0b01 << 4)      | // F32 Accum
        (0b0 << 6)       | // Reserved
        (0b001 << 7)     | // BF16 A dtype
        (0b001 << 10)    | // BF16 B dtype
        (0b0 << 13)      | // No Negation
        (0b0 << 14)      | // No Negation
        (0b0 << 15)      | // No Transpose A
        (0b1 << 16)      | // Transpose B
        ((N >> 3) << 17) | // N, encoded
        (0b0 << 23)      | // Reserved
        ((M >> 3) << 24) | // M, encoded
        (0b0 << 29)      | // Reserved
        (0b0 << 30)      | // No B resuse
    
    constexpr uint64_t adesc = make_smem_desc(sA);
    constexpr uint64_t bdesc = make_smem_desc(sB);

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.f16 [%0], %1, %2, %3, p; \n\t"
        "}\n"
        :
        : "r"(tm_addr), "l(adesc)", "l"(bdesc), "r"(idesc), "n"(1)
    );
}

__device__ __forceinline__ void tcgen05_commit_group(uint64_t *bar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "l"(__cvta_generic_to_shared(mbar_ptr))
    );
}
