#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/pipeline>  

using bf16 = __nv_bfloat16;

namespace cde = cuda::device::experimental;

// --- shared-mem matrix descriptor helpers -----------------------------------

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 0x4);
}

__device__ inline uint64_t make_smem_desc(bf16* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0;
  desc |= matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode((uint64_t)16)  << 16;  // leading dimension in bytes/elt groups
  desc |= matrix_descriptor_encode((uint64_t)1024)<< 32;  // stride (adjust as needed)
  desc |= 1ull << 62;                                     // “smem” bit
  return desc;
}

// --- barrier helpers -----------------------------------
__device__ static __forceinline__ void init_barriers(uint64_t* bar, int thread_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
  asm volatile (
      "mbarrier.init.shared::cta.b64 [%0], %1;\n"
      :: "r"(bar_ptr), "r"(thread_count)
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

__device__ static __forceinline__ void wait_cluster(uint64_t* bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
  asm volatile (
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n"
      :: "r"(mbar_ptr),
      "r"(kPhaseBit)
  );
}

__device__ static __forceinline__ void expect_bytes(uint64_t* mbar, uint32_t bytes) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile (
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(mbar_addr), "r"(bytes)
  );
}

__device__ static inline void load_async(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
        : "memory"
    );
}

__device__ static inline void load_async_3d(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
  uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile (
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
      : "memory"
  );
}


__device__ static inline void load_async_multi(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx, uint16_t mask) {
  uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64), "h"(mask)
      : "memory"
  );
}

__device__ static inline void load_async_3d_L2(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
  uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint64_t cache_hint = static_cast<uint64_t>(0x1000000000000000);

  asm volatile (
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64), "l"(cache_hint)
      : "memory"
  );
}


__device__ static inline void load_async_multi_L2(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx, uint16_t mask) {
  uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint64_t cache_hint = static_cast<uint64_t>(0x1000000000000000);


  asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64), 
      "h"(mask), 
      "l"(cache_hint)
      : "memory"
  );
}

__device__ static inline void store_async(void const* dst_tma_map, bf16 *src, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(tma_ptr), "r"(src_addr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
    );
}


__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}



__device__ void arrive_cluster(uint64_t* bar, uint32_t cta_id, uint32_t count=1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}

__device__ void prefetch_tma(CUtensorMap const* ptr) {
    uint64_t desc_addr = reinterpret_cast<uint64_t>(ptr);
    asm volatile(
        "prefetch.tensormap [%0];"
        :
        : "l"(desc_addr)
        : "memory");
}
// --- warpgroup flow helpers --------------------------------------------------

__device__ inline void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ inline void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ inline void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ inline void warpgroup_fence_memory(float regs[1][16][8]) {
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 16; ++j) {
      for (int k = 0; k < 8; ++k) {
        asm volatile("" : "+f"(regs[i][j][k])::"memory");
      }
    }
  }
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n256k16(float d[16][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n128k16(float d[8][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
            "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
            "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
            "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
            "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n192k16(float d[12][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
        " %96,"
        " %97,"
        " %98,    %99,  %100,  %101,  %102;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void __forceinline__ wgmma_m64n64k16(float d[4][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = make_smem_desc(&sA[0]);
  uint64_t desc_b = make_smem_desc(&sB[0]);
  asm volatile(
    "{\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
    " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
    " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
    " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
    " %32,"
    " %33,"
    " %34, %35, %36, %37, %38;\n"
    "}\n"
    : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
      "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
      "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
      "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
      "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
      "+f"(d[3][6]), "+f"(d[3][7])
    : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
      "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void __forceinline__ wgmma_m64n32k16(float d[2][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = make_smem_desc(&sA[0]);
  uint64_t desc_b = make_smem_desc(&sB[0]);
  asm volatile(
    "{\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
    "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
    " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
    " %16,  "
    " %17,  "
    " %18,  %19,  %20,  %21,  %22,  %23;\n"
    "}\n"
    : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
      "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
      "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
    : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
      "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void __forceinline__ wgmma_m64n16k16(float d[1][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = make_smem_desc(&sA[0]);
  uint64_t desc_b = make_smem_desc(&sB[0]);
  asm volatile(
    "{\n"
    "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
    "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
    " %8,   "
    " %9,   "
    " %10,  %11,  %12,  %13,  %14;\n"
    "}\n"
    : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
      "+f"(d[0][6]), "+f"(d[0][7])
    : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
      "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma(float d[WGMMA_N/16][8], bf16 *sA, bf16 *sB) {
  static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 192 || WGMMA_N == 256);
  if constexpr (WGMMA_N == 256)
    wgmma_m64n256k16<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 192)
    wgmma_m64n192k16<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 128)
    wgmma_m64n128k16<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 64)
    wgmma_m64n64k16<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 32)
    wgmma_m64n32k16<1, 1, 1, 0, 0>(d, sA, sB);
}
