#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_bf16.h>

#include "ptx.cuh"

constexpr int WARP_THREADS = 32;
constexpr int WARPGROUP_WARPS = 4;
constexpr int WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS;

template <int BlockMajorSize, int BlockMinorSize, int swizzle_bytes>
__host__ static inline CUtensorMap create_tensor_map(const nv_bfloat16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    constexpr int swizzle_elements = swizzle_bytes / sizeof(nv_bfloat16);
    constexpr CUtensorMapSwizzle tma_swizzle = 
        swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
                               CU_TENSOR_MAP_SWIZZLE_NONE;
    uint64_t gmem_prob_shape[5] = {(uint64_t) swizzle_elements, (uint64_t) global_height, (uint64_t) global_width / swizzle_elements, 1, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t) global_width * sizeof(nv_bfloat16), (uint64_t)swizzle_bytes * sizeof(nv_bfloat16), 0, 0, 0};
    uint32_t smem_box_shape[5] = {(uint64_t)swizzle_elements, (uint64_t)(BlockMajorSize), (uint64_t)BlockMinorSize/swizzle_elements, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        tma_swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;

template <int BM, int BN, int BK>
struct SharedStorage {
  alignas(128) nv_bfloat16 A[BM*BK];
  alignas(128) nv_bfloat16 B[BK*BN];
};

template <int BM, int BN, int BK, int NUM_THREADS, int swizzle_elements>
__global__ void __launch_bounds__(NUM_THREADS) 
matmul_kernel_v1(int M, int N, int K, nv_bfloat16* C, 
                 const __grid_constant__ CUtensorMap tensorMapA,
                 const __grid_constant__ CUtensorMap tensorMapB) 
{
    int lane_id = threadIdx.x % WARP_THREADS;
    int warp_id = threadIdx.x / WARP_THREADS;
    int warpgroup_id = threadIdx.x / WARPGROUP_THREADS;
    int tid = threadIdx.x % 128;

    constexpr uint32_t UMMA_M = BM, UMMA_N = BN, UMMA_K = 16;
    const int num_tiles_k = K / BK;
    const int num_rows_n = N / BN;
    const int tile_m = blockIdx.x / num_rows_n;
    const int tile_n = blockIdx.x % num_rows_n;

    extern __shared__ __align__(128) uint8_t smem[];
    SharedStorage<BM, BN, BK> &s = *reinterpret_cast<SharedStorage<BM, BN, BK>*>(smem);
    nv_bfloat16 *sA = s.A;
    nv_bfloat16 *sB = s.B;

    __shared__ __align__(8) uint64_t tma_bar;
    __shared__ __align__(8) uint64_t mma_bar;
    __shared__ __align__(8) uint32_t tmem_addr_shared;

    if (threadIdx.x == 0) {
        init_barriers(&tma_bar, 1);
        init_barriers(&mma_bar, 1);
    }

    uint32_t tmem_addr = 0;
    uint32_t n_cols = 32;
    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_shared, n_cols);
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    tmem_addr = tmem_addr_shared;
    
    uint32_t tma_phase_bit = 0;
    uint32_t mma_phase_bit = 0;

    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
        if (tid == 0) {
            // load to tma
            expect_bytes(&tma_bar, (BK*BN+BK*BM)*sizeof(nv_bfloat16));
            tma_load<swizzle_elements>(&sA[0], &tensorMapA, &tma_bar, k_tile*BK, tile_m*BM);
            tma_load<swizzle_elements>(&sB[0], &tensorMapB, &tma_bar, k_tile*BK, tile_n*BN);
        }
        constexpr uint32_t idesc = 
            (0b00 << 0)              | // dense mma
            (0b0 << 2)               | // no sparsity
            (0b0 << 3)               | // no saturation
            (0b01 << 4)              | // F32 Accum
            (0b0 << 6)               | // Reserved
            (0b001 << 7)             | // BF16 A dtype
            (0b001 << 10)            | // BF16 B dtype
            (0b0 << 13)              | // No Negation
            (0b0 << 14)              | // No Negation
            (0b0 << 15)              | // A K-Major
            (0b1 << 16)              | // B K-Major
            ((BN >> 3) << 17)        | // N, encoded
            (0b0 << 23)              | // Reserved
            ((BM >> 3) << 24)        | // M, encoded
            (0b0 << 29)              | // Reserved
            (0b00 << 30);              // No B reuse

        uint64_t a_desc = 
            (((static_cast<uint32_t>(__cvta_generic_to_shared(&sA[0])) & 0x3FFFF) >> 4) << 0) |
            (0b00L << 14)                             |
            (0x0L << 16)                              |
            (((BK * sizeof(nv_bfloat16) & 0x3FFFF) >> 4) << 32)  |
            (0b001L << 46)                   |
            (0b000L << 49)                   |
            (0b0L << 52)                     | 
            (0b0000'0000L << 53)             | 
            (0x6L << 61);                        
            
        uint64_t b_desc = 
            (((static_cast<uint32_t>(__cvta_generic_to_shared(&sB[0])) & 0x3FFFF) >> 4) << 0) |
            (0x0L << 16)                              |
            ((((BK * BN * sizeof(nv_bfloat16)) & 0x3'FFFF) >> 4) << 32) |
            (0b001L << 46)                            |
            (0b000L << 49)                            |
            (0b0L << 52)                              | 
            (0b0000'0000L << 53)                      | 
            (0x6L << 61);

        // wait for tma_bar to complete
        wait(tma_bar, tma_phase_bit);
        tma_phase_bit ^= 1;
        if (tid == 0) {
            tcgen05_mma(tmem_addr, idesc, a_desc, b_desc);
        }
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");

            // for (int k = 0; k < BK / UMMA_K; ++k) {
            //     a_desc += UMMA_K;
            //     b_desc += UMMA_K;
            // }

        //     tcgen05_commit_group(&mma_bar);
        }
            
            // 
            //  // tgen05 mma commit

            // wait(&mma_bar, mma_phase_bit);
            // mma_phase_bit ^= 1;
            // asm volatile("tcgen05.fence::before_thread_sync;\n");
            // asm volatile("bar.sync 0;\n");
            // asm volatile("tcgen05.fence::after_thread_sync;\n");
            // asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    //         float tmp[32];
    //         asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    //             : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
    //             "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
    //             "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
    //             "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
    //             "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
    //             "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
    //             "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
    //             "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
    //             : "r"(tmem_addr + (0 << 16) + (0))
    //         );

    //         //wait ld
    //         asm volatile("tcgen05.wait::ld.sync.aligned;");
        
    }
    if (warp_id == 0) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_addr), "r"(n_cols)
        );
    }

}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    constexpr int swizzle_bytes = 32;
    constexpr int swizzle_elements = swizzle_bytes / sizeof(nv_bfloat16);

    d_tma_map_A = create_tensor_map<BM, BK, swizzle_bytes>(A, M, K);
    d_tma_map_B = create_tensor_map<BN, BK, swizzle_bytes>(B, N, K);

    size_t smem_size = sizeof(SharedStorage<BM, BN, BK>);
    auto* kernel = matmul_kernel_v1<BM,BN,BK, NUM_THREADS, swizzle_elements>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(
        M, N, K, C, d_tma_map_A, d_tma_map_B
    );
}
