#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_bf16.h>

#include "ptx.cuh"

constexpr int WARP_THREADS = 32;
constexpr int WARPGROUP_WARPS = 4;
constexpr int WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS;

__forceinline__ __device__ uint32_t get_tmem_addr(uint32_t idx, int row_offset, int col_offset) {
    int col_idx = idx & 0xFFFF;
    int row_idx = (idx >> 16) & 0xFFFF;
    col_idx += col_offset;
    row_idx += row_offset;
    col_idx = col_idx & 0xFFFF;
    row_idx = row_idx & 0xFFFF;
  
    uint32_t new_idx = (row_idx << 16) | col_idx;
    return new_idx;
  }
  

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
    uint64_t gmem_prob_shape[5] = {
        (uint64_t) swizzle_elements, 
        (uint64_t) global_height, 
        (uint64_t) global_width / swizzle_elements, 
        1, 
        1
    };
    uint64_t gmem_prob_stride[5] = {
        (uint64_t) global_width * sizeof(nv_bfloat16), 
        (uint64_t)swizzle_bytes,
        (uint64_t) global_height * global_width * sizeof(nv_bfloat16), 
        (uint64_t) global_height * global_width * sizeof(nv_bfloat16), 
    };
    uint32_t smem_box_shape[5] = {
        (uint32_t)swizzle_elements, 
        (uint32_t)BlockMajorSize, 
        (uint32_t)BlockMinorSize/swizzle_elements, 
        1, 
        1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
                            &tma_map, 
                            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
                            5, 
                            gmem_address, 
                            gmem_prob_shape,
                            gmem_prob_stride, 
                            smem_box_shape, 
                            smem_box_stride, 
                            CU_TENSOR_MAP_INTERLEAVE_NONE,
                            tma_swizzle, 
                            CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                        );

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;

template <int BM, int BN, int BK>
struct SharedStorage {
  alignas(256) nv_bfloat16 A[BM*BK];
  alignas(256) nv_bfloat16 B[BK*BN];
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
    __syncthreads();

    uint32_t tmem_addr = 0;
    uint32_t n_cols = 512;
    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_shared, n_cols);
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

        wait(tma_bar, tma_phase_bit);
        tma_phase_bit ^= 1;

        constexpr uint32_t idesc =
            (0b00  << 0)                 | // sparsity selector
            (0b0   << 2)                 | // dense
            (0b0   << 3)                 | // no saturate
            (0b01  << 4)                 | // D (accum) type = F32
            (0b0   << 6)                 | // reserved
            (0b001 << 7)                 | // A type = BF16
            (0b001 << 10)                | // B type = BF16
            (0b0   << 13)                | // no negate A
            (0b0   << 14)                | // no negate B
            (0b0   << 15)                | // A: no transpose (K-major)
            (0b0   << 16)                | // B: bo transpose (K-major) << - Keep!
            ((BN >> 3) << 17)            | // N field (6 bits)
            (0b0   << 23)                | // reserved
            ((BM >> 4) << 24)            | // M field (5 bits)  <<— FIXED
            (0b0   << 29)                | // reserved
            (0b00  << 30);                 // no B reuse shift
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (k_tile == 0) {
                tcgen05_mma</* init =*/true>(tmem_addr, idesc, &sA[0], &sB[0]);
            } else {
                tcgen05_mma</* init =*/false>(tmem_addr, idesc, &sA[0], &sB[0]);
            }
            for (int k = 1; k < BK / UMMA_K; ++k) {  
                tcgen05_mma<false>(tmem_addr, idesc, &sA[k*UMMA_K], &sB[k*UMMA_K]);
            }
            tcgen05_commit_group(&mma_bar);
        }

        wait(mma_bar, mma_phase_bit);
        mma_phase_bit ^= 1;
    }

    nv_bfloat16 *block_C = C + tile_m*BM*N + tile_n*BN;
    float tmp[128];
    const int row = tid;
    for (int c = 0; c < 128; ++c) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
        : "=f"(tmp[c])
        : "r"(get_tmem_addr(tmem_addr, tid/32 * 32, c)));

    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    for (int c = 0; c < 128; ++c) {
        #define IDX(i, j) (i*N + j)
        block_C[IDX(row, c)] = __float2bfloat16(tmp[c]);
        #undef IDX
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

    constexpr int swizzle_bytes = 128;
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
