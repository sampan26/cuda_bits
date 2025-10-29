#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_bf16.h>

#include "ptx.cuh"

constexpr int WARP_THREADS = 32;
constexpr int WARPGROUP_WARPS = 4;
constexpr int WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS;

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap create_tensor_map(const nv_bfloat16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    constexpr int swizzle_bytes = 32;
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

template <int BM, int BN, int BK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
matmul_kernel_v1(int M, int N, int K, nv_bfloat16* C, 
                 const __grid_constant__ CUtensorMap tensorMapA,
                 const __grid_constant__ CUtensorMap tensorMapB) 
{
    constexpr int MMA_M = 64, MMA_K = 16, MMA_N = BN;

    int lane_id = threadIdx.x % WARP_THREADS;
    int warp_id = threadIdx.x / WARP_THREADS;
    int warpgroup_id = threadIdx.x / WARPGROUP_THREADS;

    constexpr int BM_MMA_M = BM / (NUM_THREADS / 128);
    const int num_tiles_k = K / BK;
    const int num_rows_n = N / BN;
    const int tile_m = blockIdx.x / num_rows_n;
    const int tile_n = blockIdx.x % num_rows_n;

    extern __shared__ __align__(128) uint8_t smem[];
    SharedStorage<BM, BN, BK> &s = *reinterpret_cast<SharedStorage<BM, BN, BK>*>(smem);
    nv_bfloat16 *sA = s.A;
    nv_bfloat16 *sB = s.B;

    __shared__ uint64_t input_arrive;
    __shared__ uint64_t tmem_addr_shared;

    if (threadIdx.x == 0) {
        init_barriers(&input_arrive, 1);
    }

    uint64_t tmem_addr = 0;
    uint32_t n_cols = 32;
    if (warp_id == 0) {
        tmem_alloc(tmem_addr_shared, n_cols);
    }

    __syncthreads();
    tmem_addr = tmem_addr_shared;
    uint32_t phase_bit = 0;
    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
        if (tid == 0) {
            expect_bytes(&input_arrive, (BK*BN+BK*BM)*sizeof(nv_bfloat16));
            tma_load(&sA, &tensorMapA, &input_arrive, k_tile*BK, tile_m*BM);
            tma_load(&sB, &tensorMapB, &input_arrive, k_tile*BK, tile_n*BN);
        }

        wait(&input_arrive, phase_bit);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

        for (int m = 0; m < BM_MMA_M / MMA_M; ++m) {
            nv_bfloat16 *m_mma_tile = sA + MMA_M*(m + warpgroup_id*BM_MMA_M/MMA_M)*BK;
            for (int k = 0; k < BK / MMA_K; ++k) {
                tcgen05_mma(&m_mma_tile[k*MMA_K], &sB[k*MMA_K]);
            }
        }
    }

}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
    d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);

    size_t smem_size = sizeof(SharedStorage<BM, BN, BK>);
    auto* kernel = matmul_kernel_v1<BM,BN,BK, NUM_THREADS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(
        M, N, K, C, d_tma_map_A, d_tma_map_B
    );
}
