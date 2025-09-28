#include "wgmma_ops.cuh" 

namespace M2 {

typedef __nv_bfloat16 bf16;

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap* tma_map, bf16 *src, int shape_major, int shape_minor) {
  void *src_addr = (void*)src;
  uint64_t gmem_prob_shape[5] = {(uint64_t)shape_minor, (uint64_t)shape_major, 1, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * shape_minor, 0, 0, 0};
  uint32_t smem_box_shape[5] = {(uint32_t)BlockMinorSize, (uint32_t)BlockMajorSize, 1, 1, 1}; // BN, BM for your block sizes
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1}; 

  CUresult result = cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
                    2, src_addr, gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape,
                    smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, 
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}


CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* init_tensor_map(bf16* src, int shape_major, int shape_minor) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_h;  // Fixed: declare as value
  create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_h, src, shape_major, shape_minor);
  cudaMemcpy(tma_map_d, &tma_map_h, sizeof(CUtensorMap), cudaMemcpyHostToDevice);  // Fixed: use tma_map_h
  return tma_map_d;
}

template <int BM, int BN, int BK>
struct SharedStorage {
  alignas(128) bf16 A[BM*BK];
  alignas(128) bf16 B[BK*BN];
};

template<int BM, int BN, int BK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
matmul_kernel_v2(int M, int N, int K, bf16* C, 
                 const CUtensorMap* tensorMapA, 
                 const CUtensorMap* tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_wg_m = BM / (NUM_THREADS / 128);
    extern __shared__ SharedStorage<BM, BN, BK> smem;
    bf16 *sA = smem.A;
    bf16 *sB = smem.B;

    float d[num_wg_m / WGMMA_M][WGMMA_N/16][8];
    memset(d, 0, sizeof(d));

    const int num_tiles_k = K / BK;
    const int num_rows_n = N / BN;
    const int tile_m = blockIdx.x / num_rows_n;
    const int tile_n = blockIdx.x % num_rows_n;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar_A, bar_B;

    if (threadIdx.x == 0) {
      init(&bar_A, blockDim.x);
      init(&bar_B, blockDim.x);
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    int wg_idx = threadIdx.x / 128;

    barrier::arrival_token token_A, token_B;

    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
      if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, k_tile*BK, tile_m*BM, bar_A);
        token_A = cuda::device::barrier_arrive_tx(bar_A, 1, BM * BK * sizeof(bf16));
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, k_tile*BK, tile_n*BN, bar_B);
        token_B = cuda::device::barrier_arrive_tx(bar_B, 1, BN * BK * sizeof(bf16));
      }
      else {
        token_A = bar_A.arrive();
        token_B = bar_B.arrive();
      }
      bar_A.wait(std::move(token_A));
      bar_B.wait(std::move(token_B));
      __syncthreads();

      warpgroup_arrive();
      for (int m = 0; m < num_wg_m /WGMMA_M; ++m) {
        bf16 *wg_a_tile_m = sA + WGMMA_M*(m + wg_idx*num_wg_m/WGMMA_M)*BK;
        for (int k = 0; k < BK / WGMMA_K; ++k) {
          wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m], &wg_a_tile_m[k*WGMMA_K], &sB[k*WGMMA_K]);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
      }
    }
    {
      int tid = threadIdx.x;
      int lane_id = tid % 32;
      int warp_id = tid / 32;
      uint32_t row = warp_id*16 + lane_id / 4;

      bf16 *out_C = C + tile_n*BN*M + tile_m*BM;

      for (int m = 0; m < num_wg_m / WGMMA_M; ++m) {
        int wg_block_m = m * WGMMA_M + wg_idx * num_wg_m;
          for (int w = 0; w < WGMMA_N / 16; ++w) {
            int col = w*16 + 2*(tid % 4);
            #define IDX(i, j) ((j)*M + (i + wg_block_m))

            out_C[IDX(row, col)] = d[m][w][0];
            out_C[IDX(row, col+1)] = d[m][w][1];
            out_C[IDX(row+8, col)] = d[m][w][2];
            out_C[IDX(row+8, col+1)] = d[m][w][3];

            out_C[IDX(row, col+8)] = d[m][w][4];
            out_C[IDX(row, col+9)] = d[m][w][5];
            out_C[IDX(row+8, col+8)] = d[m][w][6];
            out_C[IDX(row+8, col+9)] = d[m][w][7];

            #undef IDX
          
          }
    }
  }
}
void matmul_v2(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128;
  
  d_tma_map_A = init_tensor_map<BM, BK>(A, M, K);
  d_tma_map_B = init_tensor_map<BN, BK>(B, N, K);
 
  size_t smem_size = sizeof(SharedStorage<BM, BN, BK>);
  auto* kernel = matmul_kernel_v2<BM,BN,BK, NUM_THREADS>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(
    M, N, K, C, d_tma_map_A, d_tma_map_B
  );
}

};
using M2::matmul_v2;
