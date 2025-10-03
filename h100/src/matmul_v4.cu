#include "wgmma_ops.cuh" 

namespace M4 {

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
  CUtensorMap tma_map_h; 
  create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_h, src, shape_major, shape_minor);
  cudaMemcpy(tma_map_d, &tma_map_h, sizeof(CUtensorMap), cudaMemcpyHostToDevice);  // Fixed: use tma_map_h
  return tma_map_d;
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <int BM, int BN, int BK, int PIPE>
struct SharedStorage {
  alignas(128) bf16 A[BM*BK*PIPE];
  alignas(128) bf16 B[BK*BN*PIPE];
};

template<int BM, int BN, int BK, int NUM_THREADS, int PIPE>
__global__ void __launch_bounds__(NUM_THREADS) 
matmul_kernel_v4(int M, int N, int K, bf16* C, 
                 const CUtensorMap* tensorMapA, 
                 const CUtensorMap* tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int wg_m = BM / num_consumers;

    extern __shared__ __align__(128) uint8_t smem[];
    SharedStorage<BM, BN, BK, PIPE> &s = *reinterpret_cast<SharedStorage<BM, BN, BK, PIPE>*>(smem);
    bf16 *sA = s.A;
    bf16 *sB = s.B;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full_barrier[PIPE], empty_barrier[PIPE];

    const int num_tiles_k = K / BK;
    const int num_rows_n = N / BN;
    const int tile_m = blockIdx.x / num_rows_n;
    const int tile_n = blockIdx.x % num_rows_n;
    int wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
      for (int i = 0; i < PIPE; ++i) {
        init(&full_barrier[i], num_consumers*128+1);
        init(&empty_barrier[i], num_consumers*128+1);
      }
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (wg_idx == 0) 
    {
      constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
      warpgroup_reg_dealloc<num_regs>();
      if (tid == 0) {
        int pipe_lane = 0;
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile, ++pipe_lane) {
          if (pipe_lane == PIPE) pipe_lane = 0;
          empty_barrier[pipe_lane].wait(empty_barrier[pipe_lane].arrive());
          cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[pipe_lane*BM*BK], tensorMapA, k_tile*BK, tile_m*BM, full_barrier[pipe_lane]);
          cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[pipe_lane*BN*BK], tensorMapB, k_tile*BK, tile_n*BN, full_barrier[pipe_lane]);
          barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full_barrier[pipe_lane], 1, (BK*BN+BK*BM)*sizeof(bf16));
        }
      }
    }
    else 
    {
      constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
      warpgroup_reg_alloc<num_regs>();
      
      for (int i = 0; i < PIPE; ++i) {
        barrier::arrival_token _ = empty_barrier[i].arrive();
      }

      float d[wg_m/WGMMA_M][WGMMA_N/16][8];
      memset(d, 0, sizeof(d));
      int pipe_lane = 0;
      --wg_idx;

      for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
        if (pipe_lane == PIPE) pipe_lane = 0;
        full_barrier[pipe_lane].wait(full_barrier[pipe_lane].arrive());
        warpgroup_arrive();
        #pragma unroll
        for (int m = 0; m < wg_m / WGMMA_M; ++m) {
          bf16 *wgmma_sA = sA + pipe_lane*BM*BK + BK*m*WGMMA_M + wg_idx*wg_m*BK;
          #pragma unroll
          for (int k = 0; k < BK / WGMMA_K; ++k) {
            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m], &wgmma_sA[k*WGMMA_K], &sB[pipe_lane*BK*BN + k*WGMMA_K]);
          }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        barrier::arrival_token _ = empty_barrier[pipe_lane].arrive();
        ++pipe_lane;
      }

      int lane = tid % 32;
      int warp = tid / 32;
      int row = warp*16 + lane / 4;
      bf16 *block_C = C + tile_n*BN*M + tile_m*BM;
  
      #pragma unroll
      for (int m_it = 0; m_it < wg_m/WGMMA_M; ++m_it) {
          int yo = m_it*WGMMA_M + wg_idx * wg_m;
          #pragma unroll
          for (int w = 0; w < WGMMA_N/16; ++w) {
              int col = 16*w + 2*(tid % 4);
              #define IDX(i, j) ((j)*M + ((i) + yo))

              block_C[IDX(row, col)] = d[m_it][w][0];
              block_C[IDX(row, col+1)] = d[m_it][w][1];
              block_C[IDX(row+8, col)] = d[m_it][w][2];
              block_C[IDX(row+8, col+1)] = d[m_it][w][3];

              block_C[IDX(row, col+8)] = d[m_it][w][4];
              block_C[IDX(row, col+9)] = d[m_it][w][5];
              block_C[IDX(row+8, col+8)] = d[m_it][w][6];
              block_C[IDX(row+8, col+9)] = d[m_it][w][7];
              #undef IDX
          }
      }
    }
}
void matmul_v4(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128 * 3;
  constexpr int PIPE = 3;
  
  d_tma_map_A = init_tensor_map<BM, BK>(A, M, K);
  d_tma_map_B = init_tensor_map<BN, BK>(B, N, K);
 
  auto* kernel = matmul_kernel_v4<BM,BN,BK, NUM_THREADS, PIPE>;
  size_t smem_size = sizeof(SharedStorage<BM, BN, BK, PIPE>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<(M/BM) * (N/BN), NUM_THREADS, smem_size>>>(
    M, N, K, C, d_tma_map_A, d_tma_map_B
  );
}

};
using M4::matmul_v4;

