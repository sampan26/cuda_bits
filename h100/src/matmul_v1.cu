namespace M1 {

typedef __nv_bfloat16 bf16;

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(bf16* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode((uint64_t)16) << 16;
  desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
  desc |= 1llu << 62; // 128B swizzle
  return desc;
}


__device__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
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

void create_tensor_map(CUtensorMap* tma_map, bf16 *src, int shape_major, int shape_minor) {
  void *src_addr = (void*)src;
  uint64_t gmem_prob_shape[5] = {(uint64_t)shape_minor, (uint64_t)shape_major, 1, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * shape_minor, 0, 0, 0};
  uint32_t smem_box_shape[5] = {64, 64, 1, 1, 1}; // BN, BM for your block sizes
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1}; 

  CUresult result = cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
                    2, src_addr, gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape,
                    smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, 
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}


CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;

__host__ static inline CUtensorMap* init_tensor_map(bf16* src, int shape_major, int shape_minor) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_h;  // Fixed: declare as value
  create_tensor_map(&tma_map_h, src, shape_major, shape_minor);
  cudaMemcpy(tma_map_d, &tma_map_h, sizeof(CUtensorMap), cudaMemcpyHostToDevice);  // Fixed: use tma_map_h
  return tma_map_d;
}


template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
matmul_kernel_v1(int M, int N, int K, bf16* C, 
                 const CUtensorMap* tensorMapA, 
                 const CUtensorMap* tensorMapB) {
    __shared__ alignas(128) bf16 sA[BM * BK];
    __shared__ alignas(128) bf16 sB[BK * BN];
    float d[WGMMA_N/16][8];
    memset(d, 0, sizeof(d));

    const int num_tiles_k = K / BK;
    const int num_rows_n = N / BN;
    const int tile_m = blockIdx.x / num_rows_n;
    const int tile_n = blockIdx.x % num_rows_n;

    __shared__ barrier bar_A;
    __shared__ barrier bar_B;

    if (threadIdx.x == 0) {
      init(&bar_A, blockDim.x);
      init(&bar_B, blockDim.x);
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token token_A, token_B;

    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
      if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, k_tile*BK, tile_m*BM, bar_A);
        token_A = cuda::device::barrier_arrive_tx(bar_A, 1, sizeof(sA));
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, k_tile*BK, tile_n*BN, bar_B);
        token_B = cuda::device::barrier_arrive_tx(bar_B, 1, sizeof(sB));
      }
      else {
        token_A = bar_A.arrive();
        token_B = bar_B.arrive();
      }
      bar_A.wait(std::move(token_A));
      bar_B.wait(std::move(token_B));
      __syncthreads();


      warpgroup_arrive();
      wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
      wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
      wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
      wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
    }
    {
      int tid = threadIdx.x;
      int lane_id = tid % 32;
      int warp_id = tid / 32;
      uint32_t row = warp_id*16 + lane_id / 4;

      bf16 *out_C = C + tile_n*BN*M + tile_m*BM;
      for (int m = 0; m < BM / WGMMA_M; ++m) {
        for (int n = 0; n < BN/WGMMA_N; ++n) {
          for (int w = 0; w < WGMMA_N / 16; ++w) {
            int col = w*WGMMA_K + 2*(tid % 4);
            #define IDX(i, j) ((n*WGMMA_N+j)*M + ((i) + m*WGMMA_M))

            out_C[IDX(row, col)] = d[w][0];
            out_C[IDX(row, col+1)] = d[w][1];
            out_C[IDX(row+8, col)] = d[w][2];
            out_C[IDX(row+8, col+1)] = d[w][3];

            out_C[IDX(row, col+8)] = d[w][4];
            out_C[IDX(row, col+9)] = d[w][5];
            out_C[IDX(row+8, col+8)] = d[w][6];
            out_C[IDX(row+8, col+9)] = d[w][7];

            #undef IDX
          }
      }
    }
  }
}
void matmul_v1(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128;
  
  d_tma_map_A = init_tensor_map(A, M, K);
  d_tma_map_B = init_tensor_map(B, N, K);

  matmul_kernel_v1<BM,BN,BK, 64, 64, 16, NUM_THREADS><<<(M/BM) * (N/BN), NUM_THREADS>>>(
    M, N, K, C, d_tma_map_A, d_tma_map_B);
}

};
using M1::matmul_v1;