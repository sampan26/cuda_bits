#include "ptx.cuh"

namespace M10 {

typedef __nv_bfloat16 bf16;

template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true, bool padding=false>
__host__ static inline CUtensorMap create_tensor_map(bf16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64*sizeof(bf16), 0, 0, 0};
    uint32_t smem_box_shape[5] = {padding ? 72 : 64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
CUtensorMap d_tma_map_C;

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
    alignas(128) bf16 C[BN*(BM + (BM/64)*8)];
};

__device__ void calculate_tile_indices(int tile_idx, int num_blocks_n, int group_size_m, int group_size_n, int tiles_in_group, int& tile_m, int& tile_n) {
    int group_idx = tile_idx / tiles_in_group;
    int tile_idx_in_group = tile_idx % tiles_in_group;
    int group_m = group_idx / (num_blocks_n / group_size_n);
    int group_n = group_idx % (num_blocks_n / group_size_n);
    int tile_group_m = tile_idx_in_group / group_size_n;
    int tile_group_n = tile_idx_in_group % group_size_n;
    tile_m = group_m * group_size_m + tile_group_m;
    tile_n = group_n * group_size_n + tile_group_n;
}

template<int BM, int BN, int BK, int NUM_THREADS, int NUM_CONSUMERS, int PIPE, int NUM_SM, int CLUSTER_M, int CLUSTER_N>
__global__  __launch_bounds__(NUM_THREADS) 
void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
matmul_kernel_v10(
    int M, int N, int K, bf16* C, 
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    const __grid_constant__ CUtensorMap tensorMapC) 
{
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int B_WG_M = BM / NUM_CONSUMERS;
    constexpr int B_WG_M_PADDED = B_WG_M + 8;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

    if (threadIdx.x == 0) {
        prefetch_tma(&tensorMapA);
        prefetch_tma(&tensorMapB);
        prefetch_tma(&tensorMapC);
    }

    extern __shared__ __align__(128) uint8_t smem[];
    SharedStorage<BM, BN, BK, PIPE> &s = *reinterpret_cast<SharedStorage<BM, BN, BK, PIPE>*>(smem);
    bf16 *sA = s.A;
    bf16 *sB = s.B;
    bf16 *sC = s.C;

    __shared__ __align__(8) uint64_t full_barrier[PIPE], empty_barrier[PIPE];
    uint32_t cluster_id, rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);

    const int num_tiles_k = K / BK;
    const int num_blocks_m = M / (BM * CLUSTER_M);
    const int num_blocks_n = N / (BN * CLUSTER_N);
    const int num_blocks = num_blocks_m * num_blocks_n;
    constexpr int group_size_m = 16/CLUSTER_M;
    constexpr int group_size_n = 8/CLUSTER_N;
    constexpr int tiles_in_group = group_size_m * group_size_n;

    int wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE; ++i) {
            init_barriers(&full_barrier[i], 1);
            init_barriers(&empty_barrier[i], NUM_CONSUMERS * CLUSTERS);
        }
    }
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N;
    uint32_t rank_n = rank % CLUSTER_N;

    if (wg_idx == 0) {
        constexpr int num_regs = (NUM_CONSUMERS <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<num_regs>();
        
        if (tid == 0) {
            int pipe_lane = 0;
            int p = 0;
            uint32_t col_mask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                col_mask |= (1 << (i * CLUSTER_N));
            }
            int tile_m, tile_n;

            for (int tile_idx = cluster_id; tile_idx < num_blocks; tile_idx+=NUM_SM/CLUSTERS) {
                calculate_tile_indices(tile_idx, num_blocks_n, group_size_m, group_size_n, tiles_in_group, tile_m, tile_n);
                tile_m = tile_m * CLUSTER_M + rank_m;
                tile_n = tile_n * CLUSTER_N + rank_n;

                for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile, ++pipe_lane) {
                    if (pipe_lane == PIPE) {pipe_lane = 0; p ^= 1; }
                    wait(&empty_barrier[pipe_lane], p);

                    expect_bytes(&full_barrier[pipe_lane], (BK*BN+BK*BM)*sizeof(bf16));
                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                        if (rank_n == 0) {
                            load_async_multi_L2(&sA[pipe_lane*BM*BK], &tensorMapA, &full_barrier[pipe_lane], k_tile*BK, tile_m*BM, mask);
                        }
                    } else {
                        load_async_3d_L2(&sA[pipe_lane*BM*BK], &tensorMapA, &full_barrier[pipe_lane], k_tile*BK, tile_m*BM);
                    }

                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            load_async_multi_L2(&sB[pipe_lane*BN*BK], &tensorMapB, &full_barrier[pipe_lane], k_tile*BK, tile_n*BN, col_mask << rank_n);
                        }
                    } else {
                        load_async_3d_L2(&sB[pipe_lane*BN*BK], &tensorMapB, &full_barrier[pipe_lane], k_tile*BK, tile_n*BN);
                    }
                    
                }
            }
        }
      }
      else {
        constexpr int num_regs = (NUM_CONSUMERS == 1) ? 256 : (NUM_CONSUMERS == 2 ? 240 : 160);
        warpgroup_reg_alloc<num_regs>();
        float d[B_WG_M / WGMMA_M][BN/16][8];

        --wg_idx;
        for (int i = 0; i < PIPE; ++i) {
            if (tid < CLUSTERS) arrive_cluster(&empty_barrier[i], tid);
        }

        int pipe_lane = 0;
        int p = 0;
        int tile_m, tile_n;
        for (int tile_idx=cluster_id; tile_idx<num_blocks; tile_idx+=NUM_SM/CLUSTERS) {
            calculate_tile_indices(tile_idx, num_blocks_n, group_size_m, group_size_n, tiles_in_group, tile_m, tile_n);
            tile_m = tile_m * CLUSTER_M + rank_m;
            tile_n = tile_n * CLUSTER_N + rank_n;
            memset(d, 0, sizeof(d));
            warpgroup_fence_memory(d);
            {
                if (pipe_lane == PIPE) { pipe_lane = 0; p ^= 1; }
                wait(&full_barrier[pipe_lane], p);
                warpgroup_arrive();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M/WGMMA_M; m_it++) {
                    bf16 *wgmma_sA = sA + pipe_lane*BM*BK + BK*(m_it + wg_idx*(B_WG_M/WGMMA_M))*WGMMA_M;
                    bf16 *wgmma_sB = sB + pipe_lane*BN*BK;
                    {
                        wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);
                        #pragma unroll
                        for (int k_it=1; k_it < BK/WGMMA_K; ++k_it) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                        }
                        wgmma_sA+=64*BM;
                        wgmma_sB+=64*BN;
                    }
                }
            
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (tid < CLUSTERS) { arrive_cluster(&empty_barrier[pipe_lane], tid); }
                ++pipe_lane;
            }
        

            for (int k_tile = 1; k_tile < num_tiles_k; ++k_tile) {
                if (pipe_lane == PIPE) { pipe_lane = 0; p ^= 1; }
                wait(&full_barrier[pipe_lane], p);
                warpgroup_arrive();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
                    bf16 *wgmma_sA = sA + pipe_lane*BM*BK + WGMMA_M*BK*(m_it + wg_idx*B_WG_M/WGMMA_M);
                    bf16 *wgmma_sB = sB + pipe_lane*BN*BK;
                    #pragma unroll
                    for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
                        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (tid < CLUSTERS) arrive_cluster(&empty_barrier[pipe_lane], tid);
                ++pipe_lane;
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            int lane = tid % 32;
            int warp = tid / 32;
            bf16* block_sC = sC + wg_idx*B_WG_M_PADDED*BN;
            uint32_t tid_offset = warp*16+(lane % 8) * B_WG_M_PADDED;
            tid_offset += (lane / 16)*B_WG_M_PADDED*8 + (lane & 8);
            uint32_t base_addr = static_cast<uint32_t>(__cvta_generic_to_shared(block_sC)) + tid_offset * sizeof(bf16);

            bf16 d_bf16[8];
            int4* data_ptr = (int4*)d_bf16;
            float* d_frg = (float*)d;

            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                int yo = m_it * WGMMA_M;
                for (int w = 0; w < WGMMA_N; w+=16, d_frg += 8) {
                    uint32_t addr = base_addr + (w * B_WG_M_PADDED + yo) * sizeof(bf16);
                    
                    for (int k = 0; k < 8; k++) {
                        d_bf16[k] = (bf16)(d_frg[k]);
                    }
                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.trans.x4.shared::cta.b16 [%0], {%1, %2, %3, %4};"
                            :: "r"(addr), "r"(data_ptr[0].x), "r"(data_ptr[0].y), "r"(data_ptr[0].z), "r"(data_ptr[0].w));
                }
            }

            asm volatile("bar.sync %0, 128;\n" :: "r"(wg_idx + 2) : "memory");

            if (tid == 0) {
                store_async(&tensorMapC, block_sC, tile_m*BM +  wg_idx*B_WG_M, tile_n*BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
    }
}

void matmul_v10(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    constexpr int BM = 64*2;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128*3;
    constexpr int NUM_CONSUMERS = 2;
    constexpr int PIPE = 3;
    constexpr int CLUSTER_M = 2;
    constexpr int CLUSTER_N = 1;

    constexpr int NUM_SM = 128;
    d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
    d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);
    d_tma_map_C = create_tensor_map<BN, BM / NUM_CONSUMERS, false, true>(C, N, M);

    auto* kernel = matmul_kernel_v10<BM,BN,BK,NUM_THREADS,NUM_CONSUMERS, PIPE,NUM_SM,CLUSTER_M,CLUSTER_N>;
    size_t smem_size = sizeof(SharedStorage<BM, BN, BK, PIPE>);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<NUM_SM, NUM_THREADS, smem_size>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, d_tma_map_C);
}

}

using M10::matmul_v10;
