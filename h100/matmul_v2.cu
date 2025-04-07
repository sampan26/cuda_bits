#include <cuda/barrier> 
#include "utils.cuh"
#include "mma.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <int BLOCK_HEIGHT, int BLOCK_WIDTH, typename T>
__host__ static inline CUtensorMap create_tensor_map(T *in, int HEIGHT, int WIDTH) {
    CUtensorMap tma_map;
    constexpr uint32_t rank = 2;
    uint64_t global_shape[rank] = {static_cast<uint64_t>(WIDTH), static_cast<uint64_t>(HEIGHT)};
    uint64_t global_stride[rank] = {sizeof(T), sizeof(T) * WIDTH};
    uint32_t box_shape[rank] = {BLOCK_WIDTH, BLOCK_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void*)in,
        global_shape,
        global_stride + 1,
        box_shape,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled failed with error code: %d\n", res);
    }
    return tma_map;
}

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K, 
    int WGMMA_M, int WGMMA_N, int WGMMA_K, 
    int BLOCK_SIZE, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE) matmul_v2_kernel(const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, T *C, int M, int N, int K) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id / num_blocks_per_row;
    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    // The destination shared memory buffer of a bulk tensor operation should be 
    // 128 bits aligned
    __shared__ alignas(128) T A_shmem[BLOCK_M * BLOCK_K];
    __shared__ alignas(128) T B_shmem[BLOCK_N * BLOCK_K];
    float d[WGMMA_N/16][8] = {0.0f};

    //Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier A_bar;
    __shared__ barrier B_bar;

    if (tid == 0) {
        // Initalize barriers, with block size threads in block participate
        init(&A_bar, BLOCK_SIZE);
        init(&B_bar, BLOCK_SIZE);
        // Make initalized barrier visible in async proxy
        cde::fence_proxy_async_shared_cta();
    }

    __syncthreads();
    barrier::arrival_token A_token, B_token;


    for (int bk = 0; bk < K/BLOCK_K; bk++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&A_shmem[0], &A_map, bk*BLOCK_K, offset_m, A_bar);
            A_token = cuda::device::barrier_arrive_tx(A_bar, 1, sizeof(A_shmem));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&B_shmem[0], &B_map, bk*BLOCK_K, offset_n, B_bar);
            A_token = cuda::device::barrier_arrive_tx(B_bar, 1, sizeof(A_shmem));
        }
        else {
            A_token = A_bar.arrive();
            B_token = B_bar.arrive();
        }
        A_bar.wait(std::move(A_token));
        B_bar.wait(std::move(B_token));
        __syncthreads();

        warpgroup_arrive();
        // Create a wgmma-group and commit all the prior outstanding wgmma.mma_async operations into the group
        // Create descriptors for the shared memory arrays
        uint64_t desc_a = make_smem_desc(&A_shmem[0]);
        uint64_t desc_b = make_smem_desc(&B_shmem[0]);

        // Call wgmma with the descriptors
        SM90_64x64x16_F32BF16BF16::wgmma(desc_a, desc_b, d); // K = 0..15
        SM90_64x64x16_F32BF16BF16::wgmma(desc_a, desc_b, d); // K = 16..31
        SM90_64x64x16_F32BF16BF16::wgmma(desc_a, desc_b, d); // K = 32..47
        SM90_64x64x16_F32BF16BF16::wgmma(desc_a, desc_b, d); // K = 48..63
        // Wait for the completion of the required wgmma-group
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    //uint32_t row = warp_id*16 + lane_id/4;
    T *block_C = C + offset_n*M + offset_m;

    for (int bm = 0; bm < BLOCK_M/WGMMA_M; bm++) {
        for (int bn = 0; bn < BLOCK_N/WGMMA_N; bn++) {
            for (int w = 0; w < WGMMA_N/16; w++) {
                //int col = 16*w + 2*(lane_id % 4); // Use lane_id directly, tid includes warp_id
                #define IDX(i, j) ((static_cast<int64_t>(j) + bn*WGMMA_N)*M + ((i) + bm*WGMMA_M + offset_m)) // Careful with indexing, incorporate offset_m and offset_n here or use block_C base. Let's adjust IDX and use C directly.


                int base_row = offset_m + warp_id * 16 + (lane_id / 4); // Calculate base row index in C
                int base_col = offset_n + bn * WGMMA_N + w * 16 + 2 * (lane_id % 4); // Calculate base col index in C

                // Use the f32_to_b16 conversion function
                // The exact mapping d[w][idx] -> C[row][col] depends on the SM90 wgmma documentation for m64n64k16.
                // Assuming the provided store pattern is correct for the d layout:
                C[(base_row + 0) + (base_col + 0) * M] = f32_to_native<nv_bfloat16>(d[w][0]);
                C[(base_row + 0) + (base_col + 1) * M] = f32_to_native<nv_bfloat16>(d[w][1]);
                C[(base_row + 8) + (base_col + 0) * M] = f32_to_native<nv_bfloat16>(d[w][2]);
                C[(base_row + 8) + (base_col + 1) * M] = f32_to_native<nv_bfloat16>(d[w][3]);

                C[(base_row + 0) + (base_col + 8) * M] = f32_to_native<nv_bfloat16>(d[w][4]);
                C[(base_row + 0) + (base_col + 9) * M] = f32_to_native<nv_bfloat16>(d[w][5]);
                C[(base_row + 8) + (base_col + 8) * M] = f32_to_native<nv_bfloat16>(d[w][6]);
                C[(base_row + 8) + (base_col + 9) * M] = f32_to_native<nv_bfloat16>(d[w][7]);

                #undef IDX
            }
        }
    }
}

void matmul_v2(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K) {
    const int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64;
    const int WGMMA_M = 64, WGMMA_N = 64, WGMMA_K = 64;
    
    CUtensorMap tma_map_A = create_tensor_map<BLOCK_M, BLOCK_K>(A, M, K);
    CUtensorMap tma_map_B = create_tensor_map<BLOCK_N, BLOCK_K>(B, N, K);
    const int BLOCK_SIZE = 128;
    dim3 grid(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N));

    matmul_v2_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WGMMA_M, WGMMA_N, WGMMA_K, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(tma_map_A, tma_map_B, C, M, N, K);
}
