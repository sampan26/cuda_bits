#include <cmath>
#include "utils.cuh"

constexpr int WARP_SIZE = 32;

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, bool TRANSPOSED, typename T>
__device__ __forceinline__ void load_shmem_vectorized(const T *in, int in_row_stride, T *out, int tid) {
  for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * 4) {
    const int idx = offset + tid * 4;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    
    const float2 x = reinterpret_cast<const float2 *>(&in[row * in_row_stride + col])[0];
    if (TRANSPOSED) {
        T tmp[4];
        memcpy(&tmp[0], &x, sizeof(T) * 4);
        out[(col + 0) * HEIGHT + row] = tmp[0];
        out[(col + 1) * HEIGHT + row] = tmp[1];
        out[(col + 2) * HEIGHT + row] = tmp[2];
        out[(col + 3) * HEIGHT + row] = tmp[3];
    } else
      reinterpret_cast<float2 *>(&out[row * WIDTH + col])[0] = x;
  }
}

// vectorized memory access without bounds check
// only memory access is different from v5
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int THREAD_N, typename T>
__global__ void matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M; // 64 / 64 = 1
    constexpr int NUM_MMA_N = WARP_N / MMA_N; // 64 / 16 = 4
    constexpr int THREAD_M = MMA_M * MMA_N / (WARP_SIZE * THREAD_N);

    const int tid = threadIdx.x;
    const int block_id_n = blockIdx.x;
    const int block_id_m = blockIdx.y;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    constexpr int num_warps_per_row = BLOCK_N / WARP_N;
    const int warp_id_m = warp_id / num_warps_per_row;
    const int warp_id_n = warp_id % num_warps_per_row;
    const int warp_tile_offset_m = warp_id_m * WARP_M;
    const int warp_tile_offset_n = warp_id_n * WARP_N;

    constexpr int num_thread_tiles_per_row = MMA_N / THREAD_N;
    const int thread_tile_id_m = lane_id / num_thread_tiles_per_row;
    const int thread_tile_id_n = lane_id % num_thread_tiles_per_row;
    const int thread_tile_offset_m = thread_tile_id_m * THREAD_M;
    const int thread_tile_offset_n = thread_tile_id_n * THREAD_N;

    A += offset_m * K;
    B += offset_n;

    __shared__ T A_shmem[BLOCK_M * BLOCK_K];
    __shared__ T B_shmem[BLOCK_K * BLOCK_N];
    float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][THREAD_N] = {0.0f};
    T A_reg[NUM_MMA_M][THREAD_M] = {};
    T B_reg[NUM_MMA_N][THREAD_N] = {};

    const T *A_thread_tile = reinterpret_cast<const T *>(A_shmem) + (warp_tile_offset_m + thread_tile_offset_m);
    const T *B_thread_tile = reinterpret_cast<const T *>(B_shmem) + (warp_tile_offset_n + thread_tile_offset_n);

    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem_vectorized<BLOCK_SIZE, BLOCK_M, BLOCK_K, true>(A, K, A_shmem, tid);
        load_shmem_vectorized<BLOCK_SIZE, BLOCK_K, BLOCK_N, false>(B, N, B_shmem, tid);
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k++) {
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int tm = 0; tm < THREAD_M; tm++)
                    A_reg[mma_tile_id_m][tm] = 
                        A_thread_tile[(k * BLOCK_M) + (mma_tile_id_m * MMA_M + tm)];    //ToDO: vectorize this, if we don't use ldmatrix later

            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                for (int tn = 0; tn < THREAD_N; tn++) {
                    B_reg[mma_tile_id_n][tn] =
                        B_thread_tile[(k * BLOCK_N) + (mma_tile_id_n * MMA_N + tn)];
                }

            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    for (int tm = 0; tm < THREAD_M; tm++)
                        for (int tn = 0; tn < THREAD_N; tn++)
                            acc[mma_tile_id_m][mma_tile_id_n][tm][tn] += __bfloat162float(A_reg[mma_tile_id_m][tm]) * __bfloat162float(B_reg[mma_tile_id_n][tn]);
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
  }

    const int C_offset_m = offset_m + warp_tile_offset_m + thread_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n + thread_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    for (int mma_m = 0; mma_m < WARP_M; mma_m += MMA_M) {
        for (int mma_n = 0; mma_n < WARP_N; mma_n += MMA_N) {
            for (int tm = 0; tm < THREAD_M; tm++) {
                for (int tn = 0; tn < THREAD_N; tn += 4) {
                    T *C_local = C + (mma_m + tm) * N + (mma_n + tn);
                    int current_mma_tile_id_n = mma_n / MMA_N;

                    // Convert accumulated floats to bfloat16 and pack into uint2
                    ushort v0 = f32_to_b16<T>(acc[0][current_mma_tile_id_n][tm][tn + 0]);
                    ushort v1 = f32_to_b16<T>(acc[0][current_mma_tile_id_n][tm][tn + 1]);
                    ushort v2 = f32_to_b16<T>(acc[0][current_mma_tile_id_n][tm][tn + 2]);
                    ushort v3 = f32_to_b16<T>(acc[0][current_mma_tile_id_n][tm][tn + 3]);

                    uint2 data;
                    data.x = (uint32_t)v0 | ((uint32_t)v1 << 16);
                    data.y = (uint32_t)v2 | ((uint32_t)v3 << 16);
                    *reinterpret_cast<uint2*>(C_local) = data;

                }
            }
        }
    }
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;
    const int WARP_M = 64, WARP_N = 64;
    const int MMA_M = 64, MMA_N = 16;
    const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 8

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 128
    dim3 grid(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M));
    matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N><<<grid, BLOCK_SIZE>>>(A, B, C, M, N, K);
}
