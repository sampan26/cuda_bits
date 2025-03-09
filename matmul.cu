#include <cmath>
#include <stdio.h>
#include <assert.h>

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr int WARP_SIZE = 32;


// naive kernel. 1 row dot 1 column
// to compute 1 output element:
// - load 1 row from A (1xK) and 1 column from B (Kx1)
// - K multiplications and (K-1) additions
// => arithmetic intensity ~ O(1)
__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    // broadcast read from A since each warp reads the same A value
    // coalesce read from B since each warp reads consecutive B values
    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total += A[row * K + k] * B[k * N + col]; 
    }
    // coalesce write to C since each warp writes consecutive C values
    C[row * N + col] = total;
}

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    int block_size_total;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_total, matmul_v1_kernel, 0, 0);

    // NOTE: blockDim.x must be a multiple of 32 (warpSize) to ensure coalesce memory access
    int BLOCK_X = WARP_SIZE;
    int BLOCK_Y = block_size_total / WARP_SIZE;
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size(cdiv(N, BLOCK_X), cdiv(M, BLOCK_Y));
    matmul_v1_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

// thread-block tiling: read 2D block into shared memory for caching
// to compute BLOCK_SIZExBLOCK_SIZE output elements:
// - load BLOCK_SIZExBLOCK_SIZE of A and B from global memory
// - amount of compute is unchanged
// => arithmetic intensity ~ O(BLOCK_SIZE)
template <int BLOCK_SIZE>
__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int offset_m = blockIdx.y * BLOCK_SIZE;
    const int offset_n = blockIdx.x * BLOCK_SIZE;

    A += offset_m * K;
    B += offset_n;
    C += offset_m * K + offset_n;

    __shared__ float A_shmem[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shmem[BLOCK_SIZE][BLOCK_SIZE];
    float acc = 0.0f;
    
    for (int offset_k = 0; offset_k < K; offset_k+=BLOCK_SIZE) {
        // load data from global memory (DDR/HBM) to shared memory (SRAM)
        // notice now each thread only loads 2 x n_blocks elements
        // coalesced memory read for both A and B
       A_shmem[tid_y][tid_x] = tid_y < (M - offset_m) && tid_x < (K - offset_k) ? A[tid_y * K + tid_x] : 0.0f;
       B_shmem[tid_y][tid_x] = tid_y < (K - offset_k) && tid_x < (N - offset_n) ? B[tid_y * N + tid_x] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += A_shmem[tid_y][k] * B_shmem[k][tid_x];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }

    if (tid_y < (M - offset_m) && tid_x < (N - offset_n))
        C[tid_y * N + tid_x] = acc;
}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));
    matmul_v2_kernel<BLOCK_SIZE><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

template <int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                           float out[HEIGHT][WIDTH], int tid) {
    for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        out[row][col] = row < in_max_row && col < in_max_col ? in[row * in_row_stride + col] : 0.0f;
    }
}

template <int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul_v3_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id % num_blocks_per_row;

    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    A += offset_m * K;
    B += offset_n;
    C += offset_m * N + offset_n;

    __shared__ float A_shmem[BLOCK_M][BLOCK_K];
    __shared__ float B_shmem[BLOCK_K][BLOCK_N];

    static_assert((BLOCK_M * BLOCK_N) % BLOCK_SIZE == 0);
    float acc[BLOCK_M * BLOCK_N / BLOCK_SIZE] = {0.0f};

    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
        load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);

        __syncthreads();

        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx+=BLOCK_SIZE) {
            const int local_idx = idx / BLOCK_SIZE;
            const int row = idx / BLOCK_N;
            const int col = idx % BLOCK_N;

            for (int k = 0; k < BLOCK_K; k++) {
                acc[local_idx] += A_shmem[row][k] * B_shmem[k][col];
            }
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }
    
    for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx+=BLOCK_SIZE) {
        const int local_idx = idx / BLOCK_SIZE;
        const int row = idx / BLOCK_N;
        const int col = idx % BLOCK_N;

        if (row < (M - offset_m) && col < (N - offset_n)) {
            C[row * N + col] = acc[local_idx];
        }
    }
}

void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int BLOCK_SIZE = 256;
    const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
    matmul_v3_kernel<BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

