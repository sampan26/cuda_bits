#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }  // https://stackoverflow.com/a/1804686
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

// we want to load a (HEIGHT, WIDTH) tile from global to shared memory.
// just load a BLOCK_SIZE of data until the whole tile is loaded.
template <int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                           float out[HEIGHT][WIDTH], int tid) {
    for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        out[row][col] = row < in_max_row && col < in_max_col ? in[row * in_row_stride + col] : 0.0f;
    }
}

// thread coarsening
template <int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul_v3_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    // assign block linearly
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

    // each thread is responsible for (BLOCK_M * BLOCK_N / BLOCK_SIZE) output elements
    static_assert((BLOCK_M * BLOCK_N) % BLOCK_SIZE == 0);
    float acc[BLOCK_M * BLOCK_N / BLOCK_SIZE] = {0.0f};

    // we move block by block along K dim
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        // decouple global memory read, so we don't need to care about assigning which thread to read which element.
        // load (BLOCK_M, BLOCK_K) from A and (BLOCK_K, BLOCK_N) from B
        load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
        load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);

        __syncthreads();

         // do a mini matmul of (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        // simply assign a BLOCK_SIZE of threads to a BLOCK_SIZE of elements in output tile
        // there is shared memory bank conflict
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
    
    // write (BLOCK_M, BLOCK_N) to C
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

// 2D thread-tiling with register cache
// each thread will calculate (THREAD_M, THREAD_N) thread-tile of output (BLOCK_M, BLOCK_N) block-tile
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void matmul_v4_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    static_assert(BLOCK_M % THREAD_M == 0);
    static_assert(BLOCK_N % THREAD_N == 0);
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id % num_blocks_per_row;
    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    const int num_threads_per_row = BLOCK_N / THREAD_N;
    const int tile_thread_id_m = tid / num_threads_per_row;
    const int tile_thread_id_n = tid % num_threads_per_row;
    const int tile_thread_offset_m = tile_thread_id_m * THREAD_M;
    const int tile_thread_offset_n = tile_thread_id_n * THREAD_N;

    __shared__ float A_shmem[BLOCK_M][BLOCK_K];
    __shared__ float B_shmem[BLOCK_K][BLOCK_N];
    float acc[THREAD_M][THREAD_N] = {0.0f};

    A += offset_m * K;
    B += offset_n;
    
    const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + tile_thread_offset_m * BLOCK_K;
    const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + tile_thread_offset_n;

    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
        load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
        __syncthreads();

        // mini-matmul with thread-tile. same structure as block-tile.
        for (int k = 0; k < BLOCK_K; k++) {
            float A_reg[THREAD_M];
            float B_reg[THREAD_N]; // register cache

            // load data from shared memory to registers
            // there is shared memory bank conflict
            for (int m = 0; m < THREAD_M; m++)
                A_reg[m] = A_thread_tile[m * BLOCK_K + k];

            for (int n = 0; n < THREAD_N; n++)
                B_reg[n] = B_thread_tile[k * BLOCK_N + n];

            // for each (THREAD_M, THEAD_N) output, we only need to read
                // (THREAD_M, BLOCK_K) of A and (BLOCK_K, THREAD_N) for B from shared memory.
            for (int m = 0; m < THREAD_M; m++ ) {
                for (int n = 0; n < THREAD_N; n++) {
                    acc[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    C += (offset_m + tile_thread_offset_m) * N + (offset_n + tile_thread_offset_n);

    // uncoalesced memory write
    // fixing it doesn't seem to make the kernel faster.
    // vectorized write is slower.
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            if (m < (M - (offset_m + tile_thread_offset_m)) && n < (N - (offset_n + tile_thread_offset_n)))
                C[m * N + n] = acc[m][n];
        }
    }
}

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int THREAD_M = 8, THREAD_N = 8;
    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);  // 256
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v4_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

// 2D warp-tiling with register cache
// we partition output tile (BLOCK_M, BLOCK_N) into tiles of (WARP_M, WARP_N)
// we use the exact number of warps in a threadblock as the number of warp tiles
// for each output warp tile (WARP_M, WARP_N), we further divide it into MMA tiles (MMA_M, MMA_N)
// for each MMA tile (MMA_M, MMA_N), we divide it exactly to 32 thread tiles (THREAD_M, THREAD_N),
// since there are 32 threads in a warp.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int THREAD_N>
__global__ void matmul_v5_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert((MMA_M * MMA_N / THREAD_N) % WARP_SIZE == 0);

    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_SIZE * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int THREAD_M = MMA_M * MMA_N / (WARP_SIZE * THREAD_N);

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id % num_blocks_per_row;
    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    constexpr int num_warps_per_row = BLOCK_N / WARP_N;
    const int warp_id_m = warp_id / num_warps_per_row;
    const int warp_id_n = warp_id % num_warps_per_row;
    const int warp_tile_offset_m = warp_id_m * WARP_M;
    const int warp_tile_offset_n = warp_id_n * WARP_N;

    constexpr int num_threads_per_row = MMA_N / THREAD_N;
    const int tile_thread_id_m = lane_id / num_threads_per_row;
    const int tile_thread_id_n = lane_id % num_threads_per_row;
    const int tile_thread_offset_m = tile_thread_id_m * THREAD_M;
    const int tile_thread_offset_n = tile_thread_id_n * THREAD_N;

    __shared__ float A_shmem[BLOCK_M][BLOCK_K];
    __shared__ float B_shmem[BLOCK_K][BLOCK_N];
    float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][THREAD_N] = {0.0f};

    A += offset_m * K;
    B += offset_n;
    

    const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + (tile_thread_offset_m + warp_tile_offset_m) * BLOCK_K;
    const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + (tile_thread_offset_n + warp_tile_offset_n);

    // points to the corresponding thread tile of the current thread in the first MMA tile
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
        load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
        __syncthreads();
        
        // implicit WARP_K = MMA_K = THREAD_K = 1
        for (int k = 0; k < BLOCK_K; k++) {
            float A_reg[NUM_MMA_M][THREAD_M]; // 2Dregister cache
            float B_reg[NUM_MMA_N][THREAD_N]; 

            // notice we have extra loops to iterate over MMA tiles
            for (int mm_tile_id_m = 0; mm_tile_id_m < NUM_MMA_M; mm_tile_id_m++)
                for (int m = 0; m < THREAD_M; m++)
                    A_reg[mm_tile_id_m][m] = A_thread_tile[(mm_tile_id_m * MMA_M + m) * BLOCK_K + k];

            for (int mm_tile_id_n = 0; mm_tile_id_n < NUM_MMA_N; mm_tile_id_n++)
                for (int n = 0; n < THREAD_N; n++)
                    B_reg[mm_tile_id_n][n] = B_thread_tile[k * BLOCK_N + (mm_tile_id_n * MMA_N + n)];

           
             for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    for (int m = 0; m < THREAD_M; m++)
                        for (int n = 0; n < THREAD_N; n++)
                            acc[mma_tile_id_m][mma_tile_id_n][m][n] += A_reg[mma_tile_id_m][m] * B_reg[mma_tile_id_n][n];
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    // points to the corresponding thread tile in the first MMA tile
    int C_offset_m = offset_m + tile_thread_offset_m + warp_tile_offset_m;
    int C_offset_n = offset_n + tile_thread_offset_n + warp_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    // uncoalesced memory write
    // fixing it doesn't seem to make the kernel faster.
    // vectorized write is slower.
    for (int mma_m = 0; mma_m < WARP_M; mma_m += MMA_M)
        for (int mma_n = 0; mma_n < WARP_N; mma_n += MMA_N)
            for (int tm = 0; tm < THREAD_M; tm++)
                for (int tn = 0; tn < THREAD_N; tn++)
                    if ((C_offset_m + mma_m + tm < M) && (C_offset_n + mma_n + tn < N))
                        C[(mma_m + tm) * N + (mma_n + tn)] = acc[mma_m / MMA_M][mma_n / MMA_N][tm][tn];
}

void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) {
    // this config will result in identical kernel as v4

    const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
    const int WARP_M = 32, WARP_N = 32;
    const int MMA_M = 16, MMA_N = 32;
    const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 4

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 256
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v5_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

// no bounds check
// NOTE: loop can be unrolled now since everything is known at compile time
// this gives a small boost. for load_shmem() (non-vectorized), this is actually slower.
template <int BLOCK_SIZE, int HEIGHT, int WIDTH, bool TRANSPOSED>
__device__ void load_shmem_vectorized(const float *in, int in_row_stride, float *out, int tid) {
  for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * 4) {
    const int idx = offset + tid * 4;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    float4 tmp = reinterpret_cast<const float4 *>(&in[row * in_row_stride + col])[0];

    if (TRANSPOSED) {
      out[(col + 0) * HEIGHT + row] = tmp.x;
      out[(col + 1) * HEIGHT + row] = tmp.y;
      out[(col + 2) * HEIGHT + row] = tmp.z;
      out[(col + 3) * HEIGHT + row] = tmp.w;
    } else
      reinterpret_cast<float4 *>(&out[row * WIDTH + col])[0] = tmp;
  }
}

// vectorized memory access without bounds check
// only memory access is different from v5
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int THREAD_N, bool TRANSPOSE_A_shmem>
__global__ void matmul_v6_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert((MMA_M * MMA_N / THREAD_N) % WARP_SIZE == 0);
    static_assert(THREAD_N % 4 == 0);  // so we can use vectorized access
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int THREAD_M = MMA_M * MMA_N / (WARP_SIZE * THREAD_N);

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id % num_blocks_per_row;
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

    __shared__ float A_shmem[BLOCK_M * BLOCK_K];
    __shared__ float B_shmem[BLOCK_K * BLOCK_N];
    float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][THREAD_N] = {0.0f};

    const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + (warp_tile_offset_m + thread_tile_offset_m) * (TRANSPOSE_A_shmem ? 1 : BLOCK_K);
    const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + (warp_tile_offset_n + thread_tile_offset_n);

    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem_vectorized<BLOCK_SIZE, BLOCK_M, BLOCK_K, TRANSPOSE_A_shmem>(A, K, A_shmem, tid);
        load_shmem_vectorized<BLOCK_SIZE, BLOCK_K, BLOCK_N, false>(B, N, B_shmem, tid);
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k++) {
            float A_reg[NUM_MMA_M][THREAD_M];
            float B_reg[NUM_MMA_N][THREAD_N];

            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                if (TRANSPOSE_A_shmem) {
                    static_assert(THREAD_M % 4 == 0);
                    for (int tm = 0; tm < THREAD_M; tm += 4) {
                        float4 tmp = reinterpret_cast<const float4 *>(&A_thread_tile[k * BLOCK_M + (mma_tile_id_m * MMA_M + tm)])[0];
                        reinterpret_cast<float4 *>(&A_reg[mma_tile_id_m][tm])[0] = tmp;
                    }
                }
                else {
                    for (int tm = 0; tm < THREAD_M; tm++)
                        A_reg[mma_tile_id_m][tm] = A_thread_tile[(mma_tile_id_m * MMA_M + tm) * BLOCK_K + k];
                }

            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                for (int tn = 0; tn < THREAD_N; tn += 4) {
                    float4 tmp = reinterpret_cast<const float4 *>(&B_thread_tile[k * BLOCK_N + (mma_tile_id_n * MMA_N + tn)])[0];
                    reinterpret_cast<float4 *>(&B_reg[mma_tile_id_n][tn])[0] = tmp;
                }

            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    for (int tm = 0; tm < THREAD_M; tm++)
                        for (int tn = 0; tn < THREAD_N; tn++)
                            acc[mma_tile_id_m][mma_tile_id_n][tm][tn] += A_reg[mma_tile_id_m][tm] * B_reg[mma_tile_id_n][tn];
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
  }

    const int C_offset_m = offset_m + warp_tile_offset_m + thread_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n + thread_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    for (int mma_m = 0; mma_m < WARP_M; mma_m += MMA_M)
        for (int mma_n = 0; mma_n < WARP_N; mma_n += MMA_N)
            for (int tm = 0; tm < THREAD_M; tm++)
                for (int tn = 0; tn < THREAD_N; tn += 4) {
                    const float4 tmp = reinterpret_cast<const float4 *>(&acc[mma_m / MMA_M][mma_n / MMA_N][tm][tn])[0];
                    reinterpret_cast<float4 *>(&C[(mma_m + tm) * N + (mma_n + tn)])[0] = tmp;
                }
}

void matmul_v6a(const float *A, const float *B, float *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;
    const int WARP_M = 64, WARP_N = 64;
    const int MMA_M = 16, MMA_N = 32;
    const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 4

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 128
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v6_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v6b(const float *A, const float *B, float *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 64, BLOCK_N = 128, BLOCK_K = 8;
    const int WARP_M = 64, WARP_N = 64;
    const int MMA_M = 32, MMA_N = 32;
    const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 8

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 64
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v6_kernel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N, true><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}