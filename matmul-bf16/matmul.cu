#include "mma.cuh"
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

#define PRINT_IF(cond, ...) if (cond) printf(__VA_ARGS__);

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }  // https://stackoverflow.com/a/1804686
constexpr int WARP_SIZE = 32;

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_b128(const T *in, int in_row_stride, T *out, int out_row_stride, int tid) {
    // number of elements to do 128-bit/16-byte load
    // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
    using load_type = uint4;
    constexpr int num_elems = sizeof(load_type) / sizeof(T);

    for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE * num_elems) {
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        load_type tmp = reinterpret_cast<const load_type *>(&in[row * in_row_stride + col])[0];
        reinterpret_cast<load_type *>(&out[row * out_row_stride + col])[0] = tmp;
    }
}

template <typename T> __device__ ushort f32_to_b16(float x);
template <> __device__ ushort f32_to_b16<half>(float x) { return __half_as_ushort(__float2half(x)); }
template <> __device__ ushort f32_to_b16<nv_bfloat16>(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

template <
  int BLOCK_M, int BLOCK_N, int BLOCK_K,
  int WARP_M, int WARP_N, int WARP_K,
  int MMA_M, int MMA_N, int MMA_K,
  bool PAD_SHMEM_A, bool PAD_SHMEM_B,
  typename T>
__global__ void matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(BLOCK_K % WARP_K == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert(WARP_K % MMA_K == 0);
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = WARP_K / MMA_K;

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

    // A is row-major, B is column-major
    A += offset_m * K;
    B += offset_n * K;

    // we can only pad 8 elements = 16 bytes to ensure 16-byte alignment required by ldmatrix
    constexpr int A_shared_width = BLOCK_K + (PAD_SHMEM_A ? 8 : 0);
    constexpr int B_shared_width = BLOCK_K + (PAD_SHMEM_B ? 8 : 0);
    __shared__ T A_shared[BLOCK_M * A_shared_width];
    __shared__ T B_shared[BLOCK_N * B_shared_width];

    // 32-bit (4-byte) registers
    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //2
    constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //1
    float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f};  // for m16n8k8, each thread holds 4 output float
    uint32_t A_reg[NUM_MMA_M][NUM_MMA_K][num_A_regs];        //              each thread holds 2 input f16x2
    uint32_t B_reg[NUM_MMA_N][NUM_MMA_K][num_B_regs];        //              each thread holds 1 input f16x1

    // first A and B warp-tile along BLOCK_K dim (we will iterate along BLOCK_K with step_size=WARP_K)
    const T *A_warp_tile = reinterpret_cast<const T *>(A_shared) + warp_tile_offset_m * A_shared_width;
    const T *B_warp_tile = reinterpret_cast<const T *>(B_shared) + warp_tile_offset_n * B_shared_width;

    for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
        load_b128<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, A_shared_width, tid);
        load_b128<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, B_shared_width, tid);
        __syncthreads();

        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
            // load data from shared memory to registers using ldmatrix
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

            // convert generic address to .shared state space address expected by inline PTX
            // thread 0 holds address of row 0
            // thread 1 holds address of row 1, and so on
            uint32_t A_tile_addr = cvta_shared(A_warp_tile + lane_id * A_shared_width + warp_k);
            uint32_t B_tile_addr = cvta_shared(B_warp_tile + lane_id * B_shared_width + warp_k);

            // load A to registers
            // ldmatrix can only load 8x8 matrix. for 16x8 tile, we need to use x2
            // works for both m16n8k8 and m16n8k16
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t A_local = A_tile_addr + (mma_tile_id_m * MMA_M * A_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_A_regs>(A_reg[mma_tile_id_m][mma_tile_id_k], A_local);
                }

            // load B to registers
            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t B_local = B_tile_addr + (mma_tile_id_n * MMA_N * B_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_B_regs>(B_reg[mma_tile_id_n][mma_tile_id_k], B_local);
                }

            // call mma
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++)
                        mma<MMA_M, MMA_N, MMA_K, T>(A_reg[mma_tile_id_m][mma_tile_id_k],
                                                    B_reg[mma_tile_id_n][mma_tile_id_k],
                                                acc[mma_tile_id_m][mma_tile_id_n]);
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K;
    }

    const int C_offset_m = offset_m + warp_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    // check output layout here
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
    // m16n8k16 has the same layout
    const int a0_row = lane_id >> 2;
    const int a0_col = (lane_id % 4) * 2;
    C += a0_row * N + a0_col;

    for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
            T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
            float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
            ushort2 tmp;

            // write a0 and a1
            tmp.x = f32_to_b16<T>(acc_frag[0]);
            tmp.y = f32_to_b16<T>(acc_frag[1]);
            reinterpret_cast<ushort2 *>(C_local)[0] = tmp;

            // write a2 and a3
            tmp.x = f32_to_b16<T>(acc_frag[2]);
            tmp.y = f32_to_b16<T>(acc_frag[3]);
            reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
        }
}

void matmul_v1a(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v1_kernel
    <
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        false, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v1b(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v1_kernel<
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        true, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_tile_async(const T* in, int in_row_stride, T* out, int out_row_stride, int tid) {
    using load_type = uint4;
    constexpr int num_elems = sizeof(load_type) / sizeof(T);

#pragma unroll
    for (int idx = tid * num_elems; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE * num_elems) {
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        uint32_t dst_smem_addr = cvta_shared(&out[row * out_row_stride + col]);
        cp_async_cg(dst_smem_addr, &in[row * in_row_stride + col]);
        
    }
}

template <
  int BLOCK_M, int BLOCK_N, int BLOCK_K,
  int WARP_M, int WARP_N, int WARP_K,
  int MMA_M, int MMA_N, int MMA_K,
  bool PAD_SHMEM_A, bool PAD_SHMEM_B,
  typename T>
__global__ void matmul_v2_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(BLOCK_K % WARP_K == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert(WARP_K % MMA_K == 0);
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = WARP_K / MMA_K;

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

    A += offset_m * K;
    B += offset_n * K;

    constexpr int A_shared_width = BLOCK_K + (PAD_SHMEM_A ? 8 : 0);
    constexpr int B_shared_width = BLOCK_K + (PAD_SHMEM_B ? 8 : 0);
    __shared__ T A_shared[BLOCK_M * A_shared_width];
    __shared__ T B_shared[BLOCK_N * B_shared_width];

    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //2
    constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //1
    float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f};  // for m16n8k8, each thread holds 4 output float
    uint32_t A_reg[NUM_MMA_M][NUM_MMA_K][num_A_regs];        //              each thread holds 2 input f16x2
    uint32_t B_reg[NUM_MMA_N][NUM_MMA_K][num_B_regs];        //              each thread holds 1 input f16x1

    const T *A_warp_tile = reinterpret_cast<const T *>(A_shared) + warp_tile_offset_m * A_shared_width;
    const T *B_warp_tile = reinterpret_cast<const T *>(B_shared) + warp_tile_offset_n * B_shared_width;

    for (int block_k = 0; block_k < K; block_k += BLOCK_K) {

        load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, A_shared_width, tid);
        load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(B, K, B_shared, B_shared_width, tid);
        
        cp_async_commit_group();
        cp_async_wait_group();

        __syncthreads();

        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
            uint32_t A_tile_addr = cvta_shared(A_warp_tile + lane_id * A_shared_width + warp_k);
            uint32_t B_tile_addr = cvta_shared(B_warp_tile + lane_id * B_shared_width + warp_k);

            #pragma unroll
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                #pragma unroll
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t A_local = A_tile_addr + (mma_tile_id_m * MMA_M * A_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_A_regs>(A_reg[mma_tile_id_m][mma_tile_id_k], A_local);
                }

            #pragma unroll
            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                #pragma unroll
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t B_local = B_tile_addr + (mma_tile_id_n * MMA_N * B_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_B_regs>(B_reg[mma_tile_id_n][mma_tile_id_k], B_local);
                }

            #pragma unroll
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                #pragma unroll
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    #pragma unroll
                    for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++)
                        mma<MMA_M, MMA_N, MMA_K, T>(A_reg[mma_tile_id_m][mma_tile_id_k],
                                                    B_reg[mma_tile_id_n][mma_tile_id_k],
                                                acc[mma_tile_id_m][mma_tile_id_n]);
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K;
    }

    const int C_offset_m = offset_m + warp_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    const int a0_row = lane_id >> 2;
    const int a0_col = (lane_id % 4) * 2;
    C += a0_row * N + a0_col;

    for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
            T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
            float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
            ushort2 tmp;

            tmp.x = f32_to_b16<T>(acc_frag[0]);
            tmp.y = f32_to_b16<T>(acc_frag[1]);
            reinterpret_cast<ushort2 *>(C_local)[0] = tmp;

            tmp.x = f32_to_b16<T>(acc_frag[2]);
            tmp.y = f32_to_b16<T>(acc_frag[3]);
            reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
        }
}

void matmul_v2a(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v2_kernel
    <
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        false, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v2b(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v2_kernel<
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        true, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

template <
  int BLOCK_M, int BLOCK_N, int BLOCK_K,
  int WARP_M, int WARP_N, int WARP_K,
  int MMA_M, int MMA_N, int MMA_K,
  bool PAD_SHMEM_A, bool PAD_SHMEM_B,
  typename T>
__global__ void matmul_v3_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(BLOCK_K % WARP_K == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert(WARP_K % MMA_K == 0);
    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = WARP_K / MMA_K;

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

    A += offset_m * K;
    B += offset_n * K;

    constexpr int A_shared_width = BLOCK_K + (PAD_SHMEM_A ? 8 : 0);
    constexpr int B_shared_width = BLOCK_K + (PAD_SHMEM_B ? 8 : 0);
    __shared__ T A_shared[2][BLOCK_M * A_shared_width];
    __shared__ T B_shared[2][BLOCK_N * B_shared_width];

    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //2
    constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;    //1
    float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f};  // for m16n8k8, each thread holds 4 output float
    uint32_t A_reg[NUM_MMA_M][NUM_MMA_K][num_A_regs];        //              each thread holds 2 input f16x2
    uint32_t B_reg[NUM_MMA_N][NUM_MMA_K][num_B_regs];        //              each thread holds 1 input f16x1

    int smem_buf_idx = 0; // Buffer index to load into (0 or 1)

    load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared[smem_buf_idx], A_shared_width, tid);
    load_tile_async<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared[smem_buf_idx], B_shared_width, tid);

    cp_async_commit_group();
    cp_async_wait_group();

    __syncthreads();

    for (int block_k = BLOCK_K; block_k < K; block_k += BLOCK_K) {
        
        const int compute_buf_idx = smem_buf_idx;
        const int load_buf_idx = 1 - smem_buf_idx;

        const T* A_load = A + BLOCK_K;
        const T* B_load = B + BLOCK_K;
        
        load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A_load, K, A_shared[load_buf_idx], A_shared_width, tid);
        load_tile_async<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B_load, K, B_shared[load_buf_idx], B_shared_width, tid);

        //Compute using the CURRENT chunk (k_step - 1) from compute_buf_idx
        const T *A_warp_tile = reinterpret_cast<const T *>(A_shared[compute_buf_idx]) + warp_tile_offset_m * A_shared_width;
        const T *B_warp_tile = reinterpret_cast<const T *>(B_shared[compute_buf_idx]) + warp_tile_offset_n * B_shared_width;
        
        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
            uint32_t A_tile_addr = cvta_shared(A_warp_tile + lane_id * A_shared_width + warp_k);
            uint32_t B_tile_addr = cvta_shared(B_warp_tile + lane_id * B_shared_width + warp_k);

            #pragma unroll
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                #pragma unroll
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t A_local = A_tile_addr + (mma_tile_id_m * MMA_M * A_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_A_regs>(A_reg[mma_tile_id_m][mma_tile_id_k], A_local);
                }

            #pragma unroll
            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                #pragma unroll
                for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++) {
                    uint32_t B_local = B_tile_addr + (mma_tile_id_n * MMA_N * B_shared_width + mma_tile_id_k * MMA_K) * sizeof(T);
                    ldmatrix<num_B_regs>(B_reg[mma_tile_id_n][mma_tile_id_k], B_local);
                }

            #pragma unroll
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                #pragma unroll
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    #pragma unroll
                    for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; mma_tile_id_k++)
                        mma<MMA_M, MMA_N, MMA_K, T>(A_reg[mma_tile_id_m][mma_tile_id_k],
                                                    B_reg[mma_tile_id_n][mma_tile_id_k],
                                                acc[mma_tile_id_m][mma_tile_id_n]);
        }

        cp_async_commit_group();
        cp_async_wait_group();

        __syncthreads();

        smem_buf_idx = load_buf_idx;

        A += BLOCK_K;
        B += BLOCK_K;
    }

    const int compute_buf_idx = smem_buf_idx; // The buffer holding the last loaded chunk
    const T *A_warp_tile = reinterpret_cast<const T *>(A_shared[compute_buf_idx]) + warp_tile_offset_m * A_shared_width;
    const T *B_warp_tile = reinterpret_cast<const T *>(B_shared[compute_buf_idx]) + warp_tile_offset_n * B_shared_width;

    #pragma unroll
    for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
        uint32_t A_tile_addr = cvta_shared(A_warp_tile + lane_id * A_shared_width + warp_k);
        uint32_t B_tile_addr = cvta_shared(B_warp_tile + lane_id * B_shared_width + warp_k);

         #pragma unroll
            for (int mma_m = 0; mma_m < NUM_MMA_M; ++mma_m)
                #pragma unroll
                for (int mma_k = 0; mma_k < NUM_MMA_K; ++mma_k) {
                    uint32_t A_local = A_tile_addr + (mma_m * MMA_M * A_shared_width + mma_k * MMA_K) * sizeof(T);
                    ldmatrix<num_A_regs>(A_reg[mma_m][mma_k], A_local);
                }
            #pragma unroll
            for (int mma_n = 0; mma_n < NUM_MMA_N; ++mma_n)
                #pragma unroll
                for (int mma_k = 0; mma_k < NUM_MMA_K; ++mma_k) {
                    uint32_t B_local = B_tile_addr + (mma_n * MMA_N * B_shared_width + mma_k * MMA_K) * sizeof(T);
                    ldmatrix<num_B_regs>(B_reg[mma_n][mma_k], B_local);
                }

         #pragma unroll
            for (int mma_m = 0; mma_m < NUM_MMA_M; ++mma_m)
                #pragma unroll
                for (int mma_n = 0; mma_n < NUM_MMA_N; ++mma_n)
                    #pragma unroll
                    for (int mma_k = 0; mma_k < NUM_MMA_K; ++mma_k)
                        mma<MMA_M, MMA_N, MMA_K, T>(A_reg[mma_m][mma_k], B_reg[mma_n][mma_k], acc[mma_m][mma_n]);
    }
    

    const int C_offset_m = offset_m + warp_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    const int a0_row = lane_id >> 2;
    const int a0_col = (lane_id % 4) * 2;
    C += a0_row * N + a0_col;

    for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
            T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
            float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];
            ushort2 tmp;

            tmp.x = f32_to_b16<T>(acc_frag[0]);
            tmp.y = f32_to_b16<T>(acc_frag[1]);
            reinterpret_cast<ushort2 *>(C_local)[0] = tmp;

            tmp.x = f32_to_b16<T>(acc_frag[2]);
            tmp.y = f32_to_b16<T>(acc_frag[3]);
            reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
        }
}

void matmul_v3a(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v3_kernel
    <
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        false, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v3b(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v3_kernel<
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        true, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}


template <
  int BLOCK_M, int BLOCK_N, int BLOCK_K,
  int WARP_M, int WARP_N, int WARP_K,
  int MMA_M, int MMA_N, int MMA_K,
  bool PAD_SHMEM_A, bool PAD_SHMEM_B,
  typename T>
__global__ void matmul_v4_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(BLOCK_K % WARP_K == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert(WARP_K % MMA_K == 0);    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = WARP_K / MMA_K; 

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

    A += offset_m * K;
    B += offset_n * K;

    constexpr int A_shared_width = BLOCK_K + (PAD_SHMEM_A ? 8 : 0);
    constexpr int B_shared_width = BLOCK_K + (PAD_SHMEM_B ? 8 : 0);
    __shared__ T A_shared[2][BLOCK_M * A_shared_width];
    __shared__ T B_shared[2][BLOCK_N * B_shared_width];

    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
    constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;

    float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {0.0f};
    // ***  Double Buffered Registers (Removed [NUM_MMA_K] dimension) ***
    uint32_t A_reg[2][NUM_MMA_M][num_A_regs];
    uint32_t B_reg[2][NUM_MMA_N][num_B_regs];

    int smem_buf_idx = 0; 
    load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared[smem_buf_idx], A_shared_width, tid);
    load_tile_async<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared[smem_buf_idx], B_shared_width, tid);
    cp_async_commit_group();

    for (int block_k = BLOCK_K; block_k <= K; block_k += BLOCK_K) {
        cp_async_wait_group(); // Wait for A_shared[smem_buf_idx], B_shared[smem_buf_idx]
        __syncthreads();

        const int compute_smem_buf_idx = smem_buf_idx;
        const int load_smem_buf_idx = 1 - smem_buf_idx;

        // Start loading the *next* K chunk asynchronously
        A += BLOCK_K;
        B += BLOCK_K;

        load_tile_async<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared[load_smem_buf_idx], A_shared_width, tid);
        load_tile_async<BLOCK_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared[load_smem_buf_idx], B_shared_width, tid);
        cp_async_commit_group();
        
        const T *A_warp_tile = reinterpret_cast<const T *>(A_shared[compute_smem_buf_idx]) + warp_tile_offset_m * A_shared_width;
        const T *B_warp_tile = reinterpret_cast<const T *>(B_shared[compute_smem_buf_idx]) + warp_tile_offset_n * B_shared_width;

        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += WARP_K) {
            int reg_load_idx = 0; // Buffer to load current k into
            int reg_compute_idx = 1; // Buffer containing previous k's data

            // --- Load initial data (k=0) into buffer 0 --- 
            const int k0_offset_in_warp = warp_k; // + 0 * MMA_K

            #pragma unroll
            for (int mma_m = 0; mma_m < NUM_MMA_M; ++mma_m) {
                uint32_t A_local_addr = cvta_shared(A_warp_tile + (mma_m * MMA_M + lane_id) * A_shared_width + k0_offset_in_warp);
                ldmatrix<num_A_regs>(A_reg[reg_load_idx][mma_m], A_local_addr);
            }
            #pragma unroll
            for (int mma_n = 0; mma_n < NUM_MMA_N; ++mma_n) {
                 uint32_t B_local_addr = cvta_shared(B_warp_tile + (mma_n * MMA_N + lane_id) * B_shared_width + k0_offset_in_warp);
                 ldmatrix<num_B_regs>(B_reg[reg_load_idx][mma_n], B_local_addr);
            }

            // --- Loop over MMA K-steps within WARP_K ---
            #pragma unroll
            for (int mma_tile_id_k = 0; mma_tile_id_k < NUM_MMA_K; ++mma_tile_id_k) {
                // Swap buffers for next iteration (load into alternate, compute from current)
                reg_load_idx = 1 - reg_load_idx;
                reg_compute_idx = 1 - reg_compute_idx;

                const int next_k_offset_in_warp = warp_k + (mma_tile_id_k + 1) * MMA_K; // Used for Load address

                // --- Load Next (k+1) into reg_load_idx buffer ---
                #pragma unroll
                for (int mma_m = 0; mma_m < NUM_MMA_M; ++mma_m) {
                    uint32_t A_local_addr = cvta_shared(A_warp_tile + (mma_m * MMA_M + lane_id) * A_shared_width + next_k_offset_in_warp);
                    ldmatrix<num_A_regs>(A_reg[reg_load_idx][mma_m], A_local_addr);
                }
                #pragma unroll
                for (int mma_n = 0; mma_n < NUM_MMA_N; ++mma_n) {
                    uint32_t B_local_addr = cvta_shared(B_warp_tile + (mma_n * MMA_N + lane_id) * B_shared_width + next_k_offset_in_warp);
                    ldmatrix<num_B_regs>(B_reg[reg_load_idx][mma_n], B_local_addr);
                }

                // --- Compute Current (k) using data from reg_compute_idx buffer ---
                // This buffer holds data loaded in the *previous* iteration (or initial load)
                #pragma unroll
                for (int mma_m = 0; mma_m < NUM_MMA_M; ++mma_m) {
                    #pragma unroll
                    for (int mma_n = 0; mma_n < NUM_MMA_N; ++mma_n) {
                        mma<MMA_M, MMA_N, MMA_K, T>(A_reg[reg_compute_idx][mma_m], B_reg[reg_compute_idx][mma_n],acc[mma_m][mma_n]);
                    }
                }
            } 
        }

        smem_buf_idx = load_smem_buf_idx;
        // Synchronization for shared memory is handled at the top of the loop by wait+syncthreads

    } 

    // --- Store results (Unchanged, but added bounds checks earlier) ---
    const int C_offset_m = offset_m + warp_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    const int store_row_offset = lane_id >> 2;
    const int store_col_offset = (lane_id % 4) * 2;
    C += store_row_offset * N + store_col_offset;

    #pragma unroll
    for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++) {
        #pragma unroll
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++) {
            T *C_local = C + mma_tile_id_m * MMA_M * N + mma_tile_id_n * MMA_N;
            float *acc_frag = acc[mma_tile_id_m][mma_tile_id_n];

            const int write_row = C_offset_m + store_row_offset + mma_tile_id_m * MMA_M;
            const int write_col = C_offset_n + store_col_offset + mma_tile_id_n * MMA_N;

            if (write_row < M && write_col < N) {
                ushort2 tmp;
                tmp.x = f32_to_b16<T>(acc_frag[0]);
                tmp.y = f32_to_b16<T>(acc_frag[1]);
                 reinterpret_cast<ushort2 *>(C_local)[0] = tmp;
            }

            if (write_row + 8 < M && write_col < N) { // Offset for m16n8 layout
                 ushort2 tmp;
                 tmp.x = f32_to_b16<T>(acc_frag[2]);
                 tmp.y = f32_to_b16<T>(acc_frag[3]);
                 reinterpret_cast<ushort2 *>(C_local + 8 * N)[0] = tmp;
            }
        }
    }
}

void matmul_v4a(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v4_kernel
    <
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        false, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v4b(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    const int WARP_M = 64, WARP_N = 64, WARP_K = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 8;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_v4_kernel<
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, WARP_K,
        MMA_M, MMA_N, MMA_K,
        true, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}

