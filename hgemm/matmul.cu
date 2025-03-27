#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "cublas_v2.h"

#include "mma.cuh"

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }  // https://stackoverflow.com/a/1804686

template <typename T> __device__ ushort f32_to_b16(float x);
template <> __device__ ushort f32_to_b16<half>(float x) { return __half_as_ushort(__float2half(x)); }
template <> __device__ ushort f32_to_b16<nv_bfloat16>(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

constexpr int WARP_SIZE = 32;

template <typename T>
__global__ void matmul_v1_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int N, int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= M || col >= N) return;

    float total = 0.0f;

#pragma unroll
    for (int i = 0; i < K; i++) {
        total += __bfloat162float(A[row * K + i]) * __bfloat162float(B[K * col + i]); 
    }

    C[row * N + col] = __float2bfloat16(total);
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    dim3 block_size(16, 16);
    dim3 grid_size(cdiv(N, 16), cdiv(M, 16));

    matmul_v1_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, typename T>
__device__ void load_b128(const T *in, int in_row_stride, T *out, int out_row_stride, int tid) {
    // Number of elements to do 128-bit/16-byte load
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


template <
    int MMA_M, int MMA_N, int MMA_K,
    typename T>
__global__ void matmul_v2_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = WARP_SIZE;
    const int tid = threadIdx.x;
    const int warp_id = blockIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps_per_row = cdiv(N, MMA_N);
    const int warp_id_m = warp_id / num_warps_per_row;
    const int warp_id_n = warp_id % num_warps_per_row;
    const int offset_m = warp_id_m * MMA_M;
    const int offset_n = warp_id_n * MMA_N;
    
    if (offset_m >= M || offset_n >= N) return;
    
    A += offset_m * K;
    B += offset_n * K;
    
    __shared__ T A_smem[MMA_M * MMA_K];
    __shared__ T B_smem[MMA_K * MMA_N];
    __shared__ T B_smem[MMA_M * MMA_N];
    
    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(T) / 4 / WARP_SIZE;
    constexpr int num_B_regs = MMA_N * MMA_K * sizeof(T) / 4 / WARP_SIZE;
    
    float acc[num_acc_regs] = {0.0f};
    uint32_t A_reg[num_A_regs];
    uint32_t B_reg[num_B_regs];
    
    #pragma unroll
    for (int block_k = 0; block_k < K; block_k += MMA_K) {
        load_b128<BLOCK_SIZE, MMA_M, MMA_K>(A, K, A_smem, MMA_K, tid);
        if (lane_id < MMA_N * 2)
            load_b128<BLOCK_SIZE, MMA_N, MMA_K>(B, K, B_smem, MMA_K, tid);
        
        __syncthreads();
        
        uint32_t A_tile_addr = cvta_shared(A_smem + (lane_id % 16) * MMA_K + (lane_id / 16) * 8);
        uint32_t B_tile_addr = cvta_shared(B_smem + (lane_id % 8) * MMA_K + (((lane_id / 8) % 2) * 8));
        
        ldmatrix<num_A_regs>(A_reg, A_tile_addr);
        ldmatrix<num_B_regs>(B_reg, B_tile_addr);
        
        mma<MMA_M, MMA_N, MMA_K, T>(A_reg, B_reg, acc);
        
        __syncthreads();
        
        A += MMA_K;
        B += MMA_K;
    }
    
    C += offset_m * N + offset_n;
    const int a0_row = lane_id >> 2;
    const int a0_col = (lane_id % 4) * 2;
    C += a0_row * N + a0_col;
    
    ushort2 tmp;
    tmp.x = f32_to_b16<T>(acc[0]);
    tmp.y = f32_to_b16<T>(acc[1]);
    reinterpret_cast<ushort2 *>(C)[0] = tmp;

    // write a2 and a3
    tmp.x = f32_to_b16<T>(acc[2]);
    tmp.y = f32_to_b16<T>(acc[3]);
    reinterpret_cast<ushort2 *>(C + 8 * N)[0] = tmp;
}

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");
    
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int BLOCK_SIZE = WARP_SIZE;
    const int grid_size = cdiv(M * N, MMA_M * MMA_N);
    
    matmul_v2_kernel<MMA_M, MMA_N, MMA_K><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}


template <
  int BLOCK_M, int BLOCK_N,
  int WARP_M, int WARP_N,
  int MMA_M, int MMA_N, int MMA_K,
  typename T>
__global__ void matmul_v1_kernel(const T *A, const T *B, T *C, int M, int N, int K) {

}

void matmul_v3(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    assert(is_power_of_two(M) && "M must be a power of 2");
    assert(is_power_of_two(N) && "N must be a power of 2");
    assert(is_power_of_two(K) && "K must be a power of 2");

    const int BLOCK_M = 256, BLOCK_N = 128;
    const int WARP_M = 64, WARP_N = 64;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);

    matmul_v1_kernel
    <
        BLOCK_M, BLOCK_N,
        WARP_M, WARP_N,
        MMA_M, MMA_N, MMA_K,><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}