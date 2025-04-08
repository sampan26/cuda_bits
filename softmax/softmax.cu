#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__global__ void softmax_kernel_v1(const float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float max_val = -INFINITY;
    for (int col = 0; col < N; col++) {
        max_val = fmaxf(max_val, input[row * N + col]);
    }

    float denom = 0.0f;
    for (int col = 0; col < N; col++) {
        denom += expf(input[row * N + col] - max_val);
    }

    for (int col = 0; col < N; col++) {
        output[row * N + col] = expf(input[row * N + col] - max_val) / denom;
    }
}

void softmax_v1(const float* input, float* output, int M, int N) {
    const int BLOCK_SIZE = 1024;
    int grid_size = cdiv(M, BLOCK_SIZE);
    softmax_kernel_v1<<<grid_size, BLOCK_SIZE>>>(input, output, M, N);
}

__global__ void softmax_kernel_v2(const float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float max_val = -INFINITY;
    float norm = 0.0f;

    for (int col = 0; col < N; col++) {
        float cur_val = input[row * N + col];
        if (cur_val > max_val) {
            norm = norm * expf(max_val - cur_val);
            max_val = cur_val;
        }
        norm += expf(cur_val - max_val);
    }

    for (int col = 0; col < N; col++) {
        output[row * N + col] = expf(input[row * N + col] - max_val) / norm;
    }
}

void softmax_v2(const float* input, float* output, int M, int N) {
    const int BLOCK_SIZE = 1024;
    int grid_size = cdiv(M, BLOCK_SIZE);
    softmax_kernel_v2<<<grid_size, BLOCK_SIZE>>>(input, output, M, N);
}

__global__ void softmax_kernel_v3(const float*  input, float*  out, int M, int N) {
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

 
    input += row * N;
    out += row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smem[0];

    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];



    for (int i = tid; i < N; i += blockDim.x) {
        out[i] = expf(input[i] - row_max) / row_norm;
    }
}

void softmax_v3(const float* input, float* output, int M, int N) {
    const int BLOCK_SIZE = 1024;
    int grid_size = M;
    softmax_kernel_v3<<<grid_size, BLOCK_SIZE>>>(input, output, M, N);
}