#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <unistd.h>

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// Simple matrix multiplication kernel as placeholder
__global__ void matmul_v1_kernel(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

void matmul_v1(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
    matmul_v1_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

std::default_random_engine generator(69);

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(cublas_handle, 
                                        CUBLAS_OP_N, CUBLAS_OP_N, 
                                        M, N, K, 
                                        &alpha, 
                                        A, CUDA_R_16BF, M,
                                        B, CUDA_R_16BF, K, 
                                        &beta, 
                                        C, CUDA_R_16BF, M,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr) {
    switch (kernel_num) {
        case 0:
            runCublasGemmBF16(M, N, K, A, B, C);
            break;
        case 1:
            matmul_v1(M, N, K, A, B, C);
            break;
    }
}
    
int yo = 0;
void randomize_matrix(bf16 *mat, int N) {
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = __float2bfloat16(distribution(generator));
    }
    ++yo;
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        int r = i / 8192, c = i % 8192;
        int it = c*8192+r;
        diff = std::fabs(__bfloat162float(matRef[i]) - __bfloat162float(matOut[i]));
        if (diff > 0.1) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
            return false;
        }
    }
    return true;
}

int main() {
    cublasCreate(&cublas_handle);

    long max_size = 8192;
    long m = max_size, n = max_size, k = max_size;

    bf16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;  // host matrices
    bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices
    
    int *DB = nullptr; 
    int *dDB = nullptr;  

    A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
    DB = (int *)malloc(sizeof(int) * max_size * 128);
    
    cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));
    
    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));
    
    cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));

    for (int kernel_num : {0, 1}) {
        // Give the GPU some rest to avoid thermal throttling
        sleep(5);
        std::cout << "KERNEL " << kernel_num << std::endl;

        memset(C, 0, sizeof(bf16) * max_size * max_size);
        cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
        memset(DB, ~0, sizeof(int) * max_size * 128);
        cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128, cudaMemcpyHostToDevice));
        
        run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
        run_kernel(kernel_num, m, n, k, dA, dB, dC, dDB); // Executes the kernel, modifies the result matrix
        
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
        
        cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

        if (kernel_num > 0 && !verify_matrix(C_ref, C, m * n)) {
            std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~" << std::endl;
            cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);
            printf("%f\n", __bfloat162float(C_ref[m]));
        } else if (kernel_num > 0) {
            std::cout << "Correctness verification passed!" << std::endl;
        }
    }
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(DB);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cudaFree(dDB);
    cublasDestroy(cublas_handle);
    
    return 0;
}
