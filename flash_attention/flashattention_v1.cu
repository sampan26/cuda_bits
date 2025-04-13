#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

#include "flashattention_v1.h"

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v1(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    float *l, // norm intermediate [B, nh, T]
    float *m, // Row max intermediate [B, nh, T]
    int B, int num_heads, int seq_len, int head_dim, 
    float softmax_scale // Scale factor (usually 1/sqrt(head_dim))
) {
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    const int Tc = ceil((float) seq_len / Bc); const int Tr = ceil((float) seq_len / Br);
    int QKV_offset = batch_id * (num_heads * seq_len * head_dim) + head_id * (seq_len * head_dim);
    int lm_offset = batch_id * (num_heads * seq_len) + head_id * (seq_len);

    extern __shared__ float sram[];
    int tile_size = Br * head_dim;  // size of Qi, Kj, Vj
    float* s_Qi = sram;
    float* s_Kj = &sram[tile_size];
    float* s_Vj = &sram[tile_size * 2];
    float* s_S = &sram[tile_size * 3];

    // for loop over Tc
    for (int j = 0; j < Tc; j++) {
        int Kj_offset = QKV_offset + j * Bc * head_dim;
        int Vj_offset = QKV_offset + j * Bc * head_dim;
        // each thread loads a row of Kj and Vj
        for (int d = 0; d < head_dim; d++) {
            s_Kj[threadIdx.x * head_dim + d] = K[Kj_offset + threadIdx.x * head_dim + d];
            s_Vj[threadIdx.x * head_dim + d] = V[Vj_offset + threadIdx.x * head_dim + d];
        }
        __syncthreads();
        
        // for loop over Tr
        for (int i = j; i < Tr; i++) {
            if (i * Br + threadIdx.x >= seq_len)
                break;
            int Qi_offset = QKV_offset + i * Br * head_dim;
            // each thread loads a row of Qi
            for (int d = 0; d < head_dim; d ++) {
                s_Qi[threadIdx.x * head_dim + d] = Q[Qi_offset + threadIdx.x * head_dim + d];
            }

            int li_offset = lm_offset + i * Br;
            int mi_offset = lm_offset + i * Br;
            // each thread loads a value of l and m
            float li = l[li_offset + threadIdx.x];
            float mi = m[mi_offset + threadIdx.x];

            // Sij = Qi @ Kj^T
            // cur_m_ij = row_max(Sij)        
            // mi_new = max(mi, cur_m_ij)
            float cur_m_ij = -INFINITY;    
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += s_Qi[threadIdx.x * head_dim + d] * s_Kj[k * head_dim + d];
                }
                sum *= softmax_scale;
                if (i * Br + threadIdx.x < j * Bc + k) { // threadIdx.x < k is not enough
                    sum = -INFINITY;
                }
                s_S[threadIdx.x * Bc + k] = sum;
                cur_m_ij = cur_m_ij > sum ? cur_m_ij : sum;
            }
            float mi_new = mi > cur_m_ij ? mi : cur_m_ij;

            // P_ij = __expf(Sij - cur_m_ij) , s_S <-- P_ij
            // cur_l_ij = row_sum(P_ij)
            // li_new = __expf(mi-mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij
            float cur_l_ij = 0.0f;
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                if (i * Br + threadIdx.x < j * Bc + k) {
                    s_S[threadIdx.x * Bc + k] = 0.0f;
                } else {
                    s_S[threadIdx.x * Bc + k] = __expf(s_S[threadIdx.x * Bc + k] - cur_m_ij);
                }
                cur_l_ij += s_S[threadIdx.x * Bc + k];                
            }
            float li_new = __expf(mi - mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij;
            
            // alpha = __expf(mi-mi_new)
            // beta = __expf(cur_m_ij - mi_new)
            // Oi = 1 / li_new * (li * alpha * Oi + beta * Pij * Vj)
            float alpha = __expf(mi - mi_new);
            float beta = __expf(cur_m_ij - mi_new);
            
            int Oi_offset = QKV_offset + i * Br * head_dim;

            for (int d = 0; d < head_dim; d++) {
                // for loop over Bc
                float pv = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    if (j * Bc + k >= seq_len)
                    {
                        break;
                    }
                    pv += s_S[threadIdx.x * Bc + k] * s_Vj[k * head_dim + d];
                }
            
                O[Oi_offset + threadIdx.x * head_dim + d] = (1.0f / li_new) \
                * (li * alpha * O[Oi_offset + threadIdx.x * head_dim + d] \
                + beta * pv);
            }

            // li = li_new
            l[li_offset + threadIdx.x] = li_new;
            // mi = mi_new
            m[mi_offset + threadIdx.x] = mi_new;
        }
        __syncthreads(); 
        // there might be cases where one thread finishes the inner loop 
        // and increases i by 1, which means the next Kj and Vj will be loaded
        // while other threads are still in the inner loop, and use the wrong Kj and Vj.
    }
} 


// Add this function to check shared memory limits
bool check_shared_memory_requirements(size_t required_shared_mem) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // Get shared memory per block
    size_t shared_mem_per_block = deviceProp.sharedMemPerBlock;
    
    if (required_shared_mem > shared_mem_per_block) {
        printf("ERROR: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n", 
               required_shared_mem, shared_mem_per_block);
        return false;
    }
    
    return true;
}

// Host LAUNCHER function
void flashattn_v1(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, // Pass pointers to buffers allocated in C++/Python
                  int B, int nh, int T, int head_dim) {
    const int Bc =16; const int Br = 16;
    const int Tc = ceil((float) T / Bc); const int Tr = ceil((float) T / Br);
    const float softmax_scale = 1.0 / sqrt(head_dim);
    const int sram_size = (3 * Bc * head_dim * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);

    flashattn_kernel_v1<Bc, Br, Bc><<<grid_dim, block_dim, sram_size>>>(
        Q, K, V, O, l, m, B, nh, T, head_dim, softmax_scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        // Consider throwing an exception
    }
}