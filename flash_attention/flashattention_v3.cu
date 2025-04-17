#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v3(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    float *l, // norm intermediate [B, nh, T]
    float *m, // Row max intermediate [B, nh, T]
    int B, int nh, int T, int d, 
    float scale // Scale factor (usually 1/sqrt(head_dim))
) {
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;

    const int Tc = cdiv(T, Bc); 
    const int Tr = cdiv(T, Br);

    const int QKV_head_offset = batch_id * (nh * T * d) + head_id * (T * d);
    const int lm_head_offset = batch_id * (nh * T) + head_id * T;

    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_smem = &smem[Br * d];
    float* V_smem = &smem[Br * d + Bc * d];
    float* S_smem = &smem[Br * d + 2 * Bc * d];  // Bc * Br
    
    // Process tiles
    for (int j = 0; j < Tc; ++j) {
        const int KV_tile_offset = QKV_head_offset + j * Bc * d;
        
        // Collaborative loading of K and V tiles using all threads
        for (int idx = tid; idx < Bc * d; idx += BLOCK_SIZE) {
            const int col = idx / d;
            const int feature = idx % d;
            K_smem[idx] = K[KV_tile_offset + col * d + feature];
            V_smem[idx] = V[KV_tile_offset + col * d + feature];
        }
        __syncthreads();

        // Process blocks row by row
        for (int i = j; i < Tr; ++i) {
            const int Q_tile_offset = QKV_head_offset + i * Br * d;
            const int lm_tile_offset = lm_head_offset + i * Br;
            
            // Collaborative loading of Q tile
            for (int idx = tid; idx < Br * d; idx += BLOCK_SIZE) {
                const int row = idx / d;
                const int feature = idx % d;
                Q_smem[idx] = Q[Q_tile_offset + row * d + feature];
            }
            __syncthreads();
            
            // Each thread processes one or more rows
            for (int row = warp_id; row < Br; row += warps_per_block) {
                float m_i = m[lm_tile_offset + row];
                float l_i = l[lm_tile_offset + row];
                
                // Compute scores for this row
                float m_ij = -INFINITY;
                
                // Each thread in warp computes partial dot products
                for (int col = 0; col < Bc; ++col) {                    
                    // Apply causal masking
                    
                    float score = 0.0f;
                    // Vectorized dot product within each thread
                    for (int f = lane_id; f < d; f += WARP_SIZE) {
                        score += Q_smem[row * d + f] * K_smem[col * d + f];
                    }
                    
                    // Warp reduction for dot product
                    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                        score += __shfl_down_sync(0xffffffff, score, offset);
                    }
                    
                    // First thread in warp has final score
                    if (lane_id == 0) {
                        score *= scale;
                        S_smem[row * Bc + col] = score;
                        m_ij = max(m_ij, score);
                    }
                }
                
                // Broadcast max score to all threads in warp
                m_ij = __shfl_sync(0xffffffff, m_ij, 0);
                
                // Compute softmax normalization and update l_ij
                float l_ij = 0.0f;
                
                for (int col = lane_id; col < Bc; col += WARP_SIZE) {
                    if (j * Bc + col >= T) continue;
                    
                    // Apply causal masking
                    if (i * Br + row < j * Bc + col) {
                        S_smem[row * Bc + col] = 0.0f;
                    } else {
                        S_smem[row * Bc + col] = __expf(S_smem[row * Bc + col] - m_ij);
                        l_ij += S_smem[row * Bc + col];
                    }
                }
                
                // Warp reduction for l_ij
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    l_ij += __shfl_down_sync(0xffffffff, l_ij, offset);
                }
                
                // First thread in warp has final l_ij
                l_ij = __shfl_sync(0xffffffff, l_ij, 0);
                
                // Compute new m and l values
                float m_new = max(m_i, m_ij);
                float l_new = __expf(m_i - m_new) * l_i + __expf(m_ij - m_new) * l_ij;
                
                float alpha = __expf(m_i - m_new);
                float beta = __expf(m_ij - m_new) / l_new;
                
                // Each thread processes a chunk of the head dimension
                for (int f = lane_id; f < d; f += WARP_SIZE) {
                    float PV_acc = 0.0f;
                    for (int col = 0; col < Bc; ++col) {                        
                        PV_acc += S_smem[row * Bc + col] * V_smem[col * d + f];
                    }
                    O[Q_tile_offset + row * d + f] = alpha * l_i / l_new * O[Q_tile_offset + row * d + f] + beta * PV_acc;
                }
                
                // Update m and l values
                if (lane_id == 0) {
                    m[lm_tile_offset + row] = m_new;
                    l[lm_tile_offset + row] = l_new;
                }
                
            }
            __syncthreads();
        }
        __syncthreads();
    }
}



// Host LAUNCHER function
void flashattn_v3(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; const int Br = 32;
    const int BLOCK_SIZE = 512;

    const float scale = 1.0 / sqrt(d);
    const int sram_size = (2 * Bc * d * sizeof(float)) + (Br * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh);  // batch_size x num_heads


    flashattn_kernel_v3<BLOCK_SIZE, 32, 32><<<grid_dim, BLOCK_SIZE, sram_size>>>(
        Q, K, V, O, l, m, B, nh, T, d, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        // Consider throwing an exception
    }
}