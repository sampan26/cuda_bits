#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v2(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    float *l, // norm intermediate [B, nh, T]
    float *m, // Row max intermediate [B, nh, T]
    int B, int nh, int T, 
    float scale // Scale factor (usually 1/sqrt(head_dim))
) { 
    int tid = threadIdx.x;
    int batch_head_id = blockIdx.x;

    int s_row = tid / Bc;
    int s_col = tid % Bc;
    
    // Register variables for this thread's row statistics
    float my_m_i = -INFINITY;  // Each thread maintains its own m_i
    float my_l_i = 0.0f;       // Each thread maintains its own l_i

    const int Tc = cdiv(T, Bc); const int Tr = cdiv(T, Br);

    const int QKV_head_offset = batch_head_id * T * d;
    const int lm_head_offset = batch_head_id * T;

    int num_tiles_kv = cdiv(d, Br);
    int num_tiles_q = cdiv(d, Bc);

    __shared__ float Q_smem[Br * d];
    __shared__ float K_smem[Bc * d];
    __shared__ float V_smem[Bc * d];
    __shared__ float S_ij_smem[Br * Bc];
    __shared__ float O_smem[Br * d];
    __shared__ float m_new[Br];
    __shared__ float l_new[Br];
    __shared__ float m_ij_smem[Br];

    for (int j = 0; j < Tc; ++j) {
        // Load K and V tiles into shared memory
        for (int x = 0; x < num_tiles_kv; ++x) {
            int idx = x * (Bc * Br) + tid;
            if (idx < Bc * d) {
                int kv_row = idx / d;
                int kv_col = idx % d;
                if (j * Bc + kv_row < T) {
                    K_smem[kv_row * d + kv_col] = K[QKV_head_offset + (j * Bc + kv_row) * d + kv_col];
                    V_smem[kv_row * d + kv_col] = V[QKV_head_offset + (j * Bc + kv_row) * d + kv_col];
                }
            }
        }
        
        __syncthreads();

        // For causal attention, we only process blocks where i >= j
        for (int i = j; i < Tr; ++i) {
            // Load Q and O tiles into shared memory
            for (int x = 0; x < num_tiles_q; ++x) {
                int idx = x * (Br * Bc) + tid;
                if (idx < Br * d) {
                    int qo_row = idx / d;
                    int qo_col = idx % d; 
                    if (i * Br + qo_row < T) {
                        Q_smem[qo_row * d + qo_col] = Q[QKV_head_offset + (i * Br + qo_row) * d + qo_col];
                        O_smem[qo_row * d + qo_col] = O[QKV_head_offset + (i * Br + qo_row) * d + qo_col];
                    }
                }
            }
            __syncthreads();
            
            // Load m_i and l_i from global memory into registers for this thread's row
            int global_row = i * Br + s_row;
            if (global_row < T) {
                my_m_i = m[lm_head_offset + global_row];
                my_l_i = l[lm_head_offset + global_row];
            }

            // Compute S_ij = Q_i * K_j^T
            float acc = 0.0f;
            for (int k = 0; k < d; k++) {
                acc += Q_smem[s_row * d + k] * K_smem[s_col * d + k];
            }
            acc *= scale;

            // Apply causal masking
            int query_pos = i * Br + s_row;
            int key_pos = j * Bc + s_col;
            if (key_pos > query_pos) {
                acc = -INFINITY;
            }
            
            S_ij_smem[s_row * Bc + s_col] = acc;
            __syncthreads();

            // Compute row max and softmax - distributed across threads
            // Each thread handles one element in the row
            float my_val = S_ij_smem[s_row * Bc + s_col];
            
            // Find row maximum using warp shuffle reduction
            float m_ij = my_val;
            for (int offset = Bc/2; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xFFFFFFFF, m_ij, offset);
                m_ij = fmaxf(m_ij, other_val);
            }
            // Broadcast the max back to all threads in the row
            m_ij = __shfl_sync(0xFFFFFFFF, m_ij, 0);
            
            // Compute exponential and sum
            float exp_val = __expf(my_val - m_ij);
            S_ij_smem[s_row * Bc + s_col] = exp_val;
            
            // Sum reduction using warp shuffle
            float l_ij = exp_val;
            for (int offset = Bc/2; offset > 0; offset /= 2) {
                l_ij += __shfl_down_sync(0xFFFFFFFF, l_ij, offset);
            }
            // Broadcast the sum back to all threads
            l_ij = __shfl_sync(0xFFFFFFFF, l_ij, 0);
            
            // Only one thread per row updates the shared arrays
            if (s_col == 0) {
                m_ij_smem[s_row] = m_ij;
                m_new[s_row] = fmaxf(m_ij, my_m_i);
                l_new[s_row] = __expf(my_m_i - m_new[s_row]) * my_l_i + __expf(m_ij - m_new[s_row]) * l_ij;
            }
            __syncthreads();

            float alpha = __expf(my_m_i - m_new[s_row]);  // Use register value
            float beta = __expf(m_ij_smem[s_row] - m_new[s_row]);
            
            // Compute S_ij * V_j for this thread's column
            for (int col = s_col; col < d; col += Bc) {
                float PV_acc = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    PV_acc += S_ij_smem[s_row * Bc + k] * V_smem[k * d + col];
                }
                
                // Update output with the running computation
                float o_old = O_smem[s_row * d + col];
                float o_new = (1.0f / l_new[s_row]) * (my_l_i * alpha * o_old + beta * PV_acc);  // Use register value
                O_smem[s_row * d + col] = o_new;
                O[QKV_head_offset + (i * Br + s_row) * d + col] = o_new;
            }
            
            // Update register values for next iteration
            my_m_i = m_new[s_row];
            my_l_i = l_new[s_row];
            
            // Write back to global memory
            if (global_row < T) {
                m[lm_head_offset + global_row] = my_m_i;
                l[lm_head_offset + global_row] = my_l_i;
            }
            __syncthreads();
        }
        __syncthreads();
    }
} 

// Host LAUNCHER function
void flashattn_v2(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; const int Br = 32;
    const int BLOCK_SIZE = Bc * Br;
    const float scale = 1.0 / sqrt(d);    

    dim3 grid_dim(B * nh);  // batch_size x num_heads
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v2<BLOCK_SIZE, Br, Bc><<<grid_dim, block_dim>>>(
        Q, K, V, O, l, m, B, nh, T, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}