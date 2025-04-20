#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc> 
__global__ void flashattn_kernel_v3(
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, //  [B, nh, T, head_dim]
    int B, int nh, int T, 
    float scale // Scale factor (usually 1/sqrt(head_dim))
) { 
    int tid = threadIdx.x;
    int batch_head_id = blockIdx.x;
    int i = blockIdx.y;

    int s_row = tid / Bc;
    int s_col = tid % Bc;

    const int Tc = cdiv(T, Bc); const int Tr = cdiv(T, Br);

    const int KV_head_offset = batch_head_id * T * d;
    const int Q_head_offset = KV_head_offset + i * Br * d;

    int num_tiles_kv = cdiv(d, Br);
    int num_tiles_q = cdiv(d, Bc);

    __shared__ float Q_smem[Br * (d + 1)];
    __shared__ float K_smem[Bc * (d + 1)];
    __shared__ float V_smem[Bc * (d + 1)];
    __shared__ float S_ij_smem[Br * Bc];

    float O_i[2];
    for (int i = 0; i < num_tiles_q; i++) {
        O_i[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    // Load Q and O tiles into shared memory
    for (int x = 0; x < num_tiles_q; ++x) {
        int idx = x * (Br * Bc) + tid;
        int q_row = idx / d;
        int q_col = idx % d; 
        Q_smem[q_row * d + q_col] = Q[Q_head_offset + q_row * d + q_col];
    }
    __syncthreads();

    for (int j = 0; j < Tc; ++j) {
        // Load K and V tiles into shared memory
        for (int x = 0; x < num_tiles_kv; ++x) {
            int idx = x * (Bc * Br) + tid;
            int kv_row = idx / d;
            int kv_col = idx % d;
            K_smem[kv_row * d + kv_col] = K[KV_head_offset + (j * Bc + kv_row) * d + kv_col];
            V_smem[kv_row * d + kv_col] = V[KV_head_offset + (j * Bc + kv_row) * d + kv_col];
        }
        
        __syncthreads();

        float acc = 0.0f;
        for (int k = 0; k < d; k++) {
            acc += Q_smem[s_row * d + k] * K_smem[s_col * d + k];
        }
        acc *= scale;

        int query_pos = i * Br + s_row;
        int key_pos = j * Bc + s_col;
        if (key_pos > query_pos) {
            acc = -INFINITY;
        }
            
        S_ij_smem[s_row * Bc + s_col] = acc;
        __syncthreads();
        
        float m_i = m;
        float m_ij = m;
        // Compute row max and softmax
        for (int k = 0; k < Bc; k++) {
            float val = S_ij_smem[s_row * Bc + k];
            if (val > m_ij) {
                m_ij = val;
            }
        }
        m = m_ij;
        
        for (int i = 0; i < num_tiles_q; i++) {
            O_i[i] *= __expf(m_i - m_ij);
        }
        float l_i = __expf(m_i - m_ij) * l;
        float P_ij;
        for (int k = 0; k < Bc; k++) {
            P_ij = __expf(S_ij_smem[s_row * Bc + k] - m);
            l_i += P_ij;
            for (int i = 0; i < num_tiles_kv; i++) {
                O_i[i] += P_ij * V_smem[k * d + (s_col + i * Bc)];
            }
        }
        l = l_i;
        __syncthreads();
    }

    
    for (int i = 0; i < num_tiles_kv; i++) {
        O[Q_head_offset + s_row * d + i * Bc + s_col] = O_i[i] / l;
    }
} 

void flashattn_v3(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32; const int Br = 32;
    const int BLOCK_SIZE = Bc * Br;
    const float scale = 1.0 / sqrt(d);    

    dim3 grid_dim(B * nh, cdiv(T, Br));  // batch_size x num_heads
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v3<BLOCK_SIZE, Br, Bc><<<grid_dim, block_dim>>>(
        Q, K, V, O, B, nh, T, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}
