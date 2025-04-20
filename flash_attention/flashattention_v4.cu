#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int WARP_SIZE = 32;
constexpr int d = 64;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int BLOCK_SIZE, int Br, int Bc, int Bk, int TM, int TN> 
__global__ void flashattn_kernel_v4(
    const float *Q,
    const float *K,
    const float *V,
    float *__restrict__ O,
    int B, int nh, int T,
    float scale)
{
    static_assert(Br % TM == 0);
    static_assert(Bc % TN == 0);

    int tid             = threadIdx.x;
    int batch_head_id   = blockIdx.x;
    int i               = blockIdx.y;

    const int num_threads_per_row = Bc / TN;         // 8
    const int tile_thread_row_id  = tid / num_threads_per_row;
    const int tile_thread_col_id  = tid % num_threads_per_row;

    const int Tc        = cdiv(T, Bc);
    const int num_tiles = d / Bc;                     // 2 for d = 64, Bc = 32

    const int num_threads_per_k_row = BLOCK_SIZE / Bk;
    const int k_row = tid / Bk;
    const int k_col = tid % Bk;

    const int KV_head_offset = batch_head_id * T * d;
    const int Q_head_offset  = KV_head_offset + i * Br * d;

    __shared__ float Q_smem[Br * d];
    __shared__ float K_smem[Bc * (Bk + 1)];
    __shared__ float V_smem[Bc * Bk];
    __shared__ float S_ij_smem[Br * (Bc + 1)];

    float m[TM];
    float l[TM]  = {0.f};
    float m_i[TM];

    #pragma unroll
    for (int ii = 0; ii < TM; ++ii) m[ii] = -INFINITY;

    float Q_reg[TM];
    float K_reg[TN];

    float O_i[num_tiles][TM * TN] = {0.f};

    for (int idx = tid; idx < d * Br; idx += BLOCK_SIZE) {
        int q_row = idx / d;
        int q_col = idx % d;
        Q_smem[q_row * d + q_col] = Q[Q_head_offset + q_row * d + q_col];
    }
    __syncthreads();

    for (int j = 0; j < Tc && j <= i; ++j) {
        float acc[TM][TN] = {0.f};

        for (int tile = 0; tile < num_tiles; tile++) {
            int block_k = tile * Bc;

            for (int x = k_row; x < Bc; x += num_threads_per_k_row) {
                int g_idx = (j * Bc + x) * d + block_k + k_col;
                K_smem[x * (Bk + 1) + k_col] = K[KV_head_offset + g_idx];
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < Bk; ++k) {
                #pragma unroll
                for (int mm = 0; mm < TM; ++mm)
                    Q_reg[mm] = Q_smem[(tile_thread_row_id * TM + mm) * d + block_k + k];

                #pragma unroll
                for (int nn = 0; nn < TN; ++nn)
                    K_reg[nn] = K_smem[(tile_thread_col_id * TN + nn) * (Bk + 1) + k];

                #pragma unroll
                for (int mm = 0; mm < TM; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < TN; ++nn)
                        acc[mm][nn] += Q_reg[mm] * K_reg[nn] * scale;  
            }
            __syncthreads();
        }

        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            int q_pos = i * Br + tile_thread_row_id * TM + mm;
            for (int nn = 0; nn < TN; ++nn) {
                int k_pos = j * Bc + tile_thread_col_id * TN + nn;
                float val = (k_pos > q_pos) ? -INFINITY : acc[mm][nn];
                S_ij_smem[(tile_thread_row_id * TM + mm) * (Bc + 1) + tile_thread_col_id * TN + nn] = val;
            }
        }
        __syncthreads();


        #pragma unroll
        for (int mm = 0; mm < TM; ++mm) {
            float m_ij = m[mm];
            #pragma unroll
            for (int k = 0; k < Bc; ++k) {
                float val = S_ij_smem[(tile_thread_row_id * TM + mm) * (Bc + 1) + k];
                m_ij = fmaxf(m_ij, val);
            }
            m_i[mm] = m[mm];          
            m[mm]   = m_ij;            
        }


        #pragma unroll
        for (int tile = 0; tile < num_tiles; ++tile) {
            #pragma unroll
            for (int mm = 0; mm < TM; ++mm) {
                #pragma unroll
                for (int nn = 0; nn < TN; ++nn) {
                    O_i[tile][mm * TN + nn] *= __expf(m_i[mm] - m[mm]);
                }
            }
        }
        

        #pragma unroll
        for (int mm = 0; mm < TM; ++mm)
            l[mm] *= __expf(m_i[mm] - m[mm]);
        __syncthreads();

        for (int tile = 0; tile < num_tiles; ++tile) {
            int block_k = tile * Bc;
            for (int x = k_row; x < Bc; x += num_threads_per_k_row) {
                int g_idx = (j * Bc + x) * d + block_k + k_col;
                if (g_idx < T * d) {
                    V_smem[x * Bk + k_col] = V[KV_head_offset + g_idx];
                }
                else {
                    V_smem[x * Bk + k_col] = 0.0f;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < Bc; ++k) {
                #pragma unroll
                for (int mm = 0; mm < TM; ++mm) {
                    Q_reg[mm] = __expf(S_ij_smem[(tile_thread_row_id * TM + mm) * (Bc + 1) + k] - m[mm]);
                    if (tile == 0)
                        l[mm] += Q_reg[mm];                               
                }
                #pragma unroll
                for (int nn = 0; nn < TN; ++nn)
                    K_reg[nn] = V_smem[k * Bk + tile_thread_col_id * TN + nn];

                #pragma unroll
                for (int mm = 0; mm < TM; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < TN; ++nn)
                        O_i[tile][mm * TN + nn] += Q_reg[mm] * K_reg[nn];
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
        int block_k = tile * Bc;
        #pragma unroll
        for (int mm = 0; mm < TM; ++mm)
            #pragma unroll
            for (int nn = 0; nn < TN; ++nn) {
                int out_idx = (tile_thread_row_id * TM + mm) * d +
                              block_k + tile_thread_col_id * TN + nn;
                O[Q_head_offset + out_idx] = O_i[tile][mm * TN + nn] / l[mm];
            }
    }
}


void flashattn_v4(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, 
                  int B, int nh, int T, int d) {
    const int Bc = 32, Br = 32, Bk = 32;
    const int TM = 4, TN = 4;
    const int BLOCK_SIZE = Bc * Br / (TM * TN); //  64
    const float scale = 1.0 / sqrt(d);    

    dim3 grid_dim(B * nh, cdiv(T, Br));  
    dim3 block_dim(BLOCK_SIZE);

    flashattn_kernel_v4<BLOCK_SIZE, Br, Bc, Bk, TM, TN><<<grid_dim, block_dim>>>(
        Q, K, V, O, B, nh, T, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}
