#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <int BLOCK_SIZE, int Br, int Bc, int Bk, int TM, int TN>
__global__ void flashattn_kernel_v5(
    const float*__restrict__ Q,
    const float*__restrict__ K,
    const float*__restrict__ V,
    float*__restrict__ O,
    float scale,
    int T,
    int d)
{
    const int tid = threadIdx.x;
    const int batch_head_id = blockIdx.x;
    const int i = blockIdx.y;

    const int batch_head_offset = batch_head_id * T * d;
    const int QO_offset = batch_head_offset + i * Br * d;

    __shared__ float Q_smem[Br][64];
    __shared__ float K_smem[Bc][Bk + 1];
    __shared__ float V_smem[Bc][Bk];
    __shared__ float S_ij_smem[Br][Bc + 1];

    constexpr int num_threads_per_k_row = Bc / TN;
    int tile_thread_row_id = tid / num_threads_per_k_row;
    int tile_thread_col_id = tid % num_threads_per_k_row;

    const int Tc = cdiv(T, Bc);

    float l[TM] = {0.f};
    float m_i[TM];
    float m[TM];
    for (int i = 0; i < TM; ++i) m[i] = -INFINITY;

    float O_i[2 * TM * TN] = {0.f};

    const int q_col = tid % d;
    for (int row = 0; row < Br; ++row) {
        int g_idx = q_col + row * BLOCK_SIZE; // BLOCK_SIZE = 64 = d
        Q_smem[row][q_col] = Q[QO_offset + g_idx];
    }
    __syncthreads();

    float Q_reg[TM];
    float K_reg[TN];
    float V_reg[TN];

    const int k_row = tid / Bk;
    const int k_col = tid % Bk;
    const int strideK = BLOCK_SIZE / Bk;

    for (int j = 0; j < Tc && j <= i; ++j) {
        float acc[TM][TN] = {0.f};
        for (int tile = 0; tile < 2; ++tile) {
            for (int i = 0; i < Br; i += strideK) {
                int g_idx = (k_row + j * Bc) * d + i * d + k_col + tile * Bc;
                K_smem[k_row + i][k_col] = K[batch_head_offset + g_idx];
            }
            __syncthreads();

            for (int k = 0; k < Bk; ++k) {
                for (int mm = 0; mm < TM; ++mm)
                    Q_reg[mm] = Q_smem[tile_thread_row_id * TM + mm][k + tile * Bk];

                for (int nn = 0; nn < TN; ++nn)
                    K_reg[nn] = K_smem[tile_thread_col_id * TN + nn][k];

                for (int mm = 0; mm < TM; ++mm)
                    for (int nn = 0; nn < TN; ++nn)
                        acc[mm][nn] += Q_reg[mm] * K_reg[nn];
            }
            __syncthreads();
        }

        for (int mm = 0; mm < TM; ++mm) {
            for (int nn = 0; nn < TN; ++nn) {
                int key_pos = j * Bc + tile_thread_col_id * TN + nn;
                int query_pos  = i * Br + tile_thread_row_id * TM + mm;
                S_ij_smem[tile_thread_row_id * TM + mm][tile_thread_col_id * TN + nn] =
                    (key_pos <= query_pos) ? acc[mm][nn] * scale : -INFINITY;
            }
        }
        __syncthreads();

        for (int mm = 0; mm < TM; ++mm) {
            m_i[mm] = m[mm];
            float m_ij= m[mm];
            for (int k = 0; k < Bc; ++k) {
                float val = S_ij_smem[tile_thread_row_id * TM + mm][k];
                if (m_ij < val) m_ij = val;
            }
            m[mm] = m_ij;
        }


        for (int tile = 0; tile < 2; ++tile)
            for (int mm = 0; mm < TM; ++mm)
                for (int nn = 0; nn < TN; ++nn)
                    O_i[tile * TM * TN + mm * TN + nn] *= expf(m_i[mm] - m[mm]);

        for (int mm = 0; mm < TM; ++mm) {
            l[mm] *= expf(m_i[mm] - m[mm]);
        }

        for (int tile = 0; tile < 2; ++tile) {
            for (int i = 0; i < Br; i += strideK) {
                int id = (k_row + j * Bc) * d + i * d + k_col + tile * Bc;
                V_smem[k_row + i][k_col] = (id < T * d) ? V[batch_head_offset + id] : 0.f;
            }
            __syncthreads();

            for (int k = 0; k < Bc; ++k) {
                for (int p_m = 0; p_m < TM; ++p_m) {
                    Q_reg[p_m] = expf(S_ij_smem[tile_thread_row_id * TM + p_m][k] - m[p_m]);
                    if (tile == 0) {
                        l[p_m] += Q_reg[p_m];
                    }
                }
                for (int v_n = 0; v_n < TN; ++v_n)
                    V_reg[v_n] = V_smem[k][tile_thread_col_id * TN + v_n];

                for (int p_m = 0; p_m < TM; ++p_m)
                    for (int v_n = 0; v_n < TN; ++v_n)
                        O_i[tile * TM * TN + p_m * TN + v_n] += Q_reg[p_m] * V_reg[v_n];
            }
            __syncthreads();
        }
    }

    for (int tile = 0; tile < 2; ++tile)
        for (int mm = 0; mm < TM; ++mm)
            for (int nn = 0; nn < TN; ++nn) {
                int out_idx = (i * Br + tile_thread_row_id * TM + mm) * d +
                              tile * Bc + tile_thread_col_id * TN + nn;
                if (out_idx < T * d) {
                    O[batch_head_offset + out_idx] = O_i[tile * TM * TN + mm * TN + nn] / l[mm];
                }
            }
}

void flashattn_v5(const float* Q, const float* K, const float* V, float* O, float* m, float* l, int B, int nh, int T, int d)
{
    constexpr int Bc = 32, Br = 32, Bk = 32;
    constexpr int TM = 4, TN = 4;
    const int BLOCK_SIZE = Bc * Br / (TM * TN);

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(B * nh, cdiv(T, Br));
    float scale = 1.0 / sqrt(d);

    flashattn_kernel_v5<BLOCK_SIZE, Br, Bc, Bk, TM, TN><<<grid_dim, block_dim>>>(Q, K, V, O, scale, T, d);
    cudaError_t err = cudaGetLastError();
}
