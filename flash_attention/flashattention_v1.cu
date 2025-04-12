#include <cmath>
#include <limits> // For INFINITY
#include <cuda_runtime.h>
#include <stdio.h>

// Include the header declaring the launcher function
#include "flashattention_v1.h"

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Define INFINITY if not already defined (safer than relying on <limits> sometimes)
#ifndef INFINITY
#define INFINITY (__builtin_huge_valf())
#endif

template<int BLOCK_SIZE, int B_r, int B_c> // Add B_r, B_c as template params for shared mem calculation
__global__ void flashattn_kernel_v1(
    // Input/Output Pointers (HBM)
    const float *Q, //  [B, nh, T, head_dim]
    const float *K, //  [B, nh, T, head_dim]
    const float *V, //  [B, nh, T, head_dim]
    float *__restrict__ O, // Output matrix [B, nh, T, head_dim]
    float *l, // Log-sum-exp intermediate [B, nh, T] - Must be pre-initialized (e.g., to 0)
    float *m, // Row max intermediate [B, nh, T] - Must be pre-initialized (e.g., to -INFINITY)
    int B, int nh, int T, int head_dim, // dimensions
    // FlashAttention Parameters
    float scale // Scale factor (usually 1/sqrt(head_dim))
    // B_c, B_r are now template parameters
    // T_c, T_r are calculated inside or passed if needed (often calculated in launcher)
) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x; // Each block processes one head

    const int T_c = cdiv(T, B_c);
    const int T_r = cdiv(T, B_r);

    const int batch_id = block_id / nh;
    const int head_id = block_id % nh;

    // Calculate offsets for the current head
    const int head_offset = (batch_id * nh + head_id) * (T * head_dim);
    const float* Q_head = Q + head_offset;
    const float* K_head = K + head_offset;
    const float* V_head = V + head_offset;
    float* O_head = O + head_offset;

    const int lm_offset = (batch_id * nh + head_id) * T;
    float* l_head = l + lm_offset;
    float* m_head = m + lm_offset;

    // --- Shared Memory Allocation ---
    // Needs to be sized correctly based on template parameters B_r, B_c, head_dim
    extern __shared__ float smem[];

    // Pointers within shared memory
    float* Q_smem = smem;                                         // Size: B_r * head_dim
    float* K_smem = Q_smem + B_r * head_dim;                      // Size: B_c * head_dim
    float* V_smem = K_smem + B_c * head_dim;                      // Size: B_c * head_dim
    float* O_smem = V_smem + B_c * head_dim;                      // Size: B_r * head_dim (accumulator for O_i)
    float* S_ij_smem = O_smem + B_r * head_dim;                   // Size: B_r * B_c (used for S_ij, then P_ij)

    // Intermediate row-wise stats in shared memory
    float* m_tilde_smem = S_ij_smem + B_r * B_c;                 // Size: B_r (Temporary max for current block S_ij) - Overlap carefully if needed! Let's place it after S_ij
    float* l_tilde_smem = m_tilde_smem + B_r;                     // Size: B_r (Temporary sum for current block P_ij)

    float* l_i_smem = l_tilde_smem + B_r;                         // Size: B_r (Running log-sum-exp)
    float* m_i_smem = l_i_smem + B_r;                             // Size: B_r (Running max)
    // These can potentially overlap m_tilde/l_tilde if needed, but safer separate
    float* m_new_smem = m_i_smem + B_r;                           // Size: B_r (Result of merging m_i and m_tilde)
    float* l_new_smem = m_new_smem + B_r;                         // Size: B_r (Result of merging l_i and l_tilde)


    // Outer loop over K/V blocks (columns of the large conceptual attention matrix)
    for (int j = 0; j < T_c; ++j) {
        int KVj_start_row = j * B_c;

        // --- Load K_j, V_j from HBM to Shared Memory ---
        // Each thread loads multiple elements using stride loop
        // Ensure threads don't read out of bounds for K/V
        for (int k_tile_row = tid / head_dim; k_tile_row < B_c; k_tile_row += BLOCK_SIZE / head_dim) { // Group threads to load rows
            int local_k_col = tid % head_dim;
            int global_KV_row = KVj_start_row + k_tile_row;
            if (global_KV_row < T) {
                for (int k_col_offset = 0; local_k_col + k_col_offset < head_dim; k_col_offset += (BLOCK_SIZE % head_dim == 0 ? BLOCK_SIZE/B_c : 1) ) { // stride through columns
                     K_smem[k_tile_row * head_dim + local_k_col + k_col_offset] = K_head[global_KV_row * head_dim + local_k_col + k_col_offset];
                     V_smem[k_tile_row * head_dim + local_k_col + k_col_offset] = V_head[global_KV_row * head_dim + local_k_col + k_col_offset];
                }
            } else {
                 // Padding: zero out K/V rows beyond sequence length T
                for (int k_col_offset = 0; local_k_col + k_col_offset < head_dim; k_col_offset += (BLOCK_SIZE % head_dim == 0 ? BLOCK_SIZE/B_c : 1)) {
                    K_smem[k_tile_row * head_dim + local_k_col + k_col_offset] = 0.0f;
                    V_smem[k_tile_row * head_dim + local_k_col + k_col_offset] = 0.0f;
                }
            }
        }
        __syncthreads(); // Ensure K_j, V_j are loaded before proceeding

        // Inner loop over Q/O blocks (rows of the large conceptual attention matrix)
        for (int i = 0; i < T_r; ++i) {
            int QOi_start_row = i * B_r;

            // --- Load Q_i HBM to Shared Memory ---
            // --- Load O_i, l_i, m_i from HBM/Shared Memory ---
            // Similar loading pattern as K/V
            for (int q_tile_row = tid / head_dim; q_tile_row < B_r; q_tile_row += BLOCK_SIZE / head_dim) { // Group threads to load rows
                int local_q_col = tid % head_dim;
                int global_QO_row = QOi_start_row + q_tile_row;

                if (global_QO_row < T) {
                     // Load Q element by element striding over columns
                     for (int q_col_offset = 0; local_q_col + q_col_offset < head_dim; q_col_offset += (BLOCK_SIZE % head_dim == 0 ? BLOCK_SIZE/B_r : 1)) { // Stride columns
                         Q_smem[q_tile_row * head_dim + local_q_col + q_col_offset] = Q_head[global_QO_row * head_dim + local_q_col + q_col_offset];
                     }
                     // Load O, l, m row-wise (one thread per row is enough if B_r <= BLOCK_SIZE)
                     if (local_q_col == 0) { // Only first thread in row-group loads l, m, O
                        // Initialize O_smem for accumulation in this inner loop iteration
                        // If it's the first K/V block (j==0), O_smem should be 0.
                        // Otherwise, it should contain the value computed from previous K/V blocks.
                        // The current implementation implicitly reads O from HBM, overwriting previous O_smem content.
                        // Let's initialize O_smem to 0 for the first block j=0,
                        // and load the accumulated value for subsequent j.
                        if (j == 0) {
                            for(int k=0; k<head_dim; ++k) O_smem[q_tile_row * head_dim + k] = 0.0f;
                            l_i_smem[q_tile_row] = 0.0f;      // Initial l_i is 0
                            m_i_smem[q_tile_row] = -INFINITY; // Initial m_i is -inf
                        } else {
                            // Load accumulated O from HBM for intermediate steps
                             for (int q_col_offset = 0; q_col_offset < head_dim; q_col_offset += 1) { // Stride columns
                                  O_smem[q_tile_row * head_dim + q_col_offset] = O_head[global_QO_row * head_dim + q_col_offset];
                             }
                            l_i_smem[q_tile_row] = l_head[global_QO_row];
                            m_i_smem[q_tile_row] = m_head[global_QO_row];
                        }
                     }
                } else {
                    // Padding for Q/O rows beyond T
                     for (int q_col_offset = 0; local_q_col + q_col_offset < head_dim; q_col_offset += (BLOCK_SIZE % head_dim == 0 ? BLOCK_SIZE/B_r : 1)) {
                        Q_smem[q_tile_row * head_dim + local_q_col + q_col_offset] = 0.0f;
                    }
                    if (local_q_col == 0) {
                        for(int k=0; k<head_dim; ++k) O_smem[q_tile_row * head_dim + k] = 0.0f; // Zero accumulator
                        l_i_smem[q_tile_row] = 0.0f;      // Padded row stats
                        m_i_smem[q_tile_row] = -INFINITY;
                    }
                }
            }
            __syncthreads(); // Ensure Q_i, O_i, l_i, m_i are loaded/initialized

            // --- Compute S_ij = Q_i @ K_j^T ---
            // Each thread computes one element of S_ij
            // Requires BLOCK_SIZE >= B_r * B_c for simple 1 thread per element
            // For BLOCK_SIZE=128, B_r=64, B_c=64, this is not true.
            // We need a loop structure where threads cooperate.
            // Simple (but less optimal) version: Each thread computes multiple S_ij elements.
            for (int row = tid / B_c; row < B_r; row += BLOCK_SIZE / B_c) { // Thread strides over rows of S_ij
                int col = tid % B_c;                                       // Thread handles one column of S_ij
                int q_global_row = QOi_start_row + row;

                if (q_global_row < T) {
                    float sum = 0.0f;
                    int k_global_row = KVj_start_row + col; // Corresponding K row index

                    if (k_global_row < T && k_global_row <= q_global_row) { // Check bounds and apply causal mask
                        // Compute dot product Q[row,:] * K[col,:]
                        #pragma unroll
                        for (int k = 0; k < head_dim; k++) {
                            sum += Q_smem[row * head_dim + k] * K_smem[col * head_dim + k];
                        }
                        S_ij_smem[row * B_c + col] = sum * scale;
                    } else {
                        S_ij_smem[row * B_c + col] = -INFINITY; // Masked value
                    }
                } else {
                    // Padding for S_ij rows corresponding to padded Q rows
                    S_ij_smem[row * B_c + col] = -INFINITY;
                }
            }
            __syncthreads(); // Ensure S_ij is computed

            // --- Compute m_tilde, l_tilde, P_ij = softmax(S_ij) ---
            // Row-wise reduction for max (m_tilde) and sum (l_tilde)
            // Threads stride over rows
            for (int row = tid; row < B_r; row += BLOCK_SIZE) {
                int q_global_row = QOi_start_row + row;
                if (q_global_row < T) {
                    // Find row max (m_tilde)
                    float m_pq = -INFINITY;
                    for (int col = 0; col < B_c; ++col) {
                        m_pq = max(m_pq, S_ij_smem[row * B_c + col]);
                    }
                    // Store temporary row max, check for -inf case (all masked)
                    m_tilde_smem[row] = (m_pq == -INFINITY) ? 0.0f : m_pq; // Avoid NaN from exp(-inf - (-inf))

                    // Compute P_ij = exp(S_ij - m_tilde) and find row sum (l_tilde)
                    float l_pq = 0.0f;
                    for (int col = 0; col < B_c; ++col) {
                        // Use fetched m_pq which is the correct max for *this* row
                        float p_val = expf(S_ij_smem[row * B_c + col] - m_pq);
                         // Check if m_pq was -inf, meaning all elements were masked. If so, p_val is exp(-inf - (-inf)) -> NaN. Set to 0.
                        if (m_pq == -INFINITY) {
                            p_val = 0.0f;
                        }
                        S_ij_smem[row * B_c + col] = p_val; // Overwrite S_ij with P_ij
                        l_pq += p_val;
                    }
                    l_tilde_smem[row] = l_pq;
                } else {
                    // Padding
                    m_tilde_smem[row] = -INFINITY; // Use -inf for padded max
                    l_tilde_smem[row] = 0.0f;
                    // Zero out P_ij for padded rows
                    for (int col = 0; col < B_c; ++col) {
                        S_ij_smem[row * B_c + col] = 0.0f;
                    }
                }
            }
            __syncthreads(); // Ensure m_tilde, l_tilde, P_ij computed

            // --- Update O_i, l_i, m_i ---
            // Threads stride over rows
            for (int row = tid; row < B_r; row += BLOCK_SIZE) {
                int q_global_row = QOi_start_row + row;
                if (q_global_row < T) {
                    // Load current running values
                    float m_i = m_i_smem[row];
                    float l_i = l_i_smem[row];
                    // Load stats for the current block P_ij
                    float m_tilde = m_tilde_smem[row];
                    float l_tilde = l_tilde_smem[row];

                    // Compute new max
                    float m_new = max(m_i, m_tilde);
                    // Handle case where initial m_i was -inf
                    if (m_i == -INFINITY) m_new = m_tilde;


                    // Compute renormalization factors using the new max
                    // Careful with expf(very small number) -> near zero
                    float exp_m_diff = expf(m_i - m_new);
                    float exp_mtilde_diff = expf(m_tilde - m_new);
                    if (m_i == -INFINITY) exp_m_diff = 0.0f; // exp(-inf - m_new) -> 0
                    if (m_tilde == -INFINITY) exp_mtilde_diff = 0.0f; // exp(-inf - m_new) -> 0

                    // Compute new l
                    float l_new = exp_m_diff * l_i + exp_mtilde_diff * l_tilde;

                    // Update O_i: Scale previous O_i value
                    // O_i_new = (l_i * exp(m_i - m_new) * O_i + l_tilde * exp(m_tilde - m_new) * P_ij @ V_j) / l_new
                    // First part: Rescale existing O_i (stored in O_smem)
                    float scale_factor_O = (l_i * exp_m_diff) / l_new;
                    // Handle division by zero if l_new is 0 (e.g., all masked)
                    if (l_new == 0.0f || l_i == 0.0f) {
                        scale_factor_O = 0.0f; // If l_new is 0, the contribution must be 0
                    }

                    #pragma unroll
                    for (int k = 0; k < head_dim; ++k) {
                        // O_smem currently holds O from previous blocks (or 0 if j=0)
                        O_smem[row * head_dim + k] *= scale_factor_O;
                    }

                    // Store new m and l for the next iteration (or final write-back)
                    m_new_smem[row] = m_new;
                    l_new_smem[row] = l_new;

                    // Compute update for O_i from P_ij @ V_j
                    // Second part: Add contribution from (P_ij @ V_j) * scale_factor_PV
                    float scale_factor_PV = (exp_mtilde_diff) / l_new; // Note: l_tilde absorbed in P_ij sum implicitly
                    if (l_new == 0.0f || l_tilde == 0.0f) {
                        scale_factor_PV = 0.0f; // If l_new or l_tilde is 0, contribution is 0
                    }

                    // Compute O_update = P_ij[row,:] @ V_j[:, k_out]
                    #pragma unroll
                    for (int k_out = 0; k_out < head_dim; ++k_out) {
                        float o_update_val = 0.0f;
                        // Inner product P[row,:] * V[:, k_out]
                        #pragma unroll
                        for (int k_inner = 0; k_inner < B_c; ++k_inner) {
                            o_update_val += S_ij_smem[row * B_c + k_inner] * V_smem[k_inner * head_dim + k_out];
                        }
                        // Add scaled contribution to O_smem
                        O_smem[row * head_dim + k_out] += o_update_val * scale_factor_PV;
                    }

                } else {
                    // Padding: Ensure new stats are identity values for next step if any
                    m_new_smem[row] = -INFINITY;
                    l_new_smem[row] = 0.0f;
                    // O_smem for padded rows should remain 0 (was initialized)
                }
            }
            __syncthreads(); // Ensure O_i updates, m_new, l_new are complete

            // --- Write O_i, l_i, m_i back to HBM ---
            // Update l_i, m_i in shared memory for the *next* inner loop iteration
            // Write final O_i, l_i, m_i to HBM
            // Only write back *final* values after the outer loop (j) completes? No, standard FlashAttention updates HBM incrementally.
            // Let's write back O, l_new, m_new to HBM here.
             for (int q_tile_row = tid / head_dim; q_tile_row < B_r; q_tile_row += BLOCK_SIZE / head_dim) {
                int local_q_col = tid % head_dim;
                int global_QO_row = QOi_start_row + q_tile_row;

                if (global_QO_row < T) {
                    // Write O element by element striding over columns
                    for (int q_col_offset = 0; local_q_col + q_col_offset < head_dim; q_col_offset += (BLOCK_SIZE % head_dim == 0 ? BLOCK_SIZE / B_r : 1)) {
                        O_head[global_QO_row * head_dim + local_q_col + q_col_offset] = O_smem[q_tile_row * head_dim + local_q_col + q_col_offset];
                    }
                    // Write l, m row-wise (one thread per row is enough)
                    if (local_q_col == 0) {
                        l_head[global_QO_row] = l_new_smem[q_tile_row];
                        m_head[global_QO_row] = m_new_smem[q_tile_row];
                        // Update l_i_smem and m_i_smem for the *next* iteration of i (if any)
                        l_i_smem[q_tile_row] = l_new_smem[q_tile_row];
                        m_i_smem[q_tile_row] = m_new_smem[q_tile_row];
                    }
                }
            }
            __syncthreads(); // Ensure write-back is complete before next K/V block

        } // End inner loop (i over Q/O blocks)
    } // End outer loop (j over K/V blocks)
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

    // --- Block Sizes ---
    // These need to be chosen carefully based on head_dim and available shared memory
    // Keeping fixed values for simplicity, but should ideally be tuned.
    constexpr int B_c = 16;
    constexpr int B_r = 16;
    // Ensure head_dim is compatible (e.g., divisible by element size for vectorization if used later)
    // assert(head_dim % 4 == 0); // Example check

    // --- Kernel Launch Configuration ---
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const unsigned int grid_size = B * nh; // One block per head
    constexpr int BLOCK_SIZE = 128; // Number of threads per block

    // --- Shared Memory Calculation ---
    // Must match the layout within the kernel!
    // size_t shared_mem_bytes = 0;
    // shared_mem_bytes += B_r * head_dim * sizeof(float); // Q_smem
    // shared_mem_bytes += B_c * head_dim * sizeof(float); // K_smem
    // shared_mem_bytes += B_c * head_dim * sizeof(float); // V_smem
    // shared_mem_bytes += B_r * head_dim * sizeof(float); // O_smem
    // shared_mem_bytes += B_r * B_c * sizeof(float);      // S_ij_smem / P_ij_smem
    // // Intermediate stats
    // shared_mem_bytes += B_r * sizeof(float); // m_tilde_smem
    // shared_mem_bytes += B_r * sizeof(float); // l_tilde_smem
    // shared_mem_bytes += B_r * sizeof(float); // l_i_smem
    // shared_mem_bytes += B_r * sizeof(float); // m_i_smem
    // shared_mem_bytes += B_r * sizeof(float); // m_new_smem
    // shared_mem_bytes += B_r * sizeof(float); // l_new_smem
    size_t shared_mem_bytes = sizeof(float) * (
        B_r * head_dim +     // Q_smem
        B_c * head_dim * 2 + // K_smem and V_smem
        B_r * head_dim +     // O_smem
        B_r * B_c +          // S_ij_smem / P_ij_smem
        B_r * 6              // All row stats (m_tilde, l_tilde, l_i, m_i, m_new, l_new)
    );

    if (!check_shared_memory_requirements(shared_mem_bytes)) {
        // Handle error or adjust parameters
        printf("Adjusting block sizes due to shared memory constraints...\n");
        // Here you could implement fallback logic to use smaller B_r and B_c
        // or throw an exception back to the calling code
        return;
    }


    // --- Kernel Launch ---
    flashattn_kernel_v1<BLOCK_SIZE, B_r, B_c><<<grid_size, BLOCK_SIZE, shared_mem_bytes>>>(
        Q, K, V, O,
        l, m, // Pass pointers to l and m buffers
        B, nh, T, head_dim, scale
    );

    // --- Error Handling ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        // Handle error appropriately, maybe throw exception back to Python
    }
    // cudaDeviceSynchronize(); // Optional: Wait for kernel completion for debugging/profiling
}