#pragma once // Include guard

// Declare the LAUNCHER function (callable from host code)
void flashattn_v1(const float *Q, const float *K, const float *V, float *O,
                  float *l, float *m, // Pass pointers to l and m buffers
                  int B, int nh, int T, int head_dim);