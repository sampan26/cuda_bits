#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_bf16.h>

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }  // https://stackoverflow.com/a/1804686

template <typename T> __device__ ushort f32_to_b16(float x);
template <> __device__ inline ushort f32_to_b16<half>(float x) { return __half_as_ushort(__float2half(x)); }
template <> __device__ inline ushort f32_to_b16<nv_bfloat16>(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

template <typename T> __device__ T f32_to_native(float x); // Maybe rename for clarity?
template <> __device__ inline half f32_to_native<half>(float x) { return __float2half(x); }
template <> __device__ inline nv_bfloat16 f32_to_native<nv_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template <typename T>
__device__ inline uint64_t make_smem_desc(T* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
  }