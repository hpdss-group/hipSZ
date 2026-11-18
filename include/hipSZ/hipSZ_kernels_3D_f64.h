#include <hip/hip_runtime.h>

#ifndef HIPSZ_INCLUDE_HIPSZ_HIPSZ_KERNELS_3D_F64_H
#define HIPSZ_INCLUDE_HIPSZ_HIPSZ_KERNELS_3D_F64_H

static const int tblock_size = 32; // Fixed to 32, cannot be modified.
static const int block_per_thread = 32;

__global__ void hipSZ_compress_kernel_3D_fixed_vec4_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);
__global__ void hipSZ_decompress_kernel_3D_fixed_vec4_f64(double* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);
__global__ void hipSZ_compress_kernel_3D_plain_vec4_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);
__global__ void hipSZ_decompress_kernel_3D_plain_vec4_f64(double* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);
__global__ void hipSZ_compress_kernel_3D_outlier_vec4_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);
__global__ void hipSZ_decompress_kernel_3D_outlier_vec4_f64(double* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const double eb);

#endif // HIPSZ_INCLUDE_HIPSZ_HIPSZ_KERNELS_3D_F64_H