#ifndef HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F32_H
#define HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F32_H

#include <hip/hip_runtime.h>

void hipSZ_compress_3D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);
void hipSZ_compress_3D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);
void hipSZ_compress_3D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, hipStream_t stream = 0);

#endif // HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F32_H