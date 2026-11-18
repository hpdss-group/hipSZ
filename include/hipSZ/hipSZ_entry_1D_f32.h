#ifndef HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_1D_F32_H
#define HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_1D_F32_H

#include <hip/hip_runtime.h>

void hipSZ_compress_1D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_1D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream = 0);
void hipSZ_compress_1D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_1D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream = 0);
void hipSZ_compress_1D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream = 0);
void hipSZ_decompress_1D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream = 0);

#endif // HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_1D_F32_H