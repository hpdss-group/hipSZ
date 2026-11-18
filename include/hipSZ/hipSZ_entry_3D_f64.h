#ifndef HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F64_H
#define HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F64_H

#include <hip/hip_runtime.h>

void hipSZ_compress_3D_fixed_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_fixed_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);
void hipSZ_compress_3D_plain_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_plain_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);
void hipSZ_compress_3D_outlier_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);
void hipSZ_decompress_3D_outlier_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream = 0);

#endif // HIPSZ_INCLUDE_HIPSZ_HIPSZ_ENTRY_3D_F64_H