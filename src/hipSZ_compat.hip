// Compatibility layer for the pre-merge hipSZ 1D API.
//
// The engine below this file is the cuSZp v3.0.0 HIP port imported from
// port. It exposes dimension-aware entry points named
// hipSZ_{compress,decompress}_{1D,2D,3D}_{fixed,plain,outlier}_{f32,f64}.
// The functions here keep the older hipSZ names working:
//   * hipSZ_compress/hipSZ_decompress        (1D, any mode)
//   * hipSZ_{compress,decompress}_{plain,outlier}_{f32,f64}
//
// A compressed stream produced by the pre-merge f32 1D implementation is not
// guaranteed to decode here: the imported f32 1D kernel uses a different
// chunk geometry and rate-array layout. See docs/porting-notes.md.
#include "hipSZ.h"

void hipSZ_compress_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream)
{
    hipSZ_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_decompress_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream)
{
    hipSZ_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_compress_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream)
{
    hipSZ_compress_1D_outlier_f32(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_decompress_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream)
{
    hipSZ_decompress_1D_outlier_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_compress_plain_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, hipStream_t stream)
{
    hipSZ_compress_1D_plain_f64(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_decompress_plain_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, hipStream_t stream)
{
    hipSZ_decompress_1D_plain_f64(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_compress_outlier_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, hipStream_t stream)
{
    hipSZ_compress_1D_outlier_f64(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_decompress_outlier_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, hipStream_t stream)
{
    hipSZ_decompress_1D_outlier_f64(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void hipSZ_compress(void* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipsz_type_t type, hipsz_mode_t mode, hipStream_t stream)
{
    uint3 dims = make_uint3(0, 0, 0);
    hipSZ_compress_ex(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, HIPSZ_DIM_1D, dims, type, mode, stream);
}

void hipSZ_decompress(void* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipsz_type_t type, hipsz_mode_t mode, hipStream_t stream)
{
    uint3 dims = make_uint3(0, 0, 0);
    hipSZ_decompress_ex(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, HIPSZ_DIM_1D, dims, type, mode, stream);
}
