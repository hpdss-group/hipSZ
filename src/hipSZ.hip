#include "hipSZ.h"

// Wrap-up API for hipSZ compression.
void hipSZ_compress_ex(void* d_oriData, unsigned char* d_cmpBytes, 
                    size_t nbEle, size_t* cmpSize, float errorBound,  
                    hipsz_dim_t dim, uint3 dims, hipsz_type_t type, hipsz_mode_t mode, 
                    hipStream_t stream)
{
    if (type == HIPSZ_TYPE_FLOAT) {
        if (mode == HIPSZ_MODE_PLAIN) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_plain_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_plain_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_plain_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        } 
        else if (mode == HIPSZ_MODE_OUTLIER) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_outlier_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_outlier_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_outlier_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else if (mode == HIPSZ_MODE_FIXED) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_fixed_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_fixed_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_fixed_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else {
            printf("Unsupported mode in hipSZ.\n");
        }
    } 
    else if (type == HIPSZ_TYPE_DOUBLE) {
        double errorBound_f64 = (double)errorBound;
        if (mode == HIPSZ_MODE_PLAIN) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_plain_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_plain_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_plain_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        } 
        else if (mode == HIPSZ_MODE_OUTLIER) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_outlier_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_outlier_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_outlier_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else if (mode == HIPSZ_MODE_FIXED) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_compress_1D_fixed_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_compress_2D_fixed_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_compress_3D_fixed_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else {
            printf("Unsupported mode in hipSZ.\n");
        }
    }
    else {
        printf("Unsupported type in hipSZ.\n");
    }
}

// Wrap-up API for hipSZ decompression.
void hipSZ_decompress_ex(void* d_decData, unsigned char* d_cmpBytes, 
                      size_t nbEle, size_t cmpSize, float errorBound,  
                      hipsz_dim_t dim, uint3 dims, hipsz_type_t type, hipsz_mode_t mode, 
                      hipStream_t stream)
{
    if (type == HIPSZ_TYPE_FLOAT) {
        if (mode == HIPSZ_MODE_PLAIN) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_plain_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_plain_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_plain_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        } 
        else if (mode == HIPSZ_MODE_OUTLIER) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_outlier_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_outlier_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_outlier_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else if (mode == HIPSZ_MODE_FIXED) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_fixed_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_fixed_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_fixed_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else {
            printf("Unsupported mode in hipSZ.\n");
        }
    } 
    else if (type == HIPSZ_TYPE_DOUBLE) {
        double errorBound_f64 = (double)errorBound;
        if (mode == HIPSZ_MODE_PLAIN) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_plain_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_plain_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_plain_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        } 
        else if (mode == HIPSZ_MODE_OUTLIER) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_outlier_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_outlier_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_outlier_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else if (mode == HIPSZ_MODE_FIXED) {
            if (dim == HIPSZ_DIM_1D)
                hipSZ_decompress_1D_fixed_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_2D)
                hipSZ_decompress_2D_fixed_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else if (dim == HIPSZ_DIM_3D)
                hipSZ_decompress_3D_fixed_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound_f64, stream);
            else
                printf("Unsupported dimension in hipSZ.\n");
        }
        else {
            printf("Unsupported mode in hipSZ.\n");
        }
    }
    else {
        printf("Unsupported type in hipSZ.\n");
    }
}
