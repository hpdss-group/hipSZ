#ifndef HIPSZ_H
#define HIPSZ_H

#include <hip/hip_runtime.h>
#include <cstddef>
#include "hipSZ/hipSZ_utility.h"
#include "hipSZ/hipSZ_timer.h"
#include "hipSZ/hipSZ_entry_1D_f32.h"
#include "hipSZ/hipSZ_entry_1D_f64.h"
#include "hipSZ/hipSZ_entry_2D_f32.h"
#include "hipSZ/hipSZ_entry_2D_f64.h"
#include "hipSZ/hipSZ_entry_3D_f32.h"
#include "hipSZ/hipSZ_entry_3D_f64.h"
#include "hipSZ/hipSZ_entry_f32.h"
#include "hipSZ/hipSZ_entry_f64.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIPSZ_DIM_1D = 1, // 1D Processing Manner (can be used for all datasets)
    HIPSZ_DIM_2D = 2, // 2D Processing Manner (can be used for both 2D and 3D)
    HIPSZ_DIM_3D = 3  // 3D Processing Manner (can be used for only 3D datasets)
} hipsz_dim_t;

typedef enum {
    // Numeric values of PLAIN/OUTLIER are kept from the pre-merge hipSZ API;
    // FIXED is appended so existing callers that compare enum values keep
    // their meaning. The imported engine used FIXED=0/PLAIN=1/OUTLIER=2,
    // but all in-tree callers use the named constants.
    HIPSZ_MODE_PLAIN   = 0, // Plain   fixed-length encoding mode (with delta)
    HIPSZ_MODE_OUTLIER = 1, // Outlier fixed-length encoding mode (with delta and outliers)
    HIPSZ_MODE_FIXED   = 2  // No-delta fixed-length encoding mode
} hipsz_mode_t;

typedef enum {
    HIPSZ_TYPE_FLOAT  = 0,  // Single precision floating point (f32)
    HIPSZ_TYPE_DOUBLE = 1   // Double precision floating point (f64)
} hipsz_type_t;

// Legacy 1D interface, kept source-compatible with the pre-merge hipSZ API.
// Plain/outlier/fixed modes are dispatched to the 1D engine; dims is ignored.
void hipSZ_compress(void* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipsz_type_t type, hipsz_mode_t mode, hipStream_t stream = 0);
void hipSZ_decompress(void* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipsz_type_t type, hipsz_mode_t mode, hipStream_t stream = 0);

// Extended interface: adds the processing dimension and shape. The source
// imported unified entry points are exported under these names so the legacy
// hipSZ_compress/hipSZ_decompress names above stay binary- and
// source-compatible. dims.x is the fastest-varying dimension.
void hipSZ_compress_ex(void* d_oriData, unsigned char* d_cmpBytes,
                       size_t nbEle, size_t* cmpSize, float errorBound,
                       hipsz_dim_t dim, uint3 dims, hipsz_type_t type, hipsz_mode_t mode,
                       hipStream_t stream = 0);
void hipSZ_decompress_ex(void* d_decData, unsigned char* d_cmpBytes,
                         size_t nbEle, size_t cmpSize, float errorBound,
                         hipsz_dim_t dim, uint3 dims, hipsz_type_t type, hipsz_mode_t mode,
                         hipStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif // HIPSZ_H
