#include "hipSZ_entry_2D_f64.h"
#include "hipSZ_kernels_2D_f64.h"

/** ************************************************************************
 * @brief hipSZ end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   dims            dimensions of the original data
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_2D_fixed_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression (implement for now, will update later)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_2D_fixed_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (blockNum + 3) / 4 * 4;  // include the rate array's 4-byte alignment padding (matches rate_ofs in the kernels)

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
} 

/** ************************************************************************
 * @brief hipSZ end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 * 
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   dims            dimensions of the original data
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_decompress_2D_fixed_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression (implement for now, will update later with adaptive vectorization)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_2D_fixed_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
}

/** ************************************************************************
 * @brief hipSZ end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   dims            dimensions of the original data
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_2D_plain_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression (implement for now, will update later)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_2D_plain_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (blockNum + 3) / 4 * 4;  // include the rate array's 4-byte alignment padding (matches rate_ofs in the kernels)

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
} 

/** ************************************************************************
 * @brief hipSZ end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 * 
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   dims            dimensions of the original data
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_decompress_2D_plain_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression (implement for now, will update later with adaptive vectorization)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_2D_plain_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
}

/** ************************************************************************
 * @brief hipSZ end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   dims            dimensions of the original data
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_2D_outlier_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression (implement for now, will update later)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_2D_outlier_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (blockNum + 3) / 4 * 4;  // include the rate array's 4-byte alignment padding (matches rate_ofs in the kernels)

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
} 
 
/** ************************************************************************
* @brief hipSZ end-to-end decompression API for device pointers
*        Decompression is executed in GPU.
*        Compressed data is stored as device pointers (in GPU).
*        Reconstructed data is stored as device pointers (in GPU).
*        P.S. Reconstructed data and original data have the same shape.
* 
* @param   d_decData       reconstructed data (device pointer)
* @param   d_cmpBytes      compressed data (device pointer)
* @param   nbEle           reconstructed data size (number of floating point)
* @param   cmpSize         compressed data size (number of unsigned char)
* @param   dims            dimensions of the original data
* @param   errorBound      user-defined error bound
* @param   stream          HIP stream for executing compression kernel
* *********************************************************************** */
void hipSZ_decompress_2D_outlier_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, double errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking. If 3D, then process as a set of 2D slices.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimxBlock = (dims.x + 7) / 8;
    uint blockNum = dims.z * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression (implement for now, will update later with adaptive vectorization)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_2D_outlier_vec4_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
}