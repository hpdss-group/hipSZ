#include "hipSZ_entry_1D_f32.h"
#include "hipSZ_kernels_1D_f32.h"

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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_1D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
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

    // hipSZ GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_decompress_1D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // hipSZ GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_1D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
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

    // hipSZ GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_decompress_1D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // hipSZ GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_compress_1D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract: all device-side byte offsets are 32-bit; the worst
    // case output is ~136 bytes per 32-element group. Reject empty and
    // oversized inputs instead of corrupting offsets silently.
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) { *cmpSize = 0; return; }
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
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

    // hipSZ GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_compress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    hipMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), hipMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
 * @param   errorBound      user-defined error bound
 * @param   stream          HIP stream for executing compression kernel
 * *********************************************************************** */
void hipSZ_decompress_1D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, hipStream_t stream)
{
    // Capacity contract, mirroring the compress side (32-bit byte offsets).
    if(nbEle == 0 || nbEle/32*136ull > 0xFFFFFFFFull) return;
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    hipMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    hipMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    hipMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    hipMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // hipSZ GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    hipSZ_decompress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
    // Free memory that is used.
    hipFree(d_cmpOffset);
    hipFree(d_locOffset);
    hipFree(d_flag);
}