#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipSZ.h>

int main(int argc, char* argv[])
{
    // Variables for user-proposed input.
    char oriFilePath[640] = {0};
    char cmpFilePath[640] = {0};
    char decFilePath[640] = {0};
    hipsz_dim_t processingDim = HIPSZ_DIM_1D;
    hipsz_type_t dataType;
    hipsz_mode_t encodingMode;
    char errorBoundMode[4] = {0};
    float errorBound = 0.0;
    uint3 dims = {0, 0, 0};

    // Usage information
    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))) {
        printf("hipSZ: A GPU error-bounded lossy compressor for floating-point data\n");
        printf("\n");
        printf("Usage:\n");
        printf("  ./hipSZ -i [oriFilePath] -t [dataType] -m [encodingMode] [-d [dim] [dim_z] [dim_y] [dim_x]] -eb [errorBoundMode] [errorBound] -x [cmpFilePath] -o [decFilePath]\n");
        printf("\n");
        printf("Options:\n");
        printf("  -i  [oriFilePath]    Path to the input file containing the original data\n");
        printf("  -t  [dataType]       Data type of the input data. Options:\n");
        printf("                       f32      : Single precision (float)\n");
        printf("                       f64      : Double precision (double)\n");
        printf("  -m  [encodingMode]   Encoding mode to use. Options:\n");
        printf("                       fixed    : No-delta fixed-length encoding mode\n");
        printf("                       plain    : Plain fixed-length encoding mode (with delta)\n");
        printf("                       outlier  : Outlier fixed-length encoding mode (with delta and outlier preservation)\n");
        printf("  -d  [dim]            Data dimensionality and processing manner. Optional; default 1 (1D). Options:\n");
        printf("                       1                         : 1D Processing Manner (can be used for all datasets)\n");
        printf("                       2 [dim_z] [dim_y] [dim_x] : 2D Processing Manner (can be used for both 2D and 3D datasets)\n");
        printf("                       3 [dim_z] [dim_y] [dim_x] : 3D Processing Manner (can be used for only 3D datasets)\n");
        printf("                       Note: dim_x is the fastest varying dimension.\n");
        printf("                       Hint: 2D processing can be used for 3D datasets by compressing each [dim_y x dim_x] slice independently.\n");
        printf("  -eb [errorBoundMode] [errorBound]\n");
        printf("                       errorBoundMode can only be:\n");
        printf("                       abs      : Absolute error bound\n");
        printf("                       rel      : Relative error bound\n");
        printf("                       errorBound is a floating-point number representing the error bound, e.g. 1E-4, 0.03.\n");
        printf("  -x  [cmpFilePath]    Path to the compressed output file (optional)\n");
        printf("  -o  [decFilePath]    Path to the decompressed output file (optional)\n");
        printf("\n");
        printf("Examples:\n");
        printf("  ./hipSZ -i pressure_3000 -t f32 -m plain -d 1 -eb abs 1E-4 -x pressure_3000.cuszp.cmp -o pressure_3000.cuszp.dec\n");
        printf("  ./hipSZ -i pressure_3000 -t f32 -m plain -d 1 -eb rel 1e-4\n");
        printf("  ./hipSZ -i pressure_3000 -t f32 -m plain -d 2 1008 1008 352 -eb rel 1e-4\n");
        printf("  ./hipSZ -i pressure_3000 -t f32 -m plain -d 3 1008 1008 352 -eb rel 1e-4\n");
        printf("  ./hipSZ -i ccd-tst.bin.d64 -t f64 -m outlier -d 1 -eb abs 0.01\n");
        printf("  ./hipSZ -i velocity_x.f32 -t f32 -m outlier -d 1 -eb rel 0.01 -x velocity_x.f32.cuszp.cmp\n");
        printf("  ./hipSZ -i xx.f32 -m outlier -eb rel 1e-4 -t f32 -d 1\n");
        exit(EXIT_FAILURE);
    }

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            // Input file path
            if (i + 1 < argc) {
                strncpy(oriFilePath, argv[i + 1], sizeof(oriFilePath) - 1);
                i++;
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            // Data type
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "f32") == 0) {
                    dataType = HIPSZ_TYPE_FLOAT;
                } else if (strcmp(argv[i + 1], "f64") == 0) {
                    dataType = HIPSZ_TYPE_DOUBLE;
                } else {
                    printf("Error: Unsupported data type. Use 'f32' or 'f64'.\n");
                    exit(EXIT_FAILURE);
                }
                i++;
            }
        } else if (strcmp(argv[i], "-m") == 0) {
            // Encoding mode
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "plain") == 0) {
                    encodingMode = HIPSZ_MODE_PLAIN;
                } else if (strcmp(argv[i + 1], "outlier") == 0) {
                    encodingMode = HIPSZ_MODE_OUTLIER;
                } else if (strcmp(argv[i + 1], "fixed") == 0) {
                    encodingMode = HIPSZ_MODE_FIXED;
                } else {
                    printf("Error: Unsupported encoding mode. Use 'fixed', 'plain', or 'outlier'.\n");
                    exit(EXIT_FAILURE);
                }
                i++;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            // Dimensions and processing manner
            if (i + 1 < argc) {
                processingDim = (hipsz_dim_t)atoi(argv[i + 1]);
                if (processingDim == 2 || processingDim == 3) {
                    if (i + 4 >= argc) {
                        printf("Error: Missing dimension sizes for 2D/3D processing.\n");
                        exit(EXIT_FAILURE);
                    }
                    // dim_x is fastest, but stored in dims.x
                    dims.z = (unsigned int)atoi(argv[i + 2]);
                    dims.y = (unsigned int)atoi(argv[i + 3]);
                    dims.x = (unsigned int)atoi(argv[i + 4]);
                    i += 4;
                } else if (processingDim == 1) {
                    // Nothing else needed
                    i += 1;
                } else {
                    printf("Error: Unsupported processing dimension. Use 1, 2, or 3.\n");
                    exit(EXIT_FAILURE);
                }
            }
        } else if (strcmp(argv[i], "-eb") == 0) {
            // Error bound mode and value
            if (i + 2 < argc) {
                strncpy(errorBoundMode, argv[i + 1], sizeof(errorBoundMode) - 1);
                if (strcmp(errorBoundMode, "rel") != 0 && strcmp(errorBoundMode, "abs") != 0) {
                    printf("Error: Unsupported error bound mode. Use 'rel' or 'abs'.\n");
                    exit(EXIT_FAILURE);
                }
                errorBound = atof(argv[i + 2]);
                i += 2;
            }
        } else if (strcmp(argv[i], "-x") == 0) {
            // Compressed file path (optional)
            if (i + 1 < argc) {
                strncpy(cmpFilePath, argv[i + 1], sizeof(cmpFilePath) - 1);
                i++;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            // Decompressed file path (optional)
            if (i + 1 < argc) {
                strncpy(decFilePath, argv[i + 1], sizeof(decFilePath) - 1);
                i++;
            }
        } else {
            printf("Error: Unrecognized option '%s'.\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Ensure required arguments are provided
    if (strlen(oriFilePath) == 0 || errorBound == 0.0) {
        printf("Error: Missing required arguments.\n");
        exit(EXIT_FAILURE);
    }

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    void* oriData = NULL;
    void* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    int status=0;
    if(dataType == HIPSZ_TYPE_FLOAT) {
        oriData = (void*)readFloatData_Yafan(oriFilePath, &nbEle, &status);
        decData = (void*)malloc(nbEle*sizeof(float));
        cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));
        if(processingDim != 1 && nbEle != (size_t)dims.x * (size_t)dims.y * (size_t)dims.z) {
            fprintf(stderr, "Error: The number of elements in the original data does not match the dimensions\n");
            return 1;
        }
        
        // Value range calculation f32
        if(strcmp(errorBoundMode, "rel") == 0) {
            float* oriData_f32 = (float*)oriData;
            float max_val = oriData_f32[0];
            float min_val = oriData_f32[0];
            for(size_t i=0; i<nbEle; i++) {
                if(oriData_f32[i]>max_val)
                    max_val = oriData_f32[i];
                else if(oriData_f32[i]<min_val)
                    min_val = oriData_f32[i];
            }
            errorBound = errorBound * (max_val - min_val);
        }
    }
    else if(dataType == HIPSZ_TYPE_DOUBLE) {
        oriData = (void*)readDoubleData_Yafan(oriFilePath, &nbEle, &status);
        decData = (void*)malloc(nbEle*sizeof(double));
        cmpBytes = (unsigned char*)malloc(nbEle*sizeof(double));
        if(processingDim != 1 && nbEle != (size_t)dims.x * (size_t)dims.y * (size_t)dims.z) {
            fprintf(stderr, "Error: The number of elements in the original data does not match the dimensions\n");
            return 1;
        }

        // Value range calculation f64
        if(strcmp(errorBoundMode, "rel") == 0) {
            double* oriData_f64 = (double*)oriData;
            double max_val = oriData_f64[0];
            double min_val = oriData_f64[0];
            for(size_t i=0; i<nbEle; i++) {
                if(oriData_f64[i]>max_val)
                    max_val = oriData_f64[i];
                else if(oriData_f64[i]<min_val)
                    min_val = oriData_f64[i];
            }
            errorBound = errorBound * (float)(max_val - min_val);
        }
    }

    // Input data preparation on GPU.
    void* d_oriData;
    void* d_decData;
    unsigned char* d_cmpBytes;
    if(dataType == HIPSZ_TYPE_FLOAT) {
        hipMalloc((void**)&d_oriData, sizeof(float)*nbEle);
        hipMemcpy(d_oriData, oriData, sizeof(float)*nbEle, hipMemcpyHostToDevice);
        hipMalloc((void**)&d_decData, sizeof(float)*nbEle);
        hipMalloc((void**)&d_cmpBytes, sizeof(float)*nbEle);
        hipMemset(d_decData, 0, sizeof(float)*nbEle); // Initialize to 0 array in decompressed data.
    }
    else if(dataType == HIPSZ_TYPE_DOUBLE) {
        hipMalloc((void**)&d_oriData, sizeof(double)*nbEle);
        hipMemcpy(d_oriData, oriData, sizeof(double)*nbEle, hipMemcpyHostToDevice);
        hipMalloc((void**)&d_decData, sizeof(double)*nbEle);
        hipMalloc((void**)&d_cmpBytes, sizeof(double)*nbEle);
        hipMemset(d_decData, 0, sizeof(double)*nbEle); // Initialize to 0 array in decompressed data.
    }

    // Initializing HIP Stream.
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    // Warmup for NVIDIA GPU.
    for(int i=0; i<10; i++) {
        hipSZ_compress_ex(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, processingDim, dims, dataType, encodingMode, stream);
    }

    // hipSZ compression.
    timer_GPU.StartCounter(stream); // set timer
    hipSZ_compress_ex(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, processingDim, dims, dataType, encodingMode, stream);
    float cmpTime = timer_GPU.GetCounter(stream);

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    hipMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), hipMemcpyDeviceToHost);
    if(dataType == HIPSZ_TYPE_FLOAT) {
        hipMemset(d_cmpBytes, 0, sizeof(float)*nbEle); // set to zero for double check.
    }
    else if(dataType == HIPSZ_TYPE_DOUBLE) {
        hipMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    }
    hipMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), hipMemcpyHostToDevice);

    // hipSZ decompression.
    timer_GPU.StartCounter(stream); // set timer
    hipSZ_decompress_ex(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, processingDim, dims, dataType, encodingMode, stream);
    float decTime = timer_GPU.GetCounter(stream);

    // Print result.
    printf("hipSZ finished!\n");
    if(dataType == HIPSZ_TYPE_FLOAT) {
        printf("hipSZ compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/cmpTime);
        printf("hipSZ decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/decTime);
        printf("hipSZ compression ratio: %f\n\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    }
    else if(dataType == HIPSZ_TYPE_DOUBLE) {
        printf("hipSZ compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
        printf("hipSZ decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
        printf("hipSZ compression ratio: %f\n\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    }

    // Error check
    if(dataType == HIPSZ_TYPE_FLOAT) {
        hipMemcpy(decData, d_decData, sizeof(float)*nbEle, hipMemcpyDeviceToHost);
    }
    else if(dataType == HIPSZ_TYPE_DOUBLE) {
        hipMemcpy(decData, d_decData, sizeof(double)*nbEle, hipMemcpyDeviceToHost);
    }
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++) {
        if(dataType == HIPSZ_TYPE_FLOAT) {
            float* oriData_f32 = (float*)oriData;
            float* decData_f32 = (float*)decData;
            if(!std::isfinite(decData_f32[i]) || fabs(oriData_f32[i]-decData_f32[i]) > errorBound*1.1) {
                not_bound++;
                // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, ((float*)oriData)[i], ((float*)decData)[i], fabs(((float*)oriData)[i]-((float*)decData)[i]), errorBound);
            }
        }
        else if(dataType == HIPSZ_TYPE_DOUBLE) {
            double* oriData_f64 = (double*)oriData;
            double* decData_f64 = (double*)decData;
            if(!std::isfinite(decData_f64[i]) || fabs(oriData_f64[i]-decData_f64[i]) > errorBound*1.1) {
                not_bound++;
                // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, ((double*)oriData)[i], ((double*)decData)[i], fabs(((double*)oriData)[i]-((double*)decData)[i]), errorBound);
            }
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);

    // Write compressed or reconstructed files if needed.
    if (strlen(cmpFilePath) > 0) {
        hipMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), hipMemcpyDeviceToHost);
        writeByteData_Yafan(cmpBytes, cmpSize, cmpFilePath, &status);
    }
    if (strlen(decFilePath) > 0) {
        if(dataType == HIPSZ_TYPE_FLOAT) {
            writeFloatData_inBytes_Yafan((float*)decData, nbEle, decFilePath, &status);
        }
        else if(dataType == HIPSZ_TYPE_DOUBLE) {
            writeDoubleData_inBytes_Yafan((double*)decData, nbEle, decFilePath, &status);
        }
    }

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    hipFree(d_oriData);
    hipFree(d_decData);
    hipFree(d_cmpBytes);
    hipStreamDestroy(stream);

    return 0;
}
