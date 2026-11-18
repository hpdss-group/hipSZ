// Consumes an installed hipSZ exactly the way a downstream project would:
// include <hipSZ.h>, link the CMake-exported target, compress and
// decompress a small synthetic dataset, verify the error bound, and check
// the documented rejection of empty input.
#include <hipSZ.h>

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(expr)                                                     \
    do {                                                                    \
        hipError_t _s = (expr);                                             \
        if (_s != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n",                  \
                    hipGetErrorString(_s), __FILE__, __LINE__, #expr);      \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main()
{
    const size_t nbEle = 100000;
    const float eb = 1e-3f;

    float* h_ori = (float*)malloc(nbEle * sizeof(float));
    float* h_dec = (float*)malloc(nbEle * sizeof(float));
    for (size_t i = 0; i < nbEle; i++)
        h_ori[i] = sinf(i * 0.001f) * 100.0f + 0.5f * cosf(i * 0.031f);

    float* d_ori;
    float* d_dec;
    unsigned char* d_cmp;
    HIP_CHECK(hipMalloc((void**)&d_ori, nbEle * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_dec, nbEle * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_cmp, nbEle * sizeof(float) + 16384));
    HIP_CHECK(hipMemcpy(d_ori, h_ori, nbEle * sizeof(float), hipMemcpyHostToDevice));

    size_t cmpSize = 0;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, eb, HIPSZ_DIM_1D,
                    make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
    hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, eb, HIPSZ_DIM_1D,
                      make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(h_dec, d_dec, nbEle * sizeof(float), hipMemcpyDeviceToHost));

    int failures = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)h_ori[i] - (double)h_dec[i]);
        if (err > max_err) max_err = err;
        if (!(err <= (double)eb * 1.1) || !isfinite(h_dec[i])) failures++;
    }

    // Legacy documented API: the pre-merge hipSZ names must still compile
    // and produce a valid round trip through the compatibility layer.
    {
        size_t legacy_size = 0;
        hipSZ_compress(d_ori, d_cmp, nbEle, &legacy_size, eb,
                       HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
        hipSZ_decompress(d_dec, d_cmp, nbEle, legacy_size, eb,
                         HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(h_dec, d_dec, nbEle * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < nbEle; i++) {
            double err = fabs((double)h_ori[i] - (double)h_dec[i]);
            if (err > max_err) max_err = err;
            if (!(err <= (double)eb * 1.1) || !isfinite(h_dec[i])) failures++;
        }
        if (legacy_size == 0) {
            fprintf(stderr, "legacy API returned cmpSize=0\n");
            failures++;
        }
    }

    // Documented capacity contract: empty input is rejected with cmpSize = 0.
    size_t empty_size = 12345;
    hipSZ_compress_ex(d_ori, d_cmp, 0, &empty_size, eb, HIPSZ_DIM_1D,
                    make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (empty_size != 0) {
        fprintf(stderr, "empty input was not rejected (cmpSize=%zu)\n", empty_size);
        failures++;
    }

    printf("consumer: cmpSize=%zu max_err=%.6g failures=%d\n", cmpSize, max_err, failures);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_ori));
    HIP_CHECK(hipFree(d_dec));
    HIP_CHECK(hipFree(d_cmp));
    free(h_ori);
    free(h_dec);

    if (failures) {
        fprintf(stderr, "consumer FAILED\n");
        return 1;
    }
    printf("consumer PASSED\n");
    return 0;
}
