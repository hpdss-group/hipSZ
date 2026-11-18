// Regression coverage for the pre-merge hipSZ public API.
//
// The main README documents hipSZ_compress/hipSZ_decompress and the
// type-specific hipSZ_{compress,decompress}_{plain,outlier}_{f32,f64}
// entry points. The merged engine is dimension-aware, so this test proves
// the legacy 1D names still compile and round-trip within the error bound,
// and additionally smokes the new hipSZ_compress_ex 2D path.
#include <hipSZ.h>
#include "hipSZ_check.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::vector<float> make_data(size_t n)
{
    std::vector<float> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = (float)(sin(i * 0.001) * 100.0 + 0.5 * cos(i * 0.031));
    return data;
}

int verify(const std::vector<float>& ori, const std::vector<float>& dec,
           double eb, const char* label)
{
    int bad = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < ori.size(); i++) {
        double err = fabs((double)ori[i] - (double)dec[i]);
        if (!std::isfinite((double)dec[i]) || !(err <= eb * 1.1)) bad++;
        if (err > max_err) max_err = err;
    }
    if (bad) {
        printf("  FAIL %s: %d bad values (max_err=%.6g eb=%.3g)\n",
               label, bad, max_err, eb);
        return 1;
    }
    printf("  Pass error check! (%s, max_err=%.6g)\n", label, max_err);
    return 0;
}

} // namespace

int main()
{
    const size_t nbEle = 1024 * 1024; // 1M f32, small enough for CI
    const float eb = 1e-2f;
    const size_t cmpCap = nbEle * sizeof(float) + 16384;
    const std::vector<float> ori = make_data(nbEle);

    float* d_ori;
    float* d_dec;
    unsigned char* d_cmp;
    HIP_CHECK(hipMalloc((void**)&d_ori, nbEle * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_dec, nbEle * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_cmp, cmpCap));
    HIP_CHECK(hipMemcpy(d_ori, ori.data(), nbEle * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int failures = 0;
    std::vector<float> dec(nbEle);

    // 1) Documented generic API: hipSZ_compress/hipSZ_decompress.
    {
        size_t cmpSize = 0;
        hipSZ_compress(d_ori, d_cmp, nbEle, &cmpSize, eb, HIPSZ_TYPE_FLOAT,
                       HIPSZ_MODE_PLAIN, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        if (cmpSize == 0 || cmpSize > cmpCap) {
            printf("  FAIL generic API: bogus cmpSize=%zu\n", cmpSize);
            failures++;
        } else {
            // Round-trip the compressed bytes through the host to catch a
            // truncated payload, mirroring the other suites.
            std::vector<unsigned char> cmp(cmpSize);
            HIP_CHECK(hipMemcpy(cmp.data(), d_cmp, cmpSize, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemset(d_cmp, 0, cmpCap));
            HIP_CHECK(hipMemcpy(d_cmp, cmp.data(), cmpSize, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_dec, 0, nbEle * sizeof(float)));
            hipSZ_decompress(d_dec, d_cmp, nbEle, cmpSize, eb, HIPSZ_TYPE_FLOAT,
                             HIPSZ_MODE_PLAIN, stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpy(dec.data(), d_dec, nbEle * sizeof(float),
                                hipMemcpyDeviceToHost));
            failures += verify(ori, dec, eb, "generic 1D plain f32");
        }
    }

    // 2) Documented type-specific API.
    {
        size_t cmpSize = 0;
        hipSZ_compress_plain_f32(d_ori, d_cmp, nbEle, &cmpSize, eb, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        hipSZ_decompress_plain_f32(d_dec, d_cmp, nbEle, cmpSize, eb, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(dec.data(), d_dec, nbEle * sizeof(float),
                            hipMemcpyDeviceToHost));
        failures += verify(ori, dec, eb, "type-specific 1D plain f32");
    }

    // 3) Extended API, 2D path.
    {
        const uint3 dims = make_uint3(1024, 1024, 1); // x fastest
        size_t cmpSize = 0;
        hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, eb, HIPSZ_DIM_2D, dims,
                          HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemset(d_dec, 0, nbEle * sizeof(float)));
        hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, eb, HIPSZ_DIM_2D, dims,
                            HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(dec.data(), d_dec, nbEle * sizeof(float),
                            hipMemcpyDeviceToHost));
        failures += verify(ori, dec, eb, "_ex 2D plain f32");
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_ori));
    HIP_CHECK(hipFree(d_dec));
    HIP_CHECK(hipFree(d_cmp));

    if (failures) {
        printf("\033[0;31mhipSZ_compat_test FAILED (%d)\033[0m\n", failures);
        return 1;
    }
    printf("\033[0;32mhipSZ_compat_test PASSED\033[0m\n");
    return 0;
}
