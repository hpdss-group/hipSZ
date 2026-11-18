// Boundary regression suite for hipSZ on gfx936.
//
// The upstream 512M-element tests only exercise large regular grids. This
// suite targets the small and irregular corner cases: single-CTA grids
// (<= 32768 elements for every dimension mode), the 1-CTA/2-CTA boundary,
// non-divisible lengths, non-regular 2D/3D shapes, degenerate data
// (constant / all-zero / random / outliers), small error bounds, and the
// documented empty-input rejection.
//
// A case fails when any decompressed value is non-finite or exceeds
// errorBound*1.1 (the upstream-compatible acceptance rule). The count of
// values exceeding the user's original errorBound is reported separately
// as diagnostic data, not treated as failure.
#include <hipSZ.h>
#include "hipSZ_check.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static unsigned long long g_cases = 0;
static unsigned long long g_failures = 0;
static double g_worst_ratio = 0.0; // max_err / (eb*1.1) over all cases

enum Pattern { PAT_CONSTANT, PAT_ZEROS, PAT_RANDOM, PAT_OUTLIER, PAT_SINE };

template <typename T>
static void fill_pattern(std::vector<T>& data, Pattern pat, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < data.size(); i++) {
        T v;
        switch (pat) {
        case PAT_CONSTANT: v = (T)42.0; break;
        case PAT_ZEROS:    v = (T)0.0;  break;
        case PAT_RANDOM:
            v = (T)((rand() / (double)RAND_MAX) * 200.0 - 100.0);
            break;
        case PAT_OUTLIER:
            v = (T)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
            if (i % 97 == 0) v *= (T)100.0;
            break;
        default: v = (T)(sin(i * 0.001) * 100.0); break;
        }
        data[i] = v;
    }
}

template <typename T>
static int verify(const std::vector<T>& ori, const std::vector<T>& dec,
                  double eb, const char* label)
{
    int fail = 0, over_user_eb = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < ori.size(); i++) {
        double err = fabs((double)ori[i] - (double)dec[i]);
        if (!std::isfinite((double)dec[i]) || !(err <= eb * 1.1)) fail++;
        if (err > eb) over_user_eb++;
        if (err > max_err) max_err = err;
    }
    double ratio = max_err / (eb * 1.1);
    if (ratio > g_worst_ratio) g_worst_ratio = ratio;
    g_cases++;
    if (fail) {
        g_failures++;
        printf("  \033[0;31mFAIL\033[0m %s: %d bad values (max_err=%.6g eb=%.3g, %d over user eb)\n",
               label, fail, max_err, eb, over_user_eb);
        return 1;
    }
    (void)over_user_eb; // kept for the failure path printout above
    return 0;
}

template <typename T>
static void run_case(size_t nbEle, uint3 dims, hipsz_dim_t dim, hipsz_type_t type,
                     hipsz_mode_t mode, double eb, Pattern pat, const char* label)
{
    std::vector<T> ori(nbEle), dec(nbEle);
    fill_pattern(ori, pat, (unsigned)(nbEle + (int)mode + (int)pat));

    T *d_ori, *d_dec;
    unsigned char* d_cmp;
    const size_t cmpCap = nbEle * sizeof(T) + 16384; // worst-case overhead for tiny inputs
    HIP_CHECK(hipMalloc((void**)&d_ori, nbEle * sizeof(T)));
    HIP_CHECK(hipMalloc((void**)&d_dec, nbEle * sizeof(T)));
    HIP_CHECK(hipMalloc((void**)&d_cmp, cmpCap));
    HIP_CHECK(hipMemcpy(d_ori, ori.data(), nbEle * sizeof(T), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    size_t cmpSize = 0;
    hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, (float)eb, dim, dims, type, mode, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());
    if (cmpSize == 0 || cmpSize > cmpCap) {
        g_failures++;
        printf("  \033[0;31mFAIL\033[0m %s: bogus cmpSize=%zu\n", label, cmpSize);
    } else {
        // Round-trip through the host with exactly cmpSize bytes: a kernel
        // that underreports the compressed size truncates the payload here
        // and the decompression check below catches it.
        std::vector<unsigned char> cmp(cmpSize);
        HIP_CHECK(hipMemcpy(cmp.data(), d_cmp, cmpSize, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemset(d_cmp, 0, nbEle * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_cmp, cmp.data(), cmpSize, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_dec, 0, nbEle * sizeof(T)));

        hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, (float)eb, dim, dims, type, mode, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(dec.data(), d_dec, nbEle * sizeof(T), hipMemcpyDeviceToHost));
        verify(ori, dec, eb, label);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_ori));
    HIP_CHECK(hipFree(d_dec));
    HIP_CHECK(hipFree(d_cmp));
}

template <typename T>
static void run_matrix(hipsz_type_t type)
{
    const size_t sizes1D[] = {1, 31, 32, 33, 63, 64, 1000, 8191, 8192, 8193,
                           16383, 16384, 16385, 32767, 32768, 32769, 65536, 100000};
    // (y, x): 168x168 and 4096x8 are single-CTA (<=512 8x8 blocks); 184x184
    // and 4097x8 cross into two CTAs. x/y not multiples of 8 hit the
    // partial-block paths.
    const struct { unsigned y, x; } sizes2D[] = {
        {1, 1}, {8, 8}, {8, 9}, {9, 8}, {16, 16}, {64, 64}, {128, 130},
        {168, 168}, {184, 184}, {4096, 8}, {4097, 8}};
    // (z, y, x): 32x32x32 == 512 4x4x4 blocks == single CTA exactly.
    const struct { unsigned z, y, x; } sizes3D[] = {
        {4, 4, 4}, {8, 8, 16}, {32, 32, 32}, {33, 32, 32}, {16, 17, 33}, {64, 64, 64}};
    const struct { Pattern pat; const char* name; } patterns[] = {
        {PAT_CONSTANT, "const"}, {PAT_ZEROS, "zeros"}, {PAT_RANDOM, "rand"},
        {PAT_OUTLIER, "outlier"}, {PAT_SINE, "sine"}};
    const double ebs[] = {1e-2, 1e-4};
    const struct { hipsz_mode_t m; const char* name; } modes[] = {
        {HIPSZ_MODE_FIXED, "fixed"}, {HIPSZ_MODE_PLAIN, "plain"}, {HIPSZ_MODE_OUTLIER, "outlier"}};

    char label[128];
    for (size_t si = 0; si < sizeof(sizes1D) / sizeof(sizes1D[0]); si++)
        for (int pi = 0; pi < 5; pi++)
            for (int ei = 0; ei < 2; ei++)
                for (int mi = 0; mi < 3; mi++) {
                    snprintf(label, sizeof(label), "1D n=%zu %s eb=%g %s",
                             sizes1D[si], patterns[pi].name, ebs[ei], modes[mi].name);
                    run_case<T>(sizes1D[si], make_uint3(0, 0, 0),
                                HIPSZ_DIM_1D, type, modes[mi].m, ebs[ei], patterns[pi].pat, label);
                }

    for (int si = 0; si < 11; si++)
        for (int pi = 0; pi < 5; pi++)
            for (int ei = 0; ei < 2; ei++)
                for (int mi = 0; mi < 3; mi++) {
                    unsigned y = sizes2D[si].y, x = sizes2D[si].x;
                    snprintf(label, sizeof(label), "2D %ux%u %s eb=%g %s",
                             y, x, patterns[pi].name, ebs[ei], modes[mi].name);
                    run_case<T>((size_t)y * x, make_uint3(x, y, 1),
                                HIPSZ_DIM_2D, type, modes[mi].m, ebs[ei], patterns[pi].pat, label);
                }

    for (int si = 0; si < 6; si++)
        for (int pi = 0; pi < 5; pi++)
            for (int ei = 0; ei < 2; ei++)
                for (int mi = 0; mi < 3; mi++) {
                    unsigned z = sizes3D[si].z, y = sizes3D[si].y, x = sizes3D[si].x;
                    snprintf(label, sizeof(label), "3D %ux%ux%u %s eb=%g %s",
                             z, y, x, patterns[pi].name, ebs[ei], modes[mi].name);
                    run_case<T>((size_t)z * y * x, make_uint3(x, y, z),
                                HIPSZ_DIM_3D, type, modes[mi].m, ebs[ei], patterns[pi].pat, label);
                }
}

int main()
{
    // Documented capacity contract: empty input is rejected with cmpSize = 0.
    {
        float* d_buf;
        HIP_CHECK(hipMalloc((void**)&d_buf, 16));
        size_t cmpSize = 1;
        hipSZ_compress_ex(d_buf, (unsigned char*)d_buf, 0, &cmpSize, 1e-3f, HIPSZ_DIM_1D,
                        make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, 0);
        HIP_CHECK(hipDeviceSynchronize());
        if (cmpSize != 0) {
            printf("  \033[0;31mFAIL\033[0m empty input not rejected (cmpSize=%zu)\n", cmpSize);
            g_failures++;
        }
        HIP_CHECK(hipFree(d_buf));
    }

    printf("Running f32 boundary matrix...\n");
    run_matrix<float>(HIPSZ_TYPE_FLOAT);
    printf("Running f64 boundary matrix...\n");
    run_matrix<double>(HIPSZ_TYPE_DOUBLE);

    printf("\nedge suite: %llu cases, %llu failures, worst max_err/(1.1*eb)=%.4g\n",
           g_cases, g_failures, g_worst_ratio);
    if (g_failures) {
        printf("\033[0;31mhipSZ_test_edge FAILED\033[0m\n");
        return 1;
    }
    printf("\033[0;32mhipSZ_test_edge PASSED\033[0m\n");
    return 0;
}
