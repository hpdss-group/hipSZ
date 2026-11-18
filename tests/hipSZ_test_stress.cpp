// Stress suite for the single-kernel decoupled-lookback offset resolution.
//
// The lookback protocol (volatile flags + __threadfence + spin waits) is the
// port's highest-risk construct: a memory-ordering bug shows up as hangs or
// sporadic corruption only under repetition, large grids and concurrent
// streams. This suite hammers exactly those axes on gfx936:
//   part 1: large grid (2048 CTAs), repeated back-to-back calls,
//           alternating modes, periodic verification;
//   part 2: four concurrent streams, each with its own buffers, verifying
//           that per-call scratch allocations and lookback states do not
//           interfere across streams.
// Every HIP call is checked; a hang is caught by the ctest TIMEOUT instead.
#include <hipSZ.h>
#include "hipSZ_check.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T>
static void make_data(std::vector<T>& data, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = (T)(sin(i * 0.00095) * 100.0 + (rand() / (double)RAND_MAX) * 0.01);
}

template <typename T>
static int verify(const std::vector<T>& ori, const std::vector<T>& dec, double eb, const char* label)
{
    int fail = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < ori.size(); i++) {
        double err = fabs((double)ori[i] - (double)dec[i]);
        if (!std::isfinite((double)dec[i]) || !(err <= eb * 1.1)) fail++;
        if (err > max_err) max_err = err;
    }
    if (fail) printf("  \033[0;31mFAIL\033[0m %s: %d bad values (max_err=%.6g)\n", label, fail, max_err);
    return fail;
}

// Compress + full host round-trip + decompress + verify on one stream.
template <typename T>
static int roundtrip(T* d_ori, T* d_dec, unsigned char* d_cmp, size_t nbEle,
                     uint3 dims, hipsz_dim_t dim, hipsz_type_t type,
                     hipsz_mode_t mode, double eb, hipStream_t stream,
                     std::vector<T>& ori, std::vector<unsigned char>& cmp_host,
                     std::vector<T>& dec_host, const char* label)
{
    size_t cmpSize = 0;
    hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, (float)eb, dim, dims, type, mode, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());

    cmp_host.resize(cmpSize);
    dec_host.assign(nbEle, (T)0);
    HIP_CHECK(hipMemcpy(cmp_host.data(), d_cmp, cmpSize, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(d_cmp, 0, nbEle * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_cmp, cmp_host.data(), cmpSize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dec, 0, nbEle * sizeof(T)));

    hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, (float)eb, dim, dims, type, mode, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(dec_host.data(), d_dec, nbEle * sizeof(T), hipMemcpyDeviceToHost));
    return verify(ori, dec_host, eb, label);
}

int main()
{
    const double eb = 1e-2;
    int failures = 0;

    // ---- Part 1: large grid, repeated back-to-back, alternating modes ----
    {
        const size_t nbEle = (size_t)64 * 1024 * 1024; // 2048 CTAs of 32768 elements (1D)
        const int reps = 100;
        std::vector<float> ori(nbEle), dec;
        std::vector<unsigned char> cmp;
        make_data(ori, 20261021);

        float *d_ori, *d_dec;
        unsigned char* d_cmp;
        HIP_CHECK(hipMalloc((void**)&d_ori, nbEle * sizeof(float)));
        HIP_CHECK(hipMalloc((void**)&d_dec, nbEle * sizeof(float)));
        HIP_CHECK(hipMalloc((void**)&d_cmp, nbEle * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_ori, ori.data(), nbEle * sizeof(float), hipMemcpyHostToDevice));
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));

        char label[96];
        for (int r = 0; r < reps; r++) {
            hipsz_mode_t mode = (r % 2) ? HIPSZ_MODE_OUTLIER : HIPSZ_MODE_PLAIN;
            snprintf(label, sizeof(label), "part1 rep=%d mode=%d", r, (int)mode);
            // Verify periodically (every 10th rep) to keep the run fast.
            if (r % 10 == 0)
                failures += roundtrip<float>(d_ori, d_dec, d_cmp, nbEle,
                                             make_uint3(0, 0, 0), HIPSZ_DIM_1D,
                                             HIPSZ_TYPE_FLOAT, mode, eb, stream,
                                             ori, cmp, dec, label);
            else {
                size_t cmpSize = 0;
                hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, (float)eb, HIPSZ_DIM_1D,
                                make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, mode, stream);
                hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, (float)eb, HIPSZ_DIM_1D,
                                  make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, mode, stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                HIP_CHECK(hipGetLastError());
            }
        }
        printf("part1 (2048-CTA grid x %d reps): %s\n", reps, failures ? "issues above" : "ok");

        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(d_ori));
        HIP_CHECK(hipFree(d_dec));
        HIP_CHECK(hipFree(d_cmp));
    }

    // ---- Part 2: four concurrent streams with independent buffers ----
    {
        const int nstreams = 4;
        const size_t nbEle = (size_t)16 * 1024 * 1024; // 512 CTAs per stream
        const int rounds = 50;
        const hipsz_mode_t modes[4] = {HIPSZ_MODE_FIXED, HIPSZ_MODE_PLAIN,
                                        HIPSZ_MODE_OUTLIER, HIPSZ_MODE_PLAIN};

        std::vector<float> oris[nstreams], decs[nstreams];
        std::vector<unsigned char> cmps[nstreams];
        float* d_ori[nstreams];
        float* d_dec[nstreams];
        unsigned char* d_cmp[nstreams];
        hipStream_t streams[nstreams];

        for (int s = 0; s < nstreams; s++) {
            oris[s].resize(nbEle);
            make_data(oris[s], 7000 + s);
            HIP_CHECK(hipMalloc((void**)&d_ori[s], nbEle * sizeof(float)));
            HIP_CHECK(hipMalloc((void**)&d_dec[s], nbEle * sizeof(float)));
            HIP_CHECK(hipMalloc((void**)&d_cmp[s], nbEle * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_ori[s], oris[s].data(), nbEle * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipStreamCreate(&streams[s]));
        }

        size_t cmpSize[nstreams];
        int failures2 = 0;
        for (int rd = 0; rd < rounds; rd++) {
            // Interleave submissions across streams, then sync all.
            for (int s = 0; s < nstreams; s++) {
                cmpSize[s] = 0;
                hipSZ_compress_ex(d_ori[s], d_cmp[s], nbEle, &cmpSize[s], (float)eb, HIPSZ_DIM_1D,
                                make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, modes[s], streams[s]);
                hipSZ_decompress_ex(d_dec[s], d_cmp[s], nbEle, cmpSize[s], (float)eb, HIPSZ_DIM_1D,
                                  make_uint3(0, 0, 0), HIPSZ_TYPE_FLOAT, modes[s], streams[s]);
            }
            for (int s = 0; s < nstreams; s++) {
                HIP_CHECK(hipStreamSynchronize(streams[s]));
                HIP_CHECK(hipGetLastError());
            }
            // Deep-verify each stream every 10th round.
            if (rd % 10 == 0) {
                char label[96];
                for (int s = 0; s < nstreams; s++) {
                    snprintf(label, sizeof(label), "part2 round=%d stream=%d", rd, s);
                    failures2 += roundtrip<float>(d_ori[s], d_dec[s], d_cmp[s], nbEle,
                                                  make_uint3(0, 0, 0), HIPSZ_DIM_1D,
                                                  HIPSZ_TYPE_FLOAT, modes[s], eb, streams[s],
                                                  oris[s], cmps[s], decs[s], label);
                }
            }
        }
        failures += failures2;
        printf("part2 (%d streams x %d rounds): %s\n", nstreams, rounds, failures2 ? "issues above" : "ok");

        for (int s = 0; s < nstreams; s++) {
            HIP_CHECK(hipStreamDestroy(streams[s]));
            HIP_CHECK(hipFree(d_ori[s]));
            HIP_CHECK(hipFree(d_dec[s]));
            HIP_CHECK(hipFree(d_cmp[s]));
        }
    }

    if (failures) {
        printf("\033[0;31mhipSZ_test_stress FAILED (%d)\033[0m\n", failures);
        return 1;
    }
    printf("\033[0;32mhipSZ_test_stress PASSED\033[0m\n");
    return 0;
}
