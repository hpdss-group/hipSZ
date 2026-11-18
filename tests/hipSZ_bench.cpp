// Dual-metric benchmark for hipSZ on gfx936.
//
// For each (type, dim, mode, size) cell: warmup, then N timed repetitions.
// Two metrics per call:
//   kernel_ms — HIP events on the execution stream (kernel only)
//   api_ms    — host wall clock around the entry point (scratch alloc +
//               memset + launch + compressed-size readback + sync)
// Raw samples are written as CSV to stdout (one row per repetition) and a
// median summary to stderr, so the raw data can be archived as-is.
//
// Data: smooth synthetic (sine + secondary cosine), the same shape the
// upstream 512M tests use, standing in for locally-smooth scientific data.
#include <hipSZ.h>
#include "hipSZ_check.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T>
static void make_data(std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); i++)
        data[i] = (T)(sin(i * 0.001) * 100.0 + 0.5 * cos(i * 0.031));
}

template <typename T>
static void run_cell(hipsz_type_t type, hipsz_dim_t dim, uint3 dims,
                     hipsz_mode_t mode, size_t nbEle, double eb,
                     int warmup, int reps, const char* tag, FILE* csv)
{
    std::vector<T> ori(nbEle);
    make_data(ori);

    T *d_ori, *d_dec;
    unsigned char* d_cmp;
    HIP_CHECK(hipMalloc((void**)&d_ori, nbEle * sizeof(T)));
    HIP_CHECK(hipMalloc((void**)&d_dec, nbEle * sizeof(T)));
    HIP_CHECK(hipMalloc((void**)&d_cmp, nbEle * sizeof(T) + 16384));
    HIP_CHECK(hipMemcpy(d_ori, ori.data(), nbEle * sizeof(T), hipMemcpyHostToDevice));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    TimingGPU timer;
    WallTimer wall;
    const char* mode_name = mode == HIPSZ_MODE_FIXED ? "fixed"
                          : mode == HIPSZ_MODE_PLAIN ? "plain" : "outlier";

    size_t cmpSize = 0;
    for (int r = -warmup; r < reps; r++) {
        wall.start();
        timer.StartCounter(stream);
        hipSZ_compress_ex(d_ori, d_cmp, nbEle, &cmpSize, (float)eb, dim, dims, type, mode, stream);
        float cmp_kernel_ms = timer.GetCounter(stream);
        double cmp_api_ms = wall.ms();
        HIP_CHECK(hipGetLastError());

        wall.start();
        timer.StartCounter(stream);
        hipSZ_decompress_ex(d_dec, d_cmp, nbEle, cmpSize, (float)eb, dim, dims, type, mode, stream);
        float dec_kernel_ms = timer.GetCounter(stream);
        double dec_api_ms = wall.ms();
        HIP_CHECK(hipGetLastError());

        if (r >= 0)
            fprintf(csv, "%s,%s,%zu,%d,%.4f,%.4f,%.4f,%.4f\n",
                    tag, mode_name, nbEle, r,
                    cmp_kernel_ms, cmp_api_ms, dec_kernel_ms, dec_api_ms);
        fflush(csv);
    }

    // verify once
    std::vector<T> dec(nbEle);
    HIP_CHECK(hipMemcpy(dec.data(), d_dec, nbEle * sizeof(T), hipMemcpyDeviceToHost));
    int bad = 0;
    double max_err = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)ori[i] - (double)dec[i]);
        if (!std::isfinite((double)dec[i]) || err > eb * 1.1) bad++;
        max_err = std::max(max_err, err);
    }

    printf("%-28s %-8s n=%-10zu ratio=%7.3f max_err=%.3g %s\n", tag, mode_name,
           nbEle, (double)(nbEle * sizeof(T)) / (double)cmpSize, max_err,
           bad ? "BAD!" : "ok");
    if (bad) exit(1);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_ori));
    HIP_CHECK(hipFree(d_dec));
    HIP_CHECK(hipFree(d_cmp));
}

int main(int argc, char** argv)
{
    const int warmup = argc > 1 ? atoi(argv[1]) : 3;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const double eb = 1e-2; // value range is ~200; equivalent to REL 1e-2 upstream
    FILE* csv = stdout;
    fprintf(csv, "cell,mode,nbEle,rep,cmp_kernel_ms,cmp_api_ms,dec_kernel_ms,dec_api_ms\n");

    // size x dim matrix (f32)
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                    HIPSZ_MODE_FIXED, (size_t)16 << 20, eb, warmup, reps, "f32-1D-16M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                    HIPSZ_MODE_PLAIN, (size_t)16 << 20, eb, warmup, reps, "f32-1D-16M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                    HIPSZ_MODE_OUTLIER, (size_t)16 << 20, eb, warmup, reps, "f32-1D-16M", csv);

    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                    HIPSZ_MODE_PLAIN, (size_t)128 << 20, eb, warmup, reps, "f32-1D-128M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_2D, make_uint3(4096, 4096, 1),
                    HIPSZ_MODE_PLAIN, (size_t)16 << 20, eb, warmup, reps, "f32-2D-4096x4096", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_2D, make_uint3(8192, 16384, 1),
                    HIPSZ_MODE_PLAIN, (size_t)128 << 20, eb, warmup, reps, "f32-2D-128M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_3D, make_uint3(512, 512, 512),
                    HIPSZ_MODE_PLAIN, (size_t)128 << 20, eb, warmup, reps, "f32-3D-128M", csv);

    // 512M cells (one per dim, plain + one outlier)
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                    HIPSZ_MODE_PLAIN, (size_t)512 << 20, eb, warmup, reps, "f32-1D-512M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_2D, make_uint3(16384, 32768, 1),
                    HIPSZ_MODE_PLAIN, (size_t)512 << 20, eb, warmup, reps, "f32-2D-512M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_3D, make_uint3(1024, 1024, 512),
                    HIPSZ_MODE_PLAIN, (size_t)512 << 20, eb, warmup, reps, "f32-3D-512M", csv);
    run_cell<float>(HIPSZ_TYPE_FLOAT, HIPSZ_DIM_3D, make_uint3(1024, 1024, 512),
                    HIPSZ_MODE_OUTLIER, (size_t)512 << 20, eb, warmup, reps, "f32-3D-512M", csv);

    // f64 representative
    run_cell<double>(HIPSZ_TYPE_DOUBLE, HIPSZ_DIM_1D, make_uint3(0, 0, 0),
                     HIPSZ_MODE_PLAIN, (size_t)128 << 20, eb, warmup, reps, "f64-1D-128M", csv);
    run_cell<double>(HIPSZ_TYPE_DOUBLE, HIPSZ_DIM_3D, make_uint3(512, 512, 512),
                     HIPSZ_MODE_PLAIN, (size_t)128 << 20, eb, warmup, reps, "f64-3D-128M", csv);

    return 0;
}
