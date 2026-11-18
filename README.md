# hipSZ

**HIP-based error-bounded lossy compression for scientific data**

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

hipSZ compresses FP32 and FP64 scientific arrays on AMD GPUs and Hygon DCUs. It provides a command-line tool and a C++ API for data already in device memory. Both interfaces support three encoding modes (fixed, plain, and outlier) and three processing dimensions (1D, 2D, and 3D). The HIP implementation is based on [cuSZp](https://github.com/szcompressor/cuSZp) v3.0.0.

[Build](#build) · [Quick start](#quick-start) · [C++ API](#c-api) · [Hygon BW150 example](#hygon-bw150-example)

## Build

Use Linux with CMake 3.21 or later and a C++17 toolchain. Put `clang` and `hipcc` on your `PATH`.

| Hardware | Toolkit |
| --- | --- |
| Hygon DCU | DCU Toolkit (DTK) 26.04 or later; validated on BW150 (gfx936) |
| AMD GPU | ROCm 6.2 or later |

```bash
git clone https://github.com/opensdc-org/hipSZ.git
cd hipSZ
```

The build targets `gfx926` (Hygon K100) by default when `CMAKE_HIP_ARCHITECTURES` is not supplied. Pass an explicit target for your device, for example `gfx936` for a Hygon BW150 (used for the measurements below) or `gfx942` for an AMD MI300X. An explicit `CMAKE_HIP_ARCHITECTURES` always wins; `HIPSZ_HIP_ARCHITECTURES` is available as a project-level fallback.

Set `HIP_ROOT` to the prefix containing your HIP headers and CMake package. For a DCU, load the DTK environment and use its HIP installation prefix (for example `/opt/dtk-26.04`).

```bash
HIP_ROOT=/opt/rocm
export PATH="$HIP_ROOT/bin:$HIP_ROOT/llvm/bin:$PATH"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx936 \
  -DCMAKE_PREFIX_PATH="$HIP_ROOT" \
  -DCMAKE_CXX_FLAGS="-I$HIP_ROOT/include" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build --parallel --target install
```

The install tree contains the CLI, test, and benchmark executables in `install/bin/`, the hipSZ libraries in `install/lib/`, the public header `install/include/hipSZ.h`, and the CMake package under `install/cmake/`.

## Quick start

The CLI reads a raw, contiguous array of `f32` or `f64` values. For a quick round trip, generate a small FP32 input and run:

```bash
python3 - <<'PY'
from array import array
from math import sin

values = array("f", (sin(i * 0.001) for i in range(1 << 20)))
with open("sample.f32", "wb") as output:
    values.tofile(output)
PY

./install/bin/hipSZ -i sample.f32 -t f32 -m plain -eb abs 1e-3
```

The tool compresses the input, reconstructs it, and reports the compression ratio, throughput, and error check. The same 1M-element input can be processed as 2D or 3D; `dim_x` is the fastest-varying dimension:

```bash
./install/bin/hipSZ -i sample.f32 -t f32 -m plain -d 2 1 1024 1024 -eb abs 1e-3
./install/bin/hipSZ -i sample.f32 -t f32 -m fixed -d 3 1 1024 1024 -eb abs 1e-3
```

`-d` is optional and defaults to 1D, so commands written before the dimension-aware merge keep working.

| Option | Meaning |
| --- | --- |
| `-i PATH` | Raw floating-point input |
| `-t f32\|f64` | Input data type |
| `-m fixed\|plain\|outlier` | No-delta, delta, or delta-plus-outlier encoding |
| `-d DIM [Z Y X]` | Optional processing dimension: `1` (default), `2` (Z slices of Y×X), or `3` (Z×Y×X) |
| `-eb abs VALUE` | Absolute error bound |
| `-eb rel VALUE` | Range-relative error bound, converted to `VALUE × (max − min)` |
| `-x PATH` | Optional compressed output |
| `-o PATH` | Optional reconstructed output |

For `-d 2`, `Z` is the number of independent `Y×X` slices; for `-d 3`, the full volume is `Z×Y×X`. In both cases the shape is passed slowest-varying first, matching the command line above.

## Hygon BW150 example

These measurements were recorded on a single Hygon BW150 DCU (gfx936) with DTK 26.04, using the `CLDLIQ` field of the CESM-ATM dataset (`CLDLIQ_1_26_1800_3600.f32`, 26×1800×3600 FP32, 674 MB) and `-eb rel 1e-4`. The rates are 5-run medians of the values reported by the CLI for the timed compression and decompression calls; each run includes the CLI's warmup. All runs passed the reconstruction error check.

The dataset is available from [SDRBench](https://sdrbench.github.io/datasets.html) and [Hugging Face](https://huggingface.co/datasets/sdrbench/cesm-atm).

| Encoding | Compression (GB/s) | Decompression (GB/s) | Compression ratio |
| --- | ---: | ---: | ---: |
| Plain | 230.52 | 403.15 | 9.897× |
| Outlier | 189.79 | 346.95 | 17.362× |

The compression ratios match the cuSZp reference measurements for this field (9.90× and 17.36×). Both modes used the CLI's default 1D processing; the table records the current BW150 validation. The `gfx926` default remains available for Hygon K100; throughput should not be compared across GPU generations.

```bash
./install/bin/hipSZ -i CLDLIQ_1_26_1800_3600.f32 -t f32 -m plain -eb rel 1e-4
./install/bin/hipSZ -i CLDLIQ_1_26_1800_3600.f32 -t f32 -m outlier -eb rel 1e-4
```

The CLI labels the throughput field `GB/s`; its calculation uses 1024-based units, so the values are GiB/s.

## C++ API

Include [`hipSZ.h`](include/hipSZ.h) and link hipSZ with the HIP runtime. The generic functions take a data type, an encoding mode, an absolute error bound, and a HIP stream. The 1D form is unchanged from the original interface:

```cpp
#include <hipSZ.h>

void roundtrip_1d(float* input_d, unsigned char* compressed_d, float* output_d,
                  std::size_t count, float abs_error, hipStream_t stream) {
    std::size_t compressed_bytes = 0;
    hipSZ_compress(input_d, compressed_d, count, &compressed_bytes, abs_error,
                   HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
    hipSZ_decompress(output_d, compressed_d, count, compressed_bytes, abs_error,
                     HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
}
```

For 2D or 3D processing, use the extended entry points and pass the shape. `dims.x` is the fastest-varying dimension; for the CLDLIQ field above, `dims = make_uint3(3600, 1800, 26)`:

```cpp
void roundtrip_3d(float* input_d, unsigned char* compressed_d, float* output_d,
                  std::size_t count, float abs_error, uint3 dims,
                  hipStream_t stream) {
    std::size_t compressed_bytes = 0;
    hipSZ_compress_ex(input_d, compressed_d, count, &compressed_bytes, abs_error,
                      HIPSZ_DIM_3D, dims, HIPSZ_TYPE_FLOAT,
                      HIPSZ_MODE_OUTLIER, stream);
    hipSZ_decompress_ex(output_d, compressed_d, count, compressed_bytes, abs_error,
                        HIPSZ_DIM_3D, dims, HIPSZ_TYPE_FLOAT,
                        HIPSZ_MODE_OUTLIER, stream);
}
```

Available processing dimensions are `HIPSZ_DIM_1D`, `HIPSZ_DIM_2D`, and `HIPSZ_DIM_3D`; types are `HIPSZ_TYPE_FLOAT` and `HIPSZ_TYPE_DOUBLE`; modes are `HIPSZ_MODE_FIXED`, `HIPSZ_MODE_PLAIN`, and `HIPSZ_MODE_OUTLIER`.

Allocate the input, compressed, and reconstructed buffers in device memory before calling the API. Compression writes the compressed byte count to a host variable. For decoding, supply the original element count, type, mode, error bound, and — for 2D/3D — the same `dims`. Allocate at least `count * sizeof(T)` bytes for compressed output and add headroom for small inputs (the rate array is padded).

The header also exposes type-, mode-, and dimension-specific functions such as `hipSZ_compress_1D_plain_f32` and `hipSZ_decompress_3D_outlier_f64`. The pre-merge type-specific 1D names (`hipSZ_compress_plain_f32`, `hipSZ_decompress_outlier_f64`, and so on) remain as compatibility wrappers. See the [FP32 example](examples/hipSZ_test_f32.cpp) and [FP64 example](examples/hipSZ_test_f64.cpp) for buffer setup and checks in both plain and outlier modes.

## Tests

After building (and optionally installing), run:

```bash
ctest --test-dir build --verbose
```

The 512M-element suites cover FP32 and FP64 across 1D, 2D, and 3D processing and all three encoding modes. `hipSZ_test_edge` covers boundary cases such as single-CTA grids, non-divisible shapes, and empty input; `hipSZ_test_stress` exercises the lookback protocol; and `hipSZ_compat_test` checks the pre-merge 1D API. The legacy `hipSZ_test_f32` and `hipSZ_test_f64` examples remain ctest entries and print `Pass error check!` for plain and outlier.

The full 512M-element inputs use 2 GiB for FP32 and 4 GiB for FP64, plus additional buffers; run the suite serially on a device with enough memory. `hipSZ_bench` is installed for performance work but is not a ctest entry.

The [GPU CI workflow](.github/workflows/gpu-sim.yml) runs reduced-size legacy 1D round-trip tests and the edge suite in a simulated AMD MI300X environment.

## Credits and license

hipSZ was developed by Shengquan Yin, with Dingwen Tao and Guangming Tan as advisors. It is maintained in the [OpenSDC community](https://github.com/opensdc-org). For bug reports or integration questions, [open an issue](https://github.com/opensdc-org/hipSZ/issues).

The dimension-aware engine in this branch is the cuSZp v3.0.0 HIP port developed as hipSZp, imported and renamed to hipSZ. See [porting notes](docs/porting-notes.md) for the porting divergences, the merge provenance, and the compressed-stream compatibility notes.

For the underlying algorithm, cite *cuSZp: An Ultra-Fast GPU Error-Bounded Lossy Compression Framework with Optimized End-to-End Performance*, SC 2023. [DOI: 10.1145/3581784.3607048](https://doi.org/10.1145/3581784.3607048).

hipSZ is distributed under the BSD 3-Clause License. The license file also includes the third-party notices for code incorporated from cuSZp. [License and notices](LICENSE).
