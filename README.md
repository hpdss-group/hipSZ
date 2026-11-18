# hipSZ

**HIP-based error-bounded lossy compression for scientific data**

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

hipSZ compresses FP32 and FP64 scientific arrays on AMD GPUs and Hygon DCUs. It provides a command-line tool and a C++ API for data already in device memory. Both interfaces support plain and outlier-aware encoding. The HIP implementation is based on [cuSZp](https://github.com/szcompressor/cuSZp).

[Build](#build) · [Quick start](#quick-start) · [C++ API](#c-api) · [Hygon K100 example](#hygon-k100-example)

## Build

Use Linux with CMake 3.21 or later and a C++17 toolchain. Put `clang` and `hipcc` on your `PATH`.

| Hardware | Toolkit |
| --- | --- |
| Hygon DCU | DCU Toolkit (DTK) 24.04.1 or later |
| AMD GPU | ROCm 6.2 or later |

```bash
git clone https://github.com/opensdc-org/hipSZ.git
cd hipSZ
```

The current [CMake configuration](CMakeLists.txt) sets `CMAKE_HIP_ARCHITECTURES` to `gfx926`. Edit that setting before configuring if your device has a different target. For example, set it to `gfx942` for an AMD MI300X.

Set `HIP_ROOT` to the prefix containing your HIP headers and CMake package. For a DCU, load the DTK environment and use its HIP installation prefix.

```bash
HIP_ROOT=/opt/rocm
export PATH="$HIP_ROOT/bin:$HIP_ROOT/llvm/bin:$PATH"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HIP_ROOT" \
  -DCMAKE_CXX_FLAGS="-I$HIP_ROOT/include" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build --parallel --target install
```

The install tree contains the CLI and test executables in `install/bin/`, the hipSZ libraries, and the public header `install/include/hipSZ.h`.

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

The tool compresses the input, reconstructs it, and reports the compression ratio, throughput, and error check.

| Option | Meaning |
| --- | --- |
| `-i PATH` | Raw floating-point input |
| `-t f32\|f64` | Input data type |
| `-m plain\|outlier` | Plain fixed-length or outlier-aware encoding |
| `-eb abs VALUE` | Absolute error bound |
| `-eb rel VALUE` | Range-relative error bound, converted to `VALUE × (max − min)` |
| `-x PATH` | Optional compressed output |
| `-o PATH` | Optional reconstructed output |

<a name="using-hipsz-prepared-executable-binary"></a>

## Hygon K100 example

These measurements were recorded on a single Hygon K100 DCU with `CLDLIQ_1_26_1800_3600.f32` and `-eb rel 1e-4`. The rates are the values reported by the CLI for the timed compression and decompression calls.

| Encoding | Compression (GB/s) | Decompression (GB/s) | Compression ratio |
| --- | ---: | ---: | ---: |
| Plain | 135.50 | 272.55 | 9.90× |
| Outlier | 65.32 | 182.15 | 17.36× |

Both runs passed the CLI's reconstruction error check.

```bash
./install/bin/hipSZ -i CLDLIQ_1_26_1800_3600.f32 -t f32 -m plain -eb rel 1e-4
./install/bin/hipSZ -i CLDLIQ_1_26_1800_3600.f32 -t f32 -m outlier -eb rel 1e-4
```

## C++ API

Include [`hipSZ.h`](include/hipSZ.h) and link hipSZ with the HIP runtime. The generic functions take a data type, an encoding mode, an absolute error bound, and a HIP stream.

```cpp
#include <hipSZ.h>

void roundtrip(float* input_d, unsigned char* compressed_d, float* output_d,
               std::size_t count, float abs_error, hipStream_t stream) {
    std::size_t compressed_bytes = 0;
    hipSZ_compress(input_d, compressed_d, count, &compressed_bytes, abs_error,
                   HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
    hipSZ_decompress(output_d, compressed_d, count, compressed_bytes, abs_error,
                     HIPSZ_TYPE_FLOAT, HIPSZ_MODE_PLAIN, stream);
}
```

Allocate the input, compressed, and reconstructed buffers in device memory before calling the API. Compression writes the compressed byte count to a host variable. For decoding, supply the original element count, type, mode, and absolute error bound.

The header also exposes type- and mode-specific functions such as `hipSZ_compress_plain_f32` and `hipSZ_decompress_outlier_f64`. See the [FP32 example](examples/hipSZ_test_f32.cpp) and [FP64 example](examples/hipSZ_test_f64.cpp) for buffer setup and checks in both modes.

<!-- Document optimized interfaces here when they are merged. Include supported
data types and architectures, buffer and stream requirements, and a matching example. -->

## Tests

After installing, run `ctest --test-dir build --verbose`. Check the output for `Pass error check!` for both modes in each test. The full test inputs use 2 GiB for FP32 and 4 GiB for FP64, plus additional buffers.

The [GPU CI workflow](.github/workflows/gpu-sim.yml) runs smaller FP32 and FP64 round-trip tests in a simulated AMD MI300X environment.

## Credits and license

hipSZ was developed by Shengquan Yin, with Dingwen Tao and Guangming Tan as advisors. It is maintained in the [OpenSDC community](https://github.com/opensdc-org). For bug reports or integration questions, [open an issue](https://github.com/opensdc-org/hipSZ/issues).

For the underlying algorithm, cite *cuSZp: An Ultra-Fast GPU Error-Bounded Lossy Compression Framework with Optimized End-to-End Performance*, SC 2023. [DOI: 10.1145/3581784.3607048](https://doi.org/10.1145/3581784.3607048).

hipSZ is distributed under the BSD 3-Clause License. The license file also includes the third-party notices for code incorporated from cuSZp. [License and notices](LICENSE).
