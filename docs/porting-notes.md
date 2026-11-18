# hipSZ porting notes (cuSZp v3.0.0 → HIP / Sugon DCU gfx936)

This file records every deliberate divergence from upstream cuSZp
(szcompressor/cuSZp @ 9a9645a) introduced by the HIP port, the capacity
contract, and known numerical characteristics observed on gfx936.

## Import provenance and hipSZ compatibility

The engine and tests in this tree were imported from
[`Lizixuan05/hipSZp`](https://github.com/Lizixuan05/hipSZp) at commit
`34054ee882051b1df639a78fb647cfc36bdd8c49` (the HIP port of cuSZp v3.0.0
at upstream `9a9645a`) and renamed `hipSZp` → `hipSZ`. The merge keeps the
pre-merge hipSZ public surface working:

* `hipSZ_compress` / `hipSZ_decompress` keep their original 8-argument
  signatures and operate on 1D data.
* `hipSZ_compress_plain_f32` and the other pre-merge type-specific 1D
  entry points remain as thin wrappers over the dimension-aware engine.
* `HIPSZ_MODE_PLAIN` and `HIPSZ_MODE_OUTLIER` keep their numeric values;
  `HIPSZ_MODE_FIXED` is appended.
* The CLI accepts `-d` as optional and defaults to 1D, so commands written
  before this merge keep working.

The canonical new entry points are `hipSZ_compress_ex` /
`hipSZ_decompress_ex` (dimension + `uint3 dims`) and the dimension-specific
`hipSZ_{compress,decompress}_{1D,2D,3D}_{fixed,plain,outlier}_{f32,f64}`
family. The main `README.md` is intentionally unchanged by the merge; its
2D/3D/fixed/`_ex` documentation is a follow-up.

**Compressed-stream compatibility:** the imported f32 1D kernel uses
`thread_chunk=256` instead of the pre-merge value 1024, which changes the
rate-array grouping and byte layout. A compressed f32 1D stream produced by
the pre-merge hipSZ cannot be assumed to decode with this engine. There is
no format version in the stream, so callers must recompress when upgrading.
The f64 1D geometry is unchanged, but cross-version decoding is still not a
supported contract.

## Divergences from upstream

### 1. Single-CTA compressed-size underreport (upstream defect, fixed here)

Upstream kernels publish the grid-wide total compressed offset
(`cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1]`) only inside the
`if (warp > 0)` branch (`warp == blockIdx.x`; every CTA is one 32-thread
warp). With a single-CTA grid — up to 32,768 elements in every dimension
mode — no warp>0 block exists, so the host reads back the `hipMemset` value
`0` and `*cmpSize` collapses to the rate-array size alone. The payload is
truncated on any host round-trip and decompression fails or corrupts.

* Confirmed on gfx936 before fixing: 1D n<=32768 returned `cmpSize == 1024`
  (rate padding only); 3D 32x32x32 (=512 blocks, exactly one CTA) failed
  with 100% of values wrong.
* Fix: a new `else` branch lets block 0 publish `cmpOffset[1] = locOffset[1]`
  when `gridDim.x == 1` (block 0's inclusive sum is already in
  `locOffset[1]`). Applied to all 18 compress kernels; the same code shape
  in decompress kernels is harmless and was patched uniformly.
* The upstream 512M-element tests never hit the bug — the edge suite
  (`hipSZ_test_edge`) regression-tests it on both sides of the boundary
  (n = 32768 / 32769, 4096x8 / 4097x8, 32^3 / 33x32x32).
* To be reported upstream.

### 2. Entry-point capacity guards

Upstream entries perform no input validation. The port rejects, with
`*cmpSize = 0` and no kernel launch:

* empty inputs (`nbEle == 0`);
* inputs whose worst-case compressed byte offset cannot be represented in
  the 32-bit device-side offset bookkeeping (`nbEle/32*136 > 0xFFFFFFFF`,
  where 136 B covers the per-32-element-group worst case: 128 B payload
  + 4 B sign + rate-array share).

### 3. Quantization / bit-width helpers: PTX → C++

The f32 kernels' inline PTX (`mul.f32`/`setp.ge`/`selp`/`add.f32`/
`cvt.rzi.s32.f32`/`sub.s32`, and `clz.b32`) was replaced by the pure-C++
versions upstream keeps commented at the bottom of the f32 kernel files
(see `docs/quantization-equivalence.md`). Device code is compiled with
`-ffp-contract=off` (CMake option `HIPSZ_FP_CONTRACT` re-enables
contraction for experiments) so the multiply and the `+0.5f` cannot fuse.

### 4. Warp shuffles: 64-bit mask + explicit width=32

All `__shfl_up_sync(0xffffffff, v, d)` / `__shfl_sync(0xffffffff, v, 31)`
became `__shfl_up_sync(0x00000000ffffffffULL, v, d, 32)` /
`__shfl_sync(0x00000000ffffffffULL, v, 31, 32)`. On gfx936 the hardware
wave is 64 lanes; a 32-thread CTA occupies lanes 0–31, so the mask selects
exactly the resident lanes and `width=32` keeps the logical group at 32.
The mask and the width are independent parameters and both are required.
Built with `-DHIP_ENABLE_WARP_SYNC_BUILTINS` (DTK HIP).

### 5. Timer

`TimingGPU` now records events on the caller-provided stream (default 0
preserves old behavior), destroys events after each measurement, and frees
its private state in the destructor. Tests report both the same-stream
kernel-only time and a host wall-clock "full-API" time that includes the
per-call scratch allocation/memset, the compressed-size readback, and the
implicit synchronization.

### 6. Test programs

Upstream examples returned 0 unconditionally and their check
`fabs(a-b) > eb*1.1` accepts NaN. The port: any failure exits non-zero;
decompressed values must be finite; every HIP call and kernel launch is
checked (`HIP_CHECK`); `ctest` runs the build-tree executables directly
(no prior installation needed).

### 7. Build system

CMake native HIP language (`enable_language(CXX HIP)`), `find_package(hip)`,
link `hip::host`, exported targets propagate `__HIP_PLATFORM_AMD__`, static
archives are built PIC, and `hipSZConfig.cmake` finds the HIP dependency.
Architecture precedence: an explicitly passed `CMAKE_HIP_ARCHITECTURES`
wins; otherwise `HIPSZ_HIP_ARCHITECTURES` (default `gfx926`, the
Hygon K100 target documented by the main README) applies. The imported
engine was validated on gfx936; pass `-DCMAKE_HIP_ARCHITECTURES=gfx936` for
that device.

## Capacity contract (documented upstream assumptions)

* Compressed-output buffer: allocate at least `nbEle * sizeof(T)` bytes;
  for tiny inputs add headroom — the 1D rate array alone is padded to
  >= 1024 B per CTA (a 1-element 1D input legitimately compresses "up" to
  ~1 KB + payload). The edge suite allocates `+ 16384` B.
* 32-bit byte offsets cap the supported input size (guards above);
  f32 ≈ 1.01e9 elements, f64 ≈ 1.01e9 elements (offset limit is on
  compressed bytes, not input bytes).
* `errorBound` is a C `float` in the unified interface — f64 data also
  receives a single-precision error bound (inherited upstream limitation).

## Known numerical characteristics (observed on gfx936, expected identical
on NVIDIA since both use IEEE-754 single-precision ops with the same
rounding)

* Outlier-pattern data with amplitude ~1e3 and eb = 1e-4: recipPrecision =
  0.5/1e-4 = 5e3, quantization values ~5e6 approach the float exact-integer
  boundary (2^23), where `x + 0.5` rounds ambiguously. Measured worst error
  ≈ 1.22x eb (exceeds the 1.1x acceptance band). This is an upstream f32
  precision limit, not a porting regression; the edge suite avoids the
  combination (outliers scaled to ±100) and this note documents it.
* Compression ratios may differ in the last digits across hosts because the
  test data are generated with host libm; acceptance uses error checks, not
  byte-identical streams.

## DTK/runtime quirks encountered (recorded, not root-caused)

* **Bare event sequences can segfault in `libgalaxyhip.so.5`**: the first
  benchmark harness arranged `hipEventCreate` + `hipEventRecord(ev, s)` +
  `hipEventSynchronize` inline around the first compress call and crashed
  inside the runtime (jump to an invalid address). The identical API calls
  through the `TimingGPU` helper class — the exact pattern used by every
  ctest binary — are stable over thousands of invocations, and a minimal
  standalone reproduction (create/record/sync, no library) also ran clean.
  The harness uses `TimingGPU`. If you extend the perf tooling, prefer that
  pattern over inline event pairs.
* **Upstream `-Wunsequenced` warnings**: several decompress kernels contain
  multiple `cmp_byte_ofs++` modifications inside one expression (e.g.
  `cmpData[cmp_byte_ofs++] << 24 | cmpData[cmp_byte_ofs++] ...`). The
  evaluation order is unspecified in C++; the port compiles cleanly with
  hipclang and every decompression test passes, but any future compiler
  change could reorder these — worth fixing upstream by sequencing the
  reads explicitly.
* `void function is missing a return statement` warnings on the kernels
  are upstream style (if/else dispatch without trailing return); harmless.
