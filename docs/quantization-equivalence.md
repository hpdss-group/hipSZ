# Quantization equivalence: nvPTX vs. C++ (f32 kernels)

Upstream f32 kernels quantize with inline PTX; the port uses the pure-C++
form upstream itself keeps as a comment at the bottom of each f32 kernel
file (and uses in the f64 files). This note argues equivalence and marks
the residual caveats.

## Side-by-side

PTX (upstream):

    mul.f32 dataRecip, %1, %2          ; dataRecip = data * recipPrecision
    setp.ge.f32 p, dataRecip, -0.5     ; p = (dataRecip >= -0.5)
    selp.s32 s, 0, 1, p                ; s = p ? 0 : 1
    add.f32 temp1, dataRecip, 0.5      ; temp1 = dataRecip + 0.5
    cvt.rzi.s32.f32 %0, temp1          ; truncate toward zero
    sub.s32 %0, %0, s

C++ (port):

    float dataRecip = data * recipPrecision;
    int s = (dataRecip >= -0.5f) ? 0 : 1;
    float temp1 = dataRecip + 0.5f;
    return (int)temp1 - s;

Bit-width helper: `clz.b32` → `(int)(sizeof(unsigned int)*8) - __clz(x)`.

## Equivalence argument

* `mul.f32`/`add.f32` without an explicit rounding suffix default to
  round-to-nearest-even — identical to IEEE-754 `*`/`+` in C++ **provided
  the two operations are not contracted into one FMA**. Contraction would
  compute `data*recip + 0.5` with a single rounding, which changes results
  at the half-integer boundary. The port therefore (a) keeps the three
  steps as separate statements and (b) compiles device code with
  `-ffp-contract=off` (CMake `HIPSZ_FP_CONTRACT` off by default).
* `setp.ge` + `selp` map exactly to the ternary (comparison of floats,
  no NaN subtleties beyond the ones noted below).
* `cvt.rzi.s32.f32` is truncation toward zero; C++ float→int conversion
  is defined as truncation toward zero (C++17 [conv.fpint]).
* `sub.s32` is integer subtract; identical.

Net effect: round-half-to-nearest-even on the magnitude (via the `+0.5`
and `-s` adjustment) with ties resolved symmetrically for negative values —
the classic fixed-point trick; both implementations produce the same int
for every input where the intermediates are representable.

## Residual caveats (documented, not porting bugs)

* **NaN / Inf input**: PTX `setp.ge` is false for NaN → `s = 1`; the C++
  ternary also takes the `s = 1` arm. `cvt.rzi` of NaN/Inf is
  implementation-defined (0 on NVIDIA); `(int)` of NaN/Inf is undefined
  behavior in C++. Upstream never defined the behavior for non-finite
  inputs; the port matches it on finite inputs and likewise does not
  define it for non-finite ones. (Test data are finite; decompressed
  outputs are checked with `isfinite`.)
* **Out-of-int-range magnitudes**: if `data * recipPrecision + 0.5`
  exceeds INT_MAX the quantized value overflows — undefined in both the
  PTX (`cvt.rzi` saturates on NVIDIA) and the C++ path. Upstream status
  quo; the capacity guards do not attempt to bound `data/eb`.
* **Float precision boundary**: for quantization values near 2^23 (~8.4e6)
  the ULP of a float is 1.0, and near 4.2e6 it is 0.5; `+0.5f` at exactly
  0.5-ULP granularity rounds ambiguously (to even). Example measured on
  gfx936: outlier-pattern amplitude ±1000 with eb=1e-4 gives values ~5e6
  and worst error ≈ 1.22x eb. The same arithmetic on NVIDIA produces the
  same result — it is an f32 precision limit of the algorithm, recorded
  in `porting-notes.md`, and avoided by the edge suite's data generator.

## Verification

* `hipSZ_test_edge` exercises ±half-integer boundary shapes through
  sine/random/outlier patterns at eb = 1e-2 and 1e-4 across all dims,
  types, and modes: 1740 cases, worst error 0.94x of the 1.1x band.
* The 512M upstream tests (all three modes, f32/f64) pass their error
  checks on gfx936.
