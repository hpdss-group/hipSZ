# Synchronization review: single-kernel decoupled look-back on gfx936

Scope: the grid-wide offset resolution shared by all 36 kernels
(compress and decompress). One CTA = one 32-thread warp; `warp = blockIdx.x`,
`lane = threadIdx.x`. Three device buffers (`cmpOffset`, `locOffset`,
`flag`), one slot per CTA plus one, all `volatile`, zeroed by the host
before each launch.

## The protocol, link by link

Each arrow below is a release→acquire pair that must be order-preserving
end to end. Notation: W(x) write, R(x) read, F `__threadfence()`.

1. Payload publication (per CTA, `lane==31` after the intra-warp scan):
   `W(locOffset[warp+1] = thread_ofs)` — inclusive byte sum of the CTA.
2. `F` — orders (1) before (3).
3. Flag publication:
   * warp 0: `W(flag[0]=2); F; W(flag[1]=1)`
   * warp w>0: `W(flag[w+1]=1)`
   Semantics: `1` = "local sum available in locOffset", `2` = "aggregated
   prefix available in cmpOffset".
4. Look-back (lane 0 of CTAs with `warp>0`):
   `do { status = R(flag[lookback]); F } while (status == 0)` — the spin.
   The trailing F after every (possibly stale) read keeps re-reads from
   being hoisted or satisfied from a stale cache line.
5. Aggregate read: on `status==2` → `R(cmpOffset[lookback])` (prefix);
   on `status==1` → `R(locOffset[lookback])` (local sum), decrement
   lookback, `F`, repeat.
6. Prefix publication: `W(cmpOffset[warp] = excl_sum); F;`
   if last CTA: `W(cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1]); F;`
   `W(flag[warp] = 2)`.
   The fixed port adds, for single-CTA grids, block 0's
   `W(cmpOffset[1] = locOffset[1])` — same slot the host reads.
7. Base offset broadcast: `excl_sum` → `__syncthreads()` →
   `base_idx = excl_sum + rate_ofs` shared write → `__syncthreads()` →
   all lanes write payload at their scanned offsets.

## Why this holds on gfx936

* Store→fence→flag-store on the producer, flag-load(+fence)→data-load on
  the consumer is the standard release/acquire shape. `__threadfence()`
  maps to HIP's device-scope fence; `volatile` keeps the flag/offset
  accesses from being sunk, hoisted, or register-cached across the loop.
* A CTA only ever waits on strictly lower block indices (`lookback < warp`,
  monotonic dispatch order on this hardware), so the spin always terminates
  regardless of how many CTAs are co-resident — this is a forward-progress
  argument, independent of the memory-ordering one.
* The decompress kernels contain the identical shape (their total-offset
  write is dead code — nothing reads it back); the port's single-CTA fix
  lands there too, harmlessly.

## What was verified empirically (gfx936, DTK 26.04)

* `hipSZ_test_stress` part 1: 2048-CTA grid, 100 back-to-back calls
  alternating plain/outlier, deep-verified every 10th call — no hangs, no
  corruption.
* `hipSZ_test_stress` part 2: four concurrent streams with independent
  buffers, 50 interleaved rounds — per-call scratch states do not interfere.
* `hipSZ_test_edge`: 2100 small/irregular cases including both sides of
  every CTA-count boundary — all pass with worst error 0.94x of the 1.1x
  acceptance band.
* `HIP_LAUNCH_BLOCKING=1` was used only as a triage tool during
  development; all conclusions above rest on unmodified execution.

## Fallback (not needed on this platform, kept as contingency)

If a future DTK/compiler combination breaks the volatile+fence shape
(symptom: stress hangs or sporadic lookback corruption), switch the flag
and offset traffic to C11 atomics with explicit ordering
(`__atomic_load_n(..., __ATOMIC_ACQUIRE)` / `__atomic_store_n(...,
__ATOMIC_RELEASE)`) along the whole chain — mixing atomic and volatile
accesses to the same slot is not allowed, so the conversion must be
all-or-nothing per buffer.
