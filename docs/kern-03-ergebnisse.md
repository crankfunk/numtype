# NumType — Kern 03: Strided Views (Results)

Date: 2026-07-10 · Spec: docs/kern-03-strided-spec.md · Status: complete, independently verified

## Summary against acceptance criteria

Every acceptance criterion holds, grounded in commands run on 2026-07-10:

- `pnpm check` — clean.
- `cargo test` — **63/63** (was 38; +25 new strided/validation tests). Existing kernel functions and
  non-strided ABI exports byte-for-byte unchanged (verified via `git diff` in independent verification).
- `pnpm test:core` — **791/791**, untouched (v1 frozen-baseline criterion).
- `pnpm test:resident` — **1412 pass + 2 honest GC skips** (was 677+1): the entire Kern-02 suite now runs
  through the strided entry points unchanged (the built-in regression proof), plus the new differential
  (`strided.test.ts`) and refcount-lifecycle (`strided-lifecycle.test.ts`) suites.
- `pnpm test:resident:gc` — **2/2**, including the new exact-once free of a shared base+view buffer.
- `pnpm demo` — all three backends bit-identical, including the new `M2ᵀ (view) @ M2` showcase.
- `pnpm bench:strided` — three series measured, three runs (numbers below, including the unfavorable ones).
- package.json test lists updated; zero new dependencies on either side.

## What was built

**The headline mechanic: `transpose()` is now O(1).** A `WNDArray` is a *view* — `(shape, strides, offset)`
metadata over a shared, refcounted WASM allocation (`ResidentBuffer`). `transpose()` reverses shape and
strides over the same buffer: no kernel call, no allocation, no data movement. Because `WNDArray` exposes no
mutation, views are semantically indistinguishable from copies; the change is observable only through
performance, memory, and dispose semantics.

- **ABI (Rust):** five new `extern "C"` entry points — `nt_add_strided`, `nt_matmul_strided`,
  `nt_sum_all_strided`, `nt_sum_axis_strided`, `nt_materialize` — taking per-operand
  `(shape, strides, offset, data ptr/full-buffer len)` quadruples; outputs always fresh contiguous
  row-major. New status code **4 = strides out of bounds**: caller strides are the first ABI input the
  kernels cannot derive themselves, so `validate_strided_bounds` checks
  `offset + Σ (dim−1)·stride < data_len` in checked u64 arithmetic before any data access. After that check,
  every u32 offset the loops accumulate is provably in-bounds (all terms non-negative and bounded by the
  validated reach).
- **Kernels:** strided generalizations of the same `runtime.ts` loops — `compute_strides(shape)` replaced by
  caller strides plus base offset, identical iteration and accumulation order. The contiguous originals stay
  **byte-for-byte untouched** (deliberate duplication: v1 is the frozen performance/correctness baseline).
- **Resident layer (TS):** all ops route through the strided entry points (contiguous handles pass natural
  strides, offset 0 — one code path, so the whole existing suite exercises the new plumbing). New
  `contiguous()` materializes any handle into an independently owned copy (deliberately always copies, even
  when already contiguous — predictable ownership). `toArray()` fast-paths contiguous handles and gathers
  views through `nt_materialize`. Public surface additions: `strides` (readonly), `contiguous()`.
- **Lifecycle:** the buffer record is refcounted; `dispose()` releases one reference, the allocation is
  freed exactly when the last handle (base or view, any order) releases. FinalizationRegistry held value is
  the shared buffer record (plain data — never the `WNDArray`), so any interleaving of dispose/GC releases
  each reference exactly once. `getResidentFreeCount()` keeps meaning "actual buffer frees".

## The determinism trap this phase existed to get right

`sum_all` on a transposed view: float addition is order-sensitive, and the view's logical row-major order
differs from memory order. `nt_sum_all_strided` walks **logical** order (flat 0..size → unravel → strided
offset), reproducing the naive reference bit-for-bit. The cargo test
`sum_all_strided_transposed_view_uses_logical_order` pins this **non-vacuously**: a 1e100/−1e100 absorption
pattern where memory-order summation provably yields different bits (`assert_ne!` guards the test against
being vacuous — the first version of that test with "mixed magnitude" values turned out NOT to distinguish
the orders and was caught by exactly that guard). The TS differential suite re-pins it end-to-end with 120
seeded view cases.

## Bench (`pnpm bench:strided`, three runs, 2026-07-10)

Methodology per the established discipline: seeded inputs, bit-identity gate before any timing, adaptive
batch-timed reps, warmed JIT, ranges over runs reported. Machine: same dev machine as Kern 01/02 benches.

**Series A — the payoff claim, honestly: transpose-view feeding matmul LOSES at larger sizes.**
`A.transpose() @ B` on resident `[n,n]` operands, view path vs materializing path
(`.transpose().contiguous()` ≙ Kern-02 behavior); `view/mat` > 1 means the view wins:

| n | view/mat (3 runs) | verdict |
|---|---|---|
| 128 | 1.04× / 1.05× / 1.05× | view marginally faster |
| 256 | 0.72× / 0.70× / 0.71× | **view ~30 % slower** |
| 512 | 0.69× / 1.27×* / 0.71× | **view ~30 % slower** (typical) |

\* run 2's 512 row is an outlier: the *materialize* path spiked from ~120 ms to ~211 ms that run (2 reps
only at this size); runs 1 and 3 agree. Reported per the range-not-point rule.

Why: the strided matmul reads the transposed operand column-wise (`a_col_stride = n`), a cache miss per
k-step at n ≥ 256, which costs more than the one O(n²) gather copy it saves. This is precisely why
BLAS-class libraries pack/copy operands for GEMM. **Consequence:** views are not a blanket win; for a hot
matmul on a large transposed operand, `.contiguous()` first is the right call — which is exactly why the
API exposes it. *(Superseded for matmul as of Kern 04: the blocked packing kernel erases this penalty —
view operands are now as fast or faster than materializing at every measured size; see
kern-04-ergebnisse.md, Series B. The general principle — access pattern decides — stands; the concrete
guidance was kernel-dependent.)*

**Series B — the routing decision is validated: strided entry points are free on contiguous data.**
Raw ABI comparison (identical per-rep plumbing, only the entry point differs), `new/old` over
add/matmul × n ∈ {64, 256, 512}, three runs: **0.90×–1.03×** — noise-level. Routing every resident op
through the strided kernels costs nothing measurable; the single-code-path decision stands.

**Series C — where views win unconditionally: consume-once workloads.**
`A.transpose().sum()` — both paths pay the strided reads once; materializing additionally writes and
re-reads an O(n²) copy:

| n | view/mat (3 runs) |
|---|---|
| 128 | 2.01× / 1.98× / (2.25× in verification run) |
| 256 | 1.99× / 1.98× |
| 512 | 1.99× / 2.00× |

**A stable ~2× win** across all sizes. The emerging guidance: *views win when the strided data is consumed
once (or the op is cheap relative to a copy); materialize first when a strided operand feeds an
access-pattern-sensitive hot loop like matmul repeatedly.*

## Gotchas (with evidence)

- **Non-vacuity guards earn their keep:** the first order-sensitivity test values (powers of 10 plus ⅓)
  produced bit-identical sums under both orders — the `assert_ne!` non-vacuity check failed the test and
  forced the deliberate absorption pattern. Without that guard the test would have silently pinned nothing.
- **Op outputs move the free counter synchronously:** two tests initially asserted refcount deltas across a
  region that also disposed an op *output* (its buffer free increments `getResidentFreeCount()` immediately)
  — expected delta 1, observed 2. Counter-delta assertions must isolate the handles under test (fixed in
  `strided-lifecycle.test.ts` and the GC view test; the GC test now also drains straggler finalizers first).
- **Scratch-leak gap closed in the rewrite (resident.ts only):** per-call scratch buffers are now tracked in
  a list freed in a single `finally`, which also covers an OOM throw *between* marshalling allocations — the
  Kern-02-era ordering leaked scratch if the output `nt_alloc` failed. The v1 backend (`backend.ts`) still
  has the old pattern; FOLLOWUPS.md item narrowed accordingly.

## Deviations from spec (with reasons)

- **Series C added to the bench** (spec listed it as optional): after Series A came out negative at larger
  sizes, the consume-once counterpoint was needed for a complete answer to "when do views pay?".
- **One extra test beyond the spec list:** a raw-ABI status-4 boundary test (verification finding: the
  status-4 path was pinned only in cargo tests, never across the TS boundary — `resident.ts` itself cannot
  construct invalid strides today, so the test calls `nt_materialize` directly).
- No other deviations; the type layer is untouched, offsets remain ABI-supported but unexposed.

## Open issues

Tracked in FOLLOWUPS.md: OOM-hardening for the v1 backend (resident.ts's half is now done), rank/len
validation before slice construction in the ABI wrappers (pre-existing pattern, now on five more entry
points), special-value differential coverage (NaN/±Inf/±0), and the next phase candidate: SIMD128 +
blocking for matmul — Series A is direct evidence that memory access patterns, not architecture, are the
current matmul bottleneck.

## Post-verification addendum (2026-07-10)

Independent fresh-context verification (brainroute:verify, full gate re-run + line-by-line bit-identity and
memory-safety review) returned **"Kern 03 meets its spec"** with zero critical/major findings. The four
minor/nit findings and their resolution:

1. *Bench honesty (minor):* Series A's unfavorable numbers must lead the results narrative, not hide behind
   Series C — adopted; see the bench section above, which reports the view path's ~30 % matmul loss first.
2. *FOLLOWUPS.md hygiene (nit):* the strided-kernels item is this phase — checked off in the same commit.
3. *ABI-hardening surface grew (nit):* the deferred rank/len-validation item now covers five more entry
   points — FOLLOWUPS item updated to say so explicitly.
4. *Status-4 TS-boundary coverage (nit):* closed by adding the raw-ABI test described above
   (`strided.test.ts`, "raw ABI: out-of-bounds strides return status 4"); suite green at 1412+2 afterwards.

Verifier-confirmed under direct inspection: bit-identity (iteration/accumulation order) of all five strided
kernels against `runtime.ts`; frozen-v1 byte-for-byte claim; soundness of `validate_strided_bounds`
including the u32-accumulation bound argument; refcount lifecycle across all dispose/GC interleavings; held
values free of `WNDArray` references; scratch-`finally` covering mid-marshalling OOM; the memory rule in
every new code path (views derived only after the last allocation).
