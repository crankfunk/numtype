# NumType — Kern 03: Strided Views (Spec)

Date: 2026-07-10 · Status: complete — implemented, independently verified (see kern-03-ergebnisse.md)

## Why (intent)

Kern 02 made array data resident in WASM linear memory, but every op still materializes: `transpose()` runs a
full O(n) gather copy (`nt_transpose`) even though the result is just a re-indexing of the same bytes. Kern 03
cashes in the first real payoff of the residency memory model: **a transpose becomes an O(1) metadata
operation** — a *view* sharing the base array's buffer with permuted strides — and the kernels learn to read
strided (non-contiguous) operands directly. The naive TS runtime stays the correctness reference; the v1
copy-based backend stays the frozen performance baseline; both remain untouched.

Because `WNDArray` exposes **no mutation** (no in-place ops, no `set`), views are semantically
indistinguishable from copies — this is a pure optimization, observable only through performance, memory, and
the (newly refcounted) dispose semantics. That makes the existing resident differential suite a ready-made
regression harness for the new plumbing.

## Scope

### In scope

1. **ABI extension — strided entry points** (`crates/core/src/abi.rs`; the existing entry points and kernel
   functions stay **byte-for-byte untouched** — v1 is a frozen baseline, deliberate duplication over shared
   code paths):
   - A strided operand is the quadruple **(shape ptr/rank, strides ptr, offset, data ptr/len)**: `rank` u32
     dims, `rank` u32 *element* strides, a u32 *element* offset into the buffer, and the **full buffer**
     length in elements (not the view's logical size). Outputs are always freshly allocated, contiguous,
     row-major — strided *outputs* are out of scope.
   - New exports: `nt_add_strided`, `nt_matmul_strided`, `nt_sum_all_strided`, `nt_sum_axis_strided`, and
     `nt_materialize` (gather a strided view into a contiguous row-major buffer — the copy-out/`contiguous()`
     workhorse).
   - New status code **4 = strides out of bounds**: before touching data, each operand is validated with
     `offset + Σ (dim_i − 1)·stride_i ≤ data_len − 1` (checked u64 arithmetic, overflow ⇒ status 4; skipped
     when the logical size is 0 — a size-0 view never reads). After this check, every read the loops can
     issue is provably in-bounds. Status codes 0–3 keep their meanings; abi.rs module doc updated.
   - The strided kernel functions mirror the *same* `runtime.ts` loops as their contiguous originals, with
     `compute_strides(shape)` replaced by caller-supplied strides (+ base offset). Same iteration order, same
     accumulation order — bit-identity is the acceptance test, not a hope.
   - **The determinism trap to get right:** `nt_sum_all_strided` must accumulate in the view's **logical
     row-major order** (flat 0..size, unravel → strided offset), *not* in memory order — float addition is
     order-sensitive, and on a transposed view the two orders differ. A differential test on transposed views
     is the guard.
   - Matmul keeps its contract: operands arrive already promoted to rank ≥ 2. TS supplies the promoted
     strides; the axis added by 1-D promotion gets stride 0 (its dim is 1, so the stride is never multiplied
     by a nonzero index — document at the call site).
2. **Resident layer** (`spike/src/wasm/resident.ts`):
   - `WNDArray<S>` gains `readonly strides: readonly number[]` (element strides), an internal element
     `offset` (always 0 in Kern 03 — only transpose views exist; the ABI supports nonzero offsets so slicing
     later needs no ABI revision), and an internal **shared buffer record** `{ core, ptr, bytes, lenElems,
     refs }` replacing the per-handle ptr/len fields.
   - `transpose()` returns an **O(1) view**: reversed shape + reversed strides, same buffer, `refs + 1`. No
     kernel call. View-of-view composes naturally (a double transpose is a contiguous-strided view). The
     return type stays `WNDArray<Transpose<S>>` — the type layer is untouched.
   - New `contiguous(): WNDArray<S>` — always returns a **fresh, independently owned, contiguous** copy (via
     `nt_materialize`), even when the receiver is already contiguous: predictable ownership beats a
     micro-optimization; document the deliberate choice.
   - **All** ops route through the strided entry points; contiguous handles pass their natural strides and
     offset 0. One code path, and the entire existing resident suite exercises the new plumbing. The old
     non-strided exports remain in use by the v1 backend only.
   - `toArray()`: fast path (single copy) when contiguous with offset 0; otherwise `nt_materialize` into
     scratch → copy out → free scratch. `toNestedArray()` builds on `toArray()` unchanged.
   - Loader (`spike/src/wasm/loader.ts`): extend `CoreExports` with the five new signatures. `backend.ts` and
     the naive runtime stay untouched.
3. **Lifecycle under sharing (the hard part — get this exactly right):**
   - The shared buffer is **refcounted**: created with `refs = 1`; each view increments; `dispose()` marks
     *this handle* disposed, unregisters it from the FinalizationRegistry, and decrements — the WASM
     allocation is freed exactly when `refs` hits 0. Double-dispose stays a per-handle no-op.
   - Disposing the base while views live keeps the buffer alive and the views fully usable; the disposed
     base's own ops/reads throw as before. Symmetrically for disposing views.
   - The FinalizationRegistry held value becomes the shared buffer record (plain data + core handle — still
     never a reference to the `WNDArray` itself); the finalizer decrements and frees at 0, so dropping any
     subset of handles without `dispose()` still cannot leak or double-free.
   - `getResidentFreeCount()` keeps its meaning: it counts **actual buffer frees**, not dispose calls.
4. **Tests** (node:test, zero deps, seeded via `prng.ts`; **new files must be added to the explicit
   `test:resident` list in package.json** — the known footgun):
   - `spike/tests-runtime/strided.test.ts` — differential, ≥100 seeded cases per op: ops on transposed views
     (each operand position: view/contiguous × view/contiguous) vs the naive reference computing on the
     materialized equivalent — **bit-identical**, incl. broadcast cases, rank 0–4, size-0 shapes;
     `toArray()`/`contiguous()` on views ≡ reference transpose; view-of-view (double transpose) ≡ base.
   - `spike/tests-runtime/strided-lifecycle.test.ts` — refcount semantics: base dispose → view usable; buffer
     freed only after the last handle (assert via free counter); per-handle use-after-dispose throws;
     double-dispose no-op; failing op leaves all handles valid; view creation on a disposed handle throws.
   - `spike/tests-runtime/resident-gc.test.ts` — extend: dropping base+view without dispose frees the buffer
     exactly once (free-counter delta 1) under `--expose-gc`; same honest skip otherwise.
   - Cargo unit tests per strided kernel: known-value cases with transposed strides, nonzero offsets
     (natively testable — offsets index into slices, no raw pointers), stride-OOB → status 4, u64-overflow
     shapes → status 4, rank > 32, rank-0, size-0; plus a seeded equivalence test:
     strided(contiguous metadata) ≡ the untouched contiguous kernel, bit-for-bit.
5. **Bench** `spike/bench-core/strided.ts`, wired as `pnpm bench:strided` (methodology per KB note: seeded
   inputs, bit-identity gate before any timing, adaptive batch-timed reps, warmed JIT, report ranges):
   - **Series A (the payoff claim):** `A.transpose().matmul(B)` at n = 128/256/512 — materializing path
     (`.transpose().contiguous()` ≙ Kern-02 behavior) vs view path, plus naive TS for context.
   - **Series B (the routing-decision check):** contiguous ops through strided entry points vs the old
     entry points — measures the generalization overhead the "always strided" routing pays. If it is not
     ~1×, that is a finding to report, not to hide.
6. **Demo:** extend the resident section of `spike/demo.ts` with a transpose-view op asserted equal inline.
7. **Findings** `docs/kern-03-ergebnisse.md` (English): design decisions (refcount lifecycle, always-strided
   routing, frozen-v1 duplication), measured numbers for both series, gotchas with evidence, deviations with
   reasons, post-verification addendum.

### Out of scope

Slicing / nonzero offsets at the TS surface (ABI-ready, not exposed); negative strides / flips; writable
views and in-place mutation; strided *outputs*; SIMD128/threads/blocking (next phase candidate); type-level
stride or contiguity tracking; any change to the naive `NDArray` or the v1 backend.

## Acceptance criteria

- `pnpm check` green; type layer untouched.
- `cargo test` green; existing kernel functions and non-strided ABI exports byte-for-byte unchanged.
- `pnpm test:core` still 791/791 (v1 untouched ⇒ must hold trivially).
- `pnpm test:resident` green: all existing resident tests pass unchanged *through the strided entry points*,
  plus the new differential and lifecycle suites; `pnpm test:resident:gc` includes the view GC case.
- `pnpm demo` runs all three backends with equal results, including the view op.
- `pnpm bench:strided` runs both series; numbers recorded in the findings doc with an explicit statement:
  did views deliver a measurable win over materializing transpose, and what does the strided routing cost on
  contiguous data?
- package.json test lists updated for every new test file.
- Zero new dependencies (both sides).

## Honesty rule

Same as Spike 01 / Kern 01 / Kern 02. Specifically here: if the strided routing costs measurable overhead on
contiguous ops, or the view path does *not* beat materialization at some sizes, those numbers are findings to
report prominently, not failures to bury. If bit-identity forces an ordering that costs performance (e.g.
logical-order `sum_all` on views), document the cost rather than silently changing the reference semantics.
