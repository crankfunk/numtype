# NumType — Kern 04: SIMD128 + Blocked Matmul (Spec)

Date: 2026-07-10 · Status: complete — implemented, independently verified (see kern-04-ergebnisse.md)

## Why (intent)

Kern 03's Series-A bench is direct, measured evidence that the matmul bottleneck is **memory access
patterns**, not the WASM architecture: a transposed-view operand loses ~30 % at n ≥ 256 because the scalar
k-loop wastes ~7/8 of every cache line, and even the contiguous kernel is a naive triple loop. Kern 04
replaces the matmul core with a **blocked, packed, SIMD128 (f64x2) kernel** — the classic GEMM recipe,
written from scratch — under one non-negotiable constraint: **bit-identity with the naive TS reference is
preserved**. It also tests the Kern-03 prediction on record: packing normalizes operand layout tile-wise,
so the view-vs-contiguous matmul gap should collapse (docs/kern-03-ergebnisse.md).

## The bit-identity law (non-negotiable, the load-bearing design constraint)

Float addition is order-sensitive; a naive "fast matmul" reorders accumulation and breaks the differential
contract. The law that keeps speed AND bits:

1. **Per output element: one accumulator chain, products added in strictly ascending k order.** No multiple
   accumulators per element, no horizontal/pairwise reductions, no split-k with post-hoc combination.
2. **Vectorize only ACROSS output elements**: an f64x2 vector holds 2 *adjacent output columns*
   (`out[mi][nj]`, `out[mi][nj+1]`). Each k-step: splat `a[mi][ki]`, load the B pair `b[ki][nj..nj+2]`,
   `f64x2_mul` then `f64x2_add`. Each lane performs exactly the scalar sequence (one mul rounding, one add
   rounding, ascending k) — bit-identical per lane by construction.
3. **No FMA, no relaxed-simd.** WASM base SIMD128 has no FMA (good — scalar has none either);
   `relaxed-simd` (`f64x2_relaxed_madd` etc.) is explicitly banned: implementation-defined results break
   cross-engine determinism (see KB `cross-plattform-float-determinismus-verifizieren`).
4. **m/n/k cache tiling is free** under (1): tiles processed so that each output element's k-products still
   arrive in ascending k order (k-tiles in ascending order, accumulating into the same chain).
5. **Packing is pure data movement** (gather strided/contiguous operand into contiguous scratch): no
   arithmetic, no effect on bits. This is also what erases the view penalty.

The acceptance test for all of this is the existing differential machinery: bit-identity vs the naive TS
reference, `Object.is` per element.

## Scope

### In scope

1. **Rust kernel** `crates/core/src/kernels/matmul_blocked.rs`: same contract as `matmul_strided`
   (operands pre-promoted to rank ≥ 2, strided quadruples, batch broadcasting, unsqueezed output, status
   codes incl. 4 with the same `validate_strided_bounds` call), new blocked core:
   - Pack the needed A-tile / B-panel into contiguous scratch (`Vec<f64>`, kernel-internal — no ABI
     change); packing reads via the operand's strides + offset, so contiguous and view operands take the
     SAME path.
   - Blocked loops (m/n/k tiling) + the f64x2 micro-step from the law above; scalar tail loops for odd
     n / tile remainders with identical order. Tile sizes: choose by rough measurement, document the chosen
     values and the measurement in the results doc (no need for BLIS-grade tuning — this is a research
     spike; "clearly better and understood" beats "optimal").
   - Batch loop unchanged around the 2-D core (same batch-stride handling as `matmul_strided`).
   - Edge cases preserved: size-0 dims (m/n/k = 0), rank > 32, inner-dim mismatch, bounds → same statuses.
2. **SIMD mechanics** (docs-first: verify against the official Rust/WASM docs before writing intrinsics):
   - `core::arch::wasm32` intrinsics (std, zero external crates) under `#[cfg(target_arch = "wasm32")]`.
   - Enable via repo-root `.cargo/config.toml`: `[target.wasm32-unknown-unknown] rustflags = ["-C",
     "target-feature=+simd128"]` — native `cargo test` builds are unaffected.
   - **Loud guard against silently losing the flag**: in the kernel module,
     `#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))] compile_error!(...)`.
   - Native (host) builds compile a **scalar inner step with the identical arithmetic sequence** behind the
     same blocked structure (cfg switch confined to the innermost micro-step) — cargo tests validate the
     blocking/packing logic natively; the TS differential suite against the real `.wasm` is the gate for
     the SIMD path itself. Document this coverage split honestly.
   - Runtime baseline: WASM SIMD128 is on by default in Node ≥ 16.4 and all evergreen browsers (2021+);
     engines without it fail loudly at instantiation. Document as a supported-environment note.
3. **ABI:** one new export `nt_matmul_blocked`, signature identical to `nt_matmul_strided`. Everything
   existing stays untouched: the v1 entry points/kernels (frozen baseline) AND the Kern-03
   `nt_matmul_strided`/`matmul_strided` (stays as the measurable "before" and as the scalar reference for
   native equivalence tests).
4. **TS:** `loader.ts` `CoreExports` + `resident.ts` `matmul()` route to `nt_matmul_blocked`
   **unconditionally** (no size dispatch in this phase — small-size overhead is measured, not guessed;
   a threshold dispatch is a possible follow-up, recorded in FOLLOWUPS.md if the numbers demand it).
   v1 `backend.ts` untouched.
5. **Tests:**
   - Cargo: packing unit tests (contiguous + strided/view operands, offsets); seeded bit-equivalence
     `matmul_blocked(scalar step) ≡ matmul_strided` on native across shapes that cross tile boundaries
     (dims straddling the chosen tile sizes, odd remainders, k=1, n=1, batch, broadcast batch); size-0;
     status paths (1/2/4).
   - TS: new `spike/tests-runtime/blocked.test.ts` — ≥100 seeded matmul differential cases vs the naive
     reference with **larger dims than the existing suites** (existing gens cap dims at 8; tile-boundary
     logic needs dims spanning at least 1..~3× the largest tile dimension, odd sizes included), operands
     contiguous AND transpose views, batch cases; plus the existing resident/strided matmul differentials
     (120+120 cases) now implicitly exercising the blocked kernel — they must stay green unchanged.
   - **package.json: add the new test file to the explicit `test:resident` list** (the footgun).
6. **Bench** `spike/bench-core/blocked.ts` (`pnpm bench:blocked`), same discipline (seeded, bit-identity
   gate before timing, adaptive reps, warmed JIT, ≥2 runs, ranges):
   - Series A (headline): contiguous matmul n ∈ {64, 128, 256, 512, 1024} — naive TS / v1 / Kern-03
     strided scalar / blocked+SIMD.
   - Series B (the prediction on record): `Aᵀ @ B` via view on the blocked kernel vs
     `Aᵀ.contiguous() @ B` — does packing erase the Kern-03 view penalty?
   - Series C (honesty at the small end): n ∈ {4, 8, 16, 32} — blocking/packing overhead vs the Kern-03
     scalar kernel; if there's a regression, that's a finding + FOLLOWUPS candidate, not something to hide.
7. **Docs:** results doc `docs/kern-04-ergebnisse.md` (written by the orchestrator after verification, with
   post-verification addendum); FOLLOWUPS/CLAUDE.md/README/HANDOFF updates at commit time.

### Out of scope

Threads/COOP-COEP (own phase — worker infrastructure, SharedArrayBuffer); relaxed-simd (banned, see law);
SIMD for add/sum/fill/materialize (only matmul this phase — focus); small-size threshold dispatch
(measure first); transcendentals; dtypes beyond f64; any change to frozen v1 or the Kern-03 strided
kernels; wasm-opt/binaryen post-processing (external tooling).

## Acceptance criteria

- `pnpm check` clean; `cargo test` green (native, scalar inner step) including the new
  blocked≡strided equivalence tests; existing counts intact: `test:core` 791/791,
  all existing resident/strided tests green **through the blocked kernel**.
- New `blocked.test.ts` ≥100 cases bit-identical, incl. views, batch, tile-boundary and odd sizes; wired
  into package.json.
- `pnpm demo` all three backends bit-identical (now exercising the blocked matmul).
- `pnpm bench:blocked` recorded (≥2 runs) with explicit statements: (a) speedup over the Kern-03 scalar
  kernel per size, (b) verdict on the view-penalty prediction, (c) small-size overhead.
- The `.wasm` actually contains SIMD (the compile_error! guard proves the flag; the bench delta proves the
  effect) — no silent scalar fallback.
- Zero new dependencies (both sides); frozen code byte-for-byte untouched.

## Honesty rule

Same as every phase. Specifically here: if SIMD+blocking underdelivers (e.g. < 1.5× at large n), if the
view-penalty prediction is wrong, or if small sizes regress, those numbers are the findings — report them
prominently. If the bit-identity law forces giving up a known GEMM trick (multiple accumulators, split-k),
document what it costs rather than quietly relaxing the law.
