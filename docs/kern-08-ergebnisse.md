# NumType — Kern 08: Runtime reshape/flatten + Editor-Hover Measurement — Ergebnisse

Date: 2026-07-11 · Status: complete, independently verified (fresh-context pass: CONFIRMED)
Spec: docs/kern-08-reshape-flatten-spec.md (binding, pre-registered gates)

## Summary

Phase B item 5 remainder landed: runtime `reshape`/`flatten` on both surfaces, consuming Spike 04's
`LiteralShapeProduct` — `flatten()` hovers as a **statically computed literal** (`NDArray<[1048576]>`
for `[1024, 1024]`), `reshape()` rejects provable product mismatches at the argument with a message
verbatim-identical to the runtime throw. Both Spike-04 FOLLOWUPS obligations are closed: the guard
wording is fixed (message table below), and the editor-hover cost of REAL reshape/flatten call sites is
measured (new bench:editor workload W6 — in-family, ~0.06 ms hover medians, not an outlier). **Zero
Rust changes; the plain-artifact hash stayed byte-identical** (strong freeze form, `7a65d800…`
reproduced by implementer AND verifier from clean rebuilds before and after).

## What was built

- `spike/src/reshape.ts` (new) — `ReshapeCheck<S, NS>`: wide-filter → `IsUnion`-filter → literal
  product equality; stretch wiring (message built here via `ShowShape`).
- `spike/src/slice-literal.ts` (append-only, prefix-`cmp` proven) — stretch classifier
  `LiteralReshapeDimInvalid<NS>`: negative-literal and dot-form dims are provably invalid → lifted to
  compile errors; exponent forms/`0`/unions/dynamic → no claim. Returns bare verdicts (dim literal or
  `"ok"`), NOT prebuilt messages — necessary to keep the file append-only (no import-line edit for
  `ShowShape`); message construction lives one layer up in reshape.ts, mirroring the existing
  slice-literal.ts/slice.ts split (disclosed as a deviation by the implementer; it is the established
  house pattern, accepted).
- `spike/src/runtime.ts` (append-only, proven) — shared validator `assertReshapeArgs` (documented
  differential blind spot, same rationale as `normalizeSliceSpecs`/`assertVectorPair`).
- `spike/src/ndarray.ts` / `spike/src/wasm/resident.ts` (insertion-only, proven) — `reshape`/`flatten`.
  Naive: always a fresh copy (no-alias invariant). Resident: **view if contiguous** (refs+1, same
  buffer, natural strides, offset 0), **materialize otherwise** (existing `nt_materialize`, scratch/
  `finally` discipline). `isContiguous()` requires offset 0 — a contiguous-shaped but offset-shifted
  view (e.g. a 1-D window slice) conservatively takes the materialize path: verifier probed this
  adversarially — never unsound, at worst a missed optimization.
- `spike/bench-dx/gen-workloads.ts` — new workload **W6** (small reshape hover, computed flatten
  hover, big-dim flatten hover, product-mismatch toggle); harness `editor-latency.ts` untouched
  (verified); generator output deterministic.
- Tests: `spike/tests-runtime/reshape.test.ts` (187 tests, registered in test:resident),
  `spike/tests/reshape.test-d.ts` (guard/method type pins incl. the MAX_SAFE_INTEGER cap cases at
  method level). Demo: new reshape/flatten section, naive + resident bit-identical, view-routing
  proven inline via `describe().ptr`.

## Error messages (now fixed — closes the Spike-04 wording obligation)

Checked dim-validity first (per axis, left to right), then product:

1. `` `reshape: invalid dimension ${d} in shape [${ns.join(",")}] (dims must be non-negative integers)` ``
2. `` `reshape: cannot reshape array of size ${size} into shape [${ns.join(",")}]` ``

Compile-time stems mirror these verbatim (`ShowShape` renders identically to `[${ns.join(",")}]` —
verifier traced `JoinDims` to confirm). `flatten` never throws. `-1` is a plain negative dim here
(inference deferred — FOLLOWUPS).

## Gates (pre-registered; implementer + verifier measured independently)

| Gate | Rule | Measured | Verdict |
|---|---|---|---|
| G1 `pnpm check` | clean, ≤ 1.0 s | clean, 0.42–0.64 s | PASS |
| G2 `check:diag` | ≤ 250,000 | **243,446** (deterministic, 2×2 runs) — new pin | PASS |
| G3 `bench:editor` | hard gate PASS incl. W6 | PASS; W1–W5 unregressed | PASS |
| G4 suites + hash | green; hash identical | core 817 · resident **3279+2** (Δ +187) · cargo 157 · threaded 65 · demo ok · hash `7a65d800…` identical | PASS |

**G2 honesty + trend flag (verifier judgment, adopted):** the delta is +42,732, of which **+33,240
(78 %) is the new type-TEST file** (dominated by the two mandated MAX_SAFE_INTEGER cap-boundary cases
reused at method level, ~13k); product cost proper (guard + methods + demo/bench sites) is **+9,492**.
Pinning 243,446 is fine — but the pin-growth series 133,656 → 188,378 → 200,714 → 243,446 is not
monotonically shrinking, and 1–2 more phases with similarly-sized mandated stress suites would breach
the 250k line. The 250k gate is a CHOSEN affordability bound (5 % of the ~5M compiler budget), not a
hard compiler limit — the next phase's spec must either consciously re-set that bound or restructure
where stress cases live (e.g. a separately-measured stress tsconfig, like bench-dx already does
per-workload). Decision deferred to the next spec, deliberately not made mid-phase.

## Measurement obligation (closes the Spike-04 FOLLOWUPS item)

bench:editor W6, real `reshape()`/`flatten()` call sites against the live native LSP server:

| Position | Median | Range |
|---|---|---|
| small reshape hover (`NDArray<[3, 2]>`) | 0.06 ms | 0.05–0.08 |
| flatten computed-literal hover (`NDArray<[6]>`) | 0.06 ms | 0.05–1.23 (one outlier) |
| big-dim flatten hover (`[1024,1024]` → `NDArray<[1048576]>`) | 0.06 ms | 0.05–0.13 |
| toggle mismatch on/off (diagnostic appear/disappear) | 3.3 ms | 3.02–5.30 |

Per-workload instantiations (isolated tsconfig): **W6 = 23,100** vs W1 16,852 · W2 18,487 ·
W3 49,308 (the dedicated Kern-05 digit-arithmetic stress) · W4 16,833 · W5 22,610. **In-family:** W6
sits beside W5 at roughly HALF the dedicated stress workload's cost, despite a genuine 4×4-digit
multiplication plus dim-validity walk per site. W6's toggle is ~2× W4's (~1.5 ms) — plausibly the
big-dim flatten in the same file re-typechecking per diagnostic pull — still ~150× under the gate.
Conclusion: real reshape/flatten sites are NOT an editor-latency risk; the A1 headroom stands.

Bisection detail worth keeping (implementer, reproduced in structure by the verifier): merely
DECLARING `NDArray<hugeShape>` is cheap (~in the noise); it is the guard/product COMPUTATION per call
site that costs (~6k instantiations for a 3-huge-dim flatten site; typical small sites far less).
`ReshapeCheck` does three non-trivial type computations per site (two products + dim-validity) vs
Spike 04's single product — inherent, not a bug.

## Post-verification addendum (fresh-context pass, CONFIRMED)

Verifier independently reproduced: both clean-rebuild hashes (identical), all six suites, both
check:diag runs, bench:editor incl. W6 numbers (exact instantiation match), prefix-`cmp` and
insertion-only proofs, and probed the stretch classifier for false positives (negative exponent forms,
dot+exponent combos, `-0` — all correctly no-claim or `"ok"`; every compile rejection is a guaranteed
runtime throw). Guard-harness non-vacuity re-proven with a deliberately corrupted expectation in a
scratch copy. Findings, none blocking:

1. **(low)** The materialize path is never driven with a size-0 output (the size-0 test case takes the
   view branch; pre-existing gap, not introduced here) → FOLLOWUPS mini-item.
2. **(nit)** Two test titles overclaim "both surfaces" where the two shapes are split across surfaces
   (`reshape.test.ts:267`, `:286`) — cosmetic, noted here rather than churning the verified tree.
3. **(adopted above)** The G2 trend flag.

Verifier's stated boundary: no interactive VS-Code/LSP session beyond the headless harness (which is
the defined verification surface and was re-run, not trusted); template-literal numeric edge cases
probed by hand for the spec-called-out forms, not exhaustively enumerated.
