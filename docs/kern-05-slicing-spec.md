# NumType — Kern 05: Slicing at the TS Surface (Spec)

Date: 2026-07-10 · Status: complete — implemented incl. stretch goal, independently verified (see kern-05-ergebnisse.md)

## Why (intent)

The ABI has been slicing-ready since Kern 03 (strided operand quadruples with element offsets, status-4
bounds validation, cargo-tested with nonzero offsets) — no Rust work is expected in this phase. The runtime
side is O(1) view metadata over the existing refcounted buffers. **The research core is the type layer**:
this is the first USP-side phase since Spike 01 — what can the type system say about a sliced shape without
violating the empirically verified TS-7 limits (tail-recursion cap ~1000; cost axis = recursion depth in a
single type; tuple arithmetic is fine for *ranks*, never for large *dimension values* — KB
`ts7-native-compiler-empirie`, CLAUDE.md)?

The answer this spec commits to: **gradual, honest, rank-precise typing as the core** (integer indexing
drops the axis statically; range-sliced axes degrade to `number` honestly; untouched axes keep their literal
dims) **plus a hard-gated stretch goal** (digit-string arithmetic for statically computed slice dims — the
"probing the limits" content, allowed to fail and be dropped honestly).

## Slicing semantics (binding — mirrors NumPy basic slicing, minus negative steps)

`slice(...specs)` takes one spec per leading axis; trailing axes are taken in full. Spec forms per axis,
for an axis of dim `d`:

- **Integer `i`** — index the axis and REMOVE it. Normalize: `i < 0 → i + d`; then require `0 ≤ i < d`,
  else **throw** (NumPy IndexError analog; indices do NOT clamp). Effect: `offset += i · stride`.
- **`null`** — take the axis in full (dim and stride unchanged).
- **`{ start?, stop?, step? }`** — range slice, keeps the axis. `step` defaults to 1 and must be `≥ 1`
  (throw on 0 or negative — negative steps/flips need signed strides, out of scope since Kern 03).
  Defaults `start = 0`, `stop = d`. Normalize (NumPy clamping — ranges clamp, never throw):
  `start < 0 → start + d`, then clamp to `[0, d]`; same for `stop`. Result dim
  `= max(0, ceil((stop − start) / step))`; `offset += start · stride`; `stride' = stride · step`.
- More specs than the rank: **throw** at runtime; **compile error at the argument** when both are tuples
  (see type layer).

**Fixture table (hand-checked against NumPy semantics; these become unit tests pinning the shared
normalizer):** for `d = 5`: `{start:1,stop:4}` → dim 3 · `{step:2}` → dim 3 (indices 0,2,4) ·
`{start:1,step:2}` → dim 2 (1,3) · `{start:-2}` → dim 2 · `{stop:-1}` → dim 4 · `{start:10}` → dim 0 ·
`{start:3,stop:2}` → dim 0 · int `-1` → index 4 · int `5`/`-6` → throw. For `d = 0`: `{}` → dim 0 ·
int `0` → throw.

## Scope

### In scope

1. **Shared normalizer** (`spike/src/runtime.ts`, additive): `normalizeSliceSpecs(shape, specs)` →
   per-axis `{ kind: "index", i } | { kind: "range", start, dim, step }` (post-normalization), used by
   BOTH the naive and the resident implementation. This is a deliberate, documented differential blind
   spot: the two sides share spec *parsing* but diverge in *data movement* (copy vs view metadata) — which
   is where the differential value lies. The normalizer's own semantics are pinned by the fixture table
   above as direct unit tests, not by the differential.
2. **Naive reference** `NDArray.slice(...specs)` — copy-based gather (`sliceRuntime(shape, data, norm)` in
   runtime.ts, additive; existing functions untouched). Returns a fresh `NDArray`.
3. **Resident** `WNDArray.slice(...specs)` — an **O(1) view**: same buffer (`refs + 1`), integer axes
   folded into the offset and dropped, range axes reshaped per the normalization. Composes with Kern-03
   transpose views (slice-of-transpose, transpose-of-slice) and with `contiguous()`/`toArray()`
   (materialize path already handles `offset ≠ 0`). Lifecycle rules identical to transpose views
   (refcount, per-handle dispose, GC backstop) — they are the same mechanism.
4. **Type layer** (`spike/src/slice.ts`, new; `ndarray.ts`/`resident.ts` gain the method):
   - `SliceShape<S, Specs>` with the **core rules**: wide `S` (`number[]`) degrades wholly (wide-type
     guard FIRST — KB `ts-wide-types-vor-tupel-rekursion-abfangen`); integer spec → axis dropped (rank
     effect is static even when the index value is a plain `number`); `null` → literal dim preserved;
     range object → that dim becomes `number` (core); specs beyond the rank → `SliceError` surfaced via
     the existing `Guard` pattern **at the spec argument**, naming the rank and the spec count (KB
     `ts-fehler-am-argument-required-property-guard`). Trailing axes preserved literally. All recursion
     accumulator/tail-recursive; rank-level only.
   - `const` type parameter for the specs so callers never need `as const` (CLAUDE.md).
   - Hover quality is an acceptance criterion: results must display as clean resolved tuples
     (`WNDArray<[2, number, 4]>`), verified in the type tests via the existing `Expect`/equality helpers.
   - **Stretch goal (attempt AFTER the core is green, gated):** statically computed dims for range slices
     when `start`/`stop`/`step` are literals with `step = 1` (or omitted) — requires from-scratch
     digit-string arithmetic (number → `${N}` template string → digit-wise subtract/compare with borrow →
     back via `infer N extends number`; O(digits) recursion, NOT tuple-length arithmetic), implementing
     the same clamping semantics as the runtime (negative literals may degrade to `number` if they
     multiply edge cases — document the supported literal subset precisely). **Go/no-go gates:** measure
     `pnpm check:diag` Instantiations + wall time BEFORE starting (baseline) — with the stretch enabled,
     full-project Instantiations ≤ 3× baseline AND check time ≤ 2× baseline AND hovers stay clean AND all
     error messages stay at the argument. Any gate fails → ship the core rules, keep the attempt on a
     branch of the results doc as findings, add a FOLLOWUPS item. Dropping the stretch is a valid,
     honest outcome, not a failure.
5. **Rust: expected zero changes.** If anything proves necessary, stop and document why (honesty rule) —
   do not silently extend the ABI.
6. **Tests:**
   - `spike/tests/slice.test-d.ts` (type level, runs under `pnpm check`): result-shape rules (drop / keep /
     degrade / trailing axes / wide-S degradation), error-at-argument cases via `@ts-expect-error` with
     message-shape checks where the idiom allows, hover/resolved-type pins via the existing test utils;
     stretch cases if the stretch lands.
   - `spike/tests-runtime/slice.test.ts` — normalizer fixture tests (the table above, verbatim); then
     differential: ≥120 seeded cases — random shapes (rank 1–4, dims 1–10) × random spec vectors (mix of
     int/null/range with negative indices, steps 1–3, out-of-range clamps, empty results),
     `WNDArray.slice(...).toArray()` bit-identical to `NDArray.slice(...)`'s data; ops on sliced views
     (add/matmul/sum on sliced operands) vs naive ops on the naive slices — bit-identical (this
     end-to-end exercises nonzero offsets through every strided kernel and the blocked matmul for the
     first time from TS); composition cases: slice-of-transpose, transpose-of-slice, slice-of-slice;
     lifecycle: slice views share the buffer (free-counter deltas, dispose interleavings — mirror
     strided-lifecycle idioms); error cases (index OOB, step ≤ 0, too many specs).
   - **package.json: add the new runtime test file to the explicit `test:resident` list** (the footgun).
7. **Bench** `spike/bench-core/slice.ts` (`pnpm bench:slice`), deliberately small (slicing is metadata;
   the claim to measure is the pipeline effect): Series A — `row-block slice → sum` end-to-end, resident
   view path vs naive TS slice+sum, n ∈ {256, 512, 1024}; Series B — step-2 slice consumed by `sum()`
   (honest strided-read cost vs its naive equivalent). Same discipline (seeded, bit-identity gate,
   adaptive reps, ≥2 runs, ranges).
8. **Demo:** one resident slice showcase (slice view → op, asserted equal inline, disposed).
9. **Findings** `docs/kern-05-ergebnisse.md` (orchestrator-written post-verification, with addendum).

### Out of scope

Negative steps / flips (signed strides — would be an ABI revision); boolean masks & fancy/integer-array
indexing; `newaxis`/ellipsis; type-level bounds checking of integer indices against literal dims (would
need large-dim comparisons; runtime backstop only — may become a follow-up if digit arithmetic lands);
writable views / in-place ops; any change to frozen v1, the Kern-03/04 kernels, or the existing type
files (`broadcast.ts`/`matmul.ts`/`reduce.ts` untouched); threads.

## Acceptance criteria

- `pnpm check` clean, including the new type tests; `check:diag` baseline and after-numbers recorded
  (Instantiations + wall time), with the stretch gates evaluated explicitly if the stretch was attempted.
- `cargo test` green with **zero crate diff** (or the documented exception per honesty rule).
- `pnpm test:core` 791/791 untouched; `pnpm test:resident` all existing tests green plus the new slice
  suites; `pnpm test:resident:gc` green; `pnpm demo` green including the slice showcase.
- Normalizer fixtures match the spec table exactly; differential ≥120 cases bit-identical incl.
  composition with transpose views and ops through the blocked matmul with nonzero offsets.
- Error-at-argument verified for too-many-specs at the type level AND thrown at runtime; hover quality
  pinned in type tests.
- Zero new dependencies; package.json lists updated.

## Honesty rule

Same as every phase. Specifically here: if the stretch goal fails its gates, the core-only result plus the
measured reasons IS the phase's finding — report the instantiation/time numbers and the failing edge cases
rather than shipping a half-lit stretch. If the shared-normalizer design ever hides a semantics bug that
the fixtures missed, document the blind spot rather than pretending the differential covered it.
