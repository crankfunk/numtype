# NumType — Kern 07: Elementwise Family (sub/mul/div) + Vector Ops (dot/norm/cosineSimilarity) (Spec)

Date: 2026-07-11 · Status: binding, pre-implementation (roadmap Phase B, item 1)

## Why (intent)

Phase B grows the op surface along the target use cases, not along NumPy's table of contents. Item 1 is
the **embedding/RAG use case**: elementwise `sub`/`mul`/`div` (the missing three of the four basic
arithmetic ops) and `dot`/`norm`/`cosineSimilarity` (the exact primitives an embedding pipeline calls in a
loop). The type layer already carries almost everything needed (`Broadcast` from Spike 01); what is new is
**runtime kernels + differential tests** under the established bit-identity law, plus one small new
type-level guard for the vector ops.

Design economy, grounded in the KB (`bit-identische-differentialtests-zwischen-implementierungen`): the
IEEE-exact op whitelist is `+ − * / sqrt` — **`sqrt` is on it** (correctly rounded per IEEE 754, identical
JS ⇄ WASM). Therefore `norm` and `cosineSimilarity` need **no kernels of their own**: they are pinned
TS-side scalar compositions over exactly two new reduction kernels (`dot`, `norm_sq`). Total new Rust
surface: one generic strided elementwise kernel (three thin op wrappers) + two tiny reduction kernels.

## Op semantics (binding)

### `sub` / `mul` / `div` — broadcasting elementwise

Exact structural mirror of `add` on both surfaces (`NDArray`, `WNDArray`): NumPy broadcasting via the
existing `Broadcast<S, B>` type and `runtimeBroadcastShape`, error at the argument via the existing
`Guard`/`OkShape` pattern, result is a fresh array. Naive reference implementations reuse
`elementwiseBinary` with the pinned closures — `(x, y) => x - y`, `(x, y) => x * y`, `(x, y) => x / y`.

**`div` is pure IEEE 754** — no zero checks, no throws: `x/0 → ±Infinity` (sign per IEEE), `0/0 → NaN`,
signed zeros and infinities propagate per the standard. This matches NumPy's *values* (NumPy additionally
warns; we don't — documented divergence, consistent with our transcendental-free determinism story).

### `dot(other)` — 1-D inner product → `number`

`a.dot(b)` for two rank-1 arrays of equal length. Returns a **plain `number`**, deliberately leaving the
NDArray world (NumPy precedent: `np.dot(1-D, 1-D)` returns a scalar, not an array; the embedding use case
consumes plain numbers — thresholds, sorts). **Design decision, documented:** this is intentionally
asymmetric with `sum()` (which returns `NDArray<[]>`): `sum` is a *reduction that stays chainable*;
`dot`/`norm`/`cosineSimilarity` are *scalar consumer ops* that terminate a chain. Alternative considered
and rejected: returning `NDArray<[]>` would duplicate what `matmul` on `[N]·[N]` already provides and
would force `.toNestedArray()` noise on every consumer.

- Rank ≠ 1 (either operand) → error (see message table). Rank-0/rank-2+ are NOT promoted (no `np.dot`
  matrix semantics — `matmul` covers those; no scalar-multiplication special case).
- Length mismatch → error.
- Size-0 vectors (`[0]·[0]`): valid, result `0` (empty accumulation; matches `np.dot([], [])`).
- Accumulation: **single accumulator, strictly ascending index, `acc += a[i] * b[i]`,** starting from
  `acc = 0`. No FMA, no pairwise/split reduction (bit-identity law, same as matmul's inner loop —
  which makes `dot(a, b)` bit-identical to `matmul`'s `[1,k]·[k,1]` single output element by
  construction; pinned by a parity test).

### `norm()` — L2/Frobenius norm over ALL elements → `number`

Any rank, no shape guard needed (mirrors `np.linalg.norm`'s default: flatten, then L2). Defined as
`Math.sqrt(normSq)` where `normSq` is a single-accumulator, strictly ascending sum of squares
(`acc += v * v`) over the array's elements in **logical row-major order** (for strided views: the view's
logical order, NOT memory order — same law as `sum_all_strided`, and the same absorption-pattern
order-pin test idiom applies). Rank-0: `norm() = |x|` (`sqrt(x²)`); size-0: `0` (matches NumPy).

The **`sqrt` happens TS-side on both surfaces** (`Math.sqrt` over the naive JS accumulation resp. over
the kernel-returned sum of squares). `Math.sqrt` is IEEE-correctly-rounded, so bits match iff the
sum-of-squares bits match — which the differential suite asserts.

### `cosineSimilarity(other)` — 1-D × 1-D → `number`

Same operand contract as `dot` (rank-1, equal length, own error-message prefix). The value is computed by
the **pinned expression**, identical on both surfaces:

```ts
const num = dot(a, b);                                   // backend's dot path
const den = Math.sqrt(normSq(a)) * Math.sqrt(normSq(b)); // backend's normSq path, exactly this shape
return num / den;
```

All four operations after the reductions (`sqrt`, `*`, `/`) are IEEE-exact, so cross-backend bit-identity
reduces to the two reduction kernels. **Edge semantics are pure IEEE, documented, no epsilon guards:**
zero vector(s) → `0/0 = NaN`; adversarial magnitude splits can underflow `den` to `0` with `num ≠ 0` →
`±Infinity`. (sklearn adds an epsilon; scipy returns NaN — we take the honest IEEE answer and document.)

### Error messages (binding, shared stems runtime ⇄ type level)

Checked in this order (both levels), `op ∈ {dot, cosineSimilarity}`:

| # | Condition | Message |
|---|-----------|---------|
| 1 | first operand rank ≠ 1 | `` `${op}: expected a 1-D vector as the first operand (got shape [2,3])` `` |
| 2 | second operand rank ≠ 1 | `` `${op}: expected a 1-D vector as the second operand (got shape [2,3])` `` |
| 3 | lengths differ | `` `${op}: vector lengths 3 and 4 do not match` `` |

Compile-time messages mirror these **verbatim** (via `ShowShape`), per the Spike-03 discipline. The
runtime validator is ONE shared function used by both surfaces (see "documented differential blind spot"
below). `sub`/`mul`/`div` reuse the existing broadcast error stems unchanged (they come from
`Broadcast`/`runtimeBroadcastShape`).

## Scope

### In scope

1. **Naive reference (`NDArray`, correctness reference):** methods `sub`/`mul`/`div` (via
   `elementwiseBinary` + pinned closures), `dot`/`norm`/`cosineSimilarity` (via new runtime.ts functions,
   see below). `runtime.ts` is **append-only** (it is the pinned reference for the frozen v1 differential
   suite): new exports `assertVectorPair(op, aShape, bShape)`, `dotRuntime(aShape, aData, bShape, bData):
   number`, `normSqRuntime(data): number` — existing lines byte-for-byte untouched.
2. **Resident surface (`WNDArray`):** methods `sub`/`mul`/`div` (structural clones of `add` modulo entry
   point + message strings; same scratch-list/`finally` discipline, same fresh-output-buffer rule, same
   pre-allocation shape validation), `dot`/`norm`/`cosineSimilarity` (kernel-backed reductions + the
   pinned TS-side compositions; scalar results are read from an ephemeral 1-element scratch buffer —
   fresh view derived after the last allocation, memory rule — then freed in `finally`).
3. **Rust kernels (new files only):**
   - `crates/core/src/kernels/elementwise.rs`: private generic `binary_strided<F: Fn(f64, f64) -> f64>`
     mirroring `add_strided`'s loop structure line-for-line (same validation order, same iteration, same
     offset algebra), plus `pub fn sub_strided / mul_strided / div_strided` delegating with the op
     closure (monomorphized — no dynamic dispatch).
   - `crates/core/src/kernels/vector.rs`: `pub fn dot_strided(a_shape, a_strides, a_offset, a_data,
     b_shape, b_strides, b_offset, b_data) -> KResult<f64>` (validates rank == 1 on both + equal dims →
     else `ShapeIncompatible`; `validate_strided_bounds` on both; ascending single-accumulator loop) and
     `pub fn norm_sq_strided(shape, strides, offset, data) -> KResult<f64>` (any rank ≤ MAX_RANK;
     logical row-major traversal exactly like `sum_all_strided`, accumulating `v * v`).
4. **ABI (append-only, see freeze section):** `nt_sub_strided`, `nt_mul_strided`, `nt_div_strided`
   (14-argument convention identical to `nt_add_strided`, including the hardened
   rank-then-region prevalidation and the status set), `nt_dot_strided` (two strided-operand quadruples +
   `out_data_ptr`, output implicitly 1 f64 — like `nt_sum_all_strided`), `nt_norm_sq_strided` (one
   strided-operand quadruple + `out_data_ptr`, output implicitly 1 f64). **NOT cfg-gated** — these ops
   belong in both artifacts (plain + threads). No new status codes: rank → 2, regions → 3, shape/rank-1/
   length violations → 1, stride bounds → 4.
5. **Type layer:** new file `spike/src/vector.ts` with `DotCheck<S, B, Op extends string>` resolving to a
   `ShapeError` (message table above) or a pass-through; used as `Guard<DotCheck<S, B, "dot">,
   NDArray<B>>` (resp. `WNDArray<B>`, `"cosineSimilarity"`). Rules:
   - dynamic RANK on either side → no claim (pass; runtime backstop);
   - statically known rank ≠ 1 → the rank error for that operand (receiver violations also surface at
     the argument — that is where `Guard` errors land, and the message names the offending shape);
   - **union dims → no claim** (tuple-wrapped/`IsUnion` boundary filter, the Spike-04/06 house rule —
     never a union verdict misread; export the existing private `IsUnion` from `slice-literal.ts` rather
     than duplicating it);
   - both dims plain literals → equal: pass; unequal: the length error.
   - `norm()` takes no argument and gets **no guard** (any rank is valid by semantics — this is exactly
     why norm is speced as Frobenius-over-all-elements rather than rank-1-only: a niladic method has no
     argument to hang a `Guard` on).
   `NDArrayView` stays **exactly three members** (Spike-05 house rule: op methods live on the classes,
   the view never gains members).
6. **Loader:** `CoreExports` gains the five new export signatures (additive).
7. **Demo:** new section — small embedding-flavored vectors; `sub`/`mul`/`div`/`dot`/`norm`/
   `cosineSimilarity` on naive + resident, asserted bit-identical inline (`Object.is` standard). The v1
   backend deliberately does NOT appear for the new ops (frozen surface — comment says so).
8. **Tests** — see test plan.

### Out of scope (explicit)

- **SIMD/threads for elementwise or the reductions** (FOLLOWUPS: memory-bound, measure first; the
  threaded backend stays matmul-only).
- **v1 backend surface for the new ops** (`backend.ts` + `nt_add`/… stay byte-frozen; v1 is the frozen
  performance baseline of the original four ops, not a growing surface).
- **Systematic special-value injection into the differential generator** (NaN/±Inf/±0/Subnormals) — that
  is roadmap Phase B item 4, a separate slice. This phase adds *targeted fixtures* only (div-by-zero,
  zero-vector cosine) and states the coverage boundary honestly.
- **Batch/broadcast dot** (`[B,N]·[N]` etc.) — `matmul` already covers matrix/batch cases.
- **Fixing `MatMul`'s pre-existing union-dim latent hazard** (`DimEq` distributes over union dims and can
  yield a union verdict misread as `false`; discovered during this design pass) — goes to FOLLOWUPS, not
  silently fixed here. The NEW `DotCheck` filters unions from day one.
- Transcendentals (`exp`/…) — unchanged; `sqrt` is not transcendental and is explicitly on the
  IEEE-exact whitelist.

## Bit-identity law for this phase (binding)

1. **Elementwise (`sub`/`mul`/`div`):** each output element is a single IEEE op on two inputs — bit
   identity holds if iteration/indexing mirror `add_strided` exactly (they must, line-for-line).
2. **Reductions (`dot`, `norm_sq`):** ONE accumulator per result, strictly ascending logical row-major
   index order, `acc += a·b` resp. `acc += v·v`, seed `0.0`. No FMA (wasm32 scalar has none; Rust never
   auto-fuses), no reordering, no pairwise/split accumulation.
3. **Scalar compositions (`norm`, `cosineSimilarity`):** TS-side, pinned expressions above; every post-
   reduction op (`sqrt`, `*`, `/`) is IEEE-exact, so composition preserves bit identity.
4. **NaN caveat (honest):** the differential comparator is `Object.is` (the suite's standard): it
   distinguishes ±0 and treats all NaNs as equal — NaN *payloads* are not compared (WASM spec permits
   nondeterministic NaN payloads). The claim is therefore: **bit-identical for all non-NaN results;
   value-class-identical (NaN is NaN) where NaN occurs.** This matches the existing suite's de-facto
   standard and is stated in the results doc.

## Freeze discipline for this phase (binding)

This phase adds new WASM exports, so — unlike Kern 06 — the plain artifact hash **necessarily changes**
(Kern-04 precedent). The freeze claim narrows accordingly and is proven as follows:

1. **Before any Rust edit:** clean `pnpm build:wasm` (repo root!) must reproduce the pinned baseline
   SHA256 `a6622a59bc331517294f070507dfd75f8a557cee64ece431e2c847abf538ab2a` (docs/kern-06-ergebnisse.md).
   If it does not: STOP — toolchain/config drift, investigate before proceeding.
2. **Frozen files, zero diff (git):** `add.rs`, `sum.rs`, `matmul.rs`, `matmul_blocked.rs`,
   `transpose.rs`, `fill.rs`, `materialize.rs`, `shape.rs`, `backend.ts`.
3. **Append-only files:** `abi.rs`, `kernels/mod.rs`, `runtime.ts`, `loader.ts`, `resident.ts`,
   `ndarray.ts` — all new content strictly AFTER all pre-existing content (abi.rs: new entry points
   after `nt_matmul_blocked_partial`, then a NEW `#[cfg(test)] mod kern07_abi_tests` at EOF — do NOT
   insert into the existing tests module; zero mid-file inserts anywhere in abi.rs). Prefix check per
   file: `git show HEAD:<file> | cmp - <(head -n $(git show HEAD:<file> | wc -l) <file>)`.
   Exception, disclosed: `package.json` (test lists) and `demo.ts` (new section) change normally;
   `slice-literal.ts` may gain exactly the `export` keyword on the existing `IsUnion` line.
4. **After:** clean rebuild; the new plain-artifact SHA256 is measured and documented in the results doc
   as the **new pin**. Behavioral freeze evidence (the actual proof the frozen kernels still behave
   identically): `test:core` 817 green (pins v1 bit-identity), all cargo tests green, `test:resident`
   green, `pnpm demo` bit-identical, `test:threaded` 65 green (threads artifact rebuilds; RUSTFLAGS rule
   and pinned nightly per CLAUDE.md).

## Test plan (binding)

New runtime test files (both added to the **test:resident** explicit list in package.json — the
test-scripts-guard enforces registration; they need `initCore` + `WNDArray`):

1. **`spike/tests-runtime/elementwise.test.ts`** — differential, naive reference vs resident WASM path:
   - per op (`sub`/`mul`/`div`): ≥ 100 seeded cases via `genBroadcastShapes`/`genData` (own fixed seed
     per op), `assertShapeEqual` + `assertDataBitIdentical`;
   - per op: ≥ 20 strided-view operand cases (transposed and/or sliced views on either/both operands;
     reference = `elementwiseBinary` over the materialized view data);
   - div IEEE fixtures on BOTH surfaces: `[1,-1,0,42] / [0,0,0,2]` → `[+Inf,-Inf,NaN,21]` (the
     comparator's NaN semantics per the law above), plus a ±0-signedness case (`0 / -2 → -0`,
     distinguished by `Object.is`).
2. **`spike/tests-runtime/vector.test.ts`** — differential + fixtures + parity:
   - `dot`: ≥ 100 seeded 1-D pairs (lengths 0…64, including 0 and 1), naive `dotRuntime` vs
     `WNDArray.dot`, bit-compared (scalars: compare via `Object.is` and bit pattern);
   - `dot` on strided operands (sliced windows, step-2 views; reference over materialized data);
   - `norm`: ≥ 50 seeded cases across ranks 0–4, both surfaces, incl. transposed/sliced views (reference
     = `Math.sqrt(normSqRuntime(materialized))`); PLUS the absorption-pattern logical-order pin (the
     `1e100` idiom from `sum_all_strided`'s tests, adapted to squares) with its non-vacuity assert;
   - `cosineSimilarity`: ≥ 50 seeded cases; fixtures: parallel → exactly `1`, antiparallel → exactly
     `-1` (construct so den is exact), orthogonal → `0`, zero-vector → `NaN` on both surfaces;
   - parity pins: `dotRuntime(a, b)` bits == naive `a.matmul(b)` rank-0 result bits (dot ≡ matmul
     `[1,k]·[k,1]` chain); `dotRuntime(a, a)` bits == `normSqRuntime(a)` bits;
   - error paths on BOTH surfaces with the EXACT pinned messages (rank first/second, length mismatch) —
     using deliberately widened types (`as number[]`/`number` dims) so the calls compile (Spike-03
     lesson), plus disposed-handle throws for the three new WNDArray vector ops and one elementwise op.
3. **Cargo tests:** in `elementwise.rs` — the `add.rs` battery adapted (same-shape, trailing/interior
   broadcast, rank-0, size-0, incompatible, rank-too-large, contiguous-metadata ≡ naive expectation,
   transposed-view operand, offset window, broadcast-dim-1, status-4 bounds); in `vector.rs` —
   dot basic/strided-window/length-mismatch/rank-errors/size-0, norm_sq logical-order absorption pin +
   non-vacuity, dot(a,a) ≡ norm_sq(a) bits; in `abi.rs` (`kern07_abi_tests` at EOF) — garbage-rank →
   status 2 and garbage-len → status 3 per new entry point (host-safe prevalidation pattern, sentinel
   pointers only — see the existing NOTE; never reach a real kernel call natively).
4. **Type tests:** new `spike/tests/vector.test-d.ts` — dot/cosine: ok-case yields `number`; length
   mismatch → `@ts-expect-error` + message-pinning via the `Guard` property type (verbatim stems);
   rank-2 receiver and rank-2 argument each rejected; dynamic dims / dynamic rank pass; **union dim →
   no claim (passes)**; `norm()` callable on ranks 0–2 and on `number[]`, returns `number`;
   sub/mul/div: one broadcast-shape-error-at-argument pin each + result-shape `Expect<Equal<…>>` hovers
   (incl. a broadcast case, e.g. `[2,3] mul [3] → NDArray<[2,3]>`).
5. **Mutation non-vacuity proof (KB rule):** two mutants, built in an isolated target-dir copy (never
   overwriting the real artifact): (a) `div_strided`'s closure `/` → `*`; (b) `dot_strided`'s `+=` →
   `-=`. The respective differential suites MUST fail; revert MUST green. Documented in the results doc.

## Budget gates (pre-registered, ABSOLUTE — Spike-04 lesson: measurements become pins, no gates on
## guessed quantities, no stress/realistic blending)

- **G1 (hard):** `pnpm check` clean; wall clock ≤ 1.0 s (current ~0.42 s).
- **G2 (hard):** `pnpm check:diag` full-project instantiations ≤ **250,000** (= 5 % of the ~5M budget;
  current pin 188,378). The measured value becomes the new pin (CLAUDE.md + results doc). Expectation
  (recorded, NOT a gate): low-thousands increase — `DotCheck` is small, but TS7 charges declaration cost
  for template-literal-heavy aliases (~+2k/phase, observed three times).
- **G3 (hard):** `bench:editor` hard gate PASS, unchanged workload (regression pin; extending the
  workload with new-op sites is the Phase-B-2 reshape obligation, not this slice).
- **G4 (hard):** all existing suites green at their current counts (817 core / 2320+2 resident /
  65 threaded / 110 cargo) — plus the new tests; new counts recorded as pins.

## Documented differential blind spots (honest, by design)

- `assertVectorPair` (shape validation for dot/cosine) is SHARED between naive and resident — spec
  parsing/validation is shared, data paths diverge; the validator's own semantics are pinned by direct
  unit tests of the error messages (same rationale as `normalizeSliceSpecs`, Kern 05).
- The pinned scalar compositions (`norm`, `cosineSimilarity`) are structurally identical TS expressions
  on both surfaces — the differential value lies in the reductions underneath (`dot`, `norm_sq`), which
  are fully differential-tested. The compositions themselves are pinned by fixtures (exact 1/−1/0/NaN).

## Definition of Done

Spec (this doc) → implementation → all gates above → fresh-context verification (independent pass over
spec vs. diff, runs the suites itself) → results doc `docs/kern-07-ergebnisse.md` with post-verification
addendum (honesty rule: failures/gaps named) → KB capture (general lessons upserted, MOC wired, graph
rebuilt, edges verified) → FOLLOWUPS updated (MatMul union-dim hazard entered; SIMD-elementwise item
stays) → CLAUDE.md/HANDOFF pins updated → commit.
