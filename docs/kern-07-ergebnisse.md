# NumType — Kern 07: Elementwise Family + Vector Ops — Ergebnisse

Date: 2026-07-11 · Status: complete, independently verified (fresh-context pass: CONFIRMED)
Spec: docs/kern-07-elementwise-vector-spec.md (binding, pre-registered gates)

## Summary

Phase B item 1 landed: broadcasting `sub`/`mul`/`div` and the embedding primitives `dot`/`norm`/
`cosineSimilarity` on both surfaces (naive `NDArray` reference + resident `WNDArray`), backed by three
new strided elementwise kernels (one generic loop, monomorphized) and two new reduction kernels
(`dot_strided`, `norm_sq_strided`) behind five appended ABI entry points. `norm` and `cosineSimilarity`
needed **no kernels of their own**: `sqrt`/`*`/`/` are IEEE-exact (JS ⇄ WASM), so both are pinned TS-side
scalar compositions over the two reductions — bit-identity reduces to the reductions, which are fully
differential-tested. One small new type-level guard (`DotCheck` in spike/src/vector.ts) surfaces rank/
length errors at the argument with runtime-verbatim messages; union dims degrade to no-claim from day one
(Spike-04/06 rule).

## What was built (files)

- `crates/core/src/kernels/elementwise.rs` — generic `binary_strided<F>` mirroring `add_strided`
  line-for-line + `sub_strided`/`mul_strided`/`div_strided`; 25 cargo tests.
- `crates/core/src/kernels/vector.rs` — `dot_strided` (rank-1×rank-1, single ascending accumulator),
  `norm_sq_strided` (any rank, logical row-major, `acc += v*v`); 13 cargo tests incl. absorption-pattern
  order pin with non-vacuity assert.
- `crates/core/src/abi.rs` (append-only, verified) — `nt_sub_strided`/`nt_mul_strided`/`nt_div_strided`
  (nt_add_strided convention incl. hardened prevalidation), `nt_dot_strided`/`nt_norm_sq_strided`
  (implicit 1-f64 output, nt_sum_all_strided convention), new `mod kern07_abi_tests` at EOF (10 tests).
  No cfg gates — the ops exist in both artifacts. No new status codes.
- `spike/src/runtime.ts` (append-only, verified) — `assertVectorPair` (shared validator, documented
  differential blind spot à la `normalizeSliceSpecs`), `dotRuntime`, `normSqRuntime`.
- `spike/src/vector.ts` (new) — `DotCheck<S, B, Op>`; deliberately does NOT reuse `DimEq` (pre-existing
  union hazard, see FOLLOWUPS) — `VectorLenCheck` filters union dims via `IsUnion` before comparing.
- `spike/src/ndarray.ts` / `spike/src/wasm/resident.ts` — six new methods each (`sub`/`mul`/`div` mirror
  `add` with pinned closures resp. new entry points; `dot`/`norm`/`cosineSimilarity` per the pinned
  compositions; scratch-list/`finally` discipline, fresh views after last allocation). Insertion-only
  diffs (see deviation note below). `NDArrayView` unchanged (still exactly 3 members).
- `spike/src/wasm/loader.ts` (append-only, verified) — second merged `interface CoreExports` block with
  the five new signatures; original block untouched.
- `spike/src/slice-literal.ts` — exactly one edit: `type IsUnion` → `export type IsUnion`.
- Tests: `spike/tests-runtime/elementwise.test.ts` (454), `spike/tests-runtime/vector.test.ts` (318),
  `spike/tests/vector.test-d.ts` (type-level, verbatim message pins). Both runtime files registered in
  the `test:resident` explicit list (test-scripts-guard green).
- `spike/demo.ts` — new embedding-flavored section, naive + resident asserted bit-identical; v1 backend
  deliberately absent for the new ops (frozen surface).
- `spike/tests-runtime/backend-oom.test.ts` — 5 mechanical `notImplemented` stubs so its hand-written
  `CoreExports` mock satisfies the larger interface (same pattern as Kern 03/04 additions).

## API design decisions (documented)

- `dot`/`norm`/`cosineSimilarity` return **plain `number`** — deliberately asymmetric with `sum()`
  (→ `NDArray<[]>`): reductions that stay chainable vs. scalar consumer ops that terminate a chain.
  NumPy precedent (`np.dot` on vectors, `np.linalg.norm` → scalar); embedding pipelines consume plain
  numbers. Alternative (`NDArray<[]>`) rejected: duplicates `matmul` `[N]·[N]` and adds `.toNestedArray()`
  noise at every consumer.
- `norm()` is Frobenius/L2 over ALL elements, any rank (NumPy default) — which also dissolves the
  guard problem for a niladic method (no argument to hang a `Guard` on).
- `div` and `cosineSimilarity` are pure IEEE (no epsilon guards, no warns): `x/0 → ±Inf`, `0/0 → NaN`,
  zero-vector cosine → `NaN`, adversarial underflow can produce `±Inf`. Documented, fixture-tested.

## Gates (pre-registered in the spec; measured, all PASS — implementer + verifier independently)

| Gate | Rule | Measured | Verdict |
|---|---|---|---|
| G1 `pnpm check` | clean, ≤ 1.0 s | clean, 0.5–0.68 s | PASS |
| G2 `check:diag` | ≤ 250,000 instantiations | **200,714** (new pin; was 188,378 → Δ +12,336) | PASS |
| G3 `bench:editor` | hard gate PASS | PASS | PASS |
| G4 suites | all green | core **817** · resident **3092+2** (Δ +772) · cargo **157** · threaded **65** · demo bit-identical | PASS |

Honest notes on G2: the spec's recorded *expectation* was "low-thousands" — the measured Δ is ~+12.3k
(DotCheck machinery + a full new type-test file + six new generic methods on two classes). The
expectation was explicitly NOT a gate (Spike-04 lesson); the absolute gate (≤ 250k = 5 % of budget)
passes with wide margin; 200,714 is the new pin. The implementer observed one run at 199,974; the
verifier measured 200,714 three times identically — pin set to the stable value, single-run variance
noted as an unreproduced one-off.

Pre-existing doc drift found and fixed: the cargo baseline was actually **109** tests, not the 110
CLAUDE.md documented (verified by arithmetic: 157 − 48 new = 109; no destructive checkout needed).

## Freeze evidence (this phase's narrowed claim, per spec)

New WASM exports mean the plain artifact hash necessarily changes (Kern-04 precedent) — the freeze claim
is at the source/behavior level and was proven as:

1. **Baseline reproduced pre-edit:** clean `pnpm build:wasm` → `a6622a59bc331517294f070507dfd75f8a55
   7cee64ece431e2c847abf538ab2a` (matches the Kern-06 pin) BEFORE any Rust change (implementer;
   the verifier accepted this on the Kern-06 doc rather than re-derive it — re-deriving would need a
   destructive working-tree revert; honest boundary).
2. **Frozen files zero-diff** (git): add.rs, sum.rs, matmul.rs, matmul_blocked.rs, transpose.rs,
   fill.rs, materialize.rs, shape.rs, backend.ts — verified by implementer AND verifier.
3. **Append-only prefix checks** (`cmp` against the HEAD line count): abi.rs, kernels/mod.rs,
   runtime.ts, loader.ts — pure appends, zero mid-file inserts (abi.rs's pre-existing `mod tests`
   untouched; new tests live in `kern07_abi_tests` at EOF).
4. **New artifact pin:** clean rebuild →
   `7a65d80062865a5e88952ce3cfbdd974b642f6d3f4b293e3f3b39afad16885d8` — reproduced independently by the
   verifier from a fresh `rm -rf target` rebuild. This is the frozen-baseline hash from Kern 07 onward.
5. **Behavioral freeze:** test:core 817 (pins v1 bit-identity), cargo 157, resident 3092+2, threaded 65,
   demo bit-identical — all green.

**Disclosed deviation (verifier: intent-preserving):** the spec listed ndarray.ts/resident.ts as
"append-only", but both classes have private constructors — a literal post-class append is structurally
impossible without breaking the nominal-constructor invariant or turning ops into free functions (which
the spec's own `a.dot(b)` phrasing contradicts). The six methods were inserted next to their closest
analogs; verifier confirmed the diffs are insertion-only plus one import-line edit per file, zero
pre-existing lines changed otherwise. Lesson recorded: append-only discipline is an *artifact-bytes*
argument and belongs to compiled/frozen-reference files; for TS class bodies the intent-preserving
equivalent is "insertion-only, zero edits to pre-existing members" — future specs should say that.

## Bit-identity & mutation proofs (non-vacuity)

- Comparator standard (unchanged, now explicitly documented): per-element `Object.is` — bit-identical
  for all non-NaN values (±0 distinguished), NaN compared as a value class (WASM spec permits
  nondeterministic NaN payloads; payloads are deliberately not claimed).
- Differential coverage: 360 broadcast + 90 strided-view elementwise cases; dot 110 + 30 strided;
  norm 60 + 35 view cases + absorption-pattern logical-order pin (with non-vacuity assert:
  memory-order bits ≠ logical-order bits); cosine 60 + exact fixtures (parallel → exactly 1,
  antiparallel → −1, orthogonal → 0, zero-vector → NaN). Parity pins: `dotRuntime` ≡ naive matmul
  `[N]·[N]` scalar bits; `dot(a,a)` ≡ `normSq(a)` bits.
- Mutation proofs, all in isolated copies (real artifact never touched, hash restored after):
  - implementer mutant (a): `div_strided` `/`→`*` → 151/454 elementwise tests fail; revert green.
  - implementer mutant (b): `dot_strided` `+=`→`-=` → 201/318 vector tests fail; revert green.
  - **verifier's own mutant** (per KB doctrine): `mul_strided` `*`→`+` → 150/454 fail; revert green,
    hash back to the 7a65d800… pin.
- Coverage boundary (honest, unchanged from the spec): the differential generator still produces only
  normal finite values — systematic NaN/±Inf/±0/subnormal injection remains roadmap Phase B item 4.
  This phase's special-value evidence is fixture-level only (div-by-zero incl. `-0` signedness,
  zero-vector cosine).

## Post-verification addendum (fresh-context pass, CONFIRMED)

Independent verifier reproduced every load-bearing claim itself (all suites, both hashes via clean
rebuild, its own mutation probe) and audited spec conformance item by item. Findings, none blocking:

1. **(major, pre-existing, systemic — NOT introduced by Kern 07)** Union-of-WHOLE-SHAPES bypasses the
   argument-side `Guard` pattern: probed empirically, an operand typed e.g. `NDArray<[2,3]> |
   NDArray<[7,3]>` can slip through guards that are sound for each member alone — reproduces on the
   PRE-EXISTING `add` path too, so it is a property of the codebase's `Guard`/distribution pattern, not
   of the new ops. Kern 07's union handling covers union DIMS (no-claim, verified); whole-shape unions
   are a distinct, broader gap → FOLLOWUPS entry extended accordingly.
2. **(minor)** G2 delta commentary — see the honest note under Gates (expectation missed, gate passed).
3. **(minor, pre-existing)** cargo baseline doc drift 109 vs "110" — fixed in CLAUDE.md with this phase.
4. **(nit)** `WNDArray.cosineSimilarity` validates the operand pair once itself and once inside its
   internal dot path — double validation, negligible cost, kept for simplicity.

Verifier's stated boundary: could not re-derive the PRE-edit baseline-hash reproduction without a
destructive revert; relied on the committed Kern-06 pin for that step. All other claims independently
reproduced.
