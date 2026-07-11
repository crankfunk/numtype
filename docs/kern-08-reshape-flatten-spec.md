# NumType — Kern 08: Runtime reshape/flatten + Editor-Hover Measurement Obligation (Spec)

Date: 2026-07-11 · Status: binding, pre-implementation (roadmap Phase B, item 5 remainder)

## Why (intent)

Spike 04 built and measured the enabler (`LiteralShapeProduct<S>`: exact type-level shape products via
digit-string multiplication, never-wrong by construction) and explicitly deferred three things to this
phase: the runtime `reshape`/`flatten` methods, their guard/error-message wording (the compile message
must mirror a runtime throw that didn't exist yet), and the FOLLOWUPS **measurement obligation** — the
editor-hover cost of REAL `reshape()`/`flatten()` call sites, measurable only once the methods exist
(`bench:editor` workload extension). This phase delivers all three. **Zero Rust changes are expected**
(reshape is pure metadata; the materialize path has existed since Kern 03) — the plain-artifact hash
must therefore stay IDENTICAL (the strong freeze form, unlike Kern 07's export-adding mode).

## Op semantics (binding)

### `reshape(newShape)` — same elements, new shape

`a.reshape(ns)` for a shape `ns` whose element product equals `a`'s element product. NumPy semantics
minus `-1` inference (see out of scope).

- **Dim validity:** every dim of `ns` must be a non-negative integer (0 allowed — size-0 shapes are
  first-class in this codebase). Violations throw (message table). `-1` is just a negative dim here.
- **Product equality:** `product(ns) === product(a.shape)`, else throw (message table). Size-0 arrays
  reshape into any size-0 shape (`0 === 0`); rank-0 (`[]`, size 1) reshapes to any size-1 shape and
  vice versa.
- **Data order:** logical row-major order is preserved (the flat sequence is invariant) — for the naive
  reference this is literally a copy of `data`; for a resident non-contiguous view it is the
  materialized logical order.
- **Naive `NDArray`:** always a fresh copy (house invariant: `NDArray` never aliases).
- **Resident `WNDArray`:** **view if contiguous** (O(1): `refs + 1`, same buffer, offset 0, natural
  row-major strides of the new shape), **materialize-copy otherwise** (`nt_materialize` into a fresh
  buffer, then wrap with the new shape). Semantically indistinguishable (WNDArray exposes no mutation);
  observable only through memory/lifecycle — tests pin the routing via `describe().ptr` identity and
  the refcount/lifecycle behavior, exactly like transpose/slice views.
- Return type: `NDArray<Mutable<NS>>` resp. `WNDArray<Mutable<NS>>` (clean hover `NDArray<[3, 2]>`,
  `const NS` type param so callers never write `as const`).

### `flatten()` — rank-1 copy/view of all elements

`a.flatten()` ≡ `a.reshape([product(a.shape)])` in behavior (always valid, no error paths; rank-0 →
`[1]`, size-0 → `[0]`). Return type is the Spike-04 payoff: `NDArray<[LiteralShapeProduct<S>]>` — a
**statically computed literal** rank-1 shape when every dim is a supported literal (hover
`NDArray<[1048576]>` for `[1024, 1024]`), degrading to `NDArray<[number]>` (honest rank-1, dynamic dim)
whenever the product degrades. Same view-if-contiguous/else-materialize routing on `WNDArray`.

### Error messages (binding, runtime stems — compile-time mirrors verbatim via `ShowShape`)

| # | Condition | Message |
|---|-----------|---------|
| 1 | some dim of `ns` is not a non-negative integer | `` `reshape: invalid dimension ${d} in shape [${ns.join(",")}] (dims must be non-negative integers)` `` |
| 2 | products differ | `` `reshape: cannot reshape array of size ${size} into shape [${ns.join(",")}]` `` |

Checked in this order (dim validity first, per-axis left to right, then product). `flatten` throws
never. The runtime validator is one shared function used by both surfaces (documented differential
blind spot, same rationale as `assertVectorPair`/`normalizeSliceSpecs`).

## Type layer (binding)

New guard in a new file `spike/src/reshape.ts` (keeps `slice-literal.ts` untouched for the core; see
stretch), consumed by both classes:

`ReshapeCheck<S, NS>` — never-wrong-only-incomplete, in this decision order:

1. `P_old = LiteralShapeProduct<S>`, `P_new = LiteralShapeProduct<NS>`.
2. If either product is wide (`number extends P`) → **no claim** (pass; runtime backstop). This
   subsumes dynamic rank, dynamic dims, negative/non-integer/exponent-form dims, over-cap products —
   `LiteralShapeProduct` already degrades all of those.
3. If either product is a **union** (`IsUnion<P>`) → **no claim** (a product verdict is an unbounded
   value — the Spike-04 rule: no subset-check possible, filter at the boundary; a union of fixed shapes
   deliberately distributes inside `LiteralShapeProduct`, so this case is reachable).
4. Both plain literals: equal → pass; unequal → `ShapeError` with the **verbatim** stem
   `` `reshape: cannot reshape array of size ${P_old} into shape ${ShowShape<NS>}` ``.

Method signatures (Guard-at-argument, existing idiom):
`reshape<const NS extends Shape>(shape: Guard<ReshapeCheck<S, NS>, NS>): NDArray<Mutable<NS>>` (resp.
`WNDArray`). `flatten(): NDArray<[LiteralShapeProduct<S>]>` — niladic, **no guard** (always valid; same
reasoning as `norm()` in Kern 07).

Known, documented, deliberately-not-fixed boundary (FOLLOWUPS, two-facet union item): a union of whole
OPERAND types can bypass argument-side guards in general; and per-guard, step 3 above means literal
union-of-shapes arguments get no claim — pinned by type tests as no-claim, not silently assumed
covered.

### Stretch (droppable independently, Spike-04 pattern — dropping it is not a phase failure)

`LiteralReshapeDimInvalid<NS>`: lift **provably invalid literal dims** of the new shape to a compile
error at the argument (Spike-03/06 idiom, reusing the existing classification primitives appended in
`slice-literal.ts`): a negative literal (template form `` `-${...}` ``) or a dot-form literal (proven
non-integer, sign-agnostic) is a GUARANTEED runtime throw → error with the verbatim message-1 stem.
NO claim for exponent forms (`1e21` renders as `"1e+21"` — could be a valid integer), `number`, unions
(filter first), `0` (VALID here, unlike slice `step`). Checked before the product check (mirrors the
runtime order). If budget gates strain, drop the stretch and keep the core; record the decision.

## Scope

### In scope

1. Shared runtime validator + reshape logic (`runtime.ts`, **append-only**): `assertReshapeArgs(oldShape,
   newShape): void` (throws messages 1/2), plus whatever pure helper the naive path needs. Existing
   lines byte-for-byte untouched.
2. `NDArray.reshape`/`NDArray.flatten` (`ndarray.ts`, **insertion-only** — the Kern-07 rule: new members
   next to their closest analog, zero edits to pre-existing members, plus the import-line additions).
3. `WNDArray.reshape`/`WNDArray.flatten` (`resident.ts`, **insertion-only**): view-if-contiguous
   (reuse the private `isContiguous()`), else `nt_materialize` (existing entry point, scratch-list/
   `finally` discipline verbatim). `assertLive` first; validation before any allocation.
4. Type layer: `spike/src/reshape.ts` (new) with `ReshapeCheck`; stretch classifier appended to
   `slice-literal.ts` (append-only, its historical discipline) if kept.
5. **bench:editor workload extension (the measurement obligation):** one new workload in
   `spike/bench-dx/gen-workloads.ts` with REAL `reshape()`/`flatten()` call sites — at minimum: a small
   reshape hover (`NDArray<[3, 2]>`), a flatten hover with computed literal (`NDArray<[6]>`), a
   big-dim flatten hover exercising digit multiplication (`[1024, 1024]` → `NDArray<[1048576]>`), and a
   diagnostic TOGGLE flipping a product mismatch on/off (same missing-property mechanism the existing
   toggle uses). Manifest expectations hand-computed in plain JS (the generator's house rule: mirror
   the type rules by hand, never via a TS compiler call). Harness (`editor-latency.ts`) thresholds
   stay UNCHANGED; the new workload must pass the existing hard gate, and its hover medians +
   per-workload instantiation count become pins in the results doc — closing the FOLLOWUPS obligation.
6. Demo: small reshape/flatten section (naive + resident, asserted bit-identical, showing the computed
   flatten hover shape in a comment).
7. Tests — see test plan.

### Out of scope (explicit)

- **`-1` dim inference** (NumPy `reshape(-1, k)`): needs exact division + remainder check at the type
  level and a different runtime validation order. Deliberately deferred — new FOLLOWUPS item; `flatten`
  covers the dominant flatten-to-1-D case.
- Rust/ABI changes of any kind (none needed; artifact hash must prove it).
- `keepdims` (next slice), order parameters (`order='F'`), `ravel` (view-alias semantics on the naive
  surface contradict the no-alias house invariant; `flatten` is the honest primitive).
- Fixing the whole-shape-union guard bypass (FOLLOWUPS, systemic).

## Freeze discipline for this phase (binding — the STRONG form again)

1. Zero Rust diffs (`git diff` empty on all of `crates/`). Clean `pnpm build:wasm` BEFORE and AFTER
   must both reproduce the pinned hash
   `7a65d80062865a5e88952ce3cfbdd974b642f6d3f4b293e3f3b39afad16885d8` — identical, not re-pinned.
2. Per-file discipline (the Kern-07 rule — named per file, in the spec, up front): `runtime.ts`,
   `slice-literal.ts` (stretch only) **append-only** (prefix-`cmp` proof); `ndarray.ts`, `resident.ts`
   **insertion-only** (zero edits to pre-existing members beyond import lines); `loader.ts`,
   `backend.ts`, all frozen Rust files **untouched**; `package.json` (test list), `demo.ts`,
   `gen-workloads.ts` change normally (`gen-workloads.ts` is bench infrastructure, not a frozen
   reference — but its determinism rule holds: regenerated output must be byte-identical run-to-run).
3. `editor-latency.ts` (the harness) should need NO changes; if a change turns out to be genuinely
   required, it is a disclosed deviation with reasoning, and the pre-existing workloads' gate results
   must be shown unchanged.

## Test plan (binding)

New runtime test file `spike/tests-runtime/reshape.test.ts` (register in **test:resident**; guard
enforces):

1. **Naive ⇄ resident parity (differential value = data movement/routing, not shape math):** ≥ 60
   seeded cases across ranks 0–4 (shapes from a seeded generator, new shape = a seeded permutation/
   regrouping of the same product incl. added/removed 1-dims), `assertShapeEqual` +
   `assertDataBitIdentical(naive.data, resident.toArray())` for both `reshape` and `flatten`.
2. **View-routing pins (resident):** contiguous reshape/flatten shares the buffer
   (`describe().ptr` identical, `residentFreeCount` unchanged until last handle disposed; base
   dispose leaves the view usable — the transpose/slice lifecycle idioms); NON-contiguous (transposed
   and sliced-with-step views) routes through materialize (fresh ptr) and matches the naive reference
   computed over the materialized logical order — bit-identical.
3. **Edge cases:** rank-0 ↔ `[1]`/`[1,1]`; size-0 (`[0,3]` → `[0]`, `[3,0]` → `[0,5]` valid); identity
   reshape; flatten of rank-0 → `[1]`, of size-0 → `[0]`.
4. **Error paths, EXACT pinned messages on BOTH surfaces** (widened types so calls compile, Spike-03
   lesson): message 1 (negative dim, non-integer dim — incl. a `-1` case documenting the deferral) and
   message 2 (product mismatch); dim-validity-before-product ORDER pinned (a shape that violates both
   reports message 1); disposed-handle throws for both new WNDArray methods.
5. **Type tests** (`spike/tests/reshape.test-d.ts`): ok-case hovers (`Expect<Equal<…, NDArray<[3, 2]>>>`,
   flatten literal product incl. a big multi-digit case and the MAX_SAFE_INTEGER-adjacent cap cases
   from product.test-d.ts reused at the METHOD level); mismatch → `@ts-expect-error` + verbatim
   message pin via the Guard property type; dynamic dims/rank pass; union-of-shapes argument → no
   claim (pass, documented); over-cap product → no claim; stretch (if kept): negative/dot-form dim
   compile errors verbatim + runtime parity (same boundary, both levels), exponent forms pass.
6. **Guard non-vacuity:** at least one deliberately wrong `Expect` must fail when flipped (harness
   idiom), and the mismatch toggle in the new bench workload doubles as an end-to-end proof the guard
   fires in a real editor program.
7. **Measurement obligation (its own results-doc section):** `pnpm bench:editor` with the new workload —
   report hover medians per new position, toggle latency, per-workload instantiations; hard gate PASS
   required; numbers become pins. Compare the big-dim flatten hover against the existing Kern-05
   digit-arithmetic stress positions to state honestly whether reshape sites are in-family or an
   outlier.

No new cargo tests (no Rust changes). No mutation proof (no new kernels — the differential content is
routing/lifecycle, covered by the ptr/refcount pins; state this explicitly in the results doc rather
than performing an empty ritual).

## Budget gates (pre-registered, ABSOLUTE)

- **G1 (hard):** `pnpm check` clean, wall ≤ 1.0 s.
- **G2 (hard):** `check:diag` ≤ **250,000** instantiations (current pin 200,714). Measured value
  becomes the new pin. Expectation (recorded, NOT a gate): a few thousand — guard + type tests +
  method declarations.
- **G3 (hard):** `bench:editor` hard gate PASS **including the new workload**; pre-existing workloads'
  results not regressed past the harness thresholds.
- **G4 (hard):** suites green: test:core 817 · test:resident 3092+2 + new reshape tests · cargo 157 ·
  threaded 65 · demo bit-identical. Artifact hash **identical** (`7a65d800…`).

## Definition of Done

Spec → implementation → gates → fresh-context verify → results doc `docs/kern-08-ergebnisse.md` with
post-verification addendum (incl. the measurement-obligation section) → KB capture → FOLLOWUPS updates
(close the two Spike-04 obligations: hover measurement + guard wording; add `-1`-inference item) →
CLAUDE.md/HANDOFF pins → commit + push.
