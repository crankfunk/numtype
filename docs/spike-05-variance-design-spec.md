# NumType — Spike 05: Variance Design (roadmap A3) (Spec)

Date: 2026-07-11 · Status: in progress

## Why (intent)

Roadmap A3 / FOLLOWUPS: `NDArray<S>` is deliberately **invariant** — the Spike-01 two-of-three
rule: of (a) sound dynamic-rank degradation, (b) errors at the offending argument, (c) implicit
assignability of `NDArray<fixed>` to `NDArray<Shape>`, at most two are possible; we keep (a)+(b).
(c) is currently served by `AnyNDArray = NDArray<any>` — an escape hatch that erases checking in
BOTH directions. This decision must land before any API freeze (roadmap release gate). This spike
decides the deliberate variance design and ships its minimal implementation.

## Probe evidence (run this session, TS 7.0.2, against the real class — scratchpad
`variance-probes{,-2}.ts`)

| probe | result |
|---|---|
| `interface View<out S>` with only read members (`readonly shape: S`, `strides()`, `toNestedArray()`) | annotation accepted; `View<[2,3]> → View<Shape>` widens ✓ |
| real `NDArray<[2,3]>` assignable to `View<Shape>` structurally (no `implements`), incl. heterogeneous `View<Shape>[]` | ✓ |
| shape-COMPUTING member under `out` (`transpose(): View<Transpose<S>>`) | **TS2636** — the annotation check is abstract (sub-S/super-S) and `Transpose` is not *provably* monotone (dynamic-rank degrade) |
| same computed member WITHOUT annotation | widening passes (structural check on concrete instantiations — `Transpose` is *factually* monotone) — i.e. covariance-by-accident, unenforced |
| wide-return (`transpose(): View<readonly number[]>`) and self-return (`self(): View<S>`) under `out` | both accepted ✓ |
| generic free-function inference through the view (`materialize<S>(v: View<S>)`) | infers exact `S` ✓ |
| downcast `View<Shape> → View<[2,3]>` | correctly an error ✓ |
| existing `NDArray` invariance pins | unaffected ✓ |

## Decision (binding)

**Option 1 — an annotated, minimal, checker-enforced covariant read view.** Rejected
alternatives, with reasons:

- *Unannotated view with computed members* (works empirically): covariance-by-accident is
  exactly the failure mode Spike 01 taught us to reject ("only type-checked because the unsound
  degradations happened to satisfy the variance probe") — unenforced, silently breakable by any
  future member, and only pinnable point-wise by tests. The `out` annotation IS the regression
  pin, at the declaration, for every instantiation at once.
- *Annotated view with wide-returning ops* (`transpose(): View<readonly number[]>`): enforced,
  but throws away shape precision — on a library whose USP is precise shapes, a precision-losing
  surface invites exactly the confusion the view exists to remove.

Design:

1. **`NDArrayView<out S extends Shape>`** (interface, type-only, zero runtime), exported from
   `ndarray.ts`, with EXACTLY three members: `readonly shape: S` · `strides(): number[]` ·
   `toNestedArray(): unknown`. No `data` field (Float64Array is naive-backend-specific; the
   member list must stay satisfiable by `WNDArray`-style resident arrays for the Phase-C/D
   backend-choice API — adopting it there is out of scope here, tracked in FOLLOWUPS). No ops:
   guard-bearing members are impossible (two-of-three) and computed-shape members are
   annotation-hostile (probe); consumers that compute stay generic over `NDArray<S>` (generic
   functions never needed variance).
2. `class NDArray<S>` declares **`implements NDArrayView<S>`** — the drift alarm on the class
   side (if the abstract implements-check rejects this, fallback: keep the structural
   relationship and pin it with an assignability type test; document which one landed).
3. **`AnyNDArray` stays**, re-documented as the deliberately-unsafe both-ways escape hatch for
   the rare heterogeneous-AND-op-calling case; `NDArrayView<Shape>` is the recommended safe top
   type for read access. The demo's `printArray` (today: `AnyNDArray`) migrates to
   `NDArrayView<Shape>` as the acceptance demonstration.
4. Type tests (extend `spike/tests/ndarray.test-d.ts`, house idioms): widening from literal view
   and from real arrays (incl. `NDArrayView<Shape>[]` container), downcast `@ts-expect-error`,
   generic inference through the view pinned via `Equal`, and the EXISTING `NDArray` invariance
   pins unchanged. The `out` annotation needs no extra pin — it is enforced at every compile.

## Gates (absolute — Spike-04 KB lesson: gate absolutes, pin measurements)

| # | Gate |
|---|---|
| G1 | `pnpm check` clean · `test:core` 817 · `test:resident` 2319+2 · `pnpm demo` runs asserted-equal after the printArray migration |
| G2 | `pnpm bench:editor` hard gate PASS |
| G3 | `check:diag` recorded vs the 133,656 pin; the new figure BECOMES the pin (expected ≈ +0 — an interface is not template-literal machinery; the number is recorded either way, not gated on a guess) |
| G4 | wall-time medians flat (5-run, ±10%) |

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Design implemented per Decision 1–4; only `ndarray.ts`, `spike/tests/ndarray.test-d.ts`, and the demo file change. Zero runtime behavior change (interface + one type in a signature). |
| 2 | Gates G1–G4 measured and recorded. |
| 3 | Fresh-context verification before "done"; results doc `docs/spike-05-ergebnisse.md` (honesty rule; records which implements-vs-structural variant landed and the option matrix); KB upsert (sharpen the two-of-three note: guard-free but shape-computing members ALSO break annotated covariance; annotation check = abstract, structural check = concrete); FOLLOWUPS updated (A3 out; new item: backends adopt `NDArrayView` in Phase C/D). |

## Out of scope

`WNDArray`/threaded surfaces adopting the view (Phase C/D, FOLLOWUPS); any op or write member on
the view; a runtime view class (the interface is type-only by design); `materialize()` (probe
proved inference works, but no consumer needs it — generic `NDArray<S>` parameters already cover
compute-consumers; do not ship speculative API).
