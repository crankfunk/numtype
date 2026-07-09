# NumType — Spike 01: Type-Level Shape System (Spec)

Date: 2026-07-09 · Status: in progress

## Why (intent)

NumType's USP is compile-time shape checking. The competitive analysis (docs/wettbewerbsanalyse-und-usp.md) established that this territory is unclaimed and that the risk is engineering, not market: prior art (potatogpt, sebinsua) stops at literal-dimension matmul, explicitly without broadcasting, and strains the TS checker. This spike answers the load-bearing question **before** any Rust/WASM work: does a type-level shape system with full NumPy broadcast/matmul/reduce semantics hold up within TypeScript's documented limits, with acceptable DX? If it fails, the project pivots — so honest negative results are as valuable as passing tests.

## Scope

Pure type-level implementation plus a minimal naive runtime so a demo actually executes. **Zero dependencies** (dev-only: `typescript`). Everything from scratch.

### In scope

1. **Core types** (`spike/src/`):
   - `Dim` = `number` (literal `2 | 3 | …` when statically known, plain `number` when dynamic), `Shape` = `readonly Dim[]`.
   - `Broadcast<A, B>` — NumPy broadcasting: align from the right; dims compatible iff equal or one is `1`; result dim is the non-1 dim.
   - `MatMul<A, B>` — full NumPy `matmul` semantics: 2-D matrix product; 1-D promotion (prepend to A / append to B, then squeeze); batch dims broadcast; scalars (rank 0) are an error.
   - `ReduceAxis<S, Axis>` (removes the axis; supports negative axes; keepdims variant replaces with `1`; no axis → `[]`).
   - `Transpose<S>` (reverse).
   - **Gradual typing**: a plain `number` dim is "dynamic" — it never produces a compile error; it propagates (`Broadcast<[2, number], [2, 3]>` = `[2, number]`; a dynamic inner dim in matmul is accepted, checked at runtime).
   - **Error channel**: a branded `ShapeError<Message extends string>` carrying a human-readable template-literal message that names the offending shapes/dims.
2. **Function surface** (`spike/src/ndarray.ts`): `NDArray<S extends Shape>` + `zeros`/`ones`/`fromArray`, `add` (broadcasting elementwise), `matmul`, `sum(axis?)`, `transpose`. Use **`const` type parameters** so callers never write `as const`. Minimal naive runtime (plain loops over `Float64Array`, correct strides/broadcasting) — correctness only, no performance work.
3. **Type tests** (`spike/tests/*.test-d.ts`): positive cases via hand-written `Expect<Equal<X, Y>>` helpers; negative cases via `@ts-expect-error` (with `tsc --noEmit`, an unused directive is itself an error, so negatives are enforced both ways).
4. **Stress & metrics** (`spike/tests/limits.test-d.ts`, `spike/bench/`): see acceptance criteria.
5. **Findings** (`docs/spike-01-ergebnisse.md`, English): what worked, what hit which limit (with minimal repro), metrics table, error-surfacing pattern chosen and at least one alternative considered with tradeoffs.
6. **Demo** (`spike/demo.ts`): builds arrays, broadcasts an `add`, runs a `matmul`, prints results. Node 24 is installed — run TS directly (`node spike/demo.ts`); wire a `pnpm demo` script.

### Out of scope

`reshape`/`flatten` (needs type-level products of large dims — deferred, see FOLLOWUPS.md), Rust/WASM, dtypes beyond f64, fancy indexing, runtime performance.

## Hard design constraints

- **All recursive types tail-recursive/accumulator-based** (~1000-iteration ceiling vs ~100 non-tail; TS PR #45711).
- **No arithmetic on dim values** — only equality and `1`-detection. Rank-level counting via tuple length is fine (ranks are small).
- Callers never need `as const` (use `const` type params, TS 5.0+).
- Shape errors surface **at the offending argument**, message names the shapes (e.g. `matmul: inner dimensions 3 and 5 do not match`).
- Hover/inferred types resolve to clean tuples (`NDArray<[2, 4]>`), not unevaluated conditional-type soup.
- Large literal dims (e.g. `1048576`) must work — they are just literals; nothing may iterate over a dim's magnitude.

## Acceptance criteria

All of the following in type tests, `pnpm check` green, no unused `@ts-expect-error`:

### Broadcast

| A | B | Expected |
|---|---|---|
| `[2,3]` | `[3]` | `[2,3]` |
| `[8,1,6,1]` | `[7,1,5]` | `[8,7,6,5]` |
| `[256,256,3]` | `[3]` | `[256,256,3]` |
| `[]` | `[2,3]` | `[2,3]` |
| `[2,number]` | `[2,3]` | `[2,number]` (gradual, no error) |
| `[2,3]` | `[4]` | ShapeError |
| `[5,4]` | `[2,4]` | ShapeError |

### MatMul

| A | B | Expected |
|---|---|---|
| `[2,3]` | `[3,4]` | `[2,4]` |
| `[3]` | `[3]` | `[]` |
| `[3]` | `[3,4]` | `[4]` |
| `[2,3]` | `[3]` | `[2]` |
| `[10,2,3]` | `[10,3,4]` | `[10,2,4]` |
| `[10,2,3]` | `[3,4]` | `[10,2,4]` |
| `[2,1,2,3]` | `[7,3,4]` | `[2,7,2,4]` |
| `[number,3]` | `[3,4]` | `[number,4]` (gradual) |
| `[2,number]` | `[3,4]` | accepted (dynamic inner dim → runtime check), result `[2,4]` |
| `[2,3]` | `[4,4]` | ShapeError |
| `[]` | `[3,3]` | ShapeError (rank 0) |
| `[2,2,3]` | `[3,3,4]` | ShapeError (batch dims incompatible) |

### Reduce / Transpose

| Input | Expected |
|---|---|
| `ReduceAxis<[2,3,4], 1>` | `[2,4]` |
| `ReduceAxis<[2,3,4], -1>` | `[2,3]` |
| keepdims `<[2,3,4], 1>` | `[2,1,4]` |
| no axis `<[2,3,4]>` | `[]` |
| `ReduceAxis<[2,3,4], 3>` | ShapeError (out of range) |
| `Transpose<[2,3,4]>` | `[4,3,2]` |

### Stress & metrics (record actual numbers in docs/spike-01-ergebnisse.md)

- Rank-16 broadcast type-checks.
- A chain of **≥100 composed ops** (alternating add/matmul/transpose on the value level, letting inference thread the shape through) type-checks without depth errors.
- `pnpm check:diag` (`--extendedDiagnostics`): record **Instantiations** (vs the ~5M budget), **Check time**, **Memory used** — once for the test suite, once for a minimal consumer file, so we know the marginal cost per call site.
- Call-site error DX: for at least matmul and add, verify (and paste into the findings doc) the actual error message a consumer sees on a mismatch — it must name the dims, and it must point at the offending argument, not somewhere else.

### Demo

`pnpm demo` runs and prints correct numeric results for a broadcast add and a 2-D matmul (verifiable by hand, e.g. small integer matrices).

## Honesty rule

If a specific acceptance case proves impossible or pathological within TS's limits, do **not** fake or weaken it silently: implement the closest workable semantics, mark the case clearly in the findings doc with a minimal repro and the precise limit hit. The spike's purpose is truth-finding, not a green checkmark.
