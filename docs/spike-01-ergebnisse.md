# NumType — Spike 01: Type-Level Shape System (Results)

Date: 2026-07-09 · Status: complete · Toolchain: TypeScript 7.0.2 (native/Go compiler), Node v24.16.0, pnpm

This spike implements a type-level NumPy-broadcast/matmul/reduce shape system in TypeScript, plus a minimal
naive runtime, per `docs/spike-01-type-layer-spec.md`. Every number in this document was produced by a command
run in this session; commands are shown so the numbers are reproducible, not asserted.

## Headline verdict

**The type layer holds up well within realistic use, and its documented TS5-era recursion limit turns out to
still apply almost unchanged on TS 7.0.2.** All 25 acceptance-table rows pass. The required stress cases
(rank-16 broadcast, a 100-op value-level chain) are cheap (tens of thousands of instantiations, well under
0.1s). Pushing far past realistic tensor ranks (empirically, not required by the spec) finds a real ceiling —
our tail-recursive `Broadcast` breaks at rank 1000, one step past the historically-documented "~1000
tail-recursive" figure for TS 5.x — so that number appears to be a property of the *tail-recursion mechanism
itself*, not something TS7's new compiler generation changed. No case had to be faked or weakened.

## Acceptance summary

| Table | Rows required | Rows passing | Notes |
|---|---|---|---|
| Broadcast | 7 | 7/7 | `spike/tests/broadcast.test-d.ts` T1–T7 (+5 extra sanity cases, T8–T12, also passing) |
| MatMul | 12 | 12/12 | `spike/tests/matmul.test-d.ts` T1–T12 (+3 extra, T13–T15, also passing) |
| Reduce/Transpose | 6 | 6/6 | `spike/tests/reduce.test-d.ts` T1–T6 (+7 extra, T7–T13, also passing) |
| Stress: rank-16 broadcast | 1 | pass | `spike/tests/limits.test-d.ts` |
| Stress: ≥100 composed ops | 1 | pass | `spike/tests/limits.test-d.ts`, exactly 100 chained ops |
| Metrics: `check:diag` (suite + minimal consumer) | 2 | recorded | see Metrics below |
| Call-site DX: add + matmul exact message | 2 | recorded | see Error messages below |
| Demo | 1 | pass | `pnpm demo`, hand-verified below |

**25/25 required acceptance-table rows pass. 0 failed. 0 faked or weakened.** Total type-level assertions in
the suite: 51 `Expect<Equal<...>>` positive checks + 4 `@ts-expect-error` negative checks = 55, all green under
`pnpm check` (verified this session, exit code 0, zero diagnostics).

No case required deviating from the spec's stated semantics. The one place a genuine design *decision* (not
dictated by the acceptance tables) was needed — `ReduceAxis` with `KeepDims=true` and no axis given — is
called out explicitly below since it's the one behavior not directly pinned down by the acceptance table.

## What was built

- `spike/src/dim.ts` — `Dim`, `Shape`, `ShapeError<Message>` (branded), `CompatDim` (broadcast per-axis rule),
  `DimEq` (strict, non-broadcast equality for matmul's contracted axis), `IsDynamicDim`, `Reverse`, `Mutable`,
  `ShowShape`.
- `spike/src/broadcast.ts` — `Broadcast<A, B>`, tail-recursive, consuming both tuples from the tail via the
  native `[...Init, Last]` variadic pattern (no separate reverse/re-reverse needed).
- `spike/src/matmul.ts` — `MatMul<A, B>`, full NumPy semantics: 2-D product, 1-D promotion (either side) with
  squeeze, batch-dim broadcasting via `Broadcast`, rank-0 rejected on either side.
- `spike/src/reduce.ts` — `ReduceAxis<S, Axis, KeepDims>` (negative axes via reuse of `Reverse`, avoiding a
  risky generic two-argument `Subtract`), `Transpose<S>` (= `Reverse<S>`).
- `spike/src/runtime.ts` — naive `Float64Array` + row-major strides runtime: generic broadcasting elementwise
  binary op, full matmul (1-D promotion + batch broadcasting), axis-sum, full-axis-reversal transpose. All
  defensively re-validate shapes at runtime (the backstop for the gradual-typing escape hatch).
- `spike/src/ndarray.ts` — the public `NDArray<S>` class: `zeros`/`ones`/`fromArray` (all `const` type params —
  no `as const` needed), `add`, `matmul`, `sum(axis?)`, `transpose`, plus the error-surfacing `Guard`/`OkShape`
  helpers (see below).
- `spike/src/index.ts` — barrel re-export.
- `spike/tests/*.test-d.ts` — one file per acceptance table + `ndarray.test-d.ts` (class-level positive/negative
  cases) + `limits.test-d.ts` (stress cases). `spike/tests/test-utils.ts` — hand-written `Equal`/`Expect`
  (no `tsd`/vitest — zero deps, `tsc --noEmit` alone is the test runner).
- `spike/bench/single-call.ts` + `spike/bench/tsconfig.json` — an isolated minimal-consumer file for the
  marginal-cost measurement, compiled via its own scoped tsconfig (see Toolchain notes for why that's needed).
- `spike/demo.ts` — runnable demo (`pnpm demo`), hand-verified below.
- Config touched: `tsconfig.json` (added `allowImportingTsExtensions: true` — required, see Toolchain notes),
  `package.json` (added `demo` and `check:diag:bench` scripts only — no dependency changes).

Total: ~692 lines of implementation (`spike/src/*.ts`), ~295 lines of type tests, plus demo/bench (measured via
`wc -l` this session).

## Error-surfacing pattern

**Chosen: argument-side "required-property" guard.** Each op's "other operand" parameter is typed as:

```ts
type Guard<Result, Actual> = Result extends ShapeError<infer Message>
  ? { readonly __shapeError: Message }
  : Actual;

add<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> { ... }
```

When shapes are incompatible, the parameter's declared type becomes an object type requiring a
`__shapeError: "<message>"` property. The actual argument (a plain `NDArray<B>`) obviously lacks that property,
so `tsc` reports a **missing-property error directly at that argument** — and because the property's type *is*
the literal message string, the message (naming the offending shapes) appears verbatim in the diagnostic. When
shapes are compatible, the conditional resolves to plain `NDArray<B>` and the call is unconstrained. Verified
empirically (this session) that generic inference of `B` from the real argument still works correctly in both
branches — see probes below.

**Alternative considered and rejected: return-type-only error surfacing.**

```ts
add2<B extends Shape>(other: NDArray<B>): Broadcast<S, B> extends ShapeError<infer M> ? ShapeError<M> : NDArray<Broadcast<S, B>>
```

Tradeoffs (verified empirically, see probe below): this is simpler to write and inference is trivially
reliable (no guard indirection needed), but **the call site itself never errors** — `a.add2(bad)` type-checks
fine. The error only appears later, wherever the (mistyped) result is first consumed as an `NDArray` — which
can be arbitrarily far from the actual mistake, especially mid-chain. This directly violates the spec's hard
constraint ("shape errors surface at the offending argument... not somewhere else"), so it was rejected as the
primary pattern, though it's simpler and worth knowing about as a fallback if a future op's shape/guard
combination proves awkward to express as a guard.

Concrete comparison (both probed in this session, `/private/tmp/.../scratchpad/probe-guard3.ts` and
`probe-guard4.ts`, not committed — throwaway probes):

```
// Guard pattern (chosen): @ts-expect-error suppresses correctly, right at the argument.
// @ts-expect-error
const r2 = a.add1(bad);        // error IS here

// Return-type-only pattern (alternative): no error at the mistake...
const r3 = a.add2(bad);        // compiles fine — the actual mistake is silent
// ...only surfaces later, wherever the result is used:
const r4: NDArray<readonly [2, 3]> = a.add2(bad);  // error is HERE instead
```

## Error messages a consumer actually sees

Captured this session by temporarily removing the `@ts-expect-error` suppressions from
`spike/tests/ndarray.test-d.ts` and running `pnpm check` (then restoring them — the file is currently green).
These are the exact, unedited `tsc` diagnostics:

**`add` shape mismatch** (`a: NDArray<[2, 3]>`, calling `.add(badAddArg)` where `badAddArg: NDArray<[4]>`):

```
spike/tests/ndarray.test-d.ts(41,7): error TS2741: Property '__shapeError' is missing in type 'NDArray<[4]>' but required in type '{ readonly __shapeError: "cannot broadcast shapes [2,3] and [4]: dims 3 and 4 are not broadcast-compatible (neither equal nor 1)"; }'.
```

**`matmul` inner-dimension mismatch** (`m1: NDArray<[2, 3]>`, calling `.matmul(badMatMulArg)` where
`badMatMulArg: NDArray<[4, 4]>`):

```
spike/tests/ndarray.test-d.ts(44,11): error TS2741: Property '__shapeError' is missing in type 'NDArray<[4, 4]>' but required in type '{ readonly __shapeError: "matmul: inner dimensions 3 and 4 do not match"; }'.
```

Bonus (also captured, same run): **`matmul` rank-0 operand**:

```
spike/tests/ndarray.test-d.ts(47,11): error TS2741: Property '__shapeError' is missing in type 'NDArray<[]>' but required in type '{ readonly __shapeError: "matmul: scalar operand (rank 0) is not allowed as the second argument (got shape [])"; }'.
```

and **`sum` out-of-range axis** (note this uses the same guard mechanism on the `axis` parameter, not just the
return type — an extension beyond what the spec required, added for consistency with add/matmul):

```
spike/tests/ndarray.test-d.ts(49,7): error TS2345: Argument of type 'number' is not assignable to parameter of type '{ readonly __shapeError: "reduce: axis 3 is out of range for shape [2,3,4] (rank 3)"; }'.
```

All four: (a) point at the correct column/argument, (b) name the actual offending shapes/dims in the message
text, matching the hard design constraint. The `matmul` inner-dimension message matches the spec's own example
format verbatim ("matmul: inner dimensions 3 and 5 do not match" — asserted exactly in
`spike/tests/matmul.test-d.ts` T15 for the `[2,3]`/`[5,4]` case).

## Metrics (`pnpm check:diag`)

All numbers from `tsc --extendedDiagnostics`, run this session. **`Instantiations` is present on TS 7.0.2** —
contrary to the task brief's flagged concern that TS7's `--extendedDiagnostics` might lack it, it's there, so
no gap to report on that front.

| Target | Files | Instantiations | Check time | Memory used | Total time |
|---|---|---|---|---|---|
| Full test suite (`pnpm check:diag`) | 78 | 26,250 | 0.011s | 35,623K | 0.031s |
| Minimal single-call consumer (`pnpm check:diag:bench`, `spike/bench/single-call.ts`: one import, one `add`, one `matmul`) | 70 | 9,494 | 0.004s | 29,525K | 0.026s |

Both are trivial in absolute terms — nowhere near any documented instantiation budget (the ~5M figure
documented for TS5, or otherwise). The marginal cost of the *whole test suite* (55 assertions across 6 files,
including a rank-16 broadcast and a 100-op chain) over the *single-call baseline* is only ~16,800 additional
instantiations and ~7ms — i.e., each additional realistic call site is cheap.

## Empirical limit-finding (beyond the required stress cases)

The spec's required stress cases (rank-16, ≥100-op chain) both pass trivially — they don't get near any real
ceiling. To actually answer "what does TS 7.0.2 do at its documented TS5-era limits" (the task's explicit ask),
I pushed our own `Broadcast<A, B>` and the value-level op chain further than the spec requires, via throwaway
probes in the session scratchpad (not committed — these are exploratory, not acceptance tests).

**Tail-recursion depth ceiling for `Broadcast`: exactly rank 999 succeeds, rank 1000 fails.**

Minimal repro (`Broadcast<A, B>` where `A`/`B` are same-length tuples of that rank, both `readonly [1,2,...,N]`):

```
rank 999:  compiles (Instantiations: 1,077,805 · Check time: 1.257s · Memory: 732,256K)
rank 1000: error TS2589: Type instantiation is excessively deep and possibly infinite.
```

This is a **near-exact match to the historically-documented TS 5.x tail-recursive ceiling** ("~1000
tail-recursive", TS PR #45711; one cited formulation in TS issue #49459 tops out at exactly 999). The
hypothesis stated in `CLAUDE.md` — that these figures were measured on TS 5.x and might not hold on TS7's new
Go-based compiler — turns out, for this specific construct, to be **false**: the ceiling is identical. This is
a useful, concrete data point: the tail-recursion elimination mechanism (and its depth limit) appears to have
carried over essentially unchanged into the new compiler generation, at least for this recursion shape.

Cost near the ceiling is **not** linear: rank 999 alone drives the whole file to over a million instantiations
and 732MB — a dramatic jump from rank 16's 35,825 instantiations (also measured this session, `rank_probe_16`
diagnostics). This tracks with the implementation's `[D, ...Acc]` accumulator pattern: prepending via tuple
spread is itself O(n) per step, making the whole recursion roughly O(n²) in the worst case — consistent with
~1000² ≈ 1M instantiations observed. **None of this matters for realistic tensor ranks** (0–16, generously up
to a few dozen for exotic batched cases) — the cost only explodes when deliberately pushed toward the ceiling,
far past anything a real array shape would need.

**Value-level op chains do not share this ceiling.** A chain of 300 and even 1000 sequential
`.matmul()/.add()/.transpose()` calls (each independently resolved, not nested inside one recursive type)
compiles in ~0.2–0.25s with a cleanly-resolved final type (`NDArray<[4, 4]>`) — no depth error, no meaningful
slowdown versus the required 100-op case. **The ~1000 ceiling is a property of recursion depth *within a
single type's computation* (e.g., one `Broadcast<A,B>` call over very high rank), not of how many independent
operations are chained at the value level.** This distinction isn't obvious from the spec/competitive-analysis
docs and is worth carrying forward: rank is the risk axis, chain length is not.

## Design decisions and honest gaps

1. **`ReduceAxis` with `KeepDims=true` and no axis (full reduction)**: not directly specified by the
   acceptance table. Implemented to match true NumPy `keepdims` semantics — reduce every axis but preserve rank
   as all-`1`s (`ReduceAxisKeepDims<[2,3,4]>` = `[1,1,1]`), not `[]`. This is a genuine design decision (not
   dictated by the spec), documented here per the honesty rule, and covered by an extra test
   (`reduce.test-d.ts` T11).
2. **`readonly` tuples and `const` type parameters**: empirically, TS's `const` type parameters *do* attach a
   `readonly` modifier to the inferred tuple (verified this session: `zeros<const S extends readonly
   number[]>([2,3,4])` infers `S = readonly [2, 3, 4]`, confirmed via a forced type-mismatch probe). This is in
   mild tension with the spec's illustrative hover example (`NDArray<[2, 4]>`, no `readonly`). Resolved by
   introducing `Mutable<T> = [...T]` (spreading a readonly tuple into a fresh tuple literal strips the
   modifier) and applying it at the three `const`-type-param entry points (`zeros`/`ones`/`fromArray`), so every
   op — from construction through every chained call — displays as a plain tuple, matching the spec's example
   exactly. `Broadcast`/`MatMul`/`ReduceAxis`/`Transpose` already produced plain tuples without needing this
   (confirmed via the same kind of probe), so `Mutable` was only needed at the three entry points.
3. **No arithmetic on dim *values***: confirmed by design — `CompatDim`/`DimEq` only ever use `extends`
   (equality) and a literal-`1` check, never addition/multiplication of dims. The one place genuine arithmetic
   appears is `ReduceAxis`'s *axis* handling (negative-axis normalization), which the spec explicitly allows
   ("rank-level counting via tuple length is fine") since axis is a rank-scale index, not a dim value. To avoid
   a risky generic two-argument `Subtract<Len, Pos>` (which can silently produce `never` for out-of-range
   inputs via a distributive-conditional pitfall — verified this is a real risk while designing it, not just a
   theoretical one), the implementation instead reuses `Reverse` for negative axes (`-k` becomes 0-based index
   `k-1` into the reversed shape) plus a trivial, always-safe `Decrement<N> = N-1`.
4. **Toolchain gaps found on TS 7.0.2, not present in the TS5-era docs**:
   - `allowImportingTsExtensions` must be set in `tsconfig.json` for `tsc` to accept the explicit `.ts`
     extensions Node 24's native TypeScript execution requires in relative imports (confirmed empirically:
     Node 24 fails to resolve `.js`-suffixed or extensionless relative specifiers when running `.ts` directly,
     `.ts`-suffixed works). This wasn't needed by any TS5-era note in the project docs and had to be discovered
     via the first `pnpm check` run.
   - TS 7.0.2 has a new diagnostic, **TS5112**: passing explicit file arguments to `tsc` when a `tsconfig.json`
     is present in the working directory is now an error ("tsconfig.json is present but will not be loaded if
     files are specified on commandline"), requiring `--ignoreConfig` or a scoped `-p <dir>` project. This
     shaped the `spike/bench/` setup (a small scoped `tsconfig.json` extending the root one) for the
     isolated single-call-site measurement.
   - `--extendedDiagnostics` **does** report `Instantiations` on TS 7.0.2 — see Metrics above; no gap here,
     contrary to the task's flagged hypothesis.
5. **`@ts-expect-error` both-directions enforcement verified empirically**: confirmed this session that an
   `@ts-expect-error` placed on a call that actually *succeeds* is itself flagged
   (`error TS2578: Unused '@ts-expect-error' directive`) — probed by temporarily mis-annotating a passing call
   in a scratch copy of `ndarray.test-d.ts`, then reverting. This is the mechanism the spec relies on for
   "negatives enforced both ways," confirmed to actually work on TS 7.0.2, not assumed.

No acceptance case was weakened, skipped, or faked. Nothing in the acceptance tables failed.

## Demo output (hand-verified)

`pnpm demo` output, captured this session:

```
A + B (bcast) shape=[2,3]
[[11,22,33],[14,25,36]]      // 1+10, 2+20, 3+30 / 4+10, 5+20, 6+30 — correct

M1 @ M2       shape=[2,2]
[[58,64],[139,154]]           // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
                               // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154 — correct

cube.sum(1)   shape=[2,4]
[[15,18,21,24],[51,54,57,60]] // batch0 col-sums of [[1,2,3,4],[5,6,7,8],[9,10,11,12]] = [15,18,21,24] — correct

A.transpose() shape=[3,2]
[[1,4],[2,5],[3,6]]           // transpose of [[1,2,3],[4,5,6]] — correct
```

All four hand-verified against the arithmetic in the comments above (and in `spike/demo.ts`).

## Recommendation

The type layer holds up: full NumPy broadcast/matmul/reduce semantics, gradual typing via a `number` escape
hatch, clean hovers (verified via strict `Equal` checks, not just eyeballing), errors that name shapes at the
offending argument, and a real (not TS5-assumed) empirical limit that only bites at unrealistic ranks (~1000+,
three orders of magnitude past anything a real array would use). This supports proceeding to the Rust/WASM
core per `FOLLOWUPS.md` — the type-level risk this spike was meant to retire is retired.

Open items carried into `FOLLOWUPS.md` (already listed there, not modified by this spike): real VS Code/tsserver
editor-latency measurement (this doc only has `tsc` timings, a documented proxy), and `reshape`/`flatten`
(needs dim-value products, explicitly out of scope here).

---

## Post-verification addendum (2026-07-09)

An independent fresh-context verification pass reproduced everything above (all acceptance rows re-counted
against the spec tables, metrics re-run to the exact instantiation count, the rank-999/1000 boundary
re-derived from fresh probes, error-at-argument behavior confirmed robust across inline/variable/generic-wrapper
call forms, runtime hand-checked on novel cases including the stride-sensitive op-after-transpose sites).
Verdict: meets the spec — with two real gaps, both addressed below.

### Gap: dynamic-RANK shapes were unsound (fixed)

The gradual-typing acceptance rows only covered per-dim dynamism inside a known-rank tuple (`[2, number]`).
Shapes whose *rank* is unknown (`number[]`, `[2, ...number[]]`) fell through every tuple destructure and
produced confidently-wrong results instead of degrading:

- `Broadcast<number[], [2, 3]>` resolved to a confident `[2, 3]` (false static certainty);
- `MatMul<number[], [3, 4]>` resolved to `never`, making `.matmul()` **uncallable** on rank-unknown arrays;
- `Transpose<number[]>` resolved to `[]` (false rank-0);
- related case found in review: a dynamic *axis* (`sum(axisVar)` with `axisVar: number`) hit `0 extends number`
  and silently removed axis 0.

Fix: a shared `IsDynamicRank<S>` probe (`number extends S["length"]`, spike/src/dim.ts) guards
`Broadcast`/`MatMul`/`ReduceAxis`/`Transpose`; all dynamic-rank (and dynamic-axis) cases now degrade to
`readonly Dim[]` — accepted, runtime-checked, never guessed. The no-axis reductions needed no guard: `[]`
(plain) and `AllOnes<S>` = `1[]` (keepdims) are statically correct for *every* rank, known or not. Covered by
16 new positive assertions (broadcast T13–T16, matmul T16–T18, reduce T14–T18, ndarray T10–T13); the suite is
now 67 positive + 5 negative assertions, all green.

### New finding: sound rank-degradation vs. erased-supertype variance (the expensive lesson)

Fixing the soundness gap *broke* `const x: NDArray<readonly number[]> = NDArray.zeros([2, 3])` — an
assignment the demo relied on. Empirical bisection (probe classes, member by member) showed the pre-fix
version only type-checked **because** the unsound degradations happened to satisfy TypeScript's variance
probe. With sound degradation, `S` occurs in method-parameter positions (the argument-side error guards), the
class's measured variance becomes invariant, and no formulation tried escapes it: the original Guard form, an
intersection form (`NDArray<B> & ErrRequire<...>`), single-root conditionals (dynamic check folded into the
recursive root), and an explicit `out S` annotation (rejected by the checker with TS2636, whose elaboration
pinpointed the mechanism) all fail the erased-supertype assignment.

**Two-of-three rule (this spike's most transferable design finding):** of (a) sound dynamic-rank degradation,
(b) shape errors at the offending argument, and (c) implicit assignability of every `NDArray<fixed>` to
`NDArray<Shape>` on the class itself — you can have at most two. We keep (a) and (b); (c) is provided
explicitly instead: `AnyNDArray` (= `NDArray<any>`, exported from spike/src/ndarray.ts), TypeScript's own
idiom for a variance-erased handle, used by the demo's `printArray` and for heterogeneous containers. The
invariance itself is pinned in ndarray.test-d.ts via `@ts-expect-error`, so any TS version or refactor that
restores implicit assignability will surface as an unused directive. A deliberate variance design (e.g. a
covariant read-only `NDArrayView<out S>` facade without guard-bearing methods) is a library-phase task —
tracked in FOLLOWUPS.md.

### Smaller verification findings

- **keepdims is type-level only**: `ReduceAxisKeepDims` exists and is tested, but `NDArray.sum()` /
  `sumRuntime()` take no keepdims parameter — there is nothing to runtime-check it against yet. Tracked in
  FOLLOWUPS.md.
- **Metrics methodology caveat**: the rank-16 standalone probe (35,825 instantiations) was compiled with
  `--ignoreConfig` and therefore a different default-lib set (85 files vs the project's 78), so its comparison
  against the 26,250-instantiation suite figure is apples-to-oranges in the low digits. The non-linear-blowup
  conclusion near the ~1000 ceiling is unaffected.

### Updated metrics (post-fix, run this session)

Full suite: **21,151 instantiations / 0.008s check / 35.2MB** — *cheaper* than pre-fix (26,250), because the
eager rank guards short-circuit recursion that previously ran to exhaustion. `pnpm check` green,
`pnpm demo` output unchanged and correct.

Correction (2026-07-09, found during Kern 01 verification): the single-call bench figure in the Metrics
table above (9,494 instantiations) was measured BEFORE the dynamic-rank fix and never re-measured when this
addendum updated the suite figure. Post-fix it is **7,425** (stable across runs; same 70-file compile scope,
confirmed via `--listFiles`). Same cause as the suite drop: the eager rank guards short-circuit recursion.
