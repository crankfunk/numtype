# NumType — Kern 05: Slicing at the TS Surface (Results)

Date: 2026-07-10 · Spec: docs/kern-05-slicing-spec.md · Status: complete, independently verified

Process note: implementation by a delegated executor (Sonnet 5, xhigh) against the binding spec; two
independent review layers (orchestrator line-by-line review of the type arithmetic + fresh-context
verification with its own adversarial probe harness). All numbers from commands run on 2026-07-10.

## Summary against acceptance criteria

- `pnpm check` clean incl. the new type tests. **`check:diag`: baseline 29,464 Instantiations → final
  46,855 (1.59×)**, wall time flat (~0.04 s) — the stretch's ≤3×/≤2× gates hold comfortably (the verifier
  re-measured the baseline independently in an isolated git worktree at the Kern-04 commit).
- `cargo test` 76/76 with **zero crate diff** — the phase confirmed the Kern-03 prediction that the ABI
  was already slicing-ready; not one line of Rust changed.
- `pnpm test:core` 791/791 · `pnpm test:resident` **2316 + 2 honest GC skips** (was 1578+2; +738 new:
  fixture table, 150-case core differential, ops-on-sliced-operands, composition, lifecycle, error paths,
  and the post-verification broadcast-with-offset additions) · `test:resident:gc` 2/2 · demo green incl.
  the `A.slice(1).sum()` showcase.
- `pnpm bench:slice` two runs: the resident view path is 1.17–1.35× over naive TS slice+sum in both
  series (contiguous row-block and step-2 strided).
- Zero new dependencies; package.json lists updated; frozen files byte-for-byte untouched.

## What was built

**Runtime:** `slice(...specs)` on both classes — one spec per leading axis (`number` = index, drops the
axis, throws on OOB; `null` = full axis; `{start?, stop?, step?}` = clamping range, step ≥ 1). NumPy
semantics pinned as a fixture table in the spec and tested verbatim against the shared normalizer
(`normalizeSliceSpecs`, used by both backends — a deliberate, documented differential blind spot: the
backends share spec *parsing* and diverge in *data movement*, which is what the 180+ differential cases
cover). `NDArray.slice` gathers a fresh copy; **`WNDArray.slice` is an O(1) view** (shared refcounted
buffer, same mechanism as `transpose()`) and is the first operation to produce **nonzero offsets** — the
ABI capability built in Kern 03 and exercised end-to-end from TS for the first time here, through every
strided kernel and the blocked matmul.

**Type layer (the phase's research core):**
- Core rules (`spike/src/slice.ts`): integer spec drops its axis statically — even when the index is a
  non-literal `number` (the rank effect never depends on the value); `null` keeps the literal dim; range
  objects degrade to `number`; trailing axes preserved; wide `S`/wide `Specs` degrade wholly, checked
  first; too-many-specs is a compile error **at the excess argument**, naming rank and count.
- **Stretch goal: landed.** `spike/src/slice-literal.ts` computes **literal dims** for range specs with
  non-negative literal `start`/`stop` (or omitted) and `step` omitted-or-`1`, via from-scratch
  digit-string arithmetic: number → `${N}` template string → digit-wise borrow subtraction and
  length-then-lexicographic comparison → back via `infer N extends number`. Cost is O(digit count), never
  O(value) — the project's "rank is fine, value is not" rule pushed one level deeper. Examples that now
  resolve statically: `NDArray<[1024]>.slice({start: 100, stop: 1000})` hovers as `NDArray<[900]>`;
  clamping (`{start: 1, stop: 3}` on dim 2 → `[1]`) and empty results (`{start: 5, stop: 2}` → `[0]`)
  match the runtime formula exactly. Everything outside the subset (negative literals, step ≠ 1, wide
  values, dynamic dims) degrades to `number` — identical to the core rule, so the stretch is a strict
  refinement, droppable by reverting one call site.

## Gotchas (with evidence — each cost real debugging time or was caught by a guard)

- **`never` as a failure sentinel silently succeeds:** `never extends X` is true for every X, so a
  digit-arithmetic helper "failing" with `never` would satisfy all downstream checks and produce a wrong
  literal. The fix is a string sentinel (`"unsupported"`) that can never equal a `${number}` — documented
  in `NonNegDigits`'s comment after being confirmed while prototyping.
- **Optional-property inference yields `unknown`, not `undefined`:** `Spec extends {start?: infer T}`
  against `{}` infers `T = unknown` — the required-property pattern plus an explicit `: undefined` outer
  branch is the correct way to detect "genuinely absent" (verified independently by the verifier).
- **`Guard<Result, Actual>` is illegal on rest parameters (TS2370):** a conditional type that *can*
  resolve to a non-array collapses the method's own declaration, breaking every call. The working
  mechanism is a homomorphic mapped type (`ErrorTuple`) that stays a tuple at every step and retypes only
  the excess positions — error text lands on the first excess argument. Reproduced independently.
- **Non-integer slice components must be rejected eagerly** (`Number.isInteger` on index/start/stop/step):
  a fractional value would produce fractional offsets/strides that silently corrupt every strided read
  downstream. NumPy rejects the same inputs. Untested at first (verify finding), now covered.

## Bench (`pnpm bench:slice`, two runs)

Slicing itself is O(1) metadata; the honest claim measured is the pipeline effect. Resident view
slice+sum vs naive TS slice+sum: Series A (row-block, contiguous consumption) 1.19–1.35×; Series B
(step-2, strided consumption) 1.17–1.22×. Consistent across runs and sizes (n = 256/512/1024).

## Deviations from spec (with reasons)

- `Number.isInteger` validation beyond the fixture table (executor-flagged; adopted — see gotchas).
- The original ops-on-sliced-operands differentials used same-shape/same-spec pairs — deliberate
  determinism, but it left broadcasting between *differently*-sliced operands and batch-broadcast matmul
  with nonzero offsets uncovered (verify finding 2; the verifier's own probes confirmed no bug). Closed
  post-verification with 30 dedicated seeded cases.
- Stretch scoped to non-negative literals (no signed add) — documented in `slice-literal.ts`'s header;
  negative literals degrade honestly. A follow-up could add signed normalization (FOLLOWUPS.md).

## Open issues

FOLLOWUPS.md: type-level bounds-checking of literal integer indices against literal dims is now
*feasible* (the digit `Compare` exists) — deferred as its own decision; signed-literal support for the
stretch; the digit machinery also reopens the previously-deferred `reshape`/`flatten` question (products
would be O(digits²) schoolbook multiplication — a design decision, not a limit, now).

## Post-verification addendum (2026-07-10)

Fresh-context verification returned **"Kern 05 meets its spec"** with the explicit verdict that the digit
arithmetic is **correct** — verified with the verifier's own 25-case adversarial probe harness (borrow
chains 10−9, 1000−999, 10000−1, 505−406; clamps; degrade boundaries), including a non-vacuity check of the
harness itself (deliberately wrong assertions must fail — they did), plus an independent reproduction of
the TS2370 claim, the optional-property-inference claim, and the error-at-argument behavior (error column
lands on the excess argument). Slicing semantics were cross-checked against **real NumPy**. Both minor
findings (untested `Number.isInteger` paths; the same-spec differential narrowing) were closed with test
additions in the same session — suite 2285 → 2316. Orchestrator review had separately added borrow-chain
type-test pins (1000−1=999 through three zeros; 100−99=1 with leading-zero strip) before verification.
