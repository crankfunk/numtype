# NumType — Spike 03: Type-Level Bounds Checks for Literal Integer Indices (Results)

Date: 2026-07-10 · Status: complete, independently verified · Toolchain: TypeScript
7.0.2 (native), Node v24, pnpm

Per `docs/spike-03-index-bounds-spec.md`. Numbers from commands run in this session, shown below.

## Headline verdict

**A literal out-of-bounds integer index to `slice()` is now a compile error at the offending
argument** — e.g. `NDArray.zeros([2, 5]).slice(1, 5)` errors on the `5` with
`slice: index 5 is out of bounds for axis 1 with dim 5 (shape [2,5])` — for both positive and
negative literal indices (negative support costs only a comparison; the signed *addition* that
keeps negative start/stop out of the Kern-05 stretch is not needed for a bounds *check*). The
static check is **never wrong, only incomplete**: it rejects exactly the calls the runtime is
guaranteed to throw on; everything unprovable (wide `number`, dynamic dims, mixed unions,
non-plain-digit literals like `1.5`/`1e21`) passes through to the unchanged runtime backstop.
Budget cost of the machinery itself: **+3.6 % instantiations** on the pre-existing program.

## What was built

- `LiteralIndexBounds<I, D>` → `"in" | "out" | "unknown"` in `slice-literal.ts` (the file that
  owns all digit knowledge), reusing Kern 05's `Compare` — positive: `i < d`; negative: strip the
  `-` from `${I}` and require `abs(i) <= d`. Union verdicts are accepted only when EVERY member
  classifies identically (tuple-wrapped subset checks); mixed unions (`2 | 7` on dim 5) yield
  `"unknown"` — this was caught during design, not review: a naive `extends "lt"` check would
  have classified `2 | 7` as `"out"`, a false positive.
- `ValidateSpecsAcc` (`slice.ts`) — the existing argument-side guard that already flags excess
  specs — additionally retypes each provably-OOB integer-spec position to the branded
  `{ readonly __shapeError: … }` object. The axis index is `Passed["length"]` (rank-bounded
  tuple-length use, per house rules); the original shape is threaded for the message. Valid
  positions pass through byte-identically (hover/inference of existing valid calls unchanged —
  pinned by the pre-existing tests continuing to pass untouched).
- `SliceShape` untouched; runtime untouched; zero Rust.

## Immediate payoff (found during implementation)

The moment the check compiled, `pnpm check` flagged the *existing* runtime error-path test
(`tests-runtime/slice.test.ts`: `nd.slice(5)` / `nd.slice(-6)` on shape `[5]`, asserting the
runtime throw) — those literal calls are now compile errors, which is precisely the feature
working as intended: code whose only possible runtime outcome is a throw no longer compiles. The
test now widens the indices (`5 as number`) to deliberately reach the runtime backstop, and a new
parity test pins that the runtime throws exactly where the type layer rejects and succeeds
exactly where it accepts (`i = d-1` / `i = -d` fine; `i = d` / `i = -(d+1)` throw; `d = 0`
rejects everything; `1.5` still hits the runtime's own "not an integer").

## Type tests (spike/tests/slice.test-d.ts, Spike-03 section)

Boundary exactness both signs (`4/5`, `-5/-6` on dim 5), `d = 0`, multi-digit boundaries
(1023/1024, 65535/65536, ±65536/65537), conservative `"unknown"` for wide/dynamic/non-integer/
exponent/mixed-union forms, uniform-union `"out"` (`6 | 7` on dim 5 — invalid whichever member),
guard-level message + axis-index exactness (incl. axis 2 of three, and OOB + excess-spec in one
call, each with its own message), method-level error-at-the-right-argument, and gradual
pass-through (`number` index, `1.5`) all pinned.

## Budget (`pnpm check:diag`, gate: ≤ 1.25× baseline)

| state | Instantiations | vs baseline | Check time |
|---|---|---|---|
| baseline (pre-spike) | 54,700 | 1.00× | 0.033 s |
| + machinery (slice-literal.ts/slice.ts changes only) | 56,689 | **1.036×** | 0.034 s |
| + the new type tests themselves | 68,141 | 1.246× | 0.035 s |

Honest reading: the *feature* costs 3.6 %; the remaining growth to 1.246× is the ~30 new
boundary/guard type TESTS (the multi-digit cases deliberately instantiate the digit machinery
many times). Both readings sit under the spec gate; the second only barely — noted for future
type-test budgeting rather than hidden inside a single number.

## What stays runtime-only (by design)

Wide `number` indices; indices into dynamic (`number`) dims; non-integer literals (`1.5` — the
template-form classifier can't distinguish "integer in exponent form" from "non-integer" without
lying in one direction, so neither is claimed); mixed-union literals. All keep the exact
pre-existing runtime errors.

## Gates

`pnpm check` clean · `check:diag` above · `test:core` 817 · `test:resident` 2319 + 2 skips
(exactly +1: the new parity test) · `bench:editor` hard gate re-run post-change: **PASS**
(hover medians 0.04–0.08 ms, unchanged).

## Post-verification addendum (2026-07-10)

Two independent machine-checked passes beyond the test suite:

1. **Orchestrator parity grid** (scratchpad `gen-parity-grid.mjs`): 174 generated assertions —
   for d ∈ {0, 1, 2, 5, 9, 10, 11, 99, 100, 999, 1000, 1024, 65536} and i sweeping both
   boundaries ±2 plus 2⁵³−1, the expected verdict computed from the runtime rule
   (`idx = i < 0 ? i + d : i; throw unless 0 <= idx < d`) matches `LiteralIndexBounds`
   **174/174** (tsc exit 0).
2. **Fresh-context verification (brainroute:verify): CONFIRMED, spec met, no functional
   defects.** Every claimed number reproduced exactly from fresh runs, including the
   machinery-only 56,689 via a scratch reconstruction (test files at HEAD, source at new
   version). Adversarial union probes all correct-or-conservative: uniform in-bounds unions →
   `"in"`, uniform OOB → `"out"`, mixed and mixed-sign → `"unknown"` (never a false verdict for
   any member); `never` / `number & {}` → `"unknown"`; `-0` collapses to `0` (consistent with
   the runtime, where `-0 < 0` is false); `1e21` compiles with no static claim and
   `Number.isInteger(1e21) === true` confirmed in Node — the conservative-classifier rationale
   is factually right. Axis counting verified at axis 0/1/2 and axis 15 of a rank-16 shape
   (no instantiation-depth concern: 14,947 instantiations, 0.03 s). Real compiler output matches
   the documented message word-for-word, and the runtime throw stem matches `runtime.ts:293`.

**Disclosed limitation found by verification (upstream, not introduced here):** TS 7.0.2's
native checker reports only ONE diagnostic per call per compile pass when MULTIPLE argument
positions of the same call are invalid (reproduced with a plain non-generic 3-arg function too —
a general characteristic of this compiler build, equally affecting the pre-existing
too-many-specs mechanism). The computed guard type genuinely flags every bad position (pinned by
G6 directly on the type); fixing the first error surfaces the next on recompile. So "each flagged
with its own message" is true of the TYPE, while the editor shows one squiggle at a time for
multi-error calls.
