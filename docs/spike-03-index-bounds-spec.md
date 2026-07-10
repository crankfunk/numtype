# NumType — Spike 03: Type-Level Bounds Checks for Literal Integer Indices (Spec)

Date: 2026-07-10 · Status: in progress

## Why (intent)

Roadmap Phase A2 / FOLLOWUPS: since Kern 05, `slice()` with an **integer spec** throws at runtime
when the index is out of bounds (NumPy IndexError analog — integer indices do NOT clamp; range
specs DO clamp and are never errors, so ranges are explicitly not this spike's subject). Kern 05
deferred the static half: with a **literal** index against a **literal** dim, validity is fully
decided at compile time, and the digit-string `Compare` machinery to decide it has existed since
Kern 05. This spike closes that gap.

**Design decision (the FOLLOWUPS item's open question "compile error vs runtime-only"):
COMPILE ERROR, at the offending argument.** Justification: the runtime semantics are already
fixed (throw); a static rejection therefore only rejects calls that are *guaranteed* to throw at
runtime — zero false positives by construction. This is the project's USP applied literally:
a guaranteed production crash becomes an editor squiggle. Everything outside the provable subset
passes through unchanged (gradual; the runtime backstop stays untouched).

## Semantics (binding)

For each spec position where the spec is an integer and the axis dim `d` is a literal:

| index `i` (literal) | verdict | mirrors runtime |
|---|---|---|
| plain non-negative integer literal, `i < d` | OK (unchanged) | no throw |
| plain non-negative integer literal, `i >= d` | **compile error at that argument** | `slice: index i is out of bounds for axis a with dim d` |
| negative integer literal, `abs(i) <= d` | OK (i normalizes to `i + d`) | no throw |
| negative integer literal, `abs(i) > d` | **compile error at that argument** | throw |
| `d = 0`, any literal integer | **compile error** (no valid index exists) | throw |
| wide `number` index, or dynamic dim `d` | pass through (runtime backstop) | gradual rule |
| any other literal form (non-integer `1.5`, exponent form `1e21`, union `2 \| 7`) | pass through (runtime backstop) | conservative |

The last row is deliberate: the static layer must be **never wrong, only incomplete**. Literals
whose template form is not plain digits (after an optional leading `-`) get NO static claim —
e.g. `1e21` IS an integer (would hit the runtime OOB branch, not the non-integer branch), so
flagging it statically as "not an integer" would lie; unions distribute through the digit
machinery and are simply not asserted about. All such cases keep the existing runtime behavior.

Error message convention (house rule: at the offending argument, shapes/values named):
`slice: index ${i} is out of bounds for axis ${axis} with dim ${d} (shape ${ShowShape<S>})` —
same wording stem as the runtime throw, extended with the shape for editor context.

## Mechanism (binding)

- **slice-literal.ts** (owner of all digit knowledge) gains one export,
  `LiteralIndexBounds<I, D>` → `"in" | "out" | "unknown"`:
  dynamic `D` or non-number/wide `I` → `"unknown"`; `${I}` = `-${Abs}` with `Abs` plain digits →
  `Compare<Abs, ${D}>` `"gt"` ? `"out"` : `"in"`; `${I}` plain digits → `Compare<${I}, ${D}>`
  `"lt"` ? `"in"` : `"out"`; anything else `"unknown"`. Cost: O(digit count) per check — the same
  cost class as the Kern-05 stretch, no signed arithmetic needed (a bounds check is a comparison;
  the signed *addition* that kept negative start/stop out of the Kern-05 stretch is not required).
- **slice.ts**: `ValidateSpecsAcc` (the existing argument-side guard that already retypes excess
  positions) additionally binds each axis head dim and, for an integer spec `Head` with
  `LiteralIndexBounds<Head, SHead> extends "out"`, retypes THAT position to the branded
  `{ readonly __shapeError: <message> }` object — the identical error-surfacing idiom as
  too-many-specs. Valid positions keep passing `Head` through untouched (hover/inference of every
  currently-valid call byte-identical). `SliceShape` is NOT touched (a flagged call doesn't
  compile; its shape is irrelevant — same division of labor as today).
- No runtime changes. No changes to broadcast.ts/matmul.ts/reduce.ts/dim.ts. Zero Rust.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Type tests (slice.test-d.ts, house idioms): boundary exactness both sides (`i = d-1` ok / `i = d` error; `i = -d` ok / `i = -(d+1)` error), `d = 0` errors for `0` and `-1`, big dims (1024, 65536) on both sides of the boundary (digit-count > 1 paths), error lands at the RIGHT argument (later positions unaffected), wide index / dynamic dim / non-plain-digit literals pass through, valid-call hover unchanged. |
| 2 | Runtime parity pinned: the exact type-level boundary cases from #1 exercised against `sliceRuntime`/`slice()` at runtime — every statically-rejected case throws, every statically-accepted boundary case succeeds (add to tests-runtime/slice.test.ts if not already present). |
| 3 | `pnpm check` clean; `check:diag` recorded before/after — **gate: Instantiations ≤ 1.25× the pre-spike baseline** on the unchanged main program (the check only adds O(digits) work per integer-spec-against-literal-dim site). |
| 4 | `pnpm bench:editor` re-run: hard gate still PASS (hover ≤ 100 ms — expected trivially; recorded, not assumed). |
| 5 | `pnpm test:core` / `test:resident` unchanged green (817 / 2318+2). |
| 6 | Fresh-context verification before "done"; results doc `docs/spike-03-ergebnisse.md` (honesty rule, incl. what stays runtime-only and why); KB decision; FOLLOWUPS updated. |

## Out of scope

Bounds checks for range `start`/`stop` (they clamp — not errors, already statically computed
where literal); non-integer-literal rejection (conservative pass-through, see table); negative
literal start/stop dim *computation* (separate FOLLOWUPS item — needs signed addition);
`newaxis`/ellipsis/masks; any runtime change.
