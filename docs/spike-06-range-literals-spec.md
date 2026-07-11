# NumType — Spike 06: Range Slices — Negative Literal start/stop, Literal Steps (Spec)

Date: 2026-07-11 · Status: in progress

## Why (intent)

Roadmap A4 (the Phase-A "Kür") / FOLLOWUPS: the Kern-05 stretch computes literal slice dims only
for non-negative literal `start`/`stop` with `step` omitted-or-`1`; everything else honestly
degrades to `Dim`. The authoritative runtime rule (`normalizeAxisSpec`, runtime.ts:283) is richer:
negative `start`/`stop` normalize by `+d` and CLAMP to `[0, d]` (never throw); `step` must be an
integer ≥ 1 (else THROW — negative steps are out of scope since Kern 03);
`dim = max(0, ceil((stop − start) / step))`. This spike closes the gap in both directions:

- **Compute** the literal dim for negative literal `start`/`stop` and for literal `step ≥ 1`.
- **Reject at compile time** (Spike-03 idiom: error at the argument) a literal `step` the runtime
  is GUARANTEED to throw on — provably `< 1` or provably non-integer.

Two scoping insights, established up front:

1. **No signed addition needed after all** (correcting Kern 05's header note): `v + d` for a
   negative literal `v` is `d − |v|` when `|v| ≤ d`, and clamps to `0` when `|v| > d` — one
   `Compare` + the existing `MultiSub`. The Spike-03 lesson ("comparison is cheaper than
   arithmetic — check what the use case really needs") applies a second time.
2. The only genuinely new machinery is **schoolbook long division** (for `ceil(diff / step)`),
   O(digits(diff) × digits(step)) via per-position trial multiplication with Spike-04's
   `MulDigits` (quotient digit = largest q ∈ 0..9 with q·step ≤ remainder-prefix; ≤ 10 bounded
   trials per output digit, instantiation-cached across positions/sites). `ceil` = floor-division
   plus 1 iff the remainder is nonzero (needs a normal-form `MultiAdd` wrapper around Spike-04's
   `AddRev` for the +1).

## Semantics (binding — mirrors `normalizeAxisSpec` exactly)

For a range spec against a literal dim `d`, with all of `start`/`stop`/`step` in the supported
literal subset (see below):

| input | type-level result | mirrors runtime |
|---|---|---|
| `start` negative literal, `\|start\| ≤ d` | `start' = d − \|start\|` | `start += d` |
| `start` negative literal, `\|start\| > d` | `start' = 0` | clamp |
| `start` non-negative literal | `start' = min(start, d)` (existing) | clamp |
| `stop` — same three rows | `stop'` analogous | same |
| `step` omitted or literal `1` | `dim = start' < stop' ? stop' − start' : 0` (existing fast path, unchanged results) | ceil with step 1 |
| `step` literal integer ≥ 2 (plain digits) | `dim = ceil((stop' − start') / step)`, `0` when `start' ≥ stop'` | the ceil formula |
| `step` literal provably invalid: plain-digit `0`, negative (`-${digits}`), or dot-form non-integer (`${T}` contains `.` and no `e` — integers never render with a dot) | **compile error at that argument** (guard, see mechanism) | runtime THROWS `slice: step … is invalid (must be an integer >= 1; negative steps are out of scope)` |
| any of start/stop/step: wide `number`, union, exponent-form (`1e21` IS a valid integer step — flagging it would lie), or dynamic `d` | degrade to `Dim` / no guard claim | gradual rule |

Never-wrong-only-incomplete throughout: the guard rejects ONLY guaranteed runtime throws; the
computation claims a literal ONLY inside the provable subset; everything else keeps the exact
pre-existing runtime behavior.

**Union rule alignment (deliberate behavior sharpening of the Kern-05 stretch):** the existing
stretch lets union literals distribute through `NonNegDigits` into the digit pipeline
(unaudited-by-construction). Per the Spike-04 boundary rule (a computed dim is an unbounded VALUE
verdict — subset checks don't apply), this spike adds `IsUnion`/`[T] extends [never]` boundary
filters to `start`/`stop`/`step` resolution: union or `never` inputs now degrade to `Dim`
uniformly. Only exotic union-literal call sites can observe the change, in the safe direction
(computed union dim → honest `number`). Documented as a behavior change, pinned by a test.

## Mechanism (binding)

- **slice-literal.ts** (all digit knowledge): new internal machinery appended below the Spike-04
  block — `MultiAdd<A, B>` (normal-form wrapper over `AddRev`, mirror of `MultiSub`),
  `DivCeil<A, B>` (schoolbook long division, most-significant-first remainder walk, trial
  `MulDigits(B, q)` per quotient digit, then +1 via `MultiAdd` iff remainder ≠ "0"; PRECONDITION
  `B ≥ 1` established by callers). `LiteralRangeDim` itself is MODIFIED (first deliberate edit of
  Kern-05-era type code — TS files carry no freeze discipline; behavior is pinned by tests):
  `ResolveStart`/`ResolveStop` gain the negative branch (strip `-`, Compare, `MultiSub`-or-"0")
  and the boundary filters; a `ResolveStep` classifies the step (`"1"`-fast-path / plain-digit
  ≥ 2 / `"invalid"` / `"unsupported"`); `ComputeRangeDigits` gains the `DivCeil` branch. The step
  fast path must keep every currently-computed result byte-identical (existing stretch tests
  unchanged). The file-header "SUPPORTED LITERAL SUBSET" section is updated to the new truth.
- New export `LiteralStepInvalid<Spec>` → `"invalid" | "unknown"` (classifier for the guard;
  `"invalid"` ONLY for the three provable forms in the table). **slice.ts**: `ValidateSpecsAcc`
  gains a range-spec branch mirroring the Spike-03 integer branch: a provably-invalid step
  retypes THAT position to the branded error with the message
  `slice: step ${step} for axis ${axis} is invalid (must be an integer >= 1; negative steps are out of scope) (shape ${ShowShape<S>})`
  — runtime stem verbatim + house shape suffix, axis = `Passed["length"]` as in Spike 03.
- No runtime changes. Zero Rust. No new files.

## Gates (absolute — the Spike-04/05 lesson: gate absolutes, pin measurements)

| # | Gate |
|---|---|
| G1 | `pnpm check` clean · `test:core` 817 · `test:resident` 2319+2 · `pnpm demo` all-agree · ALL pre-existing type tests untouched and green (the Kern-05 stretch pins prove the fast path didn't move) |
| G2 | `pnpm bench:editor` hard gate PASS |
| G3 | `check:diag` recorded vs the 133,727 pin: machinery-only figure and with-tests figure separately (honesty split); the final figure becomes the new pin. Grid run total < 20% of the ~5M budget and grid check time < 2 s |
| G4 | 5-run wall-clock `pnpm check` median flat vs a same-session baseline (±10%; the tsc-internal Check-time field is noise at this scale — Spike-05 methodology note) |

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Type tests (slice.test-d.ts, appended section): negative boundary exactness (`start: -1` on d=5 → dim 1 with stop d; `-5` → full 5; `-6` → clamp 0 …), both-negative and mixed start/stop, step boundaries (step 2/3 on exact and non-exact divisions, step ≥ d, step 1 via both omitted and explicit), `d = 0`, big multi-digit dims on both sides, guard errors for step `0` / negative / `1.5` at the RIGHT argument (incl. a later-position case and coexistence with Spike-03 bounds errors in one call), degrades pinned (`number` step, union start, exponent step `1e21` compiles with NO claim), and the union-sharpening pin (union start now → `Dim`). Non-vacuity both directions. |
| 2 | Runtime parity grid (scratch, orchestrator-run): the generator imports the REAL `normalizeAxisSpec`/`normalizeSliceSpecs` from `spike/src/runtime.ts` (Node type-stripping) as ground truth — zero reimplementation drift; ≥ 250 generated `(d, start, stop, step)` sites sweeping negatives around ±d, clamp boundaries, step ∈ {1,2,3,7,d−1,d,d+1}, multi-digit dims; expected literal (or `number` for out-of-subset combos) computed from the runtime's returned `dim`; invalid-step combos assert the guard type; tsc exit 0. |
| 3 | Runtime parity for the guard: the exact statically-rejected step forms throw at runtime, statically-accepted boundary steps succeed (extend tests-runtime/slice.test.ts only if a pinning gap exists — widened literals per the Spike-03 pattern). |
| 4 | Gates G1–G4 recorded; fresh-context verification before "done"; results doc with addendum; KB upsert; FOLLOWUPS updated (Kern-05-stretch item out; non-integer start/stop guard as symmetric follow-on with the dot-form insight). |

## Out of scope

Negative literal steps (runtime throws — covered by the guard, not by computation); non-integer
`start`/`stop` compile errors (guaranteed throws too, via the same dot-form insight — deferred as
a symmetric follow-on, FOLLOWUPS, to keep this slice's guard surface step-only); `newaxis`/
ellipsis/masks; any runtime change; index-spec paths (Spike 03, untouched).
