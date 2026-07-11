# NumType — Spike 06: Range Slices — Negative Literal start/stop, Literal Steps (Results)

Date: 2026-07-11 · Status: complete, independently verified · Toolchain: TypeScript 7.0.2
(native), Node v24, pnpm

Per `docs/spike-06-range-literals-spec.md`. Numbers from commands run in this session.

## Headline verdict

**The Phase-A "Kür" is closed: range slices now compute literal dims for negative literal
`start`/`stop` and literal `step ≥ 1`, and a literal step the runtime is guaranteed to throw on
(`0`, negative, dot-form non-integer like `1.5`/`-1.5`) is a compile error at the offending
argument — 280/280 parity against the REAL runtime rule** (the grid's ground truth is the
imported `normalizeSliceSpecs` itself, not a reimplementation — zero drift by construction; the
generator self-checks its invariants, e.g. every "invalid"-expected step form must actually throw
and `1e21` must not). Both spec insights held: negative normalization needs **no signed
addition** (compare + existing subtract + clamp — the Spike-03 "comparison is cheaper than
arithmetic" lesson, applied a second time), and the only genuinely new machinery is schoolbook
**long division** (`DivCeil`, most-significant-first remainder walk, ≤ 10 bounded
trial-multiplications per output digit via Spike-04's `MulDigits`).

## What was built

- `slice-literal.ts`: `ResolveBoundaryDigits` (shared start/stop resolution — negative branch via
  `Compare` + `MultiSub` with clamp-to-`"0"`, plus `never`/`IsUnion` boundary filters before any
  digit string forms), `ResolveStepDigits` (fast path for omitted/`1` — including the
  `1 | undefined` union, which is step 1 either way; plain-digit ≥ 2 → division path; everything
  else degrades), `MultiAdd`, `DivCeil` (+ `FindQuotientDigit`/`TryQuotientDigit`), and the
  exported guard classifier `LiteralStepInvalid<Spec>` (`"invalid"` only for the three provable
  forms; exponent forms like `1e21` are VALID runtime steps and stay unclaimed). The file header's
  "SUPPORTED LITERAL SUBSET" section now states the new truth, including the correction that
  negative start/stop never needed a signed add.
- `slice.ts`: `ValidateSpecsAcc` range-spec branch — a provably-invalid step retypes exactly its
  position to the branded error, message = runtime stem verbatim
  (`slice: step ${step} for axis ${axis} is invalid (must be an integer >= 1; negative steps are
  out of scope)`) + house shape suffix.
- Type tests (Spike-06 section in slice.test-d.ts): negative boundary exactness, step
  exact/non-exact/oversized divisions, multi-digit paths, guard positioning (incl. coexistence
  with Spike-03 bounds errors, pinned on the guard TYPE per the one-diagnostic-per-call TS7
  caveat), degrade pins, the union-sharpening pin. Non-vacuity proven in both directions.
- Runtime tests: the three invalid step forms were already pinned as throws — their literals are
  now compile errors (the feature working; Spike-03 pattern: widened via `0 as number`), plus ONE
  new positive parity case for a valid multi-digit step (test:resident is now **2320 + 2**, +1
  disclosed).

## Deviations disclosed

1. **Spec drafting error (mine, orchestrator): G1 said "ALL pre-existing type tests untouched" —
   impossible as written.** Five pre-existing pins (T3/T4/SB1/SB2/T24) used `{step: 2}` /
   `{start: -1}` as their canonical "degrades" examples — exactly the boundary this spike moves;
   leaving them untouched would fail `pnpm check` *because the feature works*. The executor
   updated them to still-unsupported forms (`1.5`/`-1.5`/wide `number`), preserving their intent
   ("outside the subset degrades"), and added dedicated pins for the newly-computed cases. The
   genuinely load-bearing fast-path pins (Kern-05 stretch section) are byte-untouched. Lesson for
   future specs: a spike that MOVES a subset boundary must expect the old boundary's pins to be
   re-expressed, and should say so.
2. **Post-delegation review fix (orchestrator):** `LiteralStepInvalid` initially classified
   `-1.5` as `"unknown"` (the negative branch preempted the dot-form check); the spec's dot-form
   row is sign-agnostic and `-1.5` is a guaranteed runtime throw. Fixed to `"invalid"`, pinned
   (`LSI3b`), added to the parity grid (which re-anchored the runtime-throw claim empirically);
   `-1e21` stays honestly `"unknown"` (negative AND integral in fact, unprovable from its
   `"-1e+21"` template form). The verifier corrected the original illustrative example here:
   `-1e5` renders as plain `"-100000"` and IS provable — the machinery was more complete than
   the comment claimed (sound in the safe direction; comments fixed at both source sites).
3. Union-sharpening behavior change (per spec, deliberate): union literals in start/stop/step now
   degrade at the boundary (Spike-04 rule) instead of distributing through the digit pipeline —
   observable only at exotic union-literal call sites, in the safe direction; pinned.

## Measurements

Baseline (post-Spike-05 pin): **133,727** instantiations.

| state | Instantiations | note |
|---|---|---|
| machinery + the five re-expressed pre-existing pins (M2, executor-reported mid-state) | 144,303 | the live cost: `LiteralRangeDim` is wired into every existing `slice()` test site via `SliceShape`, so this is NOT "unused machinery" |
| full tree as delivered (M3) | 188,314 | + the new Spike-06 test section |
| final after the `-1.5` review fix + `LSI3b` pin | **188,378** | = **3.77 % of the ~5M budget**; the new `check:diag` pin |

Runtime-parity grid (isolated runs): empty-stub fixed cost 5,790; full grid **231,052**
instantiations, 0.05 s, **280/280 assertions, tsc exit 0** → ≈ **805 instantiations/site**
amortized — markedly cheaper than Spike 04's product sites (~1.2–4 k), because range dims and
steps are short digit strings and the bounded trial products cache heavily.

## Gates (absolute)

| gate | result |
|---|---|
| G1 check clean · 817 · 2320+2 (+1 disclosed) · demo all-agree · Kern-05 fast-path pins untouched | **PASS** (with deviation 1 disclosed for the five boundary pins) |
| G2 bench:editor | hard gate **PASS** |
| G3 check:diag recorded, honesty split above; grid 231,052 < 20 % budget, 0.05 s < 2 s | **PASS** — new pin 188,378 |
| G4 wall-clock `pnpm check` ×5 | 0.41–0.43 s, median 0.42 s vs ~0.43 s baseline — flat |

## What stays runtime-only (by design)

Wide/dynamic/union/exponent-form inputs (gradual rule); non-integer `start`/`stop` compile
errors (guaranteed throws too, via the same dot-form insight — deferred as a symmetric follow-on,
FOLLOWUPS); negative literal steps as *computation* (the runtime itself throws — the guard covers
them instead).

## Post-verification addendum (2026-07-11)

**Fresh-context verification (brainroute:verify): CONFIRMED — no correctness or soundness defect
anywhere in the arithmetic or the guard, across ~60 independent adversarial cases plus the full
280-site grid; every measurement reproduced exactly; both disclosed deviations judged
defensible; one comment/doc inaccuracy found and fixed (the `-1e5` example above).** Details:

1. **32/32 independent semantic probes** (verifier's own combo list, expected values from a
   fresh import of the REAL `normalizeSliceSpecs`): both clamp-overshoot directions, both-negative
   crossings in both orders, `step > diff` / `step == diff`, exact and non-exact divisions,
   multi-digit steps (12/100/999/47/256/873 — deliberately beyond the grid's mostly-single-digit
   steps), `d = 0` rows, `diff = 0` with a 3-digit step, 7-digit numerators, `DivCeil` edges
   (`A < B`, exact multiple, `A == B`). Zero mismatches.
2. **Grid re-derived:** byte-identical regeneration, 280/280, 231,052 instantiations exact;
   ≥ 10 lines hand-recomputed.
3. **Guard probed structurally:** message stem re-read from runtime.ts:300 and matched verbatim;
   invalid step at position 2 of a 3-arg call lands on axis 2; Spike-03 + Spike-06 errors coexist
   in one call (pinned on the guard TYPE); valid steps pass through byte-identically.
   Non-vacuity proven BY CONSTRUCTION: a scratch copy with `LiteralStepInvalid` neutered to
   always-`"unknown"` passes specs through unchanged — the real classifier is load-bearing.
4. **Exotics:** union step `2|3` and uniformly-invalid union `0|-1` both boundary-degrade
   (LSI8-consistent); `never` fields degrade; `-0` collapses to the full-axis default; readonly
   and plain spec objects identical; the `1e20`→plain / `1e21`→`"1e+21"` template-form
   switchover confirmed empirically — matching the source comments' documented threshold.
5. **Measurements:** baseline 133,727 (stash) and final 188,378 exact; suites/demo/editor gates
   reproduced; the +1 runtime parity test confirmed real and non-vacuous (asserts sliced data,
   not just "doesn't throw"); M2 correctly labeled executor-internal.
6. **Deviations:** the five re-expressed boundary pins verified intent-preserving with ST1–ST9
   byte-untouched; the sign-agnostic dot-form fix verified sound (`-1.5` genuinely throws).
   The verifier's own correction — the `-1e5` illustrative example — is incorporated above; the
   machinery was MORE complete than its comments claimed (safe direction).

The DoD items the verifier flagged as open at its snapshot (FOLLOWUPS, KB upsert, this addendum)
are closed in this same round, before the commit.
