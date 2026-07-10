# NumType — Spike 04: Type-Level Shape Products (reshape/flatten enabler) (Spec)

Date: 2026-07-10 · Status: in progress

## Why (intent)

Roadmap Phase A2 / FOLLOWUPS: `reshape`/`flatten` need the **product of a shape's dims** at the
type level (`flatten` result shape `[Product<S>]`; `reshape` validity = product equality). Since
Kern 05 the digit-string machinery makes this *feasible in principle* — schoolbook multiplication
is O(digit-count²) — but it has never been built or measured. This spike is the **budget
decision** the FOLLOWUPS item calls for: implement the multiplication, measure what it costs, and
issue a pre-registered GO/NO-GO. The runtime `reshape`/`flatten` methods themselves are Phase-B
work (roadmap B5) and are **out of scope** here — this spike ships the arithmetic those methods
will consume, not the methods.

Parity target note: Spike 03 mirrored an *existing* runtime throw. Here there is no runtime
`reshape` yet, so the ground truth for "never wrong" is **arithmetic itself**: a claimed literal
product must equal the exact mathematical product (machine-checked against BigInt), and every
unprovable case must make no claim. No runtime-throw mirroring is possible or claimed.

## Semantics (binding)

One new export, `LiteralShapeProduct<S extends Shape>` → `Dim`, in `slice-literal.ts` (the file
that owns all digit knowledge — Spike-03 precedent; renaming the file to reflect its grown scope
is a Phase-D packaging concern, not this spike's):

| shape `S` | result | why |
|---|---|---|
| every dim a non-union, non-negative, plain-digit integer literal; exact product ≤ 9 007 199 254 740 991 | **the literal product** | fully decidable |
| empty shape `[]` | `1` | empty product; matches NumPy's scalar size |
| any dim `0` (other dims arbitrary literals, even huge) | `0` | digit strings are exact at any length; the cap applies to the FINAL value only |
| exact product > 9 007 199 254 740 991 (`Number.MAX_SAFE_INTEGER`) | `number` (no claim) | round-tripping a >2⁵³−1 digit string through `` `${infer N extends number}` `` yields a double-rounded literal ≠ the true product — a WRONG claim. Degrade instead (also: `2⁵³` itself degrades — the simple `Number.isSafeInteger` line, not a representability special case) |
| dynamic rank (`number[]`, variadic) | `number` | gradual rule, checked FIRST before any tuple recursion |
| any dynamic (`number`) dim | `number` | gradual rule |
| any union dim (e.g. `2 \| 3`), any `never` dim | `number` (no claim) | see union rule below |
| any negative / non-integer / exponent-form (`1e21`) literal dim | `number` | outside the plain-digit subset (existing `NonNegDigits` sentinel) |

A union of fixed shapes (`[2,3] | [4]`) distributes at the tuple walk and yields the union of
per-member results (`6 | 4`) — each member runs the pipeline independently; pinned by test.

**Union rule (binding, differs from Spike 03 deliberately):** `LiteralIndexBounds` could accept
union verdicts via tuple-wrapped SUBSET checks because its verdict alphabet is finite
(`"in" | "out" | "unknown"`). A product's "verdict" is an unbounded VALUE — a union result cannot
be subset-validated against anything. So unions are excluded at the boundary (`IsUnion` on each
dim, plus `[Head] extends [never]` → no claim), and the digit pipeline is **union-free by
construction**: no distribution can occur anywhere inside the arithmetic. This is the whole
safety argument, and it must hold structurally (no naked union ever reaches `ReverseStr`/
`AddRev`/`MulDigits`).

## Mechanism (binding)

Appended to `slice-literal.ts` strictly below all existing content, reusing (never duplicating)
`Digit`/`DigitValue`/`Tup`/`TupDigit`/`ReverseStr`/`StripLeadingZeros`/`Compare`/`NonNegDigits`:

- `AddDigit<A, B, CarryIn>` → `[digit, carryOut]` — the carry mirror of `SubDigit`, via bounded
  tuple concat (length ≤ 19, split on `Tup<10>`).
- `AddRev<ARev, BRev, CarryIn, Acc>` — multi-digit addition on REVERSED strings, mirror of
  `SubRev` incl. unequal lengths; a surviving final carry appends `"1"`.
- `MulAdd<A, B, CarryIn>` → `[digit, carryDigit]` — digit×digit+carry via bounded tuples
  (value ≤ 89: `MulTup` = repeated concat, recursion depth ≤ 9; split by repeated `Tup<10>`
  peeling, ≤ 8 peels). Never a 100-entry lookup table, never proportional to a dim value.
- `MulDigitRev<ARev, B>` — reversed string × single digit, threading the carry.
- schoolbook accumulation, tail-recursive over `B`'s digits with a zero-string shift accumulator
  (shift-left = PREFIX zeros in reversed space):
  `MulAccRev<ARev, BRev, Shift = "", Acc = "0">`.
- `MulDigits<A, B>` = strip(reverse(MulAccRev(reverse A, reverse B))) — `StripLeadingZeros`
  is mandatory (×0 paths produce `"000…0"`).
- `LiteralShapeProduct<S>`: wide-rank guard first, then a tail-recursive `ProductAcc<S, Acc="1">`
  over the tuple (rank-bounded walk, the allowed kind of tuple recursion) with per-dim filters
  in this order: `[Head] extends [never]` → `number`; `IsUnion<Head>` → `number`;
  `NonNegDigits<Head>` = `"unsupported"` → `number` (early exit, no arithmetic on the rest);
  else `ProductAcc<Rest, MulDigits<Acc, digits>>`. At exhaustion:
  `Compare<Acc, "9007199254740991"> extends "gt"` → `number`, else back via
  `` Acc extends `${infer N extends number}` `` (existing sentinel discipline: failure states are
  strings like `"unsupported"`, never `never`).
- Cost: O(len(A)·len(B)) digit steps per multiplication, O(rank) multiplications per shape.
- No runtime changes. No changes to any other file. Zero Rust. `slice.ts` untouched.

## Budget gates (pre-registered — this spike IS the budget decision)

Baseline, measured fresh this session: **68,141 instantiations, check 0.033 s** (`check:diag`).

| # | Gate | Threshold |
|---|---|---|
| G1 | machinery must be ~free when unused: `check:diag` with the `slice-literal.ts` changes only (no new tests) | ≤ 1.02× baseline |
| G2 | per-site cost on the generated parity grid (≥ 150 product sites, ranks 1–4, dims 1–8 digits plus 16-digit cap probes), measured in an ISOLATED tsc run | mean ≤ 2,000 instantiations/site AND grid-run total < 1,000,000 AND grid check time < 2 s |
| G3 | `pnpm bench:editor` re-run (guards the existing program's editor experience) | hard gate PASS |
| G4 | `pnpm check` clean · `test:core` 817 · `test:resident` 2319+2 skips unchanged · main `check:diag` wall time within ±10% | all |

G2 rationale: the Kern-05 stretch (subtract+compare per slice site) cost order-of-hundreds
instantiations per site at 1.59× total; O(digits²) multiplication may cost one order more but
must stay in the low thousands for realistic sites, or hover/typecheck cost would grow past what
the Spike-02 headroom (~3 orders of magnitude) comfortably absorbs. The committed curated type
tests' own cost is recorded SEPARATELY (Spike-03 honesty split: feature cost vs. test cost), with
no gate — but is reported, not hidden.

**Decision rule:** all four gates pass → **GO** (machinery stays; Phase B wires
`flatten()`/`reshape()` onto it). Any gate fails → **NO-GO**: revert the appended block + tests
(single-block revert, no other file touched) and record the negative decision with the measured
numbers in the results doc.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Type tests (new `spike/tests/product.test-d.ts`, house idioms): identities (`[]`→1, `[1,1,1]`→1, `[3]`→3), zeros (`[0,5]`→0, `[2⁵²,2⁵²,0]`→0), carry/borrow-heavy multi-digit cases (`[999,999]`→998001, `[1024,768]`, `[65536,65536]`, 7-digit `[1000003,1000003]`), cap boundary BOTH sides (`[6361,69431,20394401]`→9007199254740991 literal; `[2,4503599627370496]`→`number`), every degrade row of the semantics table (dynamic rank/dim, union, `never`, negative, `1.5`, `1e21`), union-of-shapes distribution (`[2,3]\|[4]`→`6\|4`). |
| 2 | Parity grid (scratch, orchestrator-run like Spike 03's): ≥ 150 generated assertions with **BigInt** ground truth, deterministic generator (no timestamps), covering ranks 1–4 and both sides of the 2⁵³−1 cap; tsc exit 0. |
| 3 | Budget gates G1–G4 measured and recorded; GO/NO-GO issued per the decision rule. |
| 4 | Fresh-context verification before "done"; results doc `docs/spike-04-ergebnisse.md` (honesty rule: what degrades and why, test cost split out, editor-latency caveat below); KB upsert; FOLLOWUPS updated. |

## Out of scope

Runtime `reshape`/`flatten` and their method-level guards/messages (Phase B — the guard message
must mirror a runtime throw that doesn't exist yet); `-1` dim inference à la NumPy reshape (needs
division); products in editor-hover workloads (`bench:editor` measures the existing program; a
future `reshape()` call site's hover cost is measurable only once the method exists — recorded as
a Phase-B measurement obligation, not silently skipped); renaming `slice-literal.ts` (Phase D).
