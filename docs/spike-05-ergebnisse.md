# NumType — Spike 05: Variance Design, roadmap A3 (Results)

Date: 2026-07-11 · Status: complete, independently verified (one refuted claim corrected, see
addendum) · Toolchain: TypeScript 7.0.2 (native), Node v24, pnpm

Per `docs/spike-05-variance-design-spec.md`. Probe numbers from commands run in this session.

## Headline verdict

**The library's variance design is decided and shipped: `NDArray<S>` stays invariant (keeping
sound degradation + errors-at-the-argument, per the Spike-01 two-of-three rule), and the
"assignable top type" role moves from the unsafe `AnyNDArray` to a new, minimal, checker-enforced
covariant read view `NDArrayView<out S>`** (enforcement has one known, documented syntax
loophole — method-shorthand bivariance, see the addendum — carried as a house rule in the doc
comment). Read-shaped consumers (printing, logging, shape
dispatch, heterogeneous containers) take `NDArrayView<Shape>` and accept every fixed-shape array
implicitly; compute-shaped consumers stay generic over `NDArray<S>` (generic functions never
needed variance); `AnyNDArray` remains only as the documented, deliberately-unsafe both-ways
escape hatch.

## The empirical map (probes against the real class, this session)

Eight probes (scratchpad `variance-probes{,-2}.ts`) pinned the whole design space on TS 7.0.2:

1. An `out S` interface with only read members is accepted, and widening
   (`NDArrayView<[2,3]> → NDArrayView<Shape>`) works — including FROM the real, invariant,
   guard-bearing `NDArray` class, structurally, with no `implements` declared, and including
   heterogeneous `NDArrayView<Shape>[]` containers.
2. A shape-COMPUTING member (`transpose(): View<Transpose<S>>`) breaks the annotation with
   **TS2636**: the annotation check is *abstract* (compares `View<sub-S>` against `View<super-S>`
   with type variables), and `Transpose` — though factually monotone — is not *provably* monotone
   to the checker (its dynamic-rank branch degrades to `readonly number[]`).
3. **The same computed member WITHOUT the annotation widens fine** at concrete instantiations
   (the structural check evaluates the actual types, where monotonicity holds). This is
   covariance-by-accident: real, but unenforced — one future member away from silently flipping
   invariant. Exactly the "only type-checked because the probe happened to pass" failure mode
   Spike 01 documented; rejected on those grounds.
4. Wide-returning (`transpose(): View<readonly number[]>`) and self-returning members survive the
   annotation — but a precision-losing surface on a precise-shapes library was rejected as a
   design contradiction.
5. Generic inference through the view (`materialize<S>(v: View<S>): NDArray<S>`) infers exact
   literal shapes; downcasts are correctly rejected; the existing `NDArray` invariance pins are
   unaffected.

**Sharpened two-of-three finding (new, beyond Spike 01):** it is not guard-bearing (parameter)
members alone that force invariance — *computed-shape return types* also break the *annotated*
(enforced) form of covariance, because annotation checking is abstract while assignability
checking is concrete. An enforced-covariant view is therefore necessarily computation-free: a
pure read surface.

## What was built

- `export interface NDArrayView<out S extends Shape>` in `ndarray.ts` with exactly
  `readonly shape: S` · `strides(): number[]` · `toNestedArray(): unknown`. Both omissions are
  documented as load-bearing in the doc comment: no op members (argument-position `S` is
  unconditionally a variance violation — two-of-three), no computed-shape members (TS2636 under
  the abstract annotation check), no `data` (backend portability for Phase C/D).
- **The `implements NDArrayView<S>` clause landed** (the spec's primary option; the structural
  fallback was not needed) — the class-side drift alarm. Nothing in the three members uses `S`
  in argument position, so the conformance check passes abstractly.
- `AnyNDArray` retained and re-scoped in its doc comment: the deliberately-unsafe BOTH-WAYS
  escape hatch for heterogeneous-and-op-calling consumers; `NDArrayView<Shape>` documented as
  the safe read-only top type. Re-exported alongside it from `index.ts`.
- `spike/demo.ts`'s `printArray` migrated `AnyNDArray` → `NDArrayView<Shape>` (the acceptance
  demonstration: the demo's only heterogeneous consumer is read-shaped).
- Type-test pins appended to `ndarray.test-d.ts`: both widenings (literal view → `Shape` /
  `readonly number[]`), real-class-to-view widening, heterogeneous `NDArrayView<Shape>[]`,
  downcast `@ts-expect-error`, exact generic inference through the view
  (`Expect<Equal<..., [2, 3]>>`); all pre-existing invariance pins untouched. Non-vacuity proven
  in both directions (broken positive pin and un-erroring `@ts-expect-error` each fail
  `pnpm check` at the exact line, then restored).

## Gates (absolute — per the Spike-04 KB lesson)

| gate | result |
|---|---|
| G1 `pnpm check` · `test:core` · `test:resident` · `pnpm demo` | clean · 817 · 2319+2 · complete, "all agree on every showcase op" |
| G2 `bench:editor` | hard gate **PASS** (warm hover 0.05–0.08 ms, toggle 1.48/1.63 ms medians) |
| G3 `check:diag` | **133,727** instantiations (prior pin 133,656, **+71** ≈ +0.05%, deterministic ×5). Attribution corrected by the verifier's stash bisection: **+24 from the interface/`implements` declarations themselves, +47 from the new type-test sites** — a third occurrence, at interface scale, of the declaration-cost pattern (Spikes 03/04). 133,727 is the new pin. |
| G4 wall time | tsc-internal `Check time` is too noisy at sub-50 ms to gate on (53% spread across 5 immediate repeats — methodology note, recorded rather than hidden); the stable proxy, wall-clock `pnpm check` ×5, has ≤3% spread with median 0.434 s — flat. |

## Deviations disclosed

- The spec's acceptance criterion 1 listed three changeable files; `spike/src/index.ts` also
  changed (re-exporting the new public type alongside `AnyNDArray`) — a necessary elaboration
  the spec's file list missed, disclosed rather than retro-edited into the spec.

## Post-verification addendum (2026-07-11)

**Fresh-context verification (brainroute:verify): CONFIRMED with issues — all four gates PASS on
independent reproduction; one prose claim REFUTED and corrected; two honest caveats added to the
shipped doc comments; the DoD items it flagged (KB upsert, FOLLOWUPS, this addendum) are closed
in this same round.**

1. **Refuted and corrected — method-shorthand bivariance.** The shipped doc comment claimed an
   argument-position `S` member is "unconditionally a variance violation" under `out`. The
   verifier's probe disproved the enforcement half: a method-SHORTHAND member
   (`resizeTo(s: S): void`) compiles with NO error — TypeScript checks method-shorthand
   parameters bivariantly (the same long-standing exemption `strictFunctionTypes` grants
   methods); only a property-typed function member (`resizeTo: (s: S) => void`) triggers TS2636.
   Consuming `S` still genuinely breaks covariance — the checker just doesn't catch that syntax.
   Consequence: the `out` annotation is a strong but NOT complete regression pin. The doc
   comment now states the loophole and the house rule (the view never gains an `S`-consuming
   member; future function members are declared property-style, where enforcement is real).
   Corroborated twice: the same shorthand member also slips through `implements`-conformance
   checking on a class.
2. **Every gate independently reproduced:** check clean · 817 · 2319+2 · demo "all agree" ·
   bench:editor hard gate PASS · 133,727 deterministic, with the prior pin 133,656 reproduced
   exactly via stash. Wall-time: sub-second and flat in order-of-magnitude terms; run-to-run
   spread at this sub-second scale is noise-dominated (verifier saw ~11% across 5 runs excluding
   a cold first run) — recorded, not gated away.
3. **Adversarial safety probes:** no unsound direction found — downcasts and widen/narrow
   laundering rejected in every variant (TS2322); no wrong literal-shape claim reachable. Two
   findings, both documented in the shipped comments: (a) `view.shape[0] = 99` type-checks —
   `readonly` on the property does not deep-freeze tuple elements; verified PRE-EXISTING on
   `NDArray` itself and not widened by the view (only real arrays satisfy the view from real
   code today); deep-readonly shape is now a FOLLOWUPS decision (interacts with the clean-hover
   house rule). (b) The view is ordinarily structural (a hand-built `{shape, strides,
   toNestedArray}` object satisfies it), while `AnyNDArray` stays de-facto nominal via the
   private constructor — a real capability difference between the two top types, now documented.
4. **Non-vacuity of the shipped pins re-proven independently** (stripping the downcast
   `@ts-expect-error` in a scratch copy surfaces a genuine TS2322).
