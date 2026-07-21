/**
 * Kern 07: `DotCheck<S, B, Op>` — the compile-time guard shared by `dot` and
 * `cosineSimilarity` (both rank-1 x rank-1, equal length; see
 * docs/kern-07-elementwise-vector-spec.md's error-message table). `Op` is
 * the message stem (`"dot"` or `"cosineSimilarity"`) so both call sites
 * reuse one guard instead of two near-duplicates.
 *
 * Structure mirrors two existing house patterns rather than inventing a
 * third:
 *  - the `RankUnknowable` guard on EITHER operand (dynamic rank OR a
 *    mixed-rank shape union, D-V1.3, docs/phase-d-vorarbeiten-spec.md),
 *    checked first and unconditionally degrading to "no claim" (pass) —
 *    identical to `Broadcast`/`MatMul`'s own `RankUnknowable` gate
 *    (broadcast.ts, matmul.ts);
 *  - the rank-1 destructuring check `S extends readonly [infer D extends Dim]`
 *    — MatMul's own "rank(A) === 1" idiom (matmul.ts) — used here as a NAKED
 *    check on the bare type parameter so a union OF SHAPES (e.g.
 *    `[2,3] | [4]`) distributes and is processed member-by-member, the same
 *    natural distribution `MatMulStatic`/`ProductAcc` already rely on
 *    elsewhere in this codebase (never special-cased).
 *
 * Deliberately does NOT reuse `DimEq` (dim.ts) for the length comparison:
 * `DimEq`'s `A extends B` check distributes over a union DIM VALUE (e.g. a
 * single axis typed `2 | 3`) with no filter — the pre-existing `MatMul`
 * latent hazard this design pass discovered and deliberately leaves
 * unfixed (FOLLOWUPS). `VectorLenCheck` below filters union dims FIRST via
 * the newly-exported `IsUnion` (literal-arithmetic.ts, Spike-04/06 house rule) —
 * a union dim on either side is *never* misread as a verdict, it degrades
 * to "no claim" instead, same as a dynamic (`number`) dim.
 */
import { type Dim, type IsDynamicDim, type RankUnknowable, type Shape, type ShapeError, type ShowShape } from "./dim.ts";
import { type IsUnion } from "./literal-arithmetic.ts";

/** Non-error sentinel `DotCheck`/`VectorLenCheck` resolve to on every "pass"
 * branch (dynamic rank, dynamic dim, union dim, or equal literal lengths).
 * `Guard<Result, Actual>` (ndarray.ts) only branches on `Result extends
 * ShapeError<string>`, so any non-`ShapeError` type works here — `true` is
 * chosen for readability at each branch. */
type Pass = true;

/**
 * Per-axis length check for the (already rank-1-confirmed) single dim of
 * each operand. Checked in this order: union dim (either side) -> no claim;
 * dynamic dim (either side) -> no claim; equal literals -> pass; unequal
 * literals -> the length-mismatch message (spec table, row 3).
 */
type VectorLenCheck<SD extends Dim, BD extends Dim, Op extends string> = IsUnion<SD> extends true
  ? Pass
  : IsUnion<BD> extends true
    ? Pass
    : IsDynamicDim<SD> extends true
      ? Pass
      : IsDynamicDim<BD> extends true
        ? Pass
        : SD extends BD
          ? BD extends SD
            ? Pass
            : ShapeError<`${Op}: vector lengths ${SD} and ${BD} do not match`>
          : ShapeError<`${Op}: vector lengths ${SD} and ${BD} do not match`>;

/**
 * Static (both operands already known non-dynamic-rank) half of `DotCheck`.
 * Message-table order: first-operand rank, then second-operand rank, then
 * length. `S`/`B` are checked NAKED (not via `S["length"]`) so a union of
 * whole SHAPES distributes across this conditional's branches, matching
 * `MatMulStatic`'s own precedent — only a union DIM WITHIN one shape needs
 * the explicit `IsUnion` filter (`VectorLenCheck` above), never a union of
 * whole shapes.
 */
type DotCheckStatic<S extends Shape, B extends Shape, Op extends string> = S extends readonly [infer SD extends Dim]
  ? B extends readonly [infer BD extends Dim]
    ? VectorLenCheck<SD, BD, Op>
    : ShapeError<`${Op}: expected a 1-D vector as the second operand (got shape ${ShowShape<B>})`>
  : ShapeError<`${Op}: expected a 1-D vector as the first operand (got shape ${ShowShape<S>})`>;

/**
 * The `dot`/`cosineSimilarity` operand guard: both operands must be rank-1
 * with equal length. Dynamic RANK on either side, OR a MIXED-rank shape
 * union (D-V1.3, `RankUnknowable`, docs/phase-d-vorarbeiten-spec.md),
 * degrades unconditionally to "no claim" (pass; the runtime backstop is
 * `assertVectorPair` in runtime.ts). Used as `Guard<DotCheck<S, B, "dot">, NDArray<B>>` (resp.
 * `WNDArray<B>`, `"cosineSimilarity"`) — a receiver-side (`S`) violation
 * still surfaces AT THE ARGUMENT, same as every other op here, because
 * that is where `Guard` itself puts the compile error.
 */
export type DotCheck<S extends Shape, B extends Shape, Op extends string> = RankUnknowable<S> extends true
  ? Pass
  : RankUnknowable<B> extends true
    ? Pass
    : DotCheckStatic<S, B, Op>;

// ---------------------------------------------------------------------------
// Op-Scheibe W1 (docs/op-w1-argmax-topk-spec.md, D3): `TopkCheck<S, K>` +
// `TopkShape<S, K>` — the compile-time guard + resulting shape for
// `NDArray.topk`. Appended strictly after all pre-existing content in this
// file (freeze discipline, D5 — this is why the extra import below is its
// OWN statement rather than widening the `literal-arithmetic.ts` import
// line above: T5 requires this file's diff to show ONLY additions after the
// last pre-existing line). Rank-1-only (`DotCheck`'s own precedent above: a
// receiver-side rank problem surfaces AT THE `k` ARGUMENT, since `topk` has
// no other place to hang a `Guard` on). Reuses the EXISTING digit-string
// machinery from literal-arithmetic.ts (`Compare`, `NonNegDigits`,
// `LiteralReshapeDimInvalid`, `IsUnion`, already imported above) — no new
// arithmetic primitives, per D3's binding decision. `LiteralIndexBounds` is
// DELIBERATELY not used here: it has the wrong semantics for a k-bounds
// check in BOTH directions (`k == D` classifies as `"out"` under
// `LiteralIndexBounds`'s `i < d` index semantics; a negative `k` classifies
// as `"in"` under its NumPy negative-index normalization) — empirically
// proven during Baustein 0, see the spec's adversarial-verification
// addendum.
// ---------------------------------------------------------------------------
import type { Compare, LiteralReshapeDimInvalid, NonNegDigits } from "./literal-arithmetic.ts";

/** `topk`'s "wrong receiver rank" message — `DotCheck` precedent: the error
 * surfaces at the `k` argument (via `Guard`) even though the actual problem
 * is with the RECEIVER `S`, because a niladic-style rank check has nowhere
 * else to attach. Mirrors the runtime throw (`topkRuntime`, runtime.ts)
 * verbatim — pinned by string-equality unit tests, not just the
 * differential suite. */
type TopkRankMessage<S extends Shape> = `topk: expected a 1-D vector (got shape ${ShowShape<S>})`;

/** `topk`'s "k is not a valid non-negative integer" message — covers BOTH a
 * negative literal `k` and a dot-form (non-integer) literal `k` in ONE
 * check (`LiteralReshapeDimInvalid`, reused unmodified against the
 * singleton shape `[K]` — the Kern-08 reshape-dim classifier already proves
 * exactly this "provably invalid dim" verdict, and a `topk` count is
 * structurally the same kind of value as a reshape dim: a non-negative
 * integer, or wide/no-claim). Mirrors the runtime throw verbatim. */
type TopkInvalidKMessage<K extends number> = `topk: k must be a non-negative integer (got ${K})`;

/** `topk`'s "k exceeds the vector length" message. Mirrors the runtime
 * throw verbatim. `LiteralIndexBounds` is NOT used here (see file-section
 * header above) — this reuses `Compare` (+ `NonNegDigits` to obtain a
 * comparable digit string for `K`) directly instead, the Owner-decided
 * minimal-export path (D3 v2). */
type TopkBoundsMessage<K extends number, D extends Dim> = `topk: k=${K} exceeds the vector length ${D}`;

/**
 * `k`'s own validity, once the receiver is confirmed rank-1 with dim `D`:
 * dot-form/negative literal `k` -> error (structurally safe, doesn't depend
 * on `D` at all — checked FIRST, unconditionally); otherwise a bounds check
 * against `D`, itself gated by `IsUnion<D>`/`IsDynamicDim<D>` (same
 * defensive filter `VectorLenCheck` above already uses for a single dim
 * that might carry a union type) and by `NonNegDigits<K>` (wide/
 * exponent-form `k` -> no bounds claim, the same digit-machinery boundary
 * `literal-arithmetic.ts` already draws — D3's "MAX_SAFE_INTEGER-Kappe ...
 * jenseits -> no-claim"). `k = 0` and `k = D` are both VALID (`Compare`
 * yields `"lt"`/`"eq"`; only `"gt"` errors).
 */
type TopkCheckStatic<D extends Dim, K extends number> = LiteralReshapeDimInvalid<[K]> extends "ok"
  ? IsUnion<D> extends true
    ? Pass
    : IsDynamicDim<D> extends true
      ? Pass
      : NonNegDigits<K> extends infer KDigits extends string
        ? KDigits extends "unsupported"
          ? Pass
          : Compare<KDigits, `${D}`> extends "gt"
            ? ShapeError<TopkBoundsMessage<K, D>>
            : Pass
        : never
  : ShapeError<TopkInvalidKMessage<K>>;

/**
 * The `topk` operand guard: `S` must be rank-1 (mirrors `DotCheck`'s own
 * receiver-rank check above); `k` must classify as a valid non-negative
 * integer no larger than the vector's length. `IsUnion<K>` is filtered
 * FIRST, UNCONDITIONALLY — before even the rank-1 check — mirroring
 * reduce.ts's own union-AXIS filter position exactly (D3: "the filter runs
 * before any ShapeError is even produced"; a union `k` is a no-claim
 * degrade regardless of whether the receiver's rank happens to also be
 * wrong). `RankUnknowable<S>` degrades the SAME way `DotCheck` does above
 * (no claim; the runtime backstop, `topkRuntime`, stays authoritative).
 */
export type TopkCheck<S extends Shape, K extends number> = IsUnion<K> extends true
  ? Pass
  : RankUnknowable<S> extends true
    ? Pass
    : S extends readonly [infer D extends Dim]
      ? TopkCheckStatic<D, K>
      : ShapeError<TopkRankMessage<S>>;

/**
 * `topk`'s resulting shape (both `values` and `indices` share it). Always
 * rank-1 — unlike `ReduceAxis`, `topk`'s output rank never depends on the
 * receiver `S` at all (a successful call is ALWAYS `[k]`, regardless of
 * whether `S`'s rank was statically confirmed or only runtime-checked), so
 * `S` itself is UNUSED in the body below — kept as a parameter purely for
 * signature symmetry with every other shape-computing type in this codebase
 * (`SliceShape`, `ReduceAxis`, ...) and because it is what the `topk` call
 * site naturally has in scope. This mirrors `SliceShape`/`SliceSpecsGuard`'s
 * own precedent (slice.ts): the shape-computing type does not need to
 * re-validate what the paired `*Check` type already gates via `Guard` — an
 * invalid call never type-checks in the first place, so this type only
 * needs to be honest for calls the guard already accepted. Degrades to
 * `readonly [number]` (rank-1, dim unknown) for a union or wide/
 * unsupported-digit-form `K` — never `readonly Dim[]` (unbounded rank),
 * since the RANK is always known even when the exact length isn't.
 */
export type TopkShape<S extends Shape, K extends number> = IsUnion<K> extends true
  ? readonly [number]
  : NonNegDigits<K> extends "unsupported"
    ? readonly [number]
    : [K];

// ---------------------------------------------------------------------------
// Op-Scheibe W4 (docs/op-w4-stack-spec.md, D2; Baustein-0-Addendum F1-F8 in
// that spec's "Adversariale Spec-Verifikation" section): `StackCheck<Shapes>`
// + `StackShape<Shapes>`, the compile-time guard + resulting shape for
// `NDArray.stack(rows)`. Appended strictly after all pre-existing content in
// this file (freeze discipline, D5 — own import statement below, same
// per-append convention the W1 block above already established).
//
// LAYERING (F1, avoids a BLOCKER-class import cycle): both types below
// operate on `Shapes extends readonly Shape[]` — the rows' shapes, ALREADY
// unwrapped from `NDArray<S>` — never on `readonly NDArray<any>[]` directly.
// vector.ts never imports NDArray (dim.ts's own file-header precedent:
// broadcast.ts/matmul.ts/reduce.ts/slice.ts/vector.ts all import FROM
// dim.ts, never the other way, to keep the import graph acyclic). The
// NDArray -> Shape unwrap is its own named type, `RowShapesOf<Rows>`
// (ndarray.ts, a homomorphic mapped type — F2 below explains why it must be
// homomorphic) — the static `NDArray.stack` method applies it BEFORE either
// type here ever runs.
// ---------------------------------------------------------------------------
import { type DimEq } from "./dim.ts";

/** `stack`'s "no rows at all" message — the FIRST validation `stackRuntime`
 * (runtime.ts) performs, so it also takes precedence at the type level over
 * every per-row check below. Only fires at compile time for an EMPTY TUPLE
 * LITERAL (`Shapes["length"] extends 0`, F3 below) — an empty runtime ARRAY
 * (`readonly NDArray<[3]>[]`, length genuinely unknown) can't be told apart
 * from a non-empty one statically, so it stays a pure runtime backstop (D2:
 * "leeres ARRAY zur Laufzeit = Runtime-Backstop"). Mirrors the runtime throw
 * verbatim. */
type StackEmptyMessage = "stack: expected at least one row";

/** `stack`'s "this row isn't rank-1" message, TUPLE-position form — WITH the
 * offending row's index (`stackRuntime` always knows it: it iterates
 * concrete rows one at a time). Mirrors the runtime throw verbatim. */
type StackRankMessage<S extends Shape, Idx extends number> = `stack: expected 1-D rows (got shape ${ShowShape<S>} at index ${Idx})`;

/** `stack`'s "this row isn't rank-1" message, ARRAY-input form (F7): no "at
 * index" suffix — every possible call of a statically `readonly
 * NDArray<[2,3]>[]`-typed argument is a guaranteed throw (an empty array
 * throws `StackEmptyMessage` above; a non-empty one throws this), so the
 * rejection is sound but isn't about any ONE particular row. Same stem
 * prefix as `StackRankMessage` on purpose — only the missing index differs. */
type StackRankMessageArray<S extends Shape> = `stack: expected 1-D rows (got shape ${ShowShape<S>})`;

/** `stack`'s "two rows disagree on length" message. Mirrors the runtime
 * throw verbatim. */
type StackLengthMismatchMessage<Expected extends Dim, Got extends Dim, Idx extends number> =
  `stack: row length mismatch (expected ${Expected}, got ${Got} at index ${Idx})`;

/**
 * Per-row dim extractor with its OWN naked type parameter (used by the
 * ARRAY path, F5/F8): a small helper generic so a UNION row-shape type (from
 * a union-element-typed array input, F8 — e.g. `Shapes[number]` resolving to
 * `readonly [3] | readonly [4]`) distributes member-by-member when this type
 * is invoked with that union. `Shapes[number]` itself is an INDEXED-ACCESS
 * type, never a naked type parameter, so without this extra generic no
 * distribution would happen at the call site at all (`(A|B) extends readonly
 * [infer D] ? D : never` would run as ONE non-distributive check against the
 * whole union instead, inferring the union of every matched position in one
 * shot) — introducing a fresh generic purely to re-enable distribution is
 * the same idiom this file's own `DotCheckStatic`/`TopkCheckStatic` already
 * rely on via their bare `S`/`D`/`K` parameters (see `DotCheckStatic`'s doc
 * comment above: "a union OF SHAPES ... distributes ... the same natural
 * distribution MatMulStatic/ProductAcc already rely on").
 */
type ArrayRowD<RowShape extends Shape> = RowShape extends readonly [infer D extends Dim] ? D : never;

/**
 * Merge one row's dim `D` into the tuple-path fold accumulator `Acc` (D2's
 * per-row length check). `Acc` is a plain `Dim` here, never the `"none"`
 * not-yet-seen sentinel — the caller (`StackFold` below) has already
 * filtered that via `[Acc] extends ["none"]` + `Extract<Acc, Dim>` before
 * ever calling this (F4: a conditional branch does NOT narrow a type
 * PARAMETER the way value-level control flow narrows a variable, so the
 * filter has to be an explicit tuple-wrapped check at the call site, not an
 * implicit assumption inside this type).
 *
 * Union/dynamic on EITHER side widens the merged dim to the wide `Dim`
 * (`number`) UNCONDITIONALLY (F6, "CompatDim-Präzedenz", dim.ts) — the same
 * wide-wins-immediately shape `CompatDim`/`DimEq` themselves already use,
 * minus `CompatDim`'s "either side is 1" broadcast special-casing, which
 * `stack` deliberately never wants: rows of length 1 and 3 must still be
 * REJECTED, never silently broadcast into each other. Once any row has
 * widened the fold, `StackFold`'s recursion keeps the fold wide for every
 * LATER row too — `IsDynamicDim<Acc>`/`IsUnion<Acc>` keep firing on the
 * now-wide accumulator on every subsequent call, so this is a MONOTONE
 * degradation, never a silent narrowing back to some later row's literal.
 *
 * Only once both sides are confirmed non-union/non-dynamic does `DimEq` do
 * the actual comparison (Baustein-0 finding 9, "DimEq reicht nach Filtern")
 * — `DimEq` alone, unfiltered, would have the same union-distribution hazard
 * this file's own header comment already documents as MatMul's un-fixed
 * latent case; filtering first avoids it here from the start.
 */
type StackDimMerge<Acc extends Dim, D extends Dim, Idx extends number> = IsUnion<Acc> extends true
  ? Dim
  : IsUnion<D> extends true
    ? Dim
    : IsDynamicDim<Acc> extends true
      ? Dim
      : IsDynamicDim<D> extends true
        ? Dim
        : DimEq<Acc, D> extends true
          ? D
          : ShapeError<StackLengthMismatchMessage<Acc, D, Idx>>;

/**
 * Tail-recursive fold over the TUPLE-input path's rows (`Reverse`'s own
 * Head/Rest idiom, dim.ts): validates every row is rank-1 and every row's
 * dim agrees with every other (via `StackDimMerge` above), accumulating the
 * common dim in `Acc` — `"none"` is the not-yet-seen sentinel (F4); any
 * OTHER `Acc` value is a real `Dim` already confirmed compatible with every
 * row seen so far. `Seen` is a pure INDEX-tracking accumulator
 * (`Seen["length"]` = the CURRENT row's index) — small-int tuple-length
 * arithmetic, not a dim value, the same "safe" arithmetic CLAUDE.md's TS-
 * limits section scopes tuple-length math to (ranks/small ints, never dim
 * VALUES).
 *
 * A `RankUnknowable` row (dynamic rank, or itself a mixed-rank shape union —
 * D-V1.3 precedent, dim.ts) can't be proven wrong, so it degrades the WHOLE
 * fold to wide (`Dim`) rather than rejecting OR silently keeping whatever
 * literal the other rows agreed on — the same "uncertainty propagates, never
 * gets silently dropped" policy `Broadcast`/`MatMul` already apply to their
 * own `RankUnknowable` operands.
 *
 * A row whose shape is itself a union of SAME-rank shapes (e.g. `[3] | [4]`
 * — not caught by `RankUnknowable`, which only fires for mixed-RANK unions)
 * is ALSO widened here, via the `IsUnion<Head>` branch — REVISED after a
 * Verify-B finding (BLOCKER-class M2 violation, empirically reproduced):
 * an earlier revision of this type left such a `Head` to "distribute
 * naturally" through the naked `Head extends readonly [infer D extends
 * Dim]` check below, reasoning by analogy to `DotCheckStatic`'s own
 * union-of-whole-shapes precedent in this file. That analogy does NOT hold
 * here: `DotCheckStatic` runs the distributed check exactly ONCE per call
 * (a single rank confirmation), so each distributed branch resolves to an
 * independent, self-contained verdict. `StackFold` instead THREADS one
 * branch's distributed value (`D`) into the NEXT recursive call's `Acc`
 * parameter — so distribution at one row silently forks the ENTIRE REST OF
 * THE FOLD into parallel per-member continuations that can each reach a
 * DIFFERENT verdict (e.g. one member's `D` agrees with a later row, the
 * other's doesn't), producing a MIXED union of a real `Dim` and a
 * `ShapeError` as `StackFold`'s overall result. `Guard`'s tuple-wrapped
 * `[Result] extends [ShapeError<infer M>]` check (ndarray.ts) rejects only
 * a UNIFORM error union — a mixed one falls through to `Actual` (accepted),
 * and `StackShape`'s `Extract<StackFold<Shapes>, Dim>` then silently drops
 * the `ShapeError` member and keeps the surviving literal `Dim` — exactly
 * the CONFIDENTLY WRONG result M2 forbids: `NDArray.stack([fixed, union])`
 * compiled clean with a specific literal `D` that the runtime call could
 * (and empirically did) reject. This is the SAME failure shape `reduce.ts`'s
 * own `ReduceAxis` documents for its `IsUnion<Axis>` filter (see that type's
 * doc comment) — a union that survives past a naked distributive check
 * partially, with the losing branch's `ShapeError` silently discarded
 * downstream. The fix is the same one `ReduceAxis` already applies: gate on
 * `IsUnion<Head>` BEFORE the naked `Head extends readonly [infer D]` check.
 * POSITION IS LOAD-BEARING, same as `ReduceAxis`'s own filter: the moment
 * execution reaches the naked check, a union `Head` has ALREADY been
 * distributed member-by-member — a filter placed after that point would see
 * only single members, never the union as a whole (this is exactly the
 * mechanism the Verify-B finding proved). Result: a union-shaped row NEVER
 * produces a `ShapeError`, not even when EVERY member would individually
 * mismatch (the "double-mismatch" case) — the fold degrades uniformly to
 * `readonly [N, number]` no-claim instead, and the runtime backstop
 * (`stackRuntime`) stays authoritative for what the type layer can no
 * longer prove. This also fixes the STEM-union side effect a partial
 * distribution would otherwise cause on a genuine double mismatch (two
 * DIFFERENT `ShapeError` messages combining into one confusing union
 * message) — there is now only ever Pass or `Extract<..., Dim>`, never a
 * `ShapeError` union, out of this fold when a union `Head` is involved.
 *
 * Terminates (`Rows` exhausted) by returning `Acc`. Always a real `Dim` in
 * practice: the caller (`StackCheck`/`StackShape` below) gates the
 * empty-tuple case (F3) BEFORE ever invoking this fold, so `Rows` is never
 * `[]` on the very first call — `"none"` is never actually observable as a
 * final result, but the type stays honest about the parameter's full range
 * rather than asserting an unproven invariant away.
 */
type StackFold<
  Rows extends readonly Shape[],
  Seen extends readonly unknown[] = [],
  Acc extends Dim | "none" = "none",
> = Rows extends readonly [infer Head extends Shape, ...infer Rest extends readonly Shape[]]
  ? RankUnknowable<Head> extends true
    ? StackFold<Rest, [...Seen, unknown], Dim>
    : IsUnion<Head> extends true
      ? // MUST sit here — directly after the `RankUnknowable` gate and
        // BEFORE the naked `Head extends readonly [infer D]` branch below.
        // See the doc comment above (Verify-B finding, `ReduceAxis`
        // precedent): position is load-bearing, a filter placed after the
        // naked check would already be too late (the union is distributed
        // by then). Widens exactly like the `RankUnknowable` branch above —
        // a union of whole row shapes can't be proven wrong either.
        StackFold<Rest, [...Seen, unknown], Dim>
      : Head extends readonly [infer D extends Dim]
        ? [Acc] extends ["none"]
          ? StackFold<Rest, [...Seen, unknown], D>
          : StackDimMerge<Extract<Acc, Dim>, D, Seen["length"]> extends infer Merged
          ? Merged extends ShapeError<string>
            ? Merged
            : StackFold<Rest, [...Seen, unknown], Extract<Merged, Dim>>
          : never
      : ShapeError<StackRankMessage<Head, Seen["length"]>>
  : Acc;

/**
 * ARRAY-input path (F5): `Shapes` here is a genuine `readonly Shape[]`
 * (dynamic length — `number extends Shapes["length"]` already confirmed by
 * the caller), so the tuple-recursive `StackFold` above NEVER matches it
 * (empirically: no Head/Rest match against a plain array type — F5). Rank
 * validity is checked NON-distributively against `Shapes[number]` as a
 * WHOLE (an indexed-access type, never naked, so `extends` here does not
 * distribute per-member — exactly what F7's "uniform" claim needs: a union
 * of same-rank row shapes like `[3] | [4]` is still accepted here, since
 * EVERY member individually satisfies the 1-tuple pattern; only a uniformly
 * WRONG rank, where NO member does, is rejected).
 *
 * F7: a uniform, provably-wrong literal rank (e.g. every possible row typed
 * `[2, 3]`) is REJECTED outright — sound, because even the empty-array case
 * throws (`StackEmptyMessage`), so every conceivable runtime call under this
 * static type is a guaranteed throw. An `Unknowable` row rank (dynamic, or a
 * mixed-rank union) degrades to no-claim (pass) instead — can't be proven
 * wrong, so it isn't rejected.
 */
type StackCheckArray<Shapes extends readonly Shape[]> = RankUnknowable<Shapes[number]> extends true
  ? Pass
  : Shapes[number] extends readonly [Dim]
    ? Pass
    : ShapeError<StackRankMessageArray<Shapes[number]>>;

/**
 * ARRAY-input path's resulting `[N, D]` (F5/F6/F8): `N` is always the wide
 * `number` (an array's length is never statically known, so honesty
 * requires it — unlike the tuple path's literal `Shapes["length"]`). `D`
 * defaults to the honest literal shared by every row (`readonly
 * NDArray<[3]>[]` -> `readonly [number, 3]`, the F5 evidence form) unless
 * EITHER a single row's own dim is dynamic (`ArrayRowD` itself resolves to
 * the wide `Dim` for a `[number]`-shaped row, so this falls out for free) OR
 * the array's element type is itself a UNION of shapes with different dims
 * (F8, e.g. `readonly (NDArray<[3]>|NDArray<[4]>)[]`) — `ArrayRowD`'s own
 * naked type parameter distributes that union into `3 | 4`, and the
 * `IsUnion` filter here degrades that combined result to wide `number`
 * rather than exposing a misleadingly-precise `3 | 4` union type (D2's
 * general "Union-Element-Typen -> no-claim" house policy, pinned as F8).
 */
type StackShapeArray<Shapes extends readonly Shape[]> = RankUnknowable<Shapes[number]> extends true
  ? readonly [number, number]
  : IsUnion<ArrayRowD<Shapes[number]>> extends true
    ? readonly [number, number]
    : readonly [number, ArrayRowD<Shapes[number]>];

/**
 * The `NDArray.stack(rows)` operand guard (D2, D4): `Shapes` is
 * `RowShapesOf<Rows>` (ndarray.ts) — the caller's rows, already unwrapped to
 * plain `Shape`s. Two structurally different inputs are routed by
 * `number extends Shapes["length"]` (F5, the standard `IsDynamicRank`-style
 * "is this a real tuple or a length-erased array" probe, dim.ts):
 *  - a TUPLE (fixed arity, e.g. two `NDArray<[3]>` arguments) -> the
 *    empty-literal gate (F3) THEN the `StackFold` fold above;
 *  - an ARRAY (`readonly NDArray<[3]>[]`, unknown length) -> `StackCheckArray`
 *    above (F5/F7).
 *
 * The empty-TUPLE-literal gate runs strictly BEFORE any element extraction
 * (F3: `never extends NDArray<infer S>`-style helpers would otherwise
 * silently fall back to their own constraint instead of erroring on `[]`) —
 * `Shapes["length"] extends 0` is exact for a literal empty tuple, since
 * `Rows["length"]` (ndarray.ts) is exactly `0` for a `[]` call.
 */
export type StackCheck<Shapes extends readonly Shape[]> = number extends Shapes["length"]
  ? StackCheckArray<Shapes>
  : Shapes["length"] extends 0
    ? ShapeError<StackEmptyMessage>
    : StackFold<Shapes> extends infer Result
      ? Result extends ShapeError<string>
        ? Result
        : Pass
      : never;

/**
 * `stack`'s resulting shape: `[N, D]`, `N` = row count, `D` = the common row
 * length. Does NOT re-validate what `StackCheck` (via `Guard`) already gates
 * — the same precedent `TopkShape`'s own doc comment states (ndarray.ts's
 * `stack` never calls this type for a rejected call), so `Extract<StackFold<
 * Shapes>, Dim>` safely narrows away the (for an accepted call, unreachable)
 * `ShapeError` branch of `StackFold`'s return type. `Shapes["length"]` is
 * `N` for the tuple path — plain tuple-length arithmetic (a RANK-shaped
 * small int, the CLAUDE.md-sanctioned use of tuple-length arithmetic; never
 * a dim VALUE). The array path degrades to `StackShapeArray` above instead
 * (F5) — an array's length is never a literal `N`.
 */
export type StackShape<Shapes extends readonly Shape[]> = number extends Shapes["length"]
  ? StackShapeArray<Shapes>
  : readonly [Shapes["length"], Extract<StackFold<Shapes>, Dim>];
