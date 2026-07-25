/**
 * Shared bit-identity assertions for the differential suite. Per spec, v1
 * kernels use only `+`/`*` (IEEE-defined in WASM) — so a match must be
 * exact, not epsilon: shape equality (deep) plus per-element `Object.is`
 * (distinguishes -0/+0 and handles NaN correctly, unlike `===`).
 *
 * NaN-payload clarification (Kern 10, docs/kern-10-special-values-spec.md
 * D3): `assertDataBitIdentical` stays `Object.is`-based, which treats every
 * NaN as equal regardless of its payload bits — the WASM spec permits
 * implementation-defined NaN payloads for arithmetic results, so the claim
 * this comparator backs is: bit-identical for ALL non-NaN values (+/-0
 * distinguished via `Object.is`), NaN equal only as a value CLASS. Callers
 * that need the strictly stronger byte-exact payload claim (e.g. proving a
 * pure data-movement op like transpose never canonicalizes a NaN's bits) use
 * `bitsOf` directly instead of this function.
 */
import assert from "node:assert";
import type { SliceSpecInput } from "../src/slice.ts";

export function assertShapeEqual(expected: readonly number[], actual: readonly number[], context: string): void {
  assert.deepStrictEqual([...actual], [...expected], `${context}: shape mismatch, expected [${expected.join(",")}] got [${actual.join(",")}]`);
}

export function bitsOf(x: number): bigint {
  return new BigUint64Array(new Float64Array([x]).buffer)[0] ?? 0n;
}

export function assertDataBitIdentical(expected: Float64Array, actual: Float64Array, context: string): void {
  assert.strictEqual(actual.length, expected.length, `${context}: data length mismatch, expected ${expected.length} got ${actual.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] ?? 0;
    const a = actual[i] ?? 0;
    if (!Object.is(e, a)) {
      assert.ok(
        false,
        `${context}: bit mismatch at flat index ${i}: reference=${e} (0x${bitsOf(e).toString(16)}) ` +
          `wasm=${a} (0x${bitsOf(a).toString(16)})`,
      );
    }
  }
}

/** Widen a literal slice spec list so `SliceSpecsGuard`/`SliceShape` take the
 * no-claim path (`IsDynamicLength`). Runtime tests check RUNTIME behaviour;
 * the type-level claims live in `spike/tests/slice.test-d.ts`, which pins
 * `NDArray.slice` and states that this covers `WNDArray.slice` too. Paying
 * the literal machinery at a runtime call site therefore buys nothing and is
 * measurably expensive — see docs/slice-literal-budget-spec.md. Only applies
 * to receivers of statically known rank; on a dynamic shape `RankUnknowable`
 * already short-circuits and this wrapper is pointless. */
export function wideSpecs(...specs: readonly SliceSpecInput[]): readonly SliceSpecInput[] {
  return specs;
}
