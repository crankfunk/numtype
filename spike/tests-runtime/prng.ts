/**
 * Hand-written seeded PRNG (splitmix64, Vigna's algorithm) + shape/data
 * generators for the differential test suite. Test-only infrastructure —
 * not part of the product, zero new dependencies (pure BigInt arithmetic).
 *
 * splitmix64: given a 64-bit seed, produces a deterministic stream of
 * 64-bit values. Same seed -> same stream, every run — the suite's cases
 * are exactly reproducible (no `Math.random()`).
 */

const MASK64 = (1n << 64n) - 1n;
const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;

/** Raw splitmix64 generator: seed -> a closure yielding the next 64-bit
 * unsigned value on each call. */
export function splitmix64(seed: bigint): () => bigint {
  let state = seed & MASK64;
  return (): bigint => {
    state = (state + GOLDEN_GAMMA) & MASK64;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    z = z ^ (z >> 31n);
    return z & MASK64;
  };
}

export interface Rng {
  /** Raw 64-bit unsigned draw. */
  nextU64(): bigint;
  /** Uniform integer in [minInclusive, maxInclusive]. */
  nextInt(minInclusive: number, maxInclusive: number): number;
  /** A finite, moderate-magnitude signed double with a fractional part
   * (sign, integer part 0..999, 3-decimal fraction) — deliberately never
   * NaN/Infinity, matching v1's transcendental-free (+/-/* only)
   * determinism contract. */
  nextF64(): number;
  nextBool(): boolean;
}

export function makeRng(seed: bigint): Rng {
  const next = splitmix64(seed);
  return {
    nextU64: next,
    nextInt(minInclusive: number, maxInclusive: number): number {
      const span = BigInt(maxInclusive - minInclusive + 1);
      const r = next() % span;
      return minInclusive + Number(r);
    },
    nextF64(): number {
      const bits = next();
      const sign = (bits & 1n) === 1n ? -1 : 1;
      const intPart = Number((bits >> 1n) % 1000n);
      const fracPart = Number((bits >> 11n) % 1000n) / 1000;
      return sign * (intPart + fracPart);
    },
    nextBool(): boolean {
      return (next() & 1n) === 1n;
    },
  };
}

/** Generate a random f64 payload for a shape (row-major, `product(shape)`
 * elements). */
export function genData(rng: Rng, shape: readonly number[]): Float64Array {
  const n = shape.reduce((acc, d) => acc * d, 1);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.nextF64();
  return out;
}

/**
 * Generate a pair of shapes that are ALWAYS NumPy-broadcast-compatible by
 * construction (mirrors `runtimeBroadcastShape`'s rule at every aligned
 * axis: equal, or one is 1) — covers varying/independent ranks (0..maxRank)
 * for each operand and, at every axis, an independent chance of placing a
 * broadcast-1 on either side (or both).
 */
export function genBroadcastShapes(rng: Rng, maxRank = 4): { aShape: number[]; bShape: number[] } {
  const aRank = rng.nextInt(0, maxRank);
  const bRank = rng.nextInt(0, maxRank);
  const rank = Math.max(aRank, bRank);
  const aShape: number[] = new Array(aRank).fill(1);
  const bShape: number[] = new Array(bRank).fill(1);

  for (let i = 0; i < rank; i++) {
    // i counts from the right (axis position rank-1-i in the aligned frame)
    const baseDim = rng.nextInt(1, 8);
    if (i < aRank) {
      aShape[aRank - 1 - i] = rng.nextBool() ? 1 : baseDim;
    }
    if (i < bRank) {
      bShape[bRank - 1 - i] = rng.nextBool() ? 1 : baseDim;
    }
  }
  return { aShape, bShape };
}

// ---------------------------------------------------------------------------
// Kern 10 (docs/kern-10-special-values-spec.md): IEEE-754 special-value
// injection. Appended strictly after all pre-existing content in this file —
// `nextF64`/`genData`/`makeRng`/`genBroadcastShapes` above are byte-for-byte
// unchanged, so no existing seeded differential case changes its values.
// ---------------------------------------------------------------------------

/**
 * One representative of every IEEE-754 special-value CLASS the differential
 * suite must cover: NaN, +/-Infinity, +0/-0 (`Object.is`-distinguished),
 * two distinct subnormals (below `2.2250738585072014e-308` — there is no
 * `Number.MIN_NORMAL` constant, so that literal boundary is the reference),
 * and +/-`MAX_VALUE` (the closest a `+`/`*`-only chain can push a value
 * while both operands stay finite, per D5's transcendental-free scope).
 * `genData`/`nextF64` never produce any of these by construction (see
 * `nextF64`'s own doc comment) — this is the deliberate injection D1 calls
 * for.
 */
export const SPECIAL_VALUES: readonly number[] = [
  Number.NaN,
  Number.POSITIVE_INFINITY,
  Number.NEGATIVE_INFINITY,
  0,
  -0,
  Number.MIN_VALUE, // smallest positive subnormal (~4.9406564584124654e-324)
  -Number.MIN_VALUE,
  Number.MIN_VALUE * 4, // a second, distinct subnormal (still < 2.225...e-308)
  Number.MAX_VALUE,
  -Number.MAX_VALUE,
];

/**
 * One f64 draw: with probability `specialProb`, a uniformly chosen member of
 * `SPECIAL_VALUES`; otherwise an ordinary `rng.nextF64()` draw. Both the
 * "is this draw special" coin flip and the "which special value" choice are
 * `rng`-sourced draws (never `Math.random`), so a given seed's stream stays
 * fully deterministic/reproducible, exactly like every other generator in
 * this file.
 */
export function nextF64Special(rng: Rng, specialProb = 0.35): number {
  const roll = rng.nextInt(0, 999);
  if (roll < Math.round(specialProb * 1000)) {
    const idx = rng.nextInt(0, SPECIAL_VALUES.length - 1);
    return SPECIAL_VALUES[idx] ?? Number.NaN;
  }
  return rng.nextF64();
}

/** Like `genData`, but every element is drawn via `nextF64Special` instead
 * of `nextF64` — a payload shot through with IEEE-754 special values at the
 * given rate, for the differential suite's special-value coverage. */
export function genDataSpecial(rng: Rng, shape: readonly number[], specialProb = 0.35): Float64Array {
  const n = shape.reduce((acc, d) => acc * d, 1);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = nextF64Special(rng, specialProb);
  return out;
}
