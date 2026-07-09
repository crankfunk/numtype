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
