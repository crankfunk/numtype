/**
 * Kern 06 follow-up bench — calibrates `THREADED_MATMUL_MIN_POOL_WORK`
 * (the auto-routing threshold in `threaded.ts`). Run via
 * `pnpm bench:crossover`; numbers recorded in docs/kern-06-ergebnisse.md
 * (auto-routing addendum).
 *
 * What it measures: for a grid of shapes spanning the break-even region the
 * Kern-06 bench bracketed (pool loses at n=64, wins 4.3x at n=256), the SAME
 * `threadedMatmul` call forced down each of its two routes —
 * `{minPoolWork: Infinity}` (single-threaded `nt_matmul_blocked` on the main
 * thread, on the pool's own core) vs `{minPoolWork: 0}` (worker-pool
 * dispatch) — so the comparison is exactly the decision the router makes,
 * on the exact code paths it picks between, over the same core/memory.
 *
 * The grid deliberately includes NON-SQUARE shapes at matched work volumes
 * (batch*m*k*n): the FOLLOWUPS item's whole point is that side length or
 * row count alone would misroute (a [16,128]@[128,1024] call has the same
 * volume as 128^3 but only 16 rows -> a single MC block -> at most ONE
 * active worker), so the bench must show how the volume criterion behaves
 * where rows and volume disagree.
 *
 * Discipline (same as every bench here, see also the KB note
 * "js-wasm-benchmarks-jit-zustand-und-crossover"): seeded inputs,
 * bit-identity gate BEFORE any timing (both routes vs the naive runtime.ts
 * reference — never time a wrong result), warmed calls, ranges (min-max
 * over N_REPS) rather than single points. Operands are created ONCE per
 * (shape, pool) and reused across reps, so the timed unit is the matmul
 * call + result dispose itself — NOT fromArray marshalling, which is
 * identical on both routes and would only dilute the difference the router
 * cares about.
 */
import { matmulRuntime } from "../src/runtime.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { initThreadedCore, threadedMatmul, ThreadedPool } from "../src/wasm/threaded.ts";
import { assertDataBitIdentical } from "../tests-runtime/assert-helpers.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const WORKER_COUNTS = [2, 4, 8];
const N_REPS = 7;

interface Case {
  readonly label: string;
  readonly aShape: readonly number[];
  readonly bShape: readonly number[];
}

/** Work volume in multiply-accumulate ops: batch*m*k*n (batch = the
 * broadcast batch product; for the 2-D cases below simply m*k*n). */
function workVolume(aShape: readonly number[], bShape: readonly number[]): number {
  const m = aShape[aShape.length - 2]!;
  const k = aShape[aShape.length - 1]!;
  const n = bShape[bShape.length - 1]!;
  const batchA = aShape.slice(0, -2).reduce((acc, d) => acc * d, 1);
  const batchB = bShape.slice(0, -2).reduce((acc, d) => acc * d, 1);
  return Math.max(batchA, batchB) * m * k * n;
}

function totalRows(aShape: readonly number[], bShape: readonly number[]): number {
  const m = aShape[aShape.length - 2]!;
  const batchA = aShape.slice(0, -2).reduce((acc, d) => acc * d, 1);
  const batchB = bShape.slice(0, -2).reduce((acc, d) => acc * d, 1);
  return Math.max(batchA, batchB) * m;
}

const CASES: Case[] = [
  // Below the region Kern 06 bracketed: a first full-grid run (2026-07-10)
  // showed the pool route already TIES OR WINS at n=64 (0.26 Mops) for
  // every worker count — the Kern-06 Series-B "threads lose at n=64" came
  // from a different measurement unit (end-to-end incl. fromArray/toArray
  // marshalling, baselined against the STABLE core), not from dispatch
  // overhead on the actual routing alternative. So the real crossover sits
  // BELOW n=64; this tiny sweep brackets it.
  { label: "square n=8", aShape: [8, 8], bShape: [8, 8] },
  { label: "square n=16", aShape: [16, 16], bShape: [16, 16] },
  { label: "square n=24", aShape: [24, 24], bShape: [24, 24] },
  { label: "square n=32", aShape: [32, 32], bShape: [32, 32] },
  { label: "square n=48", aShape: [48, 48], bShape: [48, 48] },
  { label: "tall tiny [256,8]@[8,8]", aShape: [256, 8], bShape: [8, 8] },
  { label: "1 row [1,64]@[64,64]", aShape: [1, 64], bShape: [64, 64] },
  // Square sweep across the region Kern 06 bracketed (64^3 .. 256^3).
  { label: "square n=64", aShape: [64, 64], bShape: [64, 64] },
  { label: "square n=96", aShape: [96, 96], bShape: [96, 96] },
  { label: "square n=128", aShape: [128, 128], bShape: [128, 128] },
  { label: "square n=160", aShape: [160, 160], bShape: [160, 160] },
  { label: "square n=192", aShape: [192, 192], bShape: [192, 192] },
  { label: "square n=224", aShape: [224, 224], bShape: [224, 224] },
  { label: "square n=256", aShape: [256, 256], bShape: [256, 256] },
  // Volume-matched non-square at ~2.1 Mops (= 128^3): rows vs volume disagree.
  { label: "wide ~128^3, 16 rows", aShape: [16, 128], bShape: [128, 1024] },
  { label: "deep-k ~128^3, 8 rows", aShape: [8, 512], bShape: [512, 512] },
  { label: "tall ~128^3, 1024 rows", aShape: [1024, 128], bShape: [128, 16] },
  // Volume-matched non-square at ~16.8 Mops (= 256^3).
  { label: "wide ~256^3, 32 rows", aShape: [32, 1024], bShape: [1024, 512] },
  { label: "deep-k ~256^3, 64 rows", aShape: [64, 512], bShape: [512, 512] },
  { label: "tall ~256^3, 4096 rows", aShape: [4096, 64], bShape: [64, 64] },
  // Batched: volume includes the batch product.
  { label: "batch 8x[64,64]@[64,64]", aShape: [8, 64, 64], bShape: [64, 64] },
  { label: "batch 32x[32,32]@[32,32]", aShape: [32, 32, 32], bShape: [32, 32, 32] },
];

interface Range {
  minMs: number;
  medianMs: number;
  maxMs: number;
}

/** Per-call range via warmed, adaptively BATCHED samples (KB note
 * "js-wasm-benchmarks-jit-zustand-und-crossover": batch-timed windows, and
 * JIT state is part of the measurement). Two things per-call timing of the
 * original grid got away with but a tiny-size sweep cannot:
 * - calls down at ~1µs need batching for `performance.now()` to measure a
 *   window, not its own overhead — each sample times `batch` calls and
 *   divides;
 * - each pool has its OWN `WebAssembly.Module` (compiled per
 *   `initThreadedCore`), and V8 tiers wasm modules up independently — a
 *   couple of warmup calls left the first-measured pool partially in
 *   Liftoff (visible as a systematic main-route spread across pools in the
 *   first run). Warm until ~30ms or 64 calls, whichever first, so tier-up
 *   settles before anything is timed. */
function measureRange(fn: () => void, reps = N_REPS): Range {
  const warmupStart = performance.now();
  let warmupCalls = 0;
  let elapsed = 0;
  while (elapsed < 30 && warmupCalls < 64) {
    fn();
    warmupCalls++;
    elapsed = performance.now() - warmupStart;
  }
  const estMs = Math.max(elapsed / warmupCalls, 1e-4);
  const batch = Math.min(512, Math.max(1, Math.ceil(2 / estMs)));

  const samples: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    for (let j = 0; j < batch; j++) fn();
    samples.push((performance.now() - t0) / batch);
  }
  samples.sort((a, b) => a - b);
  return {
    minMs: samples[0]!,
    medianMs: samples[Math.floor(samples.length / 2)]!,
    maxMs: samples[samples.length - 1]!,
  };
}

function fmt(ms: number): string {
  if (ms >= 1) return `${ms.toFixed(2)}ms`;
  return `${(ms * 1000).toFixed(1)}µs`;
}

console.log("=== NumType Kern 06 follow-up bench: threadedMatmul route crossover (main-thread vs pool, per work volume) ===\n");
console.log("Spawning pools for workers in", WORKER_COUNTS, "...");
const pools = new Map<number, ThreadedPool>();
for (const wc of WORKER_COUNTS) {
  pools.set(wc, await initThreadedCore(wc));
}
console.log("Pools ready.\n");

const rng = makeRng(0x43524f53534f5652n); // "CROSSOVR" packed, arbitrary distinct seed

interface ResultRow {
  readonly c: Case;
  readonly volume: number;
  readonly rows: number;
  /** per worker count: [mainRange, poolRange] */
  readonly perWorker: Map<number, readonly [Range, Range]>;
}

const results: ResultRow[] = [];

for (const c of CASES) {
  const aData = genData(rng, c.aShape);
  const bData = genData(rng, c.bShape);
  const ref = matmulRuntime(c.aShape, aData, c.bShape, bData).data;

  const perWorker = new Map<number, readonly [Range, Range]>();
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const a = WNDArray.fromArray(pool.core, c.aShape, aData);
    const b = WNDArray.fromArray(pool.core, c.bShape, bData);
    try {
      // Bit-identity gate for BOTH routes before timing anything.
      const viaMain = threadedMatmul(pool, a, b, { minPoolWork: Infinity });
      assertDataBitIdentical(ref, viaMain.toArray(), `${c.label} workers=${wc} main route`);
      viaMain.dispose();
      const viaPool = threadedMatmul(pool, a, b, { minPoolWork: 0 });
      assertDataBitIdentical(ref, viaPool.toArray(), `${c.label} workers=${wc} pool route`);
      viaPool.dispose();

      const mainRange = measureRange(() => threadedMatmul(pool, a, b, { minPoolWork: Infinity }).dispose());
      const poolRange = measureRange(() => threadedMatmul(pool, a, b, { minPoolWork: 0 }).dispose());
      perWorker.set(wc, [mainRange, poolRange]);
    } finally {
      a.dispose();
      b.dispose();
    }
  }
  results.push({ c, volume: workVolume(c.aShape, c.bShape), rows: totalRows(c.aShape, c.bShape), perWorker });
}

results.sort((x, y) => x.volume - y.volume);

const header =
  `${"case".padEnd(26)} | ${"Mops".padStart(6)} | ${"rows".padStart(5)} | ` +
  WORKER_COUNTS.map((wc) => `w=${wc}: main med | pool med (pool/main)`.padStart(38)).join(" | ");
console.log(header);
console.log("-".repeat(header.length));
for (const r of results) {
  const cells = WORKER_COUNTS.map((wc) => {
    const [mainR, poolR] = r.perWorker.get(wc)!;
    const ratio = poolR.medianMs / mainR.medianMs;
    const mark = ratio <= 1 ? " POOL" : "     ";
    return `${fmt(mainR.medianMs).padStart(9)} | ${fmt(poolR.medianMs).padStart(9)} (${ratio.toFixed(2)})${mark}`.padStart(38);
  });
  console.log(`${r.c.label.padEnd(26)} | ${(r.volume / 1e6).toFixed(2).padStart(6)} | ${String(r.rows).padStart(5)} | ${cells.join(" | ")}`);
}
console.log(
  `\n(pool/main < 1.00 = pool dispatch wins, marked POOL; medians over ${N_REPS} batch-timed samples after adaptive warmup (~30ms or 64 calls) per route;` +
    `\n operands created once per (case, pool) — the timed unit is threadedMatmul + result dispose, both routes measured on the SAME pool core.)`,
);

// Full min-max ranges for the record (medians above are the decision input;
// ranges show the noise floor honestly).
console.log("\n--- full ranges (min-max, median in parens) ---");
for (const r of results) {
  for (const wc of WORKER_COUNTS) {
    const [mainR, poolR] = r.perWorker.get(wc)!;
    console.log(
      `${r.c.label.padEnd(26)} w=${wc}: main ${fmt(mainR.minMs)}-${fmt(mainR.maxMs)} (${fmt(mainR.medianMs)}) | pool ${fmt(poolR.minMs)}-${fmt(poolR.maxMs)} (${fmt(poolR.medianMs)})`,
    );
  }
}

console.log("\nDisposing pools...");
for (const pool of pools.values()) await pool.dispose();
console.log("Done.");
