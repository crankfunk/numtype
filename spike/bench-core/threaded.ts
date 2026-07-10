/**
 * Kern 06 bench — run via `pnpm bench:threaded`; numbers recorded in
 * docs/kern-06-ergebnisse.md. Same discipline as the other benches: seeded
 * inputs, bit-identity gate BEFORE any timing (never time a wrong result),
 * warmed calls, MULTIPLE repetitions reported as a range (min–max), not a
 * single point — the honesty rule this phase's spec calls for explicitly
 * ("scaling numbers as ranges... cases where threads LOSE... reported as
 * prominently as wins").
 *
 * Series A (headline): `[n,n] @ [n,n]` for n in {256, 512, 1024}, single-
 * threaded `nt_matmul_blocked` (stable core, raw ABI) vs `threadedMatmul`
 * at worker counts {1, 2, 4, 8} (persistent pools, reused across every n).
 *
 * Series B (honesty at the small end, explicitly required by the spec):
 * n = 64 — dispatch overhead (per-call Atomics round trips across the
 * pool) is real and not amortized by a matmul this small; reported even
 * (especially) if threads lose there, exactly as the spec asks.
 */
import { matmulRuntime } from "../src/runtime.ts";
import { initCore, type CoreExports } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { getThreadedPoolFreeCount, initThreadedCore, threadedMatmul, ThreadedPool } from "../src/wasm/threaded.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES_A = [256, 512, 1024];
const SIZE_B = 64;
const WORKER_COUNTS = [1, 2, 4, 8];
const N_REPS = 7;

interface Range {
  minMs: number;
  medianMs: number;
  maxMs: number;
}

function measureRange(fn: () => void, reps = N_REPS): Range {
  fn();
  fn(); // warmup (2 calls)
  const samples: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  const minMs = samples[0]!;
  const maxMs = samples[samples.length - 1]!;
  const medianMs = samples[Math.floor(samples.length / 2)]!;
  return { minMs, medianMs, maxMs };
}

function fmt(ms: number): string {
  if (ms >= 1) return `${ms.toFixed(2)}ms`;
  return `${(ms * 1000).toFixed(1)}µs`;
}

function fmtRange(r: Range): string {
  return `${fmt(r.minMs)}–${fmt(r.maxMs)} (med ${fmt(r.medianMs)})`;
}

function assertBitIdentical(label: string, ref: Float64Array, got: Float64Array): void {
  if (ref.length !== got.length) {
    throw new Error(`${label}: length divergence ${ref.length} vs ${got.length}`);
  }
  for (let i = 0; i < ref.length; i++) {
    if (!Object.is(ref[i], got[i])) {
      throw new Error(`${label}: bit divergence at index ${i}: ${ref[i]} vs ${got[i]} — refusing to time a wrong result`);
    }
  }
}

console.log("=== NumType Kern 06 bench: threaded matmul (nt_matmul_blocked_partial across a worker pool) ===\n");
console.log("Loading stable core (single-threaded nt_matmul_blocked baseline)...");
const stableCore: CoreExports = await initCore();
console.log("Loading threaded core + spawning pools for workers in", WORKER_COUNTS, "...");
const pools = new Map<number, ThreadedPool>();
for (const wc of WORKER_COUNTS) {
  pools.set(wc, await initThreadedCore(wc));
}
console.log("Pools ready.\n");

const rng = makeRng(0x5448524444424e4348n); // arbitrary distinct seed

/** Single-threaded `nt_matmul_blocked` baseline via the WNDArray surface on
 * the STABLE core (identical call shape to what threadedMatmul's own
 * "workers=1 pool" ultimately dispatches through, just without any Atomics
 * round trip — the honest "no threading at all" floor). */
function baselineBlocked(shape: readonly number[], aData: Float64Array, bData: Float64Array): Float64Array {
  const a = WNDArray.fromArray(stableCore, shape, aData);
  const b = WNDArray.fromArray(stableCore, shape, bData);
  const c = a.matmul(b);
  const result = c.toArray();
  a.dispose();
  b.dispose();
  c.dispose();
  return result;
}

/** Pins the POOL route explicitly: this bench's job is to measure worker
 * dispatch itself (Series A scaling, Series B overhead), so the size-based
 * auto-router must not quietly reroute small cases to the main thread. */
const FORCE_POOL = { minPoolWork: 0 } as const;

function threadedRun(pool: ThreadedPool, shape: readonly number[], aData: Float64Array, bData: Float64Array): Float64Array {
  const a = WNDArray.fromArray(pool.core, shape, aData);
  const b = WNDArray.fromArray(pool.core, shape, bData);
  const c = threadedMatmul(pool, a, b, FORCE_POOL);
  const result = c.toArray();
  a.dispose();
  b.dispose();
  c.dispose();
  return result;
}

/** The auto-routed call (default threshold) — what a caller actually gets
 * since the auto-routing follow-up. Shown in Series B for the record: at
 * n=64 (0.26 Mops, exactly the measured threshold) the router KEEPS the
 * pool, because within the threads core the pool route is faster there —
 * the crossover bench showed Series B's <1.00x readings at n=64 come from
 * the STABLE-core baseline + marshalling, not from dispatch overhead. The
 * router's savings live at sizes smaller than any Series B measures (see
 * bench:crossover, n<=48). */
function threadedRunAuto(pool: ThreadedPool, shape: readonly number[], aData: Float64Array, bData: Float64Array): Float64Array {
  const a = WNDArray.fromArray(pool.core, shape, aData);
  const b = WNDArray.fromArray(pool.core, shape, bData);
  const c = threadedMatmul(pool, a, b);
  const result = c.toArray();
  a.dispose();
  b.dispose();
  c.dispose();
  return result;
}

// --- Series A: headline sizes ------------------------------------------
console.log(`--- Series A: [n,n] @ [n,n], single-threaded nt_matmul_blocked vs threadedMatmul across worker counts ${JSON.stringify(WORKER_COUNTS)} ---`);
interface RowA {
  n: number;
  baseline: Range;
  perWorker: Map<number, Range>;
}
const rowsA: RowA[] = [];

for (const n of SIZES_A) {
  const shape = [n, n];
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);

  const ref = matmulRuntime(shape, aData, shape, bData).data;
  const baselineResult = baselineBlocked(shape, aData, bData);
  assertBitIdentical(`series A n=${n} (single-threaded nt_matmul_blocked)`, ref, baselineResult);

  const perWorker = new Map<number, Range>();
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const got = threadedRun(pool, shape, aData, bData);
    assertBitIdentical(`series A n=${n} workers=${wc}`, ref, got);
  }

  const baseline = measureRange(() => void baselineBlocked(shape, aData, bData));
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const r = measureRange(() => void threadedRun(pool, shape, aData, bData));
    perWorker.set(wc, r);
  }

  rowsA.push({ n, baseline, perWorker });
}

const headerA = `${"n".padStart(6)} | ${"single-thread".padStart(20)} | ` + WORKER_COUNTS.map((wc) => `workers=${wc}`.padStart(28)).join(" | ");
console.log(headerA);
console.log("-".repeat(headerA.length));
for (const r of rowsA) {
  const cells = WORKER_COUNTS.map((wc) => {
    const range = r.perWorker.get(wc)!;
    const speedup = r.baseline.medianMs / range.medianMs;
    return `${fmtRange(range)} (${speedup.toFixed(2)}x)`.padStart(28);
  });
  console.log(`${String(r.n).padStart(6)} | ${fmtRange(r.baseline).padStart(20)} | ${cells.join(" | ")}`);
}
console.log("(Nx in parens = speedup vs the single-threaded nt_matmul_blocked baseline's median; ranges are min–max over " + N_REPS + " timed reps after 2 warmup calls.)\n");

// --- Series B: dispatch overhead at small n (honesty, explicitly required) --
console.log(`--- Series B: n=${SIZE_B} — dispatch overhead at a size too small to amortize thread coordination ---`);
{
  const shape = [SIZE_B, SIZE_B];
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);
  const ref = matmulRuntime(shape, aData, shape, bData).data;

  const baselineResult = baselineBlocked(shape, aData, bData);
  assertBitIdentical(`series B n=${SIZE_B} (single-threaded)`, ref, baselineResult);
  const baseline = measureRange(() => void baselineBlocked(shape, aData, bData));

  console.log(`${"config".padStart(16)} | ${"range".padStart(28)} | vs single-thread`);
  console.log("-".repeat(70));
  console.log(`${"single-thread".padStart(16)} | ${fmtRange(baseline).padStart(28)} | 1.00x (baseline)`);
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const got = threadedRun(pool, shape, aData, bData);
    assertBitIdentical(`series B n=${SIZE_B} workers=${wc}`, ref, got);
    const r = measureRange(() => void threadedRun(pool, shape, aData, bData));
    const speedup = baseline.medianMs / r.medianMs;
    const mark = speedup >= 1 ? "" : "  <-- threads LOSE here";
    console.log(`${`workers=${wc}`.padStart(16)} | ${fmtRange(r).padStart(28)} | ${speedup.toFixed(2)}x${mark}`);
  }
  // Auto-routed row (default threshold), for the record — n=64 sits exactly
  // AT the measured threshold (64^3 = THREADED_MATMUL_MIN_POOL_WORK), so the
  // router keeps it on the pool; see threadedRunAuto's doc for why that is
  // the right call despite this table's <1.00x readings.
  {
    const pool = pools.get(WORKER_COUNTS[WORKER_COUNTS.length - 1]!)!;
    const got = threadedRunAuto(pool, shape, aData, bData);
    assertBitIdentical(`series B n=${SIZE_B} auto-routed`, ref, got);
    const r = measureRange(() => void threadedRunAuto(pool, shape, aData, bData));
    const speedup = baseline.medianMs / r.medianMs;
    console.log(`${"auto (routed)".padStart(16)} | ${fmtRange(r).padStart(28)} | ${speedup.toFixed(2)}x  (workers=${WORKER_COUNTS[WORKER_COUNTS.length - 1]} pool — at/above threshold)`);
  }
  console.log(
    "\n(At n=64 the per-call Atomics dispatch/wait round trip across the pool is a real, measured cost; the workers=N\n" +
      " rows pin the POOL route explicitly ({minPoolWork: 0}) to keep measuring it honestly; if any workers>1 row\n" +
      " shows <1.00x, that is an honest loss, not hidden — but note it is measured against the STABLE core incl.\n" +
      " fromArray/toArray marshalling; the router's own decision baseline is the threads core, see bench:crossover.)",
  );
}

console.log("\nDisposing pools...");
for (const pool of pools.values()) await pool.dispose();
console.log(`getThreadedPoolFreeCount() after disposing all pools: ${getThreadedPoolFreeCount()}`);
console.log("Done.");
