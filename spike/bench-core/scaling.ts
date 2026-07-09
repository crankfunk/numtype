/**
 * Kern 01 scaling bench (informational): naive TS vs WASM (incl. v1
 * copy-in/copy-out overhead) across sizes, for `add` (elementwise, same
 * shape) and `matmul` (square) — makes the small-op crossover visible and
 * quantifies the v2 (zero-copy residency) motivation. Run via
 * `pnpm bench:scaling`. Numbers recorded in docs/kern-01-ergebnisse.md.
 *
 * Methodology: seeded inputs (splitmix64, deterministic across runs);
 * per-config bit-identity gate before any timing; adaptive repetition count
 * (batch-timed, targeting ~150ms per measurement, min 3 reps) so that
 * microsecond-scale ops aren't measured inside timer noise.
 */
import { elementwiseBinary, matmulRuntime } from "../src/runtime.ts";
import { wasmAdd, wasmMatmul } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const ADD_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024];
const MATMUL_SIZES = [8, 16, 32, 64, 128, 256, 512];
const TARGET_MS = 150;
const MIN_REPS = 3;
const MAX_REPS = 5000;

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

/** Batch-timed average: warm up, calibrate one call, then time `reps` calls
 * inside a single performance.now() window. */
function measureAvgMs(fn: () => void): { avgMs: number; reps: number } {
  fn();
  fn(); // warmup
  const t0 = performance.now();
  fn();
  const oneCall = performance.now() - t0;
  const reps = Math.min(MAX_REPS, Math.max(MIN_REPS, Math.ceil(TARGET_MS / Math.max(oneCall, 0.0005))));
  const start = performance.now();
  for (let i = 0; i < reps; i++) fn();
  const total = performance.now() - start;
  return { avgMs: total / reps, reps };
}

function fmt(ms: number): string {
  if (ms >= 1) return `${ms.toFixed(3)}ms`;
  return `${(ms * 1000).toFixed(2)}µs`;
}

console.log("=== NumType Kern 01 scaling bench: naive TS vs WASM (incl. copies) ===\n");

console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x5343_414c_494e_47n); // "SCALING"

interface Row {
  n: number;
  work: string;
  tsAvg: number;
  wasmAvg: number;
  tsReps: number;
  wasmReps: number;
}

function printTable(title: string, rows: readonly Row[], workHeader: string): void {
  console.log(`--- ${title} ---`);
  const header = `${"n".padStart(5)} | ${workHeader.padStart(12)} | ${"naive TS".padStart(11)} | ${"WASM+copies".padStart(11)} | ${"speedup".padStart(8)} | reps (ts/wasm)`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const r of rows) {
    const speedup = r.tsAvg / r.wasmAvg;
    const marker = speedup >= 1 ? `${speedup.toFixed(2)}x` : `${speedup.toFixed(2)}x <`; // "<" flags WASM slower
    console.log(
      `${String(r.n).padStart(5)} | ${r.work.padStart(12)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.wasmAvg).padStart(11)} | ${marker.padStart(8)} | ${r.tsReps}/${r.wasmReps}`,
    );
  }
  console.log();
}

// --- add: [n,n] + [n,n] -----------------------------------------------------
const addRows: Row[] = [];
for (const n of ADD_SIZES) {
  const shape = [n, n];
  const a = genData(rng, shape);
  const b = genData(rng, shape);

  const ref = elementwiseBinary(shape, a, shape, b, (x, y) => x + y);
  const got = wasmAdd(core, shape, a, shape, b);
  assertBitIdentical(`add ${n}x${n}`, ref.data, got.data);

  const ts = measureAvgMs(() => {
    elementwiseBinary(shape, a, shape, b, (x, y) => x + y);
  });
  const wasm = measureAvgMs(() => {
    wasmAdd(core, shape, a, shape, b);
  });
  addRows.push({ n, work: `${n * n} elems`, tsAvg: ts.avgMs, wasmAvg: wasm.avgMs, tsReps: ts.reps, wasmReps: wasm.reps });
}
printTable("add [n,n] + [n,n]", addRows, "elements");

// --- matmul: [n,n] x [n,n] --------------------------------------------------
const matmulRows: Row[] = [];
for (const n of MATMUL_SIZES) {
  const shape = [n, n];
  const a = genData(rng, shape);
  const b = genData(rng, shape);

  const ref = matmulRuntime(shape, a, shape, b);
  const got = wasmMatmul(core, shape, a, shape, b);
  assertBitIdentical(`matmul ${n}x${n}`, ref.data, got.data);

  const ts = measureAvgMs(() => {
    matmulRuntime(shape, a, shape, b);
  });
  const wasm = measureAvgMs(() => {
    wasmMatmul(core, shape, a, shape, b);
  });
  matmulRows.push({
    n,
    work: `${(2 * n * n * n / 1e6).toFixed(1)} MFLOP`,
    tsAvg: ts.avgMs,
    wasmAvg: wasm.avgMs,
    tsReps: ts.reps,
    wasmReps: wasm.reps,
  });
}
printTable("matmul [n,n] x [n,n]", matmulRows, "work");

console.log("Notes: WASM timings include full v1 copy-in/copy-out overhead (marshalling, alloc/free).");
console.log("A '<' marker means WASM is SLOWER than naive TS at that size (copy/call overhead dominates).");
console.log("All timed configurations passed the bit-identity gate first.");
