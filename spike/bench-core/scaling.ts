/**
 * Scaling bench: naive TS vs WASM v1 (copy-in/copy-out) vs WASM resident
 * (Kern 02) across sizes, for `add` (elementwise, same shape) and `matmul`
 * (square) — makes the small-op crossover visible and quantifies whether
 * zero-copy residency delivered on the copy-overhead motivation Kern 01
 * measured. Run via `pnpm bench:scaling`. Kern 01 numbers recorded in
 * docs/kern-01-ergebnisse.md; the v2-vs-v1 delta is recorded in
 * docs/kern-02-ergebnisse.md.
 *
 * Methodology: seeded inputs (splitmix64, deterministic across runs);
 * per-config bit-identity gate before any timing; adaptive repetition count
 * (batch-timed, targeting ~150ms per measurement, min 3 reps) so that
 * microsecond-scale ops aren't measured inside timer noise.
 *
 * Resident series methodology: operands are constructed as `WNDArray`s
 * ONCE per size (outside the timed loop) — the whole point of residency is
 * that a caller does this once and reuses the operands across many ops, so
 * timing a fresh `fromArray` on every rep would misrepresent the steady
 * state this backend targets. Each *timed* call is `op(...).dispose()`:
 * the result is freed every rep so memory stays bounded across thousands
 * of reps (dispose is one `nt_free` call — negligible next to the kernel
 * work being measured, and it is what a real chained caller does between
 * resident ops anyway). Boundary copies (`fromArray`/`toArray`) are
 * deliberately EXCLUDED from the timed resident series — that is the
 * whole point of "resident": this bench isolates the pointer-to-pointer
 * cost. `chain.ts` measures the opposite, boundary-inclusive scenario.
 */
import { elementwiseBinary, matmulRuntime } from "../src/runtime.ts";
import { wasmAdd, wasmMatmul } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
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

console.log("=== NumType scaling bench: naive TS vs WASM v1 (copies) vs WASM resident ===\n");

console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x5343_414c_494e_47n); // "SCALING"

interface Row {
  n: number;
  work: string;
  tsAvg: number;
  wasmAvg: number;
  residentAvg: number;
  tsReps: number;
  wasmReps: number;
  residentReps: number;
}

function printTable(title: string, rows: readonly Row[], workHeader: string): void {
  console.log(`--- ${title} ---`);
  const header =
    `${"n".padStart(5)} | ${workHeader.padStart(12)} | ${"naive TS".padStart(11)} | ${"WASM+copies".padStart(11)} | ` +
    `${"WASM resident".padStart(13)} | ${"res/naive".padStart(10)} | ${"res/v1".padStart(8)} | reps (ts/wasm/res)`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const r of rows) {
    const vsNaive = r.tsAvg / r.residentAvg;
    const vsV1 = r.wasmAvg / r.residentAvg;
    const markNaive = vsNaive >= 1 ? `${vsNaive.toFixed(2)}x` : `${vsNaive.toFixed(2)}x <`; // "<" flags resident slower
    const markV1 = vsV1 >= 1 ? `${vsV1.toFixed(2)}x` : `${vsV1.toFixed(2)}x <`;
    console.log(
      `${String(r.n).padStart(5)} | ${r.work.padStart(12)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.wasmAvg).padStart(11)} | ` +
        `${fmt(r.residentAvg).padStart(13)} | ${markNaive.padStart(10)} | ${markV1.padStart(8)} | ${r.tsReps}/${r.wasmReps}/${r.residentReps}`,
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

  const rA = WNDArray.fromArray(core, shape, a);
  const rB = WNDArray.fromArray(core, shape, b);
  const residentCheck = rA.add(rB);
  assertBitIdentical(`add ${n}x${n} (resident)`, ref.data, residentCheck.toArray());
  residentCheck.dispose();

  const ts = measureAvgMs(() => {
    elementwiseBinary(shape, a, shape, b, (x, y) => x + y);
  });
  const wasm = measureAvgMs(() => {
    wasmAdd(core, shape, a, shape, b);
  });
  const resident = measureAvgMs(() => {
    rA.add(rB).dispose();
  });
  rA.dispose();
  rB.dispose();

  addRows.push({
    n,
    work: `${n * n} elems`,
    tsAvg: ts.avgMs,
    wasmAvg: wasm.avgMs,
    residentAvg: resident.avgMs,
    tsReps: ts.reps,
    wasmReps: wasm.reps,
    residentReps: resident.reps,
  });
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

  const rA = WNDArray.fromArray(core, shape, a);
  const rB = WNDArray.fromArray(core, shape, b);
  const residentCheck = rA.matmul(rB);
  assertBitIdentical(`matmul ${n}x${n} (resident)`, ref.data, residentCheck.toArray());
  residentCheck.dispose();

  const ts = measureAvgMs(() => {
    matmulRuntime(shape, a, shape, b);
  });
  const wasm = measureAvgMs(() => {
    wasmMatmul(core, shape, a, shape, b);
  });
  const resident = measureAvgMs(() => {
    rA.matmul(rB).dispose();
  });
  rA.dispose();
  rB.dispose();

  matmulRows.push({
    n,
    work: `${(2 * n * n * n / 1e6).toFixed(1)} MFLOP`,
    tsAvg: ts.avgMs,
    wasmAvg: wasm.avgMs,
    residentAvg: resident.avgMs,
    tsReps: ts.reps,
    wasmReps: wasm.reps,
    residentReps: resident.reps,
  });
}
printTable("matmul [n,n] x [n,n]", matmulRows, "work");

console.log("Notes: 'WASM+copies' (v1) includes full copy-in/copy-out overhead (marshalling, alloc/free).");
console.log("'WASM resident' (v2) times only the pointer-to-pointer op + fresh output alloc + dispose —");
console.log("operands are constructed once (fromArray) OUTSIDE the timed loop, per size.");
console.log("A '<' marker means the left side of that comparison is SLOWER at that size.");
console.log("All timed configurations (v1 AND resident) passed the bit-identity gate first.");
