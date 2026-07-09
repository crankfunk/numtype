/**
 * Kern 01 bench (informational, no threshold gate per spec): matmul
 * [256,256] x [256,256], wall-clock naive TS (`matmulRuntime`) vs WASM
 * (`wasmMatmul`) — the WASM timing includes the copy-in/copy-out overhead
 * (v1 is copy-in/copy-out by design, see spec). Run via `pnpm bench:core`.
 * Numbers recorded in docs/kern-01-ergebnisse.md.
 *
 * Lives in `spike/bench-core/` (NOT `spike/bench/`) deliberately:
 * `spike/bench/` is spike-01's isolated-project instantiation-diagnostics
 * harness (`check:diag:bench`, `-p spike/bench`, whose `tsconfig.json`
 * includes the whole directory) — dropping a new file in there would
 * silently change what that pre-existing, already-recorded measurement
 * compiles.
 */
import { matmulRuntime } from "../src/runtime.ts";
import { wasmMatmul } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";

const N = 256;
const ITERATIONS = 10;

function randomMatrix(n: number): Float64Array {
  const data = new Float64Array(n * n);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  return data;
}

function stats(times: readonly number[]): { min: number; avg: number; max: number } {
  const min = Math.min(...times);
  const max = Math.max(...times);
  const avg = times.reduce((acc, t) => acc + t, 0) / times.length;
  return { min, avg, max };
}

function timeIterations(fn: () => void): number[] {
  fn(); // untimed warmup
  const times: number[] = [];
  for (let i = 0; i < ITERATIONS; i++) {
    const start = performance.now();
    fn();
    const end = performance.now();
    times.push(end - start);
  }
  return times;
}

console.log(`=== NumType Kern 01 bench: matmul [${N},${N}] x [${N},${N}], ${ITERATIONS} timed iterations (+1 warmup) ===\n`);

console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const shape = [N, N];
const a = randomMatrix(N);
const b = randomMatrix(N);

// Correctness sanity check before timing — never report timings for a
// wrong result.
{
  const ref = matmulRuntime(shape, a, shape, b);
  const got = wasmMatmul(core, shape, a, shape, b);
  let identical = ref.data.length === got.data.length;
  for (let i = 0; identical && i < ref.data.length; i++) identical = Object.is(ref.data[i], got.data[i]);
  console.log(`Correctness check (bit-identical TS vs WASM): ${identical}\n`);
  if (!identical) {
    throw new Error("bench: TS and WASM matmul results diverge — refusing to report timings for a wrong result");
  }
}

const tsTimes = timeIterations(() => {
  matmulRuntime(shape, a, shape, b);
});
const wasmTimes = timeIterations(() => {
  wasmMatmul(core, shape, a, shape, b);
});

const tsStats = stats(tsTimes);
const wasmStats = stats(wasmTimes);

console.log(`naive TS            : min=${tsStats.min.toFixed(3)}ms avg=${tsStats.avg.toFixed(3)}ms max=${tsStats.max.toFixed(3)}ms`);
console.log(`WASM (incl. copies) : min=${wasmStats.min.toFixed(3)}ms avg=${wasmStats.avg.toFixed(3)}ms max=${wasmStats.max.toFixed(3)}ms`);
console.log(`speedup (avg TS / avg WASM): ${(tsStats.avg / wasmStats.avg).toFixed(2)}x`);
