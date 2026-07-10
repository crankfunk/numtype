/**
 * Kern 05 slice bench — run via `pnpm bench:slice`; numbers recorded in
 * docs/kern-05-ergebnisse.md. Same discipline as the other benches: seeded
 * inputs, bit-identity gate BEFORE any timing, adaptive batch-timed reps,
 * warmed JIT, ranges over sizes reported rather than single points.
 *
 * Deliberately SMALL: slicing itself is metadata (`WNDArray.slice` never
 * touches the buffer — see resident.ts). The claim worth measuring is the
 * PIPELINE effect of feeding a slice straight into a reducing op without
 * materializing it first.
 *
 * Series A — the payoff claim: `A.slice({start:0,stop:n/2}) -> sum()` on a
 * resident `[n,n]` operand (a contiguous row-block: the view's stride
 * pattern is UNCHANGED, only the offset/dim shrink) vs the naive TS
 * equivalent (`sliceRuntime` gather + `sumRuntime`). Both compute the exact
 * same reduction; the resident path never copies the row-block out.
 *
 * Series B — the honest strided-read cost: `A.slice({step:2}) -> sum()` —
 * a GENUINELY strided view (every other row), vs the naive TS gather+sum
 * equivalent. This is the regime where the resident kernel must walk a
 * non-unit stride, mirroring strided.ts's Series C (`transpose().sum()`)
 * but for slicing instead of transposition.
 */
import { normalizeSliceSpecs, sliceRuntime, sumRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES = [256, 512, 1024];
const TARGET_MS = 150;
const MIN_REPS = 2;
const MAX_REPS = 50;

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

console.log("=== NumType Kern 05 bench: slice -> sum, view vs naive ===\n");
console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x534c4943455f424en); // "SLICE_BN"-ish

// --- Series A: row-block slice (contiguous view) -> sum -------------------
console.log("--- Series A: A.slice({start:0,stop:n/2}) -> sum() — row-block (contiguous view) ---");
interface RowA {
  n: number;
  tsAvg: number;
  viewAvg: number;
  tsReps: number;
  viewReps: number;
}
const rowsA: RowA[] = [];

for (const n of SIZES) {
  const shape = [n, n];
  const data = genData(rng, shape);
  const half = Math.floor(n / 2);
  const specs = [{ start: 0, stop: half }];

  const rA = WNDArray.fromArray(core, shape, data);

  function naive(): Float64Array {
    const sliced = sliceRuntime(shape, data, normalizeSliceSpecs(shape, specs));
    return sumRuntime(sliced.shape, sliced.data, undefined).data;
  }

  function viaView(): Float64Array {
    const view = rA.slice(...specs);
    const s = view.sum();
    view.dispose();
    const result = s.toArray();
    s.dispose();
    return result;
  }

  const ref = naive();
  assertBitIdentical(`series A n=${n} (view)`, ref, viaView());

  const ts = measureAvgMs(() => void naive());
  const view = measureAvgMs(() => void viaView());
  rowsA.push({ n, tsAvg: ts.avgMs, viewAvg: view.avgMs, tsReps: ts.reps, viewReps: view.reps });

  rA.dispose();
}

const headerA = `${"n".padStart(5)} | ${"naive TS".padStart(11)} | ${"view".padStart(11)} | ${"view/ts".padStart(8)} | reps (ts/view)`;
console.log(headerA);
console.log("-".repeat(headerA.length));
for (const r of rowsA) {
  const ratio = r.tsAvg / r.viewAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(
    `${String(r.n).padStart(5)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.viewAvg).padStart(11)} | ${mark.padStart(8)} | ${r.tsReps}/${r.viewReps}`,
  );
}
console.log("('view/ts' > 1x means the resident view path is FASTER than naive TS slice+sum.)\n");

// --- Series B: step-2 slice (genuinely strided view) -> sum ----------------
console.log("--- Series B: A.slice({step:2}) -> sum() — every other row (honest strided-read cost) ---");
interface RowB {
  n: number;
  tsAvg: number;
  viewAvg: number;
  tsReps: number;
  viewReps: number;
}
const rowsB: RowB[] = [];

for (const n of SIZES) {
  const shape = [n, n];
  const data = genData(rng, shape);
  const specs = [{ step: 2 }];

  const rA = WNDArray.fromArray(core, shape, data);

  function naive(): Float64Array {
    const sliced = sliceRuntime(shape, data, normalizeSliceSpecs(shape, specs));
    return sumRuntime(sliced.shape, sliced.data, undefined).data;
  }

  function viaView(): Float64Array {
    const view = rA.slice(...specs);
    const s = view.sum();
    view.dispose();
    const result = s.toArray();
    s.dispose();
    return result;
  }

  const ref = naive();
  assertBitIdentical(`series B n=${n} (view)`, ref, viaView());

  const ts = measureAvgMs(() => void naive());
  const view = measureAvgMs(() => void viaView());
  rowsB.push({ n, tsAvg: ts.avgMs, viewAvg: view.avgMs, tsReps: ts.reps, viewReps: view.reps });

  rA.dispose();
}

const headerB = `${"n".padStart(5)} | ${"naive TS".padStart(11)} | ${"view".padStart(11)} | ${"view/ts".padStart(8)} | reps (ts/view)`;
console.log(headerB);
console.log("-".repeat(headerB.length));
for (const r of rowsB) {
  const ratio = r.tsAvg / r.viewAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(
    `${String(r.n).padStart(5)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.viewAvg).padStart(11)} | ${mark.padStart(8)} | ${r.tsReps}/${r.viewReps}`,
  );
}
console.log("('view/ts' > 1x means the resident view path is FASTER than naive TS slice+sum.)");
