/**
 * Kern 02 chain bench: a fixed chain of k=8 ops (mixed add/matmul,
 * alternating, all square `[n,n]` operands) at n=128/256/512, comparing
 * naive TS vs v1 (per-op copy-in/copy-out) vs WASM resident END-TO-END.
 * Run via `pnpm bench:chain`. Numbers recorded in docs/kern-02-ergebnisse.md.
 *
 * This is deliberately a DIFFERENT scenario from `scaling.ts`'s resident
 * series (which excludes boundary copies entirely, to isolate pure
 * pointer-to-pointer op cost). Here the boundary copies ARE part of what's
 * measured, once per full pipeline run, contrasted with v1's cost of
 * paying them at every step:
 *
 *  - naive TS: no WASM involved at all — everything is already a plain
 *    `Float64Array`; the pure-compute baseline.
 *  - v1 (WASM+copies): every `wasmAdd`/`wasmMatmul` call copies BOTH its
 *    operands in and its result out. Chaining 8 calls means 8 full
 *    copy-in/copy-out round trips — 7 of those "inputs" are really the
 *    previous step's own (just copied-out) result, copied straight back in.
 *  - resident (v2): the seed array and all 8 per-step operands are
 *    constructed via `fromArray` (one boundary-in round trip each, but
 *    only once for the whole pipeline, not once per op); the 8 ops then
 *    run purely pointer-to-pointer, each intermediate disposed as soon as
 *    the next op consumes it; a single `toArray()` reads the final result
 *    out. This — one bounded round of boundary copies for an entire
 *    multi-op pipeline, vs. v1's one round PER op — is the realistic
 *    "resident pipeline" usage pattern the spec calls out.
 *
 * Each timed rep runs the ENTIRE pipeline (construct -> 8 ops -> read out)
 * exactly once, for all three series — this intentionally captures the
 * one-time boundary cost the resident series still has to pay per
 * invocation, so the comparison stays apples-to-apples: "how long does one
 * full run of this pipeline take," not "how fast are only the ops in the
 * middle." Same seeded-PRNG determinism and bit-identity-before-timing
 * discipline as `scaling.ts`.
 */
import { elementwiseBinary, matmulRuntime } from "../src/runtime.ts";
import { wasmAdd, wasmMatmul } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { type AnyWNDArray, WNDArray } from "../src/wasm/resident.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES = [128, 256, 512];
const CHAIN_LEN = 8; // alternating add (even i) / matmul (odd i)
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

console.log("=== NumType Kern 02 chain bench: naive TS vs WASM v1 (copies) vs WASM resident ===\n");
console.log(`Chain: ${CHAIN_LEN} ops (add, matmul, add, matmul, ...), square [n,n] operands throughout.\n`);

console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x4348_4149_4e42_4e43n); // "CHAINBNC"-ish, arbitrary fixed seed

interface Row {
  n: number;
  tsAvg: number;
  wasmAvg: number;
  residentAvg: number;
  tsReps: number;
  wasmReps: number;
  residentReps: number;
}

const rows: Row[] = [];

for (const n of SIZES) {
  const shape = [n, n];
  const seedData = genData(rng, shape);
  const stepData: Float64Array[] = [];
  for (let i = 0; i < CHAIN_LEN; i++) stepData.push(genData(rng, shape));

  function naiveChain(): Float64Array {
    let curShape: readonly number[] = shape;
    let curData = seedData;
    for (let i = 0; i < CHAIN_LEN; i++) {
      const step = stepData[i];
      if (step === undefined) throw new Error("chain bench: missing step data");
      const r =
        i % 2 === 0
          ? elementwiseBinary(curShape, curData, shape, step, (x, y) => x + y)
          : matmulRuntime(curShape, curData, shape, step);
      curShape = r.shape;
      curData = r.data;
    }
    return curData;
  }

  function v1Chain(): Float64Array {
    let curShape: readonly number[] = shape;
    let curData = seedData;
    for (let i = 0; i < CHAIN_LEN; i++) {
      const step = stepData[i];
      if (step === undefined) throw new Error("chain bench: missing step data");
      const r = i % 2 === 0 ? wasmAdd(core, curShape, curData, shape, step) : wasmMatmul(core, curShape, curData, shape, step);
      curShape = r.shape;
      curData = r.data;
    }
    return curData;
  }

  function residentChain(): Float64Array {
    // Boundary-IN: construct every operand this pipeline run needs, once.
    let cur: AnyWNDArray = WNDArray.fromArray(core, shape, Array.from(seedData));
    const stepArrays: AnyWNDArray[] = stepData.map((d) => WNDArray.fromArray(core, shape, Array.from(d)));
    // Pure pointer-to-pointer chain: dispose each intermediate as soon as
    // the next op has consumed it.
    for (let i = 0; i < CHAIN_LEN; i++) {
      const step = stepArrays[i];
      if (step === undefined) throw new Error("chain bench: missing step operand");
      const next = i % 2 === 0 ? cur.add(step) : cur.matmul(step);
      cur.dispose();
      cur = next;
    }
    // Boundary-OUT: a single copy-out for the whole pipeline.
    const result = cur.toArray();
    cur.dispose();
    for (const s of stepArrays) s.dispose();
    return result;
  }

  // Bit-identity gate before any timing — never time a wrong chain.
  const ref = naiveChain();
  assertBitIdentical(`chain n=${n} (v1)`, ref, v1Chain());
  assertBitIdentical(`chain n=${n} (resident)`, ref, residentChain());

  const ts = measureAvgMs(naiveChain);
  const wasm = measureAvgMs(v1Chain);
  const resident = measureAvgMs(residentChain);

  rows.push({
    n,
    tsAvg: ts.avgMs,
    wasmAvg: wasm.avgMs,
    residentAvg: resident.avgMs,
    tsReps: ts.reps,
    wasmReps: wasm.reps,
    residentReps: resident.reps,
  });
}

console.log(`--- chain of ${CHAIN_LEN} ops (add/matmul alternating), square [n,n] ---`);
const header =
  `${"n".padStart(5)} | ${"naive TS".padStart(11)} | ${"WASM+copies".padStart(11)} | ${"WASM resident".padStart(13)} | ` +
  `${"res/naive".padStart(10)} | ${"res/v1".padStart(8)} | reps (ts/wasm/res)`;
console.log(header);
console.log("-".repeat(header.length));
for (const r of rows) {
  const vsNaive = r.tsAvg / r.residentAvg;
  const vsV1 = r.wasmAvg / r.residentAvg;
  const markNaive = vsNaive >= 1 ? `${vsNaive.toFixed(2)}x` : `${vsNaive.toFixed(2)}x <`;
  const markV1 = vsV1 >= 1 ? `${vsV1.toFixed(2)}x` : `${vsV1.toFixed(2)}x <`;
  console.log(
    `${String(r.n).padStart(5)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.wasmAvg).padStart(11)} | ${fmt(r.residentAvg).padStart(13)} | ` +
      `${markNaive.padStart(10)} | ${markV1.padStart(8)} | ${r.tsReps}/${r.wasmReps}/${r.residentReps}`,
  );
}
console.log();
console.log("Notes: each timed rep runs the WHOLE pipeline once, including its boundary copies —");
console.log("v1 pays copy-in/copy-out at every one of the 8 steps; resident pays it once, at the ends.");
console.log("The resident boundary-in cost includes converting Float64Array -> number[] for fromArray");
console.log("(WNDArray.fromArray mirrors NDArray.fromArray's own `readonly number[]` signature exactly).");
console.log("A '<' marker means the left side of that comparison is SLOWER. Bit-identity checked first.");
