/**
 * Kern 11 bench — run via `pnpm bench:elementwise`; numbers recorded in
 * docs/kern-11-elementwise-fastpath-ergebnisse.md. Same discipline as the
 * other benches: seeded inputs, bit-identity gate BEFORE any timing, warmed
 * JIT, adaptive batch-timed reps, ranges over sizes reported rather than
 * single points.
 *
 * Kern 11's fast path (crates/core/src/kernels/add.rs `add_strided`,
 * crates/core/src/kernels/elementwise.rs `binary_strided`) skips the
 * per-element `unravel` allocation for the most common elementwise case:
 * both operands contiguous, same shape (no broadcast), offset 0. This bench
 * measures add/sub/mul/div on resident `WNDArray`s in two configurations at
 * the SAME sizes, through the SAME binary:
 *
 *  - "contiguous": both operands `WNDArray.fromArray` (natural strides,
 *    offset 0) -> hits the new fast path.
 *  - "non-contiguous": one operand is `.transpose()`d first (swapped
 *    strides, still offset 0 for a square array) -> fails the
 *    `a_strides == compute_strides(a_shape)` guard, so it is forced onto
 *    the UNCHANGED general `unravel`-based loop.
 *
 * Both configurations are gated bit-identical against the naive TS
 * reference (`elementwiseBinary` for contiguous; `transposeRuntime` +
 * `elementwiseBinary` for the non-contiguous case) before any timing. The
 * contiguous-vs-non-contiguous ratio at each size directly evidences the
 * fast path's win — no pre-fix binary is needed for the comparison, since
 * the non-contiguous case exercises the byte-for-byte UNCHANGED general
 * path in the same artifact (Kern 11 spec D1: "NICHT angefasst" outside the
 * two fast-path guards). The opening measurement that motivated this slice
 * (docs/kern-11-elementwise-fastpath-spec.md "Messgrundlage") found ~8-10x
 * on throwaway prototypes (mul 12.9->1.65ms, div 12.7->1.31ms at 1024x1024);
 * this bench re-measures the committed kernel honestly, not that prototype.
 */
import { elementwiseBinary, transposeRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES = [128, 256, 512, 1024];
const TARGET_MS = 150;
const MIN_REPS = 2;
const MAX_REPS = 50;

type Op = "add" | "sub" | "mul" | "div";
const OPS: readonly Op[] = ["add", "sub", "mul", "div"];

function refOp(op: Op): (x: number, y: number) => number {
  if (op === "add") return (x, y) => x + y;
  if (op === "sub") return (x, y) => x - y;
  if (op === "mul") return (x, y) => x * y;
  return (x, y) => x / y;
}

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
  return `${(ms * 1000).toFixed(2)}us`;
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

function callResident(op: Op, a: AnyWNDArray, b: AnyWNDArray): AnyWNDArray {
  if (op === "add") return a.add(b);
  if (op === "sub") return a.sub(b);
  if (op === "mul") return a.mul(b);
  return a.div(b);
}

console.log("=== NumType Kern 11 bench: elementwise contiguous fast path vs general path ===\n");
console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x4b45524e31315f45n); // "KERN11_E"-ish, arbitrary fixed seed

interface Row {
  op: Op;
  n: number;
  refAvg: number;
  contigAvg: number;
  nonContigAvg: number;
  reps: string;
}
const rows: Row[] = [];

for (const op of OPS) {
  for (const n of SIZES) {
    const shape = [n, n];
    const aData = genData(rng, shape);
    const bData = genData(rng, shape);

    // --- contiguous: both operands natural-strided, offset 0 -> fast path ---
    const aRes = WNDArray.fromArray(core, shape, aData);
    const bRes = WNDArray.fromArray(core, shape, bData);

    const refContig = elementwiseBinary(shape, aData, shape, bData, refOp(op)).data;

    function runContig(): Float64Array {
      const out = callResident(op, aRes, bRes);
      const result = out.toArray();
      out.dispose();
      return result;
    }
    assertBitIdentical(`${op} n=${n} (contiguous)`, refContig, runContig());

    // --- non-contiguous: transpose one operand (square, so shape is
    // unchanged but strides are swapped) -> fails the natural-strides
    // guard, forced onto the general path. ---
    const refTransposedA = transposeRuntime(shape, aData);
    const refNonContig = elementwiseBinary(refTransposedA.shape, refTransposedA.data, shape, bData, refOp(op)).data;

    function runNonContig(): Float64Array {
      const aT = aRes.transpose();
      const out = callResident(op, aT, bRes);
      const result = out.toArray();
      out.dispose();
      return result;
    }
    assertBitIdentical(`${op} n=${n} (non-contiguous)`, refNonContig, runNonContig());

    const ref = measureAvgMs(() => void elementwiseBinary(shape, aData, shape, bData, refOp(op)));
    const contig = measureAvgMs(runContig);
    const nonContig = measureAvgMs(runNonContig);

    rows.push({
      op,
      n,
      refAvg: ref.avgMs,
      contigAvg: contig.avgMs,
      nonContigAvg: nonContig.avgMs,
      reps: `${ref.reps}/${contig.reps}/${nonContig.reps}`,
    });

    aRes.dispose();
    bRes.dispose();
  }
}

const header =
  `${"op".padStart(4)} | ${"n".padStart(6)} | ${"naive TS".padStart(11)} | ${"contiguous".padStart(11)} | ` +
  `${"non-contig".padStart(11)} | ${"fast/general".padStart(12)} | ${"naive/fast".padStart(10)} | reps (ref/contig/noncontig)`;
console.log(header);
console.log("-".repeat(header.length));
for (const r of rows) {
  const fastVsGeneral = r.nonContigAvg / r.contigAvg;
  const naiveVsFast = r.refAvg / r.contigAvg;
  console.log(
    `${r.op.padStart(4)} | ${String(r.n).padStart(6)} | ${fmt(r.refAvg).padStart(11)} | ${fmt(r.contigAvg).padStart(11)} | ` +
      `${fmt(r.nonContigAvg).padStart(11)} | ${`${fastVsGeneral.toFixed(2)}x`.padStart(12)} | ${`${naiveVsFast.toFixed(2)}x`.padStart(10)} | ${r.reps}`,
  );
}
console.log(
  "\n('fast/general' = non-contiguous time / contiguous time, SAME binary — the fast path's internal win. " +
    "'naive/fast' = naive TS time / contiguous resident time.)",
);
