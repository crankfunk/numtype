/**
 * Kern 03 strided bench — run via `pnpm bench:strided`; numbers recorded in
 * docs/kern-03-ergebnisse.md. Same discipline as the other benches: seeded
 * inputs, bit-identity gate BEFORE any timing, adaptive batch-timed reps,
 * warmed JIT, ranges over sizes reported rather than single points.
 *
 * Series A — the payoff claim: `A.transpose() @ B` end-to-end on resident
 * operands (operand construction NOT timed; per-rep work is transpose +
 * matmul + dispose of intermediates + result):
 *   - materializing path: `A.transpose().contiguous()` then matmul — what
 *     Kern 02 effectively did (transpose was a full gather copy).
 *   - view path: `A.transpose().matmul(B)` — the transpose is O(1)
 *     metadata; the strided matmul kernel reads A column-wise in place.
 *   - naive TS for context (transposeRuntime + matmulRuntime).
 *   The trade being measured: the view path SKIPS an O(n²) copy but pays
 *   cache-unfriendlier strided reads inside the matmul k-loop. Whichever
 *   way it lands, the number is the finding (honesty rule).
 *
 * Series B — the routing-decision check: resident.ts routes EVERY op
 * through the strided entry points, passing natural strides for contiguous
 * data. What does that generalization cost vs the frozen non-strided
 * entry points on identical contiguous operands? Raw ABI-level comparison
 * with identical per-rep plumbing (scratch marshalling + output alloc/free
 * inside the timed body, mirroring what resident.ts actually does per op);
 * only the entry point differs. If it is not ~1x, that is a finding to
 * report, not to hide.
 *
 * Series C — the consume-once pattern: `A.transpose().sum()`. Both paths
 * pay the strided (column-major) reads exactly once; the materializing
 * path additionally writes and re-reads an O(n²) scratch copy. This is the
 * shape of workload where a view should win unconditionally — measured,
 * not assumed.
 */
import { matmulRuntime, transposeRuntime, computeStrides } from "../src/runtime.ts";
import { initCore, type CoreExports } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES_A = [128, 256, 512];
const SIZES_B = [64, 256, 512];
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

console.log("=== NumType Kern 03 strided bench: transpose views vs materialization ===\n");
console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x5354_5249_4442_4e43n); // "STRIDBNC"-ish, arbitrary fixed seed

// --- Series A: A.transpose() @ B — materialize vs view ---------------------
console.log("--- Series A: A.transpose() @ B on resident [n,n] operands ---");
interface RowA {
  n: number;
  tsAvg: number;
  matAvg: number;
  viewAvg: number;
  tsReps: number;
  matReps: number;
  viewReps: number;
}
const rowsA: RowA[] = [];

for (const n of SIZES_A) {
  const shape = [n, n];
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);

  const rA = WNDArray.fromArray(core, shape, aData);
  const rB = WNDArray.fromArray(core, shape, bData);

  function naive(): Float64Array {
    const t = transposeRuntime(shape, aData);
    return matmulRuntime(t.shape, t.data, shape, bData).data;
  }

  function materializing(): Float64Array {
    const view = rA.transpose();
    const mat = view.contiguous(); // the Kern-02-era full gather copy
    view.dispose();
    const out = mat.matmul(rB);
    mat.dispose();
    const result = out.toArray();
    out.dispose();
    return result;
  }

  function viaView(): Float64Array {
    const view = rA.transpose(); // O(1) metadata
    const out = view.matmul(rB); // strided reads, no copy of A
    view.dispose();
    const result = out.toArray();
    out.dispose();
    return result;
  }

  // Bit-identity gate before timing.
  const ref = naive();
  assertBitIdentical(`series A n=${n} (materializing)`, ref, materializing());
  assertBitIdentical(`series A n=${n} (view)`, ref, viaView());

  const ts = measureAvgMs(() => void naive());
  const mat = measureAvgMs(() => void materializing());
  const view = measureAvgMs(() => void viaView());
  rowsA.push({ n, tsAvg: ts.avgMs, matAvg: mat.avgMs, viewAvg: view.avgMs, tsReps: ts.reps, matReps: mat.reps, viewReps: view.reps });

  rA.dispose();
  rB.dispose();
}

const headerA =
  `${"n".padStart(5)} | ${"naive TS".padStart(11)} | ${"materialize".padStart(11)} | ${"view".padStart(11)} | ` +
  `${"view/mat".padStart(9)} | reps (ts/mat/view)`;
console.log(headerA);
console.log("-".repeat(headerA.length));
for (const r of rowsA) {
  const ratio = r.matAvg / r.viewAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(
    `${String(r.n).padStart(5)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.matAvg).padStart(11)} | ${fmt(r.viewAvg).padStart(11)} | ` +
      `${mark.padStart(9)} | ${r.tsReps}/${r.matReps}/${r.viewReps}`,
  );
}
console.log("('view/mat' > 1x means the view path is FASTER than materializing; '<' marks it slower.)\n");

// --- Series B: contiguous ops, strided entry points vs non-strided ---------
// Raw ABI level so both paths share IDENTICAL per-rep plumbing (scratch
// marshalling, output alloc/free) and differ ONLY in the entry point.
console.log("--- Series B: routing cost — contiguous data via strided vs non-strided entry points ---");

interface Alloc {
  readonly ptr: number;
  readonly bytes: number;
}

function alloc(c: CoreExports, bytes: number): Alloc {
  const ptr = c.nt_alloc(bytes);
  if (ptr === 0 && bytes !== 0) throw new Error(`bench: nt_alloc(${bytes}) failed`);
  return { ptr, bytes };
}

function writeU32(c: CoreExports, values: readonly number[]): Alloc {
  const buf = alloc(c, values.length * 4);
  new Uint32Array(c.memory.buffer, buf.ptr, values.length).set(values);
  return buf;
}

function writeF64(c: CoreExports, values: Float64Array): Alloc {
  const buf = alloc(c, values.length * 8);
  new Float64Array(c.memory.buffer, buf.ptr, values.length).set(values);
  return buf;
}

interface RowB {
  op: string;
  n: number;
  oldAvg: number;
  newAvg: number;
  oldReps: number;
  newReps: number;
}
const rowsB: RowB[] = [];

for (const n of SIZES_B) {
  const shape = [n, n];
  const strides = computeStrides(shape);
  const len = n * n;
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);

  // Operand data resident once (not timed) — both paths read the same bytes.
  const aBuf = writeF64(core, aData);
  const bBuf = writeF64(core, bData);

  for (const op of ["add", "matmul"] as const) {
    function oldPath(): Float64Array {
      const aShapeBuf = writeU32(core, shape);
      const bShapeBuf = writeU32(core, shape);
      const outBuf = alloc(core, len * 8);
      const status =
        op === "add"
          ? core.nt_add(aShapeBuf.ptr, 2, aBuf.ptr, len, bShapeBuf.ptr, 2, bBuf.ptr, len, outBuf.ptr, len)
          : core.nt_matmul(aShapeBuf.ptr, 2, aBuf.ptr, len, bShapeBuf.ptr, 2, bBuf.ptr, len, outBuf.ptr, len);
      if (status !== 0) throw new Error(`bench old ${op}: status ${status}`);
      const result = Float64Array.from(new Float64Array(core.memory.buffer, outBuf.ptr, len));
      core.nt_free(outBuf.ptr, outBuf.bytes);
      core.nt_free(aShapeBuf.ptr, aShapeBuf.bytes);
      core.nt_free(bShapeBuf.ptr, bShapeBuf.bytes);
      return result;
    }

    function newPath(): Float64Array {
      const aShapeBuf = writeU32(core, shape);
      const aStridesBuf = writeU32(core, strides);
      const bShapeBuf = writeU32(core, shape);
      const bStridesBuf = writeU32(core, strides);
      const outBuf = alloc(core, len * 8);
      const status =
        op === "add"
          ? core.nt_add_strided(
              aShapeBuf.ptr, 2, aStridesBuf.ptr, 0, aBuf.ptr, len,
              bShapeBuf.ptr, 2, bStridesBuf.ptr, 0, bBuf.ptr, len,
              outBuf.ptr, len,
            )
          : core.nt_matmul_strided(
              aShapeBuf.ptr, 2, aStridesBuf.ptr, 0, aBuf.ptr, len,
              bShapeBuf.ptr, 2, bStridesBuf.ptr, 0, bBuf.ptr, len,
              outBuf.ptr, len,
            );
      if (status !== 0) throw new Error(`bench new ${op}: status ${status}`);
      const result = Float64Array.from(new Float64Array(core.memory.buffer, outBuf.ptr, len));
      core.nt_free(outBuf.ptr, outBuf.bytes);
      core.nt_free(aShapeBuf.ptr, aShapeBuf.bytes);
      core.nt_free(aStridesBuf.ptr, aStridesBuf.bytes);
      core.nt_free(bShapeBuf.ptr, bShapeBuf.bytes);
      core.nt_free(bStridesBuf.ptr, bStridesBuf.bytes);
      return result;
    }

    assertBitIdentical(`series B ${op} n=${n}`, oldPath(), newPath());
    const oldM = measureAvgMs(() => void oldPath());
    const newM = measureAvgMs(() => void newPath());
    rowsB.push({ op, n, oldAvg: oldM.avgMs, newAvg: newM.avgMs, oldReps: oldM.reps, newReps: newM.reps });
  }

  core.nt_free(aBuf.ptr, aBuf.bytes);
  core.nt_free(bBuf.ptr, bBuf.bytes);
}

const headerB =
  `${"op".padStart(7)} | ${"n".padStart(5)} | ${"non-strided".padStart(11)} | ${"strided".padStart(11)} | ` +
  `${"new/old".padStart(8)} | reps (old/new)`;
console.log(headerB);
console.log("-".repeat(headerB.length));
for (const r of rowsB) {
  const ratio = r.newAvg / r.oldAvg;
  console.log(
    `${r.op.padStart(7)} | ${String(r.n).padStart(5)} | ${fmt(r.oldAvg).padStart(11)} | ${fmt(r.newAvg).padStart(11)} | ` +
      `${`${ratio.toFixed(2)}x`.padStart(8)} | ${r.oldReps}/${r.newReps}`,
  );
}
console.log("('new/old' is overhead of the strided entry point on contiguous data: 1.00x = free routing.)\n");

// --- Series C: consume-once — A.transpose().sum() ---------------------------
console.log("--- Series C: A.transpose().sum() — view vs materialize-then-sum ---");
interface RowC {
  n: number;
  matAvg: number;
  viewAvg: number;
  matReps: number;
  viewReps: number;
}
const rowsC: RowC[] = [];

for (const n of SIZES_A) {
  const shape = [n, n];
  const aData = genData(rng, shape);
  const rA = WNDArray.fromArray(core, shape, aData);

  function materializing(): Float64Array {
    const view = rA.transpose();
    const mat = view.contiguous();
    view.dispose();
    const s = mat.sum();
    mat.dispose();
    const result = s.toArray();
    s.dispose();
    return result;
  }

  function viaView(): Float64Array {
    const view = rA.transpose();
    const s = view.sum(); // strided sum, logical order, no copy
    view.dispose();
    const result = s.toArray();
    s.dispose();
    return result;
  }

  // Bit-identity gate (reference: naive TS on the materialized transpose).
  const t = transposeRuntime(shape, aData);
  let refTotal = 0;
  for (let i = 0; i < t.data.length; i++) refTotal += t.data[i] ?? 0;
  const ref = Float64Array.from([refTotal]);
  assertBitIdentical(`series C n=${n} (materializing)`, ref, materializing());
  assertBitIdentical(`series C n=${n} (view)`, ref, viaView());

  const mat = measureAvgMs(() => void materializing());
  const view = measureAvgMs(() => void viaView());
  rowsC.push({ n, matAvg: mat.avgMs, viewAvg: view.avgMs, matReps: mat.reps, viewReps: view.reps });

  rA.dispose();
}

const headerC =
  `${"n".padStart(5)} | ${"materialize".padStart(11)} | ${"view".padStart(11)} | ${"view/mat".padStart(9)} | reps (mat/view)`;
console.log(headerC);
console.log("-".repeat(headerC.length));
for (const r of rowsC) {
  const ratio = r.matAvg / r.viewAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(
    `${String(r.n).padStart(5)} | ${fmt(r.matAvg).padStart(11)} | ${fmt(r.viewAvg).padStart(11)} | ${mark.padStart(9)} | ${r.matReps}/${r.viewReps}`,
  );
}
console.log("('view/mat' > 1x means the view path is FASTER than materialize-then-sum.)");
