/**
 * Kern 04 bench — run via `pnpm bench:blocked`; numbers recorded in
 * docs/kern-04-ergebnisse.md. Same discipline as the other benches: seeded
 * inputs, bit-identity gate BEFORE any timing, adaptive batch-timed reps,
 * warmed JIT, ranges over sizes reported rather than single points.
 *
 * `WNDArray.matmul()` now calls `nt_matmul_blocked` unconditionally (Kern
 * 04), so there is no more class-level way to reach the old
 * `nt_matmul_strided`/`nt_matmul` paths for a same-machinery comparison —
 * this bench talks to those two entry points at the raw ABI level (alloc +
 * marshal + call + free, mirroring exactly what resident.ts/backend.ts do
 * per call) so all four columns in Series A share comparable per-rep
 * plumbing, differing only in which kernel actually runs the multiply-add.
 *
 * Series A (headline): contiguous `[n,n] @ [n,n]`, four ways — naive TS /
 * v1 (`nt_matmul`, copy-based) / Kern-03 strided scalar (`nt_matmul_strided`
 * on contiguous metadata) / Kern-04 blocked+SIMD (`nt_matmul_blocked`).
 *
 * Series B (the prediction on record, Kern 03 docs/kern-03-ergebnisse.md):
 * does packing erase the view penalty Kern 03 measured (~30% slower at
 * n>=256 for a transposed operand feeding matmul)? `A.transpose().matmul(B)`
 * (view path, through the SAME blocked kernel now) vs
 * `A.transpose().contiguous().matmul(B)` (materializing first). Both go
 * through `nt_matmul_blocked` — the kernel packs either way, so the
 * question is whether the view's strided packing-gather is measurably
 * slower than reading an already-contiguous operand's packing-gather.
 *
 * Series C (honesty at the small end): n in {4,8,16,32} — blocked vs the
 * Kern-03 scalar-strided kernel on identical contiguous data at the raw ABI
 * level. Packing/blocking overhead is not free; if it isn't amortized at
 * these sizes, that regression is the finding, not something to hide.
 */
import { matmulRuntime } from "../src/runtime.ts";
import { initCore, type CoreExports } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

const SIZES_A = [64, 128, 256, 512, 1024];
const SIZES_B = [128, 256, 512];
const SIZES_C = [4, 8, 16, 32];
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

console.log("=== NumType Kern 04 bench: blocked + packed + SIMD128 matmul ===\n");
console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

const rng = makeRng(0x424c4f434b45445fn); // "BLOCKED_"-ish

// --- raw ABI helpers (mirrors bench-core/strided.ts Series B exactly) ------
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
function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array(shape.length).fill(0);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] ?? 1;
  }
  return strides;
}

/** v1 path: `nt_matmul` (contiguous-only ABI, no strides argument). */
function rawV1(core: CoreExports, aBuf: Alloc, bBuf: Alloc, shape: readonly number[], len: number): Float64Array {
  const aShapeBuf = writeU32(core, shape);
  const bShapeBuf = writeU32(core, shape);
  const outBuf = alloc(core, len * 8);
  const status = core.nt_matmul(aShapeBuf.ptr, 2, aBuf.ptr, len, bShapeBuf.ptr, 2, bBuf.ptr, len, outBuf.ptr, len);
  if (status !== 0) throw new Error(`bench v1 matmul: status ${status}`);
  const result = Float64Array.from(new Float64Array(core.memory.buffer, outBuf.ptr, len));
  core.nt_free(outBuf.ptr, outBuf.bytes);
  core.nt_free(aShapeBuf.ptr, aShapeBuf.bytes);
  core.nt_free(bShapeBuf.ptr, bShapeBuf.bytes);
  return result;
}

/** Kern-03 strided-scalar path on contiguous metadata, or Kern-04
 * blocked+SIMD path — same quadruple convention, selectable by entry point. */
function rawStridedOrBlocked(
  core: CoreExports,
  entry: "strided" | "blocked",
  aBuf: Alloc,
  bBuf: Alloc,
  shape: readonly number[],
  len: number,
): Float64Array {
  const strides = computeStrides(shape);
  const aShapeBuf = writeU32(core, shape);
  const aStridesBuf = writeU32(core, strides);
  const bShapeBuf = writeU32(core, shape);
  const bStridesBuf = writeU32(core, strides);
  const outBuf = alloc(core, len * 8);
  const fn = entry === "strided" ? core.nt_matmul_strided : core.nt_matmul_blocked;
  const status = fn(
    aShapeBuf.ptr, 2, aStridesBuf.ptr, 0, aBuf.ptr, len,
    bShapeBuf.ptr, 2, bStridesBuf.ptr, 0, bBuf.ptr, len,
    outBuf.ptr, len,
  );
  if (status !== 0) throw new Error(`bench ${entry} matmul: status ${status}`);
  const result = Float64Array.from(new Float64Array(core.memory.buffer, outBuf.ptr, len));
  core.nt_free(outBuf.ptr, outBuf.bytes);
  core.nt_free(aShapeBuf.ptr, aShapeBuf.bytes);
  core.nt_free(aStridesBuf.ptr, aStridesBuf.bytes);
  core.nt_free(bShapeBuf.ptr, bShapeBuf.bytes);
  core.nt_free(bStridesBuf.ptr, bStridesBuf.bytes);
  return result;
}

// --- Series A: naive TS / v1 / Kern-03 strided-scalar / Kern-04 blocked ----
console.log("--- Series A: [n,n] @ [n,n], four ways ---");
interface RowA {
  n: number;
  tsAvg: number;
  v1Avg: number;
  stridedAvg: number;
  blockedAvg: number;
  reps: string;
}
const rowsA: RowA[] = [];

for (const n of SIZES_A) {
  const shape = [n, n];
  const len = n * n;
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);

  const aBuf = writeF64(core, aData);
  const bBuf = writeF64(core, bData);

  const ref = matmulRuntime(shape, aData, shape, bData).data;
  assertBitIdentical(`series A n=${n} (v1)`, ref, rawV1(core, aBuf, bBuf, shape, len));
  assertBitIdentical(`series A n=${n} (strided-scalar)`, ref, rawStridedOrBlocked(core, "strided", aBuf, bBuf, shape, len));
  assertBitIdentical(`series A n=${n} (blocked)`, ref, rawStridedOrBlocked(core, "blocked", aBuf, bBuf, shape, len));

  const ts = measureAvgMs(() => void matmulRuntime(shape, aData, shape, bData));
  const v1 = measureAvgMs(() => void rawV1(core, aBuf, bBuf, shape, len));
  const strided = measureAvgMs(() => void rawStridedOrBlocked(core, "strided", aBuf, bBuf, shape, len));
  const blocked = measureAvgMs(() => void rawStridedOrBlocked(core, "blocked", aBuf, bBuf, shape, len));

  rowsA.push({
    n,
    tsAvg: ts.avgMs,
    v1Avg: v1.avgMs,
    stridedAvg: strided.avgMs,
    blockedAvg: blocked.avgMs,
    reps: `${ts.reps}/${v1.reps}/${strided.reps}/${blocked.reps}`,
  });

  core.nt_free(aBuf.ptr, aBuf.bytes);
  core.nt_free(bBuf.ptr, bBuf.bytes);
}

const headerA =
  `${"n".padStart(6)} | ${"naive TS".padStart(11)} | ${"v1 (copy)".padStart(11)} | ${"strided-scalar".padStart(14)} | ` +
  `${"blocked+SIMD".padStart(12)} | ${"blk/strided".padStart(11)} | reps (ts/v1/str/blk)`;
console.log(headerA);
console.log("-".repeat(headerA.length));
for (const r of rowsA) {
  const speedup = r.stridedAvg / r.blockedAvg;
  console.log(
    `${String(r.n).padStart(6)} | ${fmt(r.tsAvg).padStart(11)} | ${fmt(r.v1Avg).padStart(11)} | ${fmt(r.stridedAvg).padStart(14)} | ` +
      `${fmt(r.blockedAvg).padStart(12)} | ${`${speedup.toFixed(2)}x`.padStart(11)} | ${r.reps}`,
  );
}
console.log("('blk/strided' > 1x means blocked+SIMD is FASTER than the Kern-03 scalar strided kernel.)\n");

// --- Series B: does packing erase the Kern-03 view penalty? ----------------
console.log("--- Series B: A.transpose() @ B — view vs materializing, both through the blocked kernel ---");
interface RowB {
  n: number;
  matAvg: number;
  viewAvg: number;
}
const rowsB: RowB[] = [];

for (const n of SIZES_B) {
  const shape = [n, n];
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);

  const rA = WNDArray.fromArray(core, shape, aData);
  const rB = WNDArray.fromArray(core, shape, bData);

  function materializing(): Float64Array {
    const view = rA.transpose();
    const mat = view.contiguous();
    view.dispose();
    const out = mat.matmul(rB);
    mat.dispose();
    const result = out.toArray();
    out.dispose();
    return result;
  }
  function viaView(): Float64Array {
    const view = rA.transpose();
    const out = view.matmul(rB);
    view.dispose();
    const result = out.toArray();
    out.dispose();
    return result;
  }

  // Reference: naive TS transpose then matmul (same as strided.ts Series A).
  const refData = (() => {
    const t = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) t[j * n + i] = aData[i * n + j] ?? 0;
    return matmulRuntime([n, n], t, [n, n], bData).data;
  })();

  assertBitIdentical(`series B n=${n} (materializing)`, refData, materializing());
  assertBitIdentical(`series B n=${n} (view)`, refData, viaView());

  const mat = measureAvgMs(() => void materializing());
  const view = measureAvgMs(() => void viaView());
  rowsB.push({ n, matAvg: mat.avgMs, viewAvg: view.avgMs });

  rA.dispose();
  rB.dispose();
}

const headerB = `${"n".padStart(6)} | ${"materialize".padStart(11)} | ${"view".padStart(11)} | ${"view/mat".padStart(9)}`;
console.log(headerB);
console.log("-".repeat(headerB.length));
for (const r of rowsB) {
  const ratio = r.matAvg / r.viewAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(`${String(r.n).padStart(6)} | ${fmt(r.matAvg).padStart(11)} | ${fmt(r.viewAvg).padStart(11)} | ${mark.padStart(9)}`);
}
console.log("('view/mat' > 1x means the view path is FASTER than materializing first; ~1x means packing erased the penalty.)\n");

// --- Series C: small-size honesty — blocked vs Kern-03 scalar strided ------
console.log("--- Series C: small n — blocked+SIMD vs Kern-03 scalar strided kernel ---");
interface RowC {
  n: number;
  stridedAvg: number;
  blockedAvg: number;
}
const rowsC: RowC[] = [];

for (const n of SIZES_C) {
  const shape = [n, n];
  const len = n * n;
  const aData = genData(rng, shape);
  const bData = genData(rng, shape);
  const aBuf = writeF64(core, aData);
  const bBuf = writeF64(core, bData);

  const ref = matmulRuntime(shape, aData, shape, bData).data;
  assertBitIdentical(`series C n=${n} (strided)`, ref, rawStridedOrBlocked(core, "strided", aBuf, bBuf, shape, len));
  assertBitIdentical(`series C n=${n} (blocked)`, ref, rawStridedOrBlocked(core, "blocked", aBuf, bBuf, shape, len));

  const strided = measureAvgMs(() => void rawStridedOrBlocked(core, "strided", aBuf, bBuf, shape, len));
  const blocked = measureAvgMs(() => void rawStridedOrBlocked(core, "blocked", aBuf, bBuf, shape, len));
  rowsC.push({ n, stridedAvg: strided.avgMs, blockedAvg: blocked.avgMs });

  core.nt_free(aBuf.ptr, aBuf.bytes);
  core.nt_free(bBuf.ptr, bBuf.bytes);
}

const headerC = `${"n".padStart(6)} | ${"strided-scalar".padStart(14)} | ${"blocked+SIMD".padStart(12)} | ${"blk/strided".padStart(11)}`;
console.log(headerC);
console.log("-".repeat(headerC.length));
for (const r of rowsC) {
  const ratio = r.stridedAvg / r.blockedAvg;
  const mark = ratio >= 1 ? `${ratio.toFixed(2)}x` : `${ratio.toFixed(2)}x <`;
  console.log(`${String(r.n).padStart(6)} | ${fmt(r.stridedAvg).padStart(14)} | ${fmt(r.blockedAvg).padStart(12)} | ${mark.padStart(11)}`);
}
console.log("('blk/strided' < 1x (marked '<') means blocking/packing overhead is NOT amortized at this size — an honest regression, not hidden.)");
