/**
 * Negative-path tests: incompatible shapes -> non-zero status from the
 * Rust kernel (tested directly against the raw ABI, `core.nt_*`) AND a
 * thrown TS `Error` naming the offending shapes (tested against the
 * `backend.ts` wrappers, which are what real callers use). Both are
 * checked for each applicable op — this is the strongest version of the
 * spec's "incompatible shapes -> status != 0 -> TS throws with the shapes
 * named" requirement: it doesn't just trust the wrapper's pre-check, it
 * also drives the kernel's own re-validation directly.
 *
 * Also covers the two status codes normal TS-side usage structurally
 * cannot reach (rank > 32, size overflow) by calling the raw ABI directly
 * with deliberately invalid shapes.
 */
import assert from "node:assert";
import { test } from "node:test";
import { wasmAdd, wasmMatmul, wasmSum } from "../src/wasm/backend.ts";
import { initCore, type CoreExports } from "../src/wasm/loader.ts";

const core = await initCore();

const STATUS_OK = 0;
const STATUS_SHAPE_INCOMPATIBLE = 1;
const STATUS_RANK_TOO_LARGE = 2;
const STATUS_SIZE_OVERFLOW = 3;

interface ScratchBuf {
  ptr: number;
  bytes: number;
}
function allocU32(c: CoreExports, values: readonly number[]): ScratchBuf {
  const bytes = values.length * 4;
  const ptr = c.nt_alloc(bytes);
  new Uint32Array(c.memory.buffer, ptr, values.length).set(values);
  return { ptr, bytes };
}
function allocF64(c: CoreExports, values: readonly number[]): ScratchBuf {
  const bytes = values.length * 8;
  const ptr = c.nt_alloc(bytes);
  new Float64Array(c.memory.buffer, ptr, values.length).set(values);
  return { ptr, bytes };
}
function allocOut(c: CoreExports, elems: number): ScratchBuf {
  const bytes = elems * 8;
  const ptr = c.nt_alloc(bytes);
  return { ptr, bytes };
}
function free(c: CoreExports, ...bufs: ScratchBuf[]): void {
  for (const b of bufs) c.nt_free(b.ptr, b.bytes);
}

// --- add: shape-incompatible (status 1) ---------------------------------
test("add: raw nt_add returns status 1 for incompatible shapes", () => {
  const aShape = allocU32(core, [2, 3]);
  const bShape = allocU32(core, [2, 4]);
  const aData = allocF64(core, new Array(6).fill(0));
  const bData = allocF64(core, new Array(8).fill(0));
  const out = allocOut(core, 8);
  const status = core.nt_add(aShape.ptr, 2, aData.ptr, 6, bShape.ptr, 2, bData.ptr, 8, out.ptr, 8);
  assert.strictEqual(status, STATUS_SHAPE_INCOMPATIBLE);
  free(core, aShape, bShape, aData, bData, out);
});

test("add: wasmAdd throws naming both shapes for incompatible add", () => {
  assert.throws(
    () => wasmAdd(core, [2, 3], new Float64Array(6), [2, 4], new Float64Array(8)),
    (err: unknown) => {
      if (!(err instanceof Error)) return false;
      assert.ok(err.message.includes("2,3"), `message should name shape [2,3]: ${err.message}`);
      assert.ok(err.message.includes("2,4"), `message should name shape [2,4]: ${err.message}`);
      return true;
    },
  );
});

// --- matmul: inner-dim mismatch (status 1) ------------------------------
test("matmul: raw nt_matmul returns status 1 for inner-dim mismatch", () => {
  const aShape = allocU32(core, [2, 3]);
  const bShape = allocU32(core, [4, 2]);
  const aData = allocF64(core, new Array(6).fill(0));
  const bData = allocF64(core, new Array(8).fill(0));
  const out = allocOut(core, 4);
  const status = core.nt_matmul(aShape.ptr, 2, aData.ptr, 6, bShape.ptr, 2, bData.ptr, 8, out.ptr, 4);
  assert.strictEqual(status, STATUS_SHAPE_INCOMPATIBLE);
  free(core, aShape, bShape, aData, bData, out);
});

test("matmul: wasmMatmul throws naming the mismatched inner dimensions", () => {
  assert.throws(
    () => wasmMatmul(core, [2, 3], new Float64Array(6), [4, 2], new Float64Array(8)),
    /inner dimensions 3 and 4/,
  );
});

test("matmul: wasmMatmul rejects a rank-0 scalar operand (TS-side, before any WASM call)", () => {
  assert.throws(() => wasmMatmul(core, [], new Float64Array([1]), [2, 2], new Float64Array(4)), /scalar operand/);
  assert.throws(() => wasmMatmul(core, [2, 2], new Float64Array(4), [], new Float64Array([1])), /scalar operand/);
});

// --- sum: axis out of range (status 1) ----------------------------------
test("sum: raw nt_sum_axis returns status 1 for an out-of-range axis", () => {
  const shape = allocU32(core, [2, 3]);
  const data = allocF64(core, new Array(6).fill(0));
  const out = allocOut(core, 3);
  const status = core.nt_sum_axis(shape.ptr, 2, data.ptr, 6, 5, out.ptr, 3);
  assert.strictEqual(status, STATUS_SHAPE_INCOMPATIBLE);
  free(core, shape, data, out);
});

test("sum: wasmSum throws naming the axis and shape for an out-of-range axis", () => {
  assert.throws(
    () => wasmSum(core, [2, 3], new Float64Array(6), 5),
    (err: unknown) => {
      if (!(err instanceof Error)) return false;
      assert.ok(err.message.includes("5"), `message should name axis 5: ${err.message}`);
      assert.ok(err.message.includes("2,3"), `message should name shape [2,3]: ${err.message}`);
      return true;
    },
  );
});

// --- rank > 32 (status 2) — unreachable through normal TS usage, so
// tested directly against the raw ABI. ------------------------------------
test("raw ABI: rank > 32 returns status 2", () => {
  const bigShape = allocU32(core, new Array(33).fill(1));
  const smallShape = allocU32(core, [1]);
  const aData = allocF64(core, [1]);
  const bData = allocF64(core, [1]);
  const out = allocOut(core, 1);
  const status = core.nt_add(bigShape.ptr, 33, aData.ptr, 1, smallShape.ptr, 1, bData.ptr, 1, out.ptr, 1);
  assert.strictEqual(status, STATUS_RANK_TOO_LARGE);
  free(core, bigShape, smallShape, aData, bData, out);
});

test("raw ABI: rank exactly 32 is accepted (boundary check)", () => {
  const shape32 = allocU32(core, new Array(32).fill(1));
  const shape1 = allocU32(core, [1]);
  const aData = allocF64(core, [1]);
  const bData = allocF64(core, [1]);
  const out = allocOut(core, 1);
  const status = core.nt_add(shape32.ptr, 32, aData.ptr, 1, shape1.ptr, 1, bData.ptr, 1, out.ptr, 1);
  assert.strictEqual(status, STATUS_OK);
  free(core, shape32, shape1, aData, bData, out);
});

// --- size overflow (status 3) — a shape whose element count overflows
// u32, tested with tiny (mismatched) real backing buffers: the kernel's
// `checked_element_count` check on the *declared shape* fires before any
// data is ever touched, so this is safe even though the declared shape's
// true product is never actually allocated (~4.9e9 elements). -------------
test("raw ABI: shape whose product overflows u32 returns status 3", () => {
  const hugeShape = allocU32(core, [70_000, 70_000]); // product ~4.9e9 > u32::MAX
  const smallShape = allocU32(core, [1]);
  const aData = allocF64(core, [1]); // deliberately NOT sized to the huge shape's product
  const bData = allocF64(core, [1]);
  const out = allocOut(core, 1);
  const status = core.nt_add(hugeShape.ptr, 2, aData.ptr, 1, smallShape.ptr, 1, bData.ptr, 1, out.ptr, 1);
  assert.strictEqual(status, STATUS_SIZE_OVERFLOW);
  free(core, hugeShape, smallShape, aData, bData, out);
});

// --- out_len mismatch also reported as size overflow (status 3): proves
// the kernel bounds its writes against the caller-declared output length
// rather than trusting its own computation blindly. ------------------------
test("raw ABI: out_len not matching the kernel's own computed output length returns status 3", () => {
  const aShape = allocU32(core, [2, 3]);
  const bShape = allocU32(core, [3]);
  const aData = allocF64(core, [1, 2, 3, 4, 5, 6]);
  const bData = allocF64(core, [10, 20, 30]);
  const out = allocOut(core, 4); // correct out_len would be 6, not 4
  const status = core.nt_add(aShape.ptr, 2, aData.ptr, 6, bShape.ptr, 1, bData.ptr, 3, out.ptr, 4);
  assert.strictEqual(status, STATUS_SIZE_OVERFLOW);
  free(core, aShape, bShape, aData, bData, out);
});
