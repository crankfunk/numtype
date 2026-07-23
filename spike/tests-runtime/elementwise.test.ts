/**
 * Differential tests (Kern 07): broadcasting elementwise sub/mul/div.
 * Resident WASM (`WNDArray.sub`/`mul`/`div`) must be bit-identical to the
 * naive TS reference (`elementwiseBinary` with the op's pinned closure) —
 * both for contiguous operand pairs and for strided-view operands
 * (transposed views; reuses the `strided.test.ts` view-construction idiom:
 * a base of the reversed shape holding the transposed data, then
 * `.transpose()`'d, so the view reads back the intended logical content —
 * that involution is already pinned by `strided.test.ts`'s own round-trip
 * group, not re-derived here).
 *
 * Plus the spec's named `div` IEEE fixtures on BOTH surfaces (targeted, not
 * systematic special-value injection — that is a separate, later phase per
 * the spec's "documented differential blind spots").
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { elementwiseBinary, sqrtRuntime, transposeRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, makeRng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 120;
const STRIDED_CASE_COUNT = 30;

type Op = "sub" | "mul" | "div";
const OPS: readonly Op[] = ["sub", "mul", "div"];

function refOp(op: Op): (x: number, y: number) => number {
  if (op === "sub") return (x, y) => x - y;
  if (op === "mul") return (x, y) => x * y;
  return (x, y) => x / y;
}

function callResident(op: Op, a: AnyWNDArray, b: AnyWNDArray): AnyWNDArray {
  if (op === "sub") return a.sub(b);
  if (op === "mul") return a.mul(b);
  return a.div(b);
}

// One fixed seed per op (own stream, ASCII-ish, matches the codebase's
// existing seed-literal style).
const BROADCAST_SEED: Record<Op, bigint> = {
  sub: 0x5355425f4b4e3037n, // "SUB_KN07"
  mul: 0x4d554c5f4b4e3037n, // "MUL_KN07"
  div: 0x4449565f4b4e3037n, // "DIV_KN07"
};
const STRIDED_SEED: Record<Op, bigint> = {
  sub: 0x5355425f5354524cn, // "SUB_STRL"-ish
  mul: 0x4d554c5f5354524cn, // "MUL_STRL"-ish
  div: 0x4449565f5354524cn, // "DIV_STRL"-ish
};

// --- contiguous operand pairs: >=100 seeded cases per op --------------------
for (const op of OPS) {
  const rng = makeRng(BROADCAST_SEED[op]);
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);

    test(`${op} case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, refOp(op));
      const a = WNDArray.fromArray(core, aShape, Array.from(aData));
      const b = WNDArray.fromArray(core, bShape, Array.from(bData));
      try {
        const got = callResident(op, a, b);
        try {
          const ctx = `${op} case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
          // D-V2.3 fallout: `got: AnyWNDArray = WNDArray<any>`, so `.shape` is
          // `Readonly<any>` — doesn't structurally collapse to `any` (TS
          // quirk), so no longer matches `readonly number[]` (TS2740). Cast
          // only; runtime value unaffected (see demo.ts's `assertResidentAgrees`
          // for the full explanation).
          assertShapeEqual(ref.shape, got.shape as readonly number[], ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// --- strided-view operand pairs: >=20 seeded cases per op --------------------
interface Operand {
  readonly arr: AnyWNDArray;
  /** Every handle that must be disposed when the case is done (for a view:
   * the base AND the view). */
  readonly owners: readonly AnyWNDArray[];
}

function makeContiguous(shape: readonly number[], refData: Float64Array): Operand {
  const arr = WNDArray.fromArray(core, shape, refData);
  return { arr, owners: [arr] };
}

/** A transpose VIEW whose logical shape is `shape` and logical content is
 * `refData` — same construction as `strided.test.ts`'s `makeView`. */
function makeView(shape: readonly number[], refData: Float64Array): Operand {
  const baseShape = [...shape].reverse();
  const baseData = transposeRuntime(shape, refData).data;
  const base = WNDArray.fromArray(core, baseShape, baseData);
  const view = base.transpose();
  return { arr: view, owners: [base, view] };
}

function makeOperand(asView: boolean, shape: readonly number[], refData: Float64Array): Operand {
  return asView ? makeView(shape, refData) : makeContiguous(shape, refData);
}

function disposeAll(...operands: Operand[]): void {
  for (const op of operands) for (const h of op.owners) h.dispose();
}

for (const op of OPS) {
  const rng = makeRng(STRIDED_SEED[op]);
  for (let c = 0; c < STRIDED_CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    // At least one operand is a view (both-contiguous is the group above).
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }

    test(`${op} strided-view case ${c}: a=[${aShape.join(",")}]${aView ? "ᵛ" : ""} b=[${bShape.join(",")}]${bView ? "ᵛ" : ""}`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, refOp(op));
      const a = makeOperand(aView, aShape, aData);
      const b = makeOperand(bView, bShape, bData);
      try {
        const got = callResident(op, a.arr, b.arr);
        try {
          const ctx = `${op} strided-view case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}] aView=${aView} bView=${bView}`;
          // D-V2.3 fallout: `got: AnyWNDArray = WNDArray<any>`, so `.shape` is
          // `Readonly<any>` — doesn't structurally collapse to `any` (TS
          // quirk), so no longer matches `readonly number[]` (TS2740). Cast
          // only; runtime value unaffected (see demo.ts's `assertResidentAgrees`
          // for the full explanation).
          assertShapeEqual(ref.shape, got.shape as readonly number[], ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(a, b);
      }
    });
  }
}

// --- div: pinned IEEE fixtures, both surfaces --------------------------------

test("div fixture: [1,-1,0,42] / [0,0,0,2] -> [+Inf,-Inf,NaN,21] on the naive reference", () => {
  const ref = elementwiseBinary([4], new Float64Array([1, -1, 0, 42]), [4], new Float64Array([0, 0, 0, 2]), (x, y) => x / y);
  assert.strictEqual(ref.data[0], Infinity);
  assert.strictEqual(ref.data[1], -Infinity);
  assert.ok(Number.isNaN(ref.data[2]));
  assert.strictEqual(ref.data[3], 21);
});

test("div fixture: [1,-1,0,42] / [0,0,0,2] -> [+Inf,-Inf,NaN,21] on the resident backend", () => {
  const a = WNDArray.fromArray(core, [4], [1, -1, 0, 42]);
  const b = WNDArray.fromArray(core, [4], [0, 0, 0, 2]);
  try {
    const got = a.div(b);
    try {
      const arr = got.toArray();
      assert.strictEqual(arr[0], Infinity);
      assert.strictEqual(arr[1], -Infinity);
      assert.ok(Number.isNaN(arr[2]));
      assert.strictEqual(arr[3], 21);
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("div fixture: 0 / -2 -> -0 (signed zero, Object.is-distinguished) on the naive reference", () => {
  const ref = elementwiseBinary([1], new Float64Array([0]), [1], new Float64Array([-2]), (x, y) => x / y);
  assert.ok(Object.is(ref.data[0], -0));
  assert.ok(!Object.is(ref.data[0], 0));
});

test("div fixture: 0 / -2 -> -0 (signed zero, Object.is-distinguished) on the resident backend", () => {
  const a = WNDArray.fromArray(core, [1], [0]);
  const b = WNDArray.fromArray(core, [1], [-2]);
  try {
    const got = a.div(b);
    try {
      const v = got.toArray()[0] ?? Number.NaN;
      assert.ok(Object.is(v, -0));
      assert.ok(!Object.is(v, 0));
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

// =============================================================================
// WASM parity S0 (docs/wasm-parity-sqrt-spec.md, D6): `WNDArray.sqrt()` M1
// differential against the naive reference (`sqrtRuntime`/`NDArray.sqrt`) —
// contiguous, a transposed view, a sliced view, an offset window, rank-0, and
// a size-0 dim. View references are the view's own already-trusted logical
// content (transpose via this file's own `makeView`/`transposeRuntime`
// idiom above; slice via `NDArray.slice`, the naive backend's own
// `sliceRuntime`-driven implementation — same "independent oracle" technique
// keepdims.test.ts already uses for its offset-slice-view cases), never a
// second read-back through the op under test.
// =============================================================================

test("sqrt: contiguous [2,3]", () => {
  const shape = [2, 3];
  const data = new Float64Array([1, 4, 9, 16, 25, 36]);
  const ref = sqrtRuntime(data);
  const w = WNDArray.fromArray(core, shape, data);
  try {
    const got = w.sqrt();
    try {
      assertShapeEqual(shape, got.shape as readonly number[], "sqrt contiguous");
      assertDataBitIdentical(ref, got.toArray(), "sqrt contiguous");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("sqrt: transposed view (non-natural strides)", () => {
  const shape = [3, 2];
  const refData = new Float64Array([1, 4, 9, 16, 25, 36]); // logical content of the view, by makeView's own construction
  const { arr: view, owners } = makeView(shape, refData);
  try {
    const ref = sqrtRuntime(refData);
    const got = view.sqrt();
    try {
      assertShapeEqual(shape, got.shape as readonly number[], "sqrt transposed view");
      assertDataBitIdentical(ref, got.toArray(), "sqrt transposed view");
    } finally {
      got.dispose();
    }
  } finally {
    for (const h of owners) h.dispose();
  }
});

test("sqrt: sliced view (step slice, non-contiguous strides, offset 0)", () => {
  // [4,3] -> rows {0,2} via step 2 -> shape [2,3], strides [6,1] (non-natural).
  const baseShape = [4, 3];
  const baseData = Array.from({ length: 12 }, (_, i) => (i + 1) * (i + 1));
  const refView = NDArray.fromArray(baseShape, baseData).slice({ step: 2 }, null); // trusted naive slice
  const ref = refView.sqrt();

  const w = WNDArray.fromArray(core, baseShape, baseData);
  try {
    const view = w.slice({ step: 2 }, null); // O(1) view, non-natural strides
    try {
      const got = view.sqrt();
      try {
        assertShapeEqual(ref.shape, got.shape as readonly number[], "sqrt sliced view");
        assertDataBitIdentical(ref.data, got.toArray(), "sqrt sliced view");
      } finally {
        got.dispose();
      }
    } finally {
      view.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("sqrt: offset window (1-D, nonzero base offset, natural stride)", () => {
  // [6] -> rows 2.. -> shape [4], offset 2, stride 1 (natural, just shifted).
  const baseShape = [6];
  const baseData = [0, 1, 4, 9, 16, 25];
  const refView = NDArray.fromArray(baseShape, baseData).slice({ start: 2 }); // trusted naive slice
  const ref = refView.sqrt();

  const w = WNDArray.fromArray(core, baseShape, baseData);
  try {
    const view = w.slice({ start: 2 }); // O(1) view, offset 2
    try {
      const got = view.sqrt();
      try {
        assertShapeEqual(ref.shape, got.shape as readonly number[], "sqrt offset window");
        assertDataBitIdentical(ref.data, got.toArray(), "sqrt offset window");
      } finally {
        got.dispose();
      }
    } finally {
      view.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("sqrt: rank-0 scalar", () => {
  const shape: readonly number[] = [];
  const data = new Float64Array([25]);
  const ref = sqrtRuntime(data);
  const w = WNDArray.fromArray(core, shape, data);
  try {
    const got = w.sqrt();
    try {
      assertShapeEqual(shape, got.shape as readonly number[], "sqrt rank-0");
      assertDataBitIdentical(ref, got.toArray(), "sqrt rank-0");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("sqrt: size-0 dim [0,3]", () => {
  const shape = [0, 3];
  const data = new Float64Array([]);
  const ref = sqrtRuntime(data);
  const w = WNDArray.fromArray(core, shape, data);
  try {
    const got = w.sqrt();
    try {
      assertShapeEqual(shape, got.shape as readonly number[], "sqrt size-0 dim");
      assertDataBitIdentical(ref, got.toArray(), "sqrt size-0 dim");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});
