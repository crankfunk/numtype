/**
 * Dual-target, hand-written WASM loader — no wasm-bindgen/wasm-pack glue.
 * Node: `fs/promises.readFile` + `WebAssembly.instantiate`. Browser:
 * `fetch` + `WebAssembly.instantiateStreaming`. Feature-detected via
 * `typeof process` (per spec) rather than any bundler-specific mechanism.
 *
 * The crate exports no imports (no `env` object needed): every `nt_*`
 * function is self-contained, and `memory` is auto-exported by
 * `wasm32-unknown-unknown` with no linker flags (verified empirically —
 * see docs/kern-01-ergebnisse.md).
 */

/** Raw WASM exports — the ABI documented in crates/core/src/abi.rs. Every
 * pointer/length is a plain `number` (WASM32 addresses/lengths are 32-bit,
 * losslessly representable as JS numbers). */
export interface CoreExports {
  readonly memory: WebAssembly.Memory;
  nt_alloc(bytes: number): number;
  nt_free(ptr: number, bytes: number): void;
  nt_add(
    aShapePtr: number,
    aRank: number,
    aDataPtr: number,
    aLen: number,
    bShapePtr: number,
    bRank: number,
    bDataPtr: number,
    bLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_matmul(
    aShapePtr: number,
    aRank: number,
    aDataPtr: number,
    aLen: number,
    bShapePtr: number,
    bRank: number,
    bDataPtr: number,
    bLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_sum_all(dataPtr: number, len: number, outDataPtr: number): number;
  nt_sum_axis(
    shapePtr: number,
    rank: number,
    dataPtr: number,
    len: number,
    axis: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_transpose(shapePtr: number, rank: number, dataPtr: number, len: number, outDataPtr: number, outLen: number): number;
  nt_fill(outDataPtr: number, outLen: number, value: number): number;
  // --- Kern 03: strided entry points. A strided operand is the quadruple
  // (shape ptr/rank, strides ptr, element offset, data ptr/FULL-buffer len);
  // see the "Strided entry points" section of crates/core/src/abi.rs.
  nt_add_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_matmul_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_sum_all_strided(
    shapePtr: number,
    rank: number,
    stridesPtr: number,
    offset: number,
    dataPtr: number,
    dataLen: number,
    outDataPtr: number,
  ): number;
  nt_sum_axis_strided(
    shapePtr: number,
    rank: number,
    stridesPtr: number,
    offset: number,
    dataPtr: number,
    dataLen: number,
    axis: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_materialize(
    shapePtr: number,
    rank: number,
    stridesPtr: number,
    offset: number,
    dataPtr: number,
    dataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  // --- Kern 04: blocked + packed + SIMD128 matmul. Identical quadruple
  // convention to nt_matmul_strided (see abi.rs); drop-in alternative entry
  // point for matmul only.
  nt_matmul_blocked(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
}

const WASM_URL = new URL("./numtype_core.wasm", import.meta.url);

async function loadModuleBytesNode(): Promise<ArrayBuffer> {
  const { readFile } = await import("node:fs/promises");
  // `.slice()`: Node's returned Uint8Array may be a view into a pooled,
  // larger underlying buffer; copy so we hand WebAssembly exactly the
  // module's own bytes in their own (non-shared) ArrayBuffer — `readFile`
  // never actually backs its result with a `SharedArrayBuffer`, so this
  // cast is sound even though the ambient type (kept deliberately minimal;
  // see spike/src/ambient.d.ts) doesn't distinguish the two.
  const bytes = (await readFile(WASM_URL)).slice();
  return bytes.buffer as ArrayBuffer;
}

async function instantiate(): Promise<WebAssembly.Instance> {
  const isNode = typeof process !== "undefined";
  if (!isNode && typeof WebAssembly.instantiateStreaming === "function") {
    const { instance } = await WebAssembly.instantiateStreaming(fetch(WASM_URL), {});
    return instance;
  }
  const moduleBytes: ArrayBuffer = isNode ? await loadModuleBytesNode() : await (await fetch(WASM_URL)).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(moduleBytes, {});
  return instance;
}

/** Instantiate the core WASM module and return its typed exports. Call
 * once per program (or per test file); the returned `Core` is cheap to
 * reuse across many kernel calls. */
export async function initCore(): Promise<CoreExports> {
  const instance = await instantiate();
  return instance.exports as unknown as CoreExports;
}

// --- Kern 07 (docs/kern-07-elementwise-vector-spec.md): elementwise
// sub/mul/div + dot/norm_sq reductions. Declared as a SEPARATE
// `export interface CoreExports { ... }` block, appended strictly after
// every pre-existing line in this file (freeze discipline) — TypeScript
// merges same-named interface declarations in the same module
// automatically, so this augments the identical `CoreExports` type without
// touching a single pre-existing line.
export interface CoreExports {
  // sub/mul/div: identical 14-argument convention to `nt_add_strided`.
  nt_sub_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_mul_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  nt_div_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number;
  // dot: two strided-operand quadruples + outDataPtr, output implicitly 1
  // f64 (no outLen), matching `nt_sum_all_strided`'s convention.
  nt_dot_strided(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
  ): number;
  // norm_sq: one strided-operand quadruple + outDataPtr, output implicitly
  // 1 f64 — identical convention to `nt_sum_all_strided`.
  nt_norm_sq_strided(
    shapePtr: number,
    rank: number,
    stridesPtr: number,
    offset: number,
    dataPtr: number,
    dataLen: number,
    outDataPtr: number,
  ): number;
}
