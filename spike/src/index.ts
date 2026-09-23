export type { CompatDim, Dim, DimEq, IsDynamicDim, IsShapeError, Reverse, Shape, ShapeError, ShowShape } from "./dim.ts";
export type { Broadcast } from "./broadcast.ts";
export type { MatMul } from "./matmul.ts";
export type { ReduceAxis, ReduceAxisKeepDims, Transpose } from "./reduce.ts";
export { NDArray, type AnyNDArray, type NDArrayView, type NestedArray, type NestedValue } from "./ndarray.ts";
// Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md): the
// explicit, opt-in WASM/threads performance backends. `WasmBackend` is a
// value export (browser-safe, no top-level Node imports); `ThreadedBackend`
// stays TYPE-ONLY (its value lives behind threaded.ts's dynamic-import
// boundary, D2 — a value re-export here would statically pull in that
// module's top-level node:os/node:fs/promises/node:worker_threads imports
// and contaminate this barrel's browser-safe default path).
export { WasmBackend, type BackendKind, type ThreadedBackendOptions } from "./wasm/backend-api.ts";
export type { ThreadedBackend } from "./wasm/threaded.ts";
// D1 (docs/release-0.3.0-spec.md): `WNDArray` as a TYPE ONLY — consumers can
// name the type a backend factory (`WasmBackend.fromArray`, `ThreadedBackend`)
// returns, e.g. in a helper's parameter/return position. Its constructor stays
// private, so this adds no new way to construct one — only the backends do.
export type { WNDArray } from "./wasm/resident.ts";
