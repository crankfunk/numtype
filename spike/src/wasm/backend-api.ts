/**
 * Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md): the WASM
 * facade half of the public backend surface. `NDArray.backend(kind, opts?)`
 * (ndarray.ts) is the discoverable entry point; this module holds the
 * `WasmBackend` class it returns for `kind === "wasm"`, plus the shared
 * `BackendKind`/`ThreadedBackendOptions` types and the Node/artifact env
 * check `backend("threaded")` runs before dynamically importing
 * `./threaded.ts` (D2).
 *
 * Deliberately free of any TOP-LEVEL Node-specific import (only a guarded
 * DYNAMIC `import("node:fs/promises")` inside `checkThreadedEnv`, reached
 * only when already known to be running under Node) — this file is
 * statically imported by `ndarray.ts`, so it must stay as browser-safe as
 * `loader.ts` itself (same reasoning D2 applies to `threaded.ts`).
 */
import type { Mutable, Shape } from "../dim.ts";
import type { CoreExports } from "./loader.ts";
import { WNDArray, type StackRowsGuard, type StackShapeOf } from "./resident.ts";

export type BackendKind = "wasm" | "threaded";

/** Detail decision 2 (owner-abgenommen): passed through to
 * `initThreadedCore` (`workers`, `matmulTimeoutMs`) / `threadedMatmul`
 * (`minPoolWork`, as this backend's per-call default) by
 * `NDArray.backend("threaded", opts)`. */
export interface ThreadedBackendOptions {
  readonly workers?: number;
  readonly matmulTimeoutMs?: number;
  readonly minPoolWork?: number;
}

/** D2's env-detection check: Node + the threads artifact reachable. Returns
 * `null` when both hold, else the specific missing-condition reason string
 * `NDArray.backend("threaded")`'s pinned error message names (detail
 * decision 4). `artifactUrl` defaults to the real threads-artifact location
 * (same directory as `threaded.ts`'s own `WASM_URL`, i.e. this module's own
 * directory) — overridable so tests can pin the "threads artifact not
 * found" branch deterministically without depending on whether
 * `build:wasm:threads` has (or hasn't) left a real artifact on disk (this
 * module's own test-only hook, same rationale as `resident.ts`'s
 * `getResidentFreeCount()`). */
export async function checkThreadedEnv(
  artifactUrl: URL = new URL("./numtype_core_threads.wasm", import.meta.url),
): Promise<string | null> {
  if (typeof process === "undefined") {
    return "not running under Node";
  }
  try {
    const { readFile } = await import("node:fs/promises");
    await readFile(artifactUrl);
    return null;
  } catch {
    return "threads artifact not found";
  }
}

/**
 * D1 — the WASM performance backend facade. Wraps a `core: CoreExports`
 * obtained from `initCore()` and hands it straight through to the
 * EXISTING `WNDArray.*(core, ...)` statics, so `fromArray`/`zeros`/`ones`
 * here return real, unmodified `WNDArray<S>` instances — the facade only
 * heals the `core`-parameter ergonomics, it doesn't wrap or shadow the
 * class. `dispose()` marks this backend disposed so further creation
 * throws (no `CoreExports` teardown call exists — there is no `core`
 * teardown to call); it does NOT dispose any `WNDArray` created through it
 * (D1: the facade never hides WASM memory management — each array keeps its
 * own explicit `dispose()`).
 */
export class WasmBackend {
  private readonly core: CoreExports;
  private disposed = false;

  constructor(core: CoreExports) {
    this.core = core;
  }

  private assertLive(op: string): void {
    if (this.disposed) {
      throw new Error(`WasmBackend.${op}: backend has been disposed`);
    }
  }

  fromArray<const S extends Shape>(shape: S, values: readonly number[] | Float64Array): WNDArray<Mutable<S>> {
    this.assertLive("fromArray");
    return WNDArray.fromArray(this.core, shape, values);
  }

  zeros<const S extends Shape>(shape: S): WNDArray<Mutable<S>> {
    this.assertLive("zeros");
    return WNDArray.zeros(this.core, shape);
  }

  ones<const S extends Shape>(shape: S): WNDArray<Mutable<S>> {
    this.assertLive("ones");
    return WNDArray.ones(this.core, shape);
  }

  /** WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D2): reachability
   * for `WNDArray.stack` — the campaign's first STATIC op. `WNDArray` is
   * not exported from `index.ts`, so a package consumer can only reach
   * `stack` through this facade (or `ThreadedBackend`'s own copy); this
   * method is a pure one-line delegation with the `core` this backend
   * already holds, exactly like `fromArray`/`zeros`/`ones` above. */
  stack<const Rows extends readonly WNDArray<any>[]>(rows: StackRowsGuard<Rows>): WNDArray<StackShapeOf<Rows>> {
    this.assertLive("stack");
    return WNDArray.stack(this.core, rows);
  }

  dispose(): void {
    this.disposed = true;
  }
}
