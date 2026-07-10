/**
 * Hand-written ambient declarations for the handful of Node.js APIs used
 * across the WASM loader/backend, differential tests, and the core bench
 * script. `@types/node` is deliberately NOT a dependency (the hard
 * constraint is zero new `package.json` entries on either side) — these
 * are minimal, scoped-to-what-we-actually-call shims, not a general Node
 * typings shim.
 *
 * Everything else used (WebAssembly.*, fetch, URL, performance.now,
 * import.meta) is already ambiently available from TypeScript's default
 * lib set for this project's `target: "ES2022"` (verified empirically:
 * compiles clean with no explicit `lib` option) — only Node-specific
 * globals/modules need declaring here.
 */

/** Feature-detect via `typeof process` (per spec) to distinguish Node from
 * a browser environment; we never dereference any property on it. */
declare var process: unknown;

/** Kern 02: `globalThis.gc`, only defined when Node is launched with
 * `--expose-gc` (see `spike/tests-runtime/resident-gc.test.ts`). Feature-
 * detected the same way as `process` above — `typeof gc === "function"` —
 * never called without that check. */
declare var gc: (() => void) | undefined;

declare module "node:fs/promises" {
  export function readFile(path: string | URL): Promise<Uint8Array>;
}

/** Kern 06: the handful of `worker_threads` surface `threaded.ts` (main)
 * and `threaded-worker.ts` (worker) actually call — not a general Node
 * typings shim, same discipline as the rest of this file. */
declare module "node:worker_threads" {
  export class Worker {
    constructor(filename: string | URL, options?: { workerData?: unknown });
    on(event: "exit", listener: (code: number) => void): this;
    on(event: "error", listener: (err: Error) => void): this;
    once(event: "message", listener: (value: any) => void): this;
    once(event: "exit", listener: (code: number) => void): this;
    once(event: "error", listener: (err: Error) => void): this;
    terminate(): Promise<number>;
  }
  export const parentPort: {
    postMessage(value: unknown): void;
  } | null;
  export const workerData: unknown;
}

/** Kern 06: `os.availableParallelism()` only (the default worker-pool size). */
declare module "node:os" {
  export function availableParallelism(): number;
}

declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): void;
  /** Kern 02: the 3-arg options form, used to `skip` the GC-backstop test
   * with an explicit reason when `--expose-gc` wasn't passed (honesty rule:
   * never silently fake a pass). */
  export function test(name: string, options: { skip?: boolean | string }, fn: () => void | Promise<void>): void;
  /** Kern 06: runs once after every test in the file completes — used to
   * `dispose()` the module-level worker pools so the test process exits by
   * itself (a live `worker_threads.Worker` keeps the event loop alive). */
  export function after(fn: () => void | Promise<void>): void;
}

declare module "node:assert" {
  function ok(value: unknown, message?: string | Error): asserts value;
  function strictEqual<T>(actual: T, expected: T, message?: string | Error): void;
  function deepStrictEqual<T>(actual: T, expected: T, message?: string | Error): void;
  function notStrictEqual<T>(actual: T, expected: T, message?: string | Error): void;
  /** `error` may be a RegExp matched against the thrown error's message, or
   * a validation function receiving the thrown value and returning
   * whether it's acceptable (both forms used by this project's tests). */
  function throws(fn: () => unknown, error?: RegExp | ((err: unknown) => boolean), message?: string): void;
  const assertDefault: {
    ok: typeof ok;
    strictEqual: typeof strictEqual;
    deepStrictEqual: typeof deepStrictEqual;
    notStrictEqual: typeof notStrictEqual;
    throws: typeof throws;
  };
  export default assertDefault;
  export { ok, strictEqual, deepStrictEqual, notStrictEqual, throws };
}
