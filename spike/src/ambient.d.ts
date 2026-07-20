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
  /** Item 10: the async twin of `throws` — awaits a rejecting `Promise`
   * (e.g. `NDArray.backend("threaded")`'s env-detection throw, which is
   * async because `backend()` itself is). */
  function rejects(fn: () => Promise<unknown>, error?: RegExp | ((err: unknown) => boolean), message?: string): Promise<void>;
  const assertDefault: {
    ok: typeof ok;
    strictEqual: typeof strictEqual;
    deepStrictEqual: typeof deepStrictEqual;
    notStrictEqual: typeof notStrictEqual;
    throws: typeof throws;
    rejects: typeof rejects;
  };
  export default assertDefault;
  export { ok, strictEqual, deepStrictEqual, notStrictEqual, throws, rejects };
}

/** Spike 02: the handful of `Buffer` surface the editor-latency LSP client's
 * JSON-RPC byte-framing needs. `Content-Length` in the LSP wire protocol is
 * a BYTE count, not a UTF-16 code-unit count — plain JS strings can't keep
 * that correct once a message contains any non-ASCII byte (this project's
 * own generated doc comments use em-dashes), so `Buffer` (which `extends
 * Uint8Array`, itself already ambient from the ES2022 lib) is the minimal
 * correct tool, not a general Node typings shim. */
interface Buffer extends Uint8Array {
  toString(encoding?: string): string;
  /** Widened to `string | number` (real Node's Buffer.indexOf overloads
   * both) so this remains a compatible override of `Uint8Array.indexOf`'s
   * `number`-only signature — a narrower `(value: string) => number` alone
   * fails TS's interface-extension assignability check. */
  indexOf(value: string | number, byteOffset?: number): number;
  subarray(start?: number, end?: number): Buffer;
}
declare var Buffer: {
  from(data: string, encoding?: string): Buffer;
  byteLength(data: string, encoding?: string): number;
  concat(list: readonly Uint8Array[]): Buffer;
};

/** Spike 02: synchronous `fs` calls used by the workload generator and the
 * harness (reading generated files back, writing the manifest). Deliberately
 * separate from the existing `node:fs/promises` shim above — sync, not
 * async, a different module specifier in real Node too. */
declare module "node:fs" {
  export function mkdirSync(path: string, options?: { recursive?: boolean }): string | undefined;
  export function rmSync(path: string, options?: { recursive?: boolean; force?: boolean }): void;
  export function writeFileSync(path: string, data: string): void;
  export function readFileSync(path: string, encoding: "utf8"): string;
  /** Item 12 (CI): the S1 import guard (spike/tests-runtime/s1-import-guard.test.ts)
   * lists spike/src recursively to scan every source file's imports. Recursive
   * form returns each entry's path relative to `path`; same scoped-shim
   * discipline as the rest of this file. */
  export function readdirSync(path: string, options: { recursive: true }): string[];
}

/** Spike 02: the `path` helpers used to build workload/tsconfig file paths
 * portably (this project runs from a fixed repo-root CWD per CLAUDE.md, but
 * the generator/harness still compute paths relative to their own module
 * location via `import.meta.url`, not a hardcoded absolute string). */
declare module "node:path" {
  export function dirname(p: string): string;
  export function join(...paths: string[]): string;
}

/** Spike 02: `fileURLToPath` — converts `import.meta.url` (a `file://` URL)
 * to a plain filesystem path, the one `node:url` export this project needs. */
declare module "node:url" {
  export function fileURLToPath(url: string): string;
}

/** V3 (Phase-D-Vorarbeiten, Browser-Smoke-Test): the minimal `node:http`
 * surface the browser test fixture's static file server needs — same
 * "scoped shim, not a general Node typings shim" discipline as every other
 * declaration in this file. Just enough to serve a directory of static
 * files (js/wasm/html) with per-response headers, on an OS-assigned
 * ephemeral port (`listen(0, ...)` + `.address()`), and shut down cleanly. */
declare module "node:http" {
  interface IncomingMessageLike {
    readonly url?: string;
    readonly method?: string;
  }
  interface ServerResponseLike {
    writeHead(statusCode: number, headers?: Record<string, string>): void;
    end(data?: string | Uint8Array): void;
  }
  interface AddressInfoLike {
    readonly port: number;
  }
  interface ServerLike {
    listen(port: number, hostname: string, callback?: () => void): ServerLike;
    close(callback?: (err?: Error) => void): void;
    address(): AddressInfoLike | null;
  }
  export function createServer(handler: (req: IncomingMessageLike, res: ServerResponseLike) => void): ServerLike;
}

/** Spike 02: the minimal `child_process` surface the LSP harness needs —
 * `spawn` to run the native `tsc --lsp --stdio` server as a long-lived
 * subprocess (stdio piped, JSON-RPC framed over stdin/stdout), `execFileSync`
 * to run one-shot `tsc --extendedDiagnostics` instantiation-count passes.
 * Modeled loosely/structurally (not the real `ChildProcess` class) — just
 * the methods/events this file actually calls. */
declare module "node:child_process" {
  interface ReadableLike {
    on(event: "data", listener: (chunk: Buffer) => void): void;
  }
  interface WritableLike {
    write(data: string): boolean;
    end(): void;
  }
  interface ChildProcessLike {
    readonly stdin: WritableLike;
    readonly stdout: ReadableLike;
    readonly stderr: ReadableLike;
    on(event: "exit", listener: (code: number | null, signal: string | null) => void): void;
    on(event: "error", listener: (err: Error) => void): void;
    kill(signal?: string): boolean;
  }
  export function spawn(command: string, args: readonly string[], options?: { stdio?: readonly string[]; cwd?: string }): ChildProcessLike;
  /** Throws on non-zero exit (including when the compiled program itself
   * reports type errors) — the thrown value still carries `.stdout` with
   * whatever was written before exit; callers narrow it defensively rather
   * than relying on this type alone. */
  export function execFileSync(command: string, args: readonly string[], options: { encoding: "utf8"; cwd?: string }): string;
}

/** W2 verify round (Verify-B finding F1 — the diagnostic-quality pin in
 * spike/tests-runtime/scalar-mean.test.ts): that test compiles a throwaway
 * fixture OUTSIDE the repo with the real tsc and asserts the MESSAGE
 * CONTENT of the broadcast ShapeError (an `@ts-expect-error` cannot see
 * message text). It needs a collision-safe temp dir (`mkdtempSync` +
 * `tmpdir` — pid-safe, unlike a fixed path; see the parallel-test temp-dir
 * lesson) and a child-process runner that reports status + output WITHOUT
 * throwing on the expected non-zero exit (`spawnSync` — `execFileSync`
 * above throws on exactly the exit this test wants to observe). Re-opened
 * module blocks merge with the declarations above; same scoped-shim
 * discipline as every other block in this file. */
declare module "node:fs" {
  export function mkdtempSync(prefix: string): string;
}
declare module "node:os" {
  export function tmpdir(): string;
}
declare module "node:child_process" {
  export function spawnSync(
    command: string,
    args: readonly string[],
    options: { cwd?: string; encoding: "utf8" },
  ): { status: number | null; stdout: string | null; stderr: string | null };
}
