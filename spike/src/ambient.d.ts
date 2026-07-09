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

declare module "node:fs/promises" {
  export function readFile(path: string | URL): Promise<Uint8Array>;
}

declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): void;
}

declare module "node:assert" {
  function ok(value: unknown, message?: string | Error): asserts value;
  function strictEqual<T>(actual: T, expected: T, message?: string | Error): void;
  function deepStrictEqual<T>(actual: T, expected: T, message?: string | Error): void;
  /** `error` may be a RegExp matched against the thrown error's message, or
   * a validation function receiving the thrown value and returning
   * whether it's acceptable (both forms used by this project's tests). */
  function throws(fn: () => unknown, error?: RegExp | ((err: unknown) => boolean), message?: string): void;
  const assertDefault: {
    ok: typeof ok;
    strictEqual: typeof strictEqual;
    deepStrictEqual: typeof deepStrictEqual;
    throws: typeof throws;
  };
  export default assertDefault;
  export { ok, strictEqual, deepStrictEqual, throws };
}
