/**
 * Hand-written ambient declaration for the single Node.js API main.ts
 * calls (`node:assert`'s `ok`/`strictEqual`). Mirrors the root project's own
 * house rule (spike/src/ambient.d.ts): `@types/node` is deliberately not a
 * dependency here either — this example's own package.json is meant to look
 * like what an external numtype consumer would actually write (only
 * `numtype` + `typescript`), so the same "minimal, scoped-to-what-we-
 * actually-call shim" discipline applies, one level down.
 */
declare module "node:assert" {
  function ok(value: unknown, message?: string): asserts value;
  function strictEqual<T>(actual: T, expected: T, message?: string): void;
  const assertDefault: {
    ok: typeof ok;
    strictEqual: typeof strictEqual;
  };
  export default assertDefault;
  export { ok, strictEqual };
}
