/**
 * D-S3.1 (docs/item-11-api-paket-spec.md): Zero-dependency guard. Makes the
 * COVENANT-Z1 zero-dep-runtime claim a CHECKED GATE, not a promise: the
 * published package must never carry a `dependencies` field (devDependencies
 * — TypeScript, Playwright — are fine, they never reach a consumer's
 * node_modules). Non-vacuity: temporarily adding a `dependencies` field makes
 * this test fail, naming the offending keys (verified as a mutant in the S3
 * verify round).
 *
 * Reads `package.json` only — no product code, no WASM import — so it runs in
 * the fast plain `node --test` `test:core` corpus. Repo root resolved from
 * `import.meta.url` (never CWD), same convention as
 * `test-scripts-guard.test.ts` and `spike/src/wasm/loader.ts`.
 */
import { test } from "node:test";
import assert from "node:assert";

const repoRootUrl = new URL("../../", import.meta.url);
const fsp = (await import("node:fs/promises")) as unknown as {
  readFile(url: URL, encoding: string): Promise<string>;
};

interface PackageJson {
  readonly dependencies?: Record<string, string>;
  readonly optionalDependencies?: Record<string, string>;
  readonly peerDependencies?: Record<string, string>;
  readonly devDependencies?: Record<string, string>;
}

test("package.json carries NO runtime dependencies of any kind (COVENANT Z1: zero-dep runtime)", async () => {
  const raw = await fsp.readFile(new URL("package.json", repoRootUrl), "utf8");
  const pkg = JSON.parse(raw) as PackageJson;
  // S3-Verify F2: checking only `dependencies` leaves Z1-bypass vectors open —
  // `optionalDependencies` are auto-installed by npm/pnpm by default, and
  // `peerDependencies` still impose a runtime requirement on consumers. Only
  // `devDependencies` (TypeScript, Playwright) are allowed.
  const runtimeDepFields = ["dependencies", "optionalDependencies", "peerDependencies"] as const;
  const offenders: string[] = [];
  for (const field of runtimeDepFields) {
    const keys = Object.keys(pkg[field] ?? {});
    if (keys.length > 0) offenders.push(`${field}: [${keys.join(", ")}]`);
  }
  assert.strictEqual(
    offenders.length,
    0,
    `package.json must declare NO runtime dependencies of any kind (Z1: the runtime stays ` +
      `zero-dependency, kernels + type machinery from scratch), but found — ${offenders.join("; ")}. ` +
      `devDependencies are the ONLY allowed dependency field; anything else is a covenant violation.`,
  );
});
