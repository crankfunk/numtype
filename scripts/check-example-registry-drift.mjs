// Registry tripwire for the dogfooding example (owner decision 2026-07-20,
// option b of the covenant-verify Z2 finding — see FOLLOWUPS.md and
// docs/dogfooding-rag-ergebnisse.md, post-verification addendum).
//
// The example consumes the PUBLISHED numtype package, pinned by its committed
// lockfile. Without this gate, a future release would leave the example
// silently testing the OLD version forever — the CI job stays green while the
// example drifts out of representativeness. This script turns that silent rot
// into a red CI job: it fails as soon as the registry's `latest` major.minor
// differs from the version the example actually has installed.
//
// Patch drift is tolerated by design: the pre-1.0 SemVer policy (README,
// "Versioning") guarantees a patch never changes types or behavior, so a
// stale patch cannot make the example unrepresentative.
//
// Zero-dependency on purpose: global fetch (Node >= 18) + node:fs only.

import { readFileSync } from "node:fs";

const installedPath = new URL(
  "../examples/rag-demo/node_modules/numtype/package.json",
  import.meta.url,
);

let installed;
try {
  installed = JSON.parse(readFileSync(installedPath, "utf8")).version;
} catch {
  console.error(
    "check-example-registry-drift: examples/rag-demo/node_modules/numtype not found — " +
      "run `pnpm -C examples/rag-demo install --frozen-lockfile` first.",
  );
  process.exit(1);
}

let latest;
try {
  const res = await fetch("https://registry.npmjs.org/numtype/latest", {
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  latest = (await res.json()).version;
} catch (err) {
  console.error(
    `check-example-registry-drift: could not query the npm registry (${err}). ` +
      "The example gate depends on the registry by design — retry, or check the network.",
  );
  process.exit(1);
}

const majorMinor = (v) => v.split(".").slice(0, 2).join(".");

if (majorMinor(latest) !== majorMinor(installed)) {
  console.error(
    `check-example-registry-drift: registry latest is numtype@${latest}, but the example ` +
      `installs numtype@${installed} (same major.minor required — pre-1.0, minor = breaking). ` +
      'Bump "numtype" in examples/rag-demo/package.json, run `pnpm -C examples/rag-demo install` ' +
      "to refresh the lockfile, re-run `pnpm test:example`, and commit both.",
  );
  process.exit(1);
}

console.log(
  `check-example-registry-drift: ok — registry latest numtype@${latest}, ` +
    `example installs numtype@${installed} (major.minor match).`,
);
