// Item 12 (CI) — frozen-baseline artifact hash gate (COVENANT M4).
//
// Computes SHA-256 of the built `spike/src/wasm/numtype_core.wasm` and checks it
// against a SET of platform-labelled known-good pins. Green (exit 0) iff the hash
// equals ONE set member; otherwise exit 1, printing the actual hash prominently so
// the reaction is a trivial one-line edit.
//
// Why a SET, not a single pin (Item-12 spec D4 + coding-kb note
// `reproducible-wasm-build-plattform-gepinnt`): a wasm32-unknown-unknown release
// artifact is empirically NOT byte-stable across hosts (macOS-arm64 ≠ linux-x64,
// via embedded file:line panic locations). The macOS-arm64 pin below was produced
// locally with rustc 1.95.0; CI runs on ubuntu and may (likely will) produce a
// different, equally-reproducible Linux hash — which then gets added as a second
// labelled entry. M4's change-detection guarantee holds either way: ANY change to
// the frozen kernels matches NONE of the pins (SHA-256, cryptographically).
//
// PIN-SET DISCIPLINE (spec D4 / Baustein-0 D-3): when the frozen baseline
// LEGITIMATELY changes (a future M4-touching slice), REPLACE the whole set — never
// append. The set contains only the hashes of the CURRENT baseline state, one per
// platform. An appended, stale pin could silently wave through a later regression
// back to an old kernel state, defeating M4.
//
// Zero runtime deps (node:crypto / node:fs only), same discipline as
// scripts/check-dist-emit.mjs.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const WASM_PATH = join(fileURLToPath(new URL(".", import.meta.url)), "..", "spike", "src", "wasm", "numtype_core.wasm");

// The CURRENT frozen-baseline hash, per platform. Replace wholesale on a
// legitimate M4 change; never append stale entries.
const KNOWN_PINS = [
  { platform: "macos-arm64 / rustc 1.95.0", hash: "24a048c767f3949ad0a8747cecccc0e25e25bdad859c5deb45e218a39d70cea2" },
];

let bytes;
try {
  bytes = readFileSync(WASM_PATH);
} catch (e) {
  console.error(`check-freeze-hash: FAIL — could not read the WASM artifact at ${WASM_PATH}.\n` + `Did you run \`pnpm build:wasm\` first? Underlying: ${String(e)}`);
  process.exit(1);
}

const actual = createHash("sha256").update(bytes).digest("hex");

if (KNOWN_PINS.some((p) => p.hash === actual)) {
  const match = KNOWN_PINS.find((p) => p.hash === actual);
  console.log(`check-freeze-hash: OK — artifact matches the frozen-baseline pin [${match.platform}]\n  ${actual}`);
  process.exit(0);
}

console.error(
  `check-freeze-hash: FAIL — the built WASM artifact matches NONE of the known frozen-baseline pins.\n` +
    `  actual : ${actual}\n` +
    `  known  :\n` +
    KNOWN_PINS.map((p) => `    - ${p.hash}  [${p.platform}]`).join("\n") +
    `\n\n` +
    `If NO frozen kernel changed and this is the first run on a NEW platform (e.g. Linux x64 in CI),\n` +
    `this is EXPECTED (D4): add the actual hash above as a new labelled entry in KNOWN_PINS.\n` +
    `If a frozen kernel DID change, this is a real M4 violation — investigate before re-pinning.`,
);
process.exit(1);
