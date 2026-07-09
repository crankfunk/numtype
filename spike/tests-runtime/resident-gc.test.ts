/**
 * Kern 02 GC-backstop test — best-effort, honestly labeled (spec's honesty
 * rule). Only asserts anything when `globalThis.gc` is available, i.e. when
 * this file is run as:
 *
 *   node --expose-gc --test spike/tests-runtime/resident-gc.test.ts
 *
 * wired as `pnpm test:resident:gc`. Without `--expose-gc`, `gc` is
 * `undefined` and the test is explicitly `skip`ped with a reason — never
 * silently faked as a pass, and never silently omitted either (it always
 * shows up in the test-runner's output, just annotated "skipped").
 *
 * The DETERMINISTIC half of the lifecycle contract this backstop depends on
 * (dispose() unregisters via the same token it registered with, so the
 * registry can never double-free after an explicit dispose) is tested
 * unconditionally in `resident-lifecycle.test.ts`'s double-dispose case —
 * that part does not need real GC to verify.
 *
 * Observability: rather than infer a free from `core.memory.buffer
 * .byteLength` (which only ever grows or plateaus, never shrinks — not a
 * usable "was this freed" signal on its own), this test uses the dedicated
 * `getResidentFreeCount()` instrumented counter, incremented exactly once
 * per actual `nt_free` call the registry callback makes.
 */
import assert from "node:assert";
import { test } from "node:test";
import { initCore } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray } from "../src/wasm/resident.ts";

const gcAvailable = typeof gc === "function";

test(
  "GC backstop: a dropped, never-disposed WNDArray is freed by the FinalizationRegistry",
  { skip: gcAvailable ? false : "requires --expose-gc (run via `pnpm test:resident:gc`) — globalThis.gc is not available in this invocation" },
  async () => {
    const core = await initCore();
    const freeCountBefore = getResidentFreeCount();

    // Allocate several arrays in a function scope, drop every reference,
    // and never call dispose() — the only thing keeping them alive after
    // this function returns is (nothing; they become garbage).
    (function allocateAndDrop(): void {
      for (let i = 0; i < 64; i++) {
        WNDArray.fromArray(core, [16, 16], new Array(256).fill(i));
      }
    })();

    // FinalizationRegistry callbacks run on a scheduled microtask-ish
    // cadence, not synchronously inside gc() — poll for a bit, forcing GC
    // each time, rather than assuming one pass suffices.
    let freeCountAfter = getResidentFreeCount();
    for (let attempt = 0; attempt < 20 && freeCountAfter <= freeCountBefore; attempt++) {
      gc?.();
      await new Promise((resolve) => setTimeout(resolve, 25));
      freeCountAfter = getResidentFreeCount();
    }

    assert.ok(
      freeCountAfter > freeCountBefore,
      `expected the FinalizationRegistry backstop to free at least one dropped allocation ` +
        `(before=${freeCountBefore}, after=${freeCountAfter}) after forcing GC repeatedly`,
    );
  },
);

// --- Kern 03: shared-buffer (view) variant ---------------------------------
// Base + transpose view share ONE buffer via refcount. Dropping BOTH without
// dispose must free that buffer EXACTLY once: each handle's registry
// finalizer releases one reference, the second release (whichever handle is
// finalized last) does the free. Delta > 1 here would mean a double-free of
// the shared allocation; delta 0 a leak.
test(
  "GC backstop with views: dropped base+view free their shared buffer exactly once",
  { skip: gcAvailable ? false : "requires --expose-gc (run via `pnpm test:resident:gc`) — globalThis.gc is not available in this invocation" },
  async () => {
    const core = await initCore();

    // Drain: let stragglers from the previous test finish finalizing, so the
    // exact-delta assertion below can't be polluted by their late callbacks.
    let stable = getResidentFreeCount();
    for (let quiet = 0; quiet < 3; ) {
      gc?.();
      await new Promise((resolve) => setTimeout(resolve, 25));
      const now = getResidentFreeCount();
      if (now === stable) quiet++;
      else {
        stable = now;
        quiet = 0;
      }
    }

    const before = getResidentFreeCount();
    (function allocateAndDrop(): void {
      // ONLY the base+view pair — any op here would dispose its own output
      // and move the counter synchronously, polluting the exact-delta check.
      const base = WNDArray.fromArray(core, [16, 16], new Array(256).fill(3));
      base.transpose(); // view handle dropped immediately; base dropped at return
    })();

    let after = getResidentFreeCount();
    for (let attempt = 0; attempt < 20 && after < before + 1; attempt++) {
      gc?.();
      await new Promise((resolve) => setTimeout(resolve, 25));
      after = getResidentFreeCount();
    }

    assert.strictEqual(
      after,
      before + 1,
      `expected the shared base+view buffer to be freed EXACTLY once by the registry ` +
        `(before=${before}, after=${after}) — more would be a double-free, fewer a leak`,
    );
  },
);

if (!gcAvailable) {
  console.log(
    "[resident-gc.test.ts] globalThis.gc unavailable — GC-backstop assertion skipped honestly. " +
      "Run `pnpm test:resident:gc` (adds --expose-gc) to actually exercise it.",
  );
}
