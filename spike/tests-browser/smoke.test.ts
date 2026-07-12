/**
 * V3 — Browser-Smoke-Test (docs/phase-d-vorarbeiten-spec.md, D-V3.4): the
 * first proof, in a REAL Chromium (not a Node moduleLoadList trace, not a
 * bundler), of the architecture claim FOLLOWUPS has carried since Item 10 —
 * "the standard surface (plain JS `NDArray`) and the opt-in
 * `NDArray.backend('wasm')` run in a browser, COOP/COEP-free". Everything
 * this file touches is TEST-ONLY (D-V3.5) — zero `spike/src` changes beyond
 * the pre-registered `node:http` ambient-declaration prerequisite
 * (spike/src/ambient.d.ts).
 *
 * Fixture: `server.ts` serves the tsc-emitted product tree
 * (spike/tests-browser/.emit/, produced by tsconfig.emit.json +
 * `test:browser`'s wasm-copy step — see package.json) as plain static
 * files, on an OS-assigned ephemeral port, with NO COOP/COEP headers. All
 * four assertions run against the SAME running server; each test navigates
 * its own fresh `page` to a synthesized blank HTML document at that origin
 * (see server.ts) so `import()` inside `page.evaluate` resolves relative to
 * a real http origin.
 *
 * The op-matrix differential (assertion 2) runs entirely INSIDE the page —
 * `page.evaluate` ships the whole comparison function to Chromium, which
 * dynamically imports the real emitted `ndarray.js`/`backend-api.js` and
 * calls the real classes; only the plain-object `{ pass, failures }`
 * summary crosses back over CDP to Node. Comparison discipline mirrors
 * `spike/tests-runtime/assert-helpers.ts`'s `assertDataBitIdentical`:
 * `Object.is` per element (distinguishes +0/-0, treats NaN as an equal
 * value CLASS, exactly this project's established "bit-exact" standard —
 * see that file's own doc comment for why `Object.is`, not `===`), not
 * literal byte-buffer comparison — replicated inline here (not imported)
 * because the comparator must ship into the PAGE as source text via
 * `page.evaluate`, and `spike/tests-runtime/` is a different, Node-only
 * corpus (D-V3.2 keeps this directory's own tsconfig fully separate).
 *
 * Local `ArrLike`/`NDArrayNS` interfaces below are a deliberately LOOSE
 * structural shape (not the real branded `Guard`/`Shape` generics from
 * `spike/src/ndarray.ts`) — this test proves RUNTIME parity and the raw
 * loading path, not the compile-time shape-checking USP (that is fully
 * covered elsewhere, e.g. `spike/tests/*.test-d.ts`); fighting the real
 * generic machinery inside a dynamically-typed `page.evaluate` payload
 * would add friction with no corresponding coverage gain.
 */
import { expect, test } from "@playwright/test";
import { startStaticServer, type StaticServer } from "./server.ts";
import { assertEmitFresh } from "./emit-freshness.ts";

const EMIT_DIR_URL = new URL("./.emit/", import.meta.url);
const SRC_DIR_URL = new URL("../src/", import.meta.url);

// Pinned message stem, ndarray.ts:240 (Item 10, D2) — the Item-10 spec's
// browser-safety claim, verified there only via a Node `moduleLoadList`
// trace; this is that claim's first real-browser confirmation.
const THREADED_MESSAGE_STEM = "threaded backend requires Node with the threads artifact";

let server: StaticServer;

test.beforeAll(async () => {
  // F1 (verify-round finding): a direct `playwright test` invocation (as
  // opposed to the `pnpm test:browser` wrapper, which always `rm -rf`s and
  // re-emits) would otherwise serve whatever `.emit/` tree happens to be on
  // disk — including a stale one compiled from an OLDER spike/src state —
  // with no signal at all. Fail hard, with a clear message, before starting
  // the server. See spike/tests-browser/emit-freshness.ts.
  await assertEmitFresh(EMIT_DIR_URL, SRC_DIR_URL, "src/ndarray.js");
  server = await startStaticServer(EMIT_DIR_URL);
});

test.afterAll(async () => {
  await server.close();
});

test.beforeEach(async ({ page }) => {
  await page.goto(server.baseUrl + "/");
});

// =============================================================================
// D-V3.4.1 — environment proof: a real browser, COOP-free, no Node globals.
// =============================================================================

test("environment: no Node process global, not cross-origin isolated", async ({ page }) => {
  const env = await page.evaluate(() => ({
    hasProcess: typeof process !== "undefined",
    crossOriginIsolated: crossOriginIsolated,
  }));
  expect(env.hasProcess).toBe(false);
  expect(env.crossOriginIsolated).toBe(false);
});

// =============================================================================
// D-V3.4.2 — op-matrix differential, entirely in-page.
// =============================================================================

test("op-matrix differential: JS NDArray oracle vs backend('wasm') WNDArray, byte-exact", async ({ page }) => {
  const result = await page.evaluate(async (baseUrl: string) => {
    interface ArrLike {
      readonly shape: readonly number[];
      data?: Float64Array;
      add(o: ArrLike): ArrLike;
      sub(o: ArrLike): ArrLike;
      mul(o: ArrLike): ArrLike;
      div(o: ArrLike): ArrLike;
      matmul(o: ArrLike): ArrLike;
      sum(axis?: number, keepdims?: boolean): ArrLike;
      transpose(): ArrLike;
      slice(...specs: ReadonlyArray<number | null | { start?: number; stop?: number; step?: number }>): ArrLike;
      reshape(shape: readonly number[]): ArrLike;
      dot(o: ArrLike): number;
      norm(): number;
      cosineSimilarity(o: ArrLike): number;
      toArray?(): Float64Array;
      dispose?(): void;
    }
    interface WasmBackendLike {
      fromArray(shape: readonly number[], values: readonly number[]): ArrLike;
      zeros(shape: readonly number[]): ArrLike;
      ones(shape: readonly number[]): ArrLike;
      dispose?(): void;
    }
    interface NDArrayNS {
      fromArray(shape: readonly number[], values: readonly number[]): ArrLike;
      zeros(shape: readonly number[]): ArrLike;
      ones(shape: readonly number[]): ArrLike;
      backend(kind: "wasm"): Promise<WasmBackendLike>;
    }

    const mod = (await import(new URL("/src/ndarray.js", baseUrl).href)) as { NDArray: NDArrayNS };
    const NDArray = mod.NDArray;
    const wasmBackend = await NDArray.backend("wasm");

    const failures: string[] = [];
    const disposables: ArrLike[] = [];
    function track<T extends ArrLike>(x: T): T {
      disposables.push(x);
      return x;
    }
    function dataOf(a: ArrLike): Float64Array {
      return a.toArray ? a.toArray() : (a.data as Float64Array);
    }
    function shapeEqual(a: readonly number[], b: readonly number[]): boolean {
      return a.length === b.length && a.every((v, i) => v === b[i]);
    }
    // Object.is per element: same standard as
    // spike/tests-runtime/assert-helpers.ts's assertDataBitIdentical (+0/-0
    // distinguished, NaN treated as an equal value class regardless of
    // payload) — see this file's module doc comment for why.
    function dataEqual(a: Float64Array, b: Float64Array): boolean {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (!Object.is(a[i], b[i])) return false;
      }
      return true;
    }
    function check(name: string, ref: ArrLike, got: ArrLike): void {
      if (!shapeEqual(ref.shape, got.shape)) {
        failures.push(`${name}: shape mismatch ref=[${ref.shape}] got=[${got.shape}]`);
        return;
      }
      const rd = dataOf(ref);
      const gd = dataOf(got);
      if (!dataEqual(rd, gd)) {
        failures.push(`${name}: data mismatch ref=[${Array.from(rd)}] got=[${Array.from(gd)}]`);
      }
    }
    function checkScalar(name: string, ref: number, got: number): void {
      if (!Object.is(ref, got)) {
        failures.push(`${name}: scalar mismatch ref=${ref} got=${got}`);
      }
    }

    try {
      // --- fromArray / zeros / ones -----------------------------------
      const shape23 = [2, 3];
      const dataA = [1, -2, 3.5, 0, 5, -6];
      const dataB = [10, 20, 30, 40, 50, 60];

      const refFA = NDArray.fromArray(shape23, dataA);
      const gotFA = track(wasmBackend.fromArray(shape23, dataA));
      check("fromArray", refFA, gotFA);

      check("zeros", NDArray.zeros(shape23), track(wasmBackend.zeros(shape23)));
      check("ones", NDArray.ones(shape23), track(wasmBackend.ones(shape23)));

      // --- add/sub/mul/div (same-shape) + one broadcast case ----------
      const refB = NDArray.fromArray(shape23, dataB);
      const gotB = track(wasmBackend.fromArray(shape23, dataB));

      check("add", refFA.add(refB), track(gotFA.add(gotB)));
      check("sub", refFA.sub(refB), track(gotFA.sub(gotB)));
      check("mul", refFA.mul(refB), track(gotFA.mul(gotB)));
      check("div", refFA.div(refB), track(gotFA.div(gotB)));

      const bcShape = [3];
      const bcData = [1, 2, 3];
      const refBC = NDArray.fromArray(bcShape, bcData);
      const gotBC = track(wasmBackend.fromArray(bcShape, bcData));
      check("add-broadcast [2,3]+[3]", refFA.add(refBC), track(gotFA.add(gotBC)));

      // --- matmul (resident matmul always calls nt_matmul_blocked) ----
      const shapeM1 = [2, 3];
      const dataM1 = [1, 2, 3, 4, 5, 6];
      const shapeM2 = [3, 2];
      const dataM2 = [7, 8, 9, 10, 11, 12];
      const refM1 = NDArray.fromArray(shapeM1, dataM1);
      const refM2 = NDArray.fromArray(shapeM2, dataM2);
      const gotM1 = track(wasmBackend.fromArray(shapeM1, dataM1));
      const gotM2 = track(wasmBackend.fromArray(shapeM2, dataM2));
      check("matmul", refM1.matmul(refM2), track(gotM1.matmul(gotM2)));

      // --- sum: axis, keepdims -----------------------------------------
      check("sum axis=0", refFA.sum(0), track(gotFA.sum(0)));
      check("sum axis=1 keepdims=true", refFA.sum(1, true), track(gotFA.sum(1, true)));

      // --- transpose (view) --------------------------------------------
      check("transpose", refFA.transpose(), track(gotFA.transpose()));

      // --- offset slice view (integer index folds into offset) --------
      const shapeS = [4, 3];
      const dataS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
      const refS = NDArray.fromArray(shapeS, dataS);
      const gotS = track(wasmBackend.fromArray(shapeS, dataS));
      check("slice(1) offset view", refS.slice(1), track(gotS.slice(1)));

      // --- reshape: view branch (contiguous) + materialize branch -----
      // (a transposed WNDArray is never contiguous for rank >= 2 — forces
      // the nt_materialize path; resident.ts's own isContiguous() doc.)
      check("reshape view branch", refFA.reshape([3, 2]), track(gotFA.reshape([3, 2])));
      check(
        "reshape materialize branch (via transpose)",
        refFA.transpose().reshape([6]),
        track(track(gotFA.transpose()).reshape([6])),
      );

      // --- dot / norm / cosineSimilarity (scalar consumer ops) --------
      const shapeV = [3];
      const v1 = [1, 2, 3];
      const v2 = [4, 5, 6];
      const refV1 = NDArray.fromArray(shapeV, v1);
      const refV2 = NDArray.fromArray(shapeV, v2);
      const gotV1 = track(wasmBackend.fromArray(shapeV, v1));
      const gotV2 = track(wasmBackend.fromArray(shapeV, v2));
      checkScalar("dot", refV1.dot(refV2), gotV1.dot(gotV2));
      checkScalar("norm", refV1.norm(), gotV1.norm());
      checkScalar("cosineSimilarity", refV1.cosineSimilarity(refV2), gotV1.cosineSimilarity(gotV2));

      // --- special-values sample through add + matmul (Kern-10 concern:
      // SIMD-blocked matmul must NOT flush subnormals) -------------------
      const specialShape = [2, 3];
      const special = [NaN, -0, Infinity, 5e-324 /* smallest positive subnormal */, -Infinity, 1];
      const normalData = [1, 2, 3, 4, 5, 6];
      const refSpecA = NDArray.fromArray(specialShape, special);
      const refSpecB = NDArray.fromArray(specialShape, normalData);
      const gotSpecA = track(wasmBackend.fromArray(specialShape, special));
      const gotSpecB = track(wasmBackend.fromArray(specialShape, normalData));
      check("special-values add", refSpecA.add(refSpecB), track(gotSpecA.add(gotSpecB)));

      const specM1Shape = [2, 3];
      const specM1 = [NaN, -0, Infinity, 5e-324, -Infinity, 1];
      const specM2Shape = [3, 2];
      const specM2 = [1, 2, 3, 4, 5, 6];
      const refSpecM1 = NDArray.fromArray(specM1Shape, specM1);
      const refSpecM2 = NDArray.fromArray(specM2Shape, specM2);
      const gotSpecM1 = track(wasmBackend.fromArray(specM1Shape, specM1));
      const gotSpecM2 = track(wasmBackend.fromArray(specM2Shape, specM2));
      check("special-values matmul", refSpecM1.matmul(refSpecM2), track(gotSpecM1.matmul(gotSpecM2)));
    } finally {
      for (const d of disposables) {
        try {
          d.dispose?.();
        } catch {
          // best-effort cleanup only — a dispose failure here must not mask
          // an earlier comparison failure already recorded above.
        }
      }
      wasmBackend.dispose?.();
    }

    return { pass: failures.length === 0, failures };
  }, server.baseUrl);

  expect(result.pass, result.failures.join("\n")).toBe(true);
});

// =============================================================================
// D-V3.4.3 — streaming path non-vacuity: correct MIME + instantiateStreaming
// exists AND IS ACTUALLY TAKEN. The MUTATION proof (flip server.ts's .wasm
// MIME entry to application/octet-stream, observe this test go red, revert)
// is a manual step performed once by the executor — see the results doc —
// not part of the committed suite (D-V3.5: the suite always asserts the
// CORRECT MIME).
//
// F2 (verify-round finding): the two assertions above only proved
// `instantiateStreaming` EXISTS and the `.wasm` response carries the right
// content-type — neither proves loader.ts's browser branch (loader.ts:158)
// actually CALLS it. A mutant that disables that branch (e.g. `if (false &&
// !isNode && ...)`, forcing the ArrayBuffer fallback) left this test green
// before this fix. Fix: `page.addInitScript` installs a counting wrapper
// around `WebAssembly.instantiateStreaming` BEFORE any page script runs
// (Playwright guarantee), then re-navigates so the wrapper is in place for
// the dynamic `import()` that follows; the wrapper delegates to the real
// implementation (so behavior is unchanged) and records how many times it
// was actually invoked.
// =============================================================================

test("streaming: .wasm served as application/wasm, instantiateStreaming available", async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { __ntStreamingCalls: number };
    w.__ntStreamingCalls = 0;
    const original = WebAssembly.instantiateStreaming.bind(WebAssembly);
    WebAssembly.instantiateStreaming = ((...args: Parameters<typeof WebAssembly.instantiateStreaming>) => {
      w.__ntStreamingCalls++;
      return original(...args);
    }) as typeof WebAssembly.instantiateStreaming;
  });
  // Re-navigate (the shared beforeEach above already loaded the page once,
  // before this init script was registered) so the wrapper is present from
  // this page load's very first script, which is what a real page load
  // looks like too.
  await page.goto(server.baseUrl + "/");

  const supportsStreaming = await page.evaluate(() => typeof WebAssembly.instantiateStreaming === "function");
  expect(supportsStreaming).toBe(true);

  const wasmResponsePromise = page.waitForResponse((resp) => resp.url().endsWith(".wasm"));

  const initOk = await page.evaluate(async (baseUrl: string) => {
    const mod = (await import(new URL("/src/ndarray.js", baseUrl).href)) as {
      NDArray: { backend(kind: "wasm"): Promise<{ dispose?(): void }> };
    };
    const backend = await mod.NDArray.backend("wasm");
    backend.dispose?.();
    return true;
  }, server.baseUrl);
  expect(initOk).toBe(true);

  const wasmResponse = await wasmResponsePromise;
  expect(wasmResponse.status()).toBe(200);
  expect(wasmResponse.headers()["content-type"]).toBe("application/wasm");

  // F2's actual assertion: the streaming path was really taken, not merely
  // available. `toBeGreaterThanOrEqual(1)` rather than `toBe(1)` — this test
  // does not need to pin an exact call count, only non-vacuity.
  const streamingCallCount = await page.evaluate(() => (window as unknown as { __ntStreamingCalls: number }).__ntStreamingCalls);
  expect(streamingCallCount).toBeGreaterThanOrEqual(1);
});

// =============================================================================
// D-V3.4.4 — NDArray.backend("threaded") rejects in the browser with the
// pinned message stem (ndarray.ts:240) — the Item-10 browser-safety claim,
// first confirmed in a real browser rather than via a Node moduleLoadList
// trace.
// =============================================================================

test('NDArray.backend("threaded") rejects in the browser with the pinned message stem', async ({ page }) => {
  const result = await page.evaluate(async (baseUrl: string) => {
    const mod = (await import(new URL("/src/ndarray.js", baseUrl).href)) as {
      NDArray: { backend(kind: "threaded"): Promise<unknown> };
    };
    try {
      await mod.NDArray.backend("threaded");
      return { threw: false, message: "" };
    } catch (e) {
      return { threw: true, message: e instanceof Error ? e.message : String(e) };
    }
  }, server.baseUrl);

  expect(result.threw).toBe(true);
  expect(result.message).toContain(THREADED_MESSAGE_STEM);
});
