// View-test classification tracer — a `--import` PRELOAD, never a source edit.
//
// Purpose (docs/view-test-klassifikation-spec.md, D1/D2): classify test cases
// by what they actually DO with views, instead of by what their names claim.
// The predecessor slice classified 13 blocks by mutating `spike/src` and reading
// test names; that method conflates "the test cannot see the damage" (a defect)
// with "the mutant cannot reach the test" (a non-measurement), and its keyword
// filter is the very claim under test.
//
// Everything here runs from a preload, so `spike/src` is NEVER modified — not
// even for the mutants. That is deliberate: two verifiers corrupted each other's
// measurements in a previous round because both mutated the shared working tree.
// With this design there is no mutated tree to corrupt.
//
// Usage (NODE_OPTIONS must be scoped INLINE — an exported value leaks into
// unrelated processes and into the `tsc` children the diagnostic-pin tests spawn):
//
//   NT_TRACE_OUT_DIR=<dir> NT_TRACE_MODE=trace \
//     NODE_OPTIONS="--import file:///abs/path/scripts/trace-view-tests.mjs" \
//     node --test spike/tests-runtime/foo.test.ts
//
// Modes:
//   trace    (default) observe only: record every view construction + readbacks
//   perturb  observe + Perturbation P: rewrite every NON-TRIVIAL view to the
//            contiguous re-interpretation of the same shape (natural strides,
//            offset 0). Guaranteed in-bounds: max linear index becomes size-1.
//   mutantS  emulate the predecessor slice's Mutant S: `slice` ignores its specs
//   mutantT  emulate the predecessor slice's Mutant T: `transpose` is identity
//
// One ledger file per process (`ledger-<pid>.jsonl`) — `node --test` runs one
// child process per test file, and per-process files make cross-process append
// interleaving impossible. The file is created LAZILY on the first record, so
// unrelated child processes that inherit NODE_OPTIONS leave no stray files.

import { appendFileSync, mkdirSync } from "node:fs";
import { beforeEach, afterEach } from "node:test";

const MODE = process.env.NT_TRACE_MODE ?? "trace";
const OUT_DIR = process.env.NT_TRACE_OUT_DIR;

const VALID_MODES = new Set(["trace", "perturb", "mutantS", "mutantT"]);
if (!VALID_MODES.has(MODE)) {
  throw new Error(`trace-view-tests: unknown NT_TRACE_MODE ${JSON.stringify(MODE)}`);
}
if (!OUT_DIR) {
  throw new Error("trace-view-tests: NT_TRACE_OUT_DIR is required");
}

// NODE_OPTIONS is inherited by nested Node/tsc child processes, so this preload
// can be evaluated more than once per logical test run. Installing twice would
// double-wrap the prototype methods and double-count every record.
const INSTALL_FLAG = "__ntViewTracerInstalled";
if (!globalThis[INSTALL_FLAG]) {
  globalThis[INSTALL_FLAG] = true;
  await install();
}

async function install() {
  const residentUrl =
    process.env.NT_RESIDENT_MOD ?? new URL("../spike/src/wasm/resident.ts", import.meta.url).href;
  const mod = await import(residentUrl);
  const WNDArray = mod.WNDArray;
  if (typeof WNDArray !== "function") {
    throw new Error(`trace-view-tests: WNDArray not exported from ${residentUrl}`);
  }

  const ledgerPath = `${OUT_DIR}/ledger-${process.pid}.jsonl`;
  let ledgerReady = false;
  const write = (obj) => {
    if (!ledgerReady) {
      mkdirSync(OUT_DIR, { recursive: true });
      ledgerReady = true;
    }
    appendFileSync(ledgerPath, JSON.stringify(obj) + "\n");
  };

  // ---- current-case state -------------------------------------------------
  // The corpus has no nested subtests, no describe(), no concurrency and no
  // async-overlapping tests (verified across all test:core/test:resident files),
  // so a single "current case" slot is race-free.
  let cur = null;
  const freshCase = (file, name) => ({
    file,
    name,
    views: [],
    perturbedCount: 0,
    readback: 0,
  });
  // Non-trivial view handles constructed by the CURRENT case — readback only
  // counts when the receiver is one of these.
  let nontrivialHandles = new WeakSet();

  const moduleScope = freshCase("<module-scope>", "<module-scope>");

  const slot = () => cur ?? moduleScope;

  // ---- geometry -----------------------------------------------------------
  const naturalStrides = (shape) => {
    const out = new Array(shape.length);
    let acc = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      out[i] = acc;
      acc *= shape[i];
    }
    return out;
  };

  const sizeOf = (shape) => shape.reduce((a, b) => a * b, 1);

  // A view is NON-TRIVIAL iff its index->address mapping
  //   lin(idx) = offset + sum_i idx[i]*strides[i]
  // differs from the contiguous mapping of the same shape, over the set of
  // VALID index tuples. Closed form (proven against brute force in
  // scripts/classify-view-tests.mjs --selftest, case 11):
  //   - size 0: no valid index tuple exists, nothing is observable -> trivial
  //   - an axis with shape[i] == 1 always has idx[i] == 0, so its stride can
  //     never contribute to an address -> ignore it
  // The naive version (`strides !== natural || offset !== 0`) is WRONG: a
  // transposed [1,5] view has strides [1,5] vs natural [1,1] yet 0 of 5
  // addresses differ.
  const isNonTrivial = (shape, strides, offset) => {
    if (sizeOf(shape) === 0) return false;
    if (offset !== 0) return true;
    const nat = naturalStrides(shape);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] > 1 && strides[i] !== nat[i]) return true;
    }
    return false;
  };

  // ---- caller attribution -------------------------------------------------
  // Readback must only count calls made from TEST code. `toArray()`/`describe()`
  // can also be reached internally, and `.shape` (which an earlier draft of the
  // spec wanted to use as the signal) is read internally ~4x per toArray() —
  // which would have made the signal pure noise.
  const isTestFrame = (stack) => {
    const lines = (stack ?? "").split("\n");
    // lines[0] is the Error header, lines[1] is the patched wrapper itself.
    for (let i = 2; i < lines.length; i++) {
      const l = lines[i];
      if (!l) continue;
      if (l.includes("/spike/src/")) return false; // internal caller
      if (l.includes("/spike/tests-runtime/")) return true; // test file or its helpers
      if (l.includes("node:")) continue;
      // Any other frame (the tracer itself, node internals): keep looking.
    }
    return false;
  };

  // ---- patch the view producers ------------------------------------------
  // These four are the only WNDArray methods that construct a handle with its
  // own (shape, strides, offset). Only `transpose`/`slice` can ever produce a
  // NON-TRIVIAL one — `reshape`/`flatten` always build with natural strides and
  // offset 0. All four are traced anyway: that they only ever yield K1 is an
  // OUTPUT of this measurement, not an assumption of it.
  for (const op of ["transpose", "slice", "reshape", "flatten"]) {
    const orig = WNDArray.prototype[op];
    if (typeof orig !== "function") {
      throw new Error(`trace-view-tests: WNDArray.prototype.${op} is not a function`);
    }
    WNDArray.prototype[op] = function tracedViewProducer(...args) {
      let out;
      if (MODE === "mutantS" && op === "slice") {
        // Mutant S: `slice` ignores its specs. Calling through with an empty
        // spec list is exactly `normalizeSliceSpecs(this.shape, [])`, i.e. a
        // full view — and retain / handle construction / registry all run
        // unchanged, so refcount semantics match the original source mutant.
        out = orig.apply(this, []);
      } else if (MODE === "mutantT" && op === "transpose") {
        // Mutant T: `transpose` is the identity. Call through (so retainBuffer,
        // the fresh handle and the registry registration all happen exactly as
        // in the original), then undo the reversal. Deliberately NOT
        // `return this`, which would change reference accounting and silently
        // produce a different candidate set.
        out = orig.apply(this, args);
        out.shape = [...this.shape];
        out.strides = [...this.strides];
      } else {
        out = orig.apply(this, args);
      }

      const shape = [...out.shape];
      const strides = [...out.strides];
      const offset = out.offset;
      const nontrivial = isNonTrivial(shape, strides, offset);

      const rec = { op, shape, strides, offset, nontrivial };

      if (MODE === "perturb" && nontrivial) {
        // Perturbation P: the contiguous re-interpretation of the same shape.
        // Uniform across all view classes (transposed / strided / offset /
        // composed), and in-bounds by construction: the maximum linear index
        // becomes size-1, which no WNDArray-producing path can exceed.
        out.strides = naturalStrides(shape);
        out.offset = 0;
        rec.perturbedTo = { strides: [...out.strides], offset: 0 };
        slot().perturbedCount += 1;
      }

      if (nontrivial) nontrivialHandles.add(out);
      slot().views.push(rec);
      return out;
    };
  }

  // ---- patch the readback surface -----------------------------------------
  for (const op of ["toArray", "describe"]) {
    const orig = WNDArray.prototype[op];
    if (typeof orig !== "function") {
      throw new Error(`trace-view-tests: WNDArray.prototype.${op} is not a function`);
    }
    WNDArray.prototype[op] = function tracedReadback(...args) {
      if (nontrivialHandles.has(this) && isTestFrame(new Error("trace").stack)) {
        slot().readback += 1;
      }
      return orig.apply(this, args);
    };
  }

  // ---- per-case ledger ----------------------------------------------------
  beforeEach((t) => {
    cur = freshCase(t.filePath ?? "<unknown>", t.name);
    nontrivialHandles = new WeakSet();
  });

  afterEach((t) => {
    const c = cur ?? freshCase(t.filePath ?? "<unknown>", t.name);
    cur = null;
    const err = t.error ?? null;
    write({
      ev: "case",
      mode: MODE,
      file: c.file,
      name: c.name,
      passed: t.passed === true,
      error: err
        ? {
            name: String(err.name ?? ""),
            code: String(err.code ?? ""),
            // Node's assert failures carry code ERR_ASSERTION / name
            // AssertionError. Anything else means the case died BEFORE an
            // assertion ran (e.g. a throw out of the data-access layer), which
            // must not be counted as structure sensitivity.
            message: String(err.message ?? "").slice(0, 400),
          }
        : null,
      views: c.views,
      nontrivial: c.views.some((v) => v.nontrivial),
      perturbedCount: c.perturbedCount,
      readback: c.readback,
    });
  });

  process.on("exit", () => {
    if (moduleScope.views.length > 0) {
      write({ ev: "module-scope", mode: MODE, file: moduleScope.file, views: moduleScope.views });
    }
  });
}
