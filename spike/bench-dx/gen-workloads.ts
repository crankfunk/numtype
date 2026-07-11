/**
 * Spike 02 workload generator (docs/spike-02-editor-latency-spec.md). Emits
 * deterministic `.ts` workload files under `spike/bench-dx/workloads/` that
 * import the REAL type layer (`spike/src`), plus a `tsconfig.json` (so the
 * language server auto-discovers a proper project when a workload file is
 * opened), one `tsconfig.<id>.json` per workload (single-file `"files"`
 * isolation, mirroring `spike/bench/tsconfig.json`'s pattern, for the
 * per-workload `--extendedDiagnostics` instantiation counts), and a
 * `manifest.json` the harness (`editor-latency.ts`) reads to know exactly
 * which position to hover/complete/toggle and what result to expect —
 * hand-computed here in plain JS, mirroring the type-level rules in
 * `broadcast.ts`/`matmul.ts`/`reduce.ts`/`slice-literal.ts` by hand, never
 * re-deriving them via any TS compiler call.
 *
 * Determinism: no `Date.now()`/`Math.random()`/host state anywhere in the
 * generated OUTPUT (this file's own top-level console logging is fine, it
 * isn't part of the output) — re-running this generator must produce
 * byte-identical `.ts`/`.json` files every time, so the workloads directory
 * is safe to `.gitignore` and regenerate on demand.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const workloadsDir = join(__dirname, "workloads");

/** Compiler options mirrored from the repo-root `tsconfig.json`, inlined
 * (not `extends`-inherited) into every generated workloads tsconfig — see
 * the note at the `main()` call site for why: a child config that only sets
 * `"files"` does NOT override an inherited `"include"` (TS unions them),
 * which silently pulled sibling files like this very generator into the
 * "isolated" per-workload program in an earlier version of this script.
 * Inlining sidesteps that inheritance footgun entirely. */
const WORKLOAD_COMPILER_OPTIONS = {
  strict: true,
  target: "ES2022",
  module: "ESNext",
  moduleResolution: "bundler",
  noEmit: true,
  allowImportingTsExtensions: true,
  skipLibCheck: true,
  noUncheckedIndexedAccess: true,
  exactOptionalPropertyTypes: true,
};

// ---------------------------------------------------------------------------
// Shared helpers — hand computations of what the TYPE layer should resolve
// to, mirroring dim.ts/broadcast.ts/matmul.ts/reduce.ts/slice-literal.ts.
// ---------------------------------------------------------------------------

/** Render a shape (numbers, or the literal string "number" for a dynamic
 * dim) the same way NDArray's hover does: `NDArray<[2, 4]>`. */
function fmtShape(dims: readonly (number | "number")[]): string {
  return `NDArray<[${dims.join(", ")}]>`;
}

interface HoverSpec {
  label: string;
  line: number; // 0-indexed
  character: number; // 0-indexed
  expected: string; // substring the hover text must contain
}

interface CompletionSpec {
  label: string;
  line: number;
  character: number;
}

interface ToggleStateSpec {
  label: string; // "fixed" | "broken"
  text: string; // full file text for this state (didChange sends this verbatim)
  diagnosticLine: number; // 0-indexed line the toggle-target diagnostic must (dis)appear on
  expectDiagnosticPresent: boolean;
  expectedCode: number; // TS diagnostic code expected when present (2741 = "missing property")
}

interface ToggleSpec {
  label: string;
  initialState: ToggleStateSpec; // matches what's persisted to disk
  otherState: ToggleStateSpec;
}

interface WorkloadSpec {
  id: string;
  fileName: string;
  text: string;
  hovers: HoverSpec[];
  completion: CompletionSpec;
  toggle?: ToggleSpec;
}

/** Small line-buffer builder shared by every workload: tracks generated
 * source lines so hover/completion positions can be captured by exact index
 * at the moment a line is pushed, instead of re-searching text afterward.
 * `push` splits on embedded `\n` so a multi-line block (e.g. the header
 * comment) still lands as one array entry PER ACTUAL LINE — critical, since
 * `lastIdx()`/hover `line` fields must match real 0-indexed file lines once
 * `lines.join("\n")` is written out. */
function makeBuilder() {
  const lines: string[] = [];
  return {
    push: (s: string) => {
      for (const l of s.split("\n")) lines.push(l);
    },
    lines,
    /** Index of the line just pushed (for capturing a hover/completion position). */
    lastIdx: () => lines.length - 1,
  };
}

/** Character offset of `token`'s second character within `line` — safely
 * inside the identifier's hover range for any token of length >= 2 (verified
 * empirically against the live server: a position anywhere inside a token's
 * range resolves that token's hover). Throws if `token` isn't found — an
 * internal consistency check against generator bugs (e.g. a renamed var). */
function charIn(line: string, token: string): number {
  const idx = line.indexOf(token);
  if (idx === -1) throw new Error(`gen-workloads: token "${token}" not found in line: ${line}`);
  return idx + 1;
}

/** Character offset right after a `.member` access's dot, for a completion
 * probe — works on any real (already-valid) member-access call site; the
 * server computes completions at a position regardless of what characters
 * follow the cursor, so no dangling/invalid syntax is needed. */
function charAfterDot(line: string, member: string): number {
  const idx = line.indexOf(`.${member}`);
  if (idx === -1) throw new Error(`gen-workloads: member ".${member}" not found in line: ${line}`);
  return idx + 1;
}

const GENERATED_HEADER = (title: string, body: string) =>
  [`/**`, ` * GENERATED FILE — see spike/bench-dx/gen-workloads.ts (${title}).`, ...body.split("\n").map((l) => ` * ${l}`), ` */`].join("\n");

// ---------------------------------------------------------------------------
// W1 — op-chain sweep: L in {1,5,10,25,50,100}, matmul/add alternating over
// fixed [8,8] operands (shape stays [8,8] the whole way — hand-verifiable by
// construction), capped by one transpose()+sum(0) so every op kind in the
// spec (matmul/add/transpose/sum) appears in every chain. Isolates "does
// latency grow with chain length" from "is shape arithmetic correct" (W2/W3).
// ---------------------------------------------------------------------------

function buildW1(): WorkloadSpec {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "W1 op-chain sweep",
      "Chains of matmul/add over literal [8,8] shapes, length L in {1,5,10,25,50,100},\neach capped by one transpose()+sum(0). Every chain step keeps the shape [8,8]\n(matmul/add against fixed [8,8] operands never changes it), so the final shape\nafter transpose+sum is [8] for every L — trivially hand-verifiable, which lets\nthe sweep isolate chain-length effects on hover/diagnostic latency cleanly.",
    ),
  );
  b.push(`import { NDArray } from "../../src/ndarray.ts";`);
  b.push(``);
  b.push(`const w1_weight = NDArray.zeros([8, 8]);`);
  b.push(`const w1_bias = NDArray.zeros([8, 8]);`);
  b.push(``);

  const hovers: HoverSpec[] = [];
  let completion: CompletionSpec | undefined;
  const CHAIN_LENGTHS = [1, 5, 10, 25, 50, 100];

  for (const L of CHAIN_LENGTHS) {
    b.push(`// --- chain length L=${L} ---`);
    b.push(`const chain${L}_x0 = NDArray.zeros([8, 8]);`);
    let prev = `chain${L}_x0`;
    for (let i = 1; i <= L; i++) {
      const cur = `chain${L}_x${i}`;
      const op = i % 2 === 1 ? `matmul(w1_weight)` : `add(w1_bias)`;
      b.push(`const ${cur} = ${prev}.${op};`);
      if (L === 100 && i === 100) {
        completion = { label: "W1 L=100 mid-chain member access", line: b.lastIdx(), character: charAfterDot(b.lines[b.lastIdx()]!, op.split("(")[0]!) };
      }
      prev = cur;
    }
    const finalName = `chain${L}_final`;
    b.push(`const ${finalName} = ${prev}.transpose().sum(0);`);
    hovers.push({ label: `W1 L=${L} final`, line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, finalName), expected: fmtShape([8]) });
    b.push(``);
  }

  if (!completion) throw new Error("W1: completion position not captured");
  return { id: "w1", fileName: "w1-chains.ts", text: b.lines.join("\n") + "\n", hovers, completion };
}

// ---------------------------------------------------------------------------
// W2 — rank/broadcast sweep: ranks r in {2,4,8,16}, every axis alternates
// which side is 1 (k even: A[k]=dimValue,B[k]=1; k odd: A[k]=1,B[k]=dimValue)
// — guarantees every axis broadcasts cleanly (never both non-1, never a
// mismatch) so the sweep stays a pure positive/hover case; W4 owns errors.
// ---------------------------------------------------------------------------

function buildW2(): WorkloadSpec {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "W2 rank/broadcast sweep",
      "Broadcasting add() at ranks r in {2,4,8,16}, mixed size-1 axes: axis k puts the\nnon-1 dim on A when k is even, on B when k is odd (dimValue = 2 + (k % 5),\nnever 1) — every axis broadcasts cleanly by construction, so the resulting\nshape is exactly the per-axis dimValue sequence, hand-verified here in JS.",
    ),
  );
  b.push(`import { NDArray } from "../../src/ndarray.ts";`);
  b.push(``);

  const hovers: HoverSpec[] = [];
  let completion: CompletionSpec | undefined;
  const RANKS = [2, 4, 8, 16];

  for (const r of RANKS) {
    const dimA: number[] = [];
    const dimB: number[] = [];
    const result: number[] = [];
    for (let k = 0; k < r; k++) {
      const dimValue = 2 + (k % 5);
      if (k % 2 === 0) {
        dimA.push(dimValue);
        dimB.push(1);
      } else {
        dimA.push(1);
        dimB.push(dimValue);
      }
      result.push(dimValue);
    }
    b.push(`// --- rank r=${r} ---`);
    b.push(`const w2_r${r}_a = NDArray.zeros([${dimA.join(", ")}]);`);
    b.push(`const w2_r${r}_b = NDArray.zeros([${dimB.join(", ")}]);`);
    const resultName = `w2_r${r}_result`;
    b.push(`const ${resultName} = w2_r${r}_a.add(w2_r${r}_b);`);
    hovers.push({ label: `W2 rank=${r}`, line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, resultName), expected: fmtShape(result) });
    if (r === 16) completion = { label: "W2 rank=16 member access", line: b.lastIdx(), character: charAfterDot(b.lines[b.lastIdx()]!, "add") };
    b.push(``);
  }

  if (!completion) throw new Error("W2: completion position not captured");
  return { id: "w2", fileName: "w2-broadcast.ts", text: b.lines.join("\n") + "\n", hovers, completion };
}

// ---------------------------------------------------------------------------
// W3 — slice/digit-arithmetic stress: many slice({start}) calls on 1-D
// arrays with base dims 1024/4096/65536 — every step is a literal-computed
// dim via slice-literal.ts's supported subset (step omitted, non-negative
// literal start), Kern 05's own highest-risk type machinery. A borrow-heavy
// K cycle (1, 9, 90, 900, 9000) forces genuine multi-digit borrow chains,
// falling back to K=1 whenever the cycle value would exceed the remaining
// dim (keeps every step valid without needing lookahead). Plus one multi-
// axis case (both axes range-sliced in a single call).
// ---------------------------------------------------------------------------

const SLICE_K_CYCLE = [1, 9, 90, 900, 9000];
const SLICE_STEPS = 12;

function buildSliceChain(b: ReturnType<typeof makeBuilder>, varPrefix: string, baseDim: number): { finalVar: string; finalDim: number } {
  b.push(`const ${varPrefix}_x0 = NDArray.zeros([${baseDim}]);`);
  let dim = baseDim;
  let prev = `${varPrefix}_x0`;
  for (let i = 1; i <= SLICE_STEPS; i++) {
    let k = SLICE_K_CYCLE[(i - 1) % SLICE_K_CYCLE.length]!;
    if (k >= dim) k = 1;
    const newDim = dim - k;
    const cur = `${varPrefix}_x${i}`;
    b.push(`const ${cur} = ${prev}.slice({ start: ${k} });`);
    prev = cur;
    dim = newDim;
  }
  return { finalVar: prev, finalDim: dim };
}

function buildW3(): WorkloadSpec {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "W3 slice/digit-arithmetic stress",
      `${SLICE_STEPS} chained slice({start}) calls per base dim (1024/4096/65536), each a\nliteral-computed dim via slice-literal.ts's digit-string subtraction (the\nproject's own highest-risk type machinery) — K cycles through\n[${SLICE_K_CYCLE.join(", ")}] (falling back to 1 once K would exceed the\nremaining dim) to force genuine multi-digit borrow chains, hand-computed here\nin plain JS (JS number subtraction; the TYPE layer is what does digit-string\narithmetic, not this generator). Plus one multi-axis range-slice case.`,
    ),
  );
  b.push(`import { NDArray } from "../../src/ndarray.ts";`);
  b.push(``);

  const hovers: HoverSpec[] = [];
  let completion: CompletionSpec | undefined;
  const BASE_DIMS = [1024, 4096, 65536];

  for (const d of BASE_DIMS) {
    b.push(`// --- base dim ${d} ---`);
    const { finalVar, finalDim } = buildSliceChain(b, `w3_d${d}`, d);
    hovers.push({ label: `W3 dim=${d} final (${SLICE_STEPS} slices)`, line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, finalVar), expected: fmtShape([finalDim]) });
    if (d === 65536) completion = { label: "W3 dim=65536 member access", line: b.lastIdx(), character: charAfterDot(b.lines[b.lastIdx()]!, "slice") };
    b.push(``);
  }

  b.push(`// --- multi-axis range slice (both axes computed in one call) ---`);
  b.push(`const w3_multi_base = NDArray.zeros([1024, 4096]);`);
  const multiResultName = `w3_multi_result`;
  b.push(`const ${multiResultName} = w3_multi_base.slice({ start: 100, stop: 1000 }, { start: 500, stop: 3500 });`);
  hovers.push({
    label: "W3 multi-axis slice",
    line: b.lastIdx(),
    character: charIn(b.lines[b.lastIdx()]!, multiResultName),
    expected: fmtShape([1000 - 100, 3500 - 500]),
  });
  b.push(``);

  if (!completion) throw new Error("W3: completion position not captured");
  return { id: "w3", fileName: "w3-slice.ts", text: b.lines.join("\n") + "\n", hovers, completion };
}

// ---------------------------------------------------------------------------
// W4 — error file: valid code + 2 deliberate errors (matmul inner-dim
// mismatch, broadcast conflict). The broadcast conflict is permanent (never
// toggled — a stable baseline error). The matmul mismatch's inner dim is the
// M3 toggle target: a single literal digit (3 <-> 5) that fixes/reintroduces
// exactly that ONE error via a single-token full-text didChange.
// ---------------------------------------------------------------------------

function buildW4(): WorkloadSpec {
  function render(toggleDim: 3 | 5): { text: string; validLine: number; validResultName: string; toggleLine: number } {
    const b = makeBuilder();
    b.push(
      GENERATED_HEADER(
        "W4 error file",
        "Valid code + 2 deliberate errors: a PERMANENT broadcast conflict\n([4,3] vs [5]) and a matmul inner-dim mismatch whose second operand's\nfirst dim is the M3 toggle target — 3 (matches, no error) or 5 (mismatch,\nerror) — flipped by a single-token full-text didChange in editor-latency.ts.",
      ),
    );
    b.push(`import { NDArray } from "../../src/ndarray.ts";`);
    b.push(``);
    b.push(`// --- valid: hover/completion proof-of-load position ---`);
    b.push(`const w4_valid_a = NDArray.zeros([2, 3]);`);
    b.push(`const w4_valid_b = NDArray.zeros([3, 4]);`);
    const validResultName = "w4_valid_result";
    b.push(`const ${validResultName} = w4_valid_a.matmul(w4_valid_b);`);
    const validLine = b.lastIdx();
    b.push(``);

    b.push(`// --- permanent error #1: broadcast conflict (never toggled) ---`);
    b.push(`const w4_bad_broadcast_a = NDArray.zeros([4, 3]);`);
    b.push(`const w4_bad_broadcast_b = NDArray.zeros([5]);`);
    b.push(`const w4_bad_broadcast_result = w4_bad_broadcast_a.add(w4_bad_broadcast_b);`);
    b.push(``);

    b.push(`// --- M3 toggle target: error #2, matmul inner-dim mismatch ---`);
    b.push(`const w4_toggle_a = NDArray.zeros([2, 3]);`);
    b.push(`const w4_toggle_b = NDArray.zeros([${toggleDim}, 4]);`);
    b.push(`const w4_toggle_result = w4_toggle_a.matmul(w4_toggle_b);`);
    const toggleLine = b.lastIdx();
    b.push(``);

    return { text: b.lines.join("\n") + "\n", validLine, validResultName, toggleLine };
  }

  const rBroken = render(5);
  const rFixed = render(3);

  const brokenState: ToggleStateSpec = { label: "broken (2 errors)", text: rBroken.text, diagnosticLine: rBroken.toggleLine, expectDiagnosticPresent: true, expectedCode: 2741 };
  const fixedState: ToggleStateSpec = { label: "fixed (1 error)", text: rFixed.text, diagnosticLine: rFixed.toggleLine, expectDiagnosticPresent: false, expectedCode: 2741 };

  const validLine = rBroken.text.split("\n")[rBroken.validLine]!;
  const hovers: HoverSpec[] = [{ label: "W4 valid matmul result", line: rBroken.validLine, character: charIn(validLine, rBroken.validResultName), expected: fmtShape([2, 4]) }];
  const completion: CompletionSpec = { label: "W4 member access", line: rBroken.validLine, character: charAfterDot(validLine, "matmul") };

  // Persisted on-disk state = the "broken" one (2 deliberate errors), matching
  // the spec's own framing ("a file with valid code + 2 deliberate errors").
  return {
    id: "w4",
    fileName: "w4-errors.ts",
    text: brokenState.text,
    hovers,
    completion,
    toggle: { label: "W4 matmul inner-dim toggle (3<->5)", initialState: brokenState, otherState: fixedState },
  };
}

// ---------------------------------------------------------------------------
// W5 — realistic mixed consumer (~200-400 LOC): an embedding-pipeline-shaped
// program. Section A: a dynamic-batch dense stack (the `number`-dim gradual
// boundary carried through every layer). Section B: a fully literal
// positional-style pipeline (matmul/add/transpose/slice/sum). Section C: a
// `fromArray`-built small matrix. Section D: a final combine step. LAYER_
// COUNT is tuned (see generation run) to land the file in the 200-400 LOC
// target while every "layer" is a genuine, non-padding dense-layer pattern.
// ---------------------------------------------------------------------------

const W5_LAYER_COUNT = 42;

function buildW5(): WorkloadSpec {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "W5 realistic mixed consumer",
      "An embedding-pipeline-shaped program: Section A is a dynamic-batch dense\nstack (batchSize: number is the gradual-typing boundary, carried through\nevery layer as shape [number, 8]); Section B is a fully literal positional-\nstyle pipeline (matmul/add/transpose/slice/sum); Section C builds a small\nmatrix via fromArray; Section D combines B and C's outputs. Every shape is\nhand-computed below, mirrored from the same rules W1-W3 already exercise.",
    ),
  );
  b.push(`import { NDArray } from "../../src/ndarray.ts";`);
  b.push(``);

  const hovers: HoverSpec[] = [];
  let completion: CompletionSpec | undefined;

  // --- Section A: dynamic batch dimension ---------------------------------
  b.push(`// --- Section A: dynamic batch dimension (gradual typing boundary) ---`);
  b.push(`// batchSize is not known until runtime; every layer below must stay`);
  b.push(`// callable and thread the \`number\` dim through rather than erroring.`);
  b.push(``);
  b.push(`declare const batchSize: number;`);
  b.push(``);
  b.push(`const w5_embedding_table = NDArray.zeros([64, 8]); // vocab=64, embed dim=8`);
  b.push(`const w5_batch_input = NDArray.zeros([batchSize, 8]); // pre-embedded batch, dynamic batch dim`);
  b.push(``);

  let prevHidden = "w5_batch_input";
  for (let i = 1; i <= W5_LAYER_COUNT; i++) {
    const weightName = `w5_dense${i}_weight`;
    const biasName = `w5_dense${i}_bias`;
    const hiddenName = `w5_hidden${i}`;
    b.push(`const ${weightName} = NDArray.zeros([8, 8]);`);
    b.push(`const ${biasName} = NDArray.zeros([8]);`);
    b.push(`const ${hiddenName} = ${prevHidden}.matmul(${weightName}).add(${biasName});`);
    if (i === W5_LAYER_COUNT) {
      completion = { label: `W5 dense layer ${i} member access`, line: b.lastIdx(), character: charAfterDot(b.lines[b.lastIdx()]!, "matmul") };
    }
    prevHidden = hiddenName;
    // Every 5th layer: a no-op full-axis literal slice on the static axis
    // (dim stays 8, still a genuine literal-computed slice call) — exercises
    // slicing inside the dynamic-batch section too, per the spec's op mix.
    if (i % 5 === 0) {
      const slicedName = `${hiddenName}_sliced`;
      b.push(`const ${slicedName} = ${hiddenName}.slice(null, { start: 0, stop: 8 });`);
      prevHidden = slicedName;
    }
    b.push(``);
  }
  // Re-derive the exact declaration line of `prevHidden` by content match
  // (robust regardless of whether the last layer also emitted a slice line).
  {
    const declIdx = b.lines.findIndex((l) => l.startsWith(`const ${prevHidden} =`));
    if (declIdx === -1) throw new Error(`W5: could not locate declaration of ${prevHidden}`);
    hovers.push({ label: "W5 dynamic-batch final hidden layer", line: declIdx, character: charIn(b.lines[declIdx]!, prevHidden), expected: fmtShape(["number", 8]) });
  }

  // --- Section B: literal positional pipeline -----------------------------
  b.push(`// --- Section B: literal-shape positional pipeline (no dynamic dims) ---`);
  b.push(``);
  b.push(`const w5_pos_base = NDArray.zeros([16, 8]);`);
  b.push(`const w5_pos_weight = NDArray.zeros([8, 8]);`);
  b.push(`const w5_pos_bias = NDArray.zeros([8]);`);
  b.push(`const w5_pos_hidden = w5_pos_base.matmul(w5_pos_weight).add(w5_pos_bias); // [16, 8]`);
  b.push(`const w5_pos_transposed = w5_pos_hidden.transpose(); // [8, 16]`);
  b.push(`const w5_pos_sliced = w5_pos_transposed.slice(null, { start: 2, stop: 10 }); // [8, 8] (10-2=8)`);
  const posSummedName = "w5_pos_summed";
  b.push(`const ${posSummedName} = w5_pos_sliced.sum(0); // [8]`);
  hovers.push({ label: "W5 literal positional pipeline final", line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, posSummedName), expected: fmtShape([8]) });
  b.push(``);

  // --- Section C: fromArray -------------------------------------------------
  b.push(`// --- Section C: fromArray-built small matrix ---`);
  b.push(``);
  b.push(`const w5_identity4 = NDArray.fromArray([4, 4], [`);
  b.push(`  1, 0, 0, 0,`);
  b.push(`  0, 1, 0, 0,`);
  b.push(`  0, 0, 1, 0,`);
  b.push(`  0, 0, 0, 1,`);
  b.push(`]);`);
  b.push(`const w5_small_weight = NDArray.zeros([4, 4]);`);
  const identityResultName = "w5_identity_result";
  b.push(`const ${identityResultName} = w5_identity4.matmul(w5_small_weight); // [4, 4]`);
  hovers.push({ label: "W5 fromArray matmul result", line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, identityResultName), expected: fmtShape([4, 4]) });
  b.push(``);

  // --- Section D: combine B and C -------------------------------------------
  b.push(`// --- Section D: combine Section B's and C's outputs ---`);
  b.push(``);
  b.push(`const w5_summary_weight = NDArray.zeros([8, 4]);`);
  const summaryName = "w5_summary";
  b.push(`const ${summaryName} = ${posSummedName}.matmul(w5_summary_weight); // [4]`);
  hovers.push({ label: "W5 final summary", line: b.lastIdx(), character: charIn(b.lines[b.lastIdx()]!, summaryName), expected: fmtShape([4]) });
  b.push(``);

  if (!completion) throw new Error("W5: completion position not captured");
  return { id: "w5", fileName: "w5-mixed.ts", text: b.lines.join("\n") + "\n", hovers, completion };
}

// ---------------------------------------------------------------------------
// W6 — reshape/flatten measurement workload (Kern 08,
// docs/kern-08-reshape-flatten-spec.md): real reshape()/flatten() call
// sites — a small reshape hover, a flatten hover with a computed literal, a
// big-dim flatten hover exercising LiteralShapeProduct's schoolbook digit
// multiplication ([1024,1024] -> [1048576]), a PERMANENT reshape product-
// mismatch error (never toggled), and the M3 toggle target: a second
// reshape product mismatch whose new shape's first dim is 3 (matches, no
// error) or 5 (mismatch, error) — flipped by a single-token full-text
// didChange, the SAME __shapeError missing-property mechanism (code 2741)
// as W4's matmul toggle. Closes the Spike-04 FOLLOWUPS hover-measurement
// obligation now that reshape/flatten actually exist (Kern 08).
// ---------------------------------------------------------------------------

function buildW6(): WorkloadSpec {
  function render(toggleDim: 3 | 5): {
    text: string;
    smallLine: number;
    smallName: string;
    flattenLine: number;
    flattenName: string;
    bigFlattenLine: number;
    bigFlattenName: string;
    toggleLine: number;
  } {
    const b = makeBuilder();
    b.push(
      GENERATED_HEADER(
        "W6 reshape/flatten measurement",
        "Real reshape()/flatten() call sites (Kern 08, docs/kern-08-reshape-flatten-spec.md):\na small reshape hover, a flatten hover with a computed literal, a big-dim\nflatten hover exercising LiteralShapeProduct's schoolbook digit multiplication\n([1024,1024] -> [1048576]), a PERMANENT reshape product-mismatch error (never\ntoggled), and the M3 toggle target: a second reshape product mismatch whose\nnew shape's first dim is 3 (matches, no error) or 5 (mismatch, error) -\nflipped by a single-token full-text didChange, the same __shapeError\nmissing-property mechanism as W4's matmul toggle.",
      ),
    );
    b.push(`import { NDArray } from "../../src/ndarray.ts";`);
    b.push(``);

    b.push(`// --- small reshape hover ---`);
    b.push(`const w6_small_base = NDArray.zeros([2, 3]);`);
    const smallName = "w6_small_reshaped";
    b.push(`const ${smallName} = w6_small_base.reshape([3, 2]);`);
    const smallLine = b.lastIdx();
    b.push(``);

    b.push(`// --- flatten hover: computed literal product ---`);
    b.push(`const w6_flatten_base = NDArray.zeros([2, 3]);`);
    const flattenName = "w6_flattened";
    b.push(`const ${flattenName} = w6_flatten_base.flatten();`);
    const flattenLine = b.lastIdx();
    b.push(``);

    b.push(`// --- big-dim flatten hover: digit-multiplication stress (1024*1024=1048576) ---`);
    b.push(`const w6_big_base = NDArray.zeros([1024, 1024]);`);
    const bigFlattenName = "w6_big_flattened";
    b.push(`const ${bigFlattenName} = w6_big_base.flatten();`);
    const bigFlattenLine = b.lastIdx();
    b.push(``);

    b.push(`// --- permanent error: reshape product mismatch (never toggled) ---`);
    b.push(`const w6_bad_base = NDArray.zeros([4, 3]);`);
    b.push(`const w6_bad_reshaped = w6_bad_base.reshape([5, 2]); // 5*2=10 != 12`);
    b.push(``);

    b.push(`// --- M3 toggle target: reshape product mismatch (3<->5) ---`);
    b.push(`const w6_toggle_base = NDArray.zeros([4, 3]);`);
    const toggleName = "w6_toggle_reshaped";
    b.push(`const ${toggleName} = w6_toggle_base.reshape([${toggleDim}, 4]);`);
    const toggleLine = b.lastIdx();
    b.push(``);

    return { text: b.lines.join("\n") + "\n", smallLine, smallName, flattenLine, flattenName, bigFlattenLine, bigFlattenName, toggleLine };
  }

  const rBroken = render(5);
  const rFixed = render(3);

  const brokenState: ToggleStateSpec = { label: "broken (reshape mismatch)", text: rBroken.text, diagnosticLine: rBroken.toggleLine, expectDiagnosticPresent: true, expectedCode: 2741 };
  const fixedState: ToggleStateSpec = { label: "fixed (reshape ok)", text: rFixed.text, diagnosticLine: rFixed.toggleLine, expectDiagnosticPresent: false, expectedCode: 2741 };

  const brokenLines = rBroken.text.split("\n");
  const hovers: HoverSpec[] = [
    { label: "W6 small reshape", line: rBroken.smallLine, character: charIn(brokenLines[rBroken.smallLine]!, rBroken.smallName), expected: fmtShape([3, 2]) },
    { label: "W6 flatten computed literal", line: rBroken.flattenLine, character: charIn(brokenLines[rBroken.flattenLine]!, rBroken.flattenName), expected: fmtShape([6]) },
    {
      label: "W6 big-dim flatten (digit multiplication)",
      line: rBroken.bigFlattenLine,
      character: charIn(brokenLines[rBroken.bigFlattenLine]!, rBroken.bigFlattenName),
      expected: fmtShape([1048576]),
    },
  ];
  const completion: CompletionSpec = { label: "W6 member access", line: rBroken.smallLine, character: charAfterDot(brokenLines[rBroken.smallLine]!, "reshape") };

  return {
    id: "w6",
    fileName: "w6-reshape-flatten.ts",
    text: brokenState.text,
    hovers,
    completion,
    toggle: { label: "W6 reshape product-mismatch toggle (3<->5)", initialState: brokenState, otherState: fixedState },
  };
}

// ---------------------------------------------------------------------------
// Main: build all workloads, write .ts files + tsconfigs + manifest.json.
// ---------------------------------------------------------------------------

function main(): void {
  rmSync(workloadsDir, { recursive: true, force: true });
  mkdirSync(workloadsDir, { recursive: true });

  const workloads = [buildW1(), buildW2(), buildW3(), buildW4(), buildW5(), buildW6()];

  // Self-contained (no "extends") so the language server's auto-discovered
  // project for any workload file is EXACTLY this directory's .ts files plus
  // their real spike/src/*.ts imports — nothing more. See the note at
  // WORKLOAD_COMPILER_OPTIONS above for why "extends" was dropped.
  writeFileSync(join(workloadsDir, "tsconfig.json"), JSON.stringify({ compilerOptions: WORKLOAD_COMPILER_OPTIONS, include: ["*.ts"] }, null, 2) + "\n");

  const manifestEntries: unknown[] = [];
  let totalLines = 0;

  for (const w of workloads) {
    const filePath = join(workloadsDir, w.fileName);
    writeFileSync(filePath, w.text);
    const lineCount = w.text.split("\n").length - 1;
    totalLines += lineCount;

    // Per-file isolated tsconfig for --extendedDiagnostics instantiation
    // counts (mirrors spike/bench/tsconfig.json's single-file pattern) —
    // self-contained + "files" (not "include") for true single-file
    // isolation, since siblings exist in this directory.
    const perFileTsconfigName = `tsconfig.${w.id}.json`;
    writeFileSync(join(workloadsDir, perFileTsconfigName), JSON.stringify({ compilerOptions: WORKLOAD_COMPILER_OPTIONS, files: [w.fileName] }, null, 2) + "\n");

    manifestEntries.push({
      id: w.id,
      fileName: w.fileName,
      lineCount,
      instantiationTsconfig: perFileTsconfigName,
      hovers: w.hovers,
      completion: w.completion,
      toggle: w.toggle,
    });

    console.log(`  wrote ${w.fileName} (${lineCount} lines, ${w.hovers.length} hover positions${w.toggle ? ", 1 toggle" : ""})`);
  }

  writeFileSync(join(workloadsDir, "manifest.json"), JSON.stringify({ workloads: manifestEntries }, null, 2) + "\n");

  console.log(`\nGenerated ${workloads.length} workloads, ${totalLines} total lines, manifest.json written.`);
  const w5 = workloads.find((w) => w.id === "w5")!;
  const w5LineCount = w5.text.split("\n").length - 1;
  if (w5LineCount < 200 || w5LineCount > 400) {
    console.warn(`WARNING: W5 line count ${w5LineCount} is outside the spec's 200-400 LOC target (adjust W5_LAYER_COUNT).`);
  }
}

main();
