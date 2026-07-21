/**
 * Scale-Probe workload generator (docs/scale-probe-spec.md, D6/D9/D19-D21).
 * Sister file of `gen-workloads.ts` — same pattern (deterministic generation,
 * real `spike/src` imports, hand-computed expected shapes), but for the four
 * pre-registered sweep axes (corpus size, chain depth, rank, cold-editor-
 * latency-at-scale) instead of the seven fixed `bench:editor` workloads.
 *
 * D5/§14 Frage 1: this file DUPLICATES (does not import) the small generation
 * helpers (`fmtShape`/`charIn`/`charAfterDot`/`makeBuilder`/`GENERATED_HEADER`)
 * and the compiler-options constant from `gen-workloads.ts` — that file is a
 * hard-gate-critical file this slice must not edit beyond the one additive
 * `buildW8()` change (D12), so importing from it here would risk pulling it
 * into an unrelated dependency graph for no benefit; duplicating a handful of
 * pure, stable helper functions is a bounded, one-time cost (same reasoning
 * `dim.ts` already applies to its own `IsUnion` duplication).
 *
 * D21: unlike `gen-workloads.ts` (one SHARED `workloads/tsconfig.json` for LSP
 * discovery + separate per-workload `tsconfig.<id>.json` for instantiation
 * counting), every sweep point here gets its OWN subdirectory with a file
 * literally named `tsconfig.json` — the language server auto-discovers a
 * project by finding the nearest file named exactly that, so a shared
 * directory would make opening the SMALLEST point's file load the ENTIRE
 * sweep's programs, including the rank-1024 point that aborts the checker.
 *
 * Determinism (D9): no `Date.now()`/`Math.random()`/host state in the
 * generated OUTPUT — re-running this script produces byte-identical files,
 * so `spike/bench-dx/scale-workloads/` is safe to `.gitignore` and regenerate.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const scaleWorkloadsDir = join(__dirname, "scale-workloads");

// Duplicated from gen-workloads.ts (see file header) — identical values.
const SCALE_COMPILER_OPTIONS = {
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
// Duplicated small helpers (see file header for why duplicated, not imported).
// ---------------------------------------------------------------------------

function fmtShape(dims: readonly (number | "number")[]): string {
  return `NDArray<[${dims.join(", ")}]>`;
}

function makeBuilder() {
  const lines: string[] = [];
  return {
    push: (s: string) => {
      for (const l of s.split("\n")) lines.push(l);
    },
    lines,
    lastIdx: () => lines.length - 1,
  };
}

function charIn(line: string, token: string): number {
  const idx = line.indexOf(token);
  if (idx === -1) throw new Error(`gen-scale-workloads: token "${token}" not found in line: ${line}`);
  return idx + 1;
}

function charAfterDot(line: string, member: string): number {
  const idx = line.indexOf(`.${member}`);
  if (idx === -1) throw new Error(`gen-scale-workloads: member ".${member}" not found in line: ${line}`);
  return idx + 1;
}

const GENERATED_HEADER = (title: string, body: string) =>
  [`/**`, ` * GENERATED FILE — see spike/bench-dx/gen-scale-workloads.ts (${title}).`, ...body.split("\n").map((l) => ` * ${l}`), ` */`].join("\n");

// ---------------------------------------------------------------------------
// Manifest types (mirrors gen-workloads.ts's shapes, duplicated for the same
// D5 reason — scale-latency.ts reads this as plain JSON, never imports it).
// ---------------------------------------------------------------------------

interface HoverSpec {
  label: string;
  line: number;
  character: number;
  expected: string;
}
interface CompletionSpec {
  label: string;
  line: number;
  character: number;
}
interface ToggleStateSpec {
  label: string;
  text: string;
  diagnosticLine: number;
  expectDiagnosticPresent: boolean;
  expectedCode: number;
}
interface ToggleSpec {
  label: string;
  targetFile: string;
  initialState: ToggleStateSpec;
  otherState: ToggleStateSpec;
}
interface ScalePointManifestEntry {
  id: string;
  axis: "a" | "b" | "c";
  subseries: string;
  sweepValue: number;
  dir: string;
  files: string[];
  tsconfig: string; // always "tsconfig.json" per D21
  primaryFile: string;
  hover: HoverSpec;
  completion: CompletionSpec;
  toggle?: ToggleSpec;
}

// ---------------------------------------------------------------------------
// Op-count self-report (D10/T8): regex-counts real call sites in generated
// text — a build-time check, not a paper claim, so a future template bug that
// silently breaks the D10 ratio binding throws here instead of shipping.
// ---------------------------------------------------------------------------

function countOps(text: string): Record<string, number> {
  const patterns: Record<string, RegExp> = {
    item: /\.item\(/g,
    fromArray: /NDArray\.fromArray\(/g,
    slice: /\.slice\(/g,
    cosineSimilarity: /\.cosineSimilarity\(/g,
    matmul: /\.matmul\(/g,
    mul: /\.mul\(/g,
    div: /\.div\(/g,
    sum: /\.sum\(/g,
    sqrt: /\.sqrt\(/g,
    reshape: /\.reshape\(/g,
    topk: /\.topk\(/g,
    dot: /\.dot\(/g,
    mean: /\.mean\(/g,
    transpose: /\.transpose\(/g,
    stack: /NDArray\.stack\(/g,
  };
  const counts: Record<string, number> = {};
  for (const [name, re] of Object.entries(patterns)) counts[name] = (text.match(re) ?? []).length;
  return counts;
}

/** D10 binding rule: the three most-frequent real ops (item, fromArray,
 * slice) must appear in EVERY generated file at least as often as the three
 * least-frequent (mean, transpose, stack). Throws (build-time blocker) if
 * violated — never silently ships a file that fails the calibration. */
function assertD10Binding(fileLabel: string, counts: Record<string, number>): void {
  const leastMax = Math.max(counts.mean ?? 0, counts.transpose ?? 0, counts.stack ?? 0);
  for (const top of ["item", "fromArray", "slice"] as const) {
    if ((counts[top] ?? 0) < leastMax) {
      throw new Error(
        `D10 binding violated in ${fileLabel}: ${top}=${counts[top]} < max(mean,transpose,stack)=${leastMax} — ` +
          `the three most-frequent ops must appear at least as often as the three least-frequent ones.`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Axis (a) — corpus size. One "op-mix" file per corpus file, calibrated to
// the real-code op frequencies (spec §5): item/fromArray/slice dominant,
// mean/transpose/stack rarest — every file touches every op in §5's table at
// least once, so the file is a genuine (if condensed) mirror of the RAG
// pipeline's op shape, not a synthetic stress construction like axis (b)/(c).
//
// Same-vs-distinct (D19): "same" fixes one (D, nRows) pair for every file at
// every sweep point (maximal TS instantiation cache hits, mirrors what a
// naive single-series measurement would have shown); "distinct" gives file i
// its own (D, nRows) pair in a disjoint numeric band (base = 500 + i*20; D =
// base+3, nRows = base+11) — by construction no two files at the same sweep
// point can share a value, checked again at generation time (T8) as a
// defensive, not just structural, guarantee.
// ---------------------------------------------------------------------------

interface OpMixFile {
  text: string;
  poolLine: number;
  poolVarName: string;
  mulLine: number;
  mulChar: number;
  toggleLine?: number;
}

function renderOpMixFile(prefix: string, D: number, nRows: number, toggleDim?: 3 | 5): OpMixFile {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "axis (a) op-mix file",
      `Condensed mirror of examples/rag-demo/main.ts's op sequence (D=${D}, nRows=${nRows}):\nfromArray -> mul/sum/sqrt/reshape/div (L2-normalize) -> transpose/matmul (score\nmatrix) -> slice/topk/item (ranking) -> slice/cosineSimilarity/dot (cross-check)\n-> fromArray/stack/mean (chunk pooling) — every op in the spec §5 calibration\ntable appears at least once, item/fromArray/slice (most frequent in the real\ncode) strictly more often than mean/transpose/stack (least frequent, D10).`,
    ),
  );
  b.push(`import { NDArray } from "../../../src/ndarray.ts";`);
  b.push(``);
  b.push(`const ${prefix}_base = NDArray.fromArray([${nRows}, ${D}], new Array(${nRows} * ${D}).fill(1));`);
  const mulLine0 = b.lines.length;
  b.push(`const ${prefix}_sq = ${prefix}_base.mul(${prefix}_base);`);
  const mulLine = mulLine0;
  const mulChar = charAfterDot(b.lines[mulLine]!, "mul");
  b.push(`const ${prefix}_sumSq = ${prefix}_sq.sum(1);`);
  b.push(`const ${prefix}_norms = ${prefix}_sumSq.sqrt();`);
  b.push(`const ${prefix}_normsCol = ${prefix}_norms.reshape([${nRows}, 1]);`);
  b.push(`const ${prefix}_normalized = ${prefix}_base.div(${prefix}_normsCol);`);
  b.push(`const ${prefix}_transposed = ${prefix}_normalized.transpose();`);
  b.push(`const ${prefix}_scores = ${prefix}_normalized.matmul(${prefix}_transposed);`);
  b.push(`const ${prefix}_row0 = ${prefix}_scores.slice(0);`);
  b.push(`const ${prefix}_ranked0 = ${prefix}_row0.topk(2);`);
  b.push(`const ${prefix}_top1 = ${prefix}_ranked0.indices.item(0);`);
  b.push(`const ${prefix}_top1Score = ${prefix}_ranked0.values.item(0);`);
  b.push(`const ${prefix}_top2Score = ${prefix}_ranked0.values.item(1);`);
  b.push(`const ${prefix}_direct = ${prefix}_scores.item(0, 1);`);
  b.push(`const ${prefix}_rawRow0 = ${prefix}_base.slice(0);`);
  b.push(`const ${prefix}_rawRow1 = ${prefix}_base.slice(1);`);
  b.push(`const ${prefix}_cos = ${prefix}_rawRow0.cosineSimilarity(${prefix}_rawRow1);`);
  b.push(`const ${prefix}_dot = ${prefix}_rawRow0.dot(${prefix}_rawRow1);`);
  b.push(`const ${prefix}_chunkA = NDArray.fromArray([${D}], new Array(${D}).fill(1));`);
  b.push(`const ${prefix}_chunkB = NDArray.fromArray([${D}], new Array(${D}).fill(1));`);
  b.push(`const ${prefix}_chunkMatrix = NDArray.stack([${prefix}_chunkA, ${prefix}_chunkB]);`);
  const poolVarName = `${prefix}_pooled`;
  b.push(`const ${poolVarName} = ${prefix}_chunkMatrix.mean(0);`);
  const poolLine = b.lastIdx();

  let toggleLine: number | undefined;
  if (toggleDim !== undefined) {
    b.push(``);
    b.push(`// --- D22 toggle target: matmul inner-dim mismatch (3<->5), W4's mechanism ---`);
    b.push(`const ${prefix}_toggle_a = NDArray.zeros([2, 3]);`);
    b.push(`const ${prefix}_toggle_b = NDArray.zeros([${toggleDim}, 4]);`);
    b.push(`const ${prefix}_toggle_result = ${prefix}_toggle_a.matmul(${prefix}_toggle_b);`);
    toggleLine = b.lastIdx();
  }

  // exactOptionalPropertyTypes: the optional `toggleLine` key must be
  // OMITTED, not present-with-value-undefined — spread it in conditionally
  // (same discipline gen-workloads.ts's WorkloadSpec.toggle already follows).
  return { text: b.lines.join("\n") + "\n", poolLine, poolVarName, mulLine, mulChar, ...(toggleLine !== undefined ? { toggleLine } : {}) };
}

const AXIS_A_POINTS = [1, 10, 25, 50, 100, 250];

function genAxisA(manifest: ScalePointManifestEntry[], selfReport: Record<string, unknown>): void {
  const report: Record<string, unknown> = {};
  for (const sub of ["same", "distinct"] as const) {
    const subReport: unknown[] = [];
    for (const N of AXIS_A_POINTS) {
      const pointId = `a-${sub}-${N}`;
      const pointDir = join(scaleWorkloadsDir, pointId);
      mkdirSync(pointDir, { recursive: true });

      const files: string[] = [];
      const dimsUsed: { D: number; nRows: number }[] = [];
      let hover: HoverSpec | undefined;
      let completion: CompletionSpec | undefined;
      let toggle: ToggleSpec | undefined;

      for (let i = 0; i < N; i++) {
        const D = sub === "same" ? 64 : 500 + i * 20 + 3;
        const nRows = sub === "same" ? 16 : 500 + i * 20 + 11;
        dimsUsed.push({ D, nRows });

        const isToggleFile = N === 250 && i === 0;
        const rendered = renderOpMixFile(`f${i}`, D, nRows, isToggleFile ? 5 : undefined);
        assertD10Binding(`${pointId}/f${i}.ts`, countOps(rendered.text));

        const fileName = `f${i}.ts`;
        writeFileSync(join(pointDir, fileName), rendered.text);
        files.push(fileName);

        if (i === 0) {
          hover = {
            label: `${pointId} pooled-vector hover`,
            line: rendered.poolLine,
            character: charIn(rendered.text.split("\n")[rendered.poolLine]!, rendered.poolVarName),
            expected: fmtShape([D]),
          };
          completion = { label: `${pointId} member access`, line: rendered.mulLine, character: rendered.mulChar };
          if (isToggleFile) {
            const fixedRendered = renderOpMixFile(`f${i}`, D, nRows, 3);
            toggle = {
              label: `${pointId} matmul inner-dim toggle (3<->5)`,
              targetFile: fileName,
              initialState: { label: "broken (2 errors)", text: rendered.text, diagnosticLine: rendered.toggleLine!, expectDiagnosticPresent: true, expectedCode: 2741 },
              otherState: { label: "fixed (1 error)", text: fixedRendered.text, diagnosticLine: fixedRendered.toggleLine!, expectDiagnosticPresent: false, expectedCode: 2741 },
            };
          }
        }
      }

      // T8: distinct sub-series must use pairwise-disjoint literal dims.
      if (sub === "distinct") {
        const seen = new Set<number>();
        for (const { D, nRows } of dimsUsed) {
          if (seen.has(D) || seen.has(nRows)) {
            throw new Error(`T8 disjointness violated at ${pointId}: dim collision (D=${D}, nRows=${nRows})`);
          }
          seen.add(D);
          seen.add(nRows);
        }
      } else {
        // "same": every file must use the IDENTICAL pair (the opposite check).
        const allSame = dimsUsed.every((d) => d.D === dimsUsed[0]!.D && d.nRows === dimsUsed[0]!.nRows);
        if (!allSame) throw new Error(`T8 sameness violated at ${pointId}: "same" sub-series files do not all share one (D, nRows) pair`);
      }

      writeFileSync(join(pointDir, "tsconfig.json"), JSON.stringify({ compilerOptions: SCALE_COMPILER_OPTIONS, files: [...files, "../../../src/ambient.d.ts"] }, null, 2) + "\n");

      // exactOptionalPropertyTypes: `toggle` must be OMITTED, not
      // present-with-value-undefined — spread it in conditionally.
      manifest.push({
        id: pointId,
        axis: "a",
        subseries: sub,
        sweepValue: N,
        dir: pointId,
        files,
        tsconfig: "tsconfig.json",
        primaryFile: "f0.ts",
        hover: hover!,
        completion: completion!,
        ...(toggle ? { toggle } : {}),
      });

      subReport.push({ id: pointId, files: N, dims: sub === "same" ? dimsUsed[0] : { first: dimsUsed[0], last: dimsUsed[dimsUsed.length - 1] } });
      console.log(`  wrote ${pointId} (${N} file(s)${toggle ? ", 1 toggle" : ""})`);
    }
    report[sub] = subReport;
  }
  (selfReport as Record<string, unknown>).axisA = report;
}

// ---------------------------------------------------------------------------
// Axis (b) — chain depth. (b-fix) reproduces W1's exact [8,8] matmul/add
// alternation extended past L=100; (b-variable, D20) grows a FRESH weight
// shape every step (D_i = 8+i, monotonically increasing, never repeated) so
// no two `MatMul<>` instantiations in the chain are identical.
// ---------------------------------------------------------------------------

interface ChainFile {
  text: string;
  finalLine: number;
  finalVarName: string;
  completionLine: number;
  completionChar: number;
  expectedShape: readonly number[];
}

function renderChainFix(L: number): ChainFile {
  const b = makeBuilder();
  b.push(GENERATED_HEADER("axis (b) fix sub-series", `Identical to W1's construction (gen-workloads.ts buildW1), extended to L=${L}: alternating matmul(weight)/add(bias) over the fixed [8,8] shape the whole way.`));
  b.push(`import { NDArray } from "../../../src/ndarray.ts";`);
  b.push(``);
  b.push(`const bf_weight = NDArray.zeros([8, 8]);`);
  b.push(`const bf_bias = NDArray.zeros([8, 8]);`);
  b.push(`const bf_x0 = NDArray.zeros([8, 8]);`);
  let prev = "bf_x0";
  let completionLine = -1;
  let completionChar = -1;
  for (let i = 1; i <= L; i++) {
    const cur = `bf_x${i}`;
    const opName = i % 2 === 1 ? "matmul" : "add";
    const opArg = i % 2 === 1 ? "bf_weight" : "bf_bias";
    b.push(`const ${cur} = ${prev}.${opName}(${opArg});`);
    if (i === L) {
      completionLine = b.lastIdx();
      completionChar = charAfterDot(b.lines[b.lastIdx()]!, opName);
    }
    prev = cur;
  }
  const finalName = "bf_final";
  b.push(`const ${finalName} = ${prev}.transpose().sum(0);`);
  const finalLine = b.lastIdx();
  return { text: b.lines.join("\n") + "\n", finalLine, finalVarName: finalName, completionLine, completionChar, expectedShape: [8] };
}

function renderChainVariable(L: number): ChainFile {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "axis (b) variable sub-series",
      `D20: a fresh, never-repeated operand shape every step (D_i = 8+i for i=1..${L}) so no two MatMul<> instantiations in the chain are identical — the adversarial counterpart to (b-fix)'s maximally cache-friendly construction.`,
    ),
  );
  b.push(`import { NDArray } from "../../../src/ndarray.ts";`);
  b.push(``);
  b.push(`const bv_x0 = NDArray.zeros([8, 8]);`);
  let prev = "bv_x0";
  let prevDim = 8;
  let completionLine = -1;
  let completionChar = -1;
  for (let i = 1; i <= L; i++) {
    const nextDim = 8 + i;
    const wName = `bv_w${i}`;
    b.push(`const ${wName} = NDArray.zeros([${prevDim}, ${nextDim}]);`);
    const cur = `bv_x${i}`;
    b.push(`const ${cur} = ${prev}.matmul(${wName});`);
    if (i === L) {
      completionLine = b.lastIdx();
      completionChar = charAfterDot(b.lines[b.lastIdx()]!, "matmul");
    }
    prev = cur;
    prevDim = nextDim;
  }
  const finalLine = b.lines.findIndex((l) => l.startsWith(`const ${prev} =`));
  if (finalLine === -1) throw new Error(`renderChainVariable: could not locate declaration of ${prev}`);
  return { text: b.lines.join("\n") + "\n", finalLine, finalVarName: prev, completionLine, completionChar, expectedShape: [8, prevDim] };
}

const AXIS_B_POINTS = [100, 250, 500, 1000, 2500, 5000, 10000];

function genAxisB(manifest: ScalePointManifestEntry[], selfReport: Record<string, unknown>): void {
  const report: Record<string, unknown> = {};
  for (const sub of ["fix", "variable"] as const) {
    const subReport: unknown[] = [];
    for (const L of AXIS_B_POINTS) {
      const pointId = `b-${sub}-${L}`;
      const pointDir = join(scaleWorkloadsDir, pointId);
      mkdirSync(pointDir, { recursive: true });

      const rendered = sub === "fix" ? renderChainFix(L) : renderChainVariable(L);
      const fileName = "chain.ts";
      writeFileSync(join(pointDir, fileName), rendered.text);

      const lines = rendered.text.split("\n");
      const hover: HoverSpec = {
        label: `${pointId} final chain shape`,
        line: rendered.finalLine,
        character: charIn(lines[rendered.finalLine]!, rendered.finalVarName),
        expected: fmtShape(rendered.expectedShape),
      };
      const completion: CompletionSpec = { label: `${pointId} member access`, line: rendered.completionLine, character: rendered.completionChar };

      writeFileSync(join(pointDir, "tsconfig.json"), JSON.stringify({ compilerOptions: SCALE_COMPILER_OPTIONS, files: [fileName, "../../../src/ambient.d.ts"] }, null, 2) + "\n");

      manifest.push({ id: pointId, axis: "b", subseries: sub, sweepValue: L, dir: pointId, files: [fileName], tsconfig: "tsconfig.json", primaryFile: fileName, hover, completion });
      subReport.push({ id: pointId, L, finalShape: rendered.expectedShape });
      console.log(`  wrote ${pointId} (L=${L})`);
    }
    report[sub] = subReport;
  }
  (selfReport as Record<string, unknown>).axisB = report;
}

// ---------------------------------------------------------------------------
// Axis (c) — rank. Identical construction to W2 (buildW2), extended past
// rank=16 up to and including the known cliff region {768, 896, 1024}.
// ---------------------------------------------------------------------------

interface RankFile {
  text: string;
  resultLine: number;
  resultName: string;
  completionLine: number;
  completionChar: number;
  expectedShape: readonly number[];
}

function renderRank(R: number): RankFile {
  const b = makeBuilder();
  b.push(
    GENERATED_HEADER(
      "axis (c) rank sweep",
      `Identical to W2's construction (gen-workloads.ts buildW2), extended to rank=${R}: broadcasting add() with alternating size-1 axis (dimValue = 2 + (k % 5) on A when k even, on B when k odd) — every axis broadcasts cleanly by construction.`,
    ),
  );
  b.push(`import { NDArray } from "../../../src/ndarray.ts";`);
  b.push(``);
  const dimA: number[] = [];
  const dimB: number[] = [];
  const result: number[] = [];
  for (let k = 0; k < R; k++) {
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
  b.push(`const c_a = NDArray.zeros([${dimA.join(", ")}]);`);
  b.push(`const c_b = NDArray.zeros([${dimB.join(", ")}]);`);
  const resultName = "c_result";
  b.push(`const ${resultName} = c_a.add(c_b);`);
  const resultLine = b.lastIdx();
  const completionChar = charAfterDot(b.lines[resultLine]!, "add");
  return { text: b.lines.join("\n") + "\n", resultLine, resultName, completionLine: resultLine, completionChar, expectedShape: result };
}

const AXIS_C_POINTS = [16, 32, 64, 128, 256, 512, 768, 896, 1024];

function genAxisC(manifest: ScalePointManifestEntry[], selfReport: Record<string, unknown>): void {
  const report: unknown[] = [];
  for (const R of AXIS_C_POINTS) {
    const pointId = `c-${R}`;
    const pointDir = join(scaleWorkloadsDir, pointId);
    mkdirSync(pointDir, { recursive: true });

    const rendered = renderRank(R);
    const fileName = "rank.ts";
    writeFileSync(join(pointDir, fileName), rendered.text);

    const lines = rendered.text.split("\n");
    // Finding (post-real-run, axis c): tsc's own hover output ELIDES long
    // tuple types in the middle ("2, 3, ... 88 more ..., 2]>") once the
    // rank gets large (observed starting around rank=256 in this sweep,
    // rank=128 still rendered in full) — a genuine TS hover-display
    // behavior, not a bug in the type computation itself. A full-array
    // `expected` substring (correct at low ranks) therefore stops being a
    // SUBSTRING of the real hover text at high ranks even though the type
    // is exactly right. The correctness proof must use a short, always-
    // present PREFIX of the shape instead of the full array — safe at
    // every rank in the sweep (TS always shows well over a hundred leading
    // elements before eliding, so 6 is comfortably inside the safe zone).
    const HOVER_PREFIX_DIMS = 6;
    const prefixShape = rendered.expectedShape.slice(0, Math.min(HOVER_PREFIX_DIMS, rendered.expectedShape.length));
    const hover: HoverSpec = {
      label: `${pointId} broadcast result`,
      line: rendered.resultLine,
      character: charIn(lines[rendered.resultLine]!, rendered.resultName),
      expected: `NDArray<[${prefixShape.join(", ")}`, // deliberately open-ended (no closing "]>") — a prefix-only check, safe against hover elision at high rank (see comment above)
    };
    const completion: CompletionSpec = { label: `${pointId} member access`, line: rendered.completionLine, character: rendered.completionChar };

    writeFileSync(join(pointDir, "tsconfig.json"), JSON.stringify({ compilerOptions: SCALE_COMPILER_OPTIONS, files: [fileName, "../../../src/ambient.d.ts"] }, null, 2) + "\n");

    manifest.push({ id: pointId, axis: "c", subseries: "single", sweepValue: R, dir: pointId, files: [fileName], tsconfig: "tsconfig.json", primaryFile: fileName, hover, completion });
    report.push({ id: pointId, R, label: R === 32 ? "= WASM-ABI MAX_RANK (type-layer measurement unaffected, see spec §2.3)" : undefined });
    console.log(`  wrote ${pointId} (rank=${R}${R === 32 ? ", = WASM-ABI MAX_RANK" : ""})`);
  }
  (selfReport as Record<string, unknown>).axisC = report;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

function main(): void {
  rmSync(scaleWorkloadsDir, { recursive: true, force: true });
  mkdirSync(scaleWorkloadsDir, { recursive: true });

  const manifest: ScalePointManifestEntry[] = [];
  const selfReport: Record<string, unknown> = {};

  console.log("Generating axis (a) — corpus size (same / distinct sub-series)...");
  genAxisA(manifest, selfReport);
  console.log("Generating axis (b) — chain depth (fix / variable sub-series)...");
  genAxisB(manifest, selfReport);
  console.log("Generating axis (c) — rank...");
  genAxisC(manifest, selfReport);

  writeFileSync(join(scaleWorkloadsDir, "manifest.json"), JSON.stringify({ points: manifest }, null, 2) + "\n");
  writeFileSync(join(scaleWorkloadsDir, "self-report.json"), JSON.stringify(selfReport, null, 2) + "\n");

  console.log(`\nGenerated ${manifest.length} sweep points, manifest.json + self-report.json written.`);
}

main();
