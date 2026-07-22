/**
 * `topk` selection bench (docs/op-topk-selection-spec.md v6, Phase 1 — D1-D4).
 * Measures whether a size-bounded max-heap (O(n log k)) beats today's
 * production `topkRuntime` (`spike/src/runtime.ts`, full `order.sort(...)`,
 * O(n log n)) at realistic-to-stress sizes, then mechanically applies the
 * pre-registered decision rule (D6) to the numbers. Pure TS-to-TS — no WASM,
 * no `initCore()`, no `pnpm build:wasm` prerequisite. Run via `pnpm
 * bench:topk`. Informative, not a CI gate (like `bench:scaling`/`bench:chain`
 * etc.) — the one hard rule this file must never break is D3's own: never
 * time a candidate whose output hasn't first been proven bit-identical to
 * today's production `topkRuntime`.
 *
 * Design precedent this file follows (D3, explicitly pointed at by the
 * spec): `spike/bench-core/threaded-crossover.ts`'s `measureRange` — adaptive
 * warmup (30ms or 64 calls, whichever first) then N_REPS batch-timed
 * samples, min/median/max reported, never a single point. See also the
 * coding-kb note "js-wasm-benchmarks-jit-zustand-und-crossover": JIT state is
 * part of the measurement, ranges not single points, warm every candidate
 * before trusting its number.
 *
 * The correctness argument this bench relies on ("a correct heap candidate
 * MUST be bit-identical to the full sort") is `topkRuntime`'s own strict
 * total order (spec section "Die tragende Beobachtung"): the comparator
 * `topkCompareValues(a, b) || (i - j)` never ties on two distinct indices,
 * so there is exactly one correct top-k index set/order for any (data, k) —
 * a claim independently fuzzed 20,017+ times pre-spec and 3,031+ times in
 * Baustein 0's adversarial pass, zero deviations both times.
 *
 * Three candidates, per D2/D7:
 *  - Candidate A ("Sort"): the real, imported, UNCHANGED `topkRuntime` —
 *    today's production behavior, not a copy.
 *  - Candidate B ("Heap"): a size-`k` max-heap of "badness" (root = worst of
 *    the currently held k), built bench-locally per D2's numbered
 *    construction (push while `size < k`; else compare against the root and
 *    replace-on-improvement; final O(k log k) sort with the SAME comparator
 *    expression as Candidate A; `values[i]` re-read from the ORIGINAL `data`
 *    array, never a cached heap value).
 *  - Candidate C ("Hybrid", D7): only built/measured if the decision rule
 *    (D6) actually lands on HYBRID — `k/n <= thresholdRatio ? Heap : Sort`,
 *    instantiated with the ganzraster-safe `t*` from Stage A's merged
 *    verdict, measured as a "Stage B" appended to the SAME script run (no
 *    separate script, no manual step, per D7).
 *
 * `topkCompareValues` is NOT exported from runtime.ts (bench-local
 * literal copy below, same as `topkRuntime`'s own doc comment describes it —
 * see D2 point 4: "Wiederverwendung, keine unabhängig hergeleitete
 * 'äquivalente' Neuformulierung"). It is copied verbatim, not
 * reformulated.
 */
import { topkRuntime } from "../src/runtime.ts";
import { assertDataBitIdentical } from "../tests-runtime/assert-helpers.ts";
import { genData, makeRng } from "../tests-runtime/prng.ts";

// ---------------------------------------------------------------------------
// D3 — the mandated grid. Literal constants, no deviation.
// ---------------------------------------------------------------------------

const N_VALUES = [100, 1_000, 10_000, 100_000, 1_000_000] as const;
const ABSOLUTE_K = [1, 5, 10, 20, 50] as const;
const RATIO_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] as const;

const N_REPS = 7;

// D6 — pre-registered tolerance constants (not a guessed crossover size).
const TOLERANCE_RATIO = 1.15;
const ABS_RELEVANCE_US = 10;

// D7 — Stage-B (Hybrid) dispatch-overhead tolerance (informative only).
const HYBRID_OVERHEAD_REL = 0.05;
const HYBRID_OVERHEAD_ABS_US = 50;

// Arbitrary, distinct, documented seeds (no ASCII meaning, unlike
// threaded-crossover.ts's "CROSSOVR" — kept simple to avoid transcription
// mistakes at this scale of a spec).
const SEED_RUN1 = 0x544f504b00000001n;
const SEED_RUN2 = 0x544f504b00000002n;
const SEED_STAGE_B = 0x544f504b00000003n;

/** D3: `k`-values for a given `n` — union of `ABSOLUTE_K` (kept when `<= n`,
 * every N_VALUES entry is `>= 100` so this never drops anything) and
 * `RATIO_FRACTIONS` resolved to `round(n * frac)` and clamped to `<= n`
 * (defensive; never actually triggers for this grid, see spec D3), then
 * deduplicated and sorted ascending. */
function buildKValues(n: number): number[] {
  const set = new Set<number>();
  for (const k of ABSOLUTE_K) {
    if (k <= n) set.add(k);
  }
  for (const frac of RATIO_FRACTIONS) {
    const k = Math.min(n, Math.round(n * frac));
    if (k >= 1) set.add(k);
  }
  return [...set].sort((a, b) => a - b);
}

interface CellSpec {
  readonly n: number;
  readonly k: number;
  readonly kOverN: number;
}

const GRID: CellSpec[] = [];
for (const n of N_VALUES) {
  for (const k of buildKValues(n)) {
    GRID.push({ n, k, kOverN: k / n });
  }
}

// ---------------------------------------------------------------------------
// Candidates (D2)
// ---------------------------------------------------------------------------

/** Bench-local literal copy of `topkRuntime`'s selection-order comparator
 * (runtime.ts:645-654) — NOT exported from runtime.ts, so this file cannot
 * import it. Copied verbatim, never reformulated (D2 point 4). */
function topkCompareValues(a: number, b: number): number {
  const aNaN = Number.isNaN(a);
  const bNaN = Number.isNaN(b);
  if (aNaN && bNaN) return 0;
  if (aNaN) return -1;
  if (bNaN) return 1;
  if (a > b) return -1;
  if (a < b) return 1;
  return 0;
}

interface Selection {
  readonly values: Float64Array;
  readonly indices: Float64Array;
}

/** Candidate B ("Heap"), D2's numbered construction, O(n log k). Heap
 * invariant: root holds the WORST of the currently held `k` elements (a
 * max-heap of "badness" under the shared total order), so replacement
 * compares the incoming candidate against the root and resifts on
 * improvement. Node representation: two parallel typed arrays (values +
 * source indices), left to the bench author's discretion per D2 — the
 * observable semantics (bit-identity) and complexity class (O(n log k)) are
 * the binding parts, not the representation. */
function heapSelect(data: Float64Array, k: number): Selection {
  const n = data.length;
  if (k === 0) return { values: new Float64Array(0), indices: new Float64Array(0) };

  const heapVal = new Float64Array(k);
  const heapIdx = new Float64Array(k);
  let size = 0;

  function cmp(aVal: number, aIdx: number, bVal: number, bIdx: number): number {
    return topkCompareValues(aVal, bVal) || aIdx - bIdx;
  }

  function siftUp(start: number): void {
    let i = start;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      const pv = heapVal[parent] ?? 0;
      const pi = heapIdx[parent] ?? 0;
      const cv = heapVal[i] ?? 0;
      const ci = heapIdx[i] ?? 0;
      if (cmp(cv, ci, pv, pi) > 0) {
        heapVal[i] = pv;
        heapIdx[i] = pi;
        heapVal[parent] = cv;
        heapIdx[parent] = ci;
        i = parent;
      } else break;
    }
  }

  function siftDown(start: number): void {
    let i = start;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let worst = i;
      if (l < size && cmp(heapVal[l] ?? 0, heapIdx[l] ?? 0, heapVal[worst] ?? 0, heapIdx[worst] ?? 0) > 0) worst = l;
      if (r < size && cmp(heapVal[r] ?? 0, heapIdx[r] ?? 0, heapVal[worst] ?? 0, heapIdx[worst] ?? 0) > 0) worst = r;
      if (worst === i) break;
      const iv = heapVal[i] ?? 0;
      const ii = heapIdx[i] ?? 0;
      heapVal[i] = heapVal[worst] ?? 0;
      heapIdx[i] = heapIdx[worst] ?? 0;
      heapVal[worst] = iv;
      heapIdx[worst] = ii;
      i = worst;
    }
  }

  for (let i = 0; i < n; i++) {
    const v = data[i] ?? 0;
    if (size < k) {
      heapVal[size] = v;
      heapIdx[size] = i;
      siftUp(size);
      size++;
    } else {
      const rv = heapVal[0] ?? 0;
      const ri = heapIdx[0] ?? 0;
      if (cmp(v, i, rv, ri) < 0) {
        heapVal[0] = v;
        heapIdx[0] = i;
        siftDown(0);
      }
    }
  }

  // Final O(k log k) sort — SAME comparator expression as Candidate A,
  // values re-read from the ORIGINAL data array (D2 point 4).
  const order = Array.from({ length: size }, (_, i) => heapIdx[i] ?? 0);
  order.sort((ia, ib) => topkCompareValues(data[ia] ?? 0, data[ib] ?? 0) || ia - ib);

  const values = new Float64Array(size);
  const indices = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const srcIdx = order[i] ?? 0;
    indices[i] = srcIdx;
    values[i] = data[srcIdx] ?? 0;
  }
  return { values, indices };
}

/** Candidate C ("Hybrid", D7/D9 shape) — the trivial dispatch:
 * `k/n <= thresholdRatio ? Heap-path : Sort-path`, sort-path literally
 * Candidate A (unchanged `topkRuntime`). Only built/measured (Stage B) if
 * the merged Stage-A verdict is HYBRID. */
function hybridSelect(shape: readonly number[], data: Float64Array, k: number, thresholdRatio: number): Selection {
  const n = shape[0] ?? 0;
  const useHeap = n > 0 && k / n <= thresholdRatio;
  return useHeap ? heapSelect(data, k) : topkRuntime(shape, data, k);
}

// ---------------------------------------------------------------------------
// Warmup/timing protocol — identical discipline to
// threaded-crossover.ts:107-144's measureRange (D3: "nicht neu erfunden").
// ---------------------------------------------------------------------------

interface Range {
  readonly minMs: number;
  readonly medianMs: number;
  readonly maxMs: number;
}

function measureRange(fn: () => void, reps = N_REPS): Range {
  const warmupStart = performance.now();
  let warmupCalls = 0;
  let elapsed = 0;
  while (elapsed < 30 && warmupCalls < 64) {
    fn();
    warmupCalls++;
    elapsed = performance.now() - warmupStart;
  }
  const estMs = Math.max(elapsed / warmupCalls, 1e-4);
  const batch = Math.min(512, Math.max(1, Math.ceil(2 / estMs)));

  const samples: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    for (let j = 0; j < batch; j++) fn();
    samples.push((performance.now() - t0) / batch);
  }
  samples.sort((a, b) => a - b);
  return {
    minMs: samples[0] ?? 0,
    medianMs: samples[Math.floor(samples.length / 2)] ?? 0,
    maxMs: samples[samples.length - 1] ?? 0,
  };
}

function fmtMs(ms: number): string {
  if (ms >= 1) return `${ms.toFixed(3)}ms`;
  return `${(ms * 1000).toFixed(2)}us`;
}

function fmtUs(us: number): string {
  return `${us >= 0 ? "+" : ""}${us.toFixed(2)}us`;
}

function bitIdentityGate(ref: Selection, cand: Selection, context: string): void {
  assertDataBitIdentical(ref.indices, cand.indices, `${context} (indices)`);
  assertDataBitIdentical(ref.values, cand.values, `${context} (values)`);
}

// ---------------------------------------------------------------------------
// Per-cell measurement (D3): both orders, same operands, order A = [Sort
// warm/measure first, then Heap], order B = [Heap first, then Sort].
// ---------------------------------------------------------------------------

interface OrderReading {
  readonly sort: Range;
  readonly heap: Range;
  readonly ratio: number; // heap.medianMs / sort.medianMs
  readonly deltaUs: number; // (heap.medianMs - sort.medianMs) * 1000, positive = heap slower
}

interface CellRunResult extends CellSpec {
  readonly orderA: OrderReading;
  readonly orderB: OrderReading;
  /** max(orderA.ratio, orderB.ratio) — D3's pessimistic per-order merge. */
  readonly ratio: number;
  /** deltaUs of WHICHEVER order produced `ratio` (paired, traceable to one
   * concrete pair of medians — D3 only defines a merge formula for ratio,
   * not delta, at the order level; pairing keeps the reported numbers
   * reproducible from a single measured order rather than inventing an
   * independent-max rule the spec doesn't state at this level). */
  readonly deltaUs: number;
  readonly orderRelDiff: number;
  readonly orderSensitive: boolean;
}

function measureCell(spec: CellSpec, rng: ReturnType<typeof makeRng>): CellRunResult {
  const { n, k } = spec;
  const shape = [n] as const;
  const data = genData(rng, shape);

  // Bit-identity gate BEFORE any timing (D3, every cell, not sampled).
  const refA = topkRuntime(shape, data, k);
  const candB = heapSelect(data, k);
  bitIdentityGate(refA, candB, `n=${n} k=${k} Heap vs Sort`);

  const sortFn = (): void => {
    topkRuntime(shape, data, k);
  };
  const heapFn = (): void => {
    heapSelect(data, k);
  };

  // Order A: Sort first, then Heap.
  const sortA = measureRange(sortFn);
  const heapA = measureRange(heapFn);
  const ratioA = heapA.medianMs / sortA.medianMs;
  const deltaAus = (heapA.medianMs - sortA.medianMs) * 1000;

  // Order B: Heap first, then Sort.
  const heapB = measureRange(heapFn);
  const sortB = measureRange(sortFn);
  const ratioB = heapB.medianMs / sortB.medianMs;
  const deltaBus = (heapB.medianMs - sortB.medianMs) * 1000;

  const orderA: OrderReading = { sort: sortA, heap: heapA, ratio: ratioA, deltaUs: deltaAus };
  const orderB: OrderReading = { sort: sortB, heap: heapB, ratio: ratioB, deltaUs: deltaBus };

  const chosen = ratioA >= ratioB ? orderA : orderB;
  const relDiff = Math.abs(ratioA - ratioB) / Math.min(ratioA, ratioB);

  return {
    ...spec,
    orderA,
    orderB,
    ratio: chosen.ratio,
    deltaUs: chosen.deltaUs,
    orderRelDiff: relDiff,
    orderSensitive: relDiff > 0.1,
  };
}

// ---------------------------------------------------------------------------
// D6 — the decision rule, applied mechanically.
// ---------------------------------------------------------------------------

interface RatioDeltaCell {
  readonly n: number;
  readonly k: number;
  readonly kOverN: number;
  readonly ratio: number;
  readonly deltaUs: number;
}

function isViolation(c: RatioDeltaCell): boolean {
  return c.ratio > TOLERANCE_RATIO && c.deltaUs > ABS_RELEVANCE_US;
}

function isWin(c: RatioDeltaCell): boolean {
  return c.ratio <= 1 / TOLERANCE_RATIO && c.deltaUs <= -ABS_RELEVANCE_US;
}

function relOnlyViolation(c: RatioDeltaCell): boolean {
  return c.ratio > TOLERANCE_RATIO && !(c.deltaUs > ABS_RELEVANCE_US);
}

function absOnlyViolation(c: RatioDeltaCell): boolean {
  return c.deltaUs > ABS_RELEVANCE_US && !(c.ratio > TOLERANCE_RATIO);
}

function relOnlyWin(c: RatioDeltaCell): boolean {
  return c.ratio <= 1 / TOLERANCE_RATIO && !(c.deltaUs <= -ABS_RELEVANCE_US);
}

function absOnlyWin(c: RatioDeltaCell): boolean {
  return c.deltaUs <= -ABS_RELEVANCE_US && !(c.ratio <= 1 / TOLERANCE_RATIO);
}

type Verdict = "PURE_HEAP" | "HYBRID" | "STATUS_QUO";

interface CandidateCheck {
  readonly t: number;
  readonly safe: boolean;
  readonly violatingCell: RatioDeltaCell | undefined;
}

interface DecisionResult {
  readonly tStar: number;
  readonly checks: CandidateCheck[];
  readonly zoneCells: RatioDeltaCell[];
  readonly winCells: RatioDeltaCell[];
  readonly verdict: Verdict;
}

function buildCandidateSet(kOverNValues: readonly number[]): number[] {
  const set = new Set<number>([0, ...RATIO_FRACTIONS, ...kOverNValues]);
  return [...set].sort((a, b) => a - b); // ascending, for readable transition printing
}

function applyDecisionRule(cells: readonly RatioDeltaCell[], candidatesAsc: readonly number[]): DecisionResult {
  const checks: CandidateCheck[] = candidatesAsc.map((t) => {
    const inZone = cells.filter((c) => c.kOverN <= t);
    const violating = inZone.find(isViolation);
    return { t, safe: violating === undefined, violatingCell: violating };
  });

  let tStar = 0;
  for (const c of checks) {
    if (c.safe && c.t > tStar) tStar = c.t;
  }

  const zoneCells = cells.filter((c) => c.kOverN <= tStar);
  const winCells = zoneCells.filter(isWin);

  let verdict: Verdict;
  if (winCells.length === 0) verdict = "STATUS_QUO";
  else if (tStar === 1.0) verdict = "PURE_HEAP";
  else verdict = "HYBRID";

  return { tStar, checks, zoneCells, winCells, verdict };
}

function verdictLine(d: DecisionResult): string {
  if (d.verdict === "PURE_HEAP") return "PURE HEAP";
  if (d.verdict === "HYBRID") return `HYBRID (threshold=${d.tStar})`;
  return "STATUS QUO (leave as is)";
}

/** Condensed transition print: group consecutive same-safety candidates
 * (ascending t) into runs instead of dumping every one of the ~dozens of
 * candidates individually — the interesting information is WHERE safety
 * flips, not each individual redundant "still safe" reading. Always marks
 * t* explicitly. */
function printDecisionTrace(label: string, d: DecisionResult): void {
  console.log(`\n  [${label}] candidate trace (ascending t, grouped by safety run):`);
  let i = 0;
  while (i < d.checks.length) {
    const start = i;
    const status = d.checks[i]?.safe;
    while (i < d.checks.length && d.checks[i]?.safe === status) i++;
    const end = i - 1;
    const tStart = d.checks[start]?.t ?? 0;
    const tEnd = d.checks[end]?.t ?? 0;
    const count = end - start + 1;
    const marker = tStart <= d.tStar && d.tStar <= tEnd ? "  <-- t* in this run" : "";
    if (status) {
      console.log(`    t in [${tStart.toFixed(4)}, ${tEnd.toFixed(4)}]: SAFE (${count} candidates)${marker}`);
    } else {
      const first = d.checks[start]?.violatingCell;
      const fc = first ? ` first violation: n=${first.n} k=${first.k} (k/n=${first.kOverN.toFixed(4)}, ratio=${first.ratio.toFixed(3)}, delta=${fmtUs(first.deltaUs)})` : "";
      console.log(`    t in [${tStart.toFixed(4)}, ${tEnd.toFixed(4)}]: UNSAFE (${count} candidates)${fc}${marker}`);
    }
  }
  console.log(`  [${label}] t* = ${d.tStar} | zone cells = ${d.zoneCells.length} | win cells in zone = ${d.winCells.length} | verdict = ${verdictLine(d)}`);
}

// ---------------------------------------------------------------------------
// Cell table printing (D3: "Median UND volle Min-Max-Spanne", "beide
// Einzelwerte werden trotzdem vollständig berichtet").
// ---------------------------------------------------------------------------

function classify(c: RatioDeltaCell): string {
  if (isViolation(c)) return "VIOLATION";
  if (isWin(c)) return "WIN";
  if (relOnlyViolation(c)) return "rel-only-viol";
  if (absOnlyViolation(c)) return "abs-only-viol";
  if (relOnlyWin(c)) return "rel-only-win";
  if (absOnlyWin(c)) return "abs-only-win";
  return "neutral";
}

function printCellRow(r: CellRunResult): void {
  const kOverNStr = r.kOverN.toFixed(4).padStart(7);
  console.log(
    `  n=${String(r.n).padStart(7)} k=${String(r.k).padStart(7)} k/n=${kOverNStr} | ` +
      `A: sort=${fmtMs(r.orderA.sort.medianMs).padStart(10)} heap=${fmtMs(r.orderA.heap.medianMs).padStart(10)} ratio=${r.orderA.ratio.toFixed(3).padStart(7)} | ` +
      `B: sort=${fmtMs(r.orderB.sort.medianMs).padStart(10)} heap=${fmtMs(r.orderB.heap.medianMs).padStart(10)} ratio=${r.orderB.ratio.toFixed(3).padStart(7)} | ` +
      `chosen ratio=${r.ratio.toFixed(3).padStart(7)} delta=${fmtUs(r.deltaUs).padStart(10)} ${r.orderSensitive ? "[ORDER-SENSITIVE]" : ""} -> ${classify(r)}`,
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("=== NumType topk selection bench: Sort (Candidate A, production) vs size-bounded Heap (Candidate B) ===");
console.log(`docs/op-topk-selection-spec.md v6 | ${new Date().toISOString()} | grid: ${GRID.length} cells over n in [${N_VALUES.join(", ")}]`);
console.log(`Tolerance: ratio>${TOLERANCE_RATIO} AND delta>${ABS_RELEVANCE_US}us = VIOLATION | ratio<=${(1 / TOLERANCE_RATIO).toFixed(4)} AND delta<=-${ABS_RELEVANCE_US}us = WIN\n`);

const overallStart = performance.now();

function runFullGrid(seed: bigint, label: string): CellRunResult[] {
  console.log(`\n--- ${label}: measuring ${GRID.length} cells (seed=0x${seed.toString(16)}) ---`);
  const rng = makeRng(seed);
  const results: CellRunResult[] = [];
  for (const spec of GRID) {
    const r = measureCell(spec, rng);
    printCellRow(r);
    results.push(r);
  }
  return results;
}

const run1Start = performance.now();
const run1 = runFullGrid(SEED_RUN1, "RUN 1");
const run1Ms = performance.now() - run1Start;

const run2Start = performance.now();
const run2 = runFullGrid(SEED_RUN2, "RUN 2");
const run2Ms = performance.now() - run2Start;

const candidatesAsc = buildCandidateSet(GRID.map((c) => c.kOverN));

const run1Decision = applyDecisionRule(run1, candidatesAsc);
const run2Decision = applyDecisionRule(run2, candidatesAsc);

console.log("\n=== RUN 1 decision trace ===");
printDecisionTrace("RUN 1", run1Decision);
console.log("\n=== RUN 2 decision trace ===");
printDecisionTrace("RUN 2", run2Decision);

console.log(`\nRUN 1 verdict: ${verdictLine(run1Decision)}`);
console.log(`RUN 2 verdict: ${verdictLine(run2Decision)}`);
if (run1Decision.verdict !== run2Decision.verdict) {
  console.log("*** RUN-LEVEL VERDICT DIVERGENCE (reportable per D3 v6) — the pessimistic merge below is what actually decides. ***");
} else {
  console.log("(RUN 1 and RUN 2 individual verdicts agree.)");
}

// D3 v6: pessimistic merge — per cell, independently, the MAX ratio and the
// MAX (more-unfavorable-to-heap) delta across both runs.
interface MergedCell extends RatioDeltaCell {
  readonly run1Ratio: number;
  readonly run2Ratio: number;
  readonly run1DeltaUs: number;
  readonly run2DeltaUs: number;
}

const merged: MergedCell[] = GRID.map((spec, i) => {
  const r1 = run1[i];
  const r2 = run2[i];
  if (!r1 || !r2) throw new Error(`internal: missing run result at grid index ${i}`);
  return {
    n: spec.n,
    k: spec.k,
    kOverN: spec.kOverN,
    ratio: Math.max(r1.ratio, r2.ratio),
    deltaUs: Math.max(r1.deltaUs, r2.deltaUs),
    run1Ratio: r1.ratio,
    run2Ratio: r2.ratio,
    run1DeltaUs: r1.deltaUs,
    run2DeltaUs: r2.deltaUs,
  };
});

console.log("\n--- Pessimistic merge (per cell: max ratio, max delta over both runs) ---");
for (const c of merged) {
  console.log(
    `  n=${String(c.n).padStart(7)} k=${String(c.k).padStart(7)} k/n=${c.kOverN.toFixed(4).padStart(7)} | ` +
      `run1: ratio=${c.run1Ratio.toFixed(3)} delta=${fmtUs(c.run1DeltaUs)} | run2: ratio=${c.run2Ratio.toFixed(3)} delta=${fmtUs(c.run2DeltaUs)} | ` +
      `merged: ratio=${c.ratio.toFixed(3)} delta=${fmtUs(c.deltaUs)} -> ${classify(c)}`,
  );
}

const finalDecision = applyDecisionRule(merged, candidatesAsc);
console.log("\n=== FINAL (pessimistic-merge) decision trace ===");
printDecisionTrace("MERGED", finalDecision);

// Order-sensitivity report (D3: >10% relative divergence between the two
// per-cell orders is itself a reportable finding).
const orderSensitiveRun1 = run1.filter((r) => r.orderSensitive);
const orderSensitiveRun2 = run2.filter((r) => r.orderSensitive);
console.log(`\nOrder-sensitive cells (|ratioA-ratioB|/min > 10%): RUN 1 = ${orderSensitiveRun1.length}/${run1.length}, RUN 2 = ${orderSensitiveRun2.length}/${run2.length}`);
for (const r of [...orderSensitiveRun1, ...orderSensitiveRun2]) {
  console.log(`  n=${r.n} k=${r.k} k/n=${r.kOverN.toFixed(4)} ratioA=${r.orderA.ratio.toFixed(3)} ratioB=${r.orderB.ratio.toFixed(3)} relDiff=${(r.orderRelDiff * 100).toFixed(1)}%`);
}

console.log(`\n${"=".repeat(80)}`);
console.log(`FINAL VERDICT (pessimistic merge of both runs, D6 applied mechanically): ${verdictLine(finalDecision)}`);
console.log(`${"=".repeat(80)}`);

// ---------------------------------------------------------------------------
// D7 — Stage B: only if the FINAL verdict is HYBRID.
// ---------------------------------------------------------------------------

if (finalDecision.verdict === "HYBRID") {
  const thresholdRatio = finalDecision.tStar;
  console.log(`\n--- Stage B: Hybrid candidate (threshold=${thresholdRatio}) — dispatch overhead over ${GRID.length} cells ---`);
  console.log("Methodology: per cell, matchedFn = the branch this cell's k/n actually dispatches to (Heap if k/n<=threshold else Sort);");
  console.log("measured in two orders (matchedFn first / hybridFn first) like Stage A; hybridMedian = pessimistic max over both orders;");
  console.log("the OTHER (non-matched) candidate is also measured once for the min(heap,sort) baseline. Fresh seed, independent of Stage A.");

  const rngB = makeRng(SEED_STAGE_B);
  interface StageBRow {
    readonly n: number;
    readonly k: number;
    readonly kOverN: number;
    readonly branch: "heap" | "sort";
    readonly hybridMedianMs: number;
    readonly heapMedianMs: number;
    readonly sortMedianMs: number;
    readonly overheadUs: number;
    readonly overheadRel: number;
    readonly withinTolerance: boolean;
  }
  const stageBRows: StageBRow[] = [];

  for (const spec of GRID) {
    const { n, k } = spec;
    const shape = [n] as const;
    const data = genData(rngB, shape);

    const ref = topkRuntime(shape, data, k);
    const hybridOut = hybridSelect(shape, data, k, thresholdRatio);
    bitIdentityGate(ref, hybridOut, `n=${n} k=${k} Hybrid vs Sort`);

    const useHeap = k / n <= thresholdRatio;
    const sortFn = (): void => {
      topkRuntime(shape, data, k);
    };
    const heapFn = (): void => {
      heapSelect(data, k);
    };
    const hybridFn = (): void => {
      hybridSelect(shape, data, k, thresholdRatio);
    };
    const matchedFn = useHeap ? heapFn : sortFn;
    const otherFn = useHeap ? sortFn : heapFn;

    // Order A': matched candidate first, then hybrid.
    const matchedA = measureRange(matchedFn);
    const hybridA = measureRange(hybridFn);
    // Order B': hybrid first, then matched candidate.
    const hybridB = measureRange(hybridFn);
    const matchedB = measureRange(matchedFn);
    // The other (non-matched) candidate, single reading — only needed for
    // the min(heap,sort) baseline, not part of the ordered comparison.
    const otherReading = measureRange(otherFn);

    const hybridMedianMs = Math.max(hybridA.medianMs, hybridB.medianMs); // pessimistic: don't understate hybrid
    const matchedMedianMs = Math.min(matchedA.medianMs, matchedB.medianMs); // best-case baseline: don't understate overhead
    const heapMedianMs = useHeap ? matchedMedianMs : otherReading.medianMs;
    const sortMedianMs = useHeap ? otherReading.medianMs : matchedMedianMs;
    const baselineMs = Math.min(heapMedianMs, sortMedianMs);
    const overheadUs = (hybridMedianMs - baselineMs) * 1000;
    const overheadRel = baselineMs > 0 ? overheadUs / (baselineMs * 1000) : Number.POSITIVE_INFINITY;
    const withinTolerance = overheadRel <= HYBRID_OVERHEAD_REL || overheadUs <= HYBRID_OVERHEAD_ABS_US;

    const row: StageBRow = {
      n,
      k,
      kOverN: spec.kOverN,
      branch: useHeap ? "heap" : "sort",
      hybridMedianMs,
      heapMedianMs,
      sortMedianMs,
      overheadUs,
      overheadRel,
      withinTolerance,
    };
    stageBRows.push(row);
    console.log(
      `  n=${String(n).padStart(7)} k=${String(k).padStart(7)} k/n=${spec.kOverN.toFixed(4).padStart(7)} branch=${row.branch.padEnd(4)} | ` +
        `hybrid=${fmtMs(hybridMedianMs).padStart(10)} heap=${fmtMs(heapMedianMs).padStart(10)} sort=${fmtMs(sortMedianMs).padStart(10)} | ` +
        `overhead=${fmtUs(overheadUs).padStart(10)} (${(overheadRel * 100).toFixed(2)}%) ${withinTolerance ? "OK" : "*** EXCEEDS TOLERANCE ***"}`,
    );
  }

  const exceeding = stageBRows.filter((r) => !r.withinTolerance);
  console.log(`\nStage B tolerance (<=${HYBRID_OVERHEAD_REL * 100}% relative OR <=${HYBRID_OVERHEAD_ABS_US}us absolute): ${stageBRows.length - exceeding.length}/${stageBRows.length} cells within tolerance.`);
  if (exceeding.length > 0) {
    console.log(`${exceeding.length} cells exceed the informative tolerance (not a blocker per D7, but a reportable finding):`);
    for (const r of exceeding) {
      console.log(`  n=${r.n} k=${r.k} k/n=${r.kOverN.toFixed(4)} branch=${r.branch} overhead=${fmtUs(r.overheadUs)} (${(r.overheadRel * 100).toFixed(2)}%)`);
    }
  }
} else {
  console.log("\n(Final verdict is not HYBRID — Stage B skipped per D7, no threshold to instantiate a hybrid candidate with.)");
}

const overallMs = performance.now() - overallStart;
console.log(`\nWall-clock: RUN 1 = ${(run1Ms / 1000).toFixed(1)}s, RUN 2 = ${(run2Ms / 1000).toFixed(1)}s, total incl. Stage B = ${(overallMs / 1000).toFixed(1)}s.`);
console.log("Done.");
