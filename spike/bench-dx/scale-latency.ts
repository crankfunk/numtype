/**
 * Scale-Probe sweep runner (docs/scale-probe-spec.md, D5/D8/D23/G3/G4).
 *
 * D5: DUPLICATES (does not import/extract) the hand-rolled JSON-RPC/LSP
 * client and stats/narrowing helpers from `editor-latency.ts` — that file
 * carries the exact-match `bench:editor` hard gate (w1-w7 instantiation
 * pins), and this slice must leave it with ZERO non-additive changes; a
 * shared module would make its diff non-empty for no benefit to the sweep
 * itself. See `gen-scale-workloads.ts`'s header for the identical reasoning
 * (and the project's own `dim.ts`/`IsUnion` precedent for this pattern).
 *
 * D23 — cliff isolation is a MECHANISM here, not just an intention: every
 * sweep point is measured inside its own try/catch, cliff classification
 * happens from tsc's own diagnostic output BEFORE any LSP hover is
 * attempted (never from a caught LSP exception), and a point that fails for
 * a DIFFERENT reason (OOM/signal-kill, an unexpected non-cliff type error)
 * is filed under its own "other-failure" category, never counted as a cliff.
 * A failed point never aborts the sweep — the loop always continues to the
 * next point, across ALL axes.
 *
 * G3: this runner carries NO hard pass/fail threshold (unlike
 * `editor-latency.ts`'s `enforceHardGate`) — the numbers themselves are the
 * result. Correctness gates (hover text must match) still throw immediately
 * within a point's own measurement (never time/accept a wrong result), but
 * that throw is caught by the point's own try/catch so the sweep keeps going.
 *
 * D8: prints per-axis tables (G4 — never mixes axis populations) and writes
 * `scale-workloads/results.json` (gitignored) with every raw sample,
 * including memory usage from the `--extendedDiagnostics` footer.
 */
import { execFileSync, spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const scaleWorkloadsDir = join(__dirname, "scale-workloads");
const repoRoot = join(__dirname, "..", "..");

const REQUEST_TIMEOUT_MS = 30_000;
const WARMUP_SAMPLES = 3;
const TIMED_SAMPLES = 20;

// ---------------------------------------------------------------------------
// Manifest types (duplicated from gen-scale-workloads.ts — plain JSON at the
// process boundary, never imported, same reasoning as editor-latency.ts's
// own duplicated manifest types).
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
  tsconfig: string;
  primaryFile: string;
  hover: HoverSpec;
  completion: CompletionSpec;
  toggle?: ToggleSpec;
}
interface ScaleManifest {
  points: ScalePointManifestEntry[];
}

// ---------------------------------------------------------------------------
// Stats helpers — duplicated from editor-latency.ts (D5).
// ---------------------------------------------------------------------------

interface Stats {
  median: number;
  min: number;
  max: number;
  n: number;
}

function statsOf(samples: readonly number[]): Stats {
  if (samples.length === 0) throw new Error("statsOf: no samples");
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2 : (sorted[mid] ?? 0);
  return { median, min: sorted[0] ?? 0, max: sorted[sorted.length - 1] ?? 0, n: sorted.length };
}

function fmtMs(ms: number): string {
  return `${ms.toFixed(2)}ms`;
}

// ---------------------------------------------------------------------------
// tsc exe resolution — duplicated from editor-latency.ts (D5).
// ---------------------------------------------------------------------------

async function resolveTscExe(): Promise<string> {
  const getExePathPath = join(repoRoot, "node_modules", "typescript", "lib", "getExePath.js");
  const mod = (await import(`file://${getExePathPath}`)) as { default: () => string };
  return mod.default();
}

// ---------------------------------------------------------------------------
// Hand-rolled JSON-RPC/LSP client over stdio — duplicated from
// editor-latency.ts (D5), byte-for-byte identical logic.
// ---------------------------------------------------------------------------

type NotificationListener = (method: string, params: unknown) => void;

interface LspClient {
  request: (method: string, params: unknown) => Promise<unknown>;
  notify: (method: string, params: unknown) => void;
  onNotification: (listener: NotificationListener) => void;
  shutdown: () => Promise<void>;
}

function createLspClient(exe: string): LspClient {
  const child = spawn(exe, ["--lsp", "--stdio"], { stdio: ["pipe", "pipe", "pipe"] });
  let buf: Buffer = Buffer.from("");
  let nextId = 1;
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  const notificationListeners: NotificationListener[] = [];
  let exited = false;

  function send(obj: unknown): void {
    const json = JSON.stringify(obj);
    child.stdin.write(`Content-Length: ${Buffer.byteLength(json, "utf8")}\r\n\r\n${json}`);
  }

  function request(method: string, params: unknown): Promise<unknown> {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`LSP request TIMEOUT after ${REQUEST_TIMEOUT_MS}ms: method="${method}" params=${JSON.stringify(params)}`));
      }, REQUEST_TIMEOUT_MS);
      pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      send({ jsonrpc: "2.0", id, method, params });
    });
  }

  function notify(method: string, params: unknown): void {
    send({ jsonrpc: "2.0", method, params });
  }

  function onNotification(listener: NotificationListener): void {
    notificationListeners.push(listener);
  }

  child.stdout.on("data", (chunk: Buffer) => {
    buf = Buffer.concat([buf, chunk]);
    while (true) {
      const headerEnd = buf.indexOf("\r\n\r\n");
      if (headerEnd === -1) break;
      const header = buf.subarray(0, headerEnd).toString("utf8");
      const m = /Content-Length: (\d+)/i.exec(header);
      if (!m || !m[1]) break;
      const len = parseInt(m[1], 10);
      const bodyStart = headerEnd + 4;
      if (buf.length < bodyStart + len) break;
      const body = buf.subarray(bodyStart, bodyStart + len).toString("utf8");
      buf = buf.subarray(bodyStart + len);
      const msg = JSON.parse(body) as { id?: number | string; method?: string; params?: unknown; result?: unknown; error?: unknown };

      if (msg.method === undefined && msg.id !== undefined) {
        if (typeof msg.id === "number") {
          const p = pending.get(msg.id);
          if (p) {
            pending.delete(msg.id);
            if (msg.error) p.reject(new Error(`LSP error response: ${JSON.stringify(msg.error)}`));
            else p.resolve(msg.result);
          }
        }
      } else if (msg.method !== undefined && msg.id !== undefined) {
        let result: unknown = null;
        if (msg.method === "workspace/configuration") {
          const items = (msg.params as { items?: unknown[] })?.items ?? [];
          result = items.map(() => null);
        }
        send({ jsonrpc: "2.0", id: msg.id, result });
      } else if (msg.method !== undefined && msg.id === undefined) {
        const method = msg.method;
        for (const l of notificationListeners) l(method, msg.params);
      }
    }
  });

  child.stderr.on("data", (chunk: Buffer) => {
    console.error(`[tsc-lsp stderr] ${chunk.toString("utf8")}`);
  });

  child.on("error", (err: Error) => {
    for (const [, p] of pending) p.reject(new Error(`tsc --lsp process error: ${err.message}`));
    pending.clear();
  });

  child.on("exit", (code, signal) => {
    exited = true;
    for (const [, p] of pending) p.reject(new Error(`tsc --lsp process exited unexpectedly (code=${code} signal=${signal}) while a request was pending`));
    pending.clear();
  });

  async function shutdown(): Promise<void> {
    if (exited) return;
    try {
      await request("shutdown", null);
    } catch {
      // best-effort — proceed to exit/kill regardless
    }
    notify("exit", null);
    child.stdin.end();
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => {
        if (!exited) child.kill();
        resolve();
      }, 2000);
      child.on("exit", () => {
        clearTimeout(t);
        resolve();
      });
    });
  }

  return { request, notify, onNotification, shutdown };
}

// ---------------------------------------------------------------------------
// Hover / completion / diagnostic narrowing — duplicated from
// editor-latency.ts (D5).
// ---------------------------------------------------------------------------

function hoverText(result: unknown): string {
  const r = result as { contents?: unknown } | null | undefined;
  if (!r || r.contents === undefined || r.contents === null) return "";
  const c = r.contents;
  if (typeof c === "string") return c;
  if (typeof c === "object" && c !== null && "value" in c) return String((c as { value: unknown }).value);
  if (Array.isArray(c)) return c.map((x) => (typeof x === "string" ? x : String((x as { value?: unknown })?.value ?? ""))).join("\n");
  throw new Error(`hoverText: unrecognized hover contents shape: ${JSON.stringify(c)}`);
}

function completionItemCount(result: unknown): number {
  if (Array.isArray(result)) return result.length;
  const r = result as { items?: unknown[] } | null | undefined;
  return r?.items?.length ?? 0;
}

interface DiagnosticItem {
  range: { start: { line: number; character: number } };
  code?: number;
  severity?: number;
  message: string;
}

function diagnosticItems(result: unknown): DiagnosticItem[] {
  const r = result as { items?: DiagnosticItem[] } | null | undefined;
  return r?.items ?? [];
}

function hostLoadLine(): string {
  try {
    return execFileSync("uptime", [], { encoding: "utf8" }).trim();
  } catch (e) {
    return `(uptime unavailable: ${String(e)})`;
  }
}

function assertHoverCorrect(pointId: string, label: string, position: { line: number; character: number }, result: unknown, expected: string): void {
  const text = hoverText(result);
  if (!text.includes(expected)) {
    throw new Error(
      `CORRECTNESS GATE FAILED [${pointId}] hover "${label}" at line=${position.line} char=${position.character}: ` +
        `expected substring "${expected}" but got "${text}" — aborting this point's measurement (never time a wrong result).`,
    );
  }
}

// ---------------------------------------------------------------------------
// Instantiation measurement + D23 cliff/other-failure classification.
// ---------------------------------------------------------------------------

interface InstResult {
  classification: "ok" | "cliff" | "other-failure";
  files: number | null;
  lines: number | null;
  types: number | null;
  instantiations: number | null;
  checkTimeS: number | null;
  memoryKb: number | null;
  hadTypeErrors: boolean;
  detail: string | null; // populated for cliff/other-failure
}

function isExecErrorLike(e: unknown): e is { stdout?: string; stderr?: string; signal?: string | null; status?: number | null } {
  return typeof e === "object" && e !== null;
}

function parseInstOutput(out: string, classification: InstResult["classification"], hadTypeErrors: boolean, detail: string | null): InstResult {
  const parseInt_ = (label: string): number | null => {
    const m = new RegExp(`${label}:\\s+(\\d+)`).exec(out);
    return m && m[1] ? parseInt(m[1], 10) : null;
  };
  const parseFloat_ = (label: string): number | null => {
    const m = new RegExp(`${label} time:\\s+([\\d.]+)s`).exec(out);
    return m && m[1] ? parseFloat(m[1]) : null;
  };
  const memMatch = /Memory used:\s+([\d,]+)K/.exec(out);
  const memoryKb = memMatch && memMatch[1] ? parseInt(memMatch[1].replace(/,/g, ""), 10) : null;
  return {
    classification,
    files: parseInt_("Files"),
    lines: parseInt_("Lines"),
    types: parseInt_("Types"),
    instantiations: parseInt_("Instantiations"),
    checkTimeS: parseFloat_("Check"),
    memoryKb,
    hadTypeErrors,
    detail,
  };
}

/** D23: classify a point's `tsc --extendedDiagnostics` outcome. Never
 * classifies a signal-killed process (OOM et al.) or an UNEXPECTED type
 * error as "cliff" — only the documented TS2589 "excessively deep" marker
 * does. A point carrying a deliberate toggle target (D22) is expected to
 * report real (non-cliff) type errors on its persisted "broken" on-disk
 * state — that is normal data (`hadTypeErrors: true`), not a failure. */
function measureInstantiationsForPoint(exe: string, point: ScalePointManifestEntry): InstResult {
  const tsconfigPath = join(scaleWorkloadsDir, point.dir, point.tsconfig);
  try {
    const out = execFileSync(exe, ["--noEmit", "--extendedDiagnostics", "-p", tsconfigPath], { encoding: "utf8" });
    return parseInstOutput(out, "ok", false, null);
  } catch (e) {
    if (!isExecErrorLike(e)) return { classification: "other-failure", files: null, lines: null, types: null, instantiations: null, checkTimeS: null, memoryKb: null, hadTypeErrors: true, detail: `non-object exception: ${String(e)}` };
    if (e.signal) {
      return { classification: "other-failure", files: null, lines: null, types: null, instantiations: null, checkTimeS: null, memoryKb: null, hadTypeErrors: true, detail: `process killed by signal ${e.signal} (likely OOM) — stdout tail: ${(e.stdout ?? "").slice(-500)}` };
    }
    const out = e.stdout ?? "";
    if (/TS2589/.test(out) || /excessively deep/i.test(out)) {
      return parseInstOutput(out, "cliff", true, `TS2589 "Type instantiation is excessively deep" (exit status=${e.status ?? "?"})`);
    }
    if (point.toggle) {
      // Expected: the persisted "broken" on-disk state carries a deliberate,
      // permanent type error (D22, same convention as W4/W6) — normal data.
      return parseInstOutput(out, "ok", true, "deliberate toggle-target type error (expected, D22)");
    }
    return { classification: "other-failure", files: null, lines: null, types: null, instantiations: null, checkTimeS: null, memoryKb: null, hadTypeErrors: true, detail: `tsc exited non-zero without a TS2589 marker (status=${e.status ?? "?"}) — UNEXPECTED, stdout tail: ${out.slice(-500)}` };
  }
}

// ---------------------------------------------------------------------------
// LSP-based M1 (cold) / M2 (warm hover) / M3 (toggle, if present) / M4
// (completion) measurement for one sweep point — run only for points whose
// instantiation measurement did NOT classify as "cliff" (D23 point 2: cliff
// classification happens before any hover is attempted).
// ---------------------------------------------------------------------------

interface M1Result {
  initializeMs: number;
  pullDiagnosticMs: number;
  diagnosticCountOnOpen: number;
  loadProofHoverMs: number;
}
interface M2Result {
  stats: Stats;
  samples: number[];
}
interface M3DirectionResult {
  label: string;
  stats: Stats;
  samples: number[];
}
interface M4Result {
  stats: Stats;
  samples: number[];
  itemCountRange: [number, number];
}

async function measureLsp(exe: string, point: ScalePointManifestEntry): Promise<{ m1: M1Result; m2: M2Result; m3: M3DirectionResult[] | null; m4: M4Result }> {
  const filePath = join(scaleWorkloadsDir, point.dir, point.primaryFile);
  const uri = `file://${filePath}`;
  const onDiskText = readFileSync(filePath, "utf8");

  if (point.toggle && onDiskText !== point.toggle.initialState.text) {
    throw new Error(`[${point.id}] generator/runner mismatch: on-disk file does not match manifest's toggle.initialState.text`);
  }

  const client = createLspClient(exe);
  try {
  // --- M1: cold ---
  const initT0 = performance.now();
  await client.request("initialize", {
    processId: null,
    rootUri: `file://${repoRoot}`,
    capabilities: {
      workspace: { configuration: true, didChangeConfiguration: { dynamicRegistration: true } },
      textDocument: { hover: { contentFormat: ["plaintext", "markdown"] } },
    },
    workspaceFolders: [{ uri: `file://${repoRoot}`, name: "numtype" }],
  });
  const initializeMs = performance.now() - initT0;
  client.notify("initialized", {});

  client.notify("textDocument/didOpen", { textDocument: { uri, languageId: "typescript", version: 1, text: onDiskText } });
  const pullT0 = performance.now();
  const diagResp = await client.request("textDocument/diagnostic", { textDocument: { uri } });
  const pullDiagnosticMs = performance.now() - pullT0;
  const diagnosticCountOnOpen = diagnosticItems(diagResp).length;

  const proofT0 = performance.now();
  const proofResp = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: point.hover.line, character: point.hover.character } });
  const loadProofHoverMs = performance.now() - proofT0;
  assertHoverCorrect(point.id, `[M1 project-load proof] ${point.hover.label}`, point.hover, proofResp, point.hover.expected);

  const m1: M1Result = { initializeMs, pullDiagnosticMs, diagnosticCountOnOpen, loadProofHoverMs };

  // --- M2: warm hover ---
  for (let i = 0; i < WARMUP_SAMPLES; i++) {
    const r = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: point.hover.line, character: point.hover.character } });
    assertHoverCorrect(point.id, `${point.hover.label} (warmup)`, point.hover, r, point.hover.expected);
  }
  const hoverSamples: number[] = [];
  for (let i = 0; i < TIMED_SAMPLES; i++) {
    const t0 = performance.now();
    const r = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: point.hover.line, character: point.hover.character } });
    const dt = performance.now() - t0;
    assertHoverCorrect(point.id, point.hover.label, point.hover, r, point.hover.expected);
    hoverSamples.push(dt);
  }
  const m2: M2Result = { stats: statsOf(hoverSamples), samples: hoverSamples };

  // --- M3: warm toggle diagnostics (only points carrying a toggle target) ---
  let m3: M3DirectionResult[] | null = null;
  if (point.toggle) {
    const { initialState, otherState } = point.toggle;

    function diagHasToggleMarker(items: DiagnosticItem[], state: ToggleStateSpec): boolean {
      return items.some((d) => d.range.start.line === state.diagnosticLine && d.code === state.expectedCode);
    }

    async function toggleTo(state: ToggleStateSpec, version: number): Promise<number> {
      const t0 = performance.now();
      client.notify("textDocument/didChange", { textDocument: { uri, version }, contentChanges: [{ text: state.text }] });
      const resp = await client.request("textDocument/diagnostic", { textDocument: { uri } });
      const dt = performance.now() - t0;
      const items = diagnosticItems(resp);
      const has = diagHasToggleMarker(items, state);
      if (has !== state.expectDiagnosticPresent) {
        throw new Error(
          `CORRECTNESS GATE FAILED [${point.id}] M3 toggle to state "${state.label}": expected diagnostic(code=${state.expectedCode}, line=${state.diagnosticLine}) present=${state.expectDiagnosticPresent}, got present=${has}. Diagnostics: ${JSON.stringify(items)}`,
        );
      }
      return dt;
    }

    let version = 2;
    for (let i = 0; i < WARMUP_SAMPLES; i++) {
      await toggleTo(otherState, version++);
      await toggleTo(initialState, version++);
    }
    const toOtherSamples: number[] = [];
    const toInitialSamples: number[] = [];
    for (let i = 0; i < TIMED_SAMPLES; i++) {
      toOtherSamples.push(await toggleTo(otherState, version++));
      toInitialSamples.push(await toggleTo(initialState, version++));
    }
    m3 = [
      { label: `-> ${otherState.label}`, stats: statsOf(toOtherSamples), samples: toOtherSamples },
      { label: `-> ${initialState.label}`, stats: statsOf(toInitialSamples), samples: toInitialSamples },
    ];
  }

  // --- M4: completion (informational) ---
  for (let i = 0; i < WARMUP_SAMPLES; i++) {
    await client.request("textDocument/completion", { textDocument: { uri }, position: { line: point.completion.line, character: point.completion.character } });
  }
  const compSamples: number[] = [];
  let minItems = Infinity;
  let maxItems = -Infinity;
  for (let i = 0; i < TIMED_SAMPLES; i++) {
    const t0 = performance.now();
    const r = await client.request("textDocument/completion", { textDocument: { uri }, position: { line: point.completion.line, character: point.completion.character } });
    const dt = performance.now() - t0;
    compSamples.push(dt);
    const n = completionItemCount(r);
    minItems = Math.min(minItems, n);
    maxItems = Math.max(maxItems, n);
  }
  const m4: M4Result = { stats: statsOf(compSamples), samples: compSamples, itemCountRange: [minItems, maxItems] };

  return { m1, m2, m3, m4 };
  } finally {
    // Resource-leak fix (found during the real sweep run, 2026-07-21): a
    // thrown correctness-gate violation anywhere above used to skip this
    // shutdown entirely, leaking the spawned `tsc --lsp --stdio` child
    // process indefinitely (its open stdio pipes keep Node's event loop
    // alive even after every synchronous log line has printed — the
    // orphaned child, not a stuck script, was why the process never
    // exited). `finally` guarantees shutdown runs on every path, including
    // the ones D23 explicitly wants caught-and-continued by the caller.
    await client.shutdown();
  }
}

// ---------------------------------------------------------------------------
// Per-point orchestration (D23 point 1: try/catch per point, sweep continues
// regardless of any single point's outcome).
// ---------------------------------------------------------------------------

interface PointResult {
  id: string;
  axis: "a" | "b" | "c";
  subseries: string;
  sweepValue: number;
  status: "ok" | "cliff" | "other-failure";
  inst: InstResult | null;
  m1: M1Result | null;
  m2: M2Result | null;
  m3: M3DirectionResult[] | null;
  m4: M4Result | null;
  error: string | null;
}

async function measurePoint(exe: string, point: ScalePointManifestEntry): Promise<PointResult> {
  const base = { id: point.id, axis: point.axis, subseries: point.subseries, sweepValue: point.sweepValue };
  try {
    const inst = measureInstantiationsForPoint(exe, point);
    if (inst.classification === "cliff") {
      return { ...base, status: "cliff", inst, m1: null, m2: null, m3: null, m4: null, error: inst.detail };
    }
    if (inst.classification === "other-failure") {
      return { ...base, status: "other-failure", inst, m1: null, m2: null, m3: null, m4: null, error: inst.detail };
    }
    const lsp = await measureLsp(exe, point);
    return { ...base, status: "ok", inst, m1: lsp.m1, m2: lsp.m2, m3: lsp.m3, m4: lsp.m4, error: null };
  } catch (e) {
    return { ...base, status: "other-failure", inst: null, m1: null, m2: null, m3: null, m4: null, error: e instanceof Error ? (e.stack ?? e.message) : String(e) };
  }
}

// ---------------------------------------------------------------------------
// Report printing (G4: strictly grouped per axis, never mixed populations).
// ---------------------------------------------------------------------------

function printAxisTable(axis: "a" | "b" | "c", results: PointResult[]): void {
  const rows = results.filter((r) => r.axis === axis);
  if (rows.length === 0) return;
  console.log(`\n=== Axis (${axis}) — ${rows.length} point(s) ===`);
  console.log(
    `${"point".padEnd(20)} | ${"status".padEnd(13)} | ${"instantiations".padStart(14)} | ${"check time".padStart(10)} | ${"memory".padStart(10)} | ${"M1 load-proof".padStart(13)} | ${"M2 hover median".padStart(15)} | ${"M4 compl. median".padStart(17)}`,
  );
  for (const r of rows) {
    const inst = r.inst?.instantiations ?? null;
    const checkT = r.inst?.checkTimeS ?? null;
    const mem = r.inst?.memoryKb ?? null;
    const m1 = r.m1 ? fmtMs(r.m1.loadProofHoverMs) : "n/a";
    const m2 = r.m2 ? fmtMs(r.m2.stats.median) : "n/a";
    const m4 = r.m4 ? fmtMs(r.m4.stats.median) : "n/a";
    console.log(
      `${r.id.padEnd(20)} | ${r.status.padEnd(13)} | ${String(inst ?? "?").padStart(14)} | ${(checkT === null ? "?" : `${checkT.toFixed(3)}s`).padStart(10)} | ${(mem === null ? "?" : `${mem}K`).padStart(10)} | ${m1.padStart(13)} | ${m2.padStart(15)} | ${m4.padStart(17)}`,
    );
  }
  const cliffs = rows.filter((r) => r.status === "cliff");
  const failures = rows.filter((r) => r.status === "other-failure");
  if (cliffs.length > 0) {
    console.log(`  Cliff point(s): ${cliffs.map((r) => r.id).join(", ")}`);
    for (const r of cliffs) console.log(`    [${r.id}] ${r.error}`);
  }
  if (failures.length > 0) {
    console.log(`  OTHER-FAILURE point(s) (never counted as cliff): ${failures.map((r) => r.id).join(", ")}`);
    for (const r of failures) console.log(`    [${r.id}] ${r.error}`);
  }
  const withToggle = rows.filter((r) => r.m3 !== null);
  if (withToggle.length > 0) {
    console.log(`  M3 toggle (hard-gate-free, informational):`);
    for (const r of withToggle) {
      for (const d of r.m3!) console.log(`    [${r.id}] ${d.label}: median=${fmtMs(d.stats.median)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== NumType Scale-Probe: sweep runner ===");
  console.log(`Host load (uptime): ${hostLoadLine()}`);

  const manifestPath = join(scaleWorkloadsDir, "manifest.json");
  let manifest: ScaleManifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as ScaleManifest;
  } catch (e) {
    throw new Error(`Could not read manifest at ${manifestPath} — did you run gen-scale-workloads.ts first? (pnpm bench:scale runs both). Underlying: ${String(e)}`);
  }

  const exe = await resolveTscExe();
  console.log(`Resolved native tsc exe: ${exe}`);
  console.log(`${manifest.points.length} sweep points to measure. M2/M4: ${WARMUP_SAMPLES} warmup + ${TIMED_SAMPLES} timed samples. M1 per point (fresh server). M3 only on toggle-bearing points.`);

  const sweepT0 = performance.now();
  const results: PointResult[] = [];
  for (const point of manifest.points) {
    console.log(`\n>>> [${point.axis}] ${point.id} (sweepValue=${point.sweepValue}, ${point.files.length} file(s))...`);
    const r = await measurePoint(exe, point);
    results.push(r);
    if (r.status === "ok") {
      console.log(`    OK — instantiations=${r.inst?.instantiations} checkTime=${r.inst?.checkTimeS}s M1=${r.m1 ? fmtMs(r.m1.loadProofHoverMs) : "n/a"} M2=${r.m2 ? fmtMs(r.m2.stats.median) : "n/a"}`);
    } else if (r.status === "cliff") {
      console.log(`    CLIFF — ${r.error}`);
    } else {
      console.log(`    OTHER-FAILURE — ${r.error}`);
    }
  }
  const sweepWallMs = performance.now() - sweepT0;

  for (const axis of ["a", "b", "c"] as const) printAxisTable(axis, results);

  const okCount = results.filter((r) => r.status === "ok").length;
  const cliffCount = results.filter((r) => r.status === "cliff").length;
  const failCount = results.filter((r) => r.status === "other-failure").length;
  console.log(`\n=== Sweep summary: ${okCount} ok, ${cliffCount} cliff, ${failCount} other-failure, out of ${results.length} points. Wall time: ${(sweepWallMs / 1000).toFixed(1)}s ===`);
  console.log(`G1 budget (30 min = 1800s): ${sweepWallMs / 1000 <= 1800 ? "WITHIN BUDGET" : "EXCEEDED — see spec G1"}`);

  writeFileSync(
    join(scaleWorkloadsDir, "results.json"),
    JSON.stringify({ hostLoad: hostLoadLine(), sweepWallMs, pointCount: results.length, okCount, cliffCount, failCount, results }, null, 2) + "\n",
  );
  console.log("\nresults.json written (scale-workloads/results.json, gitignored).");
  console.log("\n=== Run complete. (No hard pass/fail gate — G3: this runner is informational.) ===");
}

main().catch((err: unknown) => {
  console.error("\nFATAL — scale-latency runner aborted (a bug in the runner itself, not a sweep-point failure — those are caught per-point):");
  console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
  throw err;
});
