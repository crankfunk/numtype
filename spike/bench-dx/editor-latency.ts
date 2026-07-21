/**
 * Spike 02 editor-latency harness (docs/spike-02-editor-latency-spec.md).
 * Drives the installed `typescript@7.0.2` native language server
 * (`tsc --lsp --stdio`, the same entry point VS Code's own extension talks
 * to) over a hand-rolled JSON-RPC/LSP client — no external LSP client
 * dependency, consistent with the no-external-libraries constraint. Reads
 * `spike/bench-dx/workloads/manifest.json` (written by `gen-workloads.ts`,
 * which MUST be run first — see `pnpm bench:editor`) and measures, per
 * workload:
 *
 *  - M1 cold: fresh server process, `initialize` roundtrip, `didOpen` ->
 *    first pull `textDocument/diagnostic` (push `publishDiagnostics`
 *    recorded too if earlier), then a project-load PROOF hover (must
 *    contain the expected shape text — a wrong/empty hover here means the
 *    project did not load and ABORTS the whole run, per spec).
 *  - M2 warm hover: >=3 warmup + 20 timed hovers per position, gated
 *    (hover text must contain the expected shape) EVERY sample, timed or
 *    not — never time (or silently accept) a wrong result.
 *  - M3 warm toggle diagnostics (every workload carrying a `toggle` spec —
 *    currently W4 and W6, see gen-workloads.ts; see the
 *    "M3 scope" note in main() for the per-workload details):
 *    alternating full-text didChange between the toggle's two states, N=20
 *    timed samples per direction, gated on the toggle-line diagnostic
 *    (code+line) actually flipping.
 *  - M4 completion: informational only (no gate per spec), 20 timed samples
 *    on a real member-access position.
 *
 * Correctness-gate philosophy matches this repo's benches (see
 * `spike/bench-core/chain.ts`'s `assertBitIdentical`): a violation THROWS
 * immediately, aborting the whole run loudly rather than silently
 * continuing past — never average away a wrong result.
 *
 * Every LSP request goes through one `request()` with a REQUEST_TIMEOUT_MS
 * global timeout that rejects (never hangs) naming the method + params.
 * Server->client requests (`client/registerCapability`,
 * `workspace/configuration`) are answered automatically — an unanswered one
 * stalls hover forever (verified empirically, see the spec's pitfall note).
 */
import { execFileSync, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const workloadsDir = join(__dirname, "workloads");
const repoRoot = join(__dirname, "..", "..");

const REQUEST_TIMEOUT_MS = 30_000;
const WARMUP_SAMPLES = 3;
const TIMED_SAMPLES = 20;

const HOVER_GATE_MS = 100; // M2 hard gate (roadmap A1)
const TOGGLE_GATE_MS = 500; // M3 hard gate (roadmap A1)
const GATE_2X_FACTOR = 2; // "no single workload median may exceed 2x its gate"

// ---------------------------------------------------------------------------
// Manifest types — deliberately duplicated from gen-workloads.ts (not
// imported: that file executes generation logic at import time, and the two
// scripts are meant to run as separate `node` processes per the spec).
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
  initialState: ToggleStateSpec;
  otherState: ToggleStateSpec;
}
interface WorkloadManifestEntry {
  id: string;
  fileName: string;
  lineCount: number;
  instantiationTsconfig: string;
  hovers: HoverSpec[];
  completion: CompletionSpec;
  toggle?: ToggleSpec;
}
interface Manifest {
  workloads: WorkloadManifestEntry[];
}

// ---------------------------------------------------------------------------
// Stats helpers.
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
// tsc exe resolution — the native platform binary, bypassing the Node
// wrapper (`node_modules/typescript/bin/tsc`) so cold-start timing excludes
// Node's own boot. `getExePath.js` isn't in `typescript`'s package.json
// "exports" map, so it's imported by direct filesystem path (not a bare
// specifier) — Node's export-map enforcement only applies to specifier
// resolution, not direct `file://` imports (verified empirically).
// ---------------------------------------------------------------------------

async function resolveTscExe(): Promise<string> {
  const getExePathPath = join(repoRoot, "node_modules", "typescript", "lib", "getExePath.js");
  const mod = (await import(`file://${getExePathPath}`)) as { default: () => string };
  return mod.default();
}

// ---------------------------------------------------------------------------
// Hand-rolled JSON-RPC/LSP client over stdio (Content-Length framing).
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
        // A response to one of OUR requests.
        if (typeof msg.id === "number") {
          const p = pending.get(msg.id);
          if (p) {
            pending.delete(msg.id);
            if (msg.error) p.reject(new Error(`LSP error response: ${JSON.stringify(msg.error)}`));
            else p.resolve(msg.result);
          }
        }
      } else if (msg.method !== undefined && msg.id !== undefined) {
        // Server->client REQUEST — MUST be answered or hover stalls forever
        // (verified empirically, see file header / spec pitfall note).
        let result: unknown = null;
        if (msg.method === "workspace/configuration") {
          const items = (msg.params as { items?: unknown[] })?.items ?? [];
          result = items.map(() => null);
        }
        send({ jsonrpc: "2.0", id: msg.id, result });
      } else if (msg.method !== undefined && msg.id === undefined) {
        // A notification (window/logMessage, publishDiagnostics, $/...).
        const method = msg.method;
        for (const l of notificationListeners) l(method, msg.params);
      }
    }
  });

  child.stderr.on("data", (chunk: Buffer) => {
    console.error(`[tsc-lsp stderr] ${chunk.toString("utf8")}`);
  });

  // An unhandled 'error' event on an EventEmitter crashes the process by
  // default (e.g. ENOENT if the exe path is wrong, EPIPE on a stdin write
  // after the process died) — convert it into rejections for whatever is
  // pending instead of an opaque process crash.
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
// Hover / completion result narrowing (defensive — never silently accept an
// unrecognized shape as "correct").
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

// ---------------------------------------------------------------------------
// Host load (spec: "Record host load (uptime) before each run").
// ---------------------------------------------------------------------------

function hostLoadLine(): string {
  try {
    return execFileSync("uptime", [], { encoding: "utf8" }).trim();
  } catch (e) {
    return `(uptime unavailable: ${String(e)})`;
  }
}

// ---------------------------------------------------------------------------
// Per-workload measurement (M1 cold on a fresh server, then M2/M3/M4 warm
// on that SAME instance, serialized — per spec discipline).
// ---------------------------------------------------------------------------

interface M1Result {
  initializeMs: number;
  pullDiagnosticMs: number;
  pushDiagnosticMs: number | null;
  diagnosticCountOnOpen: number;
  loadProofHoverMs: number;
  loadProofLabel: string;
}

interface M2Result {
  label: string;
  stats: Stats;
  samples: number[];
  gatePass: boolean;
  gate2xPass: boolean;
}

interface M3DirectionResult {
  label: string;
  stats: Stats;
  samples: number[];
  gatePass: boolean;
  gate2xPass: boolean;
}

interface M4Result {
  label: string;
  stats: Stats;
  samples: number[];
  itemCountRange: [number, number];
}

interface WorkloadResult {
  id: string;
  fileName: string;
  m1: M1Result;
  m2: M2Result[];
  m3: M3DirectionResult[] | null;
  m4: M4Result;
}

function assertHoverCorrect(workloadId: string, label: string, position: { line: number; character: number }, result: unknown, expected: string): void {
  const text = hoverText(result);
  if (!text.includes(expected)) {
    throw new Error(
      `CORRECTNESS GATE FAILED [${workloadId}] hover "${label}" at line=${position.line} char=${position.character}: ` +
        `expected substring "${expected}" but got "${text}" — aborting (never time a wrong result).`,
    );
  }
}

async function measureWorkload(exe: string, entry: WorkloadManifestEntry): Promise<WorkloadResult> {
  const filePath = join(workloadsDir, entry.fileName);
  const uri = `file://${filePath}`;
  const onDiskText = readFileSync(filePath, "utf8");

  if (entry.toggle && onDiskText !== entry.toggle.initialState.text) {
    throw new Error(`[${entry.id}] generator/harness mismatch: on-disk file does not match manifest's toggle.initialState.text`);
  }

  const client = createLspClient(exe);

  let firstPublishAtMs: number | null = null;
  let didOpenSentAtMs = 0;
  client.onNotification((method, params) => {
    if (method === "textDocument/publishDiagnostics" && firstPublishAtMs === null) {
      const p = params as { uri?: string };
      // Case-insensitive compare: observed empirically that the server can
      // report SOME file:// URIs lowercased (macOS's case-insensitive FS) —
      // never let that silently suppress the push-diagnostics timing.
      if (p.uri && p.uri.toLowerCase() === uri.toLowerCase()) firstPublishAtMs = performance.now() - didOpenSentAtMs;
    }
  });

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

  didOpenSentAtMs = performance.now();
  client.notify("textDocument/didOpen", { textDocument: { uri, languageId: "typescript", version: 1, text: onDiskText } });
  const pullT0 = performance.now();
  const diagResp = await client.request("textDocument/diagnostic", { textDocument: { uri } });
  const pullDiagnosticMs = performance.now() - pullT0;
  const diagnosticCountOnOpen = diagnosticItems(diagResp).length;

  const loadProof = entry.hovers[0];
  if (!loadProof) throw new Error(`[${entry.id}] manifest has no hover positions to use as the project-load proof`);
  const proofT0 = performance.now();
  const proofResp = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: loadProof.line, character: loadProof.character } });
  const loadProofHoverMs = performance.now() - proofT0;
  assertHoverCorrect(entry.id, `[M1 project-load proof] ${loadProof.label}`, loadProof, proofResp, loadProof.expected);

  const m1: M1Result = {
    initializeMs,
    pullDiagnosticMs,
    pushDiagnosticMs: firstPublishAtMs,
    diagnosticCountOnOpen,
    loadProofHoverMs,
    loadProofLabel: loadProof.label,
  };

  // --- M2: warm hover, per position ---
  const m2: M2Result[] = [];
  for (const h of entry.hovers) {
    for (let i = 0; i < WARMUP_SAMPLES; i++) {
      const r = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: h.line, character: h.character } });
      assertHoverCorrect(entry.id, `${h.label} (warmup)`, h, r, h.expected);
    }
    const samples: number[] = [];
    for (let i = 0; i < TIMED_SAMPLES; i++) {
      const t0 = performance.now();
      const r = await client.request("textDocument/hover", { textDocument: { uri }, position: { line: h.line, character: h.character } });
      const dt = performance.now() - t0;
      assertHoverCorrect(entry.id, h.label, h, r, h.expected);
      samples.push(dt);
    }
    const s = statsOf(samples);
    m2.push({ label: h.label, stats: s, samples, gatePass: s.median <= HOVER_GATE_MS, gate2xPass: s.median <= HOVER_GATE_MS * GATE_2X_FACTOR });
  }

  // --- M3: warm toggle diagnostics (only workloads with a `toggle` spec) ---
  let m3: M3DirectionResult[] | null = null;
  if (entry.toggle) {
    const { initialState, otherState } = entry.toggle;

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
          `CORRECTNESS GATE FAILED [${entry.id}] M3 toggle to state "${state.label}": expected diagnostic(code=${state.expectedCode}, line=${state.diagnosticLine}) present=${state.expectDiagnosticPresent}, got present=${has}. Diagnostics: ${JSON.stringify(items)}`,
        );
      }
      return dt;
    }

    let version = 2;
    // Warmup: a few untimed (but still gated) round trips.
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

    const sOther = statsOf(toOtherSamples);
    const sInitial = statsOf(toInitialSamples);
    m3 = [
      {
        label: `-> ${otherState.label}`,
        stats: sOther,
        samples: toOtherSamples,
        gatePass: sOther.median <= TOGGLE_GATE_MS,
        gate2xPass: sOther.median <= TOGGLE_GATE_MS * GATE_2X_FACTOR,
      },
      {
        label: `-> ${initialState.label}`,
        stats: sInitial,
        samples: toInitialSamples,
        gatePass: sInitial.median <= TOGGLE_GATE_MS,
        gate2xPass: sInitial.median <= TOGGLE_GATE_MS * GATE_2X_FACTOR,
      },
    ];
  }

  // --- M4: completion (informational, no gate) ---
  const c = entry.completion;
  for (let i = 0; i < WARMUP_SAMPLES; i++) {
    await client.request("textDocument/completion", { textDocument: { uri }, position: { line: c.line, character: c.character } });
  }
  const compSamples: number[] = [];
  let minItems = Infinity;
  let maxItems = -Infinity;
  for (let i = 0; i < TIMED_SAMPLES; i++) {
    const t0 = performance.now();
    const r = await client.request("textDocument/completion", { textDocument: { uri }, position: { line: c.line, character: c.character } });
    const dt = performance.now() - t0;
    compSamples.push(dt);
    const n = completionItemCount(r);
    minItems = Math.min(minItems, n);
    maxItems = Math.max(maxItems, n);
  }
  const m4: M4Result = { label: c.label, stats: statsOf(compSamples), samples: compSamples, itemCountRange: [minItems, maxItems] };

  await client.shutdown();

  return { id: entry.id, fileName: entry.fileName, m1, m2, m3, m4 };
}

// ---------------------------------------------------------------------------
// Instantiation counts (acceptance criterion 5): per-workload
// `--extendedDiagnostics`, same mechanism as `pnpm check:diag`/`check:diag:bench`.
// ---------------------------------------------------------------------------

interface InstantiationResult {
  id: string;
  files: number | null;
  lines: number | null;
  types: number | null;
  instantiations: number | null;
  checkTimeS: number | null;
  hadTypeErrors: boolean;
}

function isExecErrorWithStdout(e: unknown): e is { stdout: string; status: number } {
  return typeof e === "object" && e !== null && "stdout" in e;
}

function measureInstantiations(exe: string, entry: WorkloadManifestEntry): InstantiationResult {
  const tsconfigPath = join(workloadsDir, entry.instantiationTsconfig);
  let out: string;
  let hadTypeErrors = false;
  try {
    out = execFileSync(exe, ["--noEmit", "--extendedDiagnostics", "-p", tsconfigPath], { encoding: "utf8" });
  } catch (e) {
    // tsc exits non-zero when the compiled program has type errors (true for
    // W4 by design) — the extendedDiagnostics footer is still on stdout.
    if (!isExecErrorWithStdout(e)) throw e;
    out = e.stdout;
    hadTypeErrors = true;
  }
  const parseInt_ = (label: string): number | null => {
    const m = new RegExp(`${label}:\\s+(\\d+)`).exec(out);
    return m && m[1] ? parseInt(m[1], 10) : null;
  };
  const parseFloat_ = (label: string): number | null => {
    const m = new RegExp(`${label} time:\\s+([\\d.]+)s`).exec(out);
    return m && m[1] ? parseFloat(m[1]) : null;
  };
  return {
    id: entry.id,
    files: parseInt_("Files"),
    lines: parseInt_("Lines"),
    types: parseInt_("Types"),
    instantiations: parseInt_("Instantiations"),
    checkTimeS: parseFloat_("Check"),
    hadTypeErrors,
  };
}

// ---------------------------------------------------------------------------
// Report printing.
// ---------------------------------------------------------------------------

function printM1Table(results: WorkloadResult[]): void {
  console.log("\n--- M1: cold project load (informational, no gate) ---");
  console.log(`${"workload".padEnd(6)} | ${"initialize".padStart(11)} | ${"pull diag".padStart(10)} | ${"push diag".padStart(10)} | ${"diag#".padStart(6)} | ${"load-proof hover".padStart(17)}`);
  for (const r of results) {
    const push = r.m1.pushDiagnosticMs === null ? "n/a" : fmtMs(r.m1.pushDiagnosticMs);
    console.log(
      `${r.id.padEnd(6)} | ${fmtMs(r.m1.initializeMs).padStart(11)} | ${fmtMs(r.m1.pullDiagnosticMs).padStart(10)} | ${push.padStart(10)} | ${String(r.m1.diagnosticCountOnOpen).padStart(6)} | ${fmtMs(r.m1.loadProofHoverMs).padStart(17)}`,
    );
  }
}

function printM2Table(results: WorkloadResult[]): void {
  console.log(`\n--- M2: warm hover (hard gate: median <= ${HOVER_GATE_MS}ms, 2x ceiling = ${HOVER_GATE_MS * GATE_2X_FACTOR}ms) ---`);
  console.log(`${"workload".padEnd(6)} | ${"position".padEnd(38)} | ${"median".padStart(8)} | ${"min".padStart(8)} | ${"max".padStart(8)} | gate`);
  for (const r of results) {
    for (const h of r.m2) {
      const gate = h.gatePass ? "PASS" : h.gate2xPass ? "FAIL(<=2x)" : "FAIL(>2x)";
      console.log(`${r.id.padEnd(6)} | ${h.label.padEnd(38)} | ${fmtMs(h.stats.median).padStart(8)} | ${fmtMs(h.stats.min).padStart(8)} | ${fmtMs(h.stats.max).padStart(8)} | ${gate}`);
    }
  }
}

function printM3Table(results: WorkloadResult[]): void {
  console.log(`\n--- M3: warm toggle diagnostics (hard gate: median <= ${TOGGLE_GATE_MS}ms, 2x ceiling = ${TOGGLE_GATE_MS * GATE_2X_FACTOR}ms) ---`);
  const any = results.some((r) => r.m3 !== null);
  if (!any) {
    console.log("  (no workload had a toggle target)");
    return;
  }
  console.log(`${"workload".padEnd(6)} | ${"direction".padEnd(24)} | ${"median".padStart(8)} | ${"min".padStart(8)} | ${"max".padStart(8)} | gate`);
  for (const r of results) {
    if (!r.m3) continue;
    for (const d of r.m3) {
      const gate = d.gatePass ? "PASS" : d.gate2xPass ? "FAIL(<=2x)" : "FAIL(>2x)";
      console.log(`${r.id.padEnd(6)} | ${d.label.padEnd(24)} | ${fmtMs(d.stats.median).padStart(8)} | ${fmtMs(d.stats.min).padStart(8)} | ${fmtMs(d.stats.max).padStart(8)} | ${gate}`);
    }
  }
}

function printM4Table(results: WorkloadResult[]): void {
  console.log("\n--- M4: completion (informational, no gate) ---");
  console.log(`${"workload".padEnd(6)} | ${"position".padEnd(38)} | ${"median".padStart(8)} | ${"min".padStart(8)} | ${"max".padStart(8)} | items(min-max)`);
  for (const r of results) {
    const [lo, hi] = r.m4.itemCountRange;
    console.log(`${r.id.padEnd(6)} | ${r.m4.label.padEnd(38)} | ${fmtMs(r.m4.stats.median).padStart(8)} | ${fmtMs(r.m4.stats.min).padStart(8)} | ${fmtMs(r.m4.stats.max).padStart(8)} | ${lo}-${hi}`);
  }
}

function printInstantiationTable(results: InstantiationResult[]): void {
  console.log("\n--- Instantiation counts (--extendedDiagnostics, per-workload isolated project; acceptance criterion 5) ---");
  console.log(`${"workload".padEnd(6)} | ${"files".padStart(6)} | ${"lines".padStart(7)} | ${"types".padStart(7)} | ${"instantiations".padStart(14)} | ${"check time".padStart(10)} | note`);
  for (const r of results) {
    const note = r.hadTypeErrors ? "(has type errors by design)" : "";
    console.log(
      `${r.id.padEnd(6)} | ${String(r.files ?? "?").padStart(6)} | ${String(r.lines ?? "?").padStart(7)} | ${String(r.types ?? "?").padStart(7)} | ${String(r.instantiations ?? "?").padStart(14)} | ${r.checkTimeS === null ? "?" : `${r.checkTimeS.toFixed(3)}s`.padStart(10)} | ${note}`,
    );
  }
}

function printGateVerdict(results: WorkloadResult[]): void {
  console.log("\n--- Gate verdict (roadmap A1 hard release gate) ---");
  let allPass = true;
  const violations: string[] = [];
  for (const r of results) {
    for (const h of r.m2) {
      if (!h.gatePass) {
        allPass = false;
        violations.push(`[${r.id}] M2 hover "${h.label}": median ${fmtMs(h.stats.median)} > ${HOVER_GATE_MS}ms${h.gate2xPass ? "" : " (exceeds 2x ceiling)"}`);
      }
    }
    if (r.m3) {
      for (const d of r.m3) {
        if (!d.gatePass) {
          allPass = false;
          violations.push(`[${r.id}] M3 toggle "${d.label}": median ${fmtMs(d.stats.median)} > ${TOGGLE_GATE_MS}ms${d.gate2xPass ? "" : " (exceeds 2x ceiling)"}`);
        }
      }
    }
  }
  console.log(`Overall hard gate: ${allPass ? "PASS" : "FAIL"}`);
  if (!allPass) {
    console.log("Violations:");
    for (const v of violations) console.log(`  - ${v}`);
  }
}

// ---------------------------------------------------------------------------
// Item 12 (CI): hard CI gate — correctness (already thrown in assertHoverCorrect
// / M3 toggle) + instantiation pins (exact) are hard; latency is gated at the 2x
// ceiling. Additive: does NOT touch the measurement paths (M1–M4,
// measureInstantiations) or the informational printGateVerdict; it only reads
// their results and sets process.exitCode.
// ---------------------------------------------------------------------------

// Instantiation pins per workload (spec D5). Exact match — the counts are
// deterministic same-platform (Spike 02 / Baustein-0: 3x/2x byte-identical).
// Measured 2026-07-17, macos-arm64 / tsc 7.0.2. Cross-platform stability is
// checked by the first CI run (spec §6); if a Linux run differs, evolve this to
// a platform-labelled set like scripts/check-freeze-hash.mjs.
//
// Re-measured 2026-07-20 (Op-Scheibe W1, docs/op-w1-argmax-topk-spec.md, D7):
// `NDArray.argmax`/`.topk` add two overloaded generic methods to the class
// surface every workload instantiates `NDArray<S>` against — a UNIFORM
// +804 shift across all seven workloads (measured twice, byte-identical
// both times), independent of each workload's own subject matter. Pins
// updated per D7's explicit allowance ("bench:editor-Instantiation-Pins
// dürfen sich durch die ndarray.ts-Surface-Erweiterung verschieben"); the
// prior values are preserved in the same commit's diff for the
// before/after record. Latency medians and the correctness gate
// (`printGateVerdict`) were unaffected (still PASS, still under the 2x
// ceiling) — only these exact counts moved.
//
// Re-measured again 2026-07-21 (Op-Scheibe W2, docs/op-w2-scalar-mean-spec.md,
// D8): the D6-v2 scalar-overload conversion of `add`/`sub`/`mul`/`div` (each
// gains a new overload signature) plus the new `mean` method are the same
// class-of-ripple as W1's argmax/topk — every workload instantiates
// `NDArray<S>` against a now-larger overload set. Measured TWICE,
// byte-identical both times: a UNIFORM **+1181** shift on w1/w2/w3/w5/w6/w7,
// and **+1220** on w4 specifically (w4 is the deliberate-type-errors
// workload — its two `ShapeError`/`Guard` diagnostic paths resolve
// differently against the now-larger add/div overload sets than the other,
// error-free workloads; a genuine, reproducible, attributable difference,
// not measurement noise — not optimized away, per house rule). Prior values
// (post-W1) preserved in this commit's diff for the before/after record.
// Latency medians and the correctness gate were unaffected (still PASS,
// still under the 2x ceiling) — only these exact counts moved.
//
// Re-measured again 2026-07-21 (W2 Verify-B finding F1, MAJOR): the two
// add/sub/mul/div overload signatures were reordered (scalar overload
// declared FIRST, the generic `Guard`-carrying overload declared LAST) so a
// failed overload set surfaces TS's own "last candidate" diagnostic as the
// shape-naming `__shapeError` message instead of the scalar decoy ("not
// assignable to type 'number'") — a resolution-order-invariant change
// (`number` arguments still match the scalar overload first; only the
// FAILURE-path diagnostic detail moves). Measured TWICE, byte-identical
// both times: w1/w2/w3/w5/w6/w7 UNCHANGED from the post-W2 pins above
// (their code never hits a failed-overload-set diagnostic path); **w4
// alone shifts by -37** (26490 -> 26453) — w4 is the deliberate-type-errors
// workload, so its two failing calls are the only ones whose diagnostic
// resolution order actually changed. Latency medians and the correctness
// gate were unaffected (still PASS, still under the 2x ceiling).
//
// NOT re-measured for Op-Scheibe W3 (docs/op-w3-sqrt-spec.md): `sqrt()` is a
// single niladic, guard-less method (no overload signature, no `Guard`-typed
// argument) — measured, PASS WITHOUT a pin deviation (docs/op-w3-sqrt-
// ergebnisse.md), unlike W1/W2's generic/overloaded additions above. Kept
// here as the explanation for why the pins below skip straight from the W2
// Verify-B values to the W4 ones.
//
// Re-measured again 2026-07-21 (Op-Scheibe W4, docs/op-w4-stack-spec.md, D6):
// the new static `NDArray.stack` method (a `const`-generic static with a
// `Guard`-typed parameter) is the same class-of-ripple as W1's argmax/topk —
// every workload instantiates `NDArray<S>` against a now-larger static
// surface, independent of each workload's own subject matter (unlike W2's
// overload-reorder finding, this ripple does NOT differentiate the
// deliberate-type-errors workload from the others, since `stack` adds no
// overload to any of add/sub/mul/div/mean that w4's own two intentional
// errors resolve against). Measured TWICE, byte-identical both times: a
// UNIFORM **+845** shift across all seven workloads. Prior values (post-W2
// Verify-B) preserved in this commit's diff for the before/after record.
// Latency medians and the correctness gate were unaffected (still PASS,
// still under the 2x ceiling) — only these exact counts moved.
//
// Re-measured again 2026-07-21 (W4 Verify-B finding, BLOCKER-class M2 fix):
// `StackFold` (vector.ts) gained one new `IsUnion<Head>` gate, positioned
// BEFORE the naked `Head extends readonly [infer D]` destructure (the
// `ReduceAxis`/Union-Axis-Mini-Scheibe load-bearing-position precedent) — a
// tiny amount of new type machinery on an already-small fold. Measured
// TWICE, byte-identical both times: a UNIFORM **+6** shift across all seven
// workloads (the smallest ripple of any Op-Scheibe re-measurement so far —
// one extra conditional branch, not a new overload/generic surface).
// Correctness gate and latency medians unaffected (still PASS, still under
// the 2x ceiling).
// Re-measured again 2026-07-21 (Op-Scheibe W5, docs/op-w5-item-spec.md):
// `NDArray.item(...indices)` + `ItemGuard`/`ItemMark`/`ItemFoldAcc`
// (vector.ts) — new type machinery each workload's own import of
// spike/src pulls in regardless of whether the workload calls `item`
// itself (same "fixed overhead from the shared src graph" shape every
// prior Op-Scheibe re-measurement here has shown). Measured TWICE,
// byte-identical both times: a UNIFORM **+628** shift across all seven
// workloads. Correctness gate and latency medians unaffected (still PASS,
// still under the 2x ceiling).
const INSTANTIATION_PINS: Record<string, number> = {
  w1: 27769,
  w2: 29578,
  w3: 60718,
  w4: 27932,
  w5: 33223,
  w6: 34393,
  w7: 26941,
};

function enforceHardGate(results: WorkloadResult[], instResults: InstantiationResult[]): void {
  const violations: string[] = [];

  // Latency: 2x ceiling only (owner-chosen — the strict 1x gate stays
  // informational in printGateVerdict; on shared CI hardware only the 2x
  // ceiling is non-flaky, at ~3 orders of magnitude local headroom). The m3
  // loop is generic, so this covers BOTH toggle-bearing workloads (W4 and W6).
  for (const r of results) {
    for (const h of r.m2) {
      if (!h.gate2xPass) {
        violations.push(`[${r.id}] M2 hover "${h.label}": median ${fmtMs(h.stats.median)} exceeds the 2x ceiling (${HOVER_GATE_MS * GATE_2X_FACTOR}ms)`);
      }
    }
    if (r.m3) {
      for (const d of r.m3) {
        if (!d.gate2xPass) {
          violations.push(`[${r.id}] M3 toggle "${d.label}": median ${fmtMs(d.stats.median)} exceeds the 2x ceiling (${TOGGLE_GATE_MS * GATE_2X_FACTOR}ms)`);
        }
      }
    }
  }

  // Instantiation pins: exact match (deterministic). A drift is either a
  // type-budget regression to investigate, or — on the first CI run — an
  // expected platform difference; the actual value is printed so the reaction
  // is a one-line pin edit.
  for (const ir of instResults) {
    const pin = INSTANTIATION_PINS[ir.id];
    if (pin === undefined) {
      violations.push(`[${ir.id}] no instantiation pin defined — add it to INSTANTIATION_PINS`);
    } else if (ir.instantiations !== pin) {
      const delta = ir.instantiations === null ? "?" : `${ir.instantiations - pin >= 0 ? "+" : ""}${ir.instantiations - pin}`;
      violations.push(`[${ir.id}] instantiations = ${ir.instantiations}, pin = ${pin} (delta ${delta})`);
    }
  }

  console.log("\n--- Hard CI gate (spec D5: correctness + instantiation pins hard, latency at 2x ceiling) ---");
  if (violations.length === 0) {
    console.log("Hard CI gate: PASS");
    return;
  }
  console.log("Hard CI gate: FAIL");
  for (const v of violations) console.log(`  - ${v}`);
  // `process` is deliberately typed `unknown` in ambient.d.ts (feature-detect
  // only). Narrow locally just for the exit-code write, rather than widening the
  // shared shim's intentional "never dereference a property on it" contract.
  (process as { exitCode?: number }).exitCode = 1;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== NumType Spike 02: editor-latency harness ===");
  console.log(`Host load (uptime): ${hostLoadLine()}`);

  const manifestPath = join(workloadsDir, "manifest.json");
  let manifest: Manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
  } catch (e) {
    throw new Error(`Could not read manifest at ${manifestPath} — did you run gen-workloads.ts first? (pnpm bench:editor runs both). Underlying: ${String(e)}`);
  }

  const exe = await resolveTscExe();
  console.log(`Resolved native tsc exe: ${exe}`);
  // M3 scope note: M3 (toggle-diagnostic latency) is measured on every workload
  // carrying a `toggle` spec — currently W4 AND W6 (Kern 08 gave W6 a reshape/
  // flatten toggle target). W1/W2/W3/W5/W7 have no toggle target. The loop in
  // measureWorkload gates on `if (entry.toggle)`, so this is generic; M1/M2/M4
  // run on every workload. (Item-12 Baustein-0 finding F1: the earlier "W4 only"
  // wording predated Kern 08's W6 toggle and was stale.)
  console.log(`M2/M3/M4: ${WARMUP_SAMPLES} warmup + ${TIMED_SAMPLES} timed samples per position/direction. M3 measured on toggle-bearing workloads (W4, W6).`);

  const results: WorkloadResult[] = [];
  for (const entry of manifest.workloads) {
    console.log(`\n>>> Measuring workload ${entry.id} (${entry.fileName}, ${entry.lineCount} lines)...`);
    const r = await measureWorkload(exe, entry);
    results.push(r);
    console.log(`    M1 initialize=${fmtMs(r.m1.initializeMs)} pullDiag=${fmtMs(r.m1.pullDiagnosticMs)} loadProof=${fmtMs(r.m1.loadProofHoverMs)} (OK)`);
    console.log(`    M2: ${r.m2.length} hover position(s) measured, all correctness-gated OK`);
    if (r.m3) console.log(`    M3: ${r.m3.length} toggle direction(s) measured, correctness-gated OK`);
    console.log(`    M4: completion measured (informational)`);
  }

  const instResults = manifest.workloads.map((entry) => measureInstantiations(exe, entry));

  printM1Table(results);
  printM2Table(results);
  printM3Table(results);
  printM4Table(results);
  printInstantiationTable(instResults);
  printGateVerdict(results);
  enforceHardGate(results, instResults);

  console.log("\n=== Run complete. ===");
}

main().catch((err: unknown) => {
  console.error("\nFATAL — editor-latency harness aborted:");
  console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
  throw err;
});
