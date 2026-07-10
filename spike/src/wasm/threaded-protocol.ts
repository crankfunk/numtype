/**
 * Kern 06: the Atomics-based job-control-block layout shared by
 * `threaded.ts` (main thread, dispatcher) and `threaded-worker.ts` (worker
 * bootstrap) — a single source of truth for the cell indices so the two
 * sides can never drift out of sync. See docs/kern-06-threads-spec.md §3
 * "Job protocol (binding requirements; cell layout is implementer's
 * choice)".
 *
 * ## Protocol (sequence-number handshake, not a state-enum)
 *
 * Each worker gets its OWN control block (one `nt_alloc`'d region in the
 * shared WASM memory, `CONTROL_BLOCK_BYTES` long) holding, as consecutive
 * little-endian `i32` cells: a monotonically increasing POSTED sequence
 * number, a DONE sequence number the worker echoes back, a STATUS cell for
 * the completed job's kernel status, and the full 16-argument
 * `nt_matmul_blocked_partial` operand descriptor (identical for every
 * worker within one `threadedMatmul` call except ROW_START/ROW_END).
 *
 * A sequence-number handshake (rather than a small IDLE/BUSY/DONE state
 * enum) was chosen deliberately: it is monotonic, so there is no "who
 * resets the state back to IDLE" ambiguity between dispatch rounds (a real
 * race with a 3-state enum — verified empirically while designing this:
 * the worker's own wait-for-next-job call needs a stable "value I'm
 * waiting to change away from," and only a ever-increasing counter gives
 * that without an extra round-trip).
 *
 * - Main increments POSTED and calls `Atomics.notify` to dispatch a job.
 *   `POSTED === QUIT_SENTINEL` tells the worker to exit instead of treating
 *   it as a new job.
 * - The worker's loop tracks its own last-seen POSTED value; on wake, if
 *   the new POSTED differs from what it last processed (and isn't QUIT),
 *   it runs the job, writes STATUS, sets DONE to the SAME sequence number,
 *   and calls `Atomics.notify`.
 * - Main waits (`Atomics.wait`, bounded by an overall deadline — see
 *   `threaded.ts`'s "Crash detection" note for why a real-time
 *   `'exit'`/`'error'` event-based liveness check does NOT work reliably
 *   inside a synchronous `Atomics.wait` retry loop, empirically verified
 *   during this phase) until DONE reaches the sequence number it posted.
 *
 * ## Memory rule
 *
 * Every `Int32Array` used with `Atomics` here MUST be constructed fresh
 * from `memory.buffer` at each point of use — never cached across a call
 * boundary (the project-wide "never cache views" rule; doubly load-bearing
 * on the threads path per the spec's feasibility grounding: a cached
 * `.buffer` reference on a GROWABLE SHARED memory silently keeps its OLD
 * length with no error).
 */

// --- Control block cell indices (Int32, 4 bytes each) ----------------------

export const CB_POSTED = 0;
export const CB_DONE = 1;
export const CB_STATUS = 2;
export const CB_A_SHAPE_PTR = 3;
export const CB_A_RANK = 4;
export const CB_A_STRIDES_PTR = 5;
export const CB_A_OFFSET = 6;
export const CB_A_DATA_PTR = 7;
export const CB_A_DATA_LEN = 8;
export const CB_B_SHAPE_PTR = 9;
export const CB_B_RANK = 10;
export const CB_B_STRIDES_PTR = 11;
export const CB_B_OFFSET = 12;
export const CB_B_DATA_PTR = 13;
export const CB_B_DATA_LEN = 14;
export const CB_OUT_DATA_PTR = 15;
export const CB_OUT_LEN = 16;
export const CB_ROW_START = 17;
export const CB_ROW_END = 18;

/** Total control-block length in `i32` cells / bytes. */
export const CONTROL_BLOCK_CELLS = 19;
export const CONTROL_BLOCK_BYTES = CONTROL_BLOCK_CELLS * 4;

/** `POSTED` value that tells a worker to exit its job loop instead of
 * waiting for (or running) another job. Chosen as a value no real sequence
 * number ever reaches in practice (dispatch count is bounded by test/bench
 * runs, nowhere near 2^31), and negative so it can never collide with a
 * legitimate (non-negative, starts at 0) sequence number by construction. */
export const QUIT_SENTINEL = -1;

/** Reserved `STATUS` value a worker writes if a JS-level exception escapes
 * its job-processing code (never expected in normal operation — the kernel
 * call itself is infallible-in-the-JS-sense, only returns a `u32` status —
 * but defends against a bug in the bootstrap/marshalling code itself
 * hanging main instead of surfacing an error). Outside the ABI's own
 * 0..4 status range, so `threadedMatmul` can tell "a real kernel status"
 * from "the worker's OWN code broke" and report the latter distinctly. */
export const WORKER_JS_ERROR_STATUS = 999;

/** Per-worker private stack region size (`nt_alloc`'d on main before
 * spawning, per the spec: "1 MiB stack region"). */
export const WORKER_STACK_BYTES = 1024 * 1024;

// --- Shared cell accessors ---------------------------------------------
//
// Every one of these rebuilds its `Int32Array` view from `memory.buffer`
// on EVERY call — never cached — per the hard memory rule (module doc
// comment). Shared between `threaded.ts` (main) and `threaded-worker.ts`
// (worker) so both sides touch the control block through the exact same
// (and exactly as narrow) code path.

export function loadCell(memory: WebAssembly.Memory, ctrlPtr: number, idx: number): number {
  return Atomics.load(new Int32Array(memory.buffer, ctrlPtr, CONTROL_BLOCK_CELLS), idx);
}

export function storeCell(memory: WebAssembly.Memory, ctrlPtr: number, idx: number, value: number): void {
  Atomics.store(new Int32Array(memory.buffer, ctrlPtr, CONTROL_BLOCK_CELLS), idx, value);
}

export function notifyCell(memory: WebAssembly.Memory, ctrlPtr: number, idx: number): void {
  Atomics.notify(new Int32Array(memory.buffer, ctrlPtr, CONTROL_BLOCK_CELLS), idx, 1);
}

/** `timeoutMs === undefined` waits with no timeout (blocks until notified) —
 * used by the worker's own "wait for a job" loop, which should sleep
 * indefinitely while idle. Main's "wait for completion" side always passes
 * an explicit bounded `timeoutMs` (see `threaded.ts`'s deadline-based crash
 * detection). */
export function waitCell(
  memory: WebAssembly.Memory,
  ctrlPtr: number,
  idx: number,
  expected: number,
  timeoutMs?: number,
): "ok" | "not-equal" | "timed-out" {
  const view = new Int32Array(memory.buffer, ctrlPtr, CONTROL_BLOCK_CELLS);
  return timeoutMs === undefined ? Atomics.wait(view, idx, expected) : Atomics.wait(view, idx, expected, timeoutMs);
}
