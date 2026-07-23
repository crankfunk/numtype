/**
 * Kern 06: threaded runtime layer — loads the threads WASM artifact
 * (`numtype_core_threads.wasm`), spawns a persistent `worker_threads` pool
 * over one shared `WebAssembly.Memory`, and exposes a synchronous
 * `threadedMatmul` that fans a matmul call's output ROWS out across the
 * pool via `nt_matmul_blocked_partial` (see docs/kern-06-threads-spec.md) —
 * except for calls below a measured work-volume threshold, which it routes
 * to the single-threaded kernel on the main thread instead (see
 * `threadedMatmul`'s "Size-based auto-routing" doc section).
 *
 * Everything else (lifecycle, views, slicing, non-matmul ops) stays on the
 * UNCHANGED `WNDArray` class from `resident.ts`, which works unmodified
 * over the synthesized `CoreExports` object below — those paths are always
 * single-threaded/main-thread, exactly as they are for the stable core.
 *
 * ## Crash detection (honesty note, verified empirically this phase)
 *
 * A worker dying mid-job is meant to "surface as a thrown Error on main
 * within finite time — never a hang" (spec). The obvious design —
 * register `'exit'`/`'error'` listeners on each `Worker` and check an
 * `alive` flag inside the wait-for-completion retry loop — does NOT work
 * reliably here: `threadedMatmul`'s dispatch/wait path is REQUIRED to be
 * synchronous (blocking `Atomics.wait`, by design, so callers never
 * `await` a matmul call). A synchronous JS call stack does not return
 * control to Node's event loop between iterations of a plain loop, even
 * one containing `Atomics.wait` calls — so a queued `'exit'` notification
 * for a worker that died mid-loop is NOT dispatched to its listener until
 * the whole synchronous call finishes, i.e. too late to react during that
 * same call. This was verified directly (a throwaway harness that killed
 * a worker via `process.exit()` at a known instant, then busy-looped on
 * `Atomics.wait` retries checking a listener-set flag: the flag never
 * became visible mid-loop, only after the loop's synchronous work
 * finished). The actual guarantee here is therefore a DEADLINE: main gives
 * each worker's job at most `matmulTimeoutMs` before throwing — this
 * bounds "finite time" but cannot, from inside one synchronous call,
 * distinguish "crashed" from "legitimately still running" any earlier
 * than that deadline. `'exit'`/`'error'` listeners are still registered
 * (and DO fire reliably between separate calls, once control returns to
 * the event loop) — they gate a cheap pre-dispatch liveness check (refuse
 * to dispatch into a worker already known dead from a PRIOR call) and
 * `dispose()`'s clean shutdown (which is genuinely async and has no such
 * timing problem).
 *
 * ## Deferred buffer freeing on a pool-compromising error (post-verify fix)
 *
 * A fresh-context verify pass found a real use-after-free: the original
 * code froze every dispatched worker's job into shared memory in one loop,
 * then waited for each worker's completion in index order in a SECOND
 * loop. If worker `i` was found dead (pre-dispatch check) or timed out
 * (wait loop), the code threw and immediately freed the per-call scratch
 * and output buffers — but workers with an index that hadn't been reached
 * yet (dispatch already posted their job; wait not yet attempted, or not
 * yet timed out) could STILL BE ACTIVELY WRITING into those exact buffers
 * on a separate OS thread. The verifier proved this empirically: the freed
 * output pointer got reused by the very next same-size `nt_alloc`, and the
 * still-computing worker's completion signal landed measurably AFTER the
 * free.
 *
 * The fix distinguishes two failure categories:
 * - **A worker completes with a non-zero kernel status, but every worker
 *   in this dispatch is otherwise healthy.** By construction, this is only
 *   observed AFTER the wait loop has confirmed every dispatched worker
 *   reached `DONE` (the status-check loop runs strictly after the
 *   wait-for-everyone loop) — so no worker can still be writing. Freeing
 *   synchronously, then throwing, is safe and keeps the deadline-bounded,
 *   synchronous contract intact.
 * - **A worker is found dead at dispatch time, or times out while
 *   waiting.** Pool integrity is compromised: some already-dispatched
 *   worker(s) may still be running. `ThreadedPool.poison()` is called:
 *   marks the pool permanently unusable for further `threadedMatmul`
 *   calls (`assertNotPoisoned()` — fails fast from then on), then
 *   kicks off an ASYNC cleanup that `Promise.all(workers.map(terminate))`s
 *   every worker (a terminated OS thread cannot write to shared memory
 *   anymore — Node's `terminate()` promise resolves only once the thread
 *   has actually stopped) and ONLY THEN frees the per-call buffers that
 *   were in flight. This cleanup is NOT awaited by the throwing call — the
 *   synchronous contract holds, the error surfaces immediately — but IS
 *   awaited by `dispose()` (and independently observable via
 *   `pool.poisonCleanup`), so callers that want a deterministic point to
 *   observe "everything is now actually freed" have one.
 */
import { availableParallelism } from "node:os";
import { readFile } from "node:fs/promises";
import { Worker } from "node:worker_threads";
import { type Guard, type OkShape } from "../ndarray.ts";
import { type Mutable, type Shape } from "../dim.ts";
import type { MatMul } from "../matmul.ts";
import type { CoreExports } from "./loader.ts";
import { planMatmul, squeezeMatmulShape, WNDArray, type WNDArrayDescriptor } from "./resident.ts";
import {
  CB_A_DATA_LEN,
  CB_A_DATA_PTR,
  CB_A_OFFSET,
  CB_A_RANK,
  CB_A_SHAPE_PTR,
  CB_A_STRIDES_PTR,
  CB_B_DATA_LEN,
  CB_B_DATA_PTR,
  CB_B_OFFSET,
  CB_B_RANK,
  CB_B_SHAPE_PTR,
  CB_B_STRIDES_PTR,
  CB_DONE,
  CB_OUT_DATA_PTR,
  CB_OUT_LEN,
  CB_POSTED,
  CB_ROW_END,
  CB_ROW_START,
  CB_STATUS,
  CONTROL_BLOCK_BYTES,
  loadCell,
  notifyCell,
  QUIT_SENTINEL,
  storeCell,
  waitCell,
  WORKER_JS_ERROR_STATUS,
  WORKER_STACK_BYTES,
} from "./threaded-protocol.ts";

const WASM_URL = new URL("./numtype_core_threads.wasm", import.meta.url);
const WORKER_URL = new URL("./threaded-worker.ts", import.meta.url);

/** MC tile size from `crates/core/src/kernels/matmul_blocked.rs` — a
 * documented cross-language constant duplication, performance-only: split
 * boundaries not aligned to it are still CORRECT (the parallel bit-identity
 * law holds for any partition), just possibly a little less cache-friendly
 * at the boundary. */
const MC = 32;

/** Default per-job deadline. Generous relative to any size this phase's own
 * benchmarks/tests exercise (n <= 1024); see the module doc's "Crash
 * detection" note for exactly what this bounds and why it can't be tighter
 * without breaking the synchronous main-thread contract. */
const DEFAULT_MATMUL_TIMEOUT_MS = 30_000;

/** The module's own declared minimum initial page count is 17 (verified
 * empirically: `WebAssembly.instantiate` throws a `LinkError` naming the
 * exact declared minimum if given fewer — "verified loudly at
 * instantiation" per the spec). `INITIAL_PAGES` carries headroom above
 * that; `MAX_PAGES` matches the module's own `--max-memory=1073741824`
 * link flag (1 GiB / 64 KiB pages). If a future build's static footprint
 * ever grows past `INITIAL_PAGES`, instantiation fails LOUDLY (a
 * `LinkError` naming the real minimum), never silently. */
const INITIAL_PAGES = 32;
const MAX_PAGES = 16384;

/** Round `x` DOWN to the nearest multiple of 16. The spec requires the
 * per-worker shadow-stack top to be "16-byte aligned downward"; `nt_alloc`
 * only guarantees 8-byte alignment (its own documented contract), and
 * `stackPtr + WORKER_STACK_BYTES` is therefore only as aligned as
 * `stackPtr` itself (`WORKER_STACK_BYTES` = 1 MiB is a multiple of 16, so
 * adding it never changes the alignment mod 16) — i.e. NOT guaranteed
 * 16-aligned without this explicit rounding. Rounding down always stays
 * inside the allocated `[stackPtr, stackPtr + WORKER_STACK_BYTES)` region
 * (loses at most 15 bytes off the top, negligible against 1 MiB). Exported
 * as a pure function so it's directly unit-testable (see threaded.test.ts).
 * `x >>> 0`: JS bitwise ops operate on 32-bit representations; our values
 * are always well under 2^31 (max memory is 1 GiB), so this is exact and
 * just makes the "unsigned 32-bit" contract explicit rather than implicit. */
export function alignDown16(x: number): number {
  return (x >>> 0) & ~15;
}

/** `CoreExports` plus the one export unique to the threads artifact. */
interface ThreadedCoreExports extends CoreExports {
  nt_matmul_blocked_partial(
    aShapePtr: number,
    aRank: number,
    aStridesPtr: number,
    aOffset: number,
    aDataPtr: number,
    aDataLen: number,
    bShapePtr: number,
    bRank: number,
    bStridesPtr: number,
    bOffset: number,
    bDataPtr: number,
    bDataLen: number,
    outDataPtr: number,
    outLen: number,
    rowStart: number,
    rowEnd: number,
  ): number;
}

interface PoolWorker {
  readonly worker: Worker;
  readonly ctrlPtr: number;
  readonly stackPtr: number;
  postedSeq: number;
  alive: boolean;
}

interface ScratchBuf {
  readonly ptr: number;
  readonly bytes: number;
}

/** Test-only observability hook, mirroring `resident.ts`'s
 * `getResidentFreeCount()` (same rationale: a deterministic leak/plateau
 * signal instead of only inferring frees from `memory.buffer.byteLength`,
 * which can never shrink). A pool's worker stacks/control-blocks are
 * ephemeral admin allocations (analogous to `resident.ts`'s own per-call
 * `ScratchBuf` pattern, not a refcounted `ResidentBuffer`), so they are a
 * DIFFERENT resource category from `getResidentFreeCount()` and get their
 * own counter rather than sharing that one. Incremented once per
 * (stack, control-block) pair actually freed in `dispose()`. */
let threadedPoolFreeCount = 0;

export function getThreadedPoolFreeCount(): number {
  return threadedPoolFreeCount;
}

/** Test-only observability hook for the DEFERRED per-call-buffer frees a
 * `ThreadedPool.poison()` cleanup performs (see module doc "Deferred
 * buffer freeing"). A SEPARATE counter from `getThreadedPoolFreeCount()`
 * (different resource: per-call scratch/output buffers, not per-worker
 * stack/control-block pairs) — incremented once per buffer actually freed
 * (`bytes !== 0`) once every worker has been confirmed terminated. */
let poisonCleanupFreeCount = 0;

export function getPoisonCleanupFreeCount(): number {
  return poisonCleanupFreeCount;
}

/**
 * A loaded threaded core + persistent worker pool. Fields beyond
 * `core`/`workerCount` are implementation details `threadedMatmul` (a free
 * function, not a method — see module doc) needs direct access to; they are
 * NOT part of this module's stable public contract.
 */
export class ThreadedPool {
  readonly core: ThreadedCoreExports;
  readonly memory: WebAssembly.Memory;
  readonly workerCount: number;
  readonly workers: PoolWorker[];
  readonly matmulTimeoutMs: number;
  private disposed = false;
  /** Set true the moment a worker is found dead or times out — the pool is
   * permanently unusable from then on (no auto-respawn; "static split
   * only," per the spec's scope). `dispatchAndRun` checks this via
   * `assertNotPoisoned()`. */
  private poisoned = false;
  /** The in-flight (or settled) async cleanup `poison()` kicked off:
   * terminate every worker, THEN free the per-call buffers that were in
   * flight at the time of poisoning. Public/readable (not just internal)
   * so callers/tests have a deterministic point to await besides
   * `dispose()` itself — see the module doc. `null` until the pool is
   * first poisoned. */
  poisonCleanup: Promise<void> | null = null;

  constructor(core: ThreadedCoreExports, memory: WebAssembly.Memory, workers: PoolWorker[], matmulTimeoutMs: number) {
    this.core = core;
    this.memory = memory;
    this.workers = workers;
    this.workerCount = workers.length;
    this.matmulTimeoutMs = matmulTimeoutMs;
  }

  get isPoisoned(): boolean {
    return this.poisoned;
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  /** Throws if `dispose()` already ran — fails fast rather than dispatching
   * into a pool whose worker threads/allocations are gone (undefined
   * behavior otherwise: workers may already be torn down, `nt_alloc`/
   * `nt_free` calls would race whatever `dispose()` itself is doing, and
   * any completion this reached would write into freed memory). Checked
   * BEFORE `assertNotPoisoned()` in `threadedMatmul` so the error names the
   * more advanced lifecycle stage when both are true (a poisoned pool that
   * was then disposed is, from the caller's perspective, simply disposed). */
  assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error(`ThreadedPool: this pool has been disposed and can no longer dispatch threadedMatmul — create a fresh pool.`);
    }
  }

  /** Throws if a prior call already found the pool compromised — fails
   * fast rather than dispatching into a pool with a permanently-dead
   * worker (which would just re-discover the same problem after paying
   * for another round of marshalling). Only gates `threadedMatmul`;
   * non-matmul `WNDArray` ops on this pool's `core` never touch worker
   * state and remain safe to use regardless. */
  assertNotPoisoned(): void {
    if (this.poisoned) {
      throw new Error(
        `ThreadedPool: this pool is poisoned (a worker crashed or timed out in a prior call) and can no longer dispatch ` +
          `threadedMatmul — dispose() it and create a fresh pool. (Non-matmul WNDArray ops on this pool's core are unaffected.)`,
      );
    }
  }

  /** Mark the pool permanently unusable and kick off the async
   * terminate-everyone-then-free-buffers cleanup described in the module
   * doc. Idempotent: a second call (e.g. dispatch AND wait both hitting a
   * problem in the same `dispatchAndRun` invocation — cannot happen since
   * the dispatch loop always fully precedes the wait loop, but defensive
   * regardless) returns the SAME cleanup promise rather than starting a
   * second termination race. Does not block the caller — the returned
   * promise is not awaited here; `dispose()` and `poisonCleanup` are the
   * documented ways to observe completion. */
  poison(buffersToFree: readonly ScratchBuf[]): Promise<void> {
    if (this.poisoned) return this.poisonCleanup ?? Promise.resolve();
    this.poisoned = true;
    this.poisonCleanup = (async () => {
      await Promise.all(
        this.workers.map(async (pw) => {
          try {
            await pw.worker.terminate();
          } catch {
            // Already gone (e.g. it crashed on its own) — fine, that's the
            // outcome we wanted anyway.
          }
        }),
      );
      // Every worker's underlying OS thread is now confirmed stopped
      // (terminate()'s promise only resolves once the thread has actually
      // exited) -- no thread can write to shared memory anymore, so
      // freeing now is safe.
      for (const buf of buffersToFree) {
        if (buf.bytes !== 0) {
          this.core.nt_free(buf.ptr, buf.bytes);
          poisonCleanupFreeCount++;
        }
      }
    })();
    return this.poisonCleanup;
  }

  /** Signal every still-alive worker to quit, await their exit (bounded —
   * falls back to `worker.terminate()` if a worker doesn't exit promptly),
   * then free every worker's stack + control-block allocation
   * (`getThreadedPoolFreeCount()` increments once per pair freed — the
   * deterministic leak-plateau signal). Safe to call more than once (a
   * no-op after the first call). If the pool was poisoned, `poison()`
   * already terminated every worker as part of its own cleanup — this
   * AWAITS that cleanup first (never races a free against a still-
   * terminating worker) instead of repeating the polite quit-signal dance
   * against threads that are already gone. Does NOT auto-respawn a worker
   * that died earlier — the caller disposes and creates a fresh pool. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;

    if (this.poisonCleanup) {
      await this.poisonCleanup.catch(() => {});
    } else {
      await Promise.all(
        this.workers.map(async (pw) => {
          if (!pw.alive) return;
          const exited = new Promise<void>((resolve) => {
            pw.worker.once("exit", () => resolve());
          });
          storeCell(this.memory, pw.ctrlPtr, CB_POSTED, QUIT_SENTINEL);
          notifyCell(this.memory, pw.ctrlPtr, CB_POSTED);
          const timedOut = new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 5000));
          const result = await Promise.race([exited.then(() => "exited" as const), timedOut]);
          if (result === "timeout") {
            await pw.worker.terminate();
          }
        }),
      );
    }

    for (const pw of this.workers) {
      this.core.nt_free(pw.ctrlPtr, CONTROL_BLOCK_BYTES);
      this.core.nt_free(pw.stackPtr, WORKER_STACK_BYTES);
      threadedPoolFreeCount++;
    }
  }
}

function clampPoolSize(requested: number | undefined): number {
  if (requested !== undefined) {
    if (!Number.isInteger(requested) || requested < 1) {
      throw new Error(`initThreadedCore: workers must be a positive integer, got ${requested}`);
    }
    return requested;
  }
  return Math.max(1, Math.min(availableParallelism() - 1, 8));
}

/**
 * Load `numtype_core_threads.wasm`, create a shared `WebAssembly.Memory`,
 * instantiate on main, synthesize a `CoreExports`-compatible object (so the
 * UNCHANGED `WNDArray` class works over it), and spawn a persistent worker
 * pool of `workers` threads (default: see `clampPoolSize`).
 *
 * `matmulTimeoutMs` (additive beyond the spec's literal `(workers?)`
 * signature — documented deviation, see the results doc): per-job deadline
 * for `threadedMatmul`'s wait-for-completion step (default
 * `DEFAULT_MATMUL_TIMEOUT_MS`). Exposed mainly so tests can exercise the
 * "worker dies mid-job -> Error, not a hang" path without a real 30s wait.
 */
export async function initThreadedCore(workers?: number, matmulTimeoutMs = DEFAULT_MATMUL_TIMEOUT_MS): Promise<ThreadedPool> {
  const bytes = (await readFile(WASM_URL)).slice();
  const mod = await WebAssembly.compile(bytes.buffer as ArrayBuffer);
  const memory = new WebAssembly.Memory({ initial: INITIAL_PAGES, maximum: MAX_PAGES, shared: true });

  let instance: WebAssembly.Instance;
  try {
    instance = await WebAssembly.instantiate(mod, { env: { memory } });
  } catch (e) {
    throw new Error(
      `initThreadedCore: failed to instantiate ${WASM_URL.href} against a shared memory of ${INITIAL_PAGES} pages — ` +
        `if the module's own declared minimum grew past that, bump INITIAL_PAGES in threaded.ts. Original error: ${(e as Error).message}`,
    );
  }
  // Cast straight to `ThreadedCoreExports` (deliberately NOT
  // `Omit<ThreadedCoreExports, "memory">`): the trailing `memory` in the object
  // literal already overrides whatever `memory` the spread carries, so this is
  // runtime-identical to an Omit-based spread. `Omit<T,K>` = `Pick<T, Exclude<
  // keyof T, K>>` re-walks `keyof ThreadedCoreExports` here, which measured at
  // +7 type-instantiations in EVERY compilation for each `CoreExports` member
  // added — a cost that compounds across the WASM-parity op campaign. The direct
  // cast eliminates it (measured 2026-07-23, WASM-parity S0/sqrt investigation).
  const core: ThreadedCoreExports = { ...(instance.exports as unknown as ThreadedCoreExports), memory } as ThreadedCoreExports;

  const poolSize = clampPoolSize(workers);
  const poolWorkers: PoolWorker[] = [];

  try {
    const readyPromises: Promise<void>[] = [];
    for (let i = 0; i < poolSize; i++) {
      const stackPtr = core.nt_alloc(WORKER_STACK_BYTES);
      if (stackPtr === 0) throw new Error(`initThreadedCore: nt_alloc failed for worker ${i}'s stack region`);
      const ctrlPtr = core.nt_alloc(CONTROL_BLOCK_BYTES);
      if (ctrlPtr === 0) throw new Error(`initThreadedCore: nt_alloc failed for worker ${i}'s control block`);
      storeCell(memory, ctrlPtr, CB_POSTED, 0);
      storeCell(memory, ctrlPtr, CB_DONE, 0);

      const stackTop = alignDown16(stackPtr + WORKER_STACK_BYTES);
      const worker = new Worker(WORKER_URL, {
        workerData: { mod, memory, stackTop, ctrlPtr },
      });
      const pw: PoolWorker = { worker, ctrlPtr, stackPtr, postedSeq: 0, alive: true };
      worker.on("exit", () => {
        pw.alive = false;
      });
      worker.on("error", () => {
        pw.alive = false;
      });
      poolWorkers.push(pw);

      readyPromises.push(
        new Promise<void>((resolve, reject) => {
          worker.once("message", (msg: { ready: boolean; error?: string }) => {
            if (msg.ready) resolve();
            else reject(new Error(`initThreadedCore: worker ${i} failed to initialize: ${msg.error}`));
          });
          worker.once("error", (e: Error) => reject(new Error(`initThreadedCore: worker ${i} errored during setup: ${String(e)}`)));
        }),
      );
    }
    await Promise.all(readyPromises);
  } catch (e) {
    await Promise.all(poolWorkers.map((pw) => pw.worker.terminate().catch(() => {})));
    for (const pw of poolWorkers) {
      core.nt_free(pw.ctrlPtr, CONTROL_BLOCK_BYTES);
      core.nt_free(pw.stackPtr, WORKER_STACK_BYTES);
    }
    throw e;
  }

  return new ThreadedPool(core, memory, poolWorkers, matmulTimeoutMs);
}

/** Split `[0, totalRows)` into exactly `workerCount` contiguous ranges,
 * MC-aligned where possible and round-robined over the pool (spec:
 * "splits the flat row space into MC-aligned contiguous chunks
 * round-robined over the pool"). Workers past the point where rows run out
 * get an empty `[totalRows, totalRows)` range — a valid no-op per the
 * partial kernel's own contract, deliberately exercised (not just assumed
 * safe) by the differential test suite for small-n/many-worker cases. */
function computeRowRanges(totalRows: number, workerCount: number): Array<readonly [number, number]> {
  const totalBlocks = Math.ceil(totalRows / MC);
  const activeWorkers = Math.max(1, Math.min(workerCount, totalBlocks));
  const blocksPerWorker = Math.ceil(totalBlocks / activeWorkers);
  const rowsPerWorker = blocksPerWorker * MC;

  const ranges: Array<readonly [number, number]> = [];
  let row = 0;
  for (let w = 0; w < workerCount; w++) {
    if (row >= totalRows) {
      ranges.push([totalRows, totalRows]);
      continue;
    }
    const end = Math.min(row + rowsPerWorker, totalRows);
    ranges.push([row, end]);
    row = end;
  }
  return ranges;
}

function writeU32Scratch(core: ThreadedCoreExports, memory: WebAssembly.Memory, values: readonly number[], scratch: ScratchBuf[]): number {
  const bytes = values.length * 4;
  if (bytes === 0) {
    scratch.push({ ptr: 0, bytes: 0 });
    return 0;
  }
  const ptr = core.nt_alloc(bytes);
  if (ptr === 0) throw new Error(`threadedMatmul: nt_alloc(${bytes}) failed for shape/stride scratch`);
  scratch.push({ ptr, bytes });
  new Uint32Array(memory.buffer, ptr, values.length).set(values);
  return ptr;
}

/**
 * Dispatch one `threadedMatmul` call's jobs across `pool`'s workers and
 * wait for completion. Owns ALL freeing of `scratch` (the shape/stride
 * marshalling buffers) and `outPtr`/`outBytes` (the output buffer) on
 * every path — see the module doc "Deferred buffer freeing" for exactly
 * which of the two freeing strategies (synchronous, or deferred via
 * `pool.poison()`) each throw site uses and why. On success, `scratch` is
 * freed (call-scoped only) but `outPtr` is NOT (it becomes the returned
 * `WNDArray`'s buffer — the caller, `threadedMatmul`, owns that decision).
 */
function dispatchAndRun(pool: ThreadedPool, da: WNDArrayDescriptor, db: WNDArrayDescriptor, outPtr: number, outBytes: number, outLen: number, totalRows: number): void {
  const ranges = computeRowRanges(totalRows, pool.workers.length);
  const scratch: ScratchBuf[] = [];

  const aShapePtr = writeU32Scratch(pool.core, pool.memory, da.shape, scratch);
  const aStridesPtr = writeU32Scratch(pool.core, pool.memory, da.strides, scratch);
  const bShapePtr = writeU32Scratch(pool.core, pool.memory, db.shape, scratch);
  const bStridesPtr = writeU32Scratch(pool.core, pool.memory, db.strides, scratch);

  const buffersInFlight: ScratchBuf[] = [...scratch, { ptr: outPtr, bytes: outBytes }];

  for (let i = 0; i < pool.workers.length; i++) {
    const pw = pool.workers[i]!;
    if (!pw.alive) {
      // Pool integrity compromised: workers 0..i-1 (if any) already have a
      // job posted and may still be actively writing into `buffersInFlight`
      // on their own OS threads. Defer freeing to the async cleanup —
      // never free synchronously here. The throw itself is still
      // synchronous/immediate (poison() does not block on its own cleanup).
      void pool.poison(buffersInFlight);
      throw new Error(
        `threadedMatmul: worker ${i} is not alive (it crashed or exited before this call) — pool is now poisoned; ` +
          `shared buffers will be freed once every worker is confirmed terminated (see pool.poisonCleanup)`,
      );
    }
    const range = ranges[i]!;
    const ctrlPtr = pw.ctrlPtr;
    storeCell(pool.memory, ctrlPtr, CB_A_SHAPE_PTR, aShapePtr);
    storeCell(pool.memory, ctrlPtr, CB_A_RANK, da.shape.length);
    storeCell(pool.memory, ctrlPtr, CB_A_STRIDES_PTR, aStridesPtr);
    storeCell(pool.memory, ctrlPtr, CB_A_OFFSET, da.offset);
    storeCell(pool.memory, ctrlPtr, CB_A_DATA_PTR, da.ptr);
    storeCell(pool.memory, ctrlPtr, CB_A_DATA_LEN, da.lenElems);
    storeCell(pool.memory, ctrlPtr, CB_B_SHAPE_PTR, bShapePtr);
    storeCell(pool.memory, ctrlPtr, CB_B_RANK, db.shape.length);
    storeCell(pool.memory, ctrlPtr, CB_B_STRIDES_PTR, bStridesPtr);
    storeCell(pool.memory, ctrlPtr, CB_B_OFFSET, db.offset);
    storeCell(pool.memory, ctrlPtr, CB_B_DATA_PTR, db.ptr);
    storeCell(pool.memory, ctrlPtr, CB_B_DATA_LEN, db.lenElems);
    storeCell(pool.memory, ctrlPtr, CB_OUT_DATA_PTR, outPtr);
    storeCell(pool.memory, ctrlPtr, CB_OUT_LEN, outLen);
    storeCell(pool.memory, ctrlPtr, CB_ROW_START, range[0]);
    storeCell(pool.memory, ctrlPtr, CB_ROW_END, range[1]);

    const nextSeq = pw.postedSeq + 1;
    pw.postedSeq = nextSeq;
    storeCell(pool.memory, ctrlPtr, CB_POSTED, nextSeq);
    notifyCell(pool.memory, ctrlPtr, CB_POSTED);
  }

  // Every worker has now been dispatched a job. Wait for each in turn
  // (they run concurrently regardless of the order we collect completion
  // in). A timeout at worker i means workers i+1..N-1 (already dispatched,
  // not yet reached in THIS loop) may still be actively writing — same
  // deferred-free reasoning as the dead-worker branch above.
  const statuses: number[] = new Array(pool.workers.length).fill(-1);
  for (let i = 0; i < pool.workers.length; i++) {
    const pw = pool.workers[i]!;
    const nextSeq = pw.postedSeq;
    const deadline = Date.now() + pool.matmulTimeoutMs;
    for (;;) {
      const done = loadCell(pool.memory, pw.ctrlPtr, CB_DONE);
      if (done === nextSeq) break;
      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        void pool.poison(buffersInFlight);
        throw new Error(
          `threadedMatmul: worker ${i} did not complete within ${pool.matmulTimeoutMs}ms (job seq ${nextSeq}) — ` +
            `possible crash or an unexpectedly large job (see threaded.ts's "Crash detection" module-doc note); ` +
            `pool is now poisoned, shared buffers will be freed once every worker is confirmed terminated (see pool.poisonCleanup)`,
        );
      }
      waitCell(pool.memory, pw.ctrlPtr, CB_DONE, done, remaining);
    }
    statuses[i] = loadCell(pool.memory, pw.ctrlPtr, CB_STATUS);
  }

  // Every dispatched worker has now reached DONE — no thread can still be
  // writing into scratch/outPtr (a worker only touches them again once
  // dispatched a NEW job, which cannot happen until the NEXT
  // threadedMatmul call). Safe to free synchronously on every remaining
  // path, success or a non-zero-status failure alike.
  for (let i = 0; i < statuses.length; i++) {
    const status = statuses[i]!;
    if (status === WORKER_JS_ERROR_STATUS || status !== 0) {
      for (const s of scratch) {
        if (s.bytes !== 0) pool.core.nt_free(s.ptr, s.bytes);
      }
      if (outBytes !== 0) pool.core.nt_free(outPtr, outBytes);
      if (status === WORKER_JS_ERROR_STATUS) {
        throw new Error(`threadedMatmul: worker ${i} hit an internal (non-kernel) error while processing its job`);
      }
      const range = ranges[i]!;
      throw new Error(`threadedMatmul: worker ${i}'s nt_matmul_blocked_partial returned status ${status} for rows [${range[0]},${range[1]})`);
    }
  }

  for (const s of scratch) {
    if (s.bytes !== 0) pool.core.nt_free(s.ptr, s.bytes);
  }
}

/**
 * Work-volume threshold (in multiply-accumulate operations, batch·m·k·n) at
 * or above which `threadedMatmul` dispatches through the worker pool; below
 * it, the call runs the single-threaded `nt_matmul_blocked` kernel on the
 * MAIN thread instead (same core, same memory, and — by the parallel
 * bit-identity law — the identical bits; the pool path IS row-partitioned
 * `nt_matmul_blocked`, so the two routes can never disagree).
 *
 * MEASURED, not guessed (FOLLOWUPS item: "Schwelle MESSEN statt raten"), via
 * `pnpm bench:crossover` (spike/bench-core/threaded-crossover.ts) on the
 * reference machine (M-series MacBook, 8 logical cores / 4 performance
 * cores, near-idle host, two runs), 2026-07-10. The measured picture —
 * NOTABLY different from what the Kern-06 Series-B bench suggested, see
 * docs/kern-06-ergebnisse.md (auto-routing addendum) for why the two
 * measurements disagree and the full tables:
 * - at/below ~0.03 Mops (square n<=32) the MAIN route wins decisively: the
 *   pool's per-call dispatch/wait round trip costs ~13–40µs (grows with
 *   worker count) and is 1.8–21× slower than just computing on main;
 * - ~0.11 Mops (n=48) is a wash (pool/main 0.99–1.16 across 2/4/8 workers);
 * - from 0.26 Mops (= 64³, n=64) upward the POOL route wins for every
 *   measured worker count and shape family (square, wide-n, deep-k, tall-m,
 *   batched; both runs agree) — including n=64, where the Kern-06 bench had
 *   reported threads losing. That older reading compared end-to-end calls
 *   (fromArray/toArray marshalling included) against the STABLE core; the
 *   router's actual choice is between two routes on the SAME threads core,
 *   and on that comparison the pool is already ahead at n=64. Worst honest
 *   case above the threshold: a single-MC-block call (rows < 32 -> one
 *   active worker) essentially ties (0.97–1.02; one unreproduced 1.30
 *   host-noise outlier in run 2, disclosed in the results-doc addendum).
 *
 * The cut sits at the TOP of the 0.11–0.26 Mops indifference band because
 * the risk is asymmetric: routing tiny calls through the pool loses BIG
 * relatively (up to ~21×), while routing a band-edge call to main costs a
 * few percent at most. 262_144 = 64³ exactly: the smallest measured volume
 * where the pool reliably wins keeps its win (>= dispatches to the pool).
 *
 * The criterion is WORK VOLUME, not matrix side length or row count: a
 * [1, 2048]·[2048, 2048] call has ONE output row but 4.2 Mops of work
 * (pool-worthy), and a [2048, 8]·[8, 8] call has 2048 rows of trivial work
 * (0.13 Mops — main-worthy); n alone or rows alone would misroute both.
 */
export const THREADED_MATMUL_MIN_POOL_WORK = 262_144;

export interface ThreadedMatmulOptions {
  /** Override for `THREADED_MATMUL_MIN_POOL_WORK` (same semantics/units:
   * dispatch through the pool iff batch·m·k·n >= this value). Tests use the
   * two extremes: `0` forces every nonempty call through the pool (the
   * differential suite must keep exercising the REAL worker-dispatch path —
   * with the default threshold, its deliberately small shapes would silently
   * all run on main and prove nothing about the pool); `Infinity` forces the
   * main-thread route. */
  readonly minPoolWork?: number;
}

/**
 * Threaded twin of `WNDArray.matmul` (resident.ts), for `WNDArray`s bound to
 * a `ThreadedPool`'s core. Mirrors the resident matmul semantics exactly —
 * 1-D promotion, batch broadcast, final squeeze — by reusing `planMatmul`/
 * `squeezeMatmulShape` (resident.ts, Kern 06 additions), the SAME pure
 * shape logic `WNDArray.matmul` itself uses inline. Takes the pool
 * explicitly as its first argument (a deliberate, documented deviation from
 * the spec's literal `threadedMatmul(a, b)` phrasing — see the results doc
 * — matching the existing `WNDArray.zeros(core, shape)`/`fromArray(core,
 * shape, values)` precedent of taking `core` explicitly rather than
 * inferring it).
 *
 * ## Size-based auto-routing (FOLLOWUPS follow-up to Kern 06)
 *
 * Calls whose work volume (batch·m·k·n, see `THREADED_MATMUL_MIN_POOL_WORK`)
 * is below the threshold are routed to the single-threaded
 * `nt_matmul_blocked` kernel on the MAIN thread — literally `a.matmul(b)`,
 * the unchanged `WNDArray` method, over this pool's core — instead of paying
 * the pool's per-call Atomics dispatch/wait round trips (~13–40µs, measured;
 * up to ~21× slower than main for an 8×8 matmul). Both routes produce
 * bit-identical results by construction (the pool path is row-partitioned
 * `nt_matmul_blocked`; parallel bit-identity law, Kern 06). The lifecycle
 * contract stays SIZE-INDEPENDENT on purpose: a disposed or poisoned pool
 * throws for every `threadedMatmul` call, including ones the router would
 * have run on main — callers get one predictable contract, not one that
 * silently flips with operand size. `opts.minPoolWork` overrides the
 * threshold per call (`0` = force pool, `Infinity` = force main — the
 * differential tests use both to pin each route explicitly).
 *
 * Empty row spaces / size-0 output shapes short-circuit on main (allocate,
 * possibly a zero-length buffer, and return) without dispatching a single
 * job. Throws immediately (before allocating anything) if the pool was
 * poisoned by a prior call's crash/timeout. Throws immediately (before
 * touching anything else) if the pool was already disposed, or poisoned by
 * an earlier call.
 */
export function threadedMatmul<S extends Shape, B extends Shape>(
  pool: ThreadedPool,
  a: WNDArray<S>,
  b: Guard<MatMul<S, B>, WNDArray<B>>,
  opts?: ThreadedMatmulOptions,
): WNDArray<OkShape<MatMul<S, B>>> {
  pool.assertNotDisposed();
  pool.assertNotPoisoned();

  const bb = b as unknown as WNDArray<B>;
  const da = a.describe();
  const db = bb.describe();
  if (da.core !== pool.core) {
    throw new Error(`threadedMatmul: first operand is not bound to this pool's threaded core`);
  }
  if (db.core !== pool.core) {
    throw new Error(`threadedMatmul: second operand is not bound to this pool's threaded core`);
  }

  const plan = planMatmul(da.shape, da.strides, db.shape, db.strides);
  const totalRows = plan.batchOut.reduce((acc, d) => acc * d, 1) * plan.m;

  // Auto-routing (see module doc): totalRows already includes the batch
  // product, so totalRows·k·n IS batch·m·k·n. `planMatmul` has already
  // thrown on any shape error at this point, so both routes reject invalid
  // inputs identically (with the same messages — `matmul()` runs the same
  // checks). Strictly below the threshold -> single-threaded main-thread
  // kernel; at/above -> pool dispatch below.
  const minPoolWork = opts?.minPoolWork ?? THREADED_MATMUL_MIN_POOL_WORK;
  if (totalRows * plan.k * plan.n < minPoolWork) {
    return a.matmul(b);
  }

  const outBytes = plan.outLen * 8;
  const outPtr = outBytes === 0 ? 0 : pool.core.nt_alloc(outBytes);
  if (outPtr === 0 && outBytes !== 0) {
    throw new Error(`threadedMatmul: nt_alloc(${outBytes}) failed (out of memory)`);
  }

  if (totalRows > 0) {
    // dispatchAndRun owns ALL buffer freeing on every path (success and
    // failure alike) — no catch-and-free here (that was the bug: freeing
    // in a blanket wrapper regardless of whether other workers were still
    // mid-flight). See its own doc comment and the module doc.
    dispatchAndRun(pool, da, db, outPtr, outBytes, plan.outLen, totalRows);
  }

  const finalShape = squeezeMatmulShape(plan);
  return WNDArray.fresh<OkShape<MatMul<S, B>>>(pool.core, finalShape as OkShape<MatMul<S, B>>, outPtr, plan.outLen);
}

/**
 * Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md, D2): the
 * threaded performance-backend facade. Wraps a `ThreadedPool` and hands its
 * `pool.core` straight through to the EXISTING `WNDArray.*(core, ...)`
 * statics for `fromArray`/`zeros`/`ones` — the created arrays are normal
 * `WNDArray`s (their instance methods run single-threaded on the main
 * thread, as today); only `matmul` is parallel, via this backend method
 * delegating to `threadedMatmul` (the free function above, unchanged) — the
 * bewusst offengelegte Inkonsistenz D2 documents (`backend.matmul(a, b)`,
 * not `a.matmul(b)`).
 *
 * Lives in THIS module (not `backend-api.ts`) so its value export stays
 * behind `threaded.ts`'s own dynamic-import boundary: `ndarray.ts` only
 * ever `import type`s `ThreadedBackend`, and `NDArray.backend("threaded")`
 * dynamically `import()`s this whole module — including this class's
 * constructor code — strictly AFTER the D2 env check passes, so the
 * browser-safe `NDArray` default and `backend("wasm")` never pull in this
 * file's top-level `node:os`/`node:fs/promises`/`node:worker_threads`
 * imports.
 *
 * Poisoned-pool behavior at this facade (Kern-06 semantics, unchanged —
 * `ThreadedPool.assertNotPoisoned()`'s own doc comment): once a worker
 * crashes or times out mid-call, the pool is permanently poisoned and
 * `matmul` throws the poisoned-pool message on every subsequent call —
 * `threadedMatmul` gates on `pool.assertNotPoisoned()` before dispatching.
 * `fromArray`/`zeros`/`ones` are UNAFFECTED: they go straight through
 * `pool.core` to the existing `WNDArray.*` statics, which never touch worker
 * state, so non-matmul ops on this backend stay usable after a poison.
 * `dispose()` still cleans up correctly either way — `ThreadedPool.dispose()`
 * awaits the in-flight poison cleanup instead of repeating worker teardown.
 */
export class ThreadedBackend {
  readonly pool: ThreadedPool;
  private readonly defaultMinPoolWork: number | undefined;
  private disposed = false;

  constructor(pool: ThreadedPool, defaultMinPoolWork?: number) {
    this.pool = pool;
    this.defaultMinPoolWork = defaultMinPoolWork;
  }

  private assertLive(op: string): void {
    if (this.disposed) {
      throw new Error(`ThreadedBackend.${op}: backend has been disposed`);
    }
  }

  fromArray<const S extends Shape>(shape: S, values: readonly number[] | Float64Array): WNDArray<Mutable<S>> {
    this.assertLive("fromArray");
    return WNDArray.fromArray(this.pool.core, shape, values);
  }

  zeros<const S extends Shape>(shape: S): WNDArray<Mutable<S>> {
    this.assertLive("zeros");
    return WNDArray.zeros(this.pool.core, shape);
  }

  ones<const S extends Shape>(shape: S): WNDArray<Mutable<S>> {
    this.assertLive("ones");
    return WNDArray.ones(this.pool.core, shape);
  }

  /** Parallel matmul (D2): delegates to `threadedMatmul(this.pool, a, b,
   * opts)`, including its size-based auto-routing. `opts.minPoolWork`
   * overrides this backend's own default (set at
   * `NDArray.backend("threaded", { minPoolWork })` construction time,
   * detail decision 2), which in turn overrides `threadedMatmul`'s own
   * module-level `THREADED_MATMUL_MIN_POOL_WORK` default. */
  matmul<S extends Shape, B extends Shape>(
    a: WNDArray<S>,
    b: Guard<MatMul<S, B>, WNDArray<B>>,
    opts?: ThreadedMatmulOptions,
  ): WNDArray<OkShape<MatMul<S, B>>> {
    this.assertLive("matmul");
    const minPoolWork = opts?.minPoolWork ?? this.defaultMinPoolWork;
    return threadedMatmul(this.pool, a, b, minPoolWork !== undefined ? { minPoolWork } : undefined);
  }

  /** Releases the pool (D1: "backend.dispose() gibt den core/Pool frei") —
   * delegates to `ThreadedPool.dispose()` (Kern-06 lifecycle contract
   * unchanged: async, idempotent, awaits worker shutdown/poison cleanup).
   * Does NOT dispose any `WNDArray` created through this backend — those
   * keep their own explicit `dispose()` (D1: the facade never hides WASM
   * memory management). */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.pool.dispose();
  }
}
