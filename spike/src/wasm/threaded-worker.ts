/**
 * Kern 06: persistent worker bootstrap for the threaded WASM core. Spawned
 * by `threaded.ts`'s `initThreadedCore` with `workerData = { mod, memory,
 * stackTop, ctrlPtr }` (the compiled `WebAssembly.Module` and the shared
 * `WebAssembly.Memory` survive `worker_threads` structured clone with no
 * recompilation needed — confirmed in the Kern-06 feasibility spike).
 *
 * Sequence: instantiate the SAME module over the SAME shared memory, set
 * THIS instance's own private `__stack_pointer` (the per-instance shadow
 * stack the spike's PoC empirically validated — see
 * docs/kern-06-threads-spec.md's feasibility grounding: this is a design
 * convention, not a spec-guaranteed protocol), signal readiness via
 * `postMessage` (a one-time setup handshake — NOT part of the per-job
 * path, so using `postMessage`/Promises here doesn't conflict with the
 * job protocol's own synchronous Atomics requirement), then enter the
 * persistent Atomics job loop (see `threaded-protocol.ts` module doc for
 * the full sequence-number handshake).
 */
import { parentPort, workerData } from "node:worker_threads";
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
  loadCell,
  notifyCell,
  QUIT_SENTINEL,
  storeCell,
  waitCell,
  WORKER_JS_ERROR_STATUS,
} from "./threaded-protocol.ts";

/** `ambient.d.ts` declares `process` as `unknown` (only ever feature-detected
 * via `typeof`, per that file's own doc comment) — this worker script is
 * the one place in the codebase that genuinely needs `process.exit`, so a
 * local cast here is preferable to widening the shared ambient shim (which
 * every other file — Node AND the never-exercised browser path in
 * loader.ts — would then see too). */
function exitProcess(code: number): never {
  return (process as { exit(code: number): never }).exit(code);
}

interface WorkerInitData {
  readonly mod: WebAssembly.Module;
  readonly memory: WebAssembly.Memory;
  readonly stackTop: number;
  readonly ctrlPtr: number;
}

/** The one export this bootstrap calls directly — narrower than the full
 * `CoreExports` surface (this worker never touches any other op). */
interface ThreadedWasmExports {
  readonly __stack_pointer: WebAssembly.Global;
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

async function main(): Promise<void> {
  const { mod, memory, stackTop, ctrlPtr } = workerData as WorkerInitData;
  let instantiated = false;
  try {
    const instance = await WebAssembly.instantiate(mod, { env: { memory } });
    instantiated = true;
    const exports = instance.exports as unknown as ThreadedWasmExports;
    exports.__stack_pointer.value = stackTop;

    parentPort?.postMessage({ ready: true });

    let mySeq = 0;
    for (;;) {
      // No timeout: an idle persistent worker sleeps until notified (a real
      // OS-level block, no busy-waiting) — see threaded-protocol.ts.
      waitCell(memory, ctrlPtr, CB_POSTED, mySeq);
      const posted = loadCell(memory, ctrlPtr, CB_POSTED);
      if (posted === QUIT_SENTINEL) break;
      if (posted === mySeq) continue; // spurious wake guard; re-check and keep waiting

      const aShapePtr = loadCell(memory, ctrlPtr, CB_A_SHAPE_PTR);
      const aRank = loadCell(memory, ctrlPtr, CB_A_RANK);
      const aStridesPtr = loadCell(memory, ctrlPtr, CB_A_STRIDES_PTR);
      const aOffset = loadCell(memory, ctrlPtr, CB_A_OFFSET);
      const aDataPtr = loadCell(memory, ctrlPtr, CB_A_DATA_PTR);
      const aDataLen = loadCell(memory, ctrlPtr, CB_A_DATA_LEN);
      const bShapePtr = loadCell(memory, ctrlPtr, CB_B_SHAPE_PTR);
      const bRank = loadCell(memory, ctrlPtr, CB_B_RANK);
      const bStridesPtr = loadCell(memory, ctrlPtr, CB_B_STRIDES_PTR);
      const bOffset = loadCell(memory, ctrlPtr, CB_B_OFFSET);
      const bDataPtr = loadCell(memory, ctrlPtr, CB_B_DATA_PTR);
      const bDataLen = loadCell(memory, ctrlPtr, CB_B_DATA_LEN);
      const outDataPtr = loadCell(memory, ctrlPtr, CB_OUT_DATA_PTR);
      const outLen = loadCell(memory, ctrlPtr, CB_OUT_LEN);
      const rowStart = loadCell(memory, ctrlPtr, CB_ROW_START);
      const rowEnd = loadCell(memory, ctrlPtr, CB_ROW_END);

      const status = exports.nt_matmul_blocked_partial(
        aShapePtr,
        aRank,
        aStridesPtr,
        aOffset,
        aDataPtr,
        aDataLen,
        bShapePtr,
        bRank,
        bStridesPtr,
        bOffset,
        bDataPtr,
        bDataLen,
        outDataPtr,
        outLen,
        rowStart,
        rowEnd,
      );

      mySeq = posted;
      storeCell(memory, ctrlPtr, CB_STATUS, status);
      storeCell(memory, ctrlPtr, CB_DONE, mySeq);
      notifyCell(memory, ctrlPtr, CB_DONE);
    }
    exitProcess(0);
  } catch (err) {
    // Defense-in-depth: a JS-level exception here should never be possible
    // in normal operation (the ABI call itself only ever returns a `u32`
    // status, never throws), but if the bootstrap/marshalling code itself
    // has a bug, surface it through BOTH channels rather than hanging main:
    // (a) the control block, if instantiation got far enough for it to be
    // meaningful, so a worker that dies mid-job still unblocks main's
    // Atomics.wait promptly instead of only via the deadline; (b)
    // postMessage, for `initThreadedCore`'s own setup-phase await.
    if (instantiated) {
      try {
        const posted = loadCell(memory, ctrlPtr, CB_POSTED);
        storeCell(memory, ctrlPtr, CB_STATUS, WORKER_JS_ERROR_STATUS);
        storeCell(memory, ctrlPtr, CB_DONE, posted);
        notifyCell(memory, ctrlPtr, CB_DONE);
      } catch {
        // Nothing more we can do — main's deadline-based timeout is the backstop.
      }
    }
    parentPort?.postMessage({ ready: false, error: String((err as Error)?.stack ?? err) });
    exitProcess(1);
  }
}

void main();
