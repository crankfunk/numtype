/**
 * OOM-path hardening test for the v1 copy-based backend (`spike/src/wasm/
 * backend.ts`). Before this hardening, an op's scratch allocations
 * (writeShape/writeData/allocBytes) all happened before a single try/
 * finally; if a LATER allocation in the same call failed (`nt_alloc`
 * returning 0 -> "out of memory"), every scratch buffer allocated so far
 * for that call would leak (finally never ran for them). The fix (mirrors
 * the pattern already used in `resident.ts`) moves every allocation inside
 * the try, pushing each buffer onto a local `scratch: ScratchBuf[]`
 * immediately after it succeeds, so a single `finally` frees exactly the
 * buffers that were actually allocated — including partial marshalling on
 * an OOM.
 *
 * Real WASM OOM (an actual `memory.grow` failure) cannot be forced
 * deterministically from a test, so this file mocks the `CoreExports` ABI
 * directly: a bump allocator over a real backing `ArrayBuffer`, configurable
 * to fail the Nth `nt_alloc` call by returning 0, with `nt_free` recording
 * every `(ptr, bytes)` pair it's asked to free so alloc/free pairing can be
 * checked exactly (no leaks, no double-frees). This is the deterministic
 * proof; it's a conscious project decision not to attempt real WASM OOM
 * here.
 */
import assert from "node:assert";
import { test } from "node:test";
import { wasmAdd, wasmMatmul, wasmSum, wasmTranspose } from "../src/wasm/backend.ts";
import type { CoreExports } from "../src/wasm/loader.ts";

interface AllocRecord {
  readonly ptr: number;
  readonly bytes: number;
}

interface MockCore {
  readonly core: CoreExports;
  readonly allocs: AllocRecord[];
  readonly frees: AllocRecord[];
}

/** A stub for every `nt_*` kernel entry point backend.ts's v1 ops call:
 * returns status 0 (OK) normally, but throws if invoked AFTER an
 * `nt_alloc` failure already happened for this mock core — a kernel call
 * should never be reached once an earlier scratch allocation in the same
 * op has failed, since `allocBytes` throws immediately. This is a
 * defensive check on top of the physical control-flow guarantee. */
function kernelStub(name: string, failed: { value: boolean }): (...args: number[]) => number {
  return (): number => {
    if (failed.value) {
      throw new Error(`mock: ${name} invoked after an nt_alloc failure — an OOM should have aborted first`);
    }
    return 0;
  };
}

/** Strided/blocked entry points are Kern-03/04 additions that backend.ts's
 * v1 copy-based ops never call — required by the `CoreExports` interface,
 * but should be unreachable from any of the ops under test here. */
function notImplemented(name: string): (...args: number[]) => number {
  return (): number => {
    throw new Error(`mock: ${name} is not used by backend.ts's v1 ops and should never be called`);
  };
}

/** Build a fresh mock `CoreExports`. If `failAtCall` is given, the
 * `failAtCall`-th call to `nt_alloc` (1-indexed, across the whole op)
 * returns 0 ("out of memory"); every other call succeeds via a simple bump
 * allocator over a real 1 MiB backing buffer. */
function makeMockCore(failAtCall?: number): MockCore {
  const buffer = new ArrayBuffer(1 << 20); // 1 MiB — ample for these small test shapes
  const allocs: AllocRecord[] = [];
  const frees: AllocRecord[] = [];
  let bump = 8; // keep pointer 0 reserved (matches allocBytes' "ptr === 0 => OOM" check)
  let callCount = 0;
  const failed = { value: false };

  function nt_alloc(bytes: number): number {
    callCount++;
    if (failAtCall !== undefined && callCount === failAtCall) {
      failed.value = true;
      return 0;
    }
    const ptr = bump;
    bump += bytes === 0 ? 8 : bytes;
    if (bump > buffer.byteLength) {
      throw new Error(`mock backing buffer too small (need ${bump} bytes) — raise the mock's buffer size`);
    }
    allocs.push({ ptr, bytes });
    return ptr;
  }

  function nt_free(ptr: number, bytes: number): void {
    frees.push({ ptr, bytes });
  }

  const core: CoreExports = {
    memory: { buffer } as WebAssembly.Memory,
    nt_alloc,
    nt_free,
    nt_add: kernelStub("nt_add", failed),
    nt_matmul: kernelStub("nt_matmul", failed),
    nt_sum_all: kernelStub("nt_sum_all", failed),
    nt_sum_axis: kernelStub("nt_sum_axis", failed),
    nt_transpose: kernelStub("nt_transpose", failed),
    nt_fill: kernelStub("nt_fill", failed),
    nt_add_strided: notImplemented("nt_add_strided"),
    nt_matmul_strided: notImplemented("nt_matmul_strided"),
    nt_sum_all_strided: notImplemented("nt_sum_all_strided"),
    nt_sum_axis_strided: notImplemented("nt_sum_axis_strided"),
    nt_materialize: notImplemented("nt_materialize"),
    nt_matmul_blocked: notImplemented("nt_matmul_blocked"),
    nt_sub_strided: notImplemented("nt_sub_strided"),
    nt_mul_strided: notImplemented("nt_mul_strided"),
    nt_div_strided: notImplemented("nt_div_strided"),
    nt_dot_strided: notImplemented("nt_dot_strided"),
    nt_norm_sq_strided: notImplemented("nt_norm_sq_strided"),
    // WASM parity S0 (docs/wasm-parity-sqrt-spec.md): not used by backend.ts's
    // v1 ops either (sqrt only exists on the resident/threaded backends).
    nt_sqrt_strided: notImplemented("nt_sqrt_strided"),
  };

  return { core, allocs, frees };
}

/** Every allocation recorded must have been freed exactly once — no leaks,
 * no double-frees, no unexplained extra frees. */
function assertAllPaired(allocs: readonly AllocRecord[], frees: readonly AllocRecord[], ctx: string): void {
  assert.strictEqual(frees.length, allocs.length, `${ctx}: expected ${allocs.length} frees, got ${frees.length}`);
  const remaining = [...frees];
  for (const a of allocs) {
    const idx = remaining.findIndex((f) => f.ptr === a.ptr && f.bytes === a.bytes);
    assert.ok(idx !== -1, `${ctx}: allocation {ptr:${a.ptr}, bytes:${a.bytes}} was never freed`);
    remaining.splice(idx, 1);
  }
  assert.strictEqual(remaining.length, 0, `${ctx}: unexpected/extra frees: ${JSON.stringify(remaining)}`);
}

interface OpCase {
  readonly name: string;
  readonly allocCount: number;
  readonly call: (core: CoreExports) => void;
}

const cases: OpCase[] = [
  {
    name: "wasmAdd",
    allocCount: 5, // aShapeBuf, bShapeBuf, aDataBuf, bDataBuf, outDataBuf
    call: (core) => {
      wasmAdd(core, [2, 3], new Float64Array(6), [2, 3], new Float64Array(6));
    },
  },
  {
    name: "wasmMatmul",
    allocCount: 5, // aShapeBuf, bShapeBuf, aDataBuf, bDataBuf, outDataBuf
    call: (core) => {
      wasmMatmul(core, [2, 3], new Float64Array(6), [3, 2], new Float64Array(6));
    },
  },
  {
    name: "wasmSum (axis=undefined, nt_sum_all)",
    allocCount: 2, // aDataBuf, outDataBuf
    call: (core) => {
      wasmSum(core, [2, 3], new Float64Array(6), undefined);
    },
  },
  {
    name: "wasmSum (axis=0, nt_sum_axis)",
    allocCount: 3, // shapeBuf, dataBuf, outDataBuf
    call: (core) => {
      wasmSum(core, [2, 3], new Float64Array(6), 0);
    },
  },
  {
    name: "wasmTranspose",
    allocCount: 3, // shapeBuf, dataBuf, outDataBuf
    call: (core) => {
      wasmTranspose(core, [2, 3], new Float64Array(6));
    },
  },
];

for (const { name, allocCount, call } of cases) {
  for (let failAt = 1; failAt <= allocCount; failAt++) {
    test(`${name}: nt_alloc failure at call ${failAt}/${allocCount} throws "out of memory" and frees exactly the ${failAt - 1} prior allocations`, () => {
      const { core, allocs, frees } = makeMockCore(failAt);
      assert.throws(() => call(core), /out of memory/);
      assert.strictEqual(allocs.length, failAt - 1, `${name} failAt=${failAt}: expected ${failAt - 1} successful allocations before the failure`);
      assertAllPaired(allocs, frees, `${name} failAt=${failAt}`);
    });
  }

  test(`${name}: success path frees all ${allocCount} scratch buffers exactly once`, () => {
    const { core, allocs, frees } = makeMockCore(undefined);
    call(core);
    assert.strictEqual(allocs.length, allocCount, `${name}: expected exactly ${allocCount} allocations`);
    assertAllPaired(allocs, frees, `${name} success`);
  });
}
