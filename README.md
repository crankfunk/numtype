```
███╗   ██╗██╗   ██╗███╗   ███╗████████╗██╗   ██╗██████╗ ███████╗
████╗  ██║██║   ██║████╗ ████║╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝
██╔██╗ ██║██║   ██║██╔████╔██║   ██║    ╚████╔╝ ██████╔╝█████╗
██║╚██╗██║██║   ██║██║╚██╔╝██║   ██║     ╚██╔╝  ██╔═══╝ ██╔══╝
██║ ╚████║╚██████╔╝██║ ╚═╝ ██║   ██║      ██║   ██║     ███████╗
╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝   ╚═╝      ╚═╝   ╚═╝     ╚══════╝
```

> NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors.

**Status: v0.1 research preview** · Apache-2.0 · zero runtime dependencies

An n-dimensional array library for TypeScript with **compile-time shape checking**. `matmul`,
broadcasting, and reduction shape mismatches appear as editor squiggles *while you type* — not
as runtime crashes in production. It's gradual by design: literal dimensions are checked
statically, and dynamic (`number`) dimensions degrade gracefully to runtime checks, so it stays
usable in real code. The numeric backend is from-scratch Rust/WASM (opt-in); the default path is
a pure-JS reference runtime that is safe to run in the browser.

```ts
import { NDArray } from "numtype";

const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]); // NDArray<[2, 3]>
const c = a.matmul(NDArray.zeros([3, 4]));                // NDArray<[2, 4]> — hover shows it

a.matmul(NDArray.zeros([5, 4]));
//        ~~~~~~~~~~~~~~~~~~~~~ matmul: inner dimensions must match — got [2, 3] and [5, 4]
```

The red squiggle is the product; the WASM performance is a credibility feature, not the reason
to reach for it. If you've ever shipped a tensor-shape bug to production, this is the pitch.

## The core idea: the type system does arithmetic over your dimensions

Shapes aren't just *matched* — where dimensions are literal, they are *computed*, while you type:

```ts
const a = NDArray.zeros([1024]);

const b = a.slice({ start: 100, stop: 1000 });
//    ^ hover shows: NDArray<[900]>  — 1000 − 100, computed by the type checker

a.slice(1024);
//      ~~~~ compile error: slice: index 1024 is out of bounds for axis 0
//           with dim 1024 (shape [1024]) — a guaranteed runtime crash,
//           surfaced while typing (integer indices throw; ranges clamp,
//           exactly like NumPy — so ranges are computed, never rejected)
```

The received wisdom is that TypeScript can't do arithmetic over values this large: the standard
tuple-length encoding costs one recursion step *per unit of the value*, and the checker's
recursion ceiling (~1000) puts `1000 − 100` structurally out of reach. The trick is a change of
representation, not a broken limit: a literal number type becomes its decimal **digit string**
via template-literal types (`` `${1000}` `` → `"1000"`), and the arithmetic is schoolbook
subtraction with borrow, digit by digit — **O(digit count) instead of O(value), ~7 recursion
steps instead of ~1,000,000 for a 7-digit dimension**. Bounds checks come almost free on top:
they only need the comparison half of that machinery, which is why negative indices
(`a.slice(-2)`) are checked too. Everything outside the provable literal subset degrades honestly
to `number` + runtime checks — never a false error.

## Install

```sh
pnpm add numtype     # or: npm install numtype / yarn add numtype
```

Zero runtime dependencies; the `.wasm` core is bundled, so there is no native toolchain to
install. The default `NDArray` runs in pure JS and works in any modern JS environment (Node or
browser). The optional WASM/threaded backends need Node (see [Backends](#backends)). ESM only.

## Usage

Every operation below is exercised bit-for-bit against the WASM backend in [`spike/demo.ts`](spike/demo.ts)
(`pnpm demo`), so these examples run verbatim.

```ts
import { NDArray } from "numtype";

// Construction — the shape is inferred from the literal you pass.
const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]); // NDArray<[2, 3]>
const b = NDArray.fromArray([3], [10, 20, 30]);          // NDArray<[3]>

// Broadcasting add: [2, 3] + [3] -> [2, 3]
const sum = a.add(b);            // NDArray<[2, 3]>  ([[11,22,33],[14,25,36]])

// Matrix multiply: [2, 3] @ [3, 2] -> [2, 2]
const m2 = NDArray.fromArray([3, 2], [7, 8, 9, 10, 11, 12]);
const product = a.matmul(m2);    // NDArray<[2, 2]>  ([[58,64],[139,154]])

// Reduction along an axis: [2, 3] -> [3]
const colSums = a.sum(0);        // NDArray<[3]>
const kept = a.sum(0, true);     // NDArray<[1, 3]>   (keepdims)

// O(1) transpose (view): [2, 3] -> [3, 2]
const t = a.transpose();         // NDArray<[3, 2]>

// Slicing with statically computed shapes
const row = a.slice(1);          // NDArray<[3]>   — integer index drops the axis
const win = NDArray.zeros([1024]).slice({ start: 100, stop: 1000 }); // NDArray<[900]>

// Reshape / flatten — the product is computed at the type level
const cube = NDArray.zeros([2, 3, 4]);
const flat = cube.flatten();     // NDArray<[24]>
const re = cube.reshape([4, 6]); // NDArray<[4, 6]>
NDArray.zeros([1024, 1024]).flatten(); // NDArray<[1048576]> — a computed literal, not number

// Elementwise family + embedding primitives (the RAG/embedding use case)
const x = NDArray.fromArray([4], [0.2, 0.4, 0.1, 0.8]);
const y = NDArray.fromArray([4], [0.5, 0.1, 0.3, 0.2]);
const scaled = x.mul(y);         // NDArray<[4]>  (also .sub / .div)
const d = x.dot(y);              // number
const cos = x.cosineSimilarity(y); // number   (also .norm())

// Read the data back out
sum.toNestedArray();             // [[11, 22, 33], [14, 25, 36]]
```

Shape mismatches are caught at the offending argument, with the runtime's exact message:

```ts
const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);

a.matmul(NDArray.zeros([5, 4])); // ❌ compile error at the argument: inner dims [2,3] vs [5,4]
a.add(NDArray.zeros([4]));       // ❌ compile error: shapes [2,3] and [4] are not broadcastable
a.slice(9);                      // ❌ compile error: index 9 out of bounds for axis 0 (dim 2)
```

## Gradual typing

Real programs have runtime-determined shapes. Where a dimension is `number` rather than a
literal, NumType stops making static promises and defers to a runtime check — the same
escape-hatch philosophy that made TypeScript itself adoptable:

```ts
async function load(): Promise<NDArray<[number, 1536]>> { /* ... */ }

const emb = await load();        // NDArray<[number, 1536]>
emb.matmul(NDArray.zeros([1536, 8])); // NDArray<[number, 8]> — the literal dim is still tracked
```

Union and dynamic-rank shapes degrade the same way: NumType never emits a *confidently wrong*
error, only a wide (`number`) result or a deferred runtime check.

## Backends

The default `NDArray` computes in pure JS and is browser-safe (it eagerly loads nothing native).
The from-scratch Rust/WASM cores are an explicit, opt-in choice:

```ts
NDArray.backend("wasm");     // single-threaded WASM core (SIMD128 blocked matmul)
NDArray.backend("threaded"); // Node-only, experimental: multi-threaded matmul
```

Every WASM kernel is **bit-identical** to the naive JS reference — proven by a differential test
suite that includes IEEE-754 special values (NaN, ±Inf, ±0, subnormals). Threading is a
Node-only, explicitly-experimental opt-in for v0 (the browser port is gated on COOP/COEP headers
a library can't set). See [`docs/`](docs/) for the backend design and benchmarks.

## Honest qualifications

Credibility is an asset for a research project, so the scope of the guarantee is stated plainly:

1. **"Python can't do this" is verified, not asserted.** NumPy's own maintainers advise against
   relying on shape typing, and PyTorch's static-shapes request has sat open for years — the
   general problem resists Python's type system structurally.
2. **"TypeScript can do this" means: newly tractable, unproven at scale.** No existing TS library
   has delivered general compile-time shape checking with broadcasting and reductions; the prior
   art stops at literal-dimension matmul. NumType is not validating a *proven* technique — it is
   probing the limit. That is the point of the research.
3. **Scope of the guarantee: realistic ranks and op chains, not "every conceivable shape."** The
   plausible failure modes for a type-level shape system in TypeScript are high-rank tensors,
   very large broadcast dimensions, and long chains of composed operations. The gradual design
   (literal dims static, `number` dims at runtime) makes that a feature, not a footnote.

The type-checker cost is measured, not hoped: the slice arithmetic costs ~1.59× instantiations,
the bounds checks ~1.036×, and hover latency measured against the native TS 7 language server is
0.04–0.08 ms median — about three orders of magnitude under a 100 ms editor gate.

## What's implemented

The type layer (broadcast / matmul / reduce as gradual types, errors at the offending argument,
statically computed slice / reshape / flatten shapes, literal bounds checks) and a from-scratch
Rust/WASM numeric core: strided O(1) views (`transpose`, `slice`), a blocked + packed + SIMD128
matmul, zero-copy WASM residency, and a hand-rolled multi-threaded matmul — each proven
bit-identical to the JS reference. The op surface is deliberately narrow (construction/
conversion, `add`/`sub`/`mul`/`div`, `matmul`, `sum` with `keepdims`, `transpose`, `slice`,
`reshape`/`flatten`, `dot`/`norm`/`cosineSimilarity`) — a *minimum viable* NumPy, not a clone.

For the full per-phase specifications, results, and the competitive analysis, see [`docs/`](docs/)
and [`docs/roadmap.md`](docs/roadmap.md). Internal research notes there are partly in German.

## Non-goals

No 400-operation NumPy clone, no GPU/autograd, no DataFrames, no transcendental ops that would
break bit-parity with the JS reference.

## License

[Apache-2.0](LICENSE). The explicit patent grant covers the from-scratch kernel algorithms; see
[`NOTICE`](NOTICE) for attribution.
