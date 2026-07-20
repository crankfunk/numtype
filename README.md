```
███╗   ██╗██╗   ██╗███╗   ███╗████████╗██╗   ██╗██████╗ ███████╗
████╗  ██║██║   ██║████╗ ████║╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝
██╔██╗ ██║██║   ██║██╔████╔██║   ██║    ╚████╔╝ ██████╔╝█████╗
██║╚██╗██║██║   ██║██║╚██╔╝██║   ██║     ╚██╔╝  ██╔═══╝ ██╔══╝
██║ ╚████║╚██████╔╝██║ ╚═╝ ██║   ██║      ██║   ██║     ███████╗
╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝   ╚═╝      ╚═╝   ╚═╝     ╚══════╝
```

> NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors.

**Status: v0.1 research preview** · Apache-2.0 · zero runtime dependencies ·
launch post: [*Teaching the type checker arithmetic*](https://marvinmuegge.com/notes/teaching-the-checker-arithmetic/)

![Editor demo: types computed while typing — matmul result NDArray<[2, 4]>, slice arithmetic NDArray<[900]>, and a shape mismatch surfacing as a compile error at the argument](https://raw.githubusercontent.com/crankfunk/numtype/main/docs/assets/numtype-demo.gif)

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

**[Try it in your browser →](https://www.typescriptlang.org/play/?#code/JYWwDg9gTgLgBAbzgOQCIEEpQIYE84C+cAZlBCHAEQB2AriDLmAKaUDcAUBwPTdwDKAC2wsAznGxRmcAMbkwtGMwAmcAEb4Yg6YxaztMgNbMocQCgEcQRABuJuFunXJwbGoA2zUQDoOc6qPhsOABeFAwsPC9SckwcXAAKAG0AJgAaOABmAF10xIBGdLTM9IAWdIBWdIA2LIBKTj8AuDAyZVoZeFDsLxBsGBBaN3i0WMiALxMIUSSM0rr6uF5LGxMALjDR3AAeFLmAPi4l-jdgGWlnLRBmGFP05QhqaQ17bX1mIzW4PIAGX7gAWi+f1CAE5fr4Hk0AO7AaghDYRXBeCZkab5b7JEp1LyiE5neJIAKSGDrH7fdIBCBgUm-b6EeqHPjoOCiYR6EDAUS9GAyQRwTkSWTyYAeOAmMimdAAFRe5ygAHN6MxqPALNQIIE4FBaCrQNIZDhWasON1uQMhiNEcjJmjKnAsbUGTw+KhcNRsByZHBlKBxMpmPKcP64IHsGdiIM3Pg1RASNg3KJpOLoKJ0sxbFBjf6ZG5JPrIfBqOs6CA1CYGgXvW74Za4tbUUlqOk8uUMjVaj0+ubhuE6yipkkW230gAOeacIA)**
— the types load straight from npm into the TypeScript Playground; hover the variables, break
the shapes. Nothing to install.

The red squiggle is the product; the WASM performance is a credibility feature, not the reason
to reach for it. If you've ever shipped a tensor-shape bug to production, this is the pitch.

## Why NumType exists

The original motivation is a gap in the TypeScript ecosystem: there is no real counterpart to
NumPy. Python's entire scientific stack rests on one shared n-dimensional array foundation; the
JS/TS world has adjacent pieces — TensorFlow.js is an ML framework, math.js targets general
(mostly low-rank) math, scijs/ndarray predates modern TypeScript — but no general, typed n-d
array layer that feels like standard infrastructure. And the one thing a TS-native answer could
uniquely add is exactly what none of the existing options attempt: putting the type system
itself to work on shapes.

NumType tries to close that gap in the only form that is honest for a new library: a **minimum
viable NumPy** — a deliberately narrow op surface with the shape-typed core done properly — not
a 400-operation clone. The obvious limitations that come with that claim are stated, not
hidden: see [Honest qualifications](#honest-qualifications) and [Non-goals](#non-goals).

### How is this different from numpy-ts?

[numpy-ts](https://numpyts.dev/) ports NumPy's *API* to TypeScript — impressively broadly (its
headline is 94% API coverage), with outputs validated against NumPy itself. NumType teaches
TypeScript's *type checker* to do NumPy's shape arithmetic. numpy-ts's "catch errors at compile
time" is signature- and dtype-level typing: a `matmul` of `[2, 3] × [5, 4]` type-checks there
and fails at runtime, while here it is a compile error at the offending argument. The two
projects barely compete — if you need 476 functions today, use numpy-ts; if you want shape
mismatches as red squiggles while you type, on a deliberately narrow op surface, that is what
NumType exists for. (Survey of the wider landscape, numpy-ts claims re-verified 2026-07-20:
[docs/wettbewerbsanalyse-und-usp.md](docs/wettbewerbsanalyse-und-usp.md) — Appendix A is in
English.)

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

The digit-string representation itself is established prior art in the type-level community —
general-purpose arithmetic libraries (notably [ts-arithmetic](https://github.com/arielhs/ts-arithmetic))
compute over digit representations to escape the same recursion ceiling. NumType's contribution
is the application, not the trick: wiring that arithmetic into an n-d array API — dimensions,
slice lengths, bounds, shape products — where a computed number becomes a compile-time shape
error.

## Install

```sh
pnpm add numtype     # or: npm install numtype / yarn add numtype
```

Zero runtime dependencies; the `.wasm` core is bundled, so there is no native toolchain to
install. The default `NDArray` runs in pure JS and works in any modern JS environment (Node or
browser). The optional WASM/threaded backends need Node (see [Backends](#backends)). ESM only.

**The shape checking works in your editor out of the box.** Everything type-level ships as
plain `.d.ts` files inside the package, so any editor running the TypeScript language service —
VS Code, WebStorm, Neovim, Zed, … — shows the hovers and shape errors immediately after
installing: no editor extension, no compiler plugin, no codegen step. Two requirements: your
project uses TypeScript (a plain-JS project still gets hovers and IntelliSense from the
`.d.ts`, but errors only with `checkJs`), and a reasonably current compiler — NumType is
developed and verified against TypeScript 7.x (the current npm `latest`); the type machinery
uses features introduced in TS 5.0, but older majors are untested.

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

A larger, real consumer application — a from-scratch RAG (retrieval) demo that embeds documents,
scores queries against them with a single `matmul`, and ranks the results — lives in
[`examples/rag-demo`](examples/rag-demo) and installs `numtype` from npm like any other project.

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

The bundled `.wasm` is the **single-threaded** core, so `backend("wasm")` works out of the box.
The multi-threaded core is **not shipped in the npm package** — it requires a pinned nightly Rust
toolchain with `-Z build-std` to compile, which the published (stable-Rust) package deliberately
does not depend on. `backend("threaded")` is therefore build-it-yourself from a checkout: clone
the repo and run `pnpm build:wasm:threads`.

## Zero dependencies, from scratch

numtype has **no runtime dependencies** — the `package.json` has no `dependencies` field at all,
enforced by a CI guard. There is nothing transitive to audit and no supply-chain surface. Both
halves are hand-written: the Rust/WASM kernels (no `wasm-bindgen`, no BLAS, no `ndarray` crate —
a hand-rolled `extern "C"` ABI) and the entire type-level machinery (the digit-string arithmetic).
The single-threaded `.wasm` is pre-built and bundled, so `pnpm add numtype` compiles nothing
native on your machine.

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

## Versioning: what to expect before 1.0

NumType follows SemVer with 0.x semantics, sharpened for a library whose *inferred types are
part of the API*:

- **Breaking changes bump the minor** (0.1 → 0.2) and are listed in the release notes. Anything
  that can change an inferred type or reject previously-accepted code counts as breaking and
  never lands in a patch.
- **Stable through all of 0.x:** zero runtime dependencies, WASM kernels bit-identical to the JS
  reference, and the "never wrong, only incomplete" type philosophy — a confidently wrong
  compile-time claim is always a bug; please report it. These aren't aspirations: CI enforces
  them with a zero-dep guard, a frozen artifact hash, and the differential test suite.
- **Types get more precise over time:** shapes that degrade to `number` today may become
  computed literals in a later minor. If you pin exact inferred types in your own tests, expect
  those pins to move on minor bumps — never on patches.
- **The op surface grows additively.** Existing operations only change with a minor bump.
  `backend("threaded")` is experimental and sits outside any stability promise.
- **Research-preview honesty:** the stated goal is probing whether this approach holds up at
  scale. If it hits a structural wall, 0.x may see substantial redesign.

## What's implemented

The type layer (broadcast / matmul / reduce as gradual types, errors at the offending argument,
statically computed slice / reshape / flatten shapes, literal bounds checks) and a from-scratch
Rust/WASM numeric core: strided O(1) views (`transpose`, `slice`), a blocked + packed + SIMD128
matmul, zero-copy WASM residency, and a hand-rolled multi-threaded matmul — each proven
bit-identical to the JS reference. The op surface is deliberately narrow (construction/
conversion, `add`/`sub`/`mul`/`div`, `matmul`, `sum` with `keepdims`, `transpose`, `slice`,
`reshape`/`flatten`, `dot`/`norm`/`cosineSimilarity`) — a *minimum viable* NumPy, not a clone.

**`argmax`/`topk`** (ranking primitives — index of the maximum element, and the top-`k` values +
indices of a 1-D vector) are also available on `NDArray`, but — unlike every op listed above —
**TypeScript-runtime surface only, no WASM kernel yet**: a deliberate, disclosed surface
asymmetry, not an oversight.

**Scalar overloads for `add`/`sub`/`mul`/`div`, a new `mean` reduction, and a new `sqrt` op** are
also available: `x.div(2)` reads as "divide by 2" (shape-preserving, no `[1]`-wrap needed, even at
rank 0), `mean` composes `sum` with one division per output element, and `sqrt` is elementwise
`Math.sqrt` (shape-preserving at every rank — IEEE 754 requires correct rounding for square root,
same as `+`/`-`/`*`/`/`, so this stays exact, unlike a true transcendental op). Like `argmax`/`topk`
above, the new surface is **TypeScript-runtime only, no WASM kernel yet** — the existing
`NDArray`-argument forms of `add`/`sub`/`mul`/`div` keep their full WASM-backed bit-for-bit
guarantee unchanged; only the new scalar overload, `mean`, and `sqrt` are the disclosed asymmetry.

For the full per-phase specifications, results, and the competitive analysis, start at the
[research-notes reading guide](docs/README.md) (curated entry points by interest) or the
[roadmap](docs/roadmap.md). Internal research notes are partly in German — the guide says
which, and why.

## Non-goals

No 400-operation NumPy clone, no GPU/autograd, no DataFrames, no transcendental ops that would
break bit-parity with the JS reference.

## License

[Apache-2.0](LICENSE). The explicit patent grant covers the from-scratch kernel algorithms; see
[`NOTICE`](NOTICE) for attribution.
