# NumType dogfooding demo — RAG retrieval

A small, deterministic RAG (retrieval-augmented generation) *retrieval* core built on the
published [`numtype`](https://www.npmjs.com/package/numtype) package, installed from the npm
registry like any real consumer would install it — this folder is not part of the numtype build,
it is an ordinary external project that depends on it.

There is no LLM/generation step here (no API calls, no model weights) — just embedding, matrix
math, and ranking, which is the part `numtype` actually helps with.

## What it shows

- A 16-document corpus and 8 queries, embedded with a from-scratch, deterministic hashed
  character-trigram TF vectorizer (`embedding.ts` — no external embedding model or library).
- Building `[16, 256]`/`[8, 256]` matrices with `NDArray.fromArray`, L2-normalizing rows with
  `.sqrt()`, scoring every query against every document with a *single* `matmul`, ranking with
  `.topk()`, and cross-checking the top hit with `.item()`/`dot`/`cosineSimilarity`.
- `reshape`, `transpose`, `slice`, and broadcasting `mul`/`div` all doing real work in the
  pipeline, not just toy calls.
- Mean-pooling a multi-chunk document into one vector (built with `NDArray.stack`, reduced with
  `.mean(0)`) and confirming it still retrieves correctly.
- Three `@ts-expect-error`-pinned shape mistakes (a `matmul` dimension mismatch, a `dot` rank
  mismatch, and an out-of-bounds `.item()` index) — self-checks that fail `pnpm run check` if
  numtype ever stopped catching them.

### From 0.1.1 to 0.2.0: the friction that became the op list

This demo was first built against numtype 0.1.1. Every place the natural numtype-first
formulation was missing something back then was logged inline as `FRICTION Fn` in `main.ts`, with
the workaround actually used right below it — see the full write-up in
[`docs/dogfooding-rag-ergebnisse.md`](../../docs/dogfooding-rag-ergebnisse.md). Those five
friction points (F1–F5) became numtype 0.2.0's op wishlist (W1–W5), in the priority order the
friction log derived:

| Friction | Missing op | 0.1.1 workaround | 0.2.0 replacement |
| --- | --- | --- | --- |
| F4 | `argmax`/`topk` | hand-rolled `Array.from(...).sort(...)` ranking | `.topk(k)` |
| F2 | scalar overloads | wrap the scalar in a shape-`[1]` `fromArray` and broadcast | `add`/`sub`/`mul`/`div(n: number)`, `.mean(axis)` |
| F1 | elementwise `sqrt` | raw `Float64Array` loop with `Math.sqrt`, rebuilt via `fromArray` | `.sqrt()` |
| F5 | `stack` | hand-rolled `Float64Array#set`-at-row-offset flatten helper | `NDArray.stack([...])` |
| F3 | `.item(...)` | nested `.slice().slice()` or raw `.data` index arithmetic | `.item(i, j, ...)` |

`main.ts` still carries every original `FRICTION Fn` comment, now rewritten in place as
`RESOLVED (0.2.0)` — each one shows the old workaround and the real op replacing it, right where
the workaround used to live.

## Run it

```sh
pnpm install
pnpm run check   # tsc --noEmit — type-checks against the installed numtype .d.ts files
pnpm run demo    # runs main.ts, prints scores, and asserts every retrieval result
```

Requires Node 22.18 or newer (the repo itself runs Node 24, see the root `.nvmrc`) — `pnpm run
demo` executes `main.ts` directly via Node's built-in TypeScript type stripping, no build step.

## What to hover in your editor

Open `main.ts` and hover:

- `corpusMatrix` — `NDArray<[16, 256]>`, a real literal tuple, not `NDArray<[number, number]>`.
- `similarities` — `NDArray<[8, 16]>`, computed by the type checker from the `matmul` of two
  already-known shapes.
- `similarities.slice(qi)` inside the ranking loop — `NDArray<[16]>`: the rank drop from an
  integer slice spec is a *static* effect, independent of `qi`'s runtime value.
- `rowScores.topk(2)` — `{ values: NDArray<[2]>, indices: NDArray<[2]> }`: `k` is a literal, so
  the result shape is too, not `NDArray<[number]>`.
- `chunkMatrix` — `NDArray<[2, 256]>`, built by `NDArray.stack([...])` from two independently
  constructed rank-1 rows, still a literal shape.
- `similarities.item(qi, top1.docIdx)` — `number`, a direct scalar read with no intermediate
  `NDArray` allocation.
- The three `@ts-expect-error` lines near the bottom — delete any one `@ts-expect-error` comment
  and `pnpm run check` immediately reports the shape mismatch at that exact argument (the third,
  `queryMatrix.item(99, 0)`, is a literal-index bounds check: `99` against axis 0's dim `8`).
