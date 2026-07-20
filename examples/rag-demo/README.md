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
- Building `[16, 256]`/`[8, 256]` matrices with `NDArray.fromArray`, L2-normalizing rows, scoring
  every query against every document with a *single* `matmul`, and cross-checking the top hit
  with `dot`/`cosineSimilarity`.
- `reshape`, `transpose`, `slice`, and broadcasting `mul`/`div` all doing real work in the
  pipeline, not just toy calls.
- Mean-pooling a multi-chunk document into one vector and confirming it still retrieves correctly.
- Two `@ts-expect-error`-pinned shape mistakes (a `matmul` dimension mismatch and a `dot` rank
  mismatch) — self-checks that fail `pnpm run check` if numtype ever stopped catching them.
- Every place the natural numtype-first formulation was missing something is called out inline as
  `FRICTION Fn` in `main.ts`, with the actual workaround right below it (the F-numbers match the
  friction log in `docs/dogfooding-rag-ergebnisse.md`).

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
- The two `@ts-expect-error` lines near the bottom — delete either `@ts-expect-error` comment and
  `pnpm run check` immediately reports the shape mismatch at that exact argument.
