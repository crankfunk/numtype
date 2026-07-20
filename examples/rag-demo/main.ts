/**
 * NumType dogfooding demo: a small, deterministic RAG *retrieval* core (no
 * LLM/generation part — see docs/dogfooding-rag-spec.md, Nicht-Ziele). Runs
 * with `pnpm demo` (or `node main.ts` directly — Node 24 executes .ts
 * natively via its built-in type stripping, same convention as the repo's
 * own spike/demo.ts).
 *
 * Pipeline: embed a 16-document corpus and 8 queries with from-scratch
 * hashed-trigram TF vectors (embedding.ts), build matrices with
 * `NDArray.fromArray`, L2-normalize rows, score every query against every
 * document via one `matmul`, rank in JS (no argmax/topk op exists yet —
 * logged as friction below), and cross-check the top hit with `dot`/
 * `cosineSimilarity` on the RAW (non-normalized) rows.
 *
 * Every place the natural numtype-first formulation was tried and something
 * was missing is logged inline as FRICTION #n, with the workaround actually
 * used right below it — see docs/dogfooding-rag-ergebnisse.md for the full
 * write-up (this file is the primary evidence source for that doc).
 */
import assert from "node:assert";
import { NDArray } from "numtype";
import { DOCS } from "./corpus.ts";
import { embedMatrix, embedText } from "./embedding.ts";

// Literal `const` bindings, not derived — this is what keeps every NDArray
// shape below a LITERAL tuple (`NDArray<[16, 256]>`, not `NDArray<[number,
// number]>`), so hovers show real numbers end to end (spec D3).
const N = 16; // documents in the corpus
const D = 256; // embedding dimensionality
const Q = 8; // queries below

assert.strictEqual(DOCS.length, N, `corpus.ts DOCS must have exactly ${N} entries (found ${DOCS.length})`);

console.log("=== NumType dogfooding demo: from-scratch RAG retrieval ===\n");

// --- Build the corpus matrix ------------------------------------------------
const corpusFlat = embedMatrix(DOCS, D);
const corpusMatrix = NDArray.fromArray([N, D], corpusFlat); // NDArray<[16, 256]>
console.log(`corpus matrix: shape=[${corpusMatrix.shape.join(",")}]`);

// --- L2-normalize each row (natural formulation attempted first) -----------
// Intent (NumPy idiom): `X / np.linalg.norm(X, axis=1, keepdims=True)`.
// mul(self) -> sum(axis=1) -> sqrt -> reshape -> broadcast div. The first
// three steps ARE natural numtype: elementwise mul, then an axis-sum whose
// resulting shape ([16]) numtype computes for us.
const corpusSquared = corpusMatrix.mul(corpusMatrix); // NDArray<[16, 256]>
const corpusSumSquares = corpusSquared.sum(1); // NDArray<[16]>

// FRICTION F1 — Intent: `np.sqrt(sumSquares)`, an elementwise unary op.
// Workaround: the next 3 lines (and the mirrored block at :103-107 for
// queries) — no `.sqrt()`/unary-map op exists on `NDArray`, so the natural
// chain (mul -> sum -> sqrt -> div) has to drop to the raw `Float64Array`,
// apply `Math.sqrt` by hand, and rebuild
// via `fromArray`. Cost: 4 extra lines, no readability loss to speak of, and
// crucially NO type-level degradation — `N` is still a literal here, so the
// rebuilt array is still `NDArray<[16]>`, not a widened `NDArray<[number]>`.
// Candidate: an elementwise unary-map surface (sqrt at minimum; NumPy has a
// whole ufunc family here).
const corpusRowNormsData = new Float64Array(corpusSumSquares.data.length);
for (let i = 0; i < corpusRowNormsData.length; i++) corpusRowNormsData[i] = Math.sqrt(corpusSumSquares.data[i] ?? 0);
const corpusRowNorms = NDArray.fromArray([N], corpusRowNormsData); // NDArray<[16]>

// reshape([16] -> [16, 1]) so the norms broadcast against [16, 256] on div.
const corpusRowNormsCol = corpusRowNorms.reshape([N, 1]); // NDArray<[16, 1]>
const corpusNormalized = corpusMatrix.div(corpusRowNormsCol); // NDArray<[16, 256]> — back to natural numtype
console.log(`corpus normalized: shape=[${corpusNormalized.shape.join(",")}]\n`);

// --- Queries -----------------------------------------------------------
// [query text, expected top-1 document index] — expected indices and the
// measured margins below were verified empirically before being pinned
// (spec D3 v2: "if a query is too tight, replace it and note it" — none of
// these eight needed replacing after the initial corpus/query draft except
// query 0, which was reworded once to widen its margin; see the results doc).
const QUERIES: readonly (readonly [string, number])[] = [
  ["TypeScript's static types catch mistakes before your code runs.", 0],
  ["Why are NumPy arrays fast for math operations?", 1],
  ["How does a neural network learn from training data?", 2],
  ["What is matrix multiplication in linear algebra?", 3],
  ["How does Rust prevent data races without garbage collection?", 4],
  ["What does cosine similarity measure between vectors?", 8],
  ["What is retrieval augmented generation for LLMs?", 11],
  ["How does NumType check array shapes at compile time?", 15],
];
assert.strictEqual(QUERIES.length, Q, `QUERIES must have exactly ${Q} entries (found ${QUERIES.length})`);

// Margin threshold (spec D3 v2, Baustein-0 finding 4): top-1 score minus
// top-2 score must clear this to count as a stable, non-near-tie retrieval.
const MARGIN_THRESHOLD = 0.03;

const queryFlat = embedMatrix(
  QUERIES.map(([text]) => text),
  D,
);
const queryMatrix = NDArray.fromArray([Q, D], queryFlat); // NDArray<[8, 256]>

// Same normalization as the corpus, duplicated rather than factored into a
// shared helper: a shape-generic `l2NormalizeRows<Rows, Dims>` helper would
// need `NDArray<[Rows, Dims]>` with BOTH dims as free type parameters, and
// `.reshape([sumSquares.shape[0], 1])` inside it loses the literal `Rows` the
// call sites actually have (the parameter position only knows `Rows extends
// number`, not a specific literal) — so factoring it out would ironically
// COST type precision rather than save lines. Left inline, on purpose.
const querySquared = queryMatrix.mul(queryMatrix); // NDArray<[8, 256]>
const querySumSquares = querySquared.sum(1); // NDArray<[8]>
const queryRowNormsData = new Float64Array(querySumSquares.data.length);
for (let i = 0; i < queryRowNormsData.length; i++) queryRowNormsData[i] = Math.sqrt(querySumSquares.data[i] ?? 0);
const queryRowNorms = NDArray.fromArray([Q], queryRowNormsData); // NDArray<[8]>
const queryRowNormsCol = queryRowNorms.reshape([Q, 1]); // NDArray<[8, 1]>
const queryNormalized = queryMatrix.div(queryRowNormsCol); // NDArray<[8, 256]>

// --- One matmul scores every query against every document ------------------
// Both operands are already unit-row-normalized, so the dot products below
// ARE the cosine similarities: [8, 256] @ [256, 16] -> [8, 16].
const similarities = queryNormalized.matmul(corpusNormalized.transpose()); // NDArray<[8, 16]>
console.log(`similarity matrix: shape=[${similarities.shape.join(",")}] (queries x documents)\n`);

// --- Rank + assert each query's top match -----------------------------
for (let qi = 0; qi < QUERIES.length; qi++) {
  const query = QUERIES[qi];
  if (query === undefined) continue; // noUncheckedIndexedAccess — statically unreachable here, kept honest
  const [queryText, expectedTop1] = query;

  // FRICTION F3/F4 — Intent: read row `qi` of the [8, 16] score matrix as a
  // plain array to rank it. Workaround: `similarities.slice(qi)` — an
  // integer spec on the leading axis drops that axis (documented behavior,
  // spike/src/slice.ts), so this IS the natural numtype call and yields a
  // real `NDArray<[16]>`. The friction is one step later: there is no
  // argmax/topk op, so ranking the 16 scores against each other still has
  // to leave NDArray and loop over `.data` by hand (see below). Candidate:
  // `argmax`/`argsort`/`topk` — this demo's single most obvious missing op.
  const rowScores = similarities.slice(qi); // NDArray<[16]>
  const ranked = Array.from(rowScores.data)
    .map((score, docIdx) => ({ score, docIdx }))
    .sort((a, b) => b.score - a.score);
  const top1 = ranked[0];
  const top2 = ranked[1];
  assert.ok(top1 !== undefined && top2 !== undefined, `query ${qi}: expected at least 2 ranked documents`);
  const margin = top1.score - top2.score;

  console.log(
    `Q${qi}: "${queryText}"\n` +
      `  top1: doc${top1.docIdx} (score=${top1.score.toFixed(4)}) — "${DOCS[top1.docIdx]}"\n` +
      `  top2: doc${top2.docIdx} (score=${top2.score.toFixed(4)}) — margin=${margin.toFixed(4)}`,
  );

  assert.strictEqual(top1.docIdx, expectedTop1, `query ${qi} ("${queryText}"): expected top-1 doc ${expectedTop1}, got ${top1.docIdx}`);
  assert.ok(margin >= MARGIN_THRESHOLD, `query ${qi} ("${queryText}"): margin ${margin.toFixed(4)} below threshold ${MARGIN_THRESHOLD}`);

  // Cross-check (spec D3): recompute the top-1 score independently via
  // `cosineSimilarity` on the RAW (non-pre-normalized) rows — cosine
  // similarity is scale-invariant, so this should agree with the
  // matmul-on-normalized-rows score above, up to floating-point rounding.
  const rawQueryRow = queryMatrix.slice(qi); // NDArray<[256]>
  const rawDocRow = corpusMatrix.slice(top1.docIdx); // NDArray<[256]>
  const crossCheckScore = rawQueryRow.cosineSimilarity(rawDocRow);
  const crossCheckDot = rawQueryRow.dot(rawDocRow);
  assert.ok(
    Math.abs(crossCheckScore - top1.score) < 1e-9,
    `query ${qi}: cosineSimilarity cross-check ${crossCheckScore} diverges from matmul score ${top1.score}`,
  );
  console.log(`  cross-check: cosineSimilarity=${crossCheckScore.toFixed(6)} raw dot=${crossCheckDot.toFixed(4)}\n`);
}

console.log(`All ${QUERIES.length} queries retrieved their expected top-1 document (margin >= ${MARGIN_THRESHOLD}).\n`);

// --- Mean-pooling a multi-chunk document ------------------------------
// A document that arrived pre-chunked (the common RAG ingestion shape):
// embed each chunk, then mean-pool the chunk vectors into one document
// vector, and check it still retrieves as document 15 (the doc these two
// chunks are split from).
console.log("=== Mean-pooling a multi-chunk document ===\n");
const chunk1 = "NumType checks NumPy-style array shapes at compile time";
const chunk2 = "using the TypeScript type checker.";
const chunkFlat = embedMatrix([chunk1, chunk2], D);
const chunkMatrix = NDArray.fromArray([2, D], chunkFlat); // NDArray<[2, 256]>
const chunkSum = chunkMatrix.sum(0); // NDArray<[256]> — sum over the chunk axis

// FRICTION F2 — Intent: `chunkSum / 2` (divide a vector by the chunk
// count — a NumPy scalar-broadcast). Workaround: `NDArray` has no
// number-scalar overload for `div`/`mul`/`add`/`sub` — every op takes
// another `NDArray`. Wrap the scalar in a shape-[1] array and rely on
// ordinary broadcasting ([256] / [1] -> [256], NumPy's own rule) instead.
// Cost: one extra `fromArray` call per scalar op; no type-level loss (the
// [256] shape survives). Candidate: `number` overloads for the four
// elementwise binary ops (or a dedicated `.scale(n)` helper) — this is the
// SECOND most obvious missing-op signal in this demo, right after argmax.
const chunkCountArr = NDArray.fromArray([1], [2]);
const pooledVector = chunkSum.div(chunkCountArr); // NDArray<[256]>

let bestPooledIdx = -1;
let bestPooledScore = -Infinity;
let secondPooledScore = -Infinity;
for (let docIdx = 0; docIdx < N; docIdx++) {
  const score = corpusMatrix.slice(docIdx).cosineSimilarity(pooledVector);
  if (score > bestPooledScore) {
    secondPooledScore = bestPooledScore;
    bestPooledScore = score;
    bestPooledIdx = docIdx;
  } else if (score > secondPooledScore) {
    secondPooledScore = score;
  }
}
const pooledMargin = bestPooledScore - secondPooledScore;
console.log(
  `pooled(chunk1, chunk2) top match: doc${bestPooledIdx} (score=${bestPooledScore.toFixed(4)}, ` +
    `margin=${pooledMargin.toFixed(4)}) — "${DOCS[bestPooledIdx]}"\n`,
);
assert.strictEqual(bestPooledIdx, 15, `pooled chunk vector should retrieve document 15, got ${bestPooledIdx}`);
assert.ok(bestPooledScore > 0.9, `pooled chunk vector's similarity to its source document should be very high, got ${bestPooledScore}`);

// Sanity: embedding the ORIGINAL (unsplit) document text directly should be
// close to identical to the pooled vector's own best match score, since
// both are measuring "how well does this vector represent document 15".
const wholeDocVector = NDArray.fromArray([D], embedText(DOCS[15] ?? "", D));
const wholeVsPooled = wholeDocVector.cosineSimilarity(pooledVector);
console.log(`cosineSimilarity(whole-doc embedding, pooled chunks) = ${wholeVsPooled.toFixed(4)}\n`);
assert.ok(wholeVsPooled > 0.8, `whole-document embedding should be close to the pooled chunk vector, got ${wholeVsPooled}`);

// --- Shape-error self-checks (compile-time, not runtime) --------------
// These two lines are the demo's OWN self-check on numtype's core USP: if
// numtype ever stopped rejecting these at the argument, `pnpm check` inside
// this example would start failing right here (an unused `@ts-expect-error`
// is itself a compile error), so these double as a tiny regression pin.
// Wrapped in a never-called function: `tsc --noEmit` still type-checks a
// function body regardless of whether it's invoked, but Node's runtime
// (which does NOT type-check — it only strips types) would otherwise
// actually execute these two deliberately-invalid calls and crash on their
// runtime guards (`assertVectorPair`/`matmulRuntime`) before ever reaching
// this point. Keeping them uncalled is what makes them a pure compile-time
// pin, matching this section's own name.
function _shapeErrorSelfChecks(): void {
  // @ts-expect-error matmul: inner dimensions must match — [2, 3] has 3 columns, [5, 4] has 5 rows
  NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]).matmul(NDArray.fromArray([5, 4], new Array(20).fill(0)));

  // @ts-expect-error dot: both operands must be rank-1 — got rank 2 and rank 1
  NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]).dot(NDArray.fromArray([3], [1, 2, 3]));
}
void _shapeErrorSelfChecks; // referenced (not called) so it isn't flagged as dead code by any future lint pass

console.log("=== demo complete: all assertions passed ===");
