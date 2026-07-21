/**
 * From-scratch, deterministic text embeddings: hashed character trigrams,
 * term-frequency weighted. No external library, no API call — the point of
 * this example is dogfooding `numtype` itself, not the embedding technique
 * (a real RAG pipeline would use a trained model here).
 */

/** djb2 string hash, folded to an unsigned 32-bit integer via `>>> 0`. A
 * small, well-known, deterministic hash — no external dependency needed. */
function djb2(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  }
  return h >>> 0;
}

/** Overlapping 3-character windows of the lowercased text, padded with a
 * leading/trailing space so word boundaries participate too (`"cat"` inside
 * a longer word and `" cat "` as a standalone word hash differently, which
 * is exactly the extra separating power a character n-gram model wants). */
function trigrams(text: string): string[] {
  const padded = `  ${text.toLowerCase()}  `;
  const out: string[] = [];
  for (let i = 0; i < padded.length - 2; i++) out.push(padded.slice(i, i + 3));
  return out;
}

/** Term-frequency vector of hashed character trigrams, dimension `dims`.
 * Deterministic — same text always produces the same vector, which is what
 * makes this demo's assertions reproducible run to run (see D3/T1 in
 * docs/dogfooding-rag-spec.md). */
export function embedText(text: string, dims: number): Float64Array {
  const vec = new Float64Array(dims);
  for (const tg of trigrams(text)) {
    const idx = djb2(tg) % dims;
    // noUncheckedIndexedAccess makes typed-array reads `number | undefined`
    // too (an index-signature read like any other) — `?? 0` is the
    // in-bounds no-op default (`idx` is always `< dims` from `% dims`).
    vec[idx] = (vec[idx] ?? 0) + 1;
  }
  return vec;
}

/**
 * Embed several texts and concatenate into one flat row-major buffer, ready
 * for `NDArray.fromArray([texts.length, dims], flat)`.
 *
 * RESOLVED-adjacent (see docs/dogfooding-rag-ergebnisse.md, F5/W4):
 * `NDArray.stack([...])` now exists (0.2.0) and is exactly the `np.stack`
 * numtype was missing — but `stack` only accepts NDArrays already built one
 * by one, and its literal-tuple call form needs every row enumerated as a
 * separate argument at the call site. Here `texts.length` (`N`/`Q` in
 * main.ts) is a fixed literal known up front, so `fromArray` over this
 * helper's single flat buffer stays the better fit — it keeps that literal
 * row count as a type without listing N rows by hand. `main.ts`'s
 * mean-pooling chunk matrix (two independently-built rows, genuinely small
 * and fixed) is `stack`'s actual showcase; see the `NDArray.stack(...)` call
 * there.
 */
export function embedMatrix(texts: readonly string[], dims: number): Float64Array {
  const flat = new Float64Array(texts.length * dims);
  texts.forEach((text, row) => {
    flat.set(embedText(text, dims), row * dims);
  });
  return flat;
}
