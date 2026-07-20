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
 * FRICTION (see docs/dogfooding-rag-ergebnisse.md): NumPy builds a matrix
 * from a list of row vectors directly (`np.array([embed(t) for t in
 * texts])`, or `np.stack`). `NDArray.fromArray` only takes one already-flat
 * buffer for the *whole* matrix, so building a matrix out of N independently
 * computed row vectors needs this small hand-rolled flattening helper —
 * `Float64Array#set` at the right row offset — instead of a one-line stack.
 */
export function embedMatrix(texts: readonly string[], dims: number): Float64Array {
  const flat = new Float64Array(texts.length * dims);
  texts.forEach((text, row) => {
    flat.set(embedText(text, dims), row * dims);
  });
  return flat;
}
