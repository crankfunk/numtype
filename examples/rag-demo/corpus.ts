/**
 * Sixteen short English documents, deliberately spanning arrays/TypeScript/
 * numerics vocabulary (and, fittingly, referencing NumType itself) so the
 * hashed-trigram embeddings in embedding.ts have enough surface-level
 * variety to separate cleanly under cosine similarity — see main.ts's
 * QUERIES for the retrieval targets and docs/dogfooding-rag-ergebnisse.md
 * for the real, measured scores.
 */
export const DOCS = [
  "TypeScript adds static types on top of JavaScript, catching mistakes before code ever runs.",
  "NumPy arrays store numbers in contiguous memory, which makes vectorized math fast.",
  "A neural network learns by adjusting weights to minimize a loss function during training.",
  "Matrix multiplication combines two matrices by summing products of rows and columns.",
  "Rust's ownership system prevents data races without a garbage collector at runtime.",
  "WebAssembly compiles code from languages like Rust into a fast binary format for browsers.",
  "Vector databases store embeddings and support approximate nearest neighbor search at scale.",
  "Broadcasting lets arrays of different shapes combine without copying data explicitly.",
  "Cosine similarity measures the angle between two vectors, ignoring their magnitude.",
  "A hash function maps arbitrary input data to a fixed-size number deterministically.",
  "Gradient descent updates parameters by stepping opposite the gradient of a loss.",
  "Retrieval augmented generation fetches relevant documents before an LLM generates an answer.",
  "The TypeScript compiler infers precise literal types from array and tuple expressions.",
  "GPUs execute the same instruction across many threads in parallel for numeric workloads.",
  "Character trigrams split text into overlapping three letter windows for simple text features.",
  "NumType checks NumPy-style array shapes at compile time using the TypeScript type checker.",
] as const;
