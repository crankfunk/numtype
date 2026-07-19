# NumType — Wettbewerbsanalyse & USP

> Naming (2026-07-09): Markenname **NumType** — macht die NumPy-Verwandtschaft und das TypeScript-Echo sichtbar. (Ältere Erwähnungen von „numtype" in den Anhängen meinen dasselbe Projekt; npm-Paketname bleibt lowercase `numtype`.)

*Stand: 2026-07-09. Methode: zwei gezielte Recherche-Agenten (Wettbewerbslandschaft; Shape-Typing-Stand-der-Technik & Nachfrage), Zahlen live gegen npm-Registry- und GitHub-APIs erhoben. Vollständige Berichte mit allen Quell-URLs in den Anhängen A und B.*

## 1. Kernbefund

**Es existiert heute keine gepflegte, allgemeine, NumPy-artige TypeScript-Bibliothek mit Compile-Time-Shape-Checking.** Jede aktiv gepflegte Numerik-Bibliothek typisiert Shapes als `number[]` zur Laufzeit; wo überhaupt Shape-nahe Typen existieren (TensorFlow.js `Tensor1D..4D`, ONNX Runtime `dims`), kodieren sie nur den *Rang*, nie konkrete *Dimensionen*. Die Lücke ist unbesetzt — und das Risiko ist kein Markt-, sondern ein Engineering-Risiko: Skaliert die Typ-Maschinerie auf eine NumPy-große API-Fläche, ohne IDE-Performance und DX zu ruinieren?

## 2. Wettbewerbslandschaft (Kondensat)

| Wer | Status | Relevanz für numtype |
|---|---|---|
| **@stdlib/ndarray** | sehr aktiv, ~32,6K DL/Woche | Der „lebende NumPy-Ersatz" für JS — konkurriert über Breite/Robustheit. **Kein** Shape-Typing. Gegen stdlib gewinnt man nicht über API-Abdeckung, nur über die Typ-Ebene. |
| **numpy-ts** | aktiv (2026), 1 Maintainer, 94 % NumPy-API-Abdeckung, TS+WASM | Nächster Nachbar. Volle *Runtime*-Typedefs, kein Compile-Time-Shape-Checking. Beobachten. |
| **TensorFlow.js** | Releases ~21 Monate alt, 545K DL/Woche | Nur Rang-Typen (`Tensor2D`), Dimensionen bleiben Laufzeit. ML-scoped, kein NumPy-Ersatz. |
| **ONNX Runtime Web** | sehr aktiv, 2,84M DL/Woche | Reine Inferenz — aber Beleg, wie groß die Browser-ML-Schicht ist (→ Use Case Pre-/Postprocessing). |
| **scijs/ndarray** | tot seit 2020, dennoch 1,47M DL/Woche (transitiv) | Zeigt: Die Basis-Abstraktion wird massenhaft gebraucht, wird aber nicht mehr gepflegt. |
| **math.js** | aktiv, 2,83M DL/Woche | Symbolisch+numerisch, breit & flach, pure JS. Kein Wettbewerber auf der Typ-Ebene. |
| **numjs, danfo.js, GPU.js, Shumai, @hoff97/tensor-js** | tot oder stagnierend | Friedhof der „NumPy für JS"-Versuche ohne Differenzierung. |
| **Rust→WASM-Kerne (candle, burn, polars)** | aktiv | Kein einziges npm-Paket bietet einen allgemeinen ndarray-Kern mit NumPy-artiger TS-Oberfläche. Feld offen. |
| **TypeGPU** | aktiv, 32K DL/Woche | Kein Konkurrent, aber der Beweis, dass „anspruchsvolle TS-Typ-Inferenz für eine Numerik-Domäne" ein tragfähiges, adoptiertes Muster ist. Technik-Referenz. |

**Prior Art fürs Shape-Typing in TS:** ausschließlich Blog-Post-Maßstab — potatogpt (GPT-2-Forward-Pass mit typgeprüftem matmul, ~45 Sterne) und sebinsua.com. Beide explizit ohne Broadcasting, nur Literal-Dimensionen, nie als Bibliothek verpackt. numtype wäre der erste ernsthafte Versuch.

## 3. Warum Python das nicht kann (belegt, nicht behauptet)

- NumPys eigene Maintainer: *„Shape typing should not be used currently"* ([numpy#28076](https://github.com/numpy/numpy/issues/28076)); der Grundsatz-Request [numpy#16544](https://github.com/numpy/numpy/issues/16544) ist Wunschliste, keine Roadmap.
- Pyright hat statische Shape-Verfolgung über Operationen hinweg **als „not planned" abgelehnt** ([pyright#8921](https://github.com/microsoft/pyright/issues/8921)).
- PyTorchs Request „Statically checked tensor shapes" ([pytorch#26889](https://github.com/pytorch/pytorch/issues/26889)) ist seit **September 2019 offen**.
- PEP 646 (variadische Generics) existiert, aber: keine Typ-Arithmetik auf Dimensionen, nur ein entpacktes TypeVarTuple pro Signatur → allgemeines matmul/Broadcasting nicht ausdrückbar.
- Die Werkzeuge, die Python-Entwickler real benutzen (**jaxtyping**, aktiv, verbreitet), prüfen per Design **zur Laufzeit** (beartype/typeguard). torchtyping: deprecated. Googles tensor_annotations: archiviert 02/2026.

**Fazit:** Ein Python-Entwickler bekommt 2026 keinen echten statischen Shape-Fehler im Editor — nicht aus Desinteresse (6 Jahre offener PyTorch-Issue, Metas Pyrefly prototypt 2026 daran), sondern weil das Typsystem es nicht hergibt.

## 4. Warum TypeScript es kann — mit ehrlichen Grenzen

**Mechanik:** Variadische Tupel-Typen (TS 4.0) + rekursive Conditional Types mit Tail-Recursion-Elimination (TS 4.5) erlauben echte Shape-Arithmetik: Dekomposition, Broadcasting-Regeln, Achsen-Reduktion — auf Typ-Ebene.

**Harte Grenzen (einplanen, nicht wegreden):**
- Rekursionstiefe ~100 (nicht-tail-rekursiv) bzw. ~1000 (tail-rekursiv, [TS PR #45711](https://github.com/microsoft/TypeScript/pull/45711)); das Limit ist formulierungsabhängig ([#49459](https://github.com/microsoft/TypeScript/issues/49459)).
- Globales Instanziierungs-Budget ~5M pro Kompilation (TS 5, [#53514](https://github.com/microsoft/TypeScript/issues/53514)) — versionsfragil.
- Tupel-Längen-Arithmetik taugt für *Ränge* (kleine Zahlen), nicht für große *Dimensionsgrößen*.
- Dokumentierte IDE-Latenz bei schweren Typen ([#47481](https://github.com/microsoft/TypeScript/issues/47481)).

**Konsequenzen fürs Design:** (a) alle Typ-Rekursion konsequent tail-rekursiv/akkumulatorbasiert; (b) IDE-Latenz ist ab Tag 1 eine Messgröße, kein Nachgedanke; (c) dynamische Shapes brauchen einen erstklassigen Escape-Hatch (Dimension als `number` statt Literal → degradiert kontrolliert zu Laufzeit-Checks) — *graduelle* Shape-Typisierung, kein Alles-oder-Nichts.

## 5. Der geschärfte USP

> **numtype verhält sich zu NumPy wie TypeScript zu JavaScript: Fehler, die bisher zur Laufzeit knallten, werden beim Tippen sichtbar — graduell, mit Escape-Hatch für dynamische Shapes, ohne den flexiblen Kern aufzugeben.**

Die ausformulierte, verteidigbare Fassung (jede Aussage quellengestützt):

> Shape-Mismatches sind *der* klassische NumPy-Laufzeitfehler. NumPys eigene Maintainer raten von Shape-Typing ab, und PyTorchs Request für statisch geprüfte Shapes ist seit sechs Jahren offen — das Problem übersteigt Pythons Typsystem strukturell. TypeScripts variadische Tupel-Typen machen Shape-Arithmetik auf Typ-Ebene erstmals praktikabel, aber niemand hat es ernsthaft gebaut: Der Stand der Technik ist ein Blog-Post-Demo ohne Broadcasting. numtype ist die erste Bibliothek, die es ernsthaft umsetzt — matmul-, Broadcasting- und Reduktions-Fehler erscheinen als Editor-Squiggle statt als Produktions-Crash, und der Editor-Hover zeigt jederzeit die inferierte Shape.

Drei Qualifikationen, die den USP ehrlich halten (Glaubwürdigkeit als Asset):
1. **„Python kann es nicht"** ist verifiziert, nicht behauptet (siehe §3).
2. **„TypeScript kann es"** heißt: *neu machbar, im Maßstab unbewiesen*. Für ein Forschungsprojekt ist genau das der Reiz — wir validieren keine bekannte Technik, wir loten die Grenze aus.
3. **Gültigkeitsbereich:** Die Garantie gilt für realistische Ränge und Op-Ketten, nicht „für jede erdenkliche Shape". Der graduelle Ansatz (Literal-Dims statisch, `number`-Dims zur Laufzeit) macht daraus ein Feature statt einer Fußnote.

## 6. Was die Dev-Community an numtype schätzen könnte

1. **Der Editor-Moment.** Rote Wellenlinie unter `matmul(a, b)` bei [2,3]×[5,4], Hover zeigt `NDArray<[2,4]>` — das ist in 10 Sekunden demonstrierbar und genau die Art Typ-System-Kunststück, die auf HN/X zirkuliert (potatogpt bekam dafür Aufmerksamkeit — als Wegwerf-Demo).
2. **Shape-Inferenz als Feedback-Schleife.** In Python führt man Code aus, um Shapes zu sehen; hier zeigt sie der Editor beim Tippen. „Type-driven development" für Numerik — der Hover ersetzt das REPL-Experiment.
3. **Graduell statt dogmatisch.** Der `number`-Escape-Hatch macht es in echtem Code benutzbar — der Fehler von Dependent-Types-Sprachen (Ergonomie-Steuer, Nische) wird strukturell vermieden. Dieselbe Adoption-Mechanik, die TypeScript selbst groß gemacht hat.
4. **Zero-Dependency, from scratch.** Rust-Kern + TS-Schicht ohne transitive Abhängigkeiten: auditierbar, supply-chain-arm — ein wachsendes Community-Anliegen.
5. **Der Forschungs-Charakter selbst.** „Wie weit trägt TSs Typsystem Tensor-Shapes?" ist publikationsfähig (Blog-Serie, Talk), unabhängig von der Adoption der Bibliothek. Die dokumentierten Grenzen (§4) sind selbst ein Beitrag.
6. **Lehrwert.** Broadcasting-Regeln als Typen sind ausführbare Dokumentation — der Typ-Fehler *erklärt* die NumPy-Semantik.

## 7. Use Cases

1. **LLM-/Embedding-Pipelines in TS** *(der zeitgemäßeste)*: Embeddings haben feste Dimensionen (`Vec<[1536]>` vs. `Vec<[3072]>`); Modell-Verwechslungen in RAG-/Similarity-Pipelines sind exakt die Fehlerklasse, die der Compiler fängt. TS ist die Sprache der AI-App-Schicht — deren Numerik-Werkzeug ist heute handgerollte Loops.
2. **ML-Pre-/Postprocessing im Browser**: ONNX Runtime Web hat 2,84M DL/Woche, aber Bild→Tensor (NHWC/NCHW, Normalisierung) ist heute Handarbeit — Layout-Verwechslungen sind Compile-Zeit-fangbar.
3. **Node-Backends ohne Python-Sidecar**: Scoring, Statistik, Empfehlungen — kleine numerische Lasten, die heute einen Python-Microservice erzwingen.
4. **Signal-/Audio-Verarbeitung im Web**: feste Frame-/Fenstergrößen sind Literal-Dimensionen — Idealfall für die Typ-Ebene.
5. **Simulation, Robotik, Grafik jenseits von vec3/mat4**: Kinematik-Ketten, Physik — statisch bekannte Matrixdimensionen.
6. **In-Browser Data Science & Lehre**: Notebooks (Observable & Co.), interaktive Visualisierung, Unterricht mit sofortigem Typ-Feedback.

## 8. Abgeleitete nächste Schritte

1. **Typ-Ebenen-Spike zuerst** (Risiko = USP): Broadcasting, matmul, Achsen-Reduktion rein auf Typ-Ebene prototypen; gegen die Grenzen aus §4 testen (Rang-Tiefe, Op-Ketten-Länge, `tsc`-Zeit, Editor-Latenz). Erst wenn das trägt, den Rust/WASM-Kern beginnen.
2. **DX-Messharness ab Tag 1**: `tsc --extendedDiagnostics`-Budget + Editor-Latenz als CI-Metrik.
3. **Positionierung fürs spätere OSS-Release**: nicht „NumPy-Klon" (gegen stdlib/numpy-ts unverteidigbar), sondern die Typ-Ebene als Kern-Story; Forschungsnotizen von Anfang an als veröffentlichbare Artefakte schreiben.

---

## Anhang A — Vollbericht Wettbewerbslandschaft (EN, Agent A, 2026-07-09)

# Competitive Landscape: NumPy-like Numeric Libraries for JS/TS

All figures below are from live queries against the npm registry API, GitHub REST API, and page fetches on 2026-07-09; each claim is tagged with its source. Where I could not verify a number, I say so rather than estimate.

## 1. General-purpose ndarray / numeric libraries

**scijs/ndarray** — the foundational "view over a flat typed array" abstraction many other libs (numjs, cwise) build on. Core repo [scijs/ndarray](https://github.com/scijs/ndarray) — GitHub API reports last push **2022-01-29**, 1,244 stars. The underlying package is actually published from `mikolalysenko/ndarray` on npm; that GitHub repo no longer resolves via the API (moved/deleted), and its last npm publish was **2020-01-05** (`ndarray@1.0.19`, [registry](https://registry.npmjs.org/ndarray)). Despite being frozen, it's still pulled **~1.47M times/week** on npm — almost entirely as a transitive dependency of other tooling, not direct NumPy-style usage. **Verdict: effectively dead upstream, but load-bearing infrastructure.** Typing: community `@types/ndarray` exists; shapes are `number[]`, runtime-only — no compile-time shape checking.

**@stdlib/ndarray / stdlib** ([github.com/stdlib-js/stdlib](https://github.com/stdlib-js/stdlib)) — by far the most actively maintained general numerical library in this space: 5,878 stars, pushed **2026-07-09** (today), `@stdlib/ndarray` last published **2026-06-05** (v0.4.1). `@stdlib/ndarray` itself gets ~32.6K downloads/week ([npm API](https://registry.npmjs.org/@stdlib/ndarray)); the umbrella `stdlib` ecosystem is much broader (thousands of micro-packages: linear algebra, stats, BLAS bindings, string libs). Pure JS with optional native/WASM fast paths for hot ops. It ships extensive `.d.ts` typings, but shapes are always `number[]`/generic array types — I found **no evidence of literal-type or generic-parameterized static shape checking** in its type defs or docs. This is the closest thing to a "living NumPy for JS," but it competes on breadth/robustness, not on type-level shape safety.

**numjs** (`nicolaspanel/numjs`) — 2,449 stars but **frozen since npm publish 2021-09-27** and last GitHub push 2024-05-31 ([repo](https://github.com/nicolaspanel/numjs)), 512 downloads/week. Effectively dead. A community fork, **`@d4c/numjs`** ([grimmerk/numjs](https://github.com/grimmerk/numjs)), is alive (last push 2026-03-11, published 0.17.35 on 2026-03-07) but tiny (27 stars, 231 dl/week) and markets itself mainly as "numjs + TS typings," not as adding static shape safety — types describe the API surface, not shapes.

**math.js** ([josdejong/mathjs](https://github.com/josdejong/mathjs)) — 15,055 stars, actively maintained (pushed 2026-05-12, published v15.2.0 2026-04-07), and huge adoption at **2.83M downloads/week**. Scope is broader/shallower than NumPy: symbolic + numeric expression evaluation, matrices, units, complex numbers, a JS-expression parser. Pure JS, no WASM/GPU backend. TypeScript types are runtime-shaped (`Matrix`/`number[][]`) — no static shape typing.

**numpy-ts** ([dupontcyborg/numpy-ts](https://github.com/dupontcyborg/numpy-ts), [site](https://numpyts.dev/)) — newest entrant, actively developed (GitHub pushed **2026-07-09**, npm published **2026-06-19**, v1.5.0), 374 stars, ~486 dl/week. Single-maintainer project claiming 94% NumPy API coverage (476/507 functions), pure TS + WASM, "no native modules." Docs state "Full TypeScript type definitions for every function, dtype, and array operation" — but per the docs I fetched this is comprehensive **runtime** type coverage, not compile-time shape verification; there's no literal-shape generic system. This is the most direct current attempt at "NumPy for TS," worth watching, but doesn't solve the shape-typing problem either.

**GPU.js** ([gpujs/gpu.js](https://github.com/gpujs/gpu.js)) — 15,361 stars but stalling: last npm publish 2022-11-16, last GitHub push 2025-04-21 (not archived, occasional maintenance-only activity), 21,982 dl/week. JIT-compiles JS functions to WebGL/WebGPU kernels for elementwise/matrix ops; not an ndarray API, no shape typing.

## 2. ML-inference-oriented (not general ndarray math)

**TensorFlow.js** ([tensorflow/tfjs](https://github.com/tensorflow/tfjs)) — 19,130 stars, repo still active (pushed 2026-06-23) but no npm release since **2024-10-21** (`@tensorflow/tfjs@4.22.0`), i.e. ~21 months stale on releases despite ongoing commits. 545K dl/week — heavy real-world use. Backends: pure JS, WASM, WebGL, Node native (libtensorflow). TypeScript ships rank-specific types (`Tensor1D`…`Tensor4D`), but per TF's own docs and API reference these encode *rank*, not concrete *dimensions* — `tf.tensor2d([...], [2,3])` still takes a runtime `shape` array; a mismatched shape is a **runtime** error, not a compile error. Scope is ML tensors/ops/autodiff, not a general NumPy replacement (no free-form indexing/slicing ergonomics NumPy users expect).

**ONNX Runtime Web** ([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)) — 21,046 stars, very active (pushed today, 2026-07-09), latest npm publish **2026-06-19** (v1.27.0), 2.84M dl/week. Pure inference runtime (loads pre-trained ONNX graphs; you don't build/compose tensors interactively). `Tensor.dims` is typed `(number | string)[]` at runtime (string = symbolic/dynamic dim) — confirmed against the ORT JS API source ([onnxruntime.ai/docs/api/js](https://onnxruntime.ai/docs/api/js/)); no compile-time shape typing, and it isn't meant to be a general array library at all.

**danfo.js** ([javascriptdata/danfojs](https://github.com/javascriptdata/danfojs)) — 5,050 stars, but stagnant: no npm version in >12 months per Snyk, last real publish **2025-04-03** (v1.2.0), ~6,115 dl/week, still receiving scattered issues into 2026 but no active development cadence. Pandas-style dataframe API (Series/DataFrame), built atop TF.js tensors underneath — not a general ndarray math library, and no shape typing.

## 3. Columnar / dataframe (adjacent, not ndarray-math)

**Apache Arrow JS** ([apache/arrow-js](https://github.com/apache/arrow-js)) — 103 stars (this JS-only split repo; the umbrella `apache/arrow` monorepo has far more, but this is the current home), active (pushed 2026-07-06), latest publish **2025-10-07** (v21.1.0), and enormous adoption: **3.54M dl/week**. Per its docs, this is a **columnar in-memory format / IPC interchange** library (Tables, Vectors, dictionary/typed columns) — explicitly not a numeric-computation library: no matmul/linear algebra, no shape system. `Vector<T>` generics encode dtype, not shape/length.

**nodejs-polars** ([pola-rs/nodejs-polars](https://github.com/pola-rs/nodejs-polars)) — 731 stars, actively maintained (pushed 2026-07-08, npm v0.25.1 published 2026-06-10), 283K dl/week. Rust-core DataFrame engine (the actual Polars) exposed to Node via native bindings (not primarily WASM — there is a separate, much less mature `nodejs-polars-wasm` package on npm for browser use). Dataframe/query-engine semantics (columns, lazy queries, joins), not an ndarray/matrix API — no shape typing, and no browser-first WASM story for the mainstream package.

## 4. TypeScript static/compile-time shape typing — prior art

This is the crux of numtype's proposed USP, and it's the thinnest part of the landscape:

- **No shipping, general-purpose library found that does this.** Extensive search (npm keyword search, GitHub topic search, blog/HN discussion) turned up no maintained npm package offering NumPy-breadth operations with compile-time-checked shapes.
- **PotatoGPT / sebinsua write-up** ([sebinsua.com/type-safe-tensors](https://sebinsua.com/type-safe-tensors)) is the most concrete prior art: a technique (not a library) using numeric-literal shapes (`as const`), "branded" `Var<Label>` types for runtime-known dims, and heavy conditional/mapped/distributive-conditional types to type-check `zip`/`matmul` shape compatibility at compile time. Explicitly demonstrated only for a small hand-rolled GPT forward pass, not packaged as a reusable library, and the author notes it pushes TS's type checker to its performance limits.
- **@hoff97/tensor-js** ([npm](https://www.npmjs.com/package/@hoff97/tensor-js), [github.com/Hoff97/tensorjs](https://github.com/Hoff97/tensorjs)) — dead (last publish 2021-04-03, last push 2021-04-07, 37 stars). Its generic type parameter encodes **dtype**, not shape (its own description: operations only compile "when using the same data type").
- **Shumai** (Meta, [facebookresearch/shumai](https://github.com/facebookresearch/shumai)) — dead/experimental (last npm publish 2023-01-25 v0.0.14, last GitHub push 2024-07-23, 18 dl/week today), Bun+Flashlight/ArrayFire native bindings (not WASM), GPU-capable — but shapes are runtime-only, no static typing.
- **TypeGPU** ([software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU), 31,999 dl/week, actively developed) is the most interesting adjacent precedent: it does real, shipping compile-time type inference across the CPU/GPU boundary for WGSL shader I/O (structs, byte layout/alignment). It is not an ndarray/NumPy library and doesn't do tensor-shape arithmetic checking, but it proves the general "advanced TS type inference for a numeric/GPU domain" pattern is viable and has real users — worth studying as a technique reference, not a competitor.
- Academic/other-language precedent exists (Scala `TensorSafe`, Rust `dfdx` via const generics, Haskell `tensor`/`tensor-ops`) confirming the *concept* is proven elsewhere, just not ported to a maintained TS library.

## 5. Rust → WASM numeric cores aimed at JS

- **Rust `ndarray` crate + wasm-bindgen**: no ready-made npm package — only ad-hoc example projects gluing `ndarray` + `wasm-bindgen`/`js-sys`, plus an open feature request on the `ndarray` repo itself asking for canonical JS-interop tooling ([rust-ndarray/ndarray#845](https://github.com/rust-ndarray/ndarray/issues/845)). No PyO3-equivalent maturity for JS.
- **Polars** (`nodejs-polars`) — covered above; the Rust core, but native-binding-first, not WASM-first, and dataframe-scoped not ndarray-scoped.
- **candle** (Hugging Face, Rust ML framework) — first-class WASM support, runs Whisper/LLaMA2-class models in-browser via `wasm-pack`-published bundles, but it's a model-inference framework, not exposed as a general ndarray-for-JS library, and has no TS shape typing.
- **burn** — backend-agnostic Rust ML framework with a WGPU/WASM backend; same story as candle: inference/training framework, not a JS-facing ndarray API, no TS typing story at all (it's Rust-side type safety only, via const-generic-ish approaches internally, not exposed to consumers in TS).
- **Verdict**: nothing ships a general Rust-ndarray-core-to-npm package with a NumPy-like JS/TS surface today; the space is open specifically for a from-scratch Rust/WASM kernel with a hand-built TS API, which matches numtype's plan.

## Comparison table

| Library | Last release | Repo pushed | Stars | DL/week | Scope | Backend | Static shape typing? |
|---|---|---|---|---|---|---|---|
| @stdlib/ndarray (+stdlib) | 2026-06-05 | 2026-07-09 | 5,878 | 32.6K | general ndarray | JS (+native fast paths) | No |
| ndarray (scijs/mikolalysenko) | 2020-01-05 | 2022-01-29 (fork) | 1,244 | 1.47M | general ndarray core | JS | No |
| numjs | 2021-09-27 | 2024-05-31 | 2,449 | 512 | NumPy-like | JS | No |
| @d4c/numjs (fork) | 2026-03-07 | 2026-03-11 | 27 | 231 | NumPy-like + TS typings | JS | No |
| numpy-ts | 2026-06-19 | 2026-07-09 | 374 | 486 | NumPy-like (94% coverage) | TS + WASM | No (full runtime typedefs only) |
| math.js | 2026-04-07 | 2026-05-12 | 15,055 | 2.83M | symbolic+numeric math | JS | No |
| TensorFlow.js | 2024-10-21 | 2026-06-23 | 19,130 | 545K | ML tensors/autodiff | JS/WASM/WebGL/native | No (rank types only) |
| ONNX Runtime Web | 2026-06-19 | 2026-07-09 | 21,046 | 2.84M | inference only | WASM/WebGL/WebGPU | No (`dims: (number\|string)[]`) |
| danfo.js | 2025-04-03 | 2026-04-15 | 5,050 | 6.1K | dataframes (on tfjs) | JS | No |
| Apache Arrow JS | 2025-10-07 | 2026-07-06 | 103 (split repo) | 3.54M | columnar/IPC, not math | JS | No (dtype generics only) |
| nodejs-polars | 2026-06-10 | 2026-07-08 | 731 | 283K | dataframe query engine | Rust native (WASM variant separate/immature) | No |
| GPU.js | 2022-11-16 | 2025-04-21 | 15,361 | 22.0K | JIT elementwise/matrix kernels | WebGL/WebGPU | No |
| Shumai | 2023-01-25 | 2024-07-23 | 1,172 | 18 | differentiable tensors | native (Bun FFI/Flashlight) | No |
| @hoff97/tensor-js | 2021-04-03 | 2021-04-07 | 37 | low | tensor ops | WASM/WebGL | Dtype only, not shape |
| TypeGPU | active | active | n/a (not checked) | 32.0K | WebGPU shader typing | WebGPU | Yes, but for shader I/O layout, not tensor-shape arithmetic |

## Gap statement

**No maintained, general-purpose, NumPy-like TypeScript library exists today that performs compile-time shape checking of array operations.** Every actively-maintained general-numeric library found (`@stdlib/ndarray`, `math.js`, `numpy-ts`) types its arrays' shapes as plain runtime `number[]`; every library that ships any shape-adjacent typing (TensorFlow.js's `Tensor1D..4D`, ONNX Runtime's `dims`) encodes only *rank*, never concrete *dimensions*, and shape mismatches surface as runtime exceptions in all of them. The only place a shape-arithmetic-at-the-type-level technique has actually been built and shown working in TS is a single blog-post-scale demo (PotatoGPT/sebinsua) for one hand-written forward pass, not a packaged, general, reusable library — and its author is explicit that it strains TypeScript's type checker (numeric-literal-only shapes, `as const` requirements, branded types for runtime dims). The nearest *adjacent* proof that this class of TS type-inference trick can be shipped and adopted at scale is TypeGPU (32K dl/week), but it solves a different problem (CPU↔GPU struct layout, not tensor-shape arithmetic across chained ops like matmul/broadcast/reshape).

So the gap is real and, as far as this search can establish, currently unclaimed: a from-scratch library that (a) has genuine NumPy-breadth operations, (b) is actively maintained, and (c) gives you `matmul([2,3], [3,4]) → [2,4]` as a compile error on mismatch rather than a runtime one, does not exist in the current JS/TS ecosystem. The risk is not "someone already built this" — it's whether the type-level machinery (as demonstrated only in a toy-scale project so far) scales to a full NumPy-sized API surface without becoming a TS-compiler-performance or DX liability; that's an engineering risk to validate early, not a market-gap risk.

---

## Anhang B — Vollbericht Shape-Typing Stand der Technik (EN, Agent B, 2026-07-09)

# NumType USP Research: Compile-Time Shape Checking — TypeScript vs. Python

## Python state of the art

**The language primitive exists; the ecosystem does not use it yet.** PEP 646 (Variadic Generics / `TypeVarTuple`) was accepted for Python 3.11 and is the enabling mechanism for encoding array shape as a type parameter. Support is uneven: **pyright** implemented it early (v1.1.108) and **fully supports** it; **mypy** lags — as of late 2023 it still had unimplemented edge cases (e.g. `Union[*Ts]` unpacking) per [python/mypy#16720](https://github.com/python/mypy/issues/16720), and general PEP 646 tracking continues at [python/mypy#12280](https://github.com/python/mypy/issues/12280). ([peps.python.org/pep-0646](https://peps.python.org/pep-0646/))

**NumPy's own maintainers say don't use shape typing today.** NumPy added syntax like `np.ndarray[tuple[M, N], np.dtype[...]]`, but its own tracker states plainly: *"Shape typing should not be used currently, because most functions will return shape-generic results, meaning that even correct shape types will typically just [fail] type checking"* ([numpy/numpy#28076](https://github.com/numpy/numpy/issues/28076)). The foundational feature request, [numpy/numpy#16544](https://github.com/numpy/numpy/issues/16544) ("Typing support for shapes"), remains **open** with no maintainer commitment or milestone — it is a wishlist item, not a roadmap item. A related feature request asking **pyright** to infer/verify shapes across ops was explicitly **closed as "not planned"** ([microsoft/pyright#8921](https://github.com/microsoft/pyright/issues/8921)) because it would require tracking shape transformations across arbitrary numpy operations — a much larger feature than PEP 646 alone provides.

An independent proof-of-concept ([taoa.io, "Shape typing numpy with pyright and variadic generics"](https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics/)) confirms the ceiling empirically: the author got matmul/`np.add`/`np.ravel` working for 1D/2D cases via hand-written overloads, but hit hard walls — **PEP 646 disallows multiple unpacked TypeVarTuples in one signature** (blocking general N-D support), and **there is no type-level arithmetic** on dimensions (multiplying/broadcasting dimensions can't be expressed, forcing hardcoded lookup tables). The author's own conclusion: a "proof of concept," not production-viable.

**The community's actual working tools all punt to runtime.** [jaxtyping](https://github.com/patrick-kidger/jaxtyping) (Patrick Kidger, active, 53 releases, latest June 2026) is explicitly a **runtime** checker: shape/dtype annotations are only enforced when combined with the `jaxtyped` decorator plus `beartype` or `typeguard` — a static type checker parses the syntax but verifies nothing. Its predecessor **torchtyping is deprecated**; its own README tells users to migrate to jaxtyping. Google DeepMind's **tensor_annotations** was itself an incomplete attempt and was **archived** Feb 2026. **nptyping** shows no PyPI release in 12+ months, and its own tracker ([nptyping#34](https://github.com/ramonhagenaars/nptyping/issues/34)) confirms static-checker (mypy) verification through operations like reshape was never solved.

A striking demand signal: **[pytorch/pytorch#26889](https://github.com/pytorch/pytorch/issues/26889), "Statically checked tensor shapes,"** opened September 2019, is **still open 6+ years later** with no active PR — stalled because only trivial broadcasting is checkable and ops like `cat` fundamentally resist static shape inference. Even Meta's own next-gen type checker, Pyrefly, is only *prototyping* static verification of jaxtyping annotations as of 2026 (HN item 47400892) — i.e., 4+ years after jaxtyping shipped, real static shape verification is still experimental, not delivered.

**Bottom line for Python:** a developer in 2026 cannot get a genuine static, editor-time shape-mismatch error for numpy/PyTorch/JAX code in any general or reliable way. The primitive (PEP 646) is there; numpy's own function stubs don't thread shapes through real signatures; the one serious static-checker feature request was rejected as out of scope; and the tools people actually use (jaxtyping) are runtime-only by design, catching bugs only when the buggy code path executes.

## TypeScript prior art & limits

**Prior art exists but is nascent, not competitive.** No maintained TypeScript library implements broadcasting, reductions, or arbitrary-rank shape typing today — numtype would be a first serious attempt in this specific niche:

- **[potatogpt](https://github.com/newhouseb/potatogpt)** (Ben Newhouse, ~45 stars) — the most concrete prior art: a GPT-2 forward pass in TS with branded type-level dimensions catching matmul shape mismatches via `tsc`. The author states explicitly: *"I've only implemented the bare minimum of tensor math... there's nothing like broadcasting implemented."* Purpose-built, one-off, not a library.
- **[sebinsua.com/type-safe-tensors](https://sebinsua.com/type-safe-tensors)** (May 2023) extends the same approach (overloaded matmul, type-safe `zip`) but hits the same wall: dimensions must be numeric literals or branded types; plain `number` or union dimensions break the system; `as const` is required everywhere to stop widening. Blog-post grade, no published package.
- **shumai** (facebookresearch, ~1.2k stars, active) and **tensor4ts/tensor4js** are real runtime tensor libraries but encode shape only as runtime `number[]` — no type-level checking at all.
- No hits on npm for literal names like `typed-tensor`, `shape-typed`, `tensor-type-ts`. TensorFlow.js's own `Shape` type is deliberately `(number|null)[]`, and no tfjs GitHub issue proposes compile-time shape typing.
- The underlying *idea* is precedented outside TS: OCaml's gradual tensor-shape checking (ESOP'23, [arxiv 2203.08402](https://arxiv.org/abs/2203.08402)), Haskell's Hasktorch, Rust's dfdx (const generics) — establishing this is a legitimate PL research direction, just one nobody has pushed hard specifically in TypeScript.

**Mechanics that make it possible:** TS 4.0's variadic tuple types (rest elements at arbitrary tuple positions, spreads of unknown-length propagating) are the load-bearing primitive for shape decomposition/recombination ([TS 4.0 release notes](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-0.html)).

**Hard limits, with real numbers, and they are real constraints for this design:**
- Recursive conditional type instantiation depth has moved across versions: originally a 50-level cap; [PR #45025](https://github.com/microsoft/TypeScript/pull/45025) raised it to 500 but caused real browser stack overflows and was walked back to 100 for general recursion.
- [PR #45711](https://github.com/microsoft/TypeScript/pull/45711) (shipped TS 4.5) added **tail-recursion elimination**: conditional types written in accumulator/tail-recursive form loop without consuming call stack, reaching **~1000 iterations** vs. ~100 for non-tail-recursive forms. This is the single most relevant lever for shape arithmetic — dimension/rank arithmetic must be written tail-recursively to get the higher ceiling.
- The "limit" is inconsistent and shape-dependent, not a clean invariant: [issue #49459](https://github.com/microsoft/TypeScript/issues/49459) shows one formulation tops out at 999 iterations while a functionally-equivalent one reaches 3153 — whether an operation trips the ceiling depends on syntactic structure of the recursive type, not tensor rank per se.
- TS5 introduced a **global ~5,000,000-instantiation budget per compilation** ([issue #53514](https://github.com/microsoft/TypeScript/issues/53514)), and TS5 was observed to instantiate substantially more objects than TS 4.9 for equivalent code — a version-sensitive regression risk.
- Tuple-length-based type-level number tricks are bounded by the same ~1000-ish ceiling — fine for tensor rank (small integers) but not for encoding large dimension sizes.
- Documented IDE/compiler performance degradation: [issue #47481](https://github.com/microsoft/TypeScript/issues/47481) reports measured compile times for large string-literal unions of 4.5s (5,000×5,000) up to 27s (15,000×10,000), with worse-than-linear scaling — still open, unresolved in the fetched thread.

**Net assessment (inference, grounded in the above):** high-rank tensors, large broadcast dimensions, or long chains of composed operations are the plausible failure modes for a type-level shape system in TS — not "recursion in general" but recursion depth/breadth proportional to rank and dimension count colliding with these version-fragile caps. A production design needs tail-recursive accumulator patterns throughout and should budget for real IDE latency at scale, not just correctness.

## Demand signals

- **jaxtyping** is active (latest release June 2026, 53 releases) and, per its author, "pretty widely-used across quite a few companies" ([kidger.site/thoughts/jaxtyping](https://kidger.site/thoughts/jaxtyping/)) — clear demand for *some* form of shape safety, but by design only a runtime compromise (shape checks happen once during JIT trace).
- The clearest durable demand signal is **[pytorch/pytorch#26889](https://github.com/pytorch/pytorch/issues/26889)**: a 6-year-old still-open feature request for statically checked tensor shapes from the flagship framework's own tracker, stalled on the *general* problem's difficulty, not lack of interest.
- Meta's Pyrefly prototyping static verification of jaxtyping annotations in 2026 (HN 47400892, title/snippet only, not independently fetched) shows the demand is live enough that a major type-checker team is actively working the problem — years after the "pragmatic" runtime tools shipped, confirming runtime checking was never considered fully satisfying.
- Reddit searches for "shape typing"/"typed tensors" + numpy returned no results — the demand conversation appears to live on GitHub issues, blogs, and HN rather than Reddit; treat this as absence of evidence on that one channel, not absence of demand.
- One HN thread directly titled "Shape typing in Python" (item 40022628, April 2024) exists but could not be fetched (rate-limited) — flagged as **unverified**, not cited for sentiment.
- **Framing paragraph on research languages:** Google Research's **Dex** ([google-research/dex-lang](https://github.com/google-research/dex-lang)) uses value-dependent types with typed indices to make shape errors compile-time errors, explicitly validating the underlying idea — but is labeled "early-stage experimental" and "not officially supported by Google in any capacity." Dependent-typing generally (Idris, Agda) stayed niche due to weak implicit-argument ergonomics, a split between proof-assistant and general-purpose goals, and difficulty compiling verified high-abstraction code efficiently. The pattern across both is the same tension that keeps Python's shape-checking stuck at "runtime plus best-effort static heuristics": full static guarantees require either a new language or an unacceptable ergonomics tax.

## Bottom line: is the USP claim true as stated, or how must it be qualified?

**The claim is directionally true and well-supported, but needs three qualifications to be defensible rather than overclaimed:**

1. **"Python cannot catch shape errors statically today in any practical way" — verified, not just asserted.** This is grounded in numpy's own maintainers telling users not to attempt shape typing ([numpy#28076](https://github.com/numpy/numpy/issues/28076)), a rejected pyright feature request ([pyright#8921](https://github.com/microsoft/pyright/issues/8921)), a 6-year-open PyTorch static-shapes request ([pytorch#26889](https://github.com/pytorch/pytorch/issues/26889)), and every production Python tool (jaxtyping) being runtime-only by explicit design. This part of the claim survives scrutiny cleanly.

2. **"TypeScript can" needs qualification, not retraction.** No existing TS library has actually delivered general compile-time shape checking with broadcasting and reductions at arbitrary rank — the prior art (potatogpt, sebinsua's post) explicitly stops at literal-dimension matmul and disclaims broadcasting as out of scope. So numtype isn't validating a *proven* approach, it's attempting something TypeScript's mechanics make *plausible but unproven at scale*. The pitch should say "TypeScript's type system makes this newly tractable and no one has built it seriously yet" rather than implying it's a solved, merely-unadopted technique.

3. **The claim should be scoped by rank/dimension size, not stated unconditionally.** TypeScript's own documented limits — ~100 non-tail-recursive / ~1000 tail-recursive instantiation depth, a version-fragile ~5M global instantiation budget, and multi-second IDE slowdowns at large union/tuple sizes — mean numtype's compile-time guarantees will hold reliably for realistic tensor ranks and common operation chains, but the marketing claim needs an implicit or explicit "within practical rank/size bounds" rather than "for any shape whatsoever," to avoid the same overclaiming that sank Dex's and dependent-typing's broader ambitions.

**Recommended reframing of the USP:** *"NumPy's own maintainers say don't rely on shape typing yet, and PyTorch's static-shapes request has sat open for six years because the general problem resists Python's type system. TypeScript's variadic tuple types make type-level shape arithmetic newly tractable — and numtype is the first library to seriously execute on it, catching common matmul/broadcast/reduction shape errors as editor squiggles instead of runtime crashes."** This keeps every clause anchored to a fetched source, concedes prior-art nascency honestly (a credibility asset, not a weakness), and implicitly scopes the guarantee to realistic tensor sizes rather than universal shape arithmetic.

---

## Addendum 2026-07-20 — numpy-ts re-verifiziert (nach Launch, Owner-Frage)

Anlass: Owner-Frage zur Abgrenzung, einen Tag nach dem v0.1-Launch. numpy-ts heute erneut
geprüft (numpyts.dev + npm-Registry, live): Version **1.5.0 → 1.6.0** (publiziert 2026-07-17 —
das Projekt shippt weiter schnell), 6,3 MB unpacked, weiterhin kein `dependencies`-Feld.
Website-Claims der Art nach unverändert: „94% API Coverage" (476/507), „Pure TypeScript + WASM",
„1.25x faster than NumPy on average", „NumPy-Validated" (20.000+ Vergleichstests gegen NumPy),
und „Full TypeScript type definitions for every function, dtype, and array operation. Catch
errors at compile time, not runtime."

**Der letzte Claim bleibt Signatur-/dtype-Ebene:** kein Hinweis auf Shape-Level-Typen — keine
literalen Dimensionen, keine generischen Shape-Parameter, keine berechneten Ergebnis-Shapes.
`matmul([2,3] × [5,4])` typechecked dort sauber und wirft zur Laufzeit. Das Anhang-A-Verdikt
(„most direct current attempt at ‚NumPy for TS', doesn't solve the shape-typing problem")
gilt unverändert; die README-Qualifikation „no existing TS library has delivered general
compile-time shape checking" ist damit auch nach dem Launch re-verifiziert.

Abgrenzung in einem Satz (seit heute auch als FAQ-Absatz in der README): numpy-ts portiert
NumPys **API** nach TypeScript (Breite, Validierung gegen NumPy), NumType bringt dem
**Typchecker** NumPys Shape-Arithmetik bei (Tiefe, Bit-Identität zwischen eigenen Backends,
gemessene Checker-Budgets). Die Projekte konkurrieren kaum — wer heute 476 Funktionen braucht,
ist bei numpy-ts besser aufgehoben; wer Shape-Fehler beim Tippen will, hier. Beobachtungspunkt
bleibt: Sollte numpy-ts je Shape-Typen nachrüsten, ist unser Vorsprung die bewiesene
Maschinerie samt gemessener Budgets (docs/, Spikes 01–06).
