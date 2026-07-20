# Dogfooding-Scheibe: RAG-Demo auf numtype — Ergebnisse

Status: Friction-Log = Rohfassung der ausführenden Session (bewusst unkuratiert
erhalten, Primärevidenz); Op-Wunschliste unten = kuratiert (Orchestrator);
Post-Verification-Addendum am Ende. Spec: docs/dogfooding-rag-spec.md (Version 3).
Datum: 2026-07-20.

## Kontext

Erste echte Konsumenten-Anwendung auf dem veröffentlichten `numtype@0.1.1`:
`examples/rag-demo` — eine deterministische RAG-Retrieval-Demo (16 Dokumente, 8 Queries,
from-scratch gehashte Zeichen-Trigramm-Embeddings, Ähnlichkeitssuche per `matmul`). Kein
LLM-/Generierungs-Teil (Retrieval only, Nicht-Ziel laut Spec). Der eigentliche Deliverable
ist dieser Friction-Log, nicht die Demo selbst.

## Friction-Log

Jeder Eintrag: Intent (NumPy-Idiom) → Workaround (Datei:Zeile) → Kosten → Wunschlisten-Kandidat.

### F1 — Kein elementweiser unärer sqrt/map-Op

- **Intent:** `np.sqrt(sumSquares)` als letzter Schritt der natürlichen
  L2-Normalisierungs-Kette `mul → sum(axis) → sqrt → reshape → div`. Die ersten drei
  Schritte SIND natürliches numtype (elementweise `mul`, dann ein `sum(axis)`, dessen
  Ergebnis-Shape numtype selbst berechnet).
- **Workaround:** `examples/rag-demo/main.ts:60` (Korpus) und `:106` (Queries) —
  `sumSquares.data` als rohes `Float64Array` auslesen, `Math.sqrt` elementweise per
  Hand-Loop anwenden, über `NDArray.fromArray` mit derselben literalen Dimension
  wieder einbetten.
- **Kosten:** ≈4 Zeilen pro Aufruf-Stelle (zweimal dupliziert — Korpus + Queries).
  **Kein** Typ-Ebenen-Verlust: `N`/`Q` bleiben literale `const`-Bindings, das
  rekonstruierte Array ist wieder `NDArray<[16]>` bzw. `NDArray<[8]>`, nicht auf
  `NDArray<[number]>` geweitet — die Degradation ist rein syntaktisch/Boilerplate,
  nicht typ-semantisch.
- **Kandidat:** ein elementweiser Unary-Map-Op, mindestens `sqrt` (IEEE-exakt →
  determinismus-sicher laut Spec D5, kein Covenant-Non-Goal). Ein allgemeineres
  `.map(fn)` wäre NumPy-näher, aber ein einzelnes `.sqrt()` deckt diesen konkreten
  Friction-Punkt bereits vollständig ab.

### F2 — Kein Skalar-Overload für die elementweisen Binär-Ops

- **Intent:** `chunkSum / 2` — ein NumPy-Skalar-Broadcast, um eine Mean-Pooling-Summe
  durch die Chunk-Anzahl zu teilen.
- **Workaround:** `examples/rag-demo/main.ts:178-188` — den Skalar in ein
  `NDArray.fromArray([1], [2])` verpacken und auf normales Shape-Broadcasting
  (`[256] ÷ [1] → [256]`) setzen, statt eines echten Skalar-Arguments.
- **Kosten:** ein zusätzlicher `fromArray`-Aufruf pro Skalar-Operation; kein
  Typ-Ebenen-Verlust (`[256]` bleibt literal). Lesbarkeit leidet spürbar — der Leser
  muss wissen, DASS `[1]`-Broadcasting hier der Trick ist, es liest sich nicht wie
  "durch 2 teilen".
- **Kandidat:** `number`-Overloads für `add`/`sub`/`mul`/`div` (oder ein dedizierter
  `.scale(n)`-Helper). Zusammen mit F1 der am häufigsten erwartete Kandidat aus dem
  HANDOFF (dort als "mean" vorhergesagt — die eigentliche Lücke ist granularer: nicht
  `mean` selbst fehlt strukturell, sondern die Skalar-Division, die eine
  `mean`-Implementierung bräuchte).

### F3 — Kein Multi-Dim-Element-Getter (`.at(i, j)` / `.item()`)

- **Intent:** einen einzelnen Ähnlichkeits-Score aus der `[8, 16]`-Score-Matrix lesen,
  um ihn gegen den unabhängig berechneten Cross-Check-Wert zu vergleichen.
- **Workaround:** `examples/rag-demo/main.ts:131` (`similarities.slice(qi)`) — der
  tatsächlich gewählte Weg ist nicht mal ein Workaround im eigentlichen Sinn: eine
  Integer-Spec auf der führenden Achse via `slice()` IST der natürliche numtype-Aufruf
  und liefert direkt ein echtes `NDArray<[16]>` (die Rang-Reduktion ist eine
  dokumentierte, statische Eigenschaft von `slice()`, unabhängig vom Laufzeitwert).
  Die eigentliche Reibung kommt EINEN Schritt später (siehe F4): das Ranking der 16
  Scores gegeneinander verlässt trotzdem NDArray und iteriert über `.data` per Hand.
- **Kosten:** gering — dieser Punkt ist eher eine Beobachtung als eine echte Lücke;
  `slice()` deckt den 1-D-Fall bereits gut ab. Für einen echten Skalar-Read (z. B. das
  eine Element `[qi, docIdx]` direkt, ohne über ein Zwischenarray zu gehen) gäbe es
  keinen direkten Weg außer verkettetem `.slice(qi).slice(docIdx)` (Rang 0) oder
  manueller Flat-Index-Arithmetik in `.data` — beides ungenutzt in dieser Demo, aber
  als Beobachtung notiert.
- **Kandidat:** niedrige Priorität — `slice()` deckt den praktischen Bedarf ab; ein
  `.item()`/`.at(...)`-Skalar-Getter wäre nice-to-have, kein Blocker.

### F4 — Kein argmax/topk-Op

- **Intent:** die 16 Ähnlichkeits-Scores einer Query ranken, um Top-1/Top-2 samt Margin
  zu bestimmen.
- **Workaround:** `examples/rag-demo/main.ts:131-135` — `Array.from(rowScores.data)`,
  dann `.map`/`.sort` in reinem JS.
- **Kosten:** ≈6 Zeilen pro Ranking-Stelle; verlässt die NDArray-Welt komplett (die
  Typ-Ebene hat für dieses JS-Array ohnehin nichts mehr zu sagen — kein Shape-Verlust
  im numtype-Sinne, aber der Ausstieg selbst ist der Reibungspunkt). Dieselbe manuelle
  Ranking-Logik taucht in der Mean-Pooling-Sektion noch einmal auf
  (`examples/rag-demo/main.ts:189-200`, dort sogar ohne den `NDArray`-Zwischenschritt,
  direkt als lineare Max-Suche über einen `cosineSimilarity`-Aufruf pro Dokument).
- **Kandidat:** `argmax`/`argsort`/`topk`. Der klarste Einzelkandidat dieser Demo — jede
  Retrieval-Anwendung braucht Ranking, und aktuell gibt es dafür buchstäblich keine
  numtype-Op.

### F5 — `fromArray` baut nur EINE flache Matrix, kein Stapeln aus Zeilenvektoren

- **Intent:** `np.array([embed(t) for t in texts])` bzw. `np.stack(...)` — eine Matrix
  aus N unabhängig berechneten Zeilenvektoren zusammensetzen.
- **Workaround:** `examples/rag-demo/embedding.ts` (`embedMatrix`) — ein selbst
  geschriebener Flatten-Helper (`Float64Array#set` an der richtigen Zeilen-Offset).
- **Kosten:** ≈10 Zeilen einmalig (wiederverwendbar für Korpus UND Queries in dieser
  Demo). Kein Typ-Ebenen-Verlust — der Helper gibt einfach ein flaches `Float64Array`
  zurück, das `fromArray` unverändert konsumiert.
- **Kandidat:** ein `NDArray.stack(rows: readonly NDArray<[D]>[])`- oder
  `NDArray.fromRows(...)`-Konstruktor. Niedrigere Priorität als F1/F2/F4 (der
  Workaround ist klein und einmalig), aber ein sehr natürlicher NumPy-Reflex, der
  aktuell fehlt.

### F6 — `toNestedArray()` gibt `unknown` zurück

- **Beobachtung (keine aktive Nutzung in dieser Demo — main.ts druckt Shapes/Scores,
  keine verschachtelten Arrays):** aus `spike/demo.ts`s eigenem Muster bekannt
  (`JSON.stringify(arr.toNestedArray())`), hier nur als Randnotiz aufgenommen, weil die
  Demo bewusst NICHT auf `toNestedArray()` zurückgreift, sondern `.shape`/`.data`
  direkt liest — vermutlich ein Hinweis, dass `.data` für numerische
  Inspektion/Debugging oft der direktere Weg ist als der `unknown`-typisierte
  Nested-Array-Pfad. Kein eigenständiger Wunschlisten-Kandidat, nur eine Beobachtung.

## Kein-Friction-Bereich (natürlich funktionierte numtype-Formulierung)

Zur Kalibrierung, was NICHT reibte:

- `NDArray.fromArray([N, D], flatBuffer)` — literale `[16, 256]`-Shape sofort im Hover.
- `corpusMatrix.mul(corpusMatrix)` — elementweises Quadrieren, Shape unverändert.
- `squared.sum(1)` — Achsen-Reduktion, Ergebnis-Shape `[16]` vom Checker berechnet.
- `rowNorms.reshape([N, 1])` — Broadcast-Vorbereitung, Produkt-Gleichheit statisch
  geprüft.
- `corpusMatrix.div(rowNormsCol)` — Broadcast-Division `[16,256] ÷ [16,1] → [16,256]`.
- `queryNormalized.matmul(corpusNormalized.transpose())` — EIN `matmul` für die
  gesamte Query×Dokument-Ähnlichkeitsmatrix, `[8,256] @ [256,16] → [8,16]`.
- `similarities.slice(qi)` — Integer-Index dropt die Achse statisch, auch bei
  Laufzeit-`qi`.
- `rawQueryRow.cosineSimilarity(rawDocRow)` / `.dot(...)` — Cross-Check-Rechnung,
  Rang-1-Guard hält beide Operanden ehrlich.
- Die zwei `@ts-expect-error`-Pins (`matmul`-Dim-Mismatch, `dot`-Rang-Mismatch) greifen
  beide exakt an der erwarteten Argument-Stelle.

## Beobachtungen

- **Hovers:** `corpusMatrix`, `queryMatrix`, `similarities` zeigen durchgehend saubere
  literale Tupel (`NDArray<[16, 256]>`, `NDArray<[8, 256]>`, `NDArray<[8, 16]>`) — keine
  einzige Stelle in der Demo degradierte unerwartet zu `NDArray<[number, number]>`.
  Selbst `similarities.slice(qi)` mit einem Nicht-Literal-`qi` (Schleifenvariable)
  bleibt `NDArray<[16]>`, weil die Rang-Reduktion einer Integer-Spec eine rein
  strukturelle (Spec-Form-abhängige), nicht wertabhängige Eigenschaft ist — genau das
  von `spike/src/slice.ts` dokumentierte Verhalten.
- **Fehler am Argument:** beide `@ts-expect-error`-Pins (matmul-Dim-Mismatch,
  dot-Rang-Mismatch, `examples/rag-demo/main.ts` Ende) lösen zuverlässig aus; entfernt
  man eine der beiden Kommentarzeilen, meldet `pnpm run check` den Fehler exakt an der
  Argument-Position des jeweiligen Aufrufs — konsistent mit der M3-Invariante.
- **Echte Scores/Margins (reproduzierbar, `pnpm -C examples/rag-demo run demo`,
  zweimal identisch verifiziert):**

  | Query | erwartet | top-1 (Score) | top-2 (Score) | Margin |
  |---|---|---|---|---|
  | "TypeScript's static types catch mistakes before your code runs." | doc0 | doc0 (0.8006) | doc12 (0.4804) | 0.3203 |
  | "Why are NumPy arrays fast for math operations?" | doc1 | doc1 (0.4431) | doc5 (0.3653) | 0.0778 |
  | "How does a neural network learn from training data?" | doc2 | doc2 (0.5679) | doc6 (0.4087) | 0.1592 |
  | "What is matrix multiplication in linear algebra?" | doc3 | doc3 (0.5034) | doc1 (0.3005) | 0.2030 |
  | "How does Rust prevent data races without garbage collection?" | doc4 | doc4 (0.7342) | doc5 (0.4027) | 0.3315 |
  | "What does cosine similarity measure between vectors?" | doc8 | doc8 (0.6360) | doc11 (0.3246) | 0.3114 |
  | "What is retrieval augmented generation for LLMs?" | doc11 | doc11 (0.6395) | doc5 (0.4011) | 0.2384 |
  | "How does NumType check array shapes at compile time?" | doc15 | doc15 (0.6948) | doc12 (0.3892) | 0.3056 |

  Alle acht Margins liegen über der gepinnten Schwelle 0.03 — die knappste (Query 1,
  0.0778) wurde bewusst so belassen (keine künstliche Aufblähung), die einzige Query,
  die tatsächlich neu formuliert werden musste, um die Schwelle komfortabel zu
  überschreiten, war Query 0 (ursprünglich 0.0466 Margin, siehe unten).
- **Mean-Pooling-Sanity:** ein aus zwei Chunks gepoolter Dokumentvektor (Chunks von
  Dokument 15) retrievt korrekt Dokument 15 zurück (Score 0.9901, Margin 0.3983 zum
  nächstbesten Kandidaten), und die Kosinus-Ähnlichkeit zum direkt embeddeten
  Volltext desselben Dokuments liegt bei 0.9901 — ein plausibles, hohes,
  nicht-triviales Ergebnis für ein reines Trigramm-Modell.
- **Iterationsschritt bei der Query-Auswahl (D3 v2, "zu knappe Query ersetzen"):**
  Query 0 wurde EINMAL umformuliert. Die ursprüngliche Formulierung ("How does
  TypeScript catch bugs before the program runs?") erzielte nur 0.0466 Margin
  (Konkurrenz durch Dokument 12, "The TypeScript compiler infers precise literal
  types…", das ebenfalls stark TypeScript-lastiges Vokabular teilt). Die Neuformulierung
  ("TypeScript's static types catch mistakes before your code runs.") — näher an
  Dokument 0s eigenem Wortlaut, weniger Überlappung mit "compiler"/"infers" — hob die
  Margin auf 0.3203. Alle anderen sieben Queries brauchten keine Anpassung im ersten
  Anlauf.
- **Determinismus:** `pnpm -C examples/rag-demo run demo` zweimal hintereinander
  ausgeführt, Ausgabe (inkl. aller Scores) byte-identisch (`diff` leer) — erwartungsgemäß,
  da die Hash-Funktion (djb2) und alle Fließkomma-Operationen deterministisch sind und
  keine Laufzeit-Randomisierung im Pfad liegt.

## Vorläufige Priorisierung (Rohmaterial für die Orchestrator-Kuratierung)

Nach Häufigkeit × Workaround-Kosten, ungeprüft gegen die volle Spec-D5-Vorlage
(Runtime-Implikation/Determinismus-Einordnung fehlt hier noch bewusst — das macht der
Orchestrator):

1. **argmax/topk** (F4) — trat zweimal auf (Query-Ranking, Mean-Pooling-Suche), verlässt
   NDArray jedes Mal komplett, kein Ersatz vorhanden.
2. **Skalar-Overloads** (F2) — trat einmal explizit auf (Mean-Pooling-Division), aber
   strukturell in jeder Reduktions-zu-Mittelwert-Pipeline zu erwarten.
3. **Elementweiser sqrt/Unary-Map** (F1) — trat zweimal auf (Korpus- und
   Query-Normalisierung), IEEE-exakt und damit determinismus-unproblematisch.
4. **`stack`/`fromRows`** (F5) — trat einmal auf, kleiner, aber sehr NumPy-typischer
   Reflex.
5. **Skalar-Element-Getter** (F3) — niedrigste Priorität, `slice()` deckt den
   praktischen Bedarf bereits weitgehend ab.

## Op-Wunschliste (kuratiert — Spec D5/T4)

Je Kandidat: Evidenz (Friction-Log) · NumPy-Analog · Typ-Ebene (Shape-Funktion +
M2-Degradation) · Runtime/M1-Einordnung · Priorität. Umsetzung ist NICHT Teil dieser
Scheibe (Spec D9) — jede Op wird eine eigene Scheibe mit eigener Spec.

### W1 · `argmax` / `topk` (aus F4) — Priorität 1

- **Evidenz:** F4, zweimal (main.ts:132–134 Query-Ranking; main.ts:193–202
  Mean-Pooling-Suche); einzige Stelle, an der die Demo die NDArray-Welt VERLASSEN
  muss, ohne dass irgendeine numtype-Op existiert.
- **NumPy:** `np.argmax(x, axis)` / `np.argsort` / `np.take`; Retrieval braucht
  konkret „Top-k-Indizes eines Rang-1-Scores-Vektors".
- **Typ-Ebene:** `argmax()` auf Rang 1 → `number`; `argmax(axis)` → Shape via
  vorhandenem `ReduceAxis` (Maschinerie existiert seit Kern 09/Item 11 inkl.
  Union-/keepdims-Degradation). `topk(k)` mit literalem k → `[k]`-Shape
  (Literal-Arithmetik vorhanden); wide/Union-k degradiert nach M2 zu no-claim.
- **Runtime/M1:** reine Vergleichsschleife, keine Arithmetik → Bit-Parität trivial;
  zu SPEZIFIZIEREN sind Tie-Breaking (NumPy: erster Treffer) und NaN-Semantik
  (NumPy-argmax propagiert NaN-Positionen!) sowie der Rückgabe-Dtype (numtype ist
  f64-only; Indizes als exakte f64-Integrale oder `number[]` — Design-Entscheidung
  der Op-Scheibe). Kann runtime-only starten (M1 bindet KERNEL an Referenz;
  eine Op ohne WASM-Kernel verletzt M1 nicht — Surface-Asymmetrie
  NDArray/WNDArray ist dann die bewusste Scope-Frage der Op-Scheibe).
- **Warum P1:** trat doppelt auf, hat NULL Ersatz in der Surface, und JEDE
  Retrieval-/Ranking-Anwendung braucht es.

### W2 · Skalar-Overloads für `add`/`sub`/`mul`/`div` (aus F2) — Priorität 2

- **Evidenz:** F2 (main.ts:187–188, `fromArray([1],[2])`-Wrap für „÷2").
- **NumPy:** `x / 2` (Skalar-Broadcast).
- **Typ-Ebene:** Overload `div(s: number): NDArray<S>` — Shape bleibt identisch,
  keine neue Guard-Fläche, keine Degradationskanten (M2-neutral), M3 unberührt.
- **Runtime/M1:** intern als `[1]`-Broadcast auf die EXISTIERENDEN Kernel abbilden →
  Bit-Parität geschenkt (identischer Codepfad); kein neuer Rust-Bedarf, kein
  Freeze-Kontakt (reine TS-Surface-Ergänzung, append-only in ndarray.ts/resident.ts).
- **Warum P2:** billigster Kandidat im Verhältnis zum Ergonomie-Gewinn; zugleich der
  Baustein, der `mean` (HANDOFF-Erwartung) praktisch kostenlos macht:
  `x.sum(axis).div(n)`. Ein eigenes `mean(axis, keepdims)` bleibt als Folge-Kandidat
  notiert (Shape-Maschinerie = 1:1 `sum`; M1-Feinheit: „Summe, dann EINE Division
  pro Element" als bindende Reihenfolge spezifizieren, nicht `sum * (1/n)` —
  unterschiedliche Rundung).

### W3 · Elementweises `sqrt` (benannte exakte Unary-Ops, aus F1) — Priorität 3

- **Evidenz:** F1, zweimal (main.ts:59–61, 105–107) — der einzige Bruch in der
  sonst durchgehend natürlichen L2-Normalisierungs-Kette.
- **NumPy:** `np.sqrt(x)` (ufunc-Familie).
- **Typ-Ebene:** Shape-erhaltend (`NDArray<S> → NDArray<S>`), null neue
  Degradationskanten.
- **Runtime/M1:** `Math.sqrt`/`f64::sqrt` sind IEEE-754-korrekt gerundet →
  determinismus-sicher, Bit-Parität JS⇄WASM erwartbar exakt (im Differential-Harness
  beweisen). WICHTIG: nur BENANNTE exakte Ops (sqrt, abs, neg) — ein generisches
  `.map(fn)` wäre nicht kernel-spiegelbar und öffnet die Nicht-Determinismus-Tür
  durch die Hintertür. **exp/log/sin bleiben explizit AUSSEN** (Covenant-Non-Goal
  „keine transzendenten Ops ohne Determinismus-Entscheidung", FOLLOWUPS
  „Transzendente Ops"; sqrt gehört NICHT zu dieser Klasse — IEEE verlangt für sqrt
  korrekte Rundung, für exp/sin nicht).
- **Warum P3:** klarer, doppelt belegter Bedarf; minimal größerer Spec-Aufwand als
  W2 (neue Kernel-Familie, wenn WASM-Parität gewünscht).

### W4 · `stack` / `fromRows` (aus F5) — Priorität 4

- **Evidenz:** F5 (embedding.ts `embedMatrix` — selbstgebauter Row-Flatten-Helper).
- **NumPy:** `np.stack([...])` / `np.array([row for ...])`.
- **Typ-Ebene:** Tupel-Input `stack([a, b, c])` mit gleichem literalen `[D]` →
  `[3, D]` (Variadic-Tuple-Länge = Rang-Arithmetik über kleine Ints, erlaubt);
  Array-Input (Länge unbekannt) degradiert ehrlich zu `[number, D]` (M2-konform).
  Gleiche-D-Prüfung = vorhandene `DimEq`-Maschinerie.
- **Runtime/M1:** reines memcpy (Movement-Op wie transpose/reshape) — keine
  Arithmetik, Bit-/NaN-Payload-Erhalt wie bei den bestehenden Movement-Ops
  (vgl. FOLLOWUPS NaN-Payload-Fixtures).
- **Warum P4:** Workaround ist klein und einmalig pro Projekt; der NumPy-Reflex ist
  aber real und der Erste-Eindrucks-Effekt für Umsteiger hoch.

### W5 · Skalar-Getter `.item(i, j)` / `.at(...)` (aus F3) — Priorität 5

- **Evidenz:** F3 (Beobachtung; `slice()` deckte den Demo-Bedarf).
- **NumPy:** `x[i, j]` / `x.item(...)`.
- **Typ-Ebene:** Rückgabe `number`, Literal-Index-Bounds-Checks existieren bereits
  (Spike 03) und wären wiederverwendbar.
- **Runtime/M1:** trivialer strided Read, keine Kernel-Frage.
- **Warum P5:** kein Blocker; reine Ergonomie-Kür.

### Kalibrierung gegen die HANDOFF-Erwartung (ehrlich)

Erwartet waren „mean, concat/stack, argmax/topk". Bestätigt: **argmax/topk** (stark,
doppelt) und **stack** (schwächer, einmal). **mean** erschien granularer als erwartet:
strukturell fehlt nicht `mean` selbst, sondern die Skalar-Division (W2); `mean` wird
als W2-Folgekandidat geführt. **concat** trat in diesem Workload NICHT auf — bleibt
ohne Evidenz und darum bewusst OHNE Wunschlisten-Platz (Evidenz-Regel D5).
Unerwartet stark: **sqrt** (W3) — von der Erwartungsliste gar nicht genannt, aber
zweimal real gebremst. Transzendente (exp/softmax) wurden im gesamten Workload NIE
gebraucht — das Covenant-Gate hat diese Scheibe nichts gekostet.

## Post-Verification-Addendum (2026-07-20)

Verify-Runde nach docs/verify-runde-template.md, Stufe 3 (A + B + C parallel, drei
frische Kontexte), plus mechanisches covenant-check.sh + `graph-a-lama query lint`
(beide grün, 0 Befunde).

- **Baustein 0 (vor der Implementierung, brainroute:deep): 1 BLOCKER gefangen** —
  pnpm-11-`minimumReleaseAge` hätte das Gate nach jedem Release gebrochen; als
  Spec-v2 eingearbeitet (pnpm-workspace.yaml `minimumReleaseAge: 0`). Details im
  Spec-Addendum.
- **Baustein A (Spec-Konformität): CONFIRMED mit Auflagen.** Alle D1–D9 und T1–T8
  bestätigt; Gates frisch: test:example 0 · check 0 · check:diag EXAKT 187,918 @ 135 ·
  test:core 822/822 · lint 0/0 · GFM 0 `<del>` × 4 Dateien. Pflicht-Mutant
  (Query-3-Erwartung 3→7): Exit 1 am erwarteten Assert (main.ts:146), Revert
  SHA256-identisch bewiesen. T7 empirisch: entfernter `@ts-expect-error` → TS2741
  exakt an der Argument-Position. Auflage T9 (Doc-Platzierung) war zum Prüfzeitpunkt
  planmäßig noch offen und ist mit diesem Addendum + CLAUDE.md-Einzeiler +
  Projekt-Log-Append + FOLLOWUPS-Einträgen erfüllt. Klärungsauftrag test:core:
  **822 auch am Vor-Scheiben-HEAD 7fbdb28 (frischer Worktree)** — die
  CLAUDE.md-„818" war vorbestehende Drift, korrigiert.
- **Baustein B (adversarial): HÄLT-mit-Befunden.** MAJOR (behoben): die D6-/CI-
  Kommentar-Behauptung „kein Root-pnpm-install" war falsch — `pnpm test:example`
  installiert auf kaltem Runner implizit die Root-devDeps (zweifach Cold-HOME-
  reproduziert); CI-Job auf direkte `-C`-Steps umgestellt (Spec-v3-Nachtrag).
  MINOR (behoben): `engines: node >=22.18` ergänzt (README-Claim maschinenlesbar).
  Nits (behoben): Friction-Inline-Nummern an F-Nummern angeglichen; ein `~`→`≈`.
  MINOR (out of scope, pre-existing): test:core-Drift, siehe A. Gehalten haben:
  Determinismus (byte-identische Doppel-Läufe, Tabelle oben = echte Zahlen),
  Margin-Gate (Mutant 0.5 → rot), Hash-Korruptions-Mutant (%4-Bucket → rot),
  @ts-expect-error-Nicht-Vakuität (beide Pins + Unused-Directive-Probe),
  kein any-Fallback via skipLibCheck (numtype entfernt → TS2307 + Kaskade),
  Cold-Install-Repro des Baustein-0-Blockers (Mitigation ist load-bearing),
  Sprach-/GFM-Gates, keine tsconfig-/Registry-/gitignore-Leaks. Transparenz-
  Beobachtung: eine schmale Klasse längenuniformer additiver Hash-Offsets ist
  für die Assertions strukturell unsichtbar (reine Rotation des Embedding-Raums,
  Scores invariant) — kein Bug, dokumentiert.
- **Baustein C (covenant-verify): Z1 hält, S1-Linkage hält, M1–M5 unberührt,
  Nicht-Ziele konform** (Wunschliste listet Transzendente nur als explizit
  gegated; sqrt sauber als nicht-transzendent begründet). **EIN Befund mittel/
  hoch zu Z2, NICHT still aufgelöst:** die v4-Ausnahme deckt Bauergebnisse des
  aktuellen Commits; test:example prüft ein eingefrorenes Registry-Artefakt eines
  vergangenen Commits → der Korpus rottet „still zwischen Releases", gedeckt nur
  durch den manuellen Bump-Prozess. → Owner-Entscheidung offen (FOLLOWUPS:
  Covenant-v5-Präzisierung / mechanischer Registry-Tripwire / beides).
- **Akzeptierte Abweichung (A-MINOR):** die Wunschlisten-Priorisierung folgt nicht
  strikt „Häufigkeit × Workaround-Kosten" (W2 vor W3 trotz geringerer Frequenz) —
  offen begründet mit Kosten/Nutzen (W2 ist der billigste Kandidat und
  mean-Enabler); dokumentiertes Ermessen, keine stille Abweichung.

Nach den Befund-Fixes wurden die betroffenen Gates erneut gefahren (test:example,
GFM über alle geänderten .md; Zahlen im Commit-Gate-Block der Session).

### Nachtrag zum Addendum (2026-07-20, nach Owner-Entscheid)

Der offene Z2-Punkt aus Baustein C ist entschieden: **Option (b), mechanischer
Registry-Tripwire** — `scripts/check-example-registry-drift.mjs` läuft in
`test:example` und im CI-Job `example` nach dem Install und failt, sobald
Registry-`latest` eine andere Major.Minor trägt als die installierte
Example-Version (Patch-Drift bewusst toleriert, SemVer-Politik). Das „stille
Rotten zwischen Releases" ist damit strukturell unmöglich; der manuelle
Release-Bump bleibt als FOLLOWUPS-Checklisten-Punkt, wird aber vom Gate
erzwungen. Nicht-Vakuität bewiesen: simulierte 0.2.0-Drift in
node_modules → Exit 1 mit Bump-Anweisung; fehlender Install → Exit 1;
Restore → Exit 0. COVENANT.md bleibt unverändert auf v4 (Option (a),
Text-Präzisierung, bewusst nicht genommen — nach (b) Kür, kein Loch).
