# Op-Scheibe W3: `sqrt` — Ergebnisse

Status: **abgeschlossen** (Eskalationsleiter Stufe 3, Verify-Runde durch den
Orchestrator; Zahlen hier sind die im Haupt-Tree tatsächlich gemessenen, kein
Vorab-Stand). Spec: docs/op-w3-sqrt-spec.md (Version 1 + Baustein-0-Addendum).
Datum: 2026-07-21.

## Summary

Die dritte Op-Scheibe aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md
W3/F1 — zweifacher Bruch der natürlichen L2-Normalisierungs-Kette in der RAG-Demo) ist
umgesetzt: `NDArray.sqrt(): NDArray<S>`, shape-erhaltend bei jedem Rang, niladisch (kein
Guard). D1: **NDArray-only, kein WASM-Kernel** — dieselbe bewusste, COVENANT-v5-gedeckte
Surface-Asymmetrie wie W1/W2. Alle Gates grün, Hash byte-identisch (keine Rust-Änderung),
Absolut-Gate (Haupt-Pin ≤ +3,000) mit deutlichem Spielraum (+2,073). Die F1-Schließung ist
BEIDES bewiesen: der Teil-Ketten-Vergleich (`mul→sum(1)→sqrt`) UND die volle
L2-Normalisierung (`mul→sum(1)→sqrt→reshape→div`) sind byte-identisch zur alten
Hand-Loop-Formulierung aus `examples/rag-demo/main.ts`.

## Umgesetzte Form je D-Punkt

- **D1 (Scope):** Nur `sqrt(): NDArray<S>` auf `NDArray`, jede Rangstufe inkl. Rang 0.
  Kein abs/neg/`.map(fn)` (Nicht-Ziele, keine Demo-Evidenz).
- **D2 (Runtime):** `sqrtRuntime(data: Float64Array): Float64Array`, APPENDED in
  runtime.ts (strikt nach `meanRuntime`) — elementweise `Math.sqrt(data[i])` aufsteigend,
  frisches Float64Array. Doc-Kommentar begründet die IEEE-Ausnahme vom
  Transzendenten-Nicht-Ziel primärquellen-verankert (ECMA-262 `sec-math.sqrt`: exakte
  𝔽-Rundung wie `+`/`-`/`*`/`/`, im Gegensatz zu "implementation-approximated" bei jeder
  transzendenten `Math.*`-Methode).
- **D3 (Methode):** `sqrt(): NDArray<S>` als Klassenkörper-Append nach `mean` (letzte
  Methode vor der schließenden Klammer) — reine Insertion, kein Bestandsmember editiert.
  Doc-Kommentar deckt IEEE-Begründung, `norm()`-Verweis (niladisch = kein Guard),
  NaN-für-negativ + `sqrt(-0) === -0`, Surface-Hinweis (kein WASM-Kernel) ab.
- **D4 (Tests):** W3-Block APPENDED ans Ende von `spike/tests-runtime/scalar-mean.test.ts`
  (kein neues File — Dateikopf-Kommentar um einen Hinweissatz ergänzt): D2-Kanten-Matrix
  (-0/NaN/negativ→NaN/±Infinity/Subnormal-Bits via `DataView`/size-0), Rang 0/1/2 +
  Shape-Erhalt, transponierte/gesliceste Empfänger, 220 randomisierte Bit-Differential-Fälle
  gegen eine direkte `Math.sqrt`-Schleife (≥200 aus der Spec erfüllt), zwei F1-Schließungs-
  Rückproben (Teilkette + volle L2-Normalisierung), size-0, Smoke-Test.
- **D5 (Typ-Pins):** 5 `Expect<Equal<...>>`-Pins (Literal `[2,3]`, Rang 0 `[]`, wide
  `readonly number[]`, Readonly-S `readonly [4,5]`, dynamischer Rang `NDArray<Shape>`)
  + 1 `@ts-expect-error` (niladisch — `sqrt(1)` ist ein Compile-Fehler) in
  `spike/tests/ndarray.test-d.ts`, APPENDED nach dem `mean`-Block.

## F1-Schließungs-Beweis (beide Formen, byte-identisch)

**Teilkette** (`m.mul(m).sum(1).sqrt()` vs. Hand-Loop `Math.sqrt` über `.data` +
`fromArray`, shape `[6,5]`, randomisierte Daten): `assertShapeEqual` UND
`assertDataBitIdentical` — **PASS**.

**Volle L2-Normalisierung** (`m.div(m.mul(m).sum(1).sqrt().reshape([N,1]))` vs. der exakten
`examples/rag-demo/main.ts`-Hand-Loop-Struktur — `mul→sum(1)→Hand-Loop-`Math.sqrt`→
`fromArray`→`reshape([N,1])`→`div`, shape `[5,4]`, strikt positive randomisierte Daten):
`assertShapeEqual` UND `assertDataBitIdentical` — **PASS**.

Beide Tests sind Teil der 710-Test-Laufs von `scalar-mean.test.ts` (siehe Gate-Tabelle) und
liefen grün im selben Testlauf wie der komplette W3-Block. Damit ist die in der Spec
geforderte "letzte fehlende Schritt der Kette"-Behauptung nicht nur begründet, sondern an
zwei unabhängigen, realistischen Formulierungen bewiesen.

## Pin-Protokoll (D5/D8, 2× deterministisch je Messpunkt)

Baseline (frischer `git worktree` von HEAD `672ec2f`, `pnpm install --frozen-lockfile`,
zweimal gemessen) **exakt reproduziert**: `188,563 @ 137` (Haupt) — deckungsgleich mit dem
CLAUDE.md-Pin.

| Messpunkt | Instantiations | Δ zur Baseline |
|---|---|---|
| Baseline (frischer Worktree, 2×) | 188,563 | — |
| Final (D2/D3/D4/D5 komplett, Haupt-Tree, 2×) | **190,636** | **+2,073** |

Absolut-Gate (≤ +3,000) eingehalten. **stress**: `104,900 @ 82` — exakt Δ 0 gegen die
Baseline (2× deterministisch). **browser**: `2,142 @ 75` — exakt Δ 0 (2× deterministisch).
Kein neues File (D4/D5 sind reine Appends an bestehende, bereits registrierte Dateien), also
kein empty-then-fill-Schritt nötig — der Delta ist reiner Inhaltskosten-Beitrag von D2+D3+D4+D5
zusammen, nicht weiter zerlegt (kleinste Scheibe, keine Order-Noise-Komponente durch ein
neues File).

## Gates (alle frisch gemessen, Haupt-Tree, dieser Stand)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (3-Verbund: root + stress + browser) | 0 Fehler | 0 |
| `check:diag` | 190,636 @ 137 (2× deterministisch) | 0 |
| `check:diag:stress` | 104,900 @ 82 (2× deterministisch, Δ 0) | 0 |
| `check:diag:browser` | 2,142 @ 75 (2× deterministisch, Δ 0) | 0 |
| `test:core` | **1,562/1,562** (Baseline 1,335 + 227 neu) | 0 |
| `test:resident` | 4,278 pass + 2 skip (unverändert) | 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161 passed (unverändert, kein Rust berührt) | 0 |
| `check:freeze` | Hash byte-identisch `0b9df4f1…2519c7d` | 0 |
| `bench:editor` | Hard-CI-Gate PASS, keine Pin-Abweichung (Instantiation-Pins unverändert getroffen) | 0 |
| `graph-a-lama query lint` | 0 Findings, 0 Errors/Warnings | 0 |
| `pnpm test:example` | PASS, numtype@0.1.1 (Registry-Major.Minor-Match), Demo-Assertions grün | 0 |

`bench:editor`s Instantiation-Pins (`spike/bench-dx/editor-latency.ts`, `INSTANTIATION_PINS`)
trafen exakt — kein Update nötig, da `sqrt()` in keinem der sieben LSP-Workload-Fixtures
aufgerufen wird und die Klassen-Member-Fläche für W3 minimal ist (ein einzelner
shape-erhaltender Member, gegenüber W2s vier Overloads + einer Reduktionsmethode).

## Tests im Detail

**Runtime (`spike/tests-runtime/scalar-mean.test.ts`, W3-Block, 227 neue Tests):** kein
WASM-Gegenpart (D1, wie W1/W2) — Coverage: 2 gruppierte Tests für die D2-IEEE-Kanten-Matrix
(-0/NaN/negativ/±Infinity/size-0, plus eine gesonderte Subnormal-Bit-Vergleichsprobe via
`DataView.getBigUint64`); 2 Rang-0/1/2-Shape-Erhalt- und Transpose-/Slice-Empfänger-Tests;
220 randomisierte Bit-Differential-Fälle gegen eine direkte `Math.sqrt`-Schleife
(`assertDataBitIdentical`, Ränge 0-4, Spezialwert-Injektion via `genDataSpecial`); 2
F1-Schließungs-Rückproben (Teilkette + volle L2-Normalisierung, s.o.); 1 Smoke-Test
(`sqrt(2) === Math.SQRT2` bei Rang 0).

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, +6 Pins — 5 benannte `Expect<Equal<...>>` + 1
`@ts-expect-error`):** Shape-Erhalt exakt bei Literal `[2,3]`, Rang 0 `[]`, wide
`readonly number[]`, Readonly-S `readonly [4,5]`, dynamischer Rang `NDArray<Shape>` (degradiert
korrekt zu `Readonly<Shape>`); `sqrt(1)` als niladischer Compile-Fehler (`@ts-expect-error`,
nicht-vakuös — ohne die Direktive würde `pnpm check` selbst fehlschlagen, da `sqrt` keinen
Parameter akzeptiert).

## Abweichungen

Keine. Alle D-Punkte der Spec wurden wie geschrieben umgesetzt; kein Owner-Eskalationsbedarf.

## Post-Verification-Addendum

_(vom Orchestrator nach der Verify-Runde zu ergänzen.)_

## Post-Verification-Addendum (2026-07-21)

Verify-Runde Stufe 3 (kompakt skaliert), A+B+C parallel + covenant-check/lint (grün).
Baustein 0 hatte vorab beide tragenden Behauptungen mit harter Evidenz versehen
(ECMA-262-Primärquelle; Typ-Delta gemessen).

- **Baustein A: CONFIRMED, keine materiellen Befunde.** Alle D/T-Punkte am Diff;
  alle Zahlen exakt reproduziert; Runtime-Mutant (Math.sqrt→Math.abs) färbt 219/227
  Tests rot (die 8 grünen = genau die abs≡sqrt-Fälle), Typ-Mutant reißt pnpm check —
  beide per Backup-Kopie-Verfahren restauriert (W2-Lektion als harte Regel etabliert).
- **Baustein B: HÄLT-mit-Befunden — zwei niedrige Coverage-Lücken, IN-SLICE
  geschlossen:** F1 (keine explizite Buffer-Identitäts-Assertion — Aliasing-Mutant
  wurde nur indirekt von 2/227 Tests gefangen) → dedizierter Isolations-Test ergänzt;
  F2 (größter Subnormal 0x000fffffffffffff ungepinnt) → Bit-Pin ergänzt; beide grün.
  F3 (informativ, kein Widerspruch): V8 erhält quiet-NaN-Payloads durch Math.sqrt,
  kanonisiert aber signaling-artige Bitmuster — die Spec verspricht nur NaN→NaN,
  Payload-Determinismus ist bewusst KEINE sqrt-Zusage (relevant erst, falls je ein
  Kernel Payload-Parität behaupten wollte). Absteigende-Reihenfolge-Mutant beweisbar
  äquivalent (rein elementweise, kein Akkumulator) — dokumentiert, kein Bug.
  Typ-Kanten hielten: Mixed-Rank-Union bleibt exakt erhalten, NDArrayView hat kein
  sqrt (nicht-vakuös), never-Durchreichung korrekt.
- **Baustein C: NULL Befunde — sauberstes Verdikt der Serie.** Eigenständiges
  Doppel-Urteil zum Transzendenten-Non-Goal: (1) sqrt ist ALGEBRAISCH (Nullstelle von
  y²−x — das Gegenbeispiel zu „transzendent"); (2) die Bit-Paritäts-Sorge des
  Non-Goals existiert für sqrt strukturell nicht (IEEE-754-Pflichtrundung für
  +−×÷√; ECMA-262 nutzt exakte 𝔽-Rundung nur für sqrt, alle transzendenten
  Math-Methoden sind „implementation-approximated"). Klarstellung nebenbei: der
  Covenant-TEXT nennt weder exp/sin noch libm — das war FOLLOWUPS-Paraphrase.
  M1-v5-Bedingung erfüllt (FOLLOWUPS-Nachtrag W3 vorhanden).
- **Finale Zahlen (nach F1/F2-Schließung, je 2×):** check:diag **190,640 @ 137**
  (+2,077 gesamt, Gate ≤ +3,000) · stress 104,900 @ 82 (Δ0) · browser 2,142 @ 75 ·
  test:core **1,564** · Hash byte-identisch · bench:editor PASS (Pins unverändert).
