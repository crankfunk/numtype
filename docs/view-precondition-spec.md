# Mini-Scheibe: View-Preconditions — bindende Spec

**Version:** v2 (2026-07-25)
**Status:** Baustein 0 gelaufen, **zwei Blocker**, beide eingearbeitet → **Scope-Erweiterung
braucht Owner-Abnahme**, danach Implementierung

**Änderungslog v1 → v2** — beide Blocker waren Fehler in v1, nicht Design-Fragen:
- **B1 (Scope, blocker):** `elementwise.test.ts` fehlte. Die vier Skalar-Ops (`add`/`sub`/`mul`/
  `div`) × drei View-Klassen = **12 weitere Testkörper** haben denselben Defekt — ihr Helfer
  `assertScalarOpMatches` (Zeile 377) bildet die Referenz aus `w.toArray()`. FOLLOWUPS D7 hatte
  die Stellen 416/433 sogar ausdrücklich genannt. Schlimmer: v1s Gegenprobe schrieb
  „`sqrt`-View-Tests (elementwise) fangen beide Mutanten" — das gilt NUR für `sqrt` und erweckte
  den Eindruck, die Datei sei abgedeckt. Ohne diese Erweiterung bliebe Arbeitsregel 12 für die
  S1-Skalar-Kernel unbewiesen. **Aufgenommen.**
- **B2 (Beweislage, blocker):** die Zahlen der Kontrollgruppe waren um Faktor 2 zu hoch — der
  `topk`-Vier-View-Block hat **20** Testkörper, nicht 40, und unter Mutant T sind **12**
  relevant, nicht 24. Ursache: gezählt wurde auf einem Reporter-Output, der Fehlschläge zweimal
  druckt (inline + Summary). Die QUOTE (100 %) hält, die Absolutzahlen sind korrigiert. Aus der
  Quelle nachgezählt: 4 + 4 + (3×3) + 3 = 20.
- **M1:** das „—" in der Mutanten-Tabelle trug zwei unvereinbare Bedeutungen. Jetzt getrennt in
  „überlebt" / „nicht anwendbar", und die fehlende gemessene Zelle ergänzt.
- **M2:** D2s Untergrenze verlangte Ungleichungen (`stride !== 1`, `offset !== 0`) — die fangen
  keine Off-by-eine-Konstante. Auf **exakte** Pins verschärft.
- **L1/L2/L3/N1** eingearbeitet: Duplikat-Prüfung für D3 konkretisiert, die bedingte
  View-Konstruktion der special-value-Raster benannt, der transpositions-invariante
  `[2,2,2,2]`-Fall explizit gemacht, die Gegenprobe-Liste vervollständigt.
**Vorgeschichte:** FOLLOWUPS „Testqualität: 13 View-Test-Stellen haben ein selbstreferentielles
Orakel" (D7 aus `docs/slice-literal-budget-spec.md`) · Messung in dieser Spec, Abschnitt
„Beweislage"

## Ziel in einem Satz

Testblöcke, deren Existenzgrund eine **View-Klasse** ist (transponiert / gestridet /
offset-verschoben / komponiert), assertieren diese Klasse künftig explizit — heute behaupten
sie sie nur im Testnamen, und **304 Testfälle überleben eine Mutation, die genau diese Klasse
zerstört**. Maßgeblich ist die Tabelle in D1 (v1 sprach im Vorspann von „zwölf Blöcken in
`resident.test.ts`"; nach der B1-Erweiterung sind es 17 Einfügestellen über drei Dateien —
diese Zeile war in v2 zunächst stehen geblieben, Baustein-A-Nit).

## Warum das eine eigene Scheibe ist

Es ist ein **vorbestehender Testqualitäts-Defekt**, kein Folgefehler der letzten Scheibe. Er
berührt die Grundlage von **Arbeitsregel 12** („Residente Op-Tests müssen VIEWS treffen") — die
Regel existiert, weil diese Blöcke sie belegen sollen. Solange sie es nicht tun, ist die Regel
eine Absichtserklärung.

## Beweislage (gemessen, nicht gelesen)

Nach Arbeitsregel 15 wurde die Klassifikation **nicht** durch Lesen der Testkörper vorgenommen,
sondern über eine orthogonale Achse: die View-Primitive selbst wurden mutiert. Zwei Mutanten,
je in `spike/src/wasm/resident.ts`, je per Backup-Kopie mit `diff`-Beweis revertiert:

- **Mutant S** — `WNDArray.slice` ignoriert seine Specs (`normalizeSliceSpecs(this.shape, [])`),
  jeder Slice wird ein voller View.
- **Mutant T** — `WNDArray.transpose` ist die Identität (Shape/Strides nicht mehr umgekehrt).

„nicht anwendbar" heißt: der Block konstruiert seinen View gar nicht über dieses Primitiv, der
Mutant kann ihn also nicht treffen — das ist KEIN Freispruch, sondern eine Nicht-Messung.

| Block | Fälle | Mutant S | Mutant T |
|---|---|---|---|
| `mean` auf transpose / sliced / offset / composed | 26 | überlebt | überlebt |
| `item` auf transpose / sliced / offset / composed | 120 | überlebt | überlebt |
| `argmax` auf transpose / sliced / offset / composed | 26 | überlebt | überlebt |
| `argmax` special values (View-Variante) | 60 | nicht anwendbar (nur `transpose`) | **überlebt 0/60** |
| `topk` special values (View-Variante) | 60 | überlebt | nicht anwendbar (nur `slice`) |
| `add`/`sub`/`mul`/`div` (Skalar) auf transpose / sliced / offset — `elementwise.test.ts` | 12 | überlebt (8 Slice-basierte) | überlebt (4 transpose-basierte) |
| **`topk` Vier-View-Block — Kontrollgruppe** | **20** | **fängt 20/20** | **fängt 12/12 der anwendbaren** |

**Die Kontrollgruppe ist der tragende Teil des Beweises.** Der `topk`-Vier-View-Block ist der
einzige, der Precondition-Assertions hat, und der einzige, der beide Mutanten fängt — in beide
Richtungen, mit Quote 100 %. Damit ist sowohl der Defekt als auch seine Behebbarkeit im selben
Lauf belegt: der Fix ist nicht erfunden, er ist bereits im Repo und funktioniert nachweislich.
Unter Mutant T sind nur die transpose-benutzenden Unterblöcke (transponierte Zeile 9 +
komponiert 3 = 12) anwendbar; step-slice und offset-window (8) fallen dort korrekt nicht.

**Gegenprobe, dass es kein flächiges Abdeckungsproblem ist** (v2 korrigiert und ergänzt): Es
**fangen** beide Mutanten — jeweils weil sie ihre Referenz über eine unabhängige `NDArray`-Kette
oder einen handgerechneten Pin bilden statt über den Prüfling: die `sqrt`-View-Tests in
`elementwise.test.ts` (**nur `sqrt` — die Skalar-Ops derselben Datei tun es NICHT, siehe B1**),
`keepdims` offset/composed, `flatten materialize-routing`, die `stack`-View-Row-Tests (52 von 60
Fällen unter Mutant S gefallen) und der `topk`-NaN-Payload-Strided-View-Test. Der Defekt sitzt
genau dort, wo das Orakel aus dem View stammt — nicht flächig.

## D1 — Geltungsbereich: 304 Fälle in zwei Dateien

| Block (Header-Kommentar) | View-Konstruktion | Fälle |
|---|---|---|
| `resident.test.ts` · `mean` auf VIEWS (440–559) | transpose · step-slice · offset · composed | 26 |
| `resident.test.ts` · `item` auf VIEWS (621–722) | transpose · step-slice · offset · composed | 120 |
| `resident.test.ts` · `argmax` auf VIEWS (1332–1433) | transpose · step-slice · offset · composed | 26 |
| `resident.test.ts` · `argmax` special-value-Raster (1434 ff.) | transpose (View-Variante) | 60 |
| `resident.test.ts` · `topk` special-value-Raster (2058 ff.) | step-slice (View-Variante) | 60 |
| **`elementwise.test.ts` · Skalar-Ops (387–442)** — v2, B1 | transpose · step-slice · offset, je `add`/`sub`/`mul`/`div` | **12** |

Zusätzlich **eine Stelle mit einem ANDEREN Defekt**, siehe D3: `backend-api.test.ts:233`.

**Die `elementwise.test.ts`-Erweiterung ist eine Scope-Änderung gegenüber v1 und braucht
Owner-Abnahme.** Sie ist keine Ermessensfrage: derselbe Defekt, dieselbe Ursache
(`assertScalarOpMatches` bildet `refData` aus `w.toArray()`, Zeile 377), von FOLLOWUPS D7
ursprünglich sogar namentlich genannt, und ohne sie bleibt Arbeitsregel 12 für die
S1-Skalar-Kernel `nt_scalar_{add,sub,mul,div}_strided` unbewiesen.

**Nicht im Scope, und zwar ausdrücklich:** siehe D4.

## D2 — Der Fix: Precondition-Assertion auf die Strukturklasse

Wortgleich zum Muster, das der `topk`-Block bereits trägt und das die Kontrollgruppe als
wirksam belegt. Unmittelbar nach der View-Konstruktion, vor dem Aufruf des Assertions-Helfers:

```ts
const view = w.slice(...wideSpecs({ step: 2 }, null));
try {
  assert.deepStrictEqual([...(view.shape as readonly number[])], [2, 3],
    "precondition: the step slice is a [2,3] view");
  assert.notStrictEqual(view.describe().strides[0], 1,
    "precondition: the view must be genuinely non-contiguous (stride != 1)");
  assertMeanViewMatches(view, axis, keepdims, ctx);
```

Je View-Klasse ist die **unterscheidende** Eigenschaft zu assertieren, nicht bloß die Shape —
die Shape allein fängt Mutant T nicht überall.

**v2-Verschärfung (M2): EXAKTE Pins, keine Ungleichungen.** v1 verlangte `stride !== 1` und
`offset !== 0`. Solche Ungleichungen fangen die beiden Mutanten, aber keinen
Off-by-eine-Konstante-Fehler: ein falscher Startindex bei korrektem Step, ein falscher aber
von 0 verschiedener Offset, ein falscher aber von 1 verschiedener Stride blieben unsichtbar —
und D5s mutationsgetriebenes Kriterium würde das NICHT aufdecken, weil weder Mutant S noch T
solche Fehler erzeugt. Da die Sollwerte an jeder Stelle statisch bekannt sind, kostet der
exakte Pin nichts:

| View-Klasse | zu assertieren (exakt) |
|---|---|
| transponiert | Shape **und** der vollständige Strides-Vektor (die vertauschte Ordnung als Literal) |
| step-Slice | Shape **und** der exakte Stride-Wert der geslicten Achse |
| offset-Fenster | Shape **und** der exakte `offset`-Wert |
| komponiert | Shape **und** exakter Stride **und** exakter Offset |

**Achtung, nicht offensichtlich (L3):** das `argmax`-special-value-Raster enthält die Shape
`[2, 2, 2, 2]`, die **transpositions-invariant** ist — dort fängt eine Shape-Assertion Mutant T
grundsätzlich nicht, nur der Strides-Vektor (`[8,4,2,1]` gegen `[1,2,4,8]`) unterscheidet. Der
Fall gehört mit einem Kommentar markiert, damit ihn niemand später „vereinfacht".

**Bedingte View-Konstruktion (L2):** die beiden special-value-Raster wählen View gegen
contiguous per `rng.nextBool()`. Die Precondition gehört dort in den View-Zweig
(`if (asView) { … }`), nicht davor. Der Seed ist fix, es entsteht kein Flakiness-Risiko.

**Bindende Untergrenze:** Für jede Klasse muss die gewählte Assertions-Menge den jeweils
zutreffenden Mutanten fangen. Das ist kein Stilhinweis, sondern das Abnahmekriterium (D5) —
welche Assertion es leistet, ist Umsetzungsfreiheit, solange sie exakt pinnt.

## D3 — `backend-api.test.ts:233` ist ein anderer Defekt

Der Test heißt „Interop: `WNDArray.slice()` (view) → `toArray()` → `NDArray.fromArray(shape, …)`,
bit-identical" und baut seine Referenz als
`NDArray.fromArray(wndSlice.shape, wndSlice.toArray())` — also **aus dem Prüfling selbst**. Er
ist damit tautologisch: er kann nicht fehlschlagen, solange `fromArray`/`toArray` konsistent
sind, und überlebt Mutant S erwartungsgemäß.

Hier hilft **keine** Precondition. Der Fix ist ein **unabhängiges Orakel**: die Referenz aus der
`NDArray`-Seite bauen (`NDArray.fromArray(base).slice(gleiche Specs)`) und beide Wege
vergleichen — das Muster, das `slice.test.ts` bereits für den Differentialtest benutzt.

**v2-Präzisierung (L1): die Duplikat-Frage ist konkret zu prüfen, nicht zu unterstellen.** Der
Test ist NICHT einfach ein Duplikat von `slice.test.ts:105-130`: er geht über die Fassade
`NDArray.backend("wasm")`, `slice.test.ts` ruft `WNDArray.fromArray(core, …)` direkt. Diese
Fassaden-Abdeckung existiert allerdings **bereits** in den Nachbartests derselben Datei
(`backend-api.test.ts:195-231`, derselbe Interop-Pfad über dieselbe Fassade). Bindend: erst
prüfen, ob nach dem Umbau irgendeine Abdeckung übrig bleibt, die die Nachbartests nicht schon
leisten. Bleibt keine, ist **Löschen** die ehrliche Auflösung — einen Test umzubauen, damit er
etwas prüft, das nebenan schon geprüft wird, ist Beschäftigung, kein Gewinn. Die Entscheidung
gehört ins Ergebnis-Doc, mit dem Vergleich, auf dem sie beruht.

## D4 — Was NICHT im Scope ist, und warum das eine bewusste Entscheidung ist

Ein Stichwort-Sweep über beide Mutanten-Läufe markiert rund **1.100 weitere Testfälle**, die
beide Mutanten überleben und ein View-Stichwort im Namen tragen. **Das ist keine
Vakuitäts-Aussage.** Stichproben zeigen, dass die Mehrheit legitim unbetroffen ist: Tests in
`strided.test.ts` gehen über die rohe ABI statt über `WNDArray.slice`; `slice.test.ts` vergleicht
zwei unabhängige Implementierungen; Lifecycle-/Refcount-Tests behaupten gar keine View-Klasse,
sondern ein Freigabeverhalten.

Diese Menge zu klassifizieren ist **eigene Arbeit mit eigener Methode** und geht als separater
FOLLOWUPS-Eintrag raus, dessen erster Schritt eine **Klassifikation** ist, kein Fix. Sie hier
mitzunehmen hieße, auf einer ungeprüften Population zu handeln — genau der Fehler, den die
vorige Scheibe gerade korrigiert hat.

## D5 — Abnahmekriterium: mutations-getrieben, nicht assertions-gezählt

Die Scheibe ist genau dann fertig, wenn **beide Mutanten aus der Beweislage in jedem der zwölf
Blöcke fallen**. Konkret, und in dieser Reihenfolge zu belegen:

1. Mutant S anwenden → `pnpm test:resident` → **jeder** Slice-basierte Block hat ≥1 benannten
   Fehlschlag, in BEIDEN Dateien (`resident.test.ts` **und** `elementwise.test.ts`). Revert per
   Backup-Kopie mit `diff`-Beweis.
2. Mutant T anwenden → dasselbe für die transpose-basierten Blöcke, ebenfalls in beiden Dateien
   (inkl. der komponierten, die BEIDE Mutanten fangen müssen). Revert ebenso.
3. Ohne Mutanten: alle Suiten grün, Testzahlen unverändert.

Der Nachweis ist eine **Tabelle mit Fallzahlen je Block**, nicht die Aussage „Mutant war rot" —
ein einzelner Fehlschlag irgendwo beweist nichts über die anderen elf Blöcke.

**Gegenprobe (Pflicht):** Fallen unter Mutant S auch Blöcke, die gar keinen Slice benutzen, ist
der Mutant zu grob gewählt und der Beweis wertlos.

## Vorregistrierte Gates

- `check:diag` ≤ **+4.000** gegen 225.671 @ 140 (v1 registrierte ≤+3.000 für zwölf Blöcke; die
  B1-Erweiterung bringt eine dreizehnte Gruppe dazu, die Grenze wird **vor jeder Messung**
  angehoben und die Anhebung hier begründet — nicht nachträglich). Es werden ausschließlich
  Assertions in bestehende Testkörper eingefügt: kein neues File, keine neue Typ-Maschinerie,
  kein Klassen-Surface. Der Betrag ist eine Obergrenze mit Reserve, keine Erwartung — nach
  Arbeitsregel 16 existiert für so etwas keine Pro-Stelle-Schätzung, also wird nicht geschätzt,
  sondern gemessen und dekomponiert berichtet. Budgetdruck besteht nicht, die vorige Scheibe
  hat 11.708 freigemacht.
- `check:diag:stress` 116.053 @ 82 und `check:diag:browser` 2.142 @ 75: **Δ0** erwartet
  (`resident.test.ts` ist in keinem der beiden Korpora) — zu messen, nicht anzunehmen.
- `bench:editor` acht Pins, `check:freeze` `2a54d9fd…`, `cargo test` 222+1: **Δ0**, kein
  `spike/src`, kein Rust im Diff.
- **Testzahlen unverändert**: `test:core` 1591 · `test:resident` 6124+2 · `test:threaded` 139.
  Es werden Assertions in bestehende `test()`-Körper eingefügt, keine neuen Tests. Führt der
  D3-Fix wider Erwarten zu einer geänderten Testzahl, ist das offenzulegen und neu zu pinnen.
- `pnpm check` (drei Legs) und `graph-a-lama query lint` Exit 0.

## Berührte Covenant-Invarianten

- **S1** — nicht berührt, es wird ausschließlich in `spike/tests-runtime/` editiert; mechanisch
  durch das Lint gedeckt.
- **M1** — nicht berührt (kein Kernel, kein Verhalten). Aber inhaltlich **gestärkt**: M1s
  Bit-Identitäts-Anspruch für residente Ops stützt sich auf genau diese View-Tests.
- **M3/M4** — nicht berührt, keine Typ-Fläche, kein Rust.

Eskalationsstufe: **Stufe 3** vorgeschlagen (Baustein 0 gegen die Spec, dann A + B). Anders als
bei der vorigen Scheibe ist Baustein C weiterhin entbehrlich — es ändert sich weder Vertragstext
noch eine Invarianten-Fläche. Owner entscheidet.

## Disziplin-Entscheidung

Edits an Bestandstests, dieselbe Lage wie in der vorigen Scheibe: die Append-/Insertion-only-
Disziplin ist ein Artefakt-Byte-Argument für `crates/core` bzw. TS-Klassenkörper und greift für
`*.test.ts` nicht. Anders als dort sind die Änderungen hier **rein additiv** (Assertions kommen
hinzu, kein bestehender Testkörper wird umformuliert) — mit Ausnahme von D3, wo ein Orakel
ersetzt wird. Diese eine Ersetzung ist die einzige nicht-additive Änderung der Scheibe und
gehört im Ergebnis-Doc benannt.

## Doku-Pflichten am Ende

`docs/view-precondition-ergebnisse.md` mit Post-Verification-Addendum · CLAUDE.md („Aktuelle
Pins & Gates" — neuer `check:diag`-Wert; **Arbeitsregel 12 präzisieren**: sie verlangt bisher
„Tests müssen Views treffen", muss künftig „…und die View-Klasse explizit assertieren" sagen) ·
FOLLOWUPS (D7-Eintrag schließen, den D4-Klassifikations-Eintrag öffnen) · `docs/projekt-log.md`
· KB-Capture. README nicht betroffen (es ändert sich nichts daran, wo eine Op läuft) —
Prüfkommando trotzdem laufen lassen.

## Ein-Block-Revert

Rein additive Assertions in zwei Dateien plus ein ersetztes (oder gelöschtes) Orakel in einer
dritten. Rückbau = `git revert` des einen Commits; der einzige zurückwandernde Pin ist
`check:diag`.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-25)

EIN `brainroute:deep`-Agent, frischer Kontext, gegen v1 dieser Spec und den echten Code;
Auftrag aus `docs/verify-runde-template.md` „Baustein 0". Er hat beide Mutanten selbst
nachgebaut und über die Spec-Vorgabe hinaus auch gegen `elementwise.test.ts` gefahren.

**Verdikt: zwei BLOCKER, beide bestätigt und beide Fehler in v1 — kein Design-Fehler, sondern
zwei falsche Behauptungen.**

- **B1** hat den Geltungsbereich um `elementwise.test.ts` erweitert. Der Auslöser war eine
  Formulierung in v1s Gegenprobe, die aus einem korrekten Teilbefund („`sqrt` fängt") einen
  falschen Gesamteindruck machte („elementwise ist abgedeckt"). **Selbst nachgeprüft:**
  `assertScalarOpMatches` (elementwise.test.ts:377) bildet `refData` aus `w.toArray()` — dasselbe
  Muster. Erschwerend: das Signal lag bereits in meiner eigenen Messung (`n=6, gefallen=2` für
  „sliced view"), wurde als Mischwert bemerkt und **nicht verfolgt**.
- **B2** hat die Kontrollgruppen-Zahlen halbiert (20 statt 40 Testkörper, 12 statt 24 unter
  Mutant T). **Selbst nachgeprüft** durch Nachzählen der Schleifengrenzen im Quelltext:
  4 + 4 + (3×3) + 3 = 20. Ursache war ein Reporter-Output, der Fehlschläge zweimal druckt und
  ungefiltert gezählt wurde — was die ÜBERLEBENDEN Blöcke nicht verfälscht (dort ist die Zahl
  0 und bleibt 0), wohl aber jede Zeile, in der etwas fällt. Die 292 aus D1 hat der Verifier
  unabhängig aus Schleifengrenzen UND Testlauf bestätigt.

**Was das über die Methode sagt:** Beide Blocker betrafen nicht das Design, sondern die
BEWEISFÜHRUNG — eine zu weit verallgemeinerte Gegenprobe und eine falsch gezählte
Kontrollgruppe. Genau die zwei Stellen also, an denen eine Spec am überzeugendsten wirkt.
Dieselbe Klasse wie der Befund, der diese ganze Kette ausgelöst hat.

**Ebenfalls eingearbeitet:** M1 (Tabellensymbol „—" war zweideutig), M2 (Ungleichungen →
exakte Pins), L1 (Duplikat-Prüfung für D3 konkretisiert, Löschen ausdrücklich zugelassen),
L2 (bedingte View-Konstruktion im Zufalls-Raster), L3 (`[2,2,2,2]` ist
transpositions-invariant), N1 (Gegenprobe-Liste vervollständigt). M3 erledigt sich mit B1.

**Geprüft und haltend:** D1s 292 Fälle, D3s Code-Annahme, D4s Stichprobe (je ein Beispiel aus
`strided.test.ts`, `slice.test.ts`, `resident-lifecycle.test.ts` — kein vakuöser Fall
darunter), die Baseline 225.671 @ 140 und die Testzahl 6124+2, sowie D5s Pflicht-Gegenprobe
(Mutant S trifft keine Blöcke ohne Slice-Nutzung, ist also nicht zu grob).
