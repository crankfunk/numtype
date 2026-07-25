# View-Preconditions — Ergebnisse

**Datum:** 2026-07-25/26 · **Basis:** `4f46964` · **Spec:** `docs/view-precondition-spec.md` v2
**Anlass:** D7 der `slice`-Literal-Budget-Scheibe — ein vorbestehender Testqualitäts-Defekt.

**Ehrlichkeitsregel dieses Dokuments:** jede Zahl stammt aus einem selbst ausgeführten Lauf mit
geprüftem Exit-Code. Wo eine Messung durch die unten dokumentierte Kontamination fragwürdig
wurde, ist sie verworfen und neu erhoben, nicht nachträglich gerettet.

## Ergebnis in drei Sätzen

Dreizehn Testblöcke behaupteten in ihrem Namen, eine residente Operation auf einer bestimmten
View-Klasse zu prüfen; **304 Testfälle überlebten eine Mutation, die genau diese Klasse
zerstört**. Sie tragen jetzt exakte Strukturklassen-Pins (Shape, vollständiger Strides-Vektor,
Offset) direkt nach der View-Konstruktion, und **jeder Block fängt beide Mutanten**. Der
teuerste Ertrag der Runde war aber ein Prozess-Befund: zwei parallel laufende Verifier haben
sich gegenseitig die Messungen verfälscht, weil ich sie beide auf denselben Working Tree
gelassen habe.

## Der Defekt, gemessen statt gelesen

Nach Arbeitsregel 15 wurde nicht durch Lesen der Testkörper klassifiziert, sondern über eine
orthogonale Achse: die View-Primitive selbst wurden mutiert.

- **Mutant S** — `WNDArray.slice` ignoriert seine Specs (`normalizeSliceSpecs(this.shape, [])`).
- **Mutant T** — `WNDArray.transpose` ist die Identität.

Vorher überlebten alle betroffenen Blöcke beide Mutanten. Ursache in jedem Fall dieselbe: die
Assertions-Helfer (`assertMeanViewMatches`, `assertItemMatches`, `assertArgmaxMatches`,
`assertTopkMatches`, `assertScalarOpMatches`) bilden ihre Referenz aus `view.toArray()` — dem
bereits konstruierten Prüfling. Ein verfälschter Aufbau erzeugt einen anderen View, aber
Referenz und Kandidat rechnen beide konsistent darüber weiter.

**Die Kontrollgruppe war der tragende Teil des Beweises:** der `topk`-Vier-View-Block ist der
einzige, der schon vorher Precondition-Assertions hatte, und der einzige, der beide Mutanten
fing. Der Fix war also nie erfunden — er lag im Repo und funktionierte nachweislich.

## Was Baustein 0 vor der ersten Codezeile korrigierte

Zwei **Blocker**, beide Fehler in meiner v1-Spec, keiner ein Design-Fehler:

1. **Der Geltungsbereich war zu klein.** `elementwise.test.ts` fehlte — die vier Skalar-Ops ×
   drei View-Klassen = 12 weitere Testkörper mit demselben Defekt, von FOLLOWUPS D7 sogar
   namentlich genannt. Auslöser des Fehlers war eine Formulierung in v1, die aus einem
   korrekten Teilbefund („`sqrt` fängt") einen falschen Gesamteindruck machte („elementwise ist
   abgedeckt"). Erschwerend: das Signal lag bereits in meiner eigenen Messung (`n=6,
   gefallen=2`), wurde als Mischwert bemerkt und **nicht verfolgt**.
2. **Die Kontrollgruppen-Zahlen waren um Faktor 2 zu hoch.** 20 Testkörper, nicht 40; unter
   Mutant T sind 12 anwendbar, nicht 24. Ursache: der Test-Reporter druckt Fehlschläge zweimal
   (inline + Summary), und `grep -c` auf den Rohoutput zählt sie doppelt. **Live bestätigt:**
   `topk special values` zeigt roh n=85 = 35 Passes (einmal) + 25 Fehlschläge (zweimal).

Beide betrafen nicht das Design, sondern die **Beweisführung** — also genau die Stellen, an
denen eine Spec am überzeugendsten wirkt.

## Der Fix

17 Einfügestellen über drei Dateien, rein additiv. Direkt nach der View-Konstruktion:

```ts
assert.deepStrictEqual([...(view.shape as readonly number[])], [2, 3], "…: precondition — view shape");
assert.deepStrictEqual([...view.describe().strides], [6, 1], "…: precondition — EXACT strides");
assert.strictEqual(view.describe().offset, 0, "…: precondition — EXACT offset");
```

**Exakte Pins, keine Ungleichungen** (Baustein-0-Befund M2): `stride !== 1` fängt keinen
Off-by-eine-Konstante-Fehler. Da die Sollwerte statisch bekannt sind, kostet der exakte Pin
nichts. Alle 17 Sollwerte stimmten im ersten echten Testlauf.

**Kein geteilter Assertions-Helfer.** Bewusst inline, obwohl DRY dagegen spricht: die
Vorgänger-Scheibe musste gerade erst reparieren, dass ein geteilter Helfer eine
Single-Point-of-Failure-Fläche erzeugt, gegen die die ihn benutzenden Tests blind sind. Der
einzige neue Helfer ist `naturalStrides` — und der ist selbstprüfend, weil eine falsche
Implementierung sofort im unmutierten Lauf auffiele.

**`[2,2,2,2]` ist der nicht offensichtliche Fall:** transpositions-invariant, dort trägt allein
der Strides-Vektor. Im Code kommentiert. Baustein B hat empirisch belegt, dass der Fall
erreichbar ist (dreimal pro Lauf unter dem festen Seed) — die Absicherung ist nicht tot.

## D3: der tautologische Interop-Test

`backend-api.test.ts` baute seine Referenz als `NDArray.fromArray(wndSlice.shape,
wndSlice.toArray())` und verglich sie gegen sich selbst. **Entscheidung gegen die Spec-Option
„löschen", mit Begründung:** der Test ist die einzige Stelle, die den Fassaden-Pfad
`NDArray.backend("wasm")` → `slice` erreicht (die Nachbarn decken nur `fromArray` und
`transpose`). Er bekommt ein unabhängiges `NDArray`-Orakel plus eigene Preconditions, und im
Kommentar steht jetzt, dass sein Wert die Fassaden-Erreichbarkeit ist, nicht das Differential.
**Offengelegter Rest** (Baustein-B-Befund): beide Seiten gehen weiterhin durch dasselbe
`normalizeSliceSpecs` — ein vorbestehender, anderswo dokumentierter Blind Spot. Im Kommentar
benannt statt verschwiegen.

## Abnahmekriterium (D5): erfüllt, entdupliziert gezählt, in sauberem Zustand neu erhoben

| Block | Körper | Mutant S | Mutant T | Mutant O (Offset ±1, in-bounds) |
|---|---|---|---|---|
| `mean` transpose / sliced / offset / composed | 6/6/6/8 | –/6/6/8 | 6/–/–/8 | –/–/6/8 |
| `item` transpose / sliced / offset / composed | 30/30/30/30 | –/30/30/30 | 30/–/–/30 | –/–/30/30 |
| `argmax` transpose / sliced / offset / composed | 6/6/6/8 | –/6/6/8 | 6/–/–/8 | –/–/6/8 |
| `argmax` special values (View-Zweig) | 60 | – | 18/18 | – |
| `topk` special values (View-Zweig) | 60 | 25/25 | – | – |
| Skalar-Ops transpose / sliced / offset | 4/4/4 | –/4/4 | 4/–/– | –/–/4 |
| Interop-Fassade (D3) | 1 | 1/1 | – | 1/1 |

„–" heißt: der Mutant kann diesen Block strukturell nicht treffen (kein `slice` bzw. kein
`transpose` bzw. Offset 0). **Pflicht-Gegenprobe erfüllt:** kein Mutant trifft einen Block, den
er nicht treffen können sollte — keiner ist zu grob gewählt.

**Mutant O ist der wichtigste der drei** und stammt nicht aus der Spec. Baustein B hatte
angemerkt, dass sein eigener Off-by-one-Mutant in 11 Blöcken nur deshalb gefangen wurde, weil
er den Lesezugriff aus dem WASM-Puffer schob — `.toArray()` warf, bevor irgendeine Assertion
lief. Ob eine **kleine, in-bounds** Korruption still absorbiert würde, blieb damit offen.
Mutant O (`offset += Math.max(0, spec.start * stride - 1)`) ist garantiert in-bounds und trifft
nur Views mit Offset ≠ 0. Ergebnis: **93 Fehlschläge nennen namentlich
`precondition — EXACT offset`** — der Pin leistet die Arbeit selbst, nicht ein nachgelagerter
Absturz. Damit ist die Lücke geschlossen, die B offen lassen musste.

## Gates

| Gate | Wert | gegen |
|---|---|---|
| `check:diag` | **225.983 @ 140** | 225.671 = **Δ+312** gegen ≤+4.000 |
| `check:diag:stress` · `:browser` | 116.053 @ 82 · 2.142 @ 75 | Δ0 |
| `bench:editor` | 8 Pins exakt, Hard CI gate PASS | Δ0 |
| `check:freeze` · `cargo test` | `2a54d9fd…` · 222 + 1 | unverändert |
| `test:core` · `:resident` · `:threaded` | 1591 · 6124+2 · 139, je 0 fail | zahlengleich |
| `pnpm check` (3 Legs) · `graph-a-lama query lint` | Exit 0 | — |

**Dekomposition der +312:** die 17 Precondition-Blöcke kosten **+4** (sie benutzen
ausschließlich bereits instanziierte Typen), die restlichen **+308** stammen aus dem D3-Umbau
mit seiner zusätzlichen `NDArray`-Referenzkette. Baustein B hat die Zerlegung unabhängig in
zwei isolierten Worktrees reproduziert.

## Der Prozess-Befund, der die Runde teurer machte als die Scheibe

**Ich habe Baustein A und B parallel dispatcht und beiden erlaubt, Mutanten im HAUPT-Working-Tree
anzuwenden.** Beide taten das gleichzeitig. B bemerkte mitten im Lauf ein fremdes `// MUTANT S`
in `resident.ts`, später ein fremdes `offset += 0 * stride`; sein erster Messlauf kombinierte
zwei unabhängige Mutationen und lieferte ein Ergebnis, das er zunächst fehlinterpretierte, bis
er `git diff` prüfte. Er verwarf die Messung und baute sie in eigenen isolierten Worktrees neu
auf.

**Die Rückwirkung auf A war größer.** A hatte eine unerklärte Diskrepanz (zwei identische Läufe
unter demselben Mutanten ergaben 669 und 654 Fehlschläge, Differenz genau ein Block, der von
diesem Mutanten strukturell gar nicht betroffen sein kann) als „nicht-deterministische
Test-Harness-Kontamination" verbucht — also als vorbestehenden Infrastruktur-Mangel. Auf
Nachfrage zog A diese Deutung zurück: der fremde Mutant ist die weit wahrscheinlichere
Erklärung. **Ohne die Aufdeckung wäre ein Phantom-Befund über die Test-Infrastruktur ins
Projekt gewandert.**

Das Bittere daran: die KB trägt zu genau dieser Falle bereits eine Notiz
(`parallele-mutierende-verifier-worktree-patch`), die in der Vakuitäts-Notiz sogar verlinkt
ist. Ich habe sie vor dem Dispatch nicht konsultiert.

**Auflösung:** Alle entscheidenden Zahlen wurden nach Abschluss beider Agenten in einem
Zustand neu erhoben, in dem nur ich schreibe — inklusive einer Vorbedingungsprüfung
(`git diff -- spike/src/` muss leer sein), bevor überhaupt ein Mutant angewandt wird. Alle
Werte oben stammen aus dieser sauberen Runde.

**Zweiter Prozess-Befund, kleiner aber derselben Klasse:** Baustein A beendete seinen ersten
Turn auf dem Cleanup-Beweis statt auf dem Report — obwohl der Auftrag wörtlich „Der Report muss
deine ALLERLETZTE Nachricht sein" enthielt. Der Report war per `SendMessage` nachzufordern.
Arbeitsregel 9 beschreibt genau das; die Regel im Prompt zu haben, reicht nachweislich nicht.

## Offengelegte Grenzen

- Der `naturalStrides`-Mutant wird ausgerechnet auf `[2,2,2,2]` absorbiert (bei lauter gleichen
  Dimensionen liefert eine Index-Verwechslung dasselbe Ergebnis). Die Suite bleibt trotzdem rot
  (15 der 18 Aufrufe fangen ihn), aber der Fall, den die Spec als tragend benennt, prüft den
  Helfer nicht unabhängig gegen genau diese Bug-Klasse. Baustein-B-Nit, akzeptiert.
- Die rund 1.100 unklassifizierten Testfälle aus D4 bleiben **unklassifiziert** — ausdrücklich
  keine Vakuitäts-Aussage. Als eigener FOLLOWUPS-Eintrag geöffnet, dessen erster Schritt eine
  Klassifikation mit anderer Methode ist.
- Mutant O trifft nur Views mit Offset ≠ 0. Eine kleine Korruption der **Strides** (statt des
  Offsets) wurde nicht separat mutiert; die Strides-Pins sind exakt, also fängt die Assertion
  sie per Konstruktion — bewiesen ist es für den Offset, nicht für die Strides.

## Was diese Scheibe gelehrt hat

1. **Ein Testname ist kein Beleg.** Dreizehn Blöcke behaupteten jahrelang eine Coverage, die
   sie nicht hatten, und keine Verify-Runde fand es — weil alle Mutanten die OPERATION trafen,
   nie den AUFBAU. Wer einen Block schreibt, dessen Existenzgrund eine Eigenschaft des Aufbaus
   ist, muss diese Eigenschaft assertieren oder per Mutant im Aufbau belegen.
2. **Parallelität braucht Isolation, nicht Disziplin.** Beide Verifier hielten sich an die
   Regel „Mutant nur als sofort revertierter Edit mit Backup-Beweis". Die Regel ist für einen
   Schreiber richtig und für zwei wertlos: ein Backup beweist „ich habe auf meinen Schnappschuss
   zurückgesetzt", nicht „mein Schnappschuss war unverfälscht".
3. **Der zweite Blocker ist der lehrreichere.** Ein Zählfehler im Reporter-Output hat eine
   Kontrollgruppe um Faktor 2 verfälscht — in derselben Tabelle, die die Spec selbst „der
   tragende Teil des Beweises" nannte. Beweisführungen verdienen dieselbe Prüfung wie Designs.
