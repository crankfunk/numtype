# Klassifikation der View-Test-Restmenge — bindende Spec

**Version:** v2 (2026-07-29)
**GEPARKT 2026-09-23** (Owner-Entscheidung): unvollständig als WIP committet; Selbsttest 12 rot — siehe FOLLOWUPS.
**Status:** Baustein 0 gelaufen, **zwei Blocker**, beide eingearbeitet · Owner-Entscheidungen
zu Eskalation und Artefakt-Ort liegen vor → **implementierungsreif**
**Vorgeschichte:** D4 der View-Precondition-Scheibe (`docs/view-precondition-spec.md` v2,
`-ergebnisse.md` „Offengelegte Grenzen") · FOLLOWUPS-Eintrag „Klassifikation der
unklassifizierten View-Test-Restmenge" (MITTEL)

**Änderungslog v1 → v2** — beide Blocker waren **Fehler in v1s als „empirisch belegt"
ausgewiesenen Fakten**, kein Design-Fehler; dieselbe Klasse wie die zwei Blocker, die
Baustein 0 in der Vorgänger-Spec fand:

- **B1 (Achse B2, blocker):** v1s Fakt 4 verwechselte **Schreibbarkeit** der Struktur-Felder
  (stimmt, und trägt Achse B1) mit **Lese-Beobachtbarkeit** (ein anderer Mechanismus, den
  Achse B2 braucht). `shape` ist eine gewöhnliche Daten-Property, und die Bibliothek liest
  sie intern ständig: Baustein 0 hat per Accessor gezählt, dass ein einzelnes `toArray()`
  **4** interne `.shape`-Lesevorgänge auslöst. Ein naives `readback` hätte damit für JEDEN
  View gefeuert, auf dem irgendeine Op läuft — das Widerspruchs-Register wäre Rauschen
  gewesen, und zwar überzeugend aussehendes. **Fix:** `readback` triggert nur auf den
  echten Prototyp-Methoden `toArray()`/`describe()`, **und** nur wenn der unmittelbare
  Aufrufer-Frame eine `*.test.ts`-Datei ist (die Stack-Frame-Zuordnung ist in D1 Fakt 2
  ohnehin belegt). Die Verengung ist offenzulegen.
- **B2 (Entscheidungsregel, blocker):** `nontrivial` war **über-inklusiv an Achsen der
  Länge 1**. Ein transponierter `[1,5]`-View hat Strides `[1,5]` gegen natural `[1,1]` — die
  Formel sagt „non-trivial", aber Baustein 0 hat durch Aufzählen aller gültigen Indextupel
  belegt: **0 von 5 Linearadressen unterscheiden sich**. Der Stride einer Achse der Länge 1
  kann nie zu einer Adresse beitragen. Folge: solche Fälle wären als **K4
  („struktur-blind")** ausgewiesen worden — also als Testqualitäts-Defekt, obwohl dort
  überhaupt keine Korruption bemerkbar wäre. Das hätte den einzigen Teil der Ausgabe
  verfälscht, um den es geht. **Fix:** `nontrivial` ist jetzt semantisch definiert (die
  Index→Linear-Abbildung weicht ab) mit einer bewiesenen Geschlossenen Form; neuer
  Pflicht-Selbsttest gleicht die Form gegen Brute-Force ab.
- **M1 eingearbeitet:** die Preload-Emulation von Mutant S/T ist jetzt **exakt** angegeben,
  damit die Äquivalenz zu den Quell-Mutanten der Vorgänger-Scheibe falsifizierbar statt
  behauptet ist.
- **M2 eingearbeitet:** v1s Behauptung, `test:browser` sei view-frei, ist **falsch**
  (`spike/tests-browser/smoke.test.ts:213-229` ruft `.transpose()`/`.slice(1)`), und v1s
  vorgeschlagene Prüfmethode wäre **vakuös** gewesen — jene Aufrufe laufen per
  `page.evaluate` in Chromium, wo ein Node-Preload strukturell nicht hinreicht. Jetzt als
  Ausschluss **per Konstruktion** formuliert und als offengelegte Grenze geführt.
- **Neu aufgenommen (Baustein-0-Nebenbefund, reproduziert):** `NODE_OPTIONS` **leakt** in
  verschachtelte Node-Subprozesse (die real-tsc-Diagnose-Pin-Tests rufen `tsc` auf) und, wenn
  exportiert, in unbeteiligte Prozesse derselben Shell. Als Disziplinpunkt in D6.
- **Populations-Korrektur (eigener Befund vor der Messung):** übersprungene Tests feuern
  **kein** `beforeEach` — und einer der zwei übersprungenen Fälle ist ein View-Fall
  (`resident-gc.test.ts:74`, „GC backstop with views"). Ohne das `:gc`-Leg wäre er stumm aus
  der Population gefallen.
- **Owner-Entscheidungen (2026-07-29):** Baustein C entfällt · die Skripte werden unter
  `scripts/` committet.

**Änderungslog v2 → v2.1 (2026-07-29, während der Umsetzung):** D7 zählte zwei Artefakte
auf; es sind **drei**, weil die Selbsttests 12/13 Aussagen über den TRACER sind und einen
echten Testlauf brauchen. Offengelegt in D7, nicht still geändert. Keine Auswirkung auf
Design, Regel, Population oder Gates.

## Ziel in einem Satz

Die rund **1.100 Testfälle**, die die View-Precondition-Scheibe als Kandidatenmenge
zurückgelassen hat, werden **mechanisch klassifiziert** — nach einer vorregistrierten,
fuzzbaren Entscheidungsregel und über eine **andere Auffindungsmethode** als den
Stichwort-Sweep, der die Menge erzeugt hat (Arbeitsregel 15). Ergebnis ist eine
**Partition mit Fallzahlen**, kein Fix.

## Was diese Scheibe ausdrücklich NICHT ist

- **Keine Vakuitäts-Liste.** Die 1.100 sind eine Kandidatenmenge, entstanden als
  „überlebt Mutant S **und** Mutant T **und** trägt ein View-Stichwort im Namen". Stichproben
  der Vorgänger-Scheibe zeigen, dass die Mehrheit legitim unbetroffen ist. Wer auf dieser
  Menge handelt, handelt auf einer ungeprüften Population — genau der Fehler, den die
  vorletzte Scheibe korrigiert hat.
- **Kein Fix.** Es wird keine einzige Precondition-Assertion eingefügt. Was einen Pin
  braucht, entscheidet der Owner nach dieser Klassifikation, auf Basis benannter Fallzahlen.
- **Keine Aussage über Korrektheit.** Alle betroffenen Tests sind grün und prüfen ihre
  Operation korrekt. Zur Debatte steht ausschließlich ihre **Coverage-Aussage**: ob der
  Prüfling die Klasse von Eingabe *ist*, die der Testname behauptet.

## Warum die Methode der Vorgänger-Scheibe hier nicht reicht

Der Sweep hat zwei Filter verkettet, und **beide** sind für eine Klassifikation untauglich:

1. **Mutanten-Überleben** konflatiert zwei unvereinbare Sachverhalte: „der Test kann den
   Schaden nicht sehen" (Defekt) und „der Mutant kann den Test strukturell nicht erreichen"
   (Nicht-Messung). Die Vorgänger-Spec hat das für ihre 13 Blöcke per Hand getrennt
   („nicht anwendbar" vs. „überlebt"); für 1.100 Fälle skaliert das nicht, und Handarbeit
   ist genau die Fehlerquelle, um die es geht.
2. **Stichwort im Namen** ist die Behauptung, die geprüft werden soll — sie als Filter zu
   benutzen, ist zirkulär. Sie ist außerdem **einseitig unvollständig**: ein Testblock, dessen
   Existenzgrund ein View ist, dessen Name das aber nicht sagt, ist für den Sweep
   **unsichtbar**. Diese Richtung hat noch niemand gemessen.

Die KB-Notiz `vakuoeser-test-nur-per-mutant-erkennbar` (Nachtrag 2026-07-26) formuliert
den Kurzschluss sogar wörtlich: „Die Blöcke, die dabei NICHT fallen, sind exakt die
vakuösen." Für die dort betrachteten 13 Blöcke stimmte das; als allgemeine Regel ist es
**falsch**, und die Korrektur dieser Notiz gehört zum Ertrag dieser Scheibe.

## D1 — Achse A: dynamisches Call-Tracing statt Namen (die andere Methode)

Statt Testnamen zu lesen, wird **beobachtet, was ein Testfall tatsächlich tut**. Ein
Tracer protokolliert pro Testfall jede View-Konstruktion mit ihrer **exakten Struktur**.

**Vier Fakten sind vorab empirisch belegt** (2026-07-29, im Haupt-Baum, ohne Edit an
`spike/src`; Belege gehören ins Ergebnis-Doc):

1. **Ein `--import`-Preload erreicht die Testprozesse.** `node --test` startet pro Datei
   einen Kindprozess; über `NODE_OPTIONS="--import file://…"` wird der Preload dort
   ausgeführt und patcht `WNDArray.prototype` **derselben** Modulinstanz, die die Tests
   importieren. Verifiziert an `keepdims.test.ts`: 367/367 grün, Trace geschrieben.
2. **Stack-Frames tragen exakte Quellzeilen** (`…/keepdims.test.ts:173:24`) — Nodes
   TS-Type-Stripping erhält Zeilennummern.
3. **Ein root-level `beforeEach` aus dem Preload sieht JEDEN Testfall.** `beforeEach((t) => …)`
   mit `t.name` liefert für `keepdims.test.ts` exakt **367 Einträge** bei 367 Testfällen und
   für die volle `test:resident`-Liste exakt **6124** bei 6124 Passes (Baustein 0,
   unabhängig) — also eine **vollständige Fall-Enumeration**, nicht eine Stichprobe. Das ist
   die Grundlage der Restmengen-Invariante (D5). Belegt ist außerdem, dass der Korpus **keine
   verschachtelten Subtests, kein `describe()`, kein `.only/.skip/.todo` und keine
   `concurrency`** benutzt — jeder Test ist ein flacher Top-Level-`test(name, fn)`, teils in
   Modul-Schleifen erzeugt, alle sequenziell. Damit gibt es keine Race um die
   „aktueller Fall"-Zuordnung.
4. **`WNDArray`s Struktur-Felder sind zur Laufzeit SCHREIBBAR** (`private readonly` ist ein
   TS-Modifier, kein `#private`) — Achse **B1** braucht deshalb keinen Eingriff in die Quelle.
   Zusätzlich belegt: `isContiguous()` (resident.ts:422) rechnet bei jedem Aufruf aus den
   **lebenden** Feldern, cached also nichts, was eine Perturbation desynchronisieren könnte.
   **v2-Korrektur (B1):** Schreibbarkeit ist NICHT dasselbe wie Lese-Beobachtbarkeit. Für
   Achse **B2** gilt sie nicht — siehe D2.

**Konsequenz, die diese Scheibe methodisch von allen Vorgängern unterscheidet:** die
Messung ist **vollständig non-invasiv**. `spike/src` wird **nicht angefasst** — auch nicht
für die Mutation in D2. Die Isolations-Auflage für mutierende Läufe ist damit
**strukturell** erfüllt statt per Disziplin: es existiert kein mutierter Working Tree, den
ein zweiter Schreiber verfälschen könnte.

**Erfasste View-Produzenten** — alle vier Methoden, die auf `WNDArray` ein Handle mit
eigener `(shape, strides, offset)`-Metadaten konstruieren können: `transpose`
(resident.ts:1315), `slice` (1336), `reshape` (1381), `flatten` (1425).

**Baustein-0-Befund, der die Sache vereinfacht:** nur `transpose` und `slice` können
überhaupt einen **non-trivialen** View erzeugen. `reshape` und `flatten` konstruieren im
View-Zweig immer mit `computeStrides(newShape)` und Offset 0, im Materialize-Zweig über
`WNDArray.fresh(...)` — beides contiguous; `contiguous()` (1469) materialisiert immer frisch.
Alle vier werden trotzdem getract: dass die beiden nur K1 produzieren, ist eine **Ausgabe**
der Messung, keine Voraussetzung.

`NDArray.transpose`/`slice` sind **immer Kopien** (ndarray.ts:658-660/668-671 → `transposeRuntime`/
`sliceRuntime`, runtime.ts:351/394, je frische `Float64Array`), haben also keine View-Klasse
und sind kein Tracer-Ziel; ihre Rolle in dieser Scheibe ist die des unabhängigen Orakels.
`ThreadedBackend`/`WasmBackend` benutzen dieselbe `WNDArray.prototype` und sind automatisch
erfasst.

**Population P** = alle Testfälle aus `test:core` ∪ `test:resident` ∪ **`test:resident:gc`**
∪ `test:threaded`. Bestätigte Fallzahlen (eigene Läufe, Exit 0): 1591 · 6126 tests / 6124
pass / **2 skipped** · 139.

**Das `:gc`-Leg ist Pflicht, nicht Kosmetik.** Übersprungene Tests feuern **kein**
`beforeEach`. Die zwei Skips sind `resident-gc.test.ts:34` und `:74` (bedingt auf
`globalThis.gc`), und der zweite heißt **„GC backstop with views: dropped base+view free
their shared buffer exactly once"** — ein Fall, der einen View konstruiert und ein
View-Stichwort trägt. Ohne das `:gc`-Leg fiele er stumm aus der Population: genau die
„stille Ausklammerung", die D4s Selbsttest 9 als harten Fehler verbietet.

**`test:browser` ist per KONSTRUKTION ausgeschlossen, und das ist eine offengelegte Grenze**
(v2, B0-Befund M2): `spike/tests-browser/smoke.test.ts:213-229` ruft sehr wohl
`.transpose()`/`.slice(1)` — aber **innerhalb einer Chromium-Seite über `page.evaluate`**,
also in einer zweiten JS-Laufzeit ohne Verbindung zu Nodes Modulregistry oder
`NODE_OPTIONS`. Ein Node-Preload reicht dort strukturell nicht hin. v1 wollte das „durch
einen Tracer-Lauf mit null Treffern belegen" — das hätte null gemeldet, aber **aus dem
falschen Grund**, und wäre eine vakuöse Prüfung gewesen. `test:package` ist geprüft
view-frei (nur `sum`/`add`). Der Rest der Scheibe darf keine Aussage über die
Browser-Testfälle machen.

**Abgeleitete Größen pro Fall `c`:**

| Größe | Definition (mechanisch) |
|---|---|
| `views(c)` | Liste der View-Konstruktionen mit `{op, shape, strides, offset}` |
| `nontrivial(c)` | ∃ Datensatz, dessen Index→Linear-Abbildung von der contiguous Abbildung derselben Shape abweicht — Definition und Geschlossene Form unten |
| `readback(c)` | Der Fall ruft `toArray()`/`describe()` auf einem selbst konstruierten **non-trivialen** View-Handle auf, **aus einem `*.test.ts`-Frame** (Achse-B2-Signal, D2) |

### `nontrivial` — semantische Definition und Geschlossene Form (v2, B1-Blocker-Fix)

Für ein Handle mit `(shape, strides, offset)` ist die Adressabbildung
`lin(idx) = offset + Σᵢ idx[i]·strides[i]` über der Menge gültiger Indextupel. **Non-trivial
heißt: `lin` weicht von der contiguous Abbildung derselben Shape ab** (`offset = 0`,
`strides = naturalStrides(shape)`).

**Geschlossene Form** — für Achsen mit `shape[i] = 1` ist stets `idx[i] = 0`, ihr Stride kann
also nie zu einer Adresse beitragen; für `size = 0` existiert kein gültiges Indextupel:

```
nontrivial  ⇔  product(shape) > 0
               ∧ ( offset ≠ 0  ∨  ∃ i : shape[i] > 1 ∧ strides[i] ≠ naturalStrides(shape)[i] )
```

Rang 0 (`shape = []`, `product = 1`) reduziert das korrekt auf `offset ≠ 0` — der Fall, den
ein Slice mit ausschließlich Integer-Specs erzeugt.

**Bindend:** die Geschlossene Form ist gegen **Brute-Force** abzugleichen (Selbsttest D4/11).
v1s naive Fassung (`strides ≠ naturalStrides` **oder** `offset ≠ 0`) ist widerlegt: ein
transponierter `[1,5]`-View erfüllt sie, hat aber **0 von 5** abweichenden Adressen.

## D2 — Achse B: greift die Assertion überhaupt auf die Struktur zu?

`nontrivial(c)` sagt, dass ein echter View gebaut wurde — nicht, ob der Test seine
Zerstörung bemerken **würde**. Dafür zwei **unabhängige** Verfahren, plus ein
Widerspruchs-Register.

### B1 — Perturbation P (entscheidend, garantiert gültig)

Jeder non-triviale View wird nach seiner Konstruktion zur **contiguous
Re-Interpretation derselben Shape** gemacht: `strides := naturalStrides(shape)`,
`offset := 0`. Eigenschaften, die diese Wahl gegenüber Mutant S/T auszeichnen:

- **Uniform.** Ein Mutant deckt beide View-Klassen ab (transponiert *und* gestridet
  *und* offset-verschoben *und* komponiert). Keine „nicht anwendbar"-Zeilen mehr, die aus
  der Wahl des Mutanten folgen statt aus dem Test.
- **Garantiert in-bounds.** Der maximale Linearindex ist `size − 1`, also nie größer als
  bei jedem contiguous Handle derselben Shape. Die HANDOFF-Lehre „Mutanten-Grobheit" ist
  damit **konstruktiv** ausgeschlossen, nicht nachträglich geprüft — kein Wurf aus dem
  Puffer, der Fehlschläge vortäuscht, bevor eine Assertion läuft.
- **Wirksamkeit wird protokolliert, nicht unterstellt.** P wird **nur** auf non-triviale
  Views angewandt (Definition oben), und `perturbed(c)` ist genau dann wahr, wenn sich die
  **Adressabbildung** geändert hat — nicht, wenn bloß ein Strides-Array überschrieben wurde.
  Das ist der Blocker-B2-Fix an seiner zweiten Stelle: ohne diese Unterscheidung hätte P an
  Achsen der Länge 1 „gewirkt" und der Fall wäre als struktur-blind ausgewiesen worden.

**Bindend:** Der Perturbations-Preload protokolliert pro Fall (a) `perturbed(c)`, (b) bei
Fehlschlag die **Fehlermeldung**. Ein Fall gilt nur dann als struktur-sensitiv, wenn er mit
einer **Assertion**-Meldung fällt — nicht mit einem Wurf aus der Datenzugriffsschicht.
Fällt irgendein Fall mit einem Nicht-Assertion-Wurf, ist das offenzulegen und einzeln zu
klären, nicht in `K3` mitzuzählen.

### B2 — Provenienz-Signal (unabhängig, nicht entscheidend)

`readback(c)` aus D1: ruft der Fall `toArray()`/`describe()` auf dem selbst konstruierten
non-trivialen View auf? Das ist der mechanische Abdruck des Muster-4-Defekts („Orakel aus dem
Prüfling", KB-Notiz `vakuoeser-test-nur-per-mutant-erkennbar`) — alle fünf bekannten
Assertions-Helfer bilden ihre Referenz genau so.

**v2, Blocker-B1-Fix — zwei Verengungen, beide bindend:**

1. **Nur echte Prototyp-Methoden**, `toArray()` und `describe()`. v1 nannte zusätzlich
   `shape` — das ist eine **gewöhnliche Daten-Property**, und die Bibliothek liest sie intern
   fortwährend: Baustein 0 hat per Accessor gezählt, dass ein einzelnes `toArray()` **4**
   interne `.shape`-Lesevorgänge auslöst und `describe()` einen weiteren. Ein
   `.shape`-Trigger hätte für jeden View gefeuert, auf dem irgendeine Op läuft.
2. **Nur aus einem `*.test.ts`-Frame.** Auch `toArray()`/`describe()` können intern
   aufgerufen werden. Der Tracer prüft den unmittelbaren Aufrufer-Frame über die in D1
   Fakt 2 belegte Stack-Zuordnung und zählt nur Aufrufe aus Testcode.

**Ohne diese zwei Verengungen wäre das Widerspruchs-Register Rauschen gewesen** — und zwar
überzeugend aussehendes: `K3 ∩ readback` wäre groß geworden, weil K3-Fälle Ops auf dem View
laufen lassen und dabei intern `.shape` lesen. Genau die Sorte Zahl, die diese Codebasis
schon zweimal in eine falsche Richtung geschickt hat.

**B2 bleibt ein Proxy und wird ausdrücklich als solcher behandelt.** Ein `toArray()` auf dem
View kann auch ein völlig unabhängiges Orakel speisen (Vergleich gegen ein handgeschriebenes
Literal). B2 dient deshalb **nicht** der Klassifikation, sondern als **Kreuzprobe auf einer
zweiten Achse**.

### Widerspruchs-Register (Pflicht)

Beide Achsen liefern eine Vorhersage. Erwartet ist grobe Übereinstimmung
(`K4 ⊆ readback`, `K3 ∩ readback` klein). **Jede Zelle, in der sie sich widersprechen,
wird namentlich aufgeführt und einzeln erklärt** — nicht gemittelt, nicht weggerundet. Ein
leeres Register ist ein Befund und muss als solcher dastehen; ein volles Register zeigt,
welche der beiden Achsen zu grob ist.

## D3 — Die vorregistrierte Entscheidungsregel

Total, disjunkt, in dieser Reihenfolge auszuwerten. Jeder Fall `c ∈ P` landet in **genau
einer** Klasse.

| # | Bedingung | Klasse | Bedeutung |
|---|---|---|---|
| 1 | `views(c) = ∅` | **K0 — kein View** | Der Fall konstruiert keinen View. Ein View-Stichwort im Namen meint etwas anderes (rohe ABI, Lifecycle, Fehlerpfad). Mutanten-Überleben trägt **keine Information**. |
| 2 | `¬nontrivial(c)` | **K1 — nur trivialer View** | View-Produzent aufgerufen, die Adressabbildung ist aber die contiguous (Rang ≤ 1, Doppel-Transpose, Voll-Slice, `reshape`/`flatten`, `size = 0`, **und jede Abweichung, die nur an Achsen der Länge 1 hängt**). Es gibt keine View-Klasse, die man verlieren könnte. |
| 3 | `¬perturbed(c)` | **K2 — nicht messbar** | Trotz non-trivialem View hat P nichts geändert. **Ausdrücklich kein Freispruch, sondern eine Nicht-Messung** — die Zeile, deren Zweideutigkeit die Vorgänger-Spec als Blocker M1 korrigieren musste. Erwartet leer; nicht leer = eigener Befund. |
| 4 | `fellP(c)` (mit Assertion-Meldung) | **K3 — struktur-sensitiv** | Der Fall **fängt** einen falschen View. Sound, kein Handlungsbedarf. |
| 5 | sonst | **K4 — struktur-blind** | Non-trivialer View, wirksam perturbiert, Fall bleibt grün: der Test kann einen falschen View **nicht** bemerken. **Kandidat**, kein Urteil. |

**Der einzige nicht-mechanische Schritt** ist die Unterteilung von K4 — und er ist als
solcher offenzulegen:

- **K4a** — der Fall behauptet eine View-Klasse (Name, Header-Kommentar oder
  Block-Existenzgrund) → **braucht einen Precondition-Pin**.
- **K4b** — der Fall behauptet etwas anderes (Ressourcen-Bilanz, Fehlerpfad,
  Typ-Diagnose, Fassaden-Erreichbarkeit) und benutzt den View nur als Vehikel → legitim.

**Bindend für K4a/K4b:** jeder Fall wird **namentlich** gelistet, mit der Zeile, auf die
sich das Urteil stützt. Ist K4 groß (> 60 Fälle), wird nach Testblock gruppiert und die
Gruppengrenze belegt (Schleifengrenzen aus der Quelle, nicht geschätzt). Diese Handarbeit
ist zulässig, **weil** die mechanischen Stufen 1–5 die Menge vorher auf das reduziert
haben, was ein Urteil braucht — die Reihenfolge ist der ganze Punkt.

## D4 — Arbeitsregel 7: die Regel wird gefuzzt, BEVOR echte Zahlen existieren

Die Regel aus D3 wird als Skript implementiert und gegen **synthetische** Ledger geprüft,
bevor ein echter Trace vorliegt. Die topk-Messung hat gezeigt, dass eine gelesene Regel
eindeutig **wirkt** und trotzdem in vier Runden achtmal brach — und dass zwei der
gebrochenen Fassungen vom Orchestrator selbst stammten.

**Pflicht-Testfälle des Klassifizierers** (jeder einzeln, mit erwarteter Klasse):

1. Leere `views`-Liste → K0.
2. `transpose` auf Rang 1 (`shape [n]`, natural strides, offset 0) → K1.
3. Non-trivialer View, `perturbed = false` → K2 (**nicht** K3/K4).
4. Non-trivialer View, perturbiert, Fall fällt mit Assertion → K3.
5. Non-trivialer View, perturbiert, Fall grün → K4.
6. Fall fällt mit **Nicht**-Assertion-Wurf → **nicht** K3, sondern als Anomalie gemeldet.
7. Mehrere Views in einem Fall, davon einer non-trivial → wie non-trivial behandelt.
8. `shape [2,2,2,2]`, transponiert: Shape **identisch**, Strides verschieden → muss
   `nontrivial` sein (der L3-Fall der Vorgänger-Spec, an dem eine Shape-Prüfung
   grundsätzlich nichts merkt).
9. Fall in P, der im Ledger fehlt → **harter Fehler**, keine stille Ausklammerung.
10. Fall zweimal im Ledger (gleicher Name in zwei Dateien) → Kollision muss auffallen,
    nicht stumm zusammenfallen (Fall-Identität ist `datei + name`, nicht `name`).

**Drei weitere, aus Baustein 0s Testplan-Lücken (v2 — die ersten zehn hätten die zwei
Blocker NICHT gefangen):**

11. **Geschlossene Form gegen Brute-Force.** Für alle Shapes mit Rang ≤ 4 und Dimensionen
    ∈ {0,1,2,3}, über eine Auswahl von Strides/Offsets: `nontrivial` per Geschlossener Form
    gegen vollständige Aufzählung aller gültigen Indextupel. **Muss überall übereinstimmen.**
    Enthält damit zwingend den transponierten `[1,5]`-Fall, an dem v1 falsch lag, und die
    `size = 0`- und Rang-0-Kanten. Dieser eine Test ist der Grund, dass die Definition dieser
    Scheibe nicht auf Zutrauen beruht.
12. **`.shape` allein löst kein `readback` aus.** Ein synthetischer Fall, der `.shape` liest,
    aber weder `toArray()` noch `describe()` aufruft → `readback = false`. Hält die
    B1-Verengung fest, damit sie nicht später „vereinfacht" wird.
13. **Verschachtelter Node-Subprozess.** Ein Fall, der wie die real-tsc-Diagnose-Pin-Tests
    einen Node-/`tsc`-Kindprozess startet, darf den Ledger nicht verfälschen (kein
    Doppel-Eintrag, kein fehlender Eintrag). Baustein 0 hat das Leck live reproduziert.

**Die Regel darf nicht nachträglich geändert werden**, nachdem echte Zahlen vorliegen.
Wird sie es doch, ist die Änderung mit Grund und Zeitpunkt zu dokumentieren und die
Klassifikation neu zu rechnen — beides offen im Ergebnis-Doc.

## D5 — Vollständigkeit über eine andere Achse (Arbeitsregel 15)

Arbeitsregel 15 verlangt, dass die Vollständigkeits-Prüfung **nicht** dieselbe Methode
benutzt wie die Arbeit. Drei Prüfungen, alle mechanisch:

1. **Partitions-Invariante.** `|K0| + |K1| + |K2| + |K3| + |K4| = |P|`, wobei `|P|` aus der
   **`beforeEach`-Enumeration** stammt (jeder Fall genau einmal) und **nicht** aus der
   Klassifikation. Beide Zahlen müssen zusätzlich zur Summe der Reporter-Fallzahlen
   (`test:core` 1591 + `test:resident` 6124 + `test:threaded` 139) passen; jede Abweichung
   ist zu erklären, nicht zu glätten.
2. **Subset-Probe gegen die alte Menge.** Die alte Kandidatenmenge wird reproduziert
   (Mutant S und T, ebenfalls **per Preload**, ebenfalls ohne Quell-Edit) und ihre
   Verteilung über K0–K4 berichtet. Erwartung: der Großteil landet in K0/K1. **Trifft das
   nicht zu, ist es ein Befund über die alte Messung, kein Fehler dieser.**

   **v2, M1-Fix — die Emulation ist exakt anzugeben, sonst ist die Äquivalenz behauptet statt
   falsifizierbar.** Die Quell-Mutanten der Vorgänger-Scheibe sind nicht in der Git-Historie
   (per Backup-Kopie revertiert), also trägt die Konstruktion die Beweislast:
   - **Mutant S** = `orig.apply(this, [])` — `slice` durchgerufen mit leerer Spec-Liste,
     also `normalizeSliceSpecs(this.shape, [])`: wortgleich zum Original. Retain,
     Handle-Konstruktion und Registry-Registrierung bleiben unberührt.
   - **Mutant T** = `orig.apply(this)` durchgerufen, danach `shape`/`strides` des ERGEBNISSES
     auf die des Empfängers zurückgeschrieben. Damit ist `transpose` die Identität, während
     `retainBuffer` + frisches Handle + Registry **exakt** wie im Original laufen.
     Ausdrücklich **nicht** `return this` — das würde die Refcount-Buchhaltung verändern und
     eine andere Kandidatenmenge erzeugen, ohne dass irgendwo ein Fehler entstünde.
   - **Nachweis, dass die Emulation trägt:** die 304 in der Vorgänger-Scheibe reparierten
     Fälle müssen unter S bzw. T **fallen** (sie haben jetzt Preconditions). Fallen sie
     nicht, ist die Emulation nicht äquivalent — das ist ein billiger, scharfer Kontrollpunkt
     und Pflicht.
3. **Die Gegenrichtung, die noch niemand gemessen hat.** `K4 \ alte Kandidatenmenge` —
   struktur-blinde Fälle **ohne** View-Stichwort im Namen. Diese Menge ist für den alten
   Sweep prinzipiell unsichtbar und **muss** berichtet werden, auch (und besonders) wenn sie
   leer ist: leer wäre der Beleg, dass die Namensheuristik zufällig vollständig war.

**Pflicht-Gegenprobe zur Perturbation:** kein Fall aus K0 (`views = ∅`) darf unter P
fallen. Fällt einer, ist P nicht so eng wie behauptet und der ganze Lauf ungültig.

## D6 — Zählregeln (die Fallen, die diese Codebasis schon zweimal getroffen haben)

- **Der `node --test`-Reporter druckt Fehlschläge zweimal** (inline + Summary). `grep -c`
  auf den Rohoutput verdoppelt sie und lässt Passes einfach — das hat in der
  View-Precondition-Scheibe eine Kontrollgruppe um Faktor 2 verfälscht, ausgerechnet in der
  Tabelle, die die Spec „der tragende Teil des Beweises" nannte. **Bindend:** gezählt wird
  über die **eindeutige Fall-Identität** `datei + name` aus dem Ledger, nie über
  Reporter-Zeilen.
- **Exit-Code und Fehleranzahl gehören zu jeder Zahl** (Arbeitsregel 6). Ein Lauf kann mit
  Exit 1 scheitern und trotzdem plausible Zahlen drucken.
- **`$pipestatus[1]`**, nicht `${PIPESTATUS[0]}`; `status` ist in zsh read-only; `cd <root>
  2>/dev/null;` mit Semikolon, auch **innerhalb** von Skriptdateien.
- **`NODE_OPTIONS` wird inline gescopet, nie exportiert** (v2, Baustein-0-Nebenbefund, live
  reproduziert). Zwei Leckwege: (a) die real-tsc-Diagnose-Pin-Tests starten `tsc` als
  Kindprozess, der `NODE_OPTIONS` erbt und den Preload mitlädt — hier harmlos (Pass-/Skip-/
  Exit-Zahlen unverändert), aber „ein Preload-Load pro Testdatei" ist damit **nicht** wörtlich
  wahr, und der Tracer muss gegen Mehrfach-Initialisierung robust sein; (b) ein exportiertes
  `NODE_OPTIONS` leckte in völlig unbeteiligte Prozesse derselben Shell (`pnpm install`, ein
  fremder `tsc --noEmit`-Pin-Lauf). Also: `VAR=… node --test …` in **einer** Zeile.
- **Statische Grep-Kreuzproben müssen `WNDArray.slice` von `Array.prototype.slice`
  unterscheiden.** In `resident-lifecycle.test.ts` sind die meisten `.slice(`-Treffer
  Array-Slices auf Allokations-Logs. Der Tracer hat dieses Problem nicht — jede *statische*
  Gegenprobe hat es.

## D7 — Artefakte

- `scripts/trace-view-tests.mjs` — der Preload (Tracer + optional Perturbation + optional
  Mutant S/T, per Env umgeschaltet). Muss ohne Umschaltung **verhaltensneutral** sein: der
  Nachweis ist ein voller Suite-Lauf mit unveränderten Fallzahlen.
- `scripts/classify-view-tests.mjs` — die Regel aus D3 plus die Selbsttests 1–11 aus D4.
- `scripts/selftest-tracer-integration.mjs` — die Selbsttests **12 und 13**.
  **Offengelegte Abweichung von v2s Zwei-Artefakte-Aufzählung (v2.1, 2026-07-29):** diese
  zwei sind keine Aussagen über die REGEL, sondern über den TRACER, und lassen sich gegen
  einen synthetischen Ledger nicht prüfen — sie brauchen einen echten `node --test`-Lauf und
  spawnen deshalb einen. Das in `classify-view-tests.mjs` zu bündeln hätte einen reinen
  Rechenschritt mit einem Prozess-Spawn vermischt. Die Fixture liegt bewusst **außerhalb**
  von `spike/tests-runtime/` (dort failt `test-scripts-guard.test.ts` auf nicht
  registrierten Testdateien) und wird in einem tmp-Verzeichnis erzeugt und wieder gelöscht.
  Der Gate-Block ist unberührt: auch die dritte Datei liegt außerhalb von
  `tsconfig.json`s `include: ["spike"]`.
- Ledger-Rohdaten (JSONL) landen im Scratchpad, **nicht** im Repo.

Beide Skripte liegen unter `scripts/` (`.mjs`) und damit **außerhalb** von
`tsconfig.json`s `include: ["spike"]` — sie können den Instantiations-Zähler nicht
bewegen. Das ist zu **messen** (Δ0), nicht anzunehmen.

**Warum committen und nicht wegwerfen:** die Klassifikation wird nach jedem Fix erneut
gebraucht (sie ist das Abnahmekriterium des Folgeschritts), und ein Tracer, der nur im
Ergebnis-Doc beschrieben ist, ist nicht reproduzierbar. Der Preis ist ein Artefakt, das
niemand aufruft — dagegen steht, dass `scripts/` genau dafür existiert
(`check-freeze-hash.mjs`, `check-dist-emit.mjs`).

## Vorregistrierte Gates

Diese Scheibe ändert **keinen** Produktionscode und **keinen** Test. Erwartet ist überall Δ0
— was den Gate-Block zur echten Prüfung macht, nicht zur Formalie:

- `check:diag` **225.983 @ 140** · `:stress` **116.053 @ 82** · `:browser` **2.142 @ 75** —
  je **Δ0** und **Dateiset unverändert** (die neuen Skripte liegen außerhalb von `spike/`).
  Jede Bewegung ist ein Befund und aufzuklären, bevor die Scheibe abgeschlossen wird.
- `bench:editor` acht Pins exakt · `check:freeze` `2a54d9fd…` · `cargo test` 222+1 — Δ0.
- **Testzahlen zahlengleich:** `test:core` 1591 · `test:resident` 6124+2 ·
  `test:threaded` 139 · `test:browser` 4 · `test:package` 3. Es kommt kein Test hinzu und
  keiner weg; die Selbsttests des Klassifizierers laufen **im Skript**, nicht in einer
  Test-Suite (sonst müssten sie in `package.json` registriert werden und würden die
  Fallzahl bewegen — bewusste Entscheidung, im Ergebnis-Doc zu benennen).
- **Alle drei Baselines sind vor der Implementierung bestätigt** (eigene Läufe, 2026-07-29,
  je Exit 0): `test:core` 1591/1591 pass · `test:resident` 6126 tests / 6124 pass / 2 skipped
  / 0 fail · `test:threaded` 139/139 pass. Der Gate-Block vergleicht gegen **diese** Zahlen,
  nicht gegen die aus CLAUDE.md abgelesenen.
- **Verhaltensneutralität des Tracers ist ein Gate, keine Zusicherung:** ein voller
  Populationslauf MIT Preload muss dieselben Pass-/Skip-/Fail-/Exit-Zahlen liefern wie ohne.
  Weicht eine ab, ist die Messung ungültig — nicht zu interpretieren.
- `pnpm check` (drei Legs) · `graph-a-lama query lint` Exit 0.

## Abnahmekriterium

Die Scheibe ist fertig, wenn **alle** davon vorliegen:

1. Eine **Partitionstabelle** K0–K4 mit Fallzahlen, deren Summe die
   `beforeEach`-Enumeration exakt trifft (D5.1).
2. Die **Verteilung der alten ~1.100** über K0–K4 (D5.2) — inklusive der Angabe, wie stark
   die reproduzierte Menge von „~1.100" abweicht.
3. Die **Gegenrichtung** `K4 \ alte Menge` (D5.3), auch wenn leer.
4. Das **Widerspruchs-Register** B1 vs. B2 (D2), auch wenn leer.
5. **K4 namentlich** aufgeschlüsselt in K4a/K4b, je mit der Zeile, auf der das Urteil beruht.
6. Die **Selbsttests** aus D4 grün, gelaufen **vor** der ersten echten Messung — mit
   Zeitstempel-Nachweis in dieser Reihenfolge.
7. Der **Gate-Block** vollständig Δ0.
8. Die Pflicht-Gegenproben: kein K0-Fall fällt unter P; kein K3-Fall fällt mit einem
   Nicht-Assertion-Wurf (oder: benannt und geklärt).

**Was ausdrücklich NICHT zum Abnahmekriterium gehört:** eine Empfehlung, welche Fälle
gefixt werden. Die Klassifikation liefert die Entscheidungsgrundlage; die Entscheidung
gehört dem Owner.

## Berührte Covenant-Invarianten

- **S1** (Runtime importiert nie aus Test-/Bench-/Demo-Verzeichnissen) — nicht berührt. Die
  neuen Skripte liegen in `scripts/` und werden von `spike/src` nicht importiert;
  mechanisch durch `graph-a-lama query lint` gedeckt.
- **M1** — nicht berührt (kein Kernel, kein Verhalten, kein Artefakt). Inhaltlich ist die
  Scheibe **M1-stützend**: M1s Bit-Identitäts-Anspruch für residente Ops beruht auf genau
  diesen View-Tests, und diese Scheibe misst, wie viel davon trägt.
- **M3/M4** — nicht berührt: keine Typfläche, kein Rust, kein Artefakt-Byte.

**Eskalationsstufe: 3** (bindende Spec) → Baustein 0 vorab, danach A + B parallel, **je im
eigenen Worktree**. **Baustein C entfällt — Owner-entschieden 2026-07-29**, Begründung
gehört in einem Satz in den Commit (dokumentiertes Ermessen, wie die Leiter es für Stufe 2
vorsieht): kein Vertragstext, keine Invarianten-Fläche, der Diff sind zwei `.mjs`-Skripte
außerhalb von `spike/` und `crates/`. Baustein 0 hat unabhängig bestätigt, dass S1 nicht
berührt ist (`graph-a-lama.rules.json` verbietet `spike/src → tests|bench|demo`, nicht die
Gegenrichtung; keiner der drei Guard-Tests scannt `scripts/`). Baustein 0 hat außerdem
angeregt, die C-Entscheidung nach den Blocker-Fixes erneut vorzulegen — **nicht
weitergegeben, mit Begründung:** die beiden Fixes verengen eine Messachse und korrigieren
eine Definition; sie berühren keine Vertragsfläche, an der sich C etwas zu prüfen holen
könnte. Die Entscheidungsgrundlage ist unverändert.

**Isolations-Auflage (CLAUDE.md, Verify-Runden-Regel):** hier strukturell erfüllt — jede
Mutation dieser Scheibe lebt im **Preload**, nicht in der Quelle. Ein Verifier, der die
Messung nachstellt, mutiert nichts im Working Tree. Die Auflage gilt trotzdem formal: wer
darüber hinaus einen Quell-Mutanten fährt, tut es im eigenen Worktree.

## Disziplin-Entscheidung

Zwei **neue** Dateien unter `scripts/`, null Änderungen an `spike/` und `crates/`. Die
Append-/Insertion-only-Disziplin greift nicht (sie ist ein Artefakt-Byte-Argument für
`crates/core` bzw. TS-Klassenkörper). Der Diff ist reine Addition.

## Doku-Pflichten am Ende

`docs/view-test-klassifikation-ergebnisse.md` mit Post-Verification-Addendum · CLAUDE.md
(„Status" — Einzeiler; „Aktuelle Pins & Gates" nur, falls sich wider Erwarten etwas bewegt)
· FOLLOWUPS (diesen Eintrag schließen, den Folge-Entscheidungs-Eintrag für K4a öffnen) ·
`docs/projekt-log.md` (Narrativ) · **KB-Capture, mit ausdrücklicher Revision** der Notiz
`vakuoeser-test-nur-per-mutant-erkennbar`: der Satz „Die Blöcke, die dabei NICHT fallen,
sind exakt die vakuösen" ist als Verallgemeinerung falsch und durch die hier gemessene
Verteilung zu ersetzen. README nicht betroffen (es ändert sich nichts daran, wo eine Op
läuft) — Prüfkommando trotzdem laufen lassen.

## Ein-Block-Revert

Zwei neue Dateien, kein berührter Bestand. Rückbau = `git revert` des einen Commits; es
wandert **kein** Pin zurück.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-29)

EIN `brainroute:deep`-Agent, frischer Kontext, gegen v1 dieser Spec und den echten Code;
Auftrag aus `docs/verify-runde-template.md` „Baustein 0". Er hat in einem eigenen
`git worktree` gearbeitet, den Haupt-Baum nicht angefasst (belegt) und keine Gates gefahren.

**Verdikt: zwei BLOCKER, beide bestätigt, beide Fehler in v1 — und beide an derselben Stelle
wie in der Vorgänger-Spec: nicht im Design, sondern in einem als „empirisch belegt"
ausgewiesenen FAKT.**

- **B1** hat v1s Fakt 4 zerlegt: Schreibbarkeit ≠ Lese-Beobachtbarkeit. Belegt **durch
  Messung, nicht durch Lesen** — `Object.getOwnPropertyDescriptor(view, "shape")` zeigt eine
  gewöhnliche Daten-Property, und ein Accessor-Zähler ergab **4** interne `.shape`-Lesungen
  pro `toArray()`. Ein `.shape`-getriggertes `readback` hätte das Pflicht-Widerspruchsregister
  in Rauschen verwandelt, das wie ein Signal aussieht.
- **B2** hat `nontrivial` an Achsen der Länge 1 widerlegt — ebenfalls durch Rechnen, nicht
  durch Argumentieren: transponierter `[1,5]`-View, **0 von 5** Adressen weichen ab. v1 hätte
  solche Fälle als K4 („der Test kann eine Korruption nicht bemerken") ausgewiesen, wo
  überhaupt keine Korruption bemerkbar ist. Das ist ein Fehler **genau in der Ausgabe, um die
  es geht**.

**Ebenfalls eingearbeitet:** M1 (Preload-Emulation von Mutant S/T exakt angeben, plus den
304-Fälle-Kontrollpunkt), M2 (`test:browser` ist NICHT view-frei, und v1s Prüfmethode dafür
wäre vakuös gewesen — `page.evaluate` läuft außerhalb von Nodes Reichweite), der
`NODE_OPTIONS`-Leckweg, die Grep-Disambiguierung, die drei neuen Selbsttests 11–13 sowie der
Zeilennummern-Nit (`reshape` 1381).

**Geprüft und HALTEND — je mit der Methode, nicht mit „gelesen und plausibel":**

- Der `--import`-Preload erreicht jeden `node --test`-Kindprozess, und `beforeEach` feuert
  **exakt einmal pro Pass**: 367/367 auf `keepdims.test.ts`, **6124/6124** auf der vollen
  `test:resident`-Liste (unabhängig nachgestellt, Exit 0).
- Übersprungene Tests feuern kein `beforeEach` — konsistent mit der eigenständig gefundenen
  Populations-Korrektur oben.
- Der Korpus hat **keine** verschachtelten Subtests, kein `describe()`, kein
  `.only/.skip/.todo`, keine `concurrency`, keine async-überlappenden Tests — geprüft über
  alle 14 `test:resident`-Dateien plus `test:core`s Liste. Damit ist die
  „aktueller Fall"-Zuordnung race-frei.
- `transpose`/`slice` sind die **einzigen** Methoden, die einen non-trivialen View erzeugen
  können (`reshape`/`flatten`/`contiguous()` konstruieren strukturell nur contiguous) —
  gelesen an resident.ts:1381-1468.
- `NDArray.transpose`/`.slice` kopieren immer (ndarray.ts:658-660/668-671 → runtime.ts:351/394).
- „Garantiert in-bounds" hält: kein `WNDArray`-erzeugender Pfad hinterlässt einen Puffer
  kleiner als `product(shape)`, also kann die contiguous Re-Interpretation nie über den Puffer
  lesen.
- `isContiguous()` cached nichts (resident.ts:422) — eine Perturbation kann kein abgeleitetes
  Feld desynchronisieren.
- Fehlerpfad-Tests kreuzen keine View-Konstruktion (`negative-paths.test.ts`,
  `backend-oom.test.ts`: null Treffer) — die Sorge, P könne absichtlich fehlerhafte Fixtures
  verfälschen, materialisiert sich dort nicht.
- Die Δ0-Gate-Behauptung für `scripts/*.mjs` hält strukturell: **alle sieben** `tsconfig*.json`
  schließen `scripts/` aus, keiner der drei Guard-Tests scannt es, und das Lint verbietet nur
  die Gegenrichtung.
- Die IST-Zahlen der Spec stimmen mit CLAUDE.md überein.
