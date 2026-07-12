# Phase-D-Vorarbeiten, Scheibe V1 — Union-Guard-Fix (Ergebnisse)

Spec: `docs/phase-d-vorarbeiten-spec.md` (Scheibe V1 + Gemeinsame Rahmenbedingungen +
Adversariales Addendum, Baustein 0 bereits verifiziert). Datum: 2026-07-13. Alles unten ist
in tatsächlich in dieser Session ausgeführten Kommandos verankert; Honesty-Rule gilt
durchgehend — unverifizierte Aussagen sind als solche markiert.

## Was gebaut wurde

Fünf bindende Entscheidungen (D-V1.1–V1.5), alle wie spezifiziert umgesetzt, **null Rust,
null resident.ts-Edits, null runtime.ts-Edits, null slice-literal.ts-Edits**:

- **D-V1.1 (`spike/src/dim.ts`):** private, unexportierte Kopie von `IsUnion`
  (`slice-literal.ts:629` als Quelle zitiert, Querverweis-Kommentar) — kein Import, kein
  Zyklus-Risiko.
- **D-V1.2 (`dim.ts`):** Union-Filter als ERSTE Prüfung in `CompatDim`/`DimEq`, vor
  `IsDynamicDim`, exakt das `VectorLenCheck`-Muster (vector.ts). Union-Dim auf einer Seite ⇒
  `CompatDim` → `Dim` (wide), `DimEq` → `true`.
- **D-V1.3 (`dim.ts` neu, sieben Konsumenten geändert):**
  `RankUnknowable<S> = IsDynamicRank<S> extends true ? true : IsUnion<S["length"]>`.
  Ersetzt das Gate GENAU an: `Broadcast` (broadcast.ts), `MatMul` (matmul.ts), `ReduceAxis` +
  `Transpose` (reduce.ts), `SliceShape` + `SliceSpecsGuard` (slice.ts), `DotCheck`
  (vector.ts). `IsDynamicRank` selbst unverändert. `LiteralShapeProduct` (slice-literal.ts,
  frozen) und `ReshapeCheck` (reshape.ts, eigene `IsUnion`-Maschinerie) bewusst NICHT
  angefasst.
- **D-V1.4 (`ndarray.ts`):** `Guard<Result, Actual>` tuple-wrapped:
  `[Result] extends [ShapeError<infer Message>] ? {...} : Actual`. Wird von `resident.ts`
  (WNDArray) type-only mitkonsumiert — **null resident.ts-Edits**, wie vorhergesagt.
- **D-V1.5:** `OkShape` unverändert (bestätigt korrekt für die Union-der-gültigen-Member-Policy).

Geänderte Dateien (Kern-Diffs):

| Datei | Änderung |
|---|---|
| `spike/src/dim.ts` | private `IsUnion`, neu `RankUnknowable`, Union-Filter in `CompatDim`/`DimEq` |
| `spike/src/broadcast.ts` | `Broadcast`-Gate: `IsDynamicRank` → `RankUnknowable` (2 Stellen) |
| `spike/src/matmul.ts` | `MatMul`-Gate: `IsDynamicRank` → `RankUnknowable` (2 Stellen) |
| `spike/src/reduce.ts` | `ReduceAxis`- UND `Transpose`-Gate: `IsDynamicRank` → `RankUnknowable` |
| `spike/src/slice.ts` | `SliceShape`- UND `SliceSpecsGuard`-Gate: `IsDynamicRank` → `RankUnknowable` |
| `spike/src/vector.ts` | `DotCheck`-Gate: `IsDynamicRank` → `RankUnknowable` (2 Stellen) |
| `spike/src/ndarray.ts` | `Guard` tuple-wrapped |
| `spike/tests/{broadcast,matmul,reduce,ndarray,vector,slice,reshape}.test-d.ts` | neue Pins (siehe Testplan unten) — **keine neuen Root-Korpus-Dateien** |

## Policy-Umsetzung je Entscheidung

| Input-Form | Policy | Mechanismus |
|---|---|---|
| Union-DIM (`2\|7`) | No-claim, Ergebnis-Dim wide | D-V1.2 (`CompatDim`/`DimEq`) |
| Shape-Union uniformen Rangs | Natürliche Distribution; uniform-fehlerhaft → kombinierte Message; gemischt → Union der validen Member | D-V1.4 (`Guard` tuple-wrapped) — `OkShape` unverändert liefert die Union der validen Member |
| Shape-Union gemischten Rangs (`S["length"]` Union) | Uniforme Degradation an JEDEM Rank-Gate, auch `Transpose` | D-V1.3 (`RankUnknowable`) |
| Union ganzer `NDArray<A>\|NDArray<B>`-Instanzen | Bereits von TS abgelehnt (Argument UND Empfänger) — Kontroll-Pin, kein Fix | unverändert, empirisch bestätigt (s. u.) |

## PRE-FIX-ROT-BEWEIS

**Methodik:** Alle Pins wurden zuerst mit einer Probing-Technik (`const p: ExpectedType = {}
as { readonly __marker: true };` — erzwingt, dass `tsc` den vollen aufgelösten Typ in der
Fehlermeldung druckt, `--noErrorTruncation`) gegen den ECHTEN Pre-Fix-Code ermittelt, um
Transkriptionsfehler bei komplexen distribuierten Typen (insb. der 4-fach
kreuzmultiplizierten kombinierten Message) auszuschließen. Danach wurden die finalen Pins in
den bestehenden `spike/tests/*.test-d.ts`-Dateien ergänzt und — mit dem Haupt-Tree
unangetastet — in einen frischen Scratch-Worktree von HEAD (`git worktree add`, nur die
Quelldateien blieben Pre-Fix, die sieben Testdateien wurden hineinkopiert) gestellt:

```
$ ./node_modules/.bin/tsc --noEmit
EXIT: 1  (20 Fehler)
```

Vollständiger Output (worktree, Pre-Fix-Quellen + Post-Fix-Pins):

```
spike/tests/broadcast.test-d.ts(57,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(58,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(59,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(60,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(66,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(71,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/broadcast.test-d.ts(102,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/matmul.test-d.ts(61,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/matmul.test-d.ts(66,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/ndarray.test-d.ts(188,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/ndarray.test-d.ts(224,39): error TS2741: Property '__shapeError' is missing in type 'WNDArray<[3, 4]>' but required in type '{ readonly __shapeError: "matmul: inner dimensions 3 and 3 do not match" | "matmul: inner dimensions 7 and 3 do not match"; }'.
spike/tests/ndarray.test-d.ts(225,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/ndarray.test-d.ts(232,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/reduce.test-d.ts(54,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/reduce.test-d.ts(55,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/slice.test-d.ts(459,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/slice.test-d.ts(460,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/slice.test-d.ts(464,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/vector.test-d.ts(165,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
spike/tests/vector.test-d.ts(166,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
```

`reshape.test-d.ts`: **0 Fehler** — die neuen Facette-(c)-Pins dort sind "already-safe"
(reshape/flatten liefen schon über `ReshapeCheck`s eigene `IsUnion`-Maschinerie korrekt),
by design unverändert von V1.

Nach der Implementierung (Haupt-Tree, alle sieben Quelldateien gefixt): derselbe Lauf →
**0 Fehler, exit 0**, für den kompletten Korpus (bestehend + neu).

### Rot-Beweis-Zusammenfassung je Facette

**Facette (a) — Union-DIMS (echte Bugs):**
- `DimEq<2|7,2>` Pre-Fix = `boolean` (nicht `true`) — RED. `CompatDim<2|7,2>` Pre-Fix =
  `2 | ShapeError<"dims 7 and 2 ...">` (nicht `number`) — RED.
- `add`: `Broadcast<readonly[2|7,3],[2,3]>` Pre-Fix = `[2,3] | ShapeError<"...">` (2
  Fehlerzweige + 1 valider Zweig — via `Guard`/`OkShape` KONFIDENT `[2,3]`, obwohl der
  Runtime-Wert `7` sein könnte und dann werfen würde) statt sauber `[number,3]` — RED.
- `matmul`, Kontraktions-Achse: `MatMul<[2,3|7],[3,4]>` Pre-Fix = `ShapeError<...>`
  (fälschlich ABGELEHNT, obwohl Member `3` passt — schlimmer als der Broadcast-Fall, eine
  konfident-falsche ABLEHNUNG statt eines konfident-falschen Accepts) statt `[2,4]` — RED.
- `matmul`, Batch-Dim (via `Broadcast`/`CompatDim`): `MatMul<[2|9,3,4],[2,4,5]>` Pre-Fix
  kreuzmultipliziert zu einer 2-Zweig-`ShapeError`-Union statt `[number,3,5]` — RED.

**Facette (b), korrigierte Form — Shape-Union IM Typparameter (uniformer Rang):**
- **Gemischt (ein Member kompatibel, einer nicht):** bereits PRE-FIX korrekt akzeptiert und
  konfident `NDArray<[2,3]>` (der alte, nicht-tuple-wrapped `Guard` distribuiert bereits
  passend zu `Actual | {err}`, und `OkShape` streift `ShapeError`-Zweige bereits distributiv
  ab) — **kein Rot-Beweis hier, GRÜN pre- und post-fix** (bewusst als Regressions-Pin
  belassen, nicht als Bugfix verkauft).
- **Uniform fehlerhaft (alle Member inkompatibel):** bereits PRE-FIX korrekt ABGELEHNT, aber
  mit einer STRUKTURELL schlechteren Diagnose — `Guard<Broadcast<[2,3],[9,3]|[7,3]>,...>`
  Pre-Fix = eine Union aus ZWEI separaten `{__shapeError:...}`-Objekttypen (je mit einer
  eigenen 2-fach-Message-Union) statt EINEM kombinierten Objekt mit der vollen 4-fach-Union
  — RED gegen den Post-Fix-Zieltyp (Test `UB2`, broadcast.test-d.ts:102).

**Facette (c) — Mixed-Rank-Shape-Union (der schwerste, "confidently wrong"-Fall):**
- `NDArray<[2,3]|[2,3,4]>.sum(2)`: Pre-Fix akzeptiert STILL und liefert KONFIDENT
  `NDArray<[2,3]>` (Kern-09-Befund, jetzt empirisch re-bestätigt) statt ehrlich
  `NDArray<readonly number[]>` — RED, der zentrale Fix dieser Scheibe.
- `ReduceAxis<[2,3]|[2,3,4],2>` (bare type): dieselbe Mechanik, RED.
- `Transpose<[2,3]|[2,3,4]>`: Pre-Fix bereits DISTRIBUTIV KORREKT (`[3,2]|[4,3,2]`, ein
  echtes Per-Member-Ergebnis, keine Lüge) — der Fix hier ist eine **bewusste,
  Owner-entschiedene Präzisions-Abgabe** für strukturelle Uniformität, kein Bugfix. RED nur
  gegen den NEUEN Zieltyp (`readonly number[]`), nicht weil Pre-Fix falsch war.
- `SliceShape<[2,3]|[2,3,4],[1]>`: ebenfalls Pre-Fix distributiv korrekt (`[3]|[3,4]`) —
  gleiche Präzisions-Abgabe wie Transpose, kein Bugfix.
- `SliceSpecsGuard<[2,3]|[2,3,4],[1,null,2]>`: **echter Bug** — Pre-Fix distribuiert die
  Arity-Prüfung pro Rang-Member; ein Aufruf mit 3 Specs (gültig nur für den Rang-3-Member)
  matched zufällig den validen Union-Zweig und wurde OHNE Fehler durchgelassen, obwohl der
  Rang-2-Member zu viele Specs hätte — RED gegen `[1,null,2]` (Post-Fix: `Specs` unverändert
  durchgereicht, aber aus dem EHRLICHEN Grund "Rang unbekannbar", nicht aus Zufall).
- `dot` (`DotCheck`, Empfänger UND Argument mixed-rank): die beiden BAREN Typ-Pins `UC1`/`UC2`
  (`vector.test-d.ts`) hatten einen ECHTEN Rot→Grün-Übergang, sichtbar oben im
  Rot-Beweis-Transkript als `vector.test-d.ts(165,19)`/`(166,19)`. Pre-Fix distribuiert
  `DotCheckStatic` natürlich über den Mixed-Rank-Empfänger/-Argument: ein Rang-Member matcht die
  rank-1-Destrukturierung (`Pass`/`true`), der andere nicht (`ShapeError<...>`) — das Ergebnis ist
  die UNION `true | ShapeError<...>`, die gegen den gepinnten Zieltyp `true` als `Type 'false'
  does not satisfy the constraint 'true'` RED steht (kein bloßer Zieltyp-Wechsel wie bei
  Transpose/SliceShape unten — der Pre-Fix-Typ war buchstäblich eine ANDERE, unsauberere Struktur
  als das gepinnte `true`). Post-Fix degradiert `RankUnknowable` das GANZE `DotCheck` uniform zu
  `true` (No-Claim), BEVOR überhaupt distribuiert wird — ein echter Fix an der Verdikt-Struktur,
  nicht nur eine Owner-entschiedene Präzisions-Abgabe. Nur die CALL-SITE-Pin `UC3`
  (`.dot()` mit mixed-rank Empfänger akzeptiert, Ergebnistyp bleibt `number`) war BEIDSEITIG grün:
  `dot`s Rückgabetyp ist ohnehin nie shape-abhängig (Kern 07: scalar-Ops geben plain `number`
  zurück), also gab es dort nie einen konfident-falschen SHAPE-Claim zu vermeiden — anders als bei
  `.sum(2)` oben (Facette c, `UC1` in ndarray.test-d.ts), wo der Call-Site-Rückgabetyp selbst der
  Bug war.

**Kontroll-Pins (Instanzen-Union, NICHT Teil des Fixes):** empirisch verifiziert (separat,
gegen den unveränderten Pre-Fix-Code UND erneut gegen den Post-Fix-Code): sowohl
`base.add(instanceUnionArg)` (Argument) als auch `instanceUnionArg.add(base)` (Empfänger) mit
`instanceUnionArg: NDArray<[2,3]>|NDArray<[7,3]>` werden von TS abgelehnt — unabhängig von
D-V1.1–V1.4, identisch vor und nach dem Fix. Bestätigt den Baustein-0-Befund.

## Testplan — Pflicht-Pins (alle grün, post-fix)

Alle in **bestehenden** `spike/tests/*.test-d.ts`-Dateien ergänzt (keine neuen Root-Korpus-
Dateien — Pin-Bewegungen sind damit echte Typkosten, kein Order-Noise durch Dateizahl-Änderung):

| Pin | Datei | Ergebnis |
|---|---|---|
| `DimEq<2\|7,2>` → `true` | broadcast.test-d.ts (`UA1`) | ✓ |
| `CompatDim<2\|7,2>` → `number` | broadcast.test-d.ts (`UA2`) | ✓ |
| add/matmul mit Union-Dim akzeptiert, Ergebnis-Dim wide | broadcast/matmul.test-d.ts | ✓ |
| Facette (b) korrigiert: gemischt akzeptiert, Union der validen Member | broadcast.test-d.ts (`UB1`) | ✓ |
| Facette (b): alle Member inkompatibel → kombinierte Message am Argument | broadcast.test-d.ts (`UB2`) | ✓ |
| Kontroll-Pin: Instanzen-Union als Argument UND Empfänger abgelehnt | ndarray.test-d.ts | ✓ |
| Facette (c): `.sum(2)` → `NDArray<readonly number[]>` | ndarray.test-d.ts (`UC1`) | ✓ |
| Mixed-Rank-Degradation Transpose/slice/dot via `RankUnknowable` | reduce/slice/vector.test-d.ts | ✓ |
| `reshape`: eigene `IsUnion`-Maschinerie, `.reshape([6])` → konfident `NDArray<[6]>` | reshape.test-d.ts (`UC1`) | ✓ |
| `flatten()`-Ergebnis-Union `NDArray<[6\|24]>` ("already-safe") | reshape.test-d.ts (`UC2`) | ✓ |
| WNDArray-Seite: add/matmul/sum, dieselben Facetten | ndarray.test-d.ts (`UW1`–`UW3`) | ✓ |
| WNDArray-Seite: Facette (b) uniform-fehlerhaft → kombinierte Message (B-F4-Schließung, Post-Verification) | ndarray.test-d.ts (`UW4`) | ✓ |

## Re-expressierte Alt-Pins

**Keine.** Der volle bestehende Test-Korpus kompiliert nach dem Fix mit **0 Diagnosen**, VOR
dem Hinzufügen der neuen V1-Pins (separat geprüft: Quell-Fix allein, ohne neue Tests → `tsc
--noEmit` exit 0). Kein bestehender Pin musste umformuliert werden — anders als die
Spike-06-Lehre, auf die die Spec vorsorglich hinwies. Dies ist plausibel, weil V1 an KEINEM
bestehenden Gate eine bereits GEPRÜFTE Grenze verschiebt, sondern ausschließlich UNGEPRÜFTE
(bislang stillschweigend distributiv durchgereichte) Fälle neu behandelt.

## Gates (echte Läufe, Haupt-Tree, post-fix)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Dreier-Verbund root+stress+browser) | clean, exit 0 |
| `pnpm test:core` | 818/818, exit 0 |
| `pnpm test:resident` | 4279 total (4277 pass, 2 skip), exit 0 |
| `pnpm test:browser` | 4/4, exit 0 |
| `cargo test` | 161/161, exit 0 |
| `pnpm demo` | „TS, WASM v1, and WASM resident all agree on every showcase op", exit 0 |
| Artefakt-Hash (`build:wasm`, CLEAN Rebuild via `cargo clean` davor) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — byte-identisch zum Pin |
| `pnpm bench:editor` | Overall hard gate: **PASS** — warm-hover Median max 0,09 ms (Gate ≤1 ms), Toggle-Median max 3,29 ms (Gate ≤10 ms) |

## Budget-Gates

**Baseline-Kontrolle** (frischer `git worktree add` von HEAD `e0feee9`, `node_modules` per
Symlink, `tsc` direkt):

| Messung | Wert | Erwartung (Task) | Status |
|---|---|---|---|
| `check:diag` (root) | 175.712 @ 132 Dateien, exit 0 | 175.712 @ 132 | exakt reproduziert |
| `check:diag:stress` | 103.882 @ 82 Dateien, exit 0 | 103.882 @ 82 | exakt reproduziert |
| `check:diag:browser` | 2.142 @ 75 Dateien, exit 0 | (kein Pin in der Task vorgegeben) | notiert |

**End-Stand** (Haupt-Tree, uncommitted; `git status` vorher gezeigt — 7 Quell- + 7
Testdateien modifiziert, sonst clean):

| Messung | Vorher | Nachher | Δ | Dateizahl | Hart-Gate | Status |
|---|---|---|---|---|---|---|
| `check:diag` (root) | 175.712 | **180.485** | **+4.773** | 132 → 132 (unverändert) | ≤ 225.000 | PASS, weit unter Soft-Erwartung (≲+20.000) |
| `check:diag:stress` | 103.882 | **103.511** | **−371** | 82 → 82 (unverändert) | kein Gate | siehe Honesty-Residuum |
| `check:diag:browser` | 2.142 | **2.142** | **0** | 75 → 75 (unverändert) | kein Gate | exakt unverändert |

Die Dateizahl blieb in ALLEN drei Korpora exakt gleich (V1 fügt bewusst keine neuen
Root-Korpus-Dateien hinzu) — die CLAUDE.md-dokumentierte Order-Noise-Quelle (Datei-Hinzufügen
verschiebt den Fresh-vs-Cached-Split der Instantiation-Memoisierung um bis zu ±2.000) greift
hier NICHT als Erklärung; das `check:diag`-Delta von +4.773 ist mit hoher Wahrscheinlichkeit
echte Typkosten der neuen Maschinerie (`RankUnknowable`, die private `IsUnion`-Kopie, die
Union-Filter in `CompatDim`/`DimEq`, die tuple-wrapped `Guard`) plus der ~230 neuen Testzeilen
über sieben Dateien.

**Instantiations-Vergleich via `bench:editor`** (deterministische, isolierte Pro-Workload-Projekte):

| Workload | Vorher | Nachher | Δ |
|---|---|---|---|
| w1 (chains) | 25.229 | 24.918 | −311 |
| w2 (broadcast) | 26.867 | 26.723 | −144 |
| w3 (slice) | 57.738 | 57.855 | +117 |
| w4 (errors) | 25.257 | 25.067 | −190 |
| w5 (mixed) | 30.583 | 30.372 | −211 |
| w6 (reshape/flatten) | 31.906 | 31.528 | −378 |

Alle sechs Editor-Workloads bleiben **in-family** (Deltas im niedrigen Hundert-Bereich, ein
Workload sogar leicht GÜNSTIGER als vorher) — realistische Editor-Sites treffen die
Mixed-Rank-/Union-Dim-Pfade praktisch nie, daher kein spürbarer Editor-Latenz-Effekt trotz des
sichtbaren Root-Korpus-Deltas.

## Honesty-Residuum

- **check:diag-Delta größer als die Baustein-0-Grobmessung:** Baustein 0 maß für den
  KOMBINIERTEN V1+V2-Probepatch nur Δ+1.036 (176.670 @ 132). Die tatsächliche, isolierte
  V1-Implementierung (ohne V2) misst Δ+4.773 — mehr als das 4-fache der Grobschätzung für
  BEIDE Scheiben zusammen. Mögliche Gründe (nicht einzeln isoliert/bewiesen): der
  Probe-Patch war vermutlich schlanker als die volle Testabdeckung dieser Scheibe (die
  Pflicht-Pins allein sind ~230 neue Zeilen über 7 Dateien, inkl. mehrerer
  Guard-/Broadcast-Ausdrücke mit UNION-Typargumenten, die selbst Instantiations-Kosten
  erzeugen); die Grobmessung war explizit als "Baustein-0-Grobmessung", nicht als
  Scheiben-Vorhersage deklariert. Beide Werte liegen weit unter dem harten Gate (225.000)
  und auch unter der Soft-Erwartung (+20.000) — kein Blocker, aber eine ehrlich zu
  berichtende Diskrepanz.
- **check:diag:stress-Rückgang (−371) nicht separat root-caused:** Die Stress-Strecke
  importiert `dim.ts` (transitiv über Digit-Arithmetik-nahe Module) und ist damit von den
  dim.ts-Änderungen betroffen, obwohl sie selbst KEINE neue Datei bekam. Ein Rückgang trotz
  reiner Inhaltsänderung (nicht Dateizahl-Änderung) ist mit dem in Kern 11 gepinnten
  Mechanismus ("der Instantiation-Zähler ist CHECK-ORDER-abhängig") VEREINBAR, aber nicht
  eigenständig neu bewiesen — hier als plausible, nicht als bewiesene Erklärung berichtet.
- **Kreuzmultiplizierter ShapeError-Text (Kern-09 Nit 3) bleibt bestehen:** D-V1.4 räumt die
  STRUKTUR auf (ein kombiniertes Objekt statt mehrerer), der MESSAGE-INHALT nennt weiterhin
  Shape/Dim-Paare, die für kein einzelnes Union-Member tatsächlich zutreffen (siehe `UB2`s
  4-fach-Message-Union für nur 2 tatsächlich unterschiedliche Fehlerursachen). Spec-konform
  dokumentiert, nicht behoben.
- **Transpose/SliceShape/dot-Facette-(c)-Pins sind KEINE Bugfixes, sondern Präzisions-Abgaben:**
  explizit im Rot-Beweis oben markiert — Pre-Fix waren diese drei bereits distributiv/gradual
  korrekt; die uniforme Degradation ist eine bewusste, Owner-entschiedene strukturelle
  Vereinfachung (Spec, "Begründung der Uniform-Degradation"), kein Fix eines Bugs.
- **`test:threaded` nicht gelaufen:** nicht Teil der pre-registrierten V1-Gate-Liste
  (resident.ts/threaded.ts sind unberührt, Risiko strukturell null), aber nicht in dieser
  Session verifiziert — falls gewünscht, ein weiterer schneller Lauf.
- **WNDArray-Abdeckung:** wie von der Spec verlangt ("mindestens add/matmul/sum") — sub/mul/
  div/dot/transpose/slice/reshape wurden NICHT zusätzlich auf der WNDArray-Seite
  facettenspezifisch gepinnt (sie teilen dieselbe importierte Maschinerie, aber das ist hier
  nicht per WNDArray-Assertion nochmal einzeln bewiesen).

## FOLLOWUPS.md — Korrektur-Textvorschlag (NICHT angewendet, s. Arbeitsregeln)

Ersetzt den bestehenden offenen Eintrag (aktuell `- [ ]` in FOLLOWUPS.md, Zeile 50, "Vorbestehende
Union-Latenz in den Typlevel-Guards, ZWEI Facetten..."):

```markdown
- [x] Vorbestehende Union-Latenz in den Typlevel-Guards, DREI Facetten — erledigt & verifiziert
  als Phase-D V1 (docs/phase-d-vorarbeiten-v1-ergebnisse.md; Spec:
  docs/phase-d-vorarbeiten-spec.md). (a) Union-DIMS (`DimEq`/`CompatDim`, dim.ts): gefixt via
  Union-Filter als erste Prüfung (D-V1.2), Muster `VectorLenCheck` (vector.ts) verallgemeinert
  in dim.ts selbst; `DimEq<2|7,2>` → `true`, `CompatDim<2|7,2>` → wide `number`, propagiert in
  `Broadcast`/`MatMul` inkl. matmul's Kontraktionsachse (dort war der Pre-Fix-Bug eine
  konfident-falsche ABLEHNUNG, nicht nur ein konfident-falsches Accept). (b) Union GANZER
  Operanden-TYPEN (`NDArray<A>|NDArray<B>` als Instanz): der ursprünglich hier dokumentierte
  Kern-07-Addendum-1-Repro (ein solcher Operand als Argument unterläuft den Guard) ist auf dem
  heutigen Stand (TS 7.0.2, geprüft in Baustein 0 UND erneut hier: 4 Varianten, Argument- UND
  Empfänger-Form) NICHT nachvollziehbar — TS' eigene generische Inferenz/Klassen-Invarianz lehnt
  diese Form bereits ab, unabhängig von Guard/CompatDim/Broadcast (ehrliche Diskrepanz-Notiz,
  keine stillschweigende Umdeutung: entweder hat sich TS' Inferenzverhalten seit der
  Kern-07-Beobachtung geändert, oder die ursprüngliche Beobachtung war an einer anderen
  Formulierung als der hier reproduzierten festgemacht — nicht mehr rekonstruierbar). Die
  TATSÄCHLICH reproduzierbare Form ist eine Shape-Union IM Typparameter EINER Instanz
  (`x: NDArray<[2,3]|[7,3]>`), was mechanisch mit (c) zusammenfällt und von der D-V1.4-Guard-
  Härtung (tuple-wrapped `[Result] extends [ShapeError<infer Message>]`) abgedeckt wird: gemischt
  (ein Member kompatibel) → akzeptiert, Ergebnis = Union der validen Member (war bereits
  pre-fix korrekt); uniform-fehlerhaft (alle Member inkompatibel) → abgelehnt mit EINER
  kombinierten Message (M1|M2|...) statt der pre-fix mehreren separaten Fehlerobjekten (Struktur
  bereinigt, Message-Kreuzmultiplikation — Kern-09 Nit 3 — bleibt dokumentiert, nicht behoben).
  (c) Mixed-Rank-Shape-Union im Typparameter (`NDArray<[2,3]|[2,3,4]>.sum(2)` → konfident
  `NDArray<[2,3]>`): gefixt via neues `RankUnknowable<S>` (dim.ts) — degradiert JEDES Rank-Gate
  (Broadcast/MatMul/ReduceAxis/Transpose/SliceShape/SliceSpecsGuard/DotCheck) uniform auf
  `readonly Dim[]`/no-claim, sobald `S["length"]` selbst eine echte Union ist. Owner-entschiedene
  Policy: UNIFORME Degradation auch dort, wo die alte distributive Auswertung bereits korrekt
  war (Transpose, SliceShape, dot) — ein bewusster, offengelegter Präzisionsverlust für
  strukturelle Einfachheit, kein Bugfix an diesen drei Stellen. `SliceSpecsGuard` hatte
  zusätzlich einen ECHTEN Arity-Leak (ein Aufruf mit zu vielen Specs für EINEN Rang-Union-Member
  wurde durchgelassen, wenn er zufällig zum ANDEREN Member passte) — ebenfalls durch
  `RankUnknowable` geschlossen. `reshape`/`flatten` waren NIE betroffen (eigene
  `IsUnion`-Produktfilter-Maschinerie in reshape.ts, unverändert, "already-safe"-Pins ergänzt).
  Alle drei Facetten jetzt auf BEIDEN Surfaces (NDArray + WNDArray) gepinnt.
```

## KB-Capture (Kandidat für coding-kb, noch nicht durchgeführt)

Diese Session hat keine coding-kb-Schreibzugriffe durchgeführt (die verfügbaren
`mcp__coding-kb__*`-Tools in dieser Umgebung sind read-only: `find`/`neighbors`/`read`/etc. —
kein `write`/`upsert`). Vorschlag für die manuelle Erfassung (nächste Session mit
KB-Schreibzugriff, oder Owner direkt):

**Lektion (generalisierbar):** "Distributive Conditional Types sind der Leak-Punkt für
Union-Verdikte in Guard-artigen Typ-Pipelines." Ein `Guard<Result, Actual>`, das `Result`
NACKT prüft (`Result extends ErrorBrand<...> ? ... : Actual`), lässt bei einem UNION-`Result`
jeden validen Zweig einzeln durch — auch wenn ANDERE Zweige desselben Results Fehler sind. Der
Fix ist immer derselbe: `Result` TUPLE-WRAPPEN (`[Result] extends [ErrorBrand<...>]`), um die
Prüfung auf das GANZE Result zu erzwingen, statt es zu distribuieren. Zusätzlich: ein
Rank-/Dynamik-Gate, das nur die "wide"-Form prüft (`number extends S["length"]`), erkennt eine
ECHTE Union von Rang-Literalen (`2|3`) NICHT als "unbekannt" — `number extends (2|3)` ist
`false`. Ein zweites Kriterium (`IsUnion<S["length"]>`) ist nötig, um Mixed-Rank-Unions
strukturell VOR jeder tuple-destrukturierenden Rekursion abzufangen, statt den Leak an jeder
einzelnen distributiven Hilfsfunktion (die S naked in der eigenen Definition referenziert)
einzeln zu jagen.

---

## Post-Verification-Addendum (Zwei-Verifier-Runde: Baustein A + Baustein B, 2026-07-13)

Auftrag aus docs/verify-runde-template.md §Baustein A / §Baustein B, je ein frischer
`brainroute:deep`-Kontext: Baustein A prüft gegen die Spec (`docs/phase-d-vorarbeiten-spec.md`,
Scheibe V1), Baustein B arbeitet adversarial gegen den Diff.

**Baustein A (Spec-Konformität): CONFIRMED, mit Auflagen.**
- Der PRE-FIX-ROT-BEWEIS-Transkript (20 Fehler über die sieben betroffenen Testdateien) wurde in
  einem eigenen frischen Scratch-Worktree byte-identisch reproduziert.
- `test:threaded` — in der Erst-Session als offenes Honesty-Residuum benannt ("nicht in dieser
  Session verifiziert") — wurde nachgeholt: **69/69 grün**. Bestätigt empirisch, nicht nur
  argumentativ, dass resident.ts/threaded.ts von V1 strukturell unberührt bleiben.
- Budget-Diskrepanz isoliert (das Erst-Session-Honesty-Residuum "check:diag-Delta größer als die
  Baustein-0-Grobmessung"): die SRC-ONLY-Änderung (die fünf D-V1.1–V1.5-Quelldateien, ohne die
  neuen Testpins) trägt **Δ+1.060** — nah an der Baustein-0-Grobmessung (Δ+1.036 für den
  KOMBINIERTEN V1+V2-Probepatch). Der Rest des berichteten Δ+4.773, also **+3.713**, entfällt auf
  die ~230 neuen Pin-Zeilen selbst (mehrere Guard-/Broadcast-Ausdrücke mit UNION-Typargumenten
  sind selbst instantiation-teuer). Die Erst-Session-Vermutung ("die Grobmessung war vermutlich
  schlanker als die volle Testabdeckung dieser Scheibe") ist damit isoliert bestätigt, nicht mehr
  nur plausibel.
- Auflagen aus dieser Runde, in dieser Session geschlossen (s. "In-Slice-Schließungen" unten):
  B-F4 (fehlende WNDArray-Pin für UB2s Unterfall) sowie die beiden Doku-Nits A-1 (dot-Bullet) und
  A-4 (stale Kommentar in vector.ts).

**Baustein B (adversarial): HÄLT, mit Befunden, kein Blocker.**
- 5/5 selbst gebaute Mutanten (breit statt tief, an Stellen jenseits der Spec) wurden von der
  bestehenden + neuen Testabdeckung gefangen.
- Über-Degradations-Check: die 163 Alt-Korpus-Pins, die nicht Teil der V1-Änderung sind, bleiben
  durch die `RankUnknowable`-Umstellung geschützt — kein Kollateralschaden an bereits gepinnten
  Uniform-Rang-/Einzel-Shape-Fällen.
- Die sieben Typ-Kanten A1–A7 (die sieben `RankUnknowable`-Konsumenten aus D-V1.3: Broadcast,
  MatMul, ReduceAxis, Transpose, SliceShape, SliceSpecsGuard, DotCheck) wurden einzeln geprüft.
- Hover-Qualität der neuen/geänderten Signaturen über einen echten LSP-Harness (nicht nur
  `tsc`-Diagnostik) sauber.
- Befunde, keiner Blocker-Stufe: **B-F4** (minor, WNDArray-Pin-Lücke — geschlossen, s. u.),
  **B-F1/B-F2/B-F3** (vorbestehend, außerhalb des V1-Scopes — als FOLLOWUPS-Textvorschläge unten
  dokumentiert, nicht in dieser Scheibe gefixt).

## In-Slice-Schließungen (2026-07-13, nach der Verify-Runde)

Drei kleine Schließungen umgesetzt, ohne die Spec-Substanz zu ändern; alles TS-only, `tsc
--noEmit` bleibt exit 0 nach jeder einzelnen Änderung:

1. **B-F4 (minor) — WNDArray-Pin für UB2s "uniform-fehlerhaft → kombinierte Message"
   Unterfall.** Neuer Pin `UW4` in `spike/tests/ndarray.test-d.ts` (direkt nach `UW1`, dessen
   Konstruktion und Assertions-Idiom von broadcast.test-d.ts' `UB2` gespiegelt): bare-Typ-Assertion
   `Guard<Broadcast<[2,3],[9,3]|[7,3]>, WNDArray<[9,3]|[7,3]>>` gegen die kombinierte
   Fehler-Message (lokal als `AllBadMsgW` benannt, gleicher Inhalt wie UB2s `AllBadMsg`). Die
   exakte Message wurde EMPIRISCH per Marker-Probe-Technik extrahiert (Scratch-Probe außerhalb des
   Haupt-Trees, `tsc --ignoreConfig --noErrorTruncation` gegen die echten
   Guard/Broadcast/WNDArray-Quellen), nicht geraten — sie kam byte-identisch zu UB2s `AllBadMsg`
   zurück (bestätigt: `Guard`s Fehlerzweig hängt nur von `Result` = `Broadcast<S,B>` ab, nie von
   `Actual`, also ist der Message-Inhalt für NDArray- und WNDArray-Empfänger identisch). Imports
   von `Broadcast` und `Guard` in ndarray.test-d.ts ergänzt.

   **Catchability-Nachweis:** `Guard` in `spike/src/ndarray.ts` testweise auf die Pre-Fix-
   distributive Form zurückgesetzt (`Result extends ShapeError<infer Message> ? ... : Actual`,
   ohne Tuple-Wrap):
   ```
   spike/tests/broadcast.test-d.ts(102,19): error TS2344: Type 'false' does not satisfy the constraint 'true'.
   spike/tests/ndarray.test-d.ts(236,3): error TS2344: Type 'false' does not satisfy the constraint 'true'.
   EXIT: 1
   ```
   Exakt ZWEI Fehler — `UB2` UND die neue `UW4` — beide TS2344, genau wie gefordert
   ("mindestens ZWEI Pins rot"). Sofort revertiert; `diff` gegen eine vor dem Mutanten gezogene
   Kopie zeigt 0 Unterschiede (identische MD5 `a662703a3facadc487ec4f5094c21bd0` vor und nach),
   `tsc --noEmit` danach wieder exit 0.

2. **A-1 (moderat, Doku) — der `dot`-Bullet widersprach dem eigenen Rot-Beweis-Transkript.**
   Der Bullet in "Rot-Beweis-Zusammenfassung je Facette" → Facette (c) behauptete fälschlich "kein
   Rot-Beweis nötig, GRÜN pre- und post-fix" für `dot`. Korrigiert: die BAREN Typ-Pins `UC1`/`UC2`
   hatten einen echten Rot→Grün-Übergang — Pre-Fix `true | ShapeError<...>` (natürliche
   `DotCheckStatic`-Distribution über den Mixed-Rank-Empfänger/-Argument: ein Rang-Member matcht
   die rank-1-Destrukturierung, der andere nicht), gegen den gepinnten Zieltyp `true` RED —
   sichtbar im Transkript als `vector.test-d.ts(165,19)`/`(166,19)`; Post-Fix degradiert
   `RankUnknowable` uniform VOR jeder Distribution zu `true`. Nur die CALL-SITE-Pin `UC3`
   (`.dot()` mit mixed-rank Empfänger akzeptiert, Ergebnis bleibt `number`) war beidseitig grün —
   weil `dot`s Rückgabetyp nie shape-abhängig ist (Kern 07: scalar-Ops geben plain `number`
   zurück), gab es dort nie einen konfident-falschen SHAPE-Claim zu vermeiden, anders als bei
   `.sum(2)` oben (Facette c, `UC1` in ndarray.test-d.ts).

3. **A-4 (nit) — veralteter Datei-Header-Kommentar in `spike/src/vector.ts` (Zeilen ~10–13).**
   Beschrieb `DotCheck`s Gate weiterhin als "the `IsDynamicRank` guard"; tatsächlich ist es seit
   D-V1.3 `RankUnknowable` (dynamischer Rang ODER Mixed-Rank-Union). Kommentar-only-Korrektur,
   referenziert jetzt `RankUnknowable`/D-V1.3/die Spec — konsistent mit dem bereits korrekten
   lokalen Kommentar direkt über `DotCheck` (Zeilen ~75–83, unverändert gelassen).

**Nicht in dieser Scheibe geschlossen — FOLLOWUPS-Textvorschläge (Baustein B, kein Blocker, NICHT
in FOLLOWUPS.md angewendet, s. Arbeitsregeln dieser Session):**

- **B-F2 (release-relevant — eigene kleine Scheibe vor Item 11 empfohlen):** Union im
  AXIS-Parameter — `arr23.sum(0 as 0|2)` kompiliert und hovert konfident `NDArray<[3]>`, wirft
  aber für Achse 2 zur Laufzeit (außerhalb des Bereichs für eine `[2,3]`-Form). Vorbestehend
  (nicht durch V1 eingeführt oder verschlimmert); Mechanik: `ReduceAxis`s Axis-Parameter
  distribuiert weiterhin naked über die Achsen-Union, `OkShape` streift den Fehlerzweig
  distributiv ab — dieselbe confidently-wrong-Klasse wie die jetzt gefixte Facette (c), nur am
  AXIS- statt am SHAPE-Parameter.
- **B-F1 (niedrig, pathologisch):** `never`-Dims/-Shapes liefern `never`-Verdikte statt einer
  Degradation zu No-Claim. Vorbestehend, annotation-only (kein Runtime-Wert kann `never` sein),
  niedrige Priorität.
- **B-F3 (Kosmetik, gleiche Klasse wie Kern-09 Nit 3):** bei einer Specs-Tupel-Union
  (`slice`-Argumente) kreuzmultipliziert die Arity-Fehlermeldung ähnlich wie D-V1.4s
  `AllBadMsg`-Kreuzmultiplikation — kosmetisch, nicht strukturell falsch.

## Gates nach den Schließungen (echte Läufe, Haupt-Tree, 2026-07-13)

Alle Gates ein zweites Mal frisch gelaufen, NACH den drei In-Slice-Schließungen (die
Doku-Änderungen A-1/A-4-Kommentar sind TS-seitig entweder nicht vorhanden (Doku) oder
Kommentar-only; einzig B-F4s neuer Pin `UW4` + Imports in ndarray.test-d.ts ist eine echte
TS-Änderung):

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Dreier-Verbund root+stress+browser) | clean, exit 0 |
| `pnpm test:core` | 818/818, exit 0 |
| `pnpm test:resident` | 4279 total (4277 pass, 2 skip), exit 0 |
| `pnpm test:browser` | 4/4, exit 0 |
| `pnpm test:threaded` | 69/69, exit 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161 lib-tests passed (Pin unverändert) + 1 zero_alloc-Integrationstest + 0 Doc-Tests — alle grün, exit 0 |
| `pnpm demo` | „TS, WASM v1, and WASM resident all agree on every showcase op", exit 0 |
| Artefakt-Hash (`cargo clean` + `pnpm build:wasm`, echt sauberer Rebuild) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — byte-identisch zum Pin (erwartet: die einzige Rust-nahe Datei-Berührung dieser Schließungsrunde ist der Kommentar in vector.ts, ein reines TS-Quellfile — ZERO Rust) |

**Pins, neu gemessen (direkt im Haupt-Tree, uncommitted, `git status` oben gezeigt — kein
Commit-/Stash-Wechsel involviert, daher kein Worktree nötig für diese Messung des
AKTUELLEN Standes):**

| Messung | Vor den Schließungen (End-Stand oben) | Nach den Schließungen | Δ | Dateizahl | Status |
|---|---|---|---|---|---|
| `check:diag` (root) | 180.485 @ 132 | **180.794 @ 132** | **+309** | unverändert (132) | reale Typkosten des neuen `UW4`-Pins (ein weiterer Guard/Broadcast-Ausdruck mit UNION-Typargumenten — dieselbe teure Klasse wie die übrigen V1-Pins, s. Honesty-Residuum oben) |
| `check:diag:stress` | 103.511 @ 82 | **103.511 @ 82** | **0** | unverändert (82) | exakt unverändert — B-F4 berührt die Stress-Strecke nicht |
| `check:diag:browser` | 2.142 @ 75 | **2.142 @ 75** | **0** | unverändert (75) | exakt unverändert — B-F4 berührt die Browser-Strecke nicht |

Die Bewegung bleibt, wie in der Aufgabenstellung erwartet, klein und plausibel: EINE neue
bare-Typ-Assertion mit zwei Union-Shape-Typargumenten (`Guard<Broadcast<[2,3],[9,3]|[7,3]>,
WNDArray<[9,3]|[7,3]>>`) plus ein vierstelliger String-Literal-Union-Typalias (`AllBadMsgW`)
plus zwei neue Imports — Δ+309 liegt in derselben Größenordnung wie die übrigen ~230
Pflicht-Pin-Zeilen dieser Scheibe (Δ+4.773 für den gesamten V1-Testplan), weit unter dem
harten Budget-Gate (≤ 225.000) und der Soft-Erwartung (≲ +20.000). `check:diag:stress` und
`check:diag:browser` bleiben exakt unverändert, weil B-F4 ausschließlich eine bestehende
Root-Korpus-Datei (`ndarray.test-d.ts`) inhaltlich erweitert, ohne die Dateizahl in
IRGENDEINEM der drei Korpora zu ändern.
