# Phase-D-Vorarbeiten — bindende Spec (V1–V3)

Stand: 2026-07-12, nach Baustein-0-Verifikation (Addendum am Dokumentende;
der Facette-(b)-Blocker ist eingearbeitet). Owner-Richtung (Chat,
2026-07-12): vor dem Paketschnitt (Roadmap Item 11) fallen drei Vorab-Scheiben —
**V1 Union-Guard fixen**, **V2 strides/NDArrayView/readonly-shape harmonisieren**,
**V3 Browser-Smoke-Test als Frühtest**. Diese Spec bindet alle drei; jede Scheibe
durchläuft einzeln die volle Hausdisziplin (Implementierung → Zwei-Verifier-Runde
nach docs/verify-runde-template.md → Ergebnisdoc → KB-Capture → Commit).

## Gemeinsame Rahmenbedingungen (alle drei Scheiben)

- **Zero Rust.** Keine Scheibe fasst `crates/` an. Freeze-Beweis in der starken
  Form: Artefakt-Hash nach `pnpm build:wasm` byte-identisch zum Pin
  `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`.
- **Pins:** `check:diag` (Haupt 175'634 @ 132) / `check:diag:stress` (103'882 @ 82)
  werden je Scheibe im frischen `git worktree` neu gemessen (Mess-Hausregel).
  Order-Noise-Regel beachten: Scheiben, die dem ROOT-Korpus Dateien hinzufügen,
  tragen ±~2'000 Rauschen — V1/V2 fügen deshalb KEINE neuen Root-Korpus-Dateien
  hinzu (neue Typtests werden in BESTEHENDE `spike/tests/*.test-d.ts` eingefügt);
  V3 hält seine Dateien per tsconfig-Exclude ganz aus dem Root-Korpus heraus.
- **Reihenfolge der Umsetzung: V3 → V1 → V2.** V3 ist der billigste Slice und der
  Frühtest mit dem größten Informationswert für Item 11 (findet er ein Problem,
  wollen wir es VOR dem API-Schnitt wissen); V1 und V2 berühren beide die
  Typ-Surface in `ndarray.ts` und laufen deshalb strikt sequenziell, V1 zuerst
  (V2 baut auf den dann festgezurrten Guard-Semantiken auf).
- **Ausgetragene FOLLOWUPS bei Landung:** V1 → Union-Guard-Item (Facetten a/b/c);
  V2 → `NDArrayView`-Ausdehnungs-Item + deep-readonly-`shape`-Item; V3 →
  Browser-Smoke-Test-Item.

---

## Scheibe V1 — Union-Guard-Fix (FOLLOWUPS-Facetten a/b/c)

### Problem, am Code verankert

Wurzelmechanismus aller drei Facetten: **`Guard<Result, Actual>`
(spike/src/ndarray.ts:64) ist eine distributive Conditional.** Ein Union-`Result`
(z. B. `[2,3] | ShapeError<…>`) distribuiert zu
`{ __shapeError } | Actual` — jedes `Actual`-typisierte Argument ist dem
Union-Member `Actual` zuweisbar, das Fehler-Verdikt entweicht. `OkShape`
(ndarray.ts:58) streift anschließend die Fehler-Member ab und liefert einen
konfidenten Ergebnistyp.

- **Facette (a) — Union-DIMS:** `CompatDim`/`DimEq` (spike/src/dim.ts:54/73)
  haben keinen Union-Filter; `DimEq<2|7, 2>` distribuiert zu `boolean`,
  `CompatDim<2|7, 2>` zu `2 | ShapeError<…>`. Union-Verdikte wandern ungefiltert
  durch `BroadcastAcc`/`MatMulStatic` (inkl. distributiver Rekursion — auch ein
  Instantiierungs-Risiko). `vector.ts` (Kern 07) filtert bereits via `IsUnion`
  (`VectorLenCheck`, vector.ts:46) — das ist das Vorbild-Muster.
- **Facette (b) — Union ganzer Operanden (Baustein-0-KORRIGIERT):** Die
  wörtliche Form `NDArray<[2,3]> | NDArray<[7,3]>` als Argument reproduziert
  die behauptete Guard-Umgehung NICHT — sie wird bereits am IST-Stand von TSs
  eigener generischer Inferenz/Klassen-Invarianz abgelehnt (Baustein 0,
  empirisch, alle 4 Varianten inkl. beide-kompatibel; unabhängig von
  Guard/CompatDim/Broadcast, unverändert durch D-V1.1–V1.4). Der TATSÄCHLICH
  reproduzierbare Leak ist die Shape-Union IM Typparameter EINER Instanz als
  Argument: `base.add(x)` mit `x: NDArray<[2,3]|[7,3]>` typechecked und liefert
  konfident `NDArray<[2,3]>`. Damit fällt (b) mechanisch mit (c) zusammen
  (Shape-Union im Parameter) und wird von Policy-Zeile 2 + D-V1.4 abgedeckt.
  Der Kern-07-Addendum-1-Wortlaut (docs/kern-07-ergebnisse.md) ist in der
  Instanzen-Union-Form auf dem heutigen Stand (TS 7.0.2) nicht nachvollziehbar —
  V1 korrigiert das FOLLOWUPS-Item entsprechend (ehrliche Diskrepanz-Notiz,
  keine stillschweigende Umdeutung).
- **Facette (c) — Mixed-Rank-Shape-Union IM Typparameter:**
  `NDArray<[2,3] | [2,3,4]>` akzeptiert `.sum(2)` still und liefert konfident
  `NDArray<[2,3]>` (Kern-09-Verifier, empirisch; docs/kern-09-keepdims-ergebnisse.md
  Befund 1). Mechanik: `IsDynamicRank<S>` (dim.ts:46) prüft
  `number extends S["length"]` — für die Mixed-Rank-Union ist
  `S["length"] = 2 | 3`, also feuert das Rank-Gate NICHT, und
  `ResolveAndApply` distribuiert zu `ShapeError | [2,3]`, was der distributive
  Guard durchlässt. Dazu Nit: kreuzmultiplizierter ShapeError-Text.

### Bindende Policy (Erweiterung von „never wrong, only incomplete" auf Unions)

| Input-Form | Verhalten (bindend) |
|---|---|
| Union-DIM in einem Shape (`2\|7`) | **No-claim**: nie ablehnen; Ergebnis-Dim degradiert zu `number` (wide) — exakt wie ein dynamisches Dim. |
| Shape-Union UNIFORMEN Rangs (`[2,3]\|[4,5]`) | Natürliche Distribution bleibt; Verdikt-Konsum nur uniform (s. Guard-Härtung): ALLE Member fehlerhaft → Compile-Fehler am Argument (Message = Union der Member-Messages), GEMISCHT → akzeptieren (Runtime-Backstop), Ergebnistyp = Union der gültigen Member-Ergebnisse (korrekt auf jedem nicht-werfenden Pfad). |
| Shape-Union GEMISCHTEN Rangs (`[2,3]\|[2,3,4]`, `S["length"]` ist Union) | Verhält sich überall EXAKT wie dynamischer Rang: degradiert zu `readonly Dim[]`, kein Verdikt, Runtime-Backstop. Gilt uniform an JEDEM Rank-Gate (auch `Transpose` — bewusster, offengelegter Präzisionsverlust für diesen exotischen Fall, s. u.). |
| Union ganzer Operanden-TYPEN (`NDArray<A>\|NDArray<B>`) — als Argument UND als Empfänger | Wird von TSs eigener generischer Inferenz bzw. Empfänger-Semantik konservativ ABGELEHNT (Baustein-0-Befund: gilt schon heute, auch für beide-kompatible Member, und ist durch Guard-Design nicht erreichbar/änderbar) — wird als Kontroll-Pin dokumentiert, nicht „gefixt". Die erreichbare Union-Form ist die Shape-Union im Typparameter (Zeilen 2/3). |

Begründung der Uniform-Degradation bei Mixed-Rank statt per-Op-Distribution
(z. B. wäre `Transpose<[2,3]|[2,3,4]>` heute distributiv korrekt `[3,2]|[4,3,2]`):
EINE Regel „Rang kein einzelnes Literal ⇒ graduell" ist konsistent mit der
gesamten `IsDynamicRank`-Philosophie, trivial verifizierbar und schließt die
konfident-falsche Facette (c) strukturell; der Präzisionsverlust betrifft nur
konstruierte Union-Parameter, keine reale API-Nutzung. **Owner-entschieden
(2026-07-12): uniforme Degradation.**

### Bindende Entscheidungen

- **D-V1.1 — `IsUnion` nach dim.ts spiegeln.** `dim.ts` bekommt eine private
  Kopie der 1-Zeilen-Definition (slice-literal.ts:629) mit Querverweis-Kommentar.
  `slice-literal.ts` bleibt UNANGETASTET (append-only-Disziplin; sein Export
  bleibt für vector.ts/reshape.ts bestehen). Verworfen: dim.ts →
  slice-literal.ts importieren (type-only-Zyklus, fragil; dim.ts ist das
  Fundament des Import-Graphen).
- **D-V1.2 — Union-Filter in `CompatDim`/`DimEq`** (dim.ts:54/73), als ERSTE
  Prüfung vor `IsDynamicDim`, Muster `VectorLenCheck`: Union-Dim auf einer Seite
  ⇒ `CompatDim` → `Dim` (wide), `DimEq` → `true`. Fixt (a) und verhindert
  distributive Rekursions-Blowups in `BroadcastAcc`.
- **D-V1.3 — kombiniertes Rank-Gate.** Neues
  `type RankUnknowable<S extends Shape> = IsDynamicRank<S> extends true ? true : IsUnion<S["length"]>`
  in dim.ts (`IsDynamicRank` selbst bleibt unverändert — andere Konsumenten!).
  Ersetzt das Gate an ALLEN Verdikt-tragenden Stellen:
  `Broadcast` (broadcast.ts:43–47), `MatMul` (matmul.ts:27–31), `ReduceAxis`
  (reduce.ts:87), `Transpose` (reduce.ts:104), `SliceShape` (slice.ts:109),
  `SliceSpecsGuard` (slice.ts:217), `DotCheck` (vector.ts:84–86).
  **Bewusst NICHT geändert:** `LiteralShapeProduct` (slice-literal.ts:688, frozen)
  — Mixed-Rank-Unions distribuieren dort zu Union-Produkten, die `ReshapeCheck`s
  vorhandene `IsUnion`-Filter (reshape.ts:64–67) bereits zu no-claim degradieren
  und die bei `flatten()` eine KORREKTE Ergebnis-Union liefern; beides wird
  gepinnt statt umgebaut.
- **D-V1.4 — Guard-Härtung** (ndarray.ts:64), tuple-wrapped:
  `Guard<Result, Actual> = [Result] extends [ShapeError<infer Message>] ? { readonly __shapeError: Message } : Actual`.
  Uniforme Fehler-Union lehnt ab (Message = Union der Messages: aus mehreren
  `{__shapeError}`-Objekttypen wird EIN kombiniertes Objekt — das räumt die
  Fehler-STRUKTUR auf; der Message-INHALT kann bei Union-Eingaben weiterhin
  kreuzmultiplizierte Dim-Paare nennen, die für kein einzelnes Member gelten —
  Kern-09-Nit 3 wird also gemildert und dokumentiert, nicht restlos
  eliminiert), gemischte Union akzeptiert BEWUSST (graduell, Runtime-Backstop),
  Einzel-Fehler unverändert. Wird von `resident.ts`
  (WNDArray) type-only mitkonsumiert — beide Surfaces in einem Fix.
- **D-V1.5 — `OkShape` bleibt unverändert** (ndarray.ts:58): das distributive
  Abstreifen der Fehler-Member liefert genau die Union der gültigen
  Member-Ergebnisse = korrekte Claims auf jedem nicht-werfenden Pfad.

### Testplan (repro-first, bindend)

1. **Repro-Pins ZUERST:** Für jede Facette den Verifier-Repro als Typtest
   formulieren und in einem Scratch-Worktree gegen den PRE-Fix-Stand demonstriert
   ROT zeigen (Beweis der Nicht-Vakuität), dann Fix, dann grün. Einfügen in
   BESTEHENDE Dateien: `broadcast.test-d.ts`, `matmul.test-d.ts`,
   `reduce.test-d.ts`, `ndarray.test-d.ts`, `vector.test-d.ts`,
   `reshape.test-d.ts`, `slice.test-d.ts` (keine neuen Root-Korpus-Dateien).
2. **Pflicht-Pins** (mindestens): `DimEq<2|7,2>` → `true`; `CompatDim<2|7,2>` →
   `number`; add/matmul mit Union-Dim-Operand akzeptiert + Ergebnis-Dim wide;
   Facette (b) in der KORRIGIERTEN Form — Argument `x: NDArray<[2,3]|[7,3]>`
   uniformen Rangs: gemischt akzeptiert (Runtime-Backstop, Ergebnistyp = Union
   der gültigen Member-Ergebnisse), ALLE Member inkompatibel → Fehler am
   Argument mit kombinierter Message (`M1|M2`, D-V1.4); KONTROLL-Pins: die
   Instanzen-Union `NDArray<A>|NDArray<B>` wird als Argument UND als Empfänger
   abgelehnt (TS-Inferenz/Invarianz, dokumentiert, kein Fix); Facette (c)
   `NDArray<[2,3]|[2,3,4]>.sum(2)` akzeptiert mit Ergebnis
   `NDArray<readonly number[]>` (kein konfidenter Einzeltyp mehr);
   Mixed-Rank-Degradation für Transpose/slice/dot (via `RankUnknowable`) und
   getrennt gepinnt `reshape` (dessen no-claim läuft über die EIGENE
   `IsUnion`-Produktfilter-Maschinerie, reshape.ts:64–67, NICHT über
   `RankUnknowable` — inkl. des Baustein-0-Befunds: Mixed-Rank-Empfänger
   `.reshape([6])` wird akzeptiert mit konfidentem `NDArray<[6]>`, korrekt auf
   jedem nicht-werfenden Pfad); `flatten()`-Ergebnis-Union (`NDArray<[6|24]>`)
   als „already-safe"-Pin. **WNDArray-Seite explizit:** dieselben Facetten-Pins
   mindestens für `add`/`matmul`/`sum` zusätzlich als WNDArray-Assertions —
   „gleiche importierte Maschinerie" wird bewiesen, nicht angenommen.
3. **Bestehende Pins:** Der Umzug einer Verhaltens-Grenze re-formuliert
   betroffene Alt-Pins intent-erhaltend (Spike-06-Lehre); JEDE re-formulierte
   Stelle wird im Ergebnisdoc einzeln gelistet. Runtime-Suiten bleiben by
   construction unberührt (Gate: null Edits unter `spike/tests-runtime/`, null
   Edits an runtime.ts).

### Gates (pre-registriert)

- `pnpm check` (Verbund) grün; `test:core`/`test:resident`/`cargo` unverändert
  grün; Artefakt-Hash identisch.
- **Budget (hart, absolut):** Haupt-`check:diag` ≤ 225'000 (= 4,5 % des
  5M-Budgets); `bench:editor` warm-hover-Median ≤ 1 ms, Edit-Toggle ≤ 10 ms
  (beides Größenordnungs-Puffer über heutigen 0,04–0,08 ms / 1,4–3,3 ms).
  Soft-Erwartung (nicht gate-end): Delta ≲ +20'000 — `IsUnion` pro Dim-Vergleich
  liegt auf dem heißesten Typ-Pfad; die Messung entscheidet, das Ergebnisdoc
  berichtet ehrlich. **Baustein-0-Grobmessung (voller V1+V2-Probepatch,
  frischer Worktree): 176'670 @ 132 = Δ+1'036 gegen den reproduzierten Pin
  175'634** — weit unter beiden Schwellen; die Scheiben-Messung bleibt
  maßgeblich.
- **Inferenz-Regression:** Der gesamte bestehende test-d-Korpus ist der Kanarienvogel
  für die tuple-wrapped Guard-Umstellung (Inferenz von `B` durch die Conditional).

---

## Scheibe V2 — strides/NDArrayView/readonly-shape-Harmonisierung

### Befundlage, am Code verankert

- `NDArrayView.strides(): number[]` — METHODE (ndarray.ts:131);
  `NDArray.strides()` — Methode, berechnet `computeStrides(this.shape)` frisch
  (ndarray.ts:402); `WNDArray.strides` — öffentliches `readonly`-FELD vom Typ
  `readonly number[]` (resident.ts:273), semantisch tragend (Views!), ~20
  interne Feld-Zugriffe.
- **Die Methode `strides()` hat NULL externe Aufrufstellen** (repo-weiter Grep,
  2026-07-12: nur Deklaration ndarray.ts:131 + Implementierung ndarray.ts:402).
  Die 8 Assertions in 3 Testdateien (slice/strided/reshape.test.ts) benutzen
  alle FELD-Zugriff auf WNDArray-Handles (`[...x.strides]`) — sie brechen nur
  bei Harmonisierung in Richtung „Methode", NICHT in Richtung „Property".
  (Präzisierung des Item-10-D3-Befunds „bricht in beide Richtungen": die
  Feld-Richtung bricht real nichts Bestehendes.)
- `WNDArray` hat `shape` (readonly S), `toNestedArray(): unknown`
  (resident.ts:1236, wirft nach `dispose()`) — nach der strides-Harmonisierung
  fehlt für `implements NDArrayView<S>` nichts mehr.
- `ThreadedBackend.fromArray/zeros/ones` geben `WNDArray<Mutable<S>>` zurück
  (threaded.ts:804 ff.) — die Threaded-Surface ist mit der WNDArray-Konformität
  automatisch abgedeckt, null Zusatzarbeit.

### Bindende Entscheidungen

- **D-V2.1 — strides wird readonly-PROPERTY** (Owner-entschieden 2026-07-12):
  `NDArrayView` deklariert `readonly strides: readonly number[]`;
  `NDArray` ersetzt die Methode durch einen Getter
  (`get strides(): readonly number[] { return computeStrides(this.shape); }`);
  `WNDArray`-Feld konform AS-IS. Fallout: exakt die zwei Stellen
  ndarray.ts:131/402; null Aufrufer, null Test-Brüche. Dokumentierte Nuance:
  der NDArray-Getter liefert pro Zugriff ein frisches Array
  (`a.strides !== a.strides`), das WNDArray-Feld ist identitätsstabil — im
  View-Kontrakt festhalten, nicht cachen (unnötig). Property-Stil entspricht
  zudem der Spike-05-Hausregel (Funktions-Member künftig property-style).
- **D-V2.2 — `WNDArray<S> implements NDArrayView<S>`** (resident.ts:267,
  Klassen-Kopfzeile). OFFENGELEGTE Disziplin-Abweichung: die Kopfzeile einer
  bestehenden Klasse ist kein insertion-only-Diff — Owner-Bestätigung erfolgt
  über die Abnahme dieser Spec (Analog Item-10-D3-Prozess). View-Kontrakt
  dokumentiert die Residency-Semantik: Member können nach `dispose()` werfen
  (das View-Interface verspricht keine Liveness). Typtests: `WNDArray<[2,3]>`
  zuweisbar an `NDArrayView<[2,3]>` und (Kovarianz) an `NDArrayView<Shape>`;
  dito für ein `ThreadedBackend`-Erzeugnis; `printArray`-Demo optional auf ein
  WNDArray-Argument erweitert.
- **D-V2.3 — deep-readonly `shape` via `Readonly<S>`**, gated auf einen
  **empirischen Pre-Flight-Probe** (Pflicht, VOR der Umsetzung, Scratch-Datei):
  prüft, ob `readonly shape: Readonly<S>` unter `out S` den abstrakten
  Varianz-Check besteht — `Readonly<S>` ist ein Mapped Type, und Spike 05 hat
  bewiesen, dass faktisch-monotone Typfunktionen (`Transpose`) unter `out`
  TS2636 werfen; ob TSs Sonderbehandlung homomorpher Mapped Types das rettet,
  ist OFFEN und nur empirisch zu klären (TS 7.0.2!).
  **Baustein-0-Vorbefund (2026-07-12):** die Probe wurde im Scratch-Worktree
  bereits einmal gefahren — `readonly shape: Readonly<S>` unter `out S` wirft
  KEIN TS2636, und der Prüfrahmen wurde gegen eine echte known-bad-Konstruktion
  (Transpose-Member via Modul-Augmentation → TS2636 feuert) als
  nicht-false-negative validiert. Erwartung: GO. Der Executor wiederholt die
  Probe trotzdem (billig) als Teil der Scheibe — Fußnote für das Ergebnisdoc:
  TSs `out`-Monotonie-Heuristik ist SYNTAXsensitiv (eine vereinfachte
  Transpose-Nachbildung feuerte NICHT, das echte `Transpose` schon).
  - **Probe besteht** → `Readonly<S>` auf `NDArrayView.shape`, `NDArray.shape`
    UND `WNDArray.shape` (alles-oder-nichts: `Readonly<S>` ist nicht an ein
    `shape: S`-Interface-Member zuweisbar, Teil-Umsetzung kollidiert mit
    `implements`). Pins: `@ts-expect-error` auf `nd.shape[0] = 99` /
    `view.shape[0] = 99` / WNDArray dito; Klassen-Hover bleibt `NDArray<[2, 3]>`
    (Member-Hover `readonly [2, 3]` ist akzeptierte, dokumentierte Folge — die
    Clean-Hover-Hausregel bindet den Klassen-Hover, nicht den Member-Hover);
    interner Fallout (Konstruktor-Zuweisungen S → Readonly<S>, Spreads,
    runtime.ts-Konsumenten mit `readonly number[]`-Parametern) wird per
    `pnpm check` ausgemessen und im Ergebnisdoc gelistet.
  - **Probe scheitert (TS2636)** → pre-registriertes **NO-GO** für D-V2.3: das
    Loch bleibt dokumentiert (FOLLOWUPS-Item bleibt offen, um den Probe-Befund
    ergänzt), D-V2.1/D-V2.2 landen unabhängig davon.

### Gates

Alle Suiten grün (`check`-Verbund, test:core, test:resident, cargo, demo;
test:threaded, da resident.ts berührt wird); Artefakt-Hash identisch; Pins neu
gemessen; `bench:editor` in-family (harte Grenzen wie V1). Die 8 bestehenden
strides-Assertions bleiben UNVERÄNDERT grün (Beweis der Richtungs-Entscheidung).

---

## Scheibe V3 — Browser-Smoke-Test (Frühtest)

### Scope

Beweist die bislang ungeprüfte Architektur-Behauptung „das Standard-Surface läuft
im Browser" (FOLLOWUPS 2026-07-12): das reine-JS-`NDArray` UND
`NDArray.backend("wasm")` in einem ECHTEN Browser, COOP/COEP-FREI ausgeliefert
(genau das ist der Punkt: kein cross-origin-isoliertes Deployment nötig).
**Non-Goals:** keine Threads im Browser (Owner-Option 1), keine Perf-Aussagen,
kein Bundler-/Paket-Repräsentativitätstest (das ist Item 11 — hier wird der ROHE
ESM+wasm-Ladepfad gepinnt), nur Chromium in v0 (WebKit/Firefox → FOLLOWUPS).

### Bindende Entscheidungen

- **D-V3.1 — Playwright als devDependency** (`@playwright/test`, Chromium).
  Einordnung: Dev-Tooling ist unter der No-external-libs-Constraint explizit
  erlaubt (Runtime bleibt zero-dependency); Playwright ist das
  Standard-Portfolio des Owners. Neues Script `test:browser`; wird NICHT in
  test:core/test:resident-Listen aufgenommen. `test-scripts-guard.test.ts`
  wird geprüft und bei Bedarf BEWUSST erweitert, sodass Browser-Testdateien
  weder still unregistriert bleiben noch fälschlich in den node-Listen landen.
- **D-V3.2 — Korpus-Isolation (Infra-01-Muster):** neue Dateien unter
  `spike/tests-browser/` mit EIGENER tsconfig; Root-tsconfig bekommt
  `spike/tests-browser` ins `exclude` (einzige Root-Korpus-Änderung — Achtung:
  auch ein Exclude-Edit kann den Pin via Order-Noise bewegen, Messung sagt es);
  `pnpm check` wird zum erweiterten Verbund (root + stress + browser-tsconfig),
  Nicht-Vakuität in beide Richtungen per Korruptions-Test bewiesen (wie Infra 01).
  `playwright.config.ts` liegt am Repo-Root und ist damit außerhalb des
  `include: ["spike"]`-Korpus.
- **D-V3.3 — Build & Serve ohne neue Runtime-Abhängigkeiten:** TS→ESM-Emission
  per `tsc` in ein Scratch-Verzeichnis (Kandidat:
  `rewriteRelativeImportExtensions` — die `.ts`-Relativimporte müssen zu `.js`
  werden; **docs-first-Pflicht:** Verfügbarkeit/Semantik auf TS 7.0.2 VOR der
  Implementierung an der Primärquelle verifizieren; Fallback wäre esbuild als
  weiteres Dev-Tool → als Abweichung eskalieren, nicht still einführen);
  `numtype_core.wasm` wird NEBEN die emittierte loader.js kopiert (`WASM_URL`
  ist `new URL(…, import.meta.url)`, loader.ts:142 — modul-relativ, funktioniert
  nach Emission nur bei Nachbarschaft); statischer Server im Test-Fixture über
  `node:http` (null Dependencies) — **Vorbedingung (Baustein-0-Befund):**
  `spike/src/ambient.d.ts` deklariert acht `node:*`-Module, aber NICHT
  `node:http`; die Deklaration wird im selben Muster ergänzt (Achtung:
  ambient.d.ts liegt im Root-Korpus — Pin-Bewegung als Order-/Realkosten
  ausweisen) — mit korrektem `application/wasm`-MIME —
  bewusst, denn `instantiateStreaming` VERLANGT den MIME-Typ, und genau diese
  dokumentationspflichtige Deployment-Anforderung soll der Test mitprüfen.
- **D-V3.4 — Assertionen (bindend):**
  1. Umgebungsbeweis: im Seiten-Kontext `typeof process === "undefined"` UND
     `crossOriginIsolated === false` (echter Browser, echt COOP-frei).
  2. Op-Matrix differential IN der Seite: JS-`NDArray` (Referenz-Orakel, wie in
     der Differential-Suite) vs. `backend("wasm")`-`WNDArray`, byte-exakt über
     `Float64Array`-Bits: fromArray/zeros/ones; add/sub/mul/div (inkl. eines
     Broadcast-Falls); matmul (inkl. blocked-Pfad); sum (axis, keepdims);
     transpose-View; offset-Slice-View; reshape (View- und Materialize-Zweig);
     dot/norm/cosineSimilarity; ein Spezialwerte-Sample (NaN, ±0, Inf,
     Subnormal durch add + matmul).
  3. Streaming-Pfad nicht-vakuös: der `.wasm`-Fetch geschieht mit
     `content-type: application/wasm`, `WebAssembly.instantiateStreaming`
     existiert; **Mutations-Beweis:** MIME im Server absichtlich auf
     `application/octet-stream` verstellt ⇒ Test wird ROT (der Browser-Zweig
     des Loaders hat keinen Fallback bei instantiateStreaming-Reject —
     loader.ts:158–160), dann revertiert.
  4. `NDArray.backend("threaded")` im Browser wirft mit dem gepinnten
     Message-Stamm (`threaded backend requires Node with the threads
     artifact`) — der Item-10-Browser-Sicherheits-Claim, erstmals im echten
     Browser statt per moduleLoadList-Trace.
- **D-V3.5 — Befund-Semantik:** V3 ist als TEST-ONLY gespeckt (null
  `spike/src`-Änderungen erwartet, Hash identisch). Stellt sich heraus, dass der
  Browser-Pfad einen src-Fix BRAUCHT, ist das ein eigenständiger Befund → an den
  Owner eskalieren (genau dafür läuft der Frühtest), nicht still in der Scheibe
  mitfixen.

### Gates

`pnpm test:browser` grün und < ~60 s lokal; erweiterter `check`-Verbund grün +
Korruptions-Beweis; alle bestehenden Suiten unverändert; Hash identisch; Pins neu
gemessen (Erwartung: Root-Pin ändert sich nur um Order-Noise des Exclude-Edits,
wenn überhaupt).

---

## Owner-Gabeln (ENTSCHIEDEN, 2026-07-12, Chat)

1. **V2 strides-Richtung: readonly-Property** (D-V2.1) — Begründung: null
   Aufrufer der Methode, null Test-Brüche, WNDArray konform as-is; die
   Methoden-Alternative (bricht 8 Assertions + ~20 interne Stellen) verworfen.
2. **V1 Mixed-Rank-Union-Politik: uniforme Degradation** an ALLEN Rank-Gates
   inkl. `Transpose` — eine Regel, strukturell nie konfident-falsch; der
   dokumentierte Preis (Transpose-Hover für exotische Union-Parameter wird
   `readonly number[]`) ist akzeptiert. Per-Op-Ausnahmen verworfen.

## Verifikation

- **Baustein 0 (Spec-Verifier, adversarial, VOR Implementierung):** nach
  Owner-Entscheidung der Gabeln; Auftrag aus docs/verify-runde-template.md.
  Schwerpunkte: die Code-Annahmen dieser Spec (Zeilenanker oben), die
  Typ-Mechanik-Behauptungen aus V1 (Distributions-/Inferenz-Verhalten der
  tuple-wrapped Guard — empirisch im Scratch-Worktree), der
  `Readonly<S>`/TS2636-Probe-Rahmen, die V3-Emissions-Annahmen
  (`rewriteRelativeImportExtensions` auf TS 7.0.2).
- **Je Scheibe:** Zwei-Verifier-Runde (Baustein A + B) wie gehabt.

## Ergebnis-Artefakte je Scheibe

`docs/phase-d-vorarbeiten-v{1,2,3}-ergebnisse.md` mit Post-Verification-Addendum
(beide Verdikte), KB-Capture der generalisierbaren Lektionen (Kandidaten: V1
„distributive Guards sind der Leak-Punkt für Union-Verdikte", V2 der
TS2636-Probe-Ausgang für homomorphe Mapped Types unter `out`, V3
tsc-Emission+ESM+wasm-Muster ohne Bundler), FOLLOWUPS-Austragungen, CLAUDE.md-
Pin-Updates, Commit je Scheibe.

---

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-07-12)

Ein `brainroute:deep`-Verifier, frischer Kontext, Auftrag aus
docs/verify-runde-template.md Baustein 0; alle Typ-Proben empirisch in eigenen
Scratch-Worktrees gegen TS 7.0.2, Haupt-Tree unberührt (git-status-Beweis).

**Blocker (eingearbeitet):** Facette (b) reproduzierte in der ursprünglich
spezifizierten Form NICHT — `NDArray<[2,3]> | NDArray<[7,3]>` als Argument wird
schon am IST-Stand von TSs generischer Inferenz/Klassen-Invarianz abgelehnt
(alle 4 Varianten geprobt, auch beide-kompatibel; identisch vor/nach dem
Probepatch; auch bei freier generischer Funktion). Der reproduzierbare Leak ist
die Shape-Union IM Typparameter EINER Instanz (`x: NDArray<[2,3]|[7,3]>`), die
konfident `NDArray<[2,3]>` liefert. Spec-Konsequenz: Facette-(b)-Beschreibung,
Policy-Zeile 4 und Pflicht-Pin 2 korrigiert; das Kern-07-Addendum-1 ist auf dem
heutigen Stand in der Instanzen-Union-Form nicht nachvollziehbar → V1 trägt die
Diskrepanz-Notiz ins FOLLOWUPS-Item.

**Bestätigt (Auswahl):** alle Zeilenanker (zwei Off-by-one-Zitate korrigiert);
„`strides()` null Aufrufer" + die 8 Feld-Assertions; Facetten (a)/(c) prä- und
post-Fix exakt wie spezifiziert (inkl. `RankUnknowable`-Dreisatz: `2|3` → true,
uniformer Rang → false, `number[]` → true); D-V2.1/D-V2.2 gegen die echte
resident.ts kollisionsfrei; D-V2.3-Probe: KEIN TS2636 (Prüfrahmen gegen echte
known-bad-Konstruktion validiert) → GO erwartet;
`rewriteRelativeImportExtensions` existiert auf TS 7.0.2 und schreibt auch
DYNAMISCHE `./x.ts`-Importe zu `.js` um; der volle bestehende test-d-Korpus
kompiliert mit dem kombinierten V1+V2-Probepatch mit null Diagnosen;
Budget-Grobmessung Δ+1'036 (176'670 @ 132 vs. reproduzierter Pin 175'634).

**Minor/Nits (eingearbeitet):** `node:http` fehlt in ambient.d.ts (V3-
Vorbedingung ergänzt); Mixed-Rank-`reshape`-Empfänger-Pin ergänzt (läuft über
die reshape-eigene `IsUnion`-Maschinerie, nicht `RankUnknowable`); WNDArray-
seitige Facetten-Pins jetzt explizit gefordert; „Nit 3 eliminiert" zu „Struktur
aufgeräumt, Inhalt dokumentiert" präzisiert; Zeilenzitate broadcast.ts:43–47 /
matmul.ts:27–31.

**Offen geblieben (bewusst):** Zero-Rust/Hash-identisch wurde vom Verifier NICHT
geprüft (Baustein 0 führt keine Gates aus) — bleibt Executor-Gate je Scheibe.
