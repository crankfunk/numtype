# 0b — Typisiertes `toNestedArray` — bindende Spec (Stufe 3b)

**Version:** v2 (2026-09-24) · **Status:** Owner-Richtung abgenommen, Baustein 0 gelaufen (ein
Blocker, eingearbeitet; Richtung unverändert) → **implementierungsreif**

**Änderungslog v1.1 → v2 (Baustein-0-Befunde, s. Addendum am Ende):** D1 von rekursiver
Zerlegung über `S` auf Rang-Akkumulator umgestellt (behebt den Blocker `number[][] | number[][]`
bei Unions gleichen Rangs UND `NestedArray<never>` = `never`); Rang-Grenze 999 offengelegt;
D5, D6, D7 und Testplan präzisiert. Semantik und Richtung unverändert → keine neue Owner-Runde.
**Stufe:** 3b (CLAUDE.md, Eskalationsleiter): neue Typ-Maschinerie auf einer öffentlichen
Signatur + Umkehr einer gepinnten Entscheidung (D-V2.2/B5). Voller Katalog: Baustein 0 vor dem
Code, danach A + B + C parallel, Lint im Gate-Block, Ergebnis-Doc mit Addendum.
**Berührte Covenant-Invarianten:** **M2** (never wrong, only incomplete), **M3** (saubere Hovers,
seit v6 inkl. Methoden-Rückgabetypen). Nicht berührt: M1/M4 (kein Rust, Laufzeit unverändert),
M5, Z1, Z2, S1.
**Anlass:** Stand-Review 2026-09-23, Befund F6: der einzige Weg, Daten verschachtelt
herauszuholen, ist untypisiert — ausgerechnet in einer Bibliothek, deren USP statische Typen sind.

## Ziel

`NDArray<S>.toNestedArray()` und `WNDArray<S>.toNestedArray()` liefern einen am **Rang** von
`S` berechneten Typ statt `unknown`: `[2, 3]` → `number[][]`, `[]` → `number`,
`[number, 3]` → `number[][]`. Wo der Rang statisch nicht feststeht, ein ehrlicher rekursiver
Typ statt einer Behauptung.

## Vorab-Befunde (2026-09-24, Probe in Scratch gegen tsc 7.0.2, nicht im Repo)

Eine Wegwerf-Probe mit dem geplanten Typ hat zwei Designfragen VOR der Spec entschieden. Die
Probe ist informell (Arbeitsregel 8) — Baustein 0 und Verify messen am echten Code nach:

1. **`NDArrayView<out S>` kann den Typ NICHT tragen.** Ein rechnender Rückgabetyp unter `out S`
   wirft TS2636 („`NestedArray<sub-S>` is not assignable to `NestedArray<super-S>`") —
   dieselbe Klasse wie Spike 05 (`Transpose`). → D3.
2. **Der Methoden-Hover löst sauber auf** (echter `tsc --lsp --stdio`, Regel 13):
   `(method) Arr<[2, 3]>.toNestedArray(): number[][]`; Variablen `const n: number[][]`,
   Rang 0 `number`, `[number, 3]` → `number[][]`. Anders als S3s `StackResultOf` bleibt der
   Alias NICHT stehen, weil die Auflösung in einem nicht-generischen Array-Typ endet. Nur der
   dynamische Fall zeigt einen Namen: `toNestedArray(): NestedValue` — ein rekursiver Typ ist
   nicht expandierbar. → D4.

## Design (D1–D7)

- **D1 — der Typ.** Zwei exportierte Aliase auf Modulebene in `spike/src/ndarray.ts` (keine
  neue Datei: eine neue Datei kostet ±≈7.000 Order-Noise):
  ```ts
  export type NestedValue = number | NestedValue[];
  export type NestedArray<S extends Shape> =
    [S] extends [never] ? NestedValue                 // vor allem anderen: sonst Endlos-Akkumulator
    : RankUnknowable<S> extends true ? NestedValue    // dim.ts:82, wiederverwendet
    : NestedOfRank<S["length"]>;
  // privat (nicht exportiert): baut aus EINEM Rang-Literal die Verschachtelung, tail-rekursiv
  type NestedOfRank<N extends number, Acc extends readonly unknown[] = [], T = number> =
    Acc["length"] extends N ? T : NestedOfRank<N, [...Acc, unknown], T[]>;
  ```
  **v2-Begründung:** die v1-Form (rekursive Zerlegung `S extends [unknown, ...infer R] ?
  NestedArray<R>[]`) lieferte für `[2, 3] | [4, 5]` den Typ `number[][] | number[][]` — jede
  Zweig-Instanziierung trägt eigene Alias-Metadaten, TS dedupliziert nicht (Baustein 0,
  empirisch am echten `dim.ts`). Über `S["length"]` gibt es bei gleichem Rang genau EINE
  Instanziierung. Probe v2 (Scratch, tsc 7.0.2, echtes `dim.ts`): alle 12 Erwartungs-Pins grün
  inkl. Union gleichen Rangs, `never`, `any`, `Shape`, Rest-Tupel; LSP-Hover
  `toNestedArray(): number[][]` auch für die Union.
  **Nur der Rang zählt, nie die Dim-Werte** — `number[]` statt Länge-N-Tupel. Tupel pro Dim-Wert
  wäre die verbotene Tupellängen-Arithmetik über große Dimensionen (CLAUDE.md, TS-Limits) und
  würde bei `[1000, 1000]` das Budget sprengen. Tail-rekursiv über den Rang (Akkumulator-Muster der Hausregel). **Offengelegte Grenze:** ab
  Rang **999** TS2589 (Tail-Rekursions-Limit, Probe v2: 998 grün, 999/1000/1024 rot) — die
  bestehende Shape-Maschinerie bricht bei 1024 (`Reverse`). Für `toNestedArray` sinkt die
  Grenze also um 25 Ränge; praktisch irrelevant, aber im Ergebnis-Doc zu nennen.
  `RankUnknowable` schließt sowohl dynamischen Rang (`number[]`) als auch Unions über die
  Länge (`[2] | [2, 3]`) ab, **bevor** destrukturiert wird (Arbeitsregel 3). Eine Union
  gleichen Rangs (`[2, 3] | [4, 5]`) ergibt korrekt `number[][]`.
- **D2 — Signaturen der Klassen.** `NDArray.toNestedArray(): NestedArray<S>` und
  `WNDArray.toNestedArray(): NestedArray<S>`; Laufzeit unverändert, ein Cast an der Rückgabe
  (`as NestedArray<S>`), wie die Ops es mit `OkShape` tun. **Abweichung von der Hausregel
  „TS-Klassenkörper insertion-only"**: die Signatur eines BESTEHENDEN Members ändert sich. Das
  ist der Zweck der Scheibe und nicht vermeidbar (ein zweiter Member daneben wäre eine
  API-Dublette). Zur Owner-Bestätigung vorgelegt (s. „Entscheidungen").
- **D3 — `NDArrayView` bleibt `unknown`.** Belegt durch Vorab-Befund 1; `implements` erlaubt
  einen engeren Rückgabetyp auf den Klassen. Der View-Doc-Kommentar erklärt, warum.
- **D4 — `NestedValue` wird aus `index.ts` exportiert** (type-only), `NestedArray` ebenfalls —
  damit der einzige benannte Hover-Fall für Konsumenten nachschlagbar und benennbar ist.
- **D5 — M2-Kanten.** Jede Variante muss „nie falsch" sein: Rang 0 liefert zur Laufzeit wirklich
  eine Zahl (beide Klassen: `data[offset] ?? 0`); size-0-Shapes (`[0, 3]`) liefern `[]` —
  typkorrekt als `number[][]`. **Geklärt (Baustein 0 + Probe v2):** `any`, `never`, `Shape`,
  `number[]`, Rest-Tupel `[2, ...number[]]` und Unions verschiedenen Rangs → `NestedValue`;
  optionale Tupel-Elemente sind gar kein gültiges `Shape` (Constraint-Fehler vorher).
- **D6 — Pins neu.** Die drei `Equal<…, unknown>`-Pins (spike/tests/ndarray.test-d.ts:216-218)
  werden ersetzt, ihr Zweck (Drift zwischen den Deklarationen fangen) bleibt:
  View-Pin bleibt `unknown`; NDArray und WNDArray je gegen dieselben exakten Erwartungen
  (Rang 0/1/2/3, `[number, 3]`, `number[]`, `Shape`, `[2]|[2,3]` → `NestedValue`,
  `[2,3]|[4,5]` → `number[][]`, `never` → `NestedValue`, Rest-Tupel, Top-Typ: `AnyNDArray` für
  NDArray bzw. `AnyWNDArray` (resident.ts:177) für WNDArray)
  plus ein Pin, dass beide Klassen für dasselbe `S` denselben Typ liefern. Keine neue Testdatei.
- **D7 — Konsumenten-Nachweis.** Beide Konsumenten-Smokes (`consumer`, `consumer-strict`) setzen
  `noUncheckedIndexedAccess` NICHT — dort wird `NDArray.fromArray([2, 3], …).toNestedArray()[0][1]`
  als `number` gepinnt, ohne Cast (neue Zeile, bisher ruft kein Smoke `toNestedArray`). Der Fall
  MIT dem Flag (`number | undefined`) wird intern in `spike/tests/ndarray.test-d.ts` gepinnt
  (Root-tsconfig setzt das Flag). `spike/tests-package/package-smoke.test.ts:23-30` tippt
  `toNestedArray(): unknown` lokal — auf den neuen Typ nachziehen. README: erst mit dem Release
  eintragen (Arbeitsregel 19).

## Nicht-Ziele

Typisierung der View (`NDArrayView`) · Tupel mit Dim-Längen · Änderung der Laufzeit ·
`toArray()`/`data` · jede Op-Änderung · dtype (eigene Scheibe danach — `NestedArray` wird dort
ein zweites Argument brauchen; dieses Design sperrt das nicht).

## Gates (Absolut-Grenzen, VOR der Messung registriert)

- `pnpm check` Exit 0 · check:diag Root **≤ +4.000** gegen 226.220 @ 140 (Dateiset bleibt 140).
  Begründung: ein generischer Member-Rückgabetyp rippelt über die Klassen-Surface
  (Mechanismus 3); die Probe war ohne echten Korpus nicht aussagekräftig.
- stress/browser berichten · `bench:editor`: Pins nur bei uniformer, dem stress-Δ gleicher
  Verschiebung neu setzen · Freeze-Hash unverändert `2a54d9fd…` · alle Testzahlen gleich oder
  um die neuen gewachsen · `test:package` inkl. beider Konsumenten-Smokes grün · Lint auf frisch
  gebautem Graph.
- **LSP-Messung Pflicht** (M3, Regel 13): Methoden-Hover und Variablen-Hover für Rang 0/2,
  `[number, 3]` und dynamisch, auf NDArray UND WNDArray, dazu der Klassen-Hover `NDArray<[2, 3]>`
  als Kontrollpunkt.

## Testplan

- Typebene (D6) in `spike/tests/ndarray.test-d.ts`, Diagnose-Inhalte, nicht nur Existenz.
- Laufzeit: Rang 0 liefert `typeof === "number"` auf beiden Klassen; size-0 liefert `[]`;
  eine View (Transpose) liefert dieselbe Verschachtelung wie ihr materialisiertes Gegenstück
  (Klasse per Regel 12 assertiert). Vorher prüfen, ob die bestehenden View-Tests mit
  `toNestedArray` (resident.test.ts, D3-Block der Release-Scheibe) das schon leisten — dann
  erweitern statt duplizieren.
- Mutanten (A und B je eigene): u. a. Akkumulator-Start `T = number[]` (Rang um eins falsch),
  `RankUnknowable`-Gate entfernt (Union-Rang → falsche Behauptung, M2-Bruch), `never`-Gate
  entfernt (erwartet: TS2589 bzw. roter `never`-Pin).

## Entscheidungen für den Owner (vor Baustein 0) — ENTSCHIEDEN 2026-09-24

Owner: (1) Abweichung D2 bestätigt · (2) `NestedValue` · (3) nicht allein veröffentlichen,
mit dem nächsten Minor (0.4.0) bündeln.


1. **Hausregel-Abweichung D2** (Signatur eines bestehenden Members ändert sich): bestätigen?
2. **Dynamischer Fall:** `NestedValue` (rekursiv, nützlich, im Hover benannt) — oder `unknown`
   (wie heute, kein neuer Name)? Empfehlung: `NestedValue`.
3. **Release:** Die Verengung von `unknown` kann bestehenden Konsumenten-Code brechen (z. B. ein
   Cast `as string` wird zum Fehler) → nach der SemVer-Policy ein Minor, also 0.4.0 — nicht
   allein veröffentlichen, sondern mit dem nächsten Paket bündeln?

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-09-24)

`brainroute:deep`, frischer Kontext, eigener Worktree am HEAD `53c8cb0`. **Keine falsche
Code-Annahme** (alle Datei-/Zeilen-Belege exakt, Baseline 226,220 @ 140 reproduziert).
**Bestätigt am echten Code:** D3 (TS2636 auf `NDArrayView<out S>`), D2 (Entwurf kompiliert in
allen drei Korpora; nur die zwei zu ersetzenden Pins rot), M3-Hover per echtem LSP inkl.
Op-Ketten (`a.matmul(b).toNestedArray()` → `number[][]`). Budget-Entwurf: Root Δ+947, stress
Δ+43, browser Δ0, Re-Export aus `index.ts` Δ0 — Entwurfsmessung, nicht das Scheibenergebnis.
**Befunde:** (1) BLOCKER — Union gleichen Rangs → `number[][] | number[][]` (behoben durch D1 v2);
(2) minor — `NestedArray<never>` → `never` (behoben durch das `never`-Gate in D1 v2);
(3) minor — `AnyWNDArray` statt `AnyNDArray` für den WNDArray-Top-Typ-Pin (D6 v2);
(4) minor — D7-Zuordnung von `noUncheckedIndexedAccess` zu den Smokes stimmte nicht (D7 v2);
(5) minor — `package-smoke.test.ts` tippt `unknown` lokal (D7 v2); (6) nit — bestehende
View-Tests prüfen (Testplan v2). Nuance ohne Handlungsbedarf: im Rumpf einer noch GENERISCHEN
Funktion hovert `NestedArray<S>` unaufgelöst — normales TS-Verhalten wie bei allen Typen hier.
Baustein 0 hatte außerdem vermutet, die v1-Rekursion senke die Rang-Grenze, und das empirisch
WIDERLEGT (v1: kein TS2589 bis 1024). Die v2-Form senkt sie dagegen auf 999 (s. D1) — bewusst
in Kauf genommen gegen die Korrektheit bei Unions gleichen Rangs.
