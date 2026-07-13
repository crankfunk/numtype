# Phase-D-Vorarbeiten, Scheibe V2 — strides/NDArrayView/readonly-shape (Ergebnisse)

Spec: `docs/phase-d-vorarbeiten-spec.md` (Scheibe V2 + Gemeinsame Rahmenbedingungen,
Baustein-0-Vorbefund bereits im Dokument). Datum: 2026-07-13. Alles unten ist in tatsächlich
in dieser Session ausgeführten Kommandos verankert; Honesty-Rule gilt durchgehend —
unverifizierte Aussagen sind als solche markiert. HEAD zu Beginn: `4f4414e` (V3+V1 gelandet).

## Was gebaut wurde

Drei bindende Entscheidungen (D-V2.1–D-V2.3), alle umgesetzt, **null Rust, null neue
Root-Korpus-Dateien** (nur bestehende Dateien editiert — 14 Dateien, 0 neue).

- **D-V2.0 (Pflicht-Probe, ZUERST, Scratch):** bestanden — `Readonly<S>` unter `out S` wirft
  KEIN TS2636 auf TS 7.0.2; der Prüfrahmen wurde im selben Lauf gegen eine echte
  known-bad-Konstruktion (Transpose-Member in Rückgabeposition) validiert, die zuverlässig
  TS2636 wirft → **GO** für D-V2.3 (Details unten).
- **D-V2.1 (strides → readonly-Property):** `NDArrayView.strides` von
  `strides(): number[]` zu `readonly strides: readonly number[]`; `NDArray.strides()`
  (Methode) zu einem Getter `get strides(): readonly number[]`; `WNDArray.strides`
  (Feld) unverändert — konform as-is.
- **D-V2.2 (`WNDArray<S> implements NDArrayView<S>`):** Klassen-Kopfzeile in resident.ts,
  offengelegte Abweichung vom insertion-only-Muster (spec-bestätigt).
- **D-V2.3 (deep-readonly `shape` via `Readonly<S>`):** auf `NDArrayView.shape`,
  `NDArray.shape` UND `WNDArray.shape`, alles-oder-nichts, nach GO aus D-V2.0.

**Präzisierung (verify-round closure, B-Befund 8):** "alles-oder-nichts, weil ein Teil-Rollout
mit `implements` kollidiert" ist nur in EINER Richtung strukturell korrekt, nicht symmetrisch —
empirisch nachgeprüft in dieser Session (drei isolierte Mutationen im Scratch-Worktree, je
sofort auf den V2-Stand zurückgedreht):
- **Klasse readonly-er als Interface** (`NDArray`/`WNDArray` bleiben `Readonly<S>`, nur
  `NDArrayView.shape` wird probeweise auf `S` zurückgedreht): der Typchecker meldet
  eigenständig `TS2322`-Fehler an den Nutzstellen, die eine solche Klasse als `NDArrayView`
  verwenden (z. B. `wnd23AsExactView`) — STRUKTURELL gefangen, unabhängig von den Equal-Pins.
- **Klasse mutabler als Interface** (`NDArrayView.shape` bleibt `Readonly<S>`, nur
  `NDArray.shape`/`WNDArray.shape` werden probeweise auf `S` zurückgedreht): `implements
  NDArrayView<S>` selbst meldet dabei GAR KEINEN Fehler (eine mutable Klassen-Property ist
  immer einer readonly-Interface-Property zuweisbar — Mutabilität darf beim Implementieren
  eines Interfaces immer VERSCHÄRFT werden) — dieser Fall wird AUSSCHLIESSLICH von den 56
  `Equal<(typeof x)["shape"], readonly TUPEL>`-Pins gefangen (Fund 1 unten), nicht vom
  Typsystem selbst. Die "kollidiert mit implements"-Formulierung in `ndarray.ts`s Doc-Kommentar
  ist also eine Vereinfachung: ein TEIL-Rollout (nur die Interface-Seite readonly) würde vom
  Compiler NICHT strukturell verhindert, sondern nur von der Test-Suite bemerkt — der
  "all-or-nothing"-Praxisgrund war (und bleibt) trotzdem richtig, nur die BEGRÜNDUNG war
  unvollständig.

Geänderte Dateien (Kern-Diffs):

| Datei | Änderung |
|---|---|
| `spike/src/ndarray.ts` | `NDArrayView.strides` → readonly Property; `NDArrayView.shape` → `Readonly<S>`; `NDArray.strides()` → Getter; `NDArray.shape` → `Readonly<S>`; Doc-Kommentare (NDArrayView-Vertrag, `AnyNDArray`-Varianz-Klausel) aktualisiert |
| `spike/src/wasm/resident.ts` | Klassen-Kopfzeile `implements NDArrayView<S>`; `WNDArray.shape` → `Readonly<S>`; Import von `NDArrayView` |
| `spike/demo.ts` | `printArray("rA …", rA)` — WNDArray jetzt direkt als `NDArrayView`-Argument (D-V2.2-Beleg); ein `.shape`-Cast-Fallout (D-V2.3, s. u.) |
| `spike/tests-runtime/{elementwise,special-values,strided}.test.ts` | 6 `.shape`-Lesestellen auf `AnyWNDArray`-Handles: `as readonly number[]` (TS2740-Fallout, s. u.) — **keine Assertion-Logik geändert** |
| `spike/tests/{backend-api,broadcast,limits,ndarray,reshape,slice,vector}.test-d.ts`, `spike/tests-stress/reshape-stress.test-d.ts` | **56** (korrigiert, verify-round closure — s. Fund 1 unten) bestehende `Equal<(typeof x)["shape"], TUPEL>`-Pins intent-erhaltend zu `Equal<…, readonly TUPEL>` re-formuliert; 1 `@ts-expect-error`-Pin zu einem positiven Pin + neuem Gegen-Pin umgebaut (Varianz-Fund, s. u., inzwischen per Re-Invariantierung wieder zu einem `@ts-expect-error`-Pin re-expressiert — s. Schließungsrunde unten); neue D-V2.2-Typtests (WNDArray-Kovarianz) in `ndarray.test-d.ts`/`backend-api.test-d.ts` |

## D-V2.0: die Pflicht-Probe (voller Output)

Scratch-Datei außerhalb des `spike`-Korpus (`probe-v2-0/probe.ts`, eigene `tsconfig.json`,
frischer `git worktree add … HEAD`), zwei Interfaces: der echte D-V2.3-Kandidat plus ein
Kontroll-Interface mit einem shape-berechnenden Member (`Transpose<S>` in
Rückgabeposition — genau das bereits in Spike 05 dokumentierte TS2636-Muster):

```ts
interface ProbeReadonlyShape<out S extends Shape> {
  readonly shape: Readonly<S>;
}
interface ProbeKnownBad<out S extends Shape> {
  readonly shape: S;
  transpose(): ProbeKnownBad<Transpose<S>>;
}
```

```
$ ./node_modules/.bin/tsc --noEmit -p probe-v2-0/tsconfig.json
probe-v2-0/probe.ts(32,25): error TS2636: Type 'ProbeKnownBad<sub-S>' is not assignable to type 'ProbeKnownBad<super-S>' as implied by variance annotation.
  The types returned by 'transpose()' are incompatible between these types.
    Type 'ProbeKnownBad<Transpose<sub-S>>' is not assignable to type 'ProbeKnownBad<Transpose<super-S>>'.
      Type 'Transpose<sub-S>' is not assignable to type 'Transpose<super-S>'.
        Type 'readonly number[] | Reverse<sub-S, []>' is not assignable to type 'Transpose<super-S>'.
          Type 'readonly number[]' is not assignable to type 'Transpose<super-S>'.
EXIT: 1
```

Genau EIN Fehler, auf `ProbeKnownBad` (Zeile 32), NICHT auf `ProbeReadonlyShape` (Zeile 23).
Das bestätigt beides gleichzeitig: (1) der Prüfrahmen ist kein False-Negative — eine echte
schlechte Konstruktion wird gefangen; (2) `Readonly<S>` unter `out S` besteht sauber → **GO**.
Deckt sich mit dem Baustein-0-Vorbefund.

## D-V2.1/D-V2.2: Fallout wie vorhergesagt (klein, null)

Faktenlage vorab bestätigt: `.strides()` (Methode) hat repo-weit NULL Aufrufstellen
(`grep -rn "\.strides("` über `spike/` → nur die Doc-Kommentar-Erwähnung selbst); die 8
Feld-Assertions liegen alle auf `[...x.strides]`-Zugriff und sind **byte-identisch
unverändert** (`git diff` zeigt keine `-`/`+`-Zeilen auf diesen exakten Zeilen):

```
spike/tests-runtime/strided.test.ts:115   assert.deepStrictEqual([...v.arr.strides], expectedStrides, …);
spike/tests-runtime/strided.test.ts:128   assert.deepStrictEqual([...a.strides], [12, 4, 1]);
spike/tests-runtime/strided.test.ts:129   assert.deepStrictEqual([...t.strides], [1, 4, 12]);
spike/tests-runtime/strided.test.ts:132   assert.deepStrictEqual([...tt.strides], [12, 4, 1]);
spike/tests-runtime/strided.test.ts:333   assert.deepStrictEqual([...cCopy.strides], computeStrides(shape), …);
spike/tests-runtime/reshape.test.ts:151   assert.deepStrictEqual([...r.strides], computeStrides([2, 6]));
spike/tests-runtime/slice.test.ts:500     assert.deepStrictEqual([...view.strides], [1]);
spike/tests-runtime/slice.test.ts:799     assert.deepStrictEqual([...sliced.strides], computeStrides(shape));
```

D-V2.1 + D-V2.2 isoliert (vor D-V2.3) gegen den vollen `check`-Verbund gemessen: **0 Fehler**,
`test:core` 818/818, `test:resident` 4279 (4277 pass + 2 skip) — exakt wie spezifiziert.
`WNDArray implements NDArrayView<S>` kompiliert kollisionsfrei gegen die echte resident.ts
(Baustein-0-Vorbefund bestätigt).

## D-V2.3: zwei ECHTE, unerwartete Funde (Fallout größer als die Spec-Erwartung)

Die Spec erwartete "klein" ("runtime.ts-Signaturen nehmen bereits `readonly number[]`").
Empirisch war der Fallout **55 Fehler** beim ersten vollen Check nach dem Feld-Typwechsel,
in zwei qualitativ verschiedenen Kategorien — beide unten einzeln aufgeschlüsselt und
gefixt, keine stillschweigend übergangen.

### Fund 1 (erwartete Kategorie, größer als gedacht): 56 test-d-Pins + `Readonly<any>`-Quirk

**Korrektur (verify-round closure, 2026-07-13 — A-Auflage, Honesty-Rule):** die ursprüngliche
Zahl unten ("52 von 55 Fehlern") war intern inkonsistent — die zugehörige Pro-Datei-Tabelle
summierte sich auf 59, nicht 52 (Verifier-A-Befund), und A's eigene Nachzählung kam auf 56.
Diese Session hat unabhängig nachgezählt, auf ZWEI Wegen, beide konvergieren exakt auf
**56**, nicht 52:
1. **Mechanisch** (Diff-basiert, auf dem unveränderten V2-Original-Diff, gesichert im
   `wt-v2-probe`-Scratch-Worktree vor dieser Session): jedes `-type NAME = Expect<Equal<…,
   TUPEL>>` mit einem `+type NAME = Expect<Equal<…, readonly TUPEL>>` GLEICHEN Namens
   zusammengeführt — also eine echte Re-Formulierung eines VORHANDENEN Pins, nicht ein neuer
   Pin (die D-V2.2-Kovarianz-Assertions wie `wnd23AsExactView` zählen NICHT mit, auch wenn
   ihre Zeilen "readonly" enthalten — sie sind neu, nicht re-formuliert; ebenso die eine
   umgebaute `@ts-expect-error`→positiv-Pin-Stelle aus Fund 2, s. u., die keinen gleichnamigen
   Vorgänger hat).
2. **Empirisch, per Mutation** (derselbe Scratch-Worktree, `Readonly<S>` an allen drei Stellen
   — `NDArrayView.shape`, `NDArray.shape`, `WNDArray.shape` — einzeln UND gemeinsam auf
   `S` zurückgedreht, `pnpm exec tsc --noEmit` gegen root+stress): `NDArray.shape` allein löst
   47 Fehler im root-Korpus aus, davon **46 `TS2344`-Equal-Pin-Fehler** (broadcast 2 + limits 1
   + ndarray 15 + reshape 12 + slice 13 + vector 3) und 1 separater `TS2322`-Fehler (die
   ehemalige Weitungspin-Stelle, Fund 2 — nicht Teil dieser Zählung), plus 2 weitere
   `TS2344`-Fehler im stress-Korpus; `WNDArray.shape` allein löst 8 weitere `TS2344`-Fehler im
   root-Korpus aus (backend-api 4 + ndarray 4, disjunkt von den 46 oben), 0 im stress-Korpus;
   `NDArrayView.shape` allein löst NULL `TS2344`-Equal-Pin-Fehler aus (nur 2
   `TS2322`-Konformitäts-Fehler an Nutzstellen, s. Fund 8/(b) unten — ebenfalls nicht Teil
   dieser Zählung). Root-Vereinigung der `TS2344`-Fehler: 46 + 8 + 0 = **54**; stress-
   Vereinigung: 2 + 0 + 0 = **2**; Gesamt: 54 + 2 = **56** — exakt konsistent mit Weg 1 (56,
   s. Tabelle unten; die Datei-für-Datei-Aufschlüsselung root-only summiert sich ebenfalls auf
   54, s. Tabelle: 19+13+12+4+2+3+1 = 54). Der kombinierte Drei-Stellen-Mutant (alle drei
   Stellen gleichzeitig zurückgedreht) reproduziert exakt dieselben 54 root- + 2
   stress-`TS2344`-Fehler (plus 1 zusätzlichen `TS2322`-Fehler von der ehemaligen
   Weitungspin-Stelle, separat, s. Fund 2). Alle drei Einzel- und der kombinierte
   Drei-Stellen-Mutant wurden nach der Messung exakt auf den V2-Pristine-Stand zurückgedreht
   (MD5/`git diff --stat`-Identitätsbeweis je Schritt).

Woher die alten Zahlen (52/55/59) kamen, ist nicht mit letzter Sicherheit rekonstruierbar
(diese Session hat keinen Zugriff auf den genauen Lauf, der sie erzeugt hat) — plausibel:
die "55 Fehler beim ersten vollen Check" wurden gegen einen NOCH-NICHT-migrierten
Test-Korpus gemessen (die Zwischenstufe während der Implementierung, nicht der fertige
V2-Endstand), während die "56" dieser Session eine SAUBERE Post-hoc-Abhängigkeitszählung
gegen den fertigen Diff ist — zwei unterschiedliche Messungen, keine zwingend "falsch". Die
Tabelle unten ist die korrigierte, zweifach (mechanisch + Mutation) verifizierte Fassung;
**was gezählt wird:** genau die `Equal<(typeof x)["shape"], TUPEL>`-artigen Typtest-Pins, die
VOR D-V2.3 einen nackten Tupel-Typ prüften und NACH D-V2.3 denselben benannten Pin (gleicher
Bezeichner) mit `readonly`-Präfix re-formuliert weiterführen — nicht neu hinzugefügte Pins,
nicht die 7 `Readonly<any>`-Cast-Stellen (eigene Kategorie, unten), nicht die eine
Fund-2-Pin-Umbau-Stelle (eigene Kategorie, s. u.).

| Datei | Pins (korrigiert, zweifach verifiziert) |
|---|---|
| `ndarray.test-d.ts` | 19 (15 NDArray-seitig + 4 WNDArray-seitig: T7f/T7g/UW1/UW2) |
| `slice.test-d.ts` | 13 |
| `reshape.test-d.ts` | 12 |
| `backend-api.test-d.ts` | 4 |
| `vector.test-d.ts` | 3 |
| `broadcast.test-d.ts` | 2 |
| `limits.test-d.ts` | 1 |
| `tests-stress/reshape-stress.test-d.ts` | 2 (Infra-01-Korpus) |
| **Summe** | **56** |

`Equal` ist strikte Typgleichheit (nicht bloße gegenseitige
Zuweisbarkeit, `spike/tests/test-utils.ts:14`), und `.shape` ist jetzt tatsächlich
`readonly [...]` statt `[...]` — der Member-Hover hat sich GENAU wie spezifiziert geändert,
und jeder Pin, der den alten (mutable) Member-Typ exakt prüfte, musste intent-erhaltend
re-formuliert werden (Spike-06-Lehre: eine Scheibe, die eine Verhaltensgrenze verschiebt,
re-formuliert betroffene Alt-Pins statt sie unangetastet zu lassen). Reine `TUPEL` →
`readonly TUPEL`-Umschreibungen, jede Stelle einzeln editiert, keine Massentransformation
per Skript — Guard-reine Typ-Level-Checks (`Equal<Broadcast<…>, …>`, `Equal<SliceSpecsGuard<…>>`
etc., die NICHT über `.shape` gehen) waren durchweg NICHT betroffen (bestätigt: keiner dieser
RG/G/LDI/SD-Pins erscheint in der Fehlerliste).

**7 der 55 Fehler** waren ein separater, TS-spezifischer Quirk: `TS2740: Type
'Readonly<any>' is missing the following properties from type 'readonly number[]'`, an
jeder Stelle, wo `.shape` von einem `AnyWNDArray = WNDArray<any>`-Handle gelesen und einer
`readonly number[]`-typisierten Variable/einem Parameter zugewiesen wurde (`demo.ts`,
`elementwise.test.ts` ×2, `special-values.test.ts`, `strided.test.ts` ×3). Root Cause:
`Readonly<any>` kollabiert in TS NICHT strukturell zurück zu `any` — ein bekannter,
docs-first bestätigter Mapped-Type-über-`any`-Quirk, spezifisch für die `<any>`-Erasure-Form
(`AnyNDArray`/`AnyWNDArray`), NICHT reproduzierbar mit dem regulären `Shape`-Bound (per
gezieltem Gegen-Probe bestätigt: `NDArray<Shape>`/`NDArray<readonly number[]>` haben dieses
Problem nicht). Fix: sieben lokale `as readonly number[]`-Casts an den Lese-Stellen, mit
Inline-Kommentar — der Laufzeitwert ist identisch, nur der Compile-Zeit-Marker ändert sich;
KEINE Assertion-Logik in den betroffenen Testdateien geändert.

**Präzisierung (verify-round closure, B-Befund 4, empirisch nachgeprüft in dieser Session per
Scratch-Probe):** der Cast ist NICHT an jeder `.shape`-Lesestelle eines `AnyWNDArray`-Handles
nötig — nur an ZWEI der vier möglichen Zugriffsformen. Probe (`declare const x: Readonly<any>`):
`x.length` und `x[0]` (Property-/Index-Zugriff) lösen `any` klaglos auf und brauchen KEINEN
Cast; erst **Zuweisung** an eine `readonly number[]`-typisierte Stelle (`const w: readonly
number[] = x` → `TS2740`) und **Iteration/Spread** (`[...x]` → `TS2488`, "must have a
`[Symbol.iterator]()` method") lösen den Quirk aus. Die sieben Fundstellen in dieser Scheibe
brauchten den Cast, weil sie alle Zuweisung oder Spread sind (z. B. `[...v.arr.shape]` in
`strided.test.ts`) — die Formulierung oben ("an jeder Stelle, wo `.shape` … gelesen … wird")
ist also zu weit gefasst; präzise: an jeder Stelle, wo das gelesene `.shape` einer
`readonly number[]`-Stelle zugewiesen oder gespreadet wird.

### Fund 2 (UNERWARTET, nicht in der Spec antizipiert): Varianz-Verschiebung bei `NDArray`

Die verbleibende Fehlerstelle (`ndarray.test-d.ts`, `@ts-expect-error - invariant S: fixed
shape not assignable to NDArray<Shape>` → "Unused '@ts-expect-error' directive") deckte
etwas Substanzielleres auf: **`NDArray<[2, 3]>` ist nach D-V2.3 neu einer
`NDArray<readonly number[]>` zuweisbar** — Spike 01s dokumentierter Befund "die Klasse ist
komplett invariant" gilt für diese Richtung nicht mehr.

**Isolation (A/B-Probe gegen die echte Klasse, Scratch-Worktree):**

```
$ ./node_modules/.bin/tsc --noEmit -p probe-variance-baseline/tsconfig.json   # HEAD (unmodified)
probe-variance-baseline/probe.ts(4,7): error TS2322: Type 'NDArray<[2, 3]>' is not assignable to type 'NDArray<readonly number[]>'.
  The types returned by 'sum(...).sum(...)' are incompatible between these types.
    Type 'NDArray<AllOnes<[] | [1, 1]>>' is not assignable to type 'NDArray<AllOnes<readonly 1[] | []>>'.
      …
        The type 'readonly 1[]' is 'readonly' and cannot be assigned to the mutable type '[1, 1]'.
EXIT: 1

$ # gleiche Probe, D-V2.1/V2.2 vorhanden, shape NOCH `S` (nicht Readonly<S>) -> IDENTISCHER Fehler, EXIT 1
$ # gleiche Probe, shape jetzt `Readonly<S>` -> EXIT 0 (kein Fehler)
```

Der Pre-Fix-Blocker kam ausschließlich aus `sum(...).sum(...)`s keepdims-Rückgabetyp-Maschinerie
(`AllOnes<S>`, Kern 09): für ein literales `S` erzeugt sie ein MUTABLES Tupel (`[1, 1]`), das
ein `readonly`-Ergebnis für das breite `S` (`readonly 1[]`) nicht erfüllen konnte — eine eher
zufällige Konsequenz der Typ-Maschinerie-Struktur, keine bewusst konstruierte
Kontravarianz-Markierung (`NDArray` trägt keine `in`/`out`-Annotation; seine Varianz war
schon immer gemessen/emergent, s. Spike 01s eigenes Eingeständnis "unsound degradations
happened to satisfy the variance probe"). Das Einwickeln von `shape` in `Readonly<S>` ändert,
wie dieser EINE Members-Vergleich auflöst, und die Weitung gelingt jetzt.

**Sicherheitsbeleg (dieselbe Probe, erweitert):**

```ts
const widened: NDArray<readonly number[]> = nd23;              // jetzt GÜLTIG (war Fehler)
const narrowed: NDArray<[2, 3]> = ndWide;                       // weiterhin FEHLER (Verengung bleibt abgelehnt)
const crossShape: NDArray<[2, 3]> = nd45;                       // weiterhin FEHLER (fremde Literal-Shape bleibt abgelehnt)
```

Verengung UND fremde Shapes bleiben korrekt abgelehnt — nur die Weitungs-Richtung öffnete
sich. Generalität über mehrere Ränge bestätigt (`NDArray<[4]>`, `NDArray<[7,2,9]>`,
`NDArray<[]>` weiten alle sauber zu `NDArray<readonly number[]>`), ebenso dass ops (add/
matmul) über einen so geweiteten Handle weiterhin normal aufrufbar bleiben (das
Gradual-Typing-Verhalten war nie an die Klassen-Varianz gekoppelt). Unter M2 ("never wrong,
only incomplete") ist eine WEITERE statische Aussage nie eine FALSCHERE — eine geweitete
Bindung verspricht bloß weniger, nie etwas Falsches; die Verengungsrichtung (die tatsächlich
gefährliche) bleibt strukturell blockiert.

**Behandlung:** kein Redesign von `AnyNDArray`/der Erasure-Architektur (out of scope für
V2). Stattdessen: (a) der stale gewordene `@ts-expect-error`-Pin wurde durch einen positiven
Pin ersetzt, PLUS einen neuen `@ts-expect-error`-Gegen-Pin für die tatsächlich
sicherheitsrelevante Verengungsrichtung (Regressionsschutz für die eine Richtung, die
wirklich zählt); (b) zwei Doc-Kommentare in `ndarray.ts` (`AnyNDArray`, die "Two honest
caveats"-Passage von `NDArrayView`) wurden korrigiert, weil sie nach D-V2.3 sachlich falsch
geworden wären (Kern-11-Lehre: Doc-Behauptungen müssen der Realität folgen); (c) neuer
FOLLOWUPS-Vorschlag unten (offene Architektur-Frage, nicht in dieser Scheibe entschieden).

**ÜBERHOLT durch die Schließungsrunde (2026-07-13) — s. "Post-Verification-Addendum" unten:**
alles in diesem "Fund 2"-Abschnitt ist eine akkurate historische Aufzeichnung dessen, was
WÄHREND der V2-Implementierung tatsächlich passierte (die Kommandos liefen wirklich, die
Befunde sind real) — es beschreibt aber NICHT mehr den aktuellen Endstand. Der Owner hat die
hier dokumentierte Weitungsfähigkeit als ungeplante Nebenwirkung eingestuft und die
Rück-Invariantierung angeordnet (expliziter `__variance`-Marker auf `NDArray` UND `WNDArray`,
docs/phase-d-vorarbeiten-spec.md-Schließungsauftrag Punkt 1): `NDArray<[2, 3]>` ist NACH der
Schließungsrunde wieder NICHT einer `NDArray<readonly number[]>` zuweisbar — der `widened`-Pin
oben ist re-expressiert zurück zu einer Ablehnung. Details, Beweis und Owner-Begründung im
Addendum.

## Gates (alle frisch, echte Zahlen)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund) | grün | 0 |
| root `check:diag` | **188.520 @ 132** (Budget ≤ 225.000 hart) | 0 |
| `check:diag:stress` | **103.780 @ 82** | 0 |
| `check:diag:browser` | **2.142 @ 75** | 0 |
| `test:core` | 818/818 pass | 0 |
| `test:resident` | 4279 (4277 pass, 2 skip) | 0 |
| `test:browser` | 4/4 pass (~1,1 s) | 0 |
| `test:threaded` | 69/69 pass | 0 |
| `cargo test` | 161/161 pass | 0 |
| `pnpm demo` | läuft durch, inkl. neuer `printArray(rA)`-Zeile via `NDArrayView` | 0 |
| `graph-a-lama . --symbols && graph-a-lama query lint` (Covenant S1) | „keine Verstöße", 0 errors, 0 warnings | 0/0 |
| Artefakt-Hash | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — **exakt der Pin** | — |
| 8 strides-Feld-Assertions | byte-identisch unverändert (git diff bestätigt, s. o.) | — |
| `bench:editor` warm-hover | 0,04–0,07 ms Median (Gate ≤ 1 ms) | PASS |
| `bench:editor` Edit-Toggle | 1,51–2,99 ms Median (Gate ≤ 10 ms) | PASS |
| `bench:editor` Overall-Verdikt | „Overall hard gate: PASS" | 0 |

### Pin-Baseline-Reproduktion (frischer HEAD-Worktree, vor Implementierung)

```
root:    Instantiations: 180794   Files: 132
stress:  Instantiations: 103511   Files: 82
browser: Instantiations: 2142     Files: 75
Artefakt-Hash (clean cargo rebuild): 0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d
```

Alle drei exakt wie im Auftrag vorgegeben reproduziert (180.794 @ 132 / 103.511 @ 82 /
2.142 @ 75), Hash exakt reproduziert.

### Pin-Deltas (Endstand, Haupt-Tree)

| Korpus | Baseline | Endstand | Δ | Dateizahl |
|---|---|---|---|---|
| root | 180.794 | **188.520** | **+7.726** | 132 (unverändert — keine neuen Dateien, Δ ist also NICHT Order-Noise im Sinne der Datei-Additions-Regel) |
| stress | 103.511 | **103.780** | **+269** | 82 (unverändert) |
| browser | 2.142 | **2.142** | **+0** | 75 (unverändert) |

Ehrlich: Δ+7.726 liegt deutlich über der Baustein-0-Grobmessung (der kombinierte
V1+V2-Probepatch maß seinerzeit nur +1.036 gesamt gegen denselben Basis-Pin — V2s Anteil war
dort vermutlich klein/negativ geschätzt). Die tatsächliche Messung dieser Scheibe zeigt einen
deutlich größeren Typkosten-Zuwachs, plausibel erklärt durch: (a) D-V2.3s
`Readonly<S>`-Wrapping an mehreren Klassen/Interface-Stellen, das die Instantiierungs-Struktur
JEDER `.shape`-konsumierenden Stelle im gesamten Korpus berührt (nicht nur die editierten
Zeilen — jede bestehende Site, die `.shape` liest, wird jetzt durch einen zusätzlichen
`Readonly<...>`-Wrap typgecheckt); (b) 56 (korrigiert, s. Fund 1 oben) re-formulierte Test-Pins
selbst tragen neue `readonly`-Wraps in ihren Soll-Typen; (c) die neuen D-V2.2-Kovarianz-Pins
(WNDArray ↔
NDArrayView) sind zusätzliche Instantiierungen. Der Gesamt-Wert bleibt weit unter dem harten
225.000-Budget (188.520 = 3,77 % des 5M-Gesamtbudgets), aber die Diskrepanz zur
Baustein-0-Grobschätzung wird hier bewusst NICHT kleingeredet — die Scheiben-Messung ist
maßgeblich, wie die Spec selbst festlegt, und sie liegt deutlich höher als die Vorab-Grobmessung.

### Pin-Deltas nach der Schließungsrunde (2026-07-13, frisch gemessen, Haupt-Tree)

| Korpus | Prä-Schließung | Post-Schließung | Δ | Dateizahl |
|---|---|---|---|---|
| root (`check:diag`) | 188.520 | **178.212** | **−10.308** | 132 (unverändert) |
| stress (`check:diag:stress`) | 103.780 | **102.096** | **−1.684** | 82 (unverändert) |
| browser (`check:diag:browser`) | 2.142 | **2.142** | **±0** | 75 (unverändert) |

Überraschend auf den ersten Blick: die Schließungsrunde FÜGT Code hinzu (zwei `__variance`-
Marker, sechs neue `@ts-expect-error`-Mutationspins, drei neue `Equal`-Rückgabetyp-Pins, ein
neuer `strided.test.ts`-Runtime-Test, mehrere stark erweiterte Doc-Kommentare) und senkt den
Typkosten-Zähler trotzdem deutlich. Isoliert per Dekomposition (separater Scratch-Worktree,
pristiner V2-Diff appliziert, `check:diag` VOR/NACH einzelnen Schließungs-Schritten gemessen):

- **Item 1 ALLEIN** (nur die zwei Marker + der re-expressierte `stillInvariant`-Pin, sonst
  nichts): 178.140 @ 132 — Δ = **−10.380** gegen den 188.520-Prä-Schließungs-Pin. Das erklärt
  praktisch die GESAMTE Bewegung.
- **Items 2+3+4 zusammen** (die drei Mutationspins, die drei `toNestedArray`-Pins, der neue
  Runtime-Test — Item 3 wirkt sich auf `check:diag` gar nicht aus, da er nur eine
  `tests-runtime`-Datei betrifft, die außerhalb des `check:diag`-Root-Korpus liegt): tragen den
  Rest bei, +72 gegenüber der Item-1-alleine-Messung (178.212 − 178.140 = 72) — ein kleiner,
  plausibler Mehrkosten-Betrag für neun neue Typ-Level-Assertions.

**Plausibler Mechanismus (beobachtet, dekomponiert, NICHT bis auf die Compiler-Ebene
zurückverfolgt — dieselbe epistemische Grenze wie Kern 11s "MECHANISM PINNED ohne vollständige
Erklärung"):** vor der Schließung war `widenedNoLongerErrors` ein ERFOLGREICHER
Zuweisbarkeits-Check (`NDArray<[2,3]>` → `NDArray<readonly number[]>`), und ein ERFOLGREICHER
strukturelle Vergleich muss (mutmaßlich) JEDES Members vollständig auflösen, inklusive der
teuren `sum(...).sum(...)`-`AllOnes`-Kette (Fund 2s eigener Befund: genau DIESE Kette war die
Ursache des Vergleichs). Nach der Schließung ist derselbe Vergleich ein FEHLSCHLAGENDER Check,
der am `__variance`-Member (früh in der Klassen-Deklaration platziert) scheitert — ein
fehlschlagender struktureller Vergleich kann beim ERSTEN inkompatiblen Member abbrechen, ohne
die übrigen (teuren) Member je aufzulösen. Das ist eine plausible, mit der Dekomposition
konsistente Erklärung, aber keine bis auf `tsc`-Interna verifizierte. Root-Zahl bleibt weit
unter dem 225.000-Budget (178.212 = 3,56 % des 5M-Gesamtbudgets, sogar niedriger als vor der
Schließung).

## Covenant-Abschnitt

- **M2 (never wrong, only incomplete):** `Guard`/`OkShape` (ndarray.ts) wurden NICHT
  angefasst — die Guard-Semantik ist byte-identisch zu V1s Endstand. Der einzige
  semantik-relevante Fund (die Varianz-Verschiebung, oben) wurde explizit auf M2-Konformität
  geprüft: nur die WEITUNGS-Richtung öffnete sich (eine weitere statische Aussage, nie eine
  falschere), die Verengungs- und Fremd-Shape-Richtungen bleiben strukturell abgelehnt
  (empirisch bestätigt, s. o.) — kein konfident-falscher Claim entsteht.
- **M3 (Fehler am Argument, Klassen-Hover bleibt sauber):** kein Guard-Message-Pfad wurde
  berührt (D-V2 fasst keine Fehlermeldungen an). Klassen-Hover-Beleg PER ECHTER LSP-Probe:
  `bench:editor` stellt via headless TS7-Server an 22+ Positionen (op-Ketten, matmul,
  reshape, flatten, Slices, dynamische Batch-Shapes über w1–w6) echte
  `textDocument/hover`-Requests und gated JEDEN Sample-Hover gegen den exakten String
  `NDArray<[…]>` (aus `fmtShape()`, gen-workloads.ts:54–56 — eine saubere Tupel-Form ohne
  `readonly`-Präfix); der Lauf endete mit „Overall hard gate: PASS" und ohne eine einzige
  `CORRECTNESS GATE FAILED`-Exception. ZUSÄTZLICH ein direkter Typ-Level-Beweis (Scratch,
  gegen den finalen Haupt-Tree-Stand, `diff` bestätigt Identität):
  ```ts
  const nd = NDArray.zeros([2, 3]);
  type ClassHoverProof  = Expect<Equal<typeof nd, NDArray<[2, 3]>>>;              // PASS
  type MemberHoverProof = Expect<Equal<(typeof nd)["shape"], readonly [2, 3]>>;   // PASS
  // @ts-expect-error - beweist, dass die ALTE (mutable) Member-Form jetzt abgelehnt wird
  type MemberHoverWasMutable = Expect<Equal<(typeof nd)["shape"], [2, 3]>>;       // korrekt abgelehnt
  ```
  Alle drei Assertions bestehen — Klassen-Hover bleibt exakt `NDArray<[2, 3]>`, Member-Hover
  ist jetzt `readonly [2, 3]`, und die Änderung ist nicht vakuos.
- **M5 (browser-sicherer Default-Pfad):** keine neuen `node:*`-Imports in ndarray.ts oder
  resident.ts (beide Diffs enthalten ausschließlich Typ-Änderungen an bestehenden Membern +
  eine Import-Ergänzung von `NDArrayView`, type-only, aus dem bereits-vorhandenen
  `../ndarray.ts`-Modul). `test:browser` (4/4, Playwright/Chromium) lief nach den Änderungen
  grün — keine Regression am COOP-freien Standard-Ladepfad.
- **M4/M1 (ZERO Rust, Hash byte-identisch):** kein `crates/`-Edit; Hash exakt
  `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`, reproduziert sowohl im
  frischen Baseline-Worktree (Clean-Rebuild) als auch nach jedem `pnpm build:wasm`-Aufruf im
  Haupt-Tree während dieser Session.
- **Z1 (keine Dependencies):** unverändert — keine `package.json`-Edits in dieser Scheibe.

## Honesty-Residuum

- Der Fallout-Umfang (ursprünglich als "55 Fehler" berichtet, davon "52" mechanische
  Pin-Re-Formulierungen — beide Zahlen in der Schließungsrunde als intern inkonsistent
  korrigiert: die tatsächliche, zweifach verifizierte Pin-Re-Formulierungszahl ist **56**,
  s. Fund 1 oben — plus 7 `Readonly<any>`-Casts + 1 substanzieller Varianz-Fund) übertraf die
  Spec-Erwartung ("klein") deutlich. Das ist hier vollständig aufgeschlüsselt, nicht
  kleingeredet — inklusive der Zählfehler selbst, die erst die Verify-Runde fand.
- Der Varianz-Fund (Fund 2) wurde EMPIRISCH isoliert (A/B-Probe auf der echten Klasse,
  Baseline-Kontrollprobe gegen den unveränderten HEAD-Stand), aber die TS-interne
  Erklärung, WARUM genau das Ändern des `shape`-Feld-Typs den `sum().sum()`-Members-Vergleich
  beeinflusst (ein strukturell unabhängiger Member), ist nicht bis auf die Compiler-Ebene
  zurückverfolgt — dieselbe Art epistemischer Grenze wie Kern 11s "check-order-dependent,
  non-monotone, MECHANISM PINNED [ohne vollständige Erklärung]". Was verifiziert ist: der
  Effekt ist real, reproduzierbar, isoliert auf genau diese eine Zeile, und in beiden
  sicherheitsrelevanten Richtungen geprüft (Verengung/Fremd-Shape bleiben blockiert).
- Δ+7.726 im Haupt-Pin liegt deutlich über der Baustein-0-Grobmessung (+1.036 für den
  KOMBINIERTEN V1+V2-Probepatch) — die Diskrepanz ist nicht vollständig kausal
  auf einzelne Zeilen zurückgeführt (siehe Pin-Deltas-Abschnitt für die plausiblen
  Beitragsfaktoren), bleibt aber weit unter dem harten Budget.
- `AnyNDArray`s praktische Notwendigkeit für rein lesende heterogene Container — die
  Beobachtung, dass Fund 2 sie möglicherweise teilweise redundant machte — ist durch die
  Schließungsrunde erledigt, nicht offengeblieben: der Owner hat direkt entschieden
  (Re-Invariantierung statt Architekturfrage offenlassen), s. Post-Verification-Addendum.

## FOLLOWUPS.md — Vorschläge zum Austragen/Ergänzen

**Zwei bestehende Items schließen (D-V2 erledigt sie):**

1. Zeile 44 (`NDArrayView` auf die weiteren Backends ausdehnen) → `[x]`, mit Verweis auf
   diese Scheibe: `WNDArray<S> implements NDArrayView<S>` (D-V2.2), strides-Signatur
   harmonisiert (D-V2.1, readonly-Property-Richtung, 0 Aufrufer/0 Test-Brüche), Typtests in
   `ndarray.test-d.ts`/`backend-api.test-d.ts` (Kovarianz für WasmBackend- UND
   ThreadedBackend-erzeugte Handles). Der dort dokumentierte KONKRETE BLOCKER
   (strides-Methode-vs-Feld-Kollision) ist aufgelöst.
2. Zeile 47 (Deep-readonly `shape` entscheiden) → `[x]`, mit Verweis: `Readonly<S>` auf
   allen drei Stellen (NDArrayView/NDArray/WNDArray), gated auf die bestandene D-V2.0-Probe
   (kein TS2636), `@ts-expect-error`-Pins für `nd.shape[0] = 2` / `view.shape[0] = 2` /
   `wnd.shape[0] = 2` (Honesty-Rule-Korrektur, Schließungsrunde 2026-07-13: diese drei Pins
   waren NICHT im ursprünglichen V2-Diff, obwohl die Doc-Kommentare in ndarray.ts/resident.ts
   die Garantie bereits behaupteten — A-Auflage der Verify-Runde, in der Schließungsrunde
   nachgeliefert, s. dort). Klassen-Hover bleibt sauber (LSP-Probe + Typ-Level-Beweis, s. o.),
   Member-Hover trägt jetzt `readonly`.

**Item aus Fund 2 dieser Scheibe — ENTFÄLLT (Schließungsrunde, 2026-07-13):**

3. ~~„`NDArray<[2,3]>` ist seit D-V2.3 ... neu einer `NDArray<readonly number[]>`/
   `NDArray<Shape>` zuweisbar ... offene Architekturfrage für eine spätere Scheibe"~~ — NICHT
   mehr als offenes FOLLOWUPS-Item vorzuschlagen: der Owner hat in der Schließungsrunde
   direkt entschieden (Punkt 1 des Schließungsauftrags), statt die Frage offenzulassen.
   `NDArray`/`WNDArray` sind per explizitem `__variance`-Marker wieder DELIBERAT invariant;
   die Weitungsfähigkeit, die Fund 2 dokumentierte, existiert im Endstand dieser Scheibe nicht
   mehr. Die zugrundeliegende Architekturfrage ("macht eine sichere Weitung `AnyNDArray` für
   den rein-lesenden Fall redundant?") bleibt akademisch interessant, ist aber durch die
   Entscheidung erledigt, nicht offen — kein FOLLOWUPS-Eintrag nötig.

**Neuer Mini-Vorschlag (aus Verifier C's Covenant-Prüfung, Schließungsrunde):**

4. COVENANT.md M3 ("Klassen-Hover bleiben saubere Tupel, `NDArray<[2, 3]>`") ist wörtlich
   mehrdeutig zwischen "der Hover der Klasseninstanz selbst" (`NDArray<[2, 3]>`, unverändert
   durch D-V2.3) und "jeder Member-Hover innerhalb der Klasse" (`.shape` zeigt seit D-V2.3
   `readonly [2, 3]`, nicht mehr `[2, 3]`) — diese Scheibe hat die engere Lesart ("Klassen-
   Hover" = nur der Instanz-Hover) gewählt und begründet (s. `NDArrayView`-Doc-Kommentar,
   ndarray.ts) und Verifier C fand keine Vertragsverletzung darin, aber die Formulierung selbst
   lädt zur Verwechslung ein. Vorschlag für eine künftige Covenant-v2-Überarbeitung (NICHT in
   dieser Scheibe am Covenant geändert, nur als Beobachtung vorgeschlagen): M3 explizit auf
   "die Klasseninstanz-Hover-Signatur, nicht jeden einzelnen Member-Hover" präzisieren.

## Nächste Schritte (laut Spec-Reihenfolge)

Phase-D-Vorarbeiten V1/V2/V3 sind jetzt alle gelandet. Laut
`docs/phase-d-vorarbeiten-spec.md` folgen als Nächstes die union-AXIS-Mini-Scheibe
(`sum(0 as 0|2)`, aus V1s FOLLOWUPS-Fund, release-relevant) und danach Item 11
(Paketschnitt).

## Post-Verification-Addendum (Schließungsrunde, 2026-07-13)

Drei-Verifier-Runde gegen den obigen Diff (Baustein A: Spec-Konformität; Baustein B:
adversarial; Baustein C: Covenant-Check). Diese Session war NICHT die Verifier-Session selbst
— die drei Verdikte unten sind wie an den Executor der Schließungsrunde übergeben; die
"In-Slice-Schließungen" darunter sind vollständig in DIESER Session ausgeführt und verankert
(Kommandos, MD5-Beweise, tsc-Läufe, alle oben bzw. unten zitiert).

### Verdikt A — CONFIRMED mit Auflage

Spec-konform bestätigt, inklusive eines eigenen Mutationsbeweises (`AllOnes`-Rückgabetyp-
Maschinerie als wörtlich bestätigte Ursache von Fund 2, s. o.) und eines Hinweises auf das
`implements`-Keyword-Verhalten (Basis für Fund 2s Isolation). Auflage: die drei
spec-geforderten `@ts-expect-error`-Pins für `nd.shape[0] = ...` / `view.shape[0] = ...` /
das WNDArray-Äquivalent fehlten im ursprünglichen Diff, obwohl D-V2.3 sie explizit
vorschreibt (spec.md: "Pins: `@ts-expect-error` auf `nd.shape[0] = 99` / `view.shape[0] = 99`
/ WNDArray dito") und obwohl die Doc-Kommentare in ndarray.ts/resident.ts die Garantie bereits
behaupteten. Außerdem fand A die Pro-Datei-Pin-Tabelle intern inkonsistent (Summe 59 ≠
berichtete 52; eigene Nachzählung 56).

**Geschlossen (diese Session, Punkt 2 des Schließungsauftrags):** drei neue Pins in
`spike/tests/ndarray.test-d.ts` (`ndForMutation`/`viewForMutation`/`wndForMutation`, jeweils
`.shape[0] = 2` — NICHT `= 99`, s. u. für den Grund). Catchability für alle drei einzeln
bewiesen (drei isolierte Scratch-Mutationen im Haupt-Tree, sofort revertiert, MD5-Identität
vor/nach jedem Revert bestätigt): `NDArray.shape` → `S` lässt NUR den `ndForMutation`-Pin als
"Unused '@ts-expect-error' directive" rot werden (Zeile 193); `NDArrayView.shape` → `S` lässt
NUR den `viewForMutation`-Pin rot werden (Zeile 197, plus zwei erwartete Kollateral-Fehler an
Nutzstellen, s. Fund 8/(b)-Präzisierung oben); `WNDArray.shape` → `S` lässt NUR den
`wndForMutation`-Pin rot werden (Zeile 201). Zusatzbefund während der Umsetzung: die
Spec-wörtliche Formulierung `nd.shape[0] = 99` ist ein schlechter Testwert — `99` ist für ein
literales `[2, 3]`-Tupel bereits durch Literal-Typ-Verengung (nicht durch `readonly`) ein
Fehler (`TS2322`, empirisch per Scratch-Probe bestätigt: `declare const t: [2, 3]; t[0] = 99`
schlägt auch OHNE jedes `readonly` fehl), sodass ein `= 99`-Pin auch nach einem Revert von
`Readonly<S>` fälschlich "grün" bliebe (der `@ts-expect-error` würde weiterhin einen — nur
anderen — Fehler unterdrücken und NICHT als "unused" auffallen). Gepinnt wurde stattdessen
`= 2` (derselbe Wert, den die Position bereits trägt) — das isoliert den Test eindeutig auf
`readonly`, bewiesen durch dieselben drei Mutationen. Ergebnisdoc-Korrektur zur
Pin-Tabelle: s. Fund 1 oben (korrigiert auf 56, zweifach verifiziert — mechanisch UND per
Mutation, in dieser Session neu und unabhängig nachgemessen, nicht nur A's Zahl übernommen).

### Verdikt B — HÄLT mit Befunden

Adversarial geprüft: alle 12 Methoden auf geweiteten Handles bleiben sicher aufrufbar (Fund
2s "ops funktionieren weiter"-Behauptung bestätigt); Getter-Lücke gefunden (kein Pin auf den
tatsächlichen `NDArray.strides`-Getter-WERT — nur auf Aufrufstellen-Zahl und WNDArray-Feld);
`toNestedArray`-Rückgabetyp-Grenze gefunden (kein Schutz gegen eine künftige Verengung von
`unknown`); Richtungs-Asymmetrie bei "alles-oder-nichts kollidiert mit `implements`" (Befund
8, s. u.); Hover-Gate strukturell nicht-vakuös bestätigt (die drei Typ-Level-Assertions im
Covenant-Abschnitt oben).

**Geschlossen (diese Session, Punkte 3+4 des Schließungsauftrags):**
- **B3 (Getter-Lücke):** neuer Test `"NDArray.strides getter: values match hand-computed
  row-major strides (rank 1/2/3)"` in `spike/tests-runtime/strided.test.ts`, drei Shapes
  (Rang 1 `[5]`→`[1]`, Rang 2 `[3,4]`→`[4,1]`, Rang 3 `[2,3,4]`→`[12,4,1]`), Erwartungswerte
  von Hand aus der Zeilenmajor-Formel abgeleitet (NICHT über `computeStrides` — das wäre
  zirkulär gegen exakt die Funktion, die der Getter selbst aufruft), plus eine
  Getter-Frischheits-Assertion (`a.strides !== a.strides`, gleiche Werte trotzdem).
  Catchability: Getter-Mutant `[...computeStrides(this.shape)].reverse()` in-Tree, sofort
  revertiert (MD5-Identität bestätigt) — der neue Test schlägt exakt an der Rang-2-Assertion
  fehl (`AssertionError`, `actual: [1, 4]` vs `expected: [4, 1]`); alle 729 Tests in
  `strided.test.ts` grün nach Revert.
- **B5 (toNestedArray-Grenze):** drei `Equal`-Pins in `ndarray.test-d.ts`
  (`ToNestedArrayViewReturn`/`ToNestedArrayNDReturn`/`ToNestedArrayWNDReturn`, je
  `Equal<ReturnType<...["toNestedArray"]>, unknown>` für `NDArrayView<[2,3]>`/`NDArray<[2,3]>`/
  `WNDArray<[2,3]>`). Catchability: `NDArray.toNestedArray()`-Rückgabetyp in-Tree auf
  `unknown[]` verengt (Body-Cast `as unknown[]` hinzugefügt, damit die Funktion selbst noch
  compiliert), sofort revertiert (MD5-Identität bestätigt) — GENAU der `ToNestedArrayNDReturn`-
  Pin (Zeile 214) wird rot, keine Kollateralschäden (da `unknown[]` weiterhin `unknown`
  erfüllt, bleibt `implements NDArrayView<S>` unberührt — die Isolation ist sauber).
- **Befund 8 (Richtungs-Asymmetrie):** präzisiert im "alles-oder-nichts"-Abschnitt oben, mit
  frischer empirischer Bestätigung dieser Session (drei isolierte Einzel-Stellen-Mutationen:
  Klasse-readonly-er-als-Interface wird strukturell an Nutzstellen gefangen, TS2322;
  Klasse-mutabler-als-Interface wird NUR von den 56 Equal-Pins gefangen, KEIN
  `implements`-Fehler).
- **B4 (Readonly<any>-Cast-Präzision):** präzisiert oben (Fund 1, `Readonly<any>`-Absatz):
  nur Zuweisung/Iteration brauchen den Cast, `.length`/Index lösen `any` auf — per
  Scratch-Probe bestätigt (`TS2740` bei Zuweisung, `TS2488` bei Spread, KEIN Fehler bei
  `.length`/`[0]`).

### Verdikt C (Covenant) — keine Invarianten-Verletzung

M2/M3/M5/M4/M1/Z1/S1 einzeln geprüft, keine Verletzung. Zwei niedrig-priore
Vertrags-Präzisions-Notizen: (i) M2 wird im `AnyNDArray`-Doc-Kommentar als Analogie
zitiert, nicht als Zertifikat für Klassen-Assignability (der Covenant-Text selbst deckt
Assignability nicht ab) — **geschlossen** (diese Session, Punkt 5(d)): der
`AnyNDArray`-Doc-Kommentar in ndarray.ts wurde ohnehin komplett neu gefasst (Re-
Invariantierung, s. u.) und zitiert M2 jetzt explizit als Analogie ("die gleiche Form wie
COVENANT.md's M2-Prinzip … dies ist eine Analogie, kein Zertifikat, das M2 selbst ausstellt").
(ii) M3s Wortlaut ist zwischen Klassen- und Member-Hover mehrdeutig — als Mini-Vorschlag ins
FOLLOWUPS-Textproposal aufgenommen (s. FOLLOWUPS-Abschnitt oben, Punkt 4), NICHT am Covenant
selbst geändert (Auftragsgrenze).

### Owner-Entscheidung: Re-Invariantierung (Punkt 1 des Schließungsauftrags)

Der Owner hat die in Fund 2 dokumentierte, ungeplante Weitungsfähigkeit NICHT als
Architekturfrage offengelassen, sondern direkt entschieden: **Re-Invariantieren** — ein
bewusster, deklarierter Marker statt der zufälligen (post-D-V2.3: zufällig AUFGEHOBENEN)
Kovarianz. Begründung (aus dem Schließungsauftrag): Fund 2 selbst zeigt, dass die Invarianz
bis V2 ein ZUFALLSPRODUKT der keepdims-`AllOnes`-Rückgabetyp-Maschinerie war (von Verdikt A
wörtlich im Baseline-`tsc`-Fehler bestätigt) — eine Eigenschaft, die als Nebenwirkung eines
strukturell unabhängigen Members entstand und durch D-V2.3 als Nebenwirkung wieder verschwand.
Unbeabsichtigte Varianz ist genau das Fehlerbild, das dieses Codebase sonst ablehnt (Spike 01s
Prinzip: Varianz muss ENFORCED sein, nicht bloß GEMESSEN — vgl. `NDArrayView`s geprüftes
`out S`). Die Umsetzung:

- **Mechanismus:** `private declare readonly __variance: (s: S) => S;` als erstes Member in
  `NDArray` (`spike/src/ndarray.ts`) UND `WNDArray` (`spike/src/wasm/resident.ts`).
  Property-Stil ist zwingend (Method-Shorthand wird bivariant geprüft und wäre wirkungslos —
  exakt Baustein Bs Probe-3-Befund, wörtlich im Schließungsauftrag zitiert und hier
  respektiert, nicht neu hergeleitet). `declare` = kein Laufzeit-Feld (nichts initialisiert
  oder weist zu — reines Compile-Zeit-Gerät); `private` = nie in der öffentlichen Surface
  oder einem Hover.
- **Geprüft (diese Session):** `implements NDArrayView<S>` bleibt für beide Klassen intakt
  (private Member stören `implements` nicht — bestätigt: `pnpm exec tsc --noEmit` nach
  Marker-Zugabe zeigt GENAU EINEN Fehler, exakt an der einen Stelle, die neu abgelehnt werden
  soll — s. u.); Klassen-Hover bleibt `NDArray<[2, 3]>` (der Marker hat keinen eigenen Shape-
  Beitrag, unverändert durch den Covenant-Abschnitt oben belegt); `AnyNDArray`/`AnyWNDArray`-
  Flüsse unverändert (`any` umgeht die Varianzprüfung unabhängig vom Klassen-Marker — die
  drei `anyErased`/`anyList`/`wndAssignable`-Pins bleiben grün).
- **Konsequenz-Pin:** `widenedNoLongerErrors` (ndarray.test-d.ts, ehemals Zeile ~156) wurde zu
  `stillInvariant` umbenannt und intent-erhaltend zu einer `@ts-expect-error`-Ablehnung
  re-expressiert, mit einer History-Passage, die alle DREI Zustände (pre-D-V2.3 zufällig
  invariant → D-V2.3 zufällig geweitet → Schließungsrunde bewusst re-invariantiert) ehrlich
  festhält.
- **Vollständige `tsc --noEmit`-Suche nach WEITEREN betroffenen Pins:** in einem separaten,
  isolierten `git worktree` (pristiner V2-Diff, VOR jeder anderen Schließungs-Änderung dieser
  Session appliziert, per `git apply` desselben gesicherten Diffs) beide Marker probeweise
  zugefügt und `tsc --noEmit` gegen root/stress/browser gefahren: root lieferte GENAU EINEN
  Fehler (an der damaligen `widenedNoLongerErrors`-Zeile, der Original-Position vor der
  Umbenennung), stress und browser je null. Dieser isolierte Vorablauf war die Grundlage für
  die gezielte Pin-Korrektur im Haupt-Tree (die dort sofort mit re-expressiertem Pin statt
  mit dem alten positiven Pin eingeführt wurde — der Haupt-Tree hat den "einen Fehler"-Zustand
  selbst nie durchlaufen, nur der isolierte Vorablauf). Nach der Umsetzung im Haupt-Tree (Marker
  + re-expressierter Pin gemeinsam) liefen alle drei Korpora dort grün, s. Gates unten. Keine
  weiteren V2-Pins pinnen die KLASSEN-Weitung; die NDArrayView-WIDENING-Pins aus D-V2.2
  (`widenedToShape`, `nd23AsView`, `wnd23AsWidenedView` etc.) bleiben unberührt, weil sie die
  INTERFACE-Weitung prüfen (unverändert kovariant), nicht die Klassen-Weitung.
- **Catchability-Nachweis (Marker entfernt → Pin feuert):** `__variance`-Member in
  `NDArray` in-Tree auskommentiert, sofort revertiert (MD5-Identität bestätigt) — der
  re-expressierte `stillInvariant`-Pin wird exakt an seiner Zeile als "Unused
  '@ts-expect-error' directive" rot; nach Revert wieder grün.
- **Nebenbefund (WNDArray war ebenfalls betroffen, nicht in Fund 2 dokumentiert):** ein
  Scratch-Probe vor der Umsetzung (isolierter `git worktree`, pristiner V2-Diff appliziert)
  zeigte, dass `WNDArray<[2, 3]>` VOR dem Marker ebenfalls silently einer
  `WNDArray<readonly number[]>` zuweisbar war (dieselbe D-V2.3-Nebenwirkung wie bei `NDArray`,
  aber ohne dass ein bestehender Pin das je bemerkt hätte — kein Pin im V2-Korpus prüfte
  Klassen-Weitung auf der WNDArray-Seite direkt). Der `AnyWNDArray`-Doc-Kommentar in
  resident.ts (der bereits VOR V2 unverändert eine "measured variance invariant"-Behauptung
  trug) war also seit D-V2.3 bereits sachlich veraltet, ohne dass es dokumentiert worden wäre
  — jetzt korrigiert (s. Diff).
- **Doc-Kommentare neu gefasst:** `AnyNDArray` (ndarray.ts) komplett neu — trägt jetzt die
  volle Drei-Zustände-Geschichte (zufällig invariant → zufällig geweitet → bewusst
  re-invariantiert) und zitiert M2 als Analogie, nicht als Zertifikat (Verdikt-C-Fix
  eingearbeitet); `AnyWNDArray` (resident.ts) aktualisiert (der Nebenbefund oben); die
  `NDArrayView`-Doc-Kommentar-Passagen zu `strides`/`shape: Readonly<S>` blieben unverändert
  (sie beschreiben die View, nicht die Klassen-Varianz — durch den vollen Root-Check als
  weiterhin akkurat bestätigt).

### Gates (Schließungsrunde, frisch, Haupt-Tree, 2026-07-13)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund: root + stress + browser) | grün | 0 |
| `pnpm test:core` | 818/818 pass | 0 |
| `pnpm test:resident` | 4280 (4278 pass, 2 skip — Δ+1 ggü. 4279 vor der Schließung, von B3s neuem Runtime-Test) | 0 |
| `pnpm test:browser` | 4/4 pass (~1,1 s) | 0 |
| `pnpm test:threaded` | 69/69 pass | 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161 (Haupt-Suite) + 1 (`zero_alloc.rs`) + 0 (Doctests) = 162, alle bestanden | 0 |
| `pnpm demo` | läuft komplett durch, TS/WASM-v1/WASM-resident stimmen auf jeder gezeigten Op überein | 0 |
| `graph-a-lama . --symbols` | 145 files, 1102 symbols, 2650 references (1 mehrdeutige inherits-Kante, keine Kante gesetzt — informativ, kein Fehler) | 0 |
| `graph-a-lama query lint` (Covenant S1) | „keine Verstöße", 0 errors, 0 warnings | 0 |
| Artefakt-Hash | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — exakt der Pin, bestätigt per ECHTEM Clean-Rebuild (`cargo clean` + `cargo build --release`, 1,04 s Neu-Kompilierung, nicht aus dem Cache) | — |
| `pnpm bench:editor` M2 (warm hover, Gate ≤ 100 ms/200 ms) | 0,05–0,07 ms Median über alle 22 Positionen, jede PASS | 0 |
| `pnpm bench:editor` M3 (Edit-Toggle, Gate ≤ 500 ms/1000 ms) | 1,56–3,37 ms Median über 4 Positionen, jede PASS (leicht über dem V2-Original-Bereich 1,51–2,99 ms, aber weit im 10-ms-harten-Gate und ohne jede `CORRECTNESS GATE FAILED`-Exception — 0 Treffer bei `grep -c "CORRECTNESS GATE FAILED"`) | 0 |
| `pnpm bench:editor` Overall-Verdikt | „Overall hard gate: PASS" | 0 |
| `check:diag` (root) | 178.212 @ 132 Dateien | 0 |
| `check:diag:stress` | 102.096 @ 82 Dateien | 0 |
| `check:diag:browser` | 2.142 @ 75 Dateien (unverändert) | 0 |

M3-Hover-Korrektheit (der Marker darf NICHTS an Hovern ändern): `bench:editor`s M2-Gate prüft
JEDEN Sample-Hover gegen den exakten String `NDArray<[…]>` (`fmtShape()`) — alle 22 Positionen
PASS, keine `CORRECTNESS GATE FAILED`-Exception im gesamten Lauf. Ergänzend strukturell
garantiert: der Marker ist `private` (nie in der öffentlichen Surface) und trägt selbst keinen
Shape-Beitrag — `NDArray.zeros`s Rückgabetyp-Signatur (`NDArray<Mutable<S>>`) ist durch die
Schließung nicht editiert worden, der Klassen-Hover kann sich also strukturell gar nicht
geändert haben.

### Was bleibt offen

- Das Kern-07-Addendum-1/Facette-(b)-Diskrepanz-Item (V1, unverändert von dieser Scheibe).
- Die union-AXIS-Mini-Scheibe (`sum(0 as 0|2)`) — als Nächstes laut Spec-Reihenfolge.
- `-1`-Inferenz auf `reshape` (Kern 08, unverändert).
- NaN-Payload-Regression nur für `transpose` getestet (Kern 10, unverändert).
- Der COVENANT-v2-Präzisierungsvorschlag zu M3 (FOLLOWUPS-Textproposal Punkt 4 oben) — bewusst
  NICHT in dieser Scheibe am Covenant umgesetzt.
