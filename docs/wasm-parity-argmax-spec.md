# WASM-Parität S4: `argmax` auf `WNDArray`/threaded — bindende Spec

Status: **bindend** (Owner-Auftrag 2026-07-24: „Starte mit S4 der WASM-Parität-Kampagne
(argmax auf WNDArray/threaded) nach dem etablierten Muster") ·
Version: **3** (v2 nach Baustein 0 — Befunde F1/D1/D2/F2/D3 eingearbeitet, Absolut-Gate
begründet angehoben; **v3 nach der Verify-Runde A+B+C** — zwei Testpflichten aus
Baustein-B-Befunden ergänzt, `keepDimsShape`-Zählung erneut korrigiert; beide Addenda am
Dateiende) · Datum: 2026-07-24 ·
Eskalationsleiter: **Stufe 3** (voller Verify-Katalog A+B+C, Baustein 0 vor der ersten
Codezeile).

Vorgänger-Scheiben derselben Kampagne: S0 `sqrt` (docs/wasm-parity-sqrt-spec.md), S1
Skalar-Overloads (docs/wasm-parity-scalar-spec.md), S2 `mean`
(docs/wasm-parity-mean-spec.md), S3 `item` + `stack`
(docs/wasm-parity-item-stack-spec.md). Semantische Primärquelle für die Op selbst:
**docs/op-w1-argmax-topk-spec.md, D4** (Totalordnung) — sie wird hier ZITIERT, nicht neu
hergeleitet.

## Ziel & Warum

`NDArray.argmax()` existiert seit Op-Scheibe W1 (2026-07-20) bewusst nur auf der naiven
JS-Klasse — dokumentierte Surface-Asymmetrie, FOLLOWUPS-getrackt. Diese Scheibe schließt
die Lücke für `argmax`: `WNDArray` (und damit automatisch der threaded-Pfad, gleiches
Crate) bekommt dieselbe Op mit derselben gepinnten Semantik, bit-identisch zur
TS-Referenz `argmaxRuntime`. `topk` bleibt S5.

## Arbeitsregel-11-Prüfung (Pflicht-Vorabfrage) — **Verdikt: KERNEL**

Arbeitsregel 11 (CLAUDE.md) verlangt vor jedem neuen Kernel die Frage, ob die Op
definitorisch eine **Komposition bereits verifizierter Kernel** ist. Für `argmax` lautet
die Antwort **nein**, mit zwei geprüften Alternativen:

- **(a) Komposition bestehender Kernel — ausgeschlossen.** Der Kernel-Bestand
  (`add`/`sub`/`mul`/`div`, `scalar_*`, `sum_all`/`sum_axis`, `matmul`, `dot`,
  `norm_sq`, `sqrt`, `transpose`, `materialize`, `fill`) enthält **keine
  Vergleichs-/Maximum-Operation und keine Index-produzierende Operation**. Es gibt keinen
  Ausdruck über diesen Kerneln, der `argmax` berechnet. Anders als S2 (`mean` =
  `sum ∘ scalar_div`) und S3 (`stack` = N × `nt_materialize`) existiert hier kein
  Kompositionspfad.
- **(b) Kernel-los in TS über `core.memory.buffer` (das S3-`item`-Muster) — geprüft und
  VERWORFEN.** Technisch möglich (eine `Float64Array`-Sicht auf den Core-Speicher, dann
  `argmaxRuntime`s Schleife in TS), aber:
  - `item` war ein **einzelner** strided Skalar-Read (O(1), keine Schleife). `argmax` ist
    eine **echte O(N)-Reduktion mit Totalordnung** — dieselbe Klasse wie `sum`, `dot`,
    `norm_sq`, die alle drei einen Kernel haben. Ein TS-Loop über residenten Speicher
    bricht die Konsistenz der Reduktions-Fläche.
  - Der Zweck der Residenz ist, dass die Rechnung im Core stattfindet; eine O(N)-Schleife
    in TS über den WASM-Puffer macht `WNDArray.argmax` strukturell zur JS-Op mit
    WASM-Speicher, nicht zu einer residenten Op.
  Diese Alternative ist ausdrücklich als Alternative dokumentiert, damit Baustein 0 sie
  angreifen kann; die Entscheidung ist bindend, wenn Baustein 0 sie nicht kippt.

**Konsequenz:** **M1 bindet neu** (erster echt neuer Kernel seit S1) und der
**Freeze-Hash bewegt sich legitim** — Beweismuster S0/S1 (additive-only-Dekomposition +
Pre-Edit-Clean-Rebuild, D7). **Arbeitsregel 10 greift** (zwei neue `CoreExports`-Member
→ zwei `notImplemented`-Stubs in `backend-oom.test.ts`, D6).

## Berührte Covenant-Invarianten (v5)

- **M1** (Anker `crates/core/src/`, `spike/src/runtime.ts`): **bindet neu.** Die beiden
  neuen Kernel müssen bit-identisch zu `argmaxRuntime` (runtime.ts:577) sein, auch für
  IEEE-Spezialwerte. Weil das Ergebnis ein **integraler Index** ist (kein arithmetisch
  gerundeter Wert), ist Bit-Identität hier eine Aussage über die **Totalordnung und die
  Tie-Auflösung**, nicht über Float-Akkumulation — genau deshalb ist der Spezialwert-Teil
  des Differentialtests (NaN, ±0, ±Inf, Subnormals) load-bearing, nicht Kür.
- **M3** (`sym:Guard`, `sym:ShowShape`): die drei Overloads reichen den `ReduceAxis`-Guard
  durch; Compile-Ablehnung erscheint AM `axis`-Argument, der Runtime-Throw trägt denselben
  Stem. Klassen-Hover bleiben `WNDArray<[2, 3]>`. **Arbeitsregel 13 greift** → LSP-Messung
  ist Pflichtbestandteil des Verify-Katalogs (D8).
- **M4** (Anker `abi.rs`, `matmul_blocked.rs`, `shape.rs`): append-only-Disziplin, D7.
- **M2**: keine neue Typ-Maschinerie (vierte Call-Site von `ReduceAxis`/`Guard`/`OkShape`
  nach `sum`/`mean` auf `WNDArray` und `sum`/`argmax` auf `NDArray`) — `reduce.ts` wird
  NICHT editiert. Die Degradationskanten sind die bereits bewiesenen.
- **M5**: NICHT berührt (keine node:*-Imports). **Z1**: NICHT berührt.
  **Z2**: keine neue Quelltext-Korpus-Datei außerhalb der bestehenden Gates.

## Bindende Entscheidungen

### D1 — Surface-Scope

`WNDArray.argmax` als **Instanz-Methode** (drei Overloads, D2). Threaded-Parität ergibt
sich automatisch (`ThreadedBackend` liefert `WNDArray`, `ThreadedCoreExports extends
CoreExports`, dasselbe Crate) und wird per Test belegt, nicht per Code.

**KEINE Änderung an `WasmBackend`/`ThreadedBackend`** (spike/src/wasm/backend-api.ts,
backend.ts, threaded.ts). Begründung/Abgrenzung zu S3: dort war `stack` **statisch** und
wäre für Paketkonsumenten unerreichbar gewesen, weil `WNDArray` nicht aus index.ts
exportiert ist. `argmax` ist eine Instanz-Methode auf einem Handle, das Konsumenten
ohnehin über die Facade erhalten — dieselbe Lage wie `sum`/`mean`/`sqrt`, die alle keine
Facaden-Änderung bekamen.

`topk` ist **NICHT** im Scope (S5).

### D2 — API-Form (exakter Spiegel von `sum`/`mean` auf `WNDArray` + `NDArray.argmax`)

```ts
argmax(): number;
argmax<const Axis extends number | undefined>(
  axis: Guard<ReduceAxis<S, Axis>, Axis>,
): WNDArray<OkShape<ReduceAxis<S, Axis, false>>>;
argmax<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
  axis: Guard<ReduceAxis<S, Axis>, Axis>,
  keepdims: KeepDims,
): WNDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
```

- Die **niladische Form gibt `number`** zurück (nicht `WNDArray<[]>`) — exakt die
  W1-D2-Abweichung, damit die beiden Flächen dieselbe Signatur tragen. Präzedenz auf
  `WNDArray` selbst: `dot(): number` (resident.ts:1191 ff.) und `norm(): number`
  (resident.ts:1251) lesen ebenso einen Skalar aus einem ephemeren Scratch-Puffer und
  verlassen die `WNDArray`-Welt.
- **Overload-Reihenfolge ist load-bearing** (Arbeitsregel 2): Guard-Träger zuletzt. Die
  Reihenfolge niladisch → 1-Arg → 2-Arg ist byte-strukturgleich zu `sum` (resident.ts:1028)
  und `mean` (resident.ts:1161); die Implementierungssignatur trägt `axis?`/`keepdims?`
  wie dort.
- `ReduceAxis`/`OkShape`/`Guard` werden **unverändert wiederverwendet** — `reduce.ts` wird
  nicht editiert.
- Ergebnis-DATEN der Achsen-Formen sind **f64-integrale Indizes** (numtype ist f64-only) —
  im Doc-Kommentar zu benennen, wie auf `NDArray.argmax`.
- Der Rückgabetyp wird **ausgeschrieben** (`WNDArray<OkShape<ReduceAxis<…>>>`), **kein
  Top-Level-Alias in Rückgabeposition** — S3-Arbeitsregel-13-Lektion: ein Alias in
  RÜCKGABE-Position bleibt in der Quick Info namentlich stehen und verletzt M3; in
  TYP-ARGUMENT-Position (wie hier) wird er aufgelöst.

### D3 — Kernel-Design (zwei neue ABI-Einstiegspunkte)

Beide spiegeln **signatur-identisch** die `sum`-Zwillinge:

```rust
#[no_mangle] pub extern "C" fn nt_argmax_all_strided(
    shape_ptr: u32, rank: u32, strides_ptr: u32, offset: u32,
    data_ptr: u32, data_len: u32, out_data_ptr: u32,
) -> u32;                          // 7 Parameter, wie nt_sum_all_strided (abi.rs:512)

#[no_mangle] pub extern "C" fn nt_argmax_axis_strided(
    shape_ptr: u32, rank: u32, strides_ptr: u32, offset: u32,
    data_ptr: u32, data_len: u32, axis: i32, out_data_ptr: u32, out_len: u32,
) -> u32;                          // 9 Parameter, wie nt_sum_axis_strided (abi.rs:552)
```

Rechenkern in **neuer Datei** `crates/core/src/kernels/argmax.rs` mit
`argmax_all_strided` / `argmax_axis_strided`, deren **Schleifenstruktur byte-strukturgleich
zu `sum_all_strided` / `sum_axis_strided`** (kernels/sum.rs:68 / :89) ist — dieselbe
Validierung (`checked_element_count`, `validate_strided_bounds`), dieselbe
Achsen-Normalisierung, dieselbe `data.get(off).copied().unwrap_or(0.0)`-Konvention
(spiegelt `argmaxRuntime`s `?? 0`), derselbe `KernelError::ShapeIncompatible` für eine
Achse außer Reichweite. Gegenüber den `sum`-Vorbildern ändern sich **genau zwei Dinge**
(v2-Präzisierung nach Baustein-0-Befund D3 — „ersetzt wird ausschließlich die
Akkumulation" war zu eng formuliert): (i) die Akkumulation `total += v` wird durch den
Challenger-Vergleich aus D4 plus die Index-Buchführung ersetzt, und (ii) es kommen die
beiden **neuen** Leer-Zweige hinzu (`size == 0` bzw. `axis_dim == 0` → früher
`ShapeIncompatible`), die `sum` nicht hat (dort laufen null Iterationen und das Ergebnis
`0.0` ist wohldefiniert).

**Pflicht-Doc-Kommentar zur `unwrap_or(0.0)`-Konvention** (v2, Baustein-0-Befund D2):
Der Fallback ist bei `sum` das additive Neutrale und damit ein echtes No-op; bei `argmax`
wäre dieselbe `0.0` ein **Vergleichs-Kandidat in der Totalordnung** — sie könnte gewinnen
oder verlieren. Der Pfad ist nicht erreichbar (`validate_strided_bounds`,
`crates/core/src/shape.rs:165-180`, beweist `max_reach < data_len` für jede
Achsen-Kombination, und beide Einstiegspunkte rufen es vor der Schleife auf), aber
`argmax`s **Korrektheit** — nicht nur seine Speichersicherheit wie bei `sum` — hängt ab
jetzt an dieser Zusicherung. Das gehört als ein Satz in den Doc-Kommentar von
`argmax.rs`, nicht in mündliche Überlieferung.

**Der tragende Semantik-Punkt (Regel 12 in Kernel-Form):** `nt_argmax_all_strided` liefert
den Index in die **LOGISCHE row-major-Abflachung der VIEW**, nicht in den Speicher. Auf
einer transponierten/geslicten View müssen `iterationsindex` und `Speicheroffset`
auseinanderfallen — das ist exakt die Falle, die `sum_all_strided`s Doc-Kommentar
(kernels/sum.rs:60-67) für die Akkumulationsreihenfolge beschreibt, hier in der
Index-Domäne. Der Rückgabewert ist der `flat`-Zähler, NIE der berechnete `off`.
Bei der Achsen-Form ist der Index der Index **entlang der Achse** (`0..axis_dim`) und
damit stride-unabhängig.

**Leerer Fall im Kernel:** eine Reduktion über null Elemente hat keinen Index. Der Kernel
gibt dafür `KernelError::ShapeIncompatible` zurück (`nt_argmax_all_strided` bei
`size == 0`; `nt_argmax_axis_strided` bei `axis_dim == 0`). Dieser Pfad ist von TS aus
**unerreichbar** (D4 prävalidiert), wird aber per cargo-Test gepinnt (defense in depth,
wie die bestehenden abi.rs-Prävalidierungs-Tests).

### D4 — Runtime-Semantik: Totalordnung + Fehlerpfade (zitiert, nicht neu erfunden)

**Totalordnung — wörtlich aus docs/op-w1-argmax-topk-spec.md D4, implementiert in
`beatsMax` (runtime.ts:555):** NaN gilt als MAXIMAL; bei mehreren NaN gewinnt der ERSTE
Index; bei Wert-Gleichheit (inkl. `0 === -0`, plain `>`, nie `Object.is`) gewinnt der
ERSTE Index. Ein Element schlägt das laufende Maximum **gdw.**

```
(isNaN(el) && !isNaN(max)) || el > max
```

Rust-Entsprechung: `(el.is_nan() && !max.is_nan()) || el > max`. `f64`-`>` ist in Rust
wie in JS IEEE-754-konform (NaN-Vergleiche falsch, `0.0 > -0.0` falsch), `f64::is_nan`
entspricht `Number.isNaN` — die Bit-Identität folgt aus der Gleichheit dieser drei
Primitiven, wird aber **nicht** als selbstverständlich behauptet, sondern differentiell
bewiesen (D6/T3).

**Fehlerpfade — TS prävalidiert, damit die Message-Stämme wortgleich sind (M3,
Cross-Surface-Parität):** `WNDArray.argmax` prüft VOR jeder Allokation und vor jedem
Kernel-Aufruf und wirft die **wortgleichen** Stämme von `argmaxRuntime`:

| Fall | Stem |
|---|---|
| niladisch, `product(shape) === 0` | `argmax: attempt to get argmax of an empty array` |
| Achse außer Reichweite | `reduce: axis ${axis} is out of range for shape [${shape}] (rank ${rank})` |
| Achsen-Dim `=== 0` | `argmax: attempt to get argmax of an empty array` |

Die Achsen-Prüfung ist **byte-strukturgleich** zu der, die `WNDArray.sum` bereits
durchführt (resident.ts:1077-1081) — inklusive Negativ-Achsen-Normalisierung. Die
size-0-Prüfung ist neu (weder `sum` noch `mean` werfen dort). Ein Kernel-Status ≠ 0 wirft
weiterhin die etablierte `wasm resident nt_…: status …`-Form.

**Rang 0 (`[]`, ein Element):** niladisch → `0`. `keepdims` folgt dem bestehenden
`keepDimsShape`-Helfer (runtime.ts:532); dessen dokumentierter „Call-Sites
prävalidieren"-Kontrakt wird eingehalten (Validierung läuft davor). Der Helfer hat vor dieser Scheibe
**fünf** Produktions-Call-Sites (`ndarray.ts` in `sum`:612, `argmax`:788, `mean`:860;
`resident.ts` in `sum`:1065 und :1087). **`WNDArray.argmax` fügt ZWEI hinzu** (je eine im
Voll-Reduktions- und im Achsen-Zweig, symmetrisch zu `sum`s zweien) — nachher also
**sieben** (v3-Korrektur nach Baustein-A-Befund; v2 sagte „die sechste" und zählte nur
eine der beiden neuen Stellen, v1 zählte die Basis mit vier statt fünf und hatte übersehen,
dass `NDArray.argmax` seit W1 selbst eine Call-Site ist. Reine Zähl-Präzision — das
Code-Muster war in allen drei Fassungen dasselbe und korrekt). Der offene FOLLOWUPS-Mini
„defensiver Achsen-Assert" bleibt unverschärft offen, keine Edits an bestehenden
runtime.ts-Funktionen.

### D5 — Lifecycle / Speicher-Disziplin

- Beide Zweige folgen dem `sum`-Muster: `scratch: ScratchBuf[]` + `try/finally`, Shape- und
  Strides-Puffer im Scratch, Ausgabepuffer per `allocBytes`, bei Status ≠ 0 wird der
  Ausgabepuffer **vor** dem Throw freigegeben (resident.ts:1062/1109).
- **Niladisch:** Ausgabe ist 1 f64. Der Skalar wird — wie bei `dot`/`norm` — aus dem
  ephemeren Puffer gelesen und der Puffer **im selben Aufruf freigegeben**; die Methode
  gibt ein plain `number` zurück und hinterlässt **null** residente Allokation. Das ist
  eine eigene Testpflicht (exakte Alloc/Free-Bilanz, D6 — S3-Lektion: Leck-Tests, die nur
  Seitengranularität messen, sind für 8-Byte-Scratch vakuös).
- **Achsen-Form:** Ausgabepuffer wird per `WNDArray.fresh` an das Ergebnis-Handle
  übergeben (kein Free im Erfolgspfad), exakt wie `sum`.
- `this.assertLive("argmax")` als erste Anweisung, wie jede andere Methode.

### D6 — Datei-Disziplin (Freeze + Pin-Schutz)

| Datei | Disziplin |
|---|---|
| `crates/core/src/kernels/argmax.rs` | **NEU** (verschiebt keine Bestands-Bytes) |
| `crates/core/src/kernels/mod.rs` | **ein Append** (`pub mod argmax;`, Präzedenz S0/S1) |
| `crates/core/src/abi.rs` | **zwei Appends strikt ans Dateiende** (nach `nt_scalar_div_strided`, dem heutigen letzten Item) — M4-Anker |
| `spike/src/wasm/loader.ts` | zwei neue `CoreExports`-Member (Append im Interface) |
| `spike/tests-runtime/backend-oom.test.ts` | **zwei `notImplemented`-Stubs** (Arbeitsregel 10 — sonst TS2739 bei grün aussehender Instantiations-Zeile) |
| `spike/tests-runtime/resident-lifecycle.test.ts` | **zwei `notImplemented`-Stubs** (v2, Baustein-0-Befund F1) — `:601` trägt ein **zweites** exhaustiv strukturell getipptes `CoreExports`-Literal mit eigenem lokalem `notImplemented` (`:595`). CLAUDE.mds Arbeitsregel 10 nennt `backend-oom.test.ts` fälschlich als „das EINZIGE" solche Literal; empirisch belegt sind es genau zwei (Dummy-Member-Probe → **beide** Dateien werfen TS2739, Exit 1, während `check:diag` weiterhin `Instantiations: 231240` druckt). **Die Regel-Korrektur in CLAUDE.md ist Teil dieser Scheibe** (T8), damit S5 nicht erneut darauf hereinfällt |
| `spike/src/wasm/resident.ts` | **insertion-only** in den Klassenkörper, null Edits an Bestandsmembern |
| `spike/src/runtime.ts` | **UNVERÄNDERT** (`argmaxRuntime` ist das Orakel und wird in dieser Scheibe nicht angefasst) |
| `spike/src/wasm/threaded.ts` | **UNVERÄNDERT** erwartet (`ThreadedCoreExports extends CoreExports`, der S0/D10-Direkt-Cast trägt neue Member ohne Edit) — falls ein Edit nötig wird, ist das ein Baustein-0-/Implementierungs-Befund und wird berichtet, nicht still gemacht |
| `spike/src/index.ts`, `backend*.ts`, `reduce.ts` | **UNVERÄNDERT** |

**KEIN neues Source-File unter `spike/src`.** Tests: kein neues Testfile (Anhänge an
bestehende Dateien, D7) → **kein Order-Noise** auf den `check:diag`-Pins.

### D7 — Testplan

**Rust (cargo, in `argmax.rs`):** contiguous gegen eine Hand-Referenz; **transponierte
View** — Ergebnis muss dem argmax des materialisierten Transponierten entsprechen, mit
**Nicht-Vakuitäts-Assertion** (der memory-order-Index unterscheidet sich hier
nachweislich, Muster kernels/sum.rs:205); Offset/Bounds; Rang 0; size-0 → `ShapeIncompatible`;
negative Achse ≡ positive; Rang zu groß; NaN-maximal; mehrere NaN → erster; ±0-Tie; Wert-Tie
→ erster Index. Zusätzlich zwei abi.rs-Prävalidierungs-Tests im bestehenden Stil.

**Orakel-Konstruktion für View- und keepdims-Fälle (v2, BINDEND — Baustein-0-Befund D1).**
Das Repo trägt heute **zwei methodisch unvereinbare Präzedenzfälle** für genau diese
Frage, und v1 hat keinen von beiden gewählt: S2/`mean` benutzt `keepDimsShape` als
Shape-Orakel (`assertMeanViewMatches`, resident.test.ts:457-472, dreifach verifiziert),
während `NDArray.argmax`s eigener keepdims-Test das ausdrücklich **ablehnt** und
stattdessen strukturelle Invarianten prüft, weil `keepDimsShape` als Orakel für eine
Funktion, die `keepDimsShape` selbst aufruft, **zirkulär** ist (argmax-topk.test.ts:21-24,
216-240). Beide haben recht — für verschiedene Fragen. Auflösung, dreiteilig, alle drei
Teile Pflicht:

1. **DATEN (die eigentliche M1-Frage) — Referenz ist `argmaxRuntime`, nie zirkulär.**
   Für einen View-Empfänger wird die Referenz aus `view.toArray()` gebildet, **gelesen
   BEVOR** die zu prüfende Methode läuft, zusammen mit `view.shape` an `argmaxRuntime`
   übergeben. Die Daten werden **nicht** aus dem Basis-Puffer neu hergeleitet (die
   S2-F1-Lektion, die dort ein eigenes Baustein-0-Addendum kostete).
2. **keepdims-SHAPE — strukturelle Invarianten, NICHT `keepDimsShape`** (das
   argmax-topk-Muster, weil hier tatsächlich zirkulär): Rang bleibt erhalten, `product`
   der Shape ist invariant gegenüber der non-keepdims-Form, die reduzierte Achse hat
   Größe 1, und „keepdims-Shape ohne die Achsen-Position" ist elementgleich zur
   non-keepdims-Shape.
3. **Cross-Surface-Shape-Pin** — zusätzlich und ohne jedes Orakel:
   `WNDArray.argmax(axis, kd).shape` ist deep-gleich zu
   `NDArray.argmax(axis, kd).shape` auf dem materialisierten Äquivalent. Das pinnt die
   Parität der beiden Flächen direkt und ist gegen einen gemeinsamen `keepDimsShape`-Bug
   zwar blind — dagegen wirken (2) und die bestehenden `keepDimsShape`-Tests.

**TS-Differential (`spike/tests-runtime/resident.test.ts`, Anhang) — M1-Beweis:**
- Randomisiertes Differential `WNDArray.argmax(...)` vs. `argmaxRuntime` über: contiguous,
  **und explizit die vier View-Klassen (Arbeitsregel 12): transponiert, geschnitten,
  offset-verschoben, zusammengesetzt** — je niladisch, positive Achse, negative Achse,
  `keepdims` true/false. Vergleich der Ergebnis-Bits (`Float64Array` → `to_bits`-Äquivalent
  via `DataView`), nicht nur `===`.
- **Spezialwert-Raster** (NaN inkl. nicht-kanonischer Payload, ±0, ±Inf, Subnormals) über
  dieselben Empfänger-Klassen.
- Rang 0, size-0-Achse (`[0,3]` Achse 1 → leeres Ergebnis, **kein** Throw), size-0-Throws
  (niladisch und Achsen-Dim 0).
- **Cross-Surface-Message-Parität** (S3-T4-Muster): die drei Stämme aus D4 als
  String-Gleichheit `NDArray` ⇄ `WNDArray`.
- **Lifecycle:** exakte Alloc/Free-Bilanz über den Zähl-Mock für beide Zweige; für die
  niladische Form der Nachweis **null residenter Netto-Allokation** über N Aufrufe auf
  einem persistenten Empfänger (S2-Muster `getResidentFreeCount()`-Delta).

**Threaded (`spike/tests-runtime/threaded.test.ts`, Anhang):** threaded-vs-stable-Parität
nach dem S0-`sqrt`-Muster (resident.test-Fälle gespiegelt: contiguous, transponierte View,
Rang 0, Spezialwerte).

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, Anhang — die Datei trägt bereits die
`WNDArray.sum`-keepdims-Pins, Zeile 73-76):** exakte Ergebnis-Shapes für literale Fälle
auf allen drei Formen; `argmax()` → `number`; Degradationskanten (Union-Achse, wide
`number`-Achse, `boolean`-keepdims-Union, dynamischer Rang) enden in no-claim;
`@ts-expect-error`-Pin für eine out-of-range-Literal-Achse **am Argument**;
`Equal<…>`-Hover-Pins. **Budget-Warnung (W5-D6-Befund):** `Equal<…>`-Message-Pins können
≈1.700 Instantiations pro Pin kosten — Pins konsolidieren statt vervielfachen.

**Nachträgliche Testpflichten (v3, aus der Verify-Runde — Baustein-B-Befunde V1/V2 plus
eine benannte Lücke):**

- **V2 — Kernel-Fehlerpfad (`status !== 0`) muss getestet sein, in BEIDEN Zweigen.**
  Baustein B hat den `freeBuf`-Aufruf vor dem Throw entfernt und **null** Testfehlschläge
  gemessen (0/334 argmax-Tests, 0/7 Lifecycle, 0/1 threaded); mit einem eigenen zählenden
  Core-Wrapper, der Status 1 erzwingt, hat B ein reales 16-Byte-Leck nachgewiesen, das die
  committete Suite nicht sieht. Das ist genau die Garantie, die der Doc-Kommentar des
  Moduls behauptet. Präzedenz existiert (S3: „`stack`: a row that fails mid-loop (kernel
  status != 0) frees BOTH…"). Pflicht: ein Mock/Wrapper, der `nt_argmax_all_strided` und
  `nt_argmax_axis_strided` je auf Status ≠ 0 zwingt, mit exakter Alloc/Free-Bilanz —
  **beide** Zweige direkt, nicht einer per Symmetrie-Argument.
- **V1 — die Leck-Nicht-Vakuitäts-Assertion darf nicht überclaimen.** Ihre zweite
  Assertion („500 Aufrufe dürfen den Speicher nicht wachsen lassen — ein 8-Byte-Leck würde
  eine Seitengrenze reißen") ist bei N=500 arithmetisch unfähig, das zu leisten
  (500 × 8 = 4.000 Byte gegen 64-KiB-Seiten) und gibt sich dennoch als unabhängige
  Bestätigung aus — dieselbe S3-Lektion im Test, der sie verhindern sollte. Auflösung: N so
  wählen, dass die Seitengrenze **beweisbar** überschritten würde (und das im Kommentar
  vorrechnen), ODER den Anspruch auf „ergänzende Plausibilitätsprüfung, der Ledger-Test ist
  der Beweis" zurücknehmen. Kein Zwischending mit unbelegtem Anspruch.
- **`memory.grow` mitten im Aufruf — Regressionstest ergänzen.** Baustein B hat den Fall
  selbst gebaut (`[4, 2000000]`, Wachstum nachweislich WÄHREND des `argmax(0)`-Aufrufs) und
  **0 von 2.000.000 Abweichungen** gemessen — der Angriff ist gescheitert, die Disziplin
  hält. Es existiert dafür aber kein committeter Test, obwohl die harte Repo-Regel („nie
  `memory.buffer`/Views cachen — mit shared memory schlägt das STILL fehl") genau diese
  Klasse adressiert und S3 für `stack` einen Präzedenz-Test hat. Der Fall wird als
  committeter Test übernommen, ggf. mit kleinerer Dimensionierung, solange das Wachstum
  nachweislich im Aufruf stattfindet.

**Pflicht-Mutant** (Implementierer, zusätzlich zu Baustein A/B): der Challenger-Vergleich
im Rust-Kernel wird invertiert (`>` → `>=`); der Test-Katalog muss den First-Index-Tie
fangen. Revert **nur** per Backup-Kopie + SHA-256-Beweis, nie `git checkout`
(Arbeitsregel 1).

### D8 — Pins, Budget, Gates

**Baseline** (HEAD `3fe47ae`, im **frischen `git worktree`** zu reproduzieren, Mess-Hausregel):
`check:diag` **226.690 @ 140** · `stress` **115.498 @ 82** · `browser` **2.142 @ 75** ·
Freeze `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176` ·
`bench:editor` `{w1 37.018, w2 38.851, w3 69.993, w4 37.173, w5 42.472, w6 43.666,
w7 36.222, w8 43.911}`.

**Vorregistriertes Absolut-Gate: Gesamtwachstum des Haupt-Pins ≤ +8.000 Instantiations**
(v2 angehoben von +6.000, Baustein-0-Befund; die Zahl wird **vor** jeder Messung
festgelegt, nicht nachträglich an ein Ergebnis angepasst).

Begründung, aus den Vorgängerscheiben abgeleitet statt geraten: S4 fügt **keine neue
Typ-Maschinerie** hinzu — es ist die sechste Call-Site der bereits bezahlten
`ReduceAxis`/`Guard`/`OkShape`-Maschinerie, strukturell die Lage von S2 (`mean`). S2s
**realisierte Gesamtkosten** waren aber nicht die zuerst gemessenen +1.500, sondern
**+5.689** — der View-Coverage-Nachtrag (Verify-B-Befund, 26 View-Fälle) kostete
allein +4.189. S4 fordert diese View-Fälle **von Anfang an** (D7/Arbeitsregel 12) und hat
zusätzlich eine Rust/ABI/`CoreExports`-Schicht, die S2 gar nicht hatte (S1s
vergleichbarer Umbau: +1.165). Realistische Erwartung damit ≈+4.000 bis +6.500; +8.000
ist die pre-S3-Standard-Marge und lässt ehrliche Luft, ohne eine Fehlkalkulation zu
kaschieren. Der S3-Ausreißer (+12.986) kam aus neuer `StackFold`-Maschinerie pro
Aufrufstelle plus Hover-Fix — beides existiert hier nicht.
**Bei Riss: STOPP, Befund an den Owner, kein „Optimieren ins Gate".**

**Mess-Protokoll:** kein neues File → **kein Order-Noise erwartet**; die Deltas werden
**gestuft attribuiert** (① Rust/ABI/CoreExports + oom-Stubs, ② `WNDArray.argmax`-Methode,
③ Test-Anhänge, ④ Typ-Pins). `stress`/`browser` sind ungated by design, ihre Deltas werden
deterministisch reproduziert und ausgewiesen. `bench:editor`-Pins deterministisch **doppelt**
messen und neu setzen. **Gate-Gesundheit (Arbeitsregel 6):** jeder Messlauf wird mit
Exit-Code und Fehlerausgabe berichtet, nie nur die gegrepte Kennzahl.

**Freeze-Beweis (S0/S1-Muster, dreiteilig):** (1) **Pre-Edit-Clean-Rebuild** reproduziert
den alten Pin `8255821b…` exakt; (2) additive-only-Dekomposition am Diff belegt (neue
Datei + drei Appends, null Zeilenverschiebung in Bestandsfunktionen); (3) der neue Hash
aus einem Clean-Rebuild wird der neue Pin, `check:freeze` und CLAUDE.md werden
aktualisiert. Threads-Artefakt: kein persistierter Pin (Bit-Identität beweist
`test:threaded`).

**Gate-Block der Scheibe:** `pnpm check` (Dreier-Verbund) · `check:diag`(+`:stress`/`:browser`)
· `test:core` · `test:resident` · `test:threaded` · `test:browser` · `test:package` ·
`test:example` · `cargo test --manifest-path crates/core/Cargo.toml` · `check:freeze` ·
`bench:editor` · `graph-a-lama query lint` · GFM-Gate auf allen neuen/geänderten `.md`
(Arbeitsregel 4: keine `~~…~~`-Strikethroughs).

### D9 — README (Hausregel 5, seit 2026-07-24 explizit)

Diese Scheibe ändert, **WO** `argmax` läuft. Die README macht heute an drei Stellen
Verfügbarkeits-Zusagen (`README.md:183/187/317-319`), die danach veraltet wären
(„only `argmax`/`topk` remain TypeScript-runtime only", „**TypeScript-runtime surface
only, no WASM kernel yet**"). Sie werden in **derselben** Scheibe korrigiert — nach S4
bleibt nur noch `topk` TypeScript-only. Prüf-Kommando:
`grep -n "TypeScript-runtime only\|no WASM kernel" README.md`; die verbleibenden
Behauptungen danach **empirisch** gegen `spike/src/index.ts` verifizieren (Wegwerf-Skript
im Scratchpad mit **absoluten** Importen), nicht nur lesen.

### D10 — Sprache

Code, Kommentare, Tests, Commit-Message, README: **Englisch**. Spec + Ergebnisse-Doc:
Deutsch. `≈` statt `~`.

## Akzeptanzkriterien

- **T1:** Alle D8-Gates grün mit berichteten Exit-Codes. Freeze-Beweis dreiteilig geführt,
  neuer Hash gepinnt.
- **T2:** Pin-Protokoll vollständig: Baseline im frischen Worktree reproduziert, Deltas
  gestuft attribuiert, neue Pins (root/stress/browser/bench:editor) dokumentiert,
  Absolut-Gate ≤ +8.000 eingehalten (oder STOPP + Owner-Befund).
- **T3 (M1):** Bit-Identität `WNDArray.argmax` ⇄ `argmaxRuntime` über contiguous **und
  alle vier View-Klassen** × alle drei Formen × Spezialwert-Raster, 0 Abweichungen, mit
  berichteter Fallzahl. Threaded-Parität separat belegt.
- **T4 (M3):** Compile-Ablehnung sitzt AM `axis`-Argument (empirisch geprüft); die drei
  Message-Stämme sind cross-surface wortgleich gepinnt; **LSP-Messung** (Arbeitsregel 13)
  mit einer Bestandsmethode als Kontrollpunkt belegt saubere Hover der neuen Signaturen.
- **T5 (M4):** Datei-Disziplin am Diff bewiesen: `argmax.rs` neu, `mod.rs`/`abi.rs` nur
  Appends ans Ende, `resident.ts` insertion-only, `runtime.ts`/`reduce.ts`/`index.ts`
  byte-unverändert.
- **T6:** Lifecycle-Beweise nicht-vakuös: exakte Alloc/Free-Bilanz (nicht
  Seitengranularität), niladische Form mit null residentem Netto-Delta über N Aufrufe.
- **T7:** Pflicht-Mutant gefangen, mit benannten fehlschlagenden Tests; Revert per
  Backup-Kopie mit SHA-256-Beweis.
- **T8:** README nach D9 korrigiert und die verbleibenden Behauptungen empirisch
  verifiziert; Doc-Platzierung nach Hausregel 5 (CLAUDE.md-Einzeiler + Pins,
  projekt-log-Append, roadmap, FOLLOWUPS, Ergebnisse-Doc mit Post-Verification-Addendum).
  **Zusätzlich (v2):** CLAUDE.mds Arbeitsregel 10 wird korrigiert — sie nennt
  `backend-oom.test.ts` als „das EINZIGE strukturell getippte `CoreExports`-Literal im
  Repo"; empirisch sind es **zwei** (auch `resident-lifecycle.test.ts:601`). Die
  Regel-Korrektur gehört in diese Scheibe, weil S5 sonst denselben Stolperstein erbt.

## Nicht-Ziele

- Kein `topk` (S5), kein `argmin`, keine Achsen-Variante von `topk`.
- Keine Performance-Aussage und keine Optimierung des Kernels (kein SIMD, kein
  Fast-Path) — die Scheibe stellt **Parität** her, nicht Geschwindigkeit. Der Kernel
  spiegelt die Referenzschleife.
- Keine Facaden-Änderung (D1), keine Edits an `runtime.ts`/`reduce.ts`, kein neuer
  `bench:editor`-Workload, kein Release/Publish.
- Keine Auflösung der offenen COVENANT-v6-Kandidaten (eigene Vertrags-Scheibe).

## Verify-Plan (Stufe 3)

**Baustein 0** (`brainroute:deep`, adversarial gegen DIESE Spec, VOR der ersten
Codezeile) — Auftrag aus docs/verify-runde-template.md, mit empirischen Proben im
eigenen Worktree: das Arbeitsregel-11-Verdikt angreifen (ist (b) doch die bessere Wahl?);
alle Code-Annahmen prüfen (abi.rs-Dateiende, `ThreadedCoreExports`-Vererbung,
`keepDimsShape`-Call-Site-Zahl, `ndarray.test-d.ts` als Pin-Ort, Regel-10-Wirksamkeit);
Overload-Auflösung inkl. Kollision mit Bestandsmembern empirisch typchecken;
Kernel-Design-Löcher (logischer vs. Speicher-Index, size-0-Achse mit nicht-leerer
Ausgabe, `unwrap_or(0.0)`-Konvention auf einem NaN-Datensatz); Freeze-Behauptung;
Testplan-Lücken.

Danach **A + B + C parallel** (A: Spec-Konformität + alle Gates + eigener Mutant;
B: adversarial, breite Mutanten, Messrandbedingungen, **Pflicht-LSP-Messung nach
Arbeitsregel 13**; C: `covenant:covenant-verify` gegen COVENANT.md v5 + Diff + Lint).
Ergebnisse-Doc mit Post-Verification-Addendum, KB-Capture, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-24)

Verifier: `brainroute:deep`, frischer Kontext, empirische Proben in einem eigenen
Scratch-Worktree auf HEAD `3fe47ae` (sauber entfernt; Haupt-Baum unberührt bestätigt).
Befunde und Auflösung:

1. **MAJOR (Konfidenz sehr hoch) — F1: es gibt ein ZWEITES strukturell getipptes
   `CoreExports`-Literal.** `spike/tests-runtime/resident-lifecycle.test.ts:601` (eigener
   lokaler `notImplemented`-Helfer bei `:595`) ist exhaustiv getippt und trägt bereits
   alle S0/S1-Member. Empirisch reproduziert: zwei Dummy-Member ins `CoreExports`-Interface
   → **beide** Dateien werfen TS2739, Exit 1, während `check:diag` weiterhin eine plausible
   `Instantiations: 231240`-Zeile druckt (Spezialfall von Arbeitsregel 6). Ein
   repo-weiter Grep belegt, dass es genau diese zwei Literale gibt. → **D6-Tabelle um die
   Datei ergänzt; die falsche „EINZIGE"-Formulierung in CLAUDE.mds Arbeitsregel 10 wird in
   dieser Scheibe korrigiert (T8).**
2. **MAJOR-ish (Konfidenz mittel-hoch) — D1: die Orakel-Methodik für View- + keepdims-Fälle
   war unspezifiziert, und das Repo trägt zwei unvereinbare Präzedenzfälle** (S2/`mean`
   benutzt `keepDimsShape` als Shape-Orakel; `NDArray.argmax`s eigener Test lehnt genau das
   als zirkulär ab). v1 wählte keinen — dieselbe Lückenform, die S2 ein eigenes
   Nachtrags-Addendum kostete. → **D7 um eine dreiteilige, bindende Auflösung ergänzt**
   (Daten gegen `argmaxRuntime` mit vorab gelesenem `view.toArray()`; keepdims-Shape über
   strukturelle Invarianten statt `keepDimsShape`; zusätzlich ein orakelfreier
   Cross-Surface-Shape-Pin).
3. **MINOR (Konfidenz mittel) — D2: die `unwrap_or(0.0)`-Konvention ist für `argmax` nicht
   gleich harmlos wie für `sum`.** Bei `sum` ist die Fallback-`0.0` das additive Neutrale,
   bei `argmax` ein echter Vergleichs-Kandidat. Der Pfad ist per `validate_strided_bounds`
   (shape.rs:165-180) unerreichbar — aber `argmax`s **Korrektheit** hängt damit an einer
   Zusicherung, an der bei `sum` nur die Speichersicherheit hing. → **Pflicht-Doc-Kommentar
   in D3 verankert.**
4. **NIT (Konfidenz hoch) — F2: `keepDimsShape` hat fünf Call-Sites, nicht vier**
   (`NDArray.argmax:788` war vergessen — es ruft den Helfer seit W1 selbst auf). → D4
   korrigiert, argmax wird die sechste.
5. **NIT — D3: „ersetzt wird ausschließlich die Akkumulation" war zu eng** (die beiden
   Leer-Zweige sind echte Neuzugänge gegenüber `sum`). → D3 präzisiert.
6. **Informationell (Konfidenz niedrig) — das +6.000-Gate war zu optimistisch begründet:**
   S2s realisierte Gesamtkosten waren +5.689 (nicht +1.500 — der View-Coverage-Nachtrag
   kostete +4.189), und S4 hat zusätzlich eine Rust/ABI/`CoreExports`-Schicht. → **Gate vor
   jeder Messung auf ≤ +8.000 angehoben, mit offengelegter Herleitung** (D8); die
   STOPP-Klausel bleibt unverändert.

**Positiv verifiziert (empirisch, nicht bloß gelesen):** `nt_scalar_div_strided` ist
tatsächlich das letzte Symbol in abi.rs (Funktion endet :1594, kein nachfolgendes
`#[cfg(test)]`) — Append-Punkt korrekt · die 7-/9-Parameter-Signaturen der `sum`-Zwillinge ·
`argmaxRuntime`/`beatsMax` wie zitiert · `dot`/`norm` als korrekte Präzedenz für die
niladische „lesen-und-im-selben-Aufruf-freigeben"-Form · `ThreadedCoreExports extends
CoreExports` + der S0/D10-Direkt-Cast tragen neue Member **ohne** Edit an threaded.ts ·
`ndarray.test-d.ts` ≈73-80 ist der richtige Pin-Anker · alle benötigten Rust-Helfer sind
bereits `pub` in shape.rs (keine Sichtbarkeits-Edits) · `kernels/argmax.rs` existiert noch
nicht · die README-Zeilen aus D9 existieren wörtlich · **die vollständige
D2-Overload-Form wurde in einer Scratch-Kopie end-to-end typgeprüft**: niladisch →
`number`, literale ±Achse → korrekte Shape, keepdims true/false → korrekte Shape,
`axis=undefined` in 1-/2-Arg-Form → Voll-Reduktions-Shapes, dynamische Achse/keepdims →
korrekte no-claim-/Union-Degradation, out-of-range-Literal-Achse wirft **AM Argument** in
beiden Formen, null Kollision mit `sum`/`mean`/`norm`, `tsc --noEmit` Exit 0 ·
**Baselines exakt reproduziert:** `check:diag` 226.690 @ 140 Files (Exit 0) und `cargo test`
184+1 (Exit 0) · das Arbeitsregel-11-**KERNEL-Verdikt hält**: ein Grep über alle
Kernel-Dateien belegt, dass keine Vergleichs-/Max-/Index-Primitive existiert (Alternative
(a) ausgeschlossen), und `dot`/`norm_sq` setzen als ebenfalls O(N)-Reduktionen den
Präzedenzfall „Reduktionen bekommen einen Kernel" (Alternative (b) nicht überzeugender).

**Bewusst NICHT verifiziert (offengelegt):** die Clean-Rebuild-Reproduktion des
Freeze-Hashes `8255821b…` (braucht die wasm32-Toolchain) und die Bit-Identität des
Threads-Artefakts (braucht die gepinnte nightly) — beides bleibt planmäßig Baustein A/B
nach der Implementierung.
