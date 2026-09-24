# dtype-Design — bindende Spec (Stufe 3b)

**Version:** v1 (2026-09-24) · **Status:** Owner-Richtung abgenommen 2026-09-24 (E1–E7 „wie
empfohlen") → Baustein 0 steht aus
**Stufe:** 3b — neue Typ-Maschinerie-Klasse (Promotion), Mess-/Forschungsanteil, zwei
Covenant-Entwürfe (M1-Erweiterung, M3 v8).
**Berührte Covenant-Invarianten:** **M1** (Bit-Identität — muss für neue dtypes erweitert werden),
**M2** (never wrong — Promotion und Skalar-Regel), **M3** (Klassen-Hover — ändert sich
zwangsläufig), **M5** (Browser-Pfad — neue Typed-Array-Arten), **Z1** (keine Abhängigkeit).
**Grundlage:** Bestandsaufnahme vom 2026-09-24 (Agent-Bericht, zusammengefasst unten, Fakten mit
Datei:Zeile). Roadmap: „dtype-Design" vor Phase 2 (Owner, 2026-09-23).

## Ziel und Zuschnitt

Diese Scheibe **entscheidet** das dtype-Design verbindlich für alle Folge-Scheiben und **misst**
es an einem echten Prototyp. Sie liefert KEIN veröffentlichbares dtype-Feature.

1. **Auf `main`:** diese Spec (die Entscheidungen D1–D10 binden die Folge-Scheiben), ein
   Ergebnis-Doc mit den Messungen, der Entwurf der Covenant-Texte (M1-Erweiterung, M3 v8) als
   Vorlage — **nicht** in COVENANT.md eingetragen (s. D10).
2. **Auf Branch `proto/dtype`** (wird nicht nach `main` gemergt): ein Prototyp auf `NDArray`, der
   die teuerste und die heikelste Maschinerie real baut, damit Budget und Hover gemessen statt
   geschätzt werden. Umfang in D11.

**Warum Branch statt `main`:** ein dtype, den nur drei Ops unterstützen, wäre auf `main` eine
halbfertige öffentliche API (Arbeitsregel 19: nichts bewerben, was nicht fertig ist) und würde
den Hover im ganzen Repo ändern, bevor M3 v8 gilt.

## Fakten aus der Bestandsaufnahme (2026-09-24, am Code belegt)

- **Oberfläche:** 67 öffentliche Signaturen (NDArray 27, WNDArray 26, WasmBackend 6,
  ThreadedBackend 7, `threadedMatmul` 1). 15 Referenzfunktionen in `runtime.ts`, alle `Float64Array`.
  28 `nt_*`-Einstiege, davon 26 f64-Datenkernel.
- **Orthogonalität:** rund 30 exportierte Shape-Typen (`Broadcast`, `MatMul`, `ReduceAxis`,
  `SliceShape`, `Guard`, `OkShape`, `NestedArray`, …) referenzieren keinen dtype — die
  Shape-Maschinerie bleibt unverändert. `crates/core/src/shape.rs` ist dtype-frei; `read_slice<T>`
  (abi.rs:127/141) ist bereits generisch.
- **Hover (echter LSP, zwei Varianten — String-Literal- und Interface-Default):** ein zweiter
  Typparameter mit Default wird IMMER angezeigt: `NDArray<[2, 3]>` → `NDArray<[2, 3], "f64">`,
  auch bei Inferenz (`fromArray`, `matmul`, `zeros`). → M3 kann nicht unverändert bleiben (E1).
- **Kosten des bloßen Parameters (Sondage, Arbeitsregel 8 — keine Messung):** +539 Instantiations,
  0 Folgefehler in 140 Dateien. Die Promotions-Maschinerie ist darin NICHT enthalten.
- **Vorhandene Lücken im heutigen Code:** `topk` liefert `{ values: NDArray<…>; indices:
  NDArray<…> }` mit demselben Typ (ndarray.ts:857-866) — Indizes als f64. `toArray(): Float64Array`
  (resident.ts:1635) muss zu einem berechneten Rückgabetyp werden.
- **NumPy-Referenz (Primärquellen: NEP 50, `numpy.sum`, `numpy.mean`, `numpy.floor_divide`,
  `basics.types`):** Promotion über Arten `bool < integral < inexact`; Ganzzahl-Überlauf ist
  modular ohne Fehler; `mean` von Ganzzahlen → float64; `sum` von Ganzzahlen → int64 (nicht
  abbildbar, int64 fehlt im Zielsatz); `/` ergibt immer Gleitkomma; `bool - bool` ist ein Fehler.
  Skalare sind „weak" und werden WERTABHÄNGIG zur Laufzeit behandelt — mit M2 unvereinbar.

## Owner-Entscheidungen (2026-09-24)

E1 Hover zeigt den dtype, M3 per v8 präzisiert · E2 NumPy-Namen · E3 `sum`/`mean` über int32/bool
→ float64 · E4 Skalare auf int32 bleiben int32, nicht-ganzzahlige Literale Compile-Fehler,
nicht-ganzzahlige Laufzeitwerte Throw · E5 bool nur aus Vergleichen, nur für `where`/`any`/`all`/
Zählen, Arithmetik braucht `astype` · E6 Index-dtype int32 · E7 Zuschnitt: Spec + Prototyp, Umsetzung
danach in Scheiben.

## Design (D1–D11)

**D1 — Namen und Typparameter.** `export type DType = "float64" | "float32" | "int32" | "bool"`.
`NDArray<S extends Shape, D extends DType = "float64">`, ebenso `WNDArray` (Folge-Scheibe).
`NDArrayView<out S>` bleibt dtype-frei (liest nur Shape/Strides/`unknown`). `AnyNDArray =
NDArray<any, any>`. Der Default hält bestehenden Code gültig: `NDArray<[2, 3]>` ist weiterhin
schreibbar und bedeutet float64.

**D2 — Speicher.** `DataOf<D>`: float64 → `Float64Array`, float32 → `Float32Array`, int32 →
`Int32Array`, bool → `Uint8Array` (0/1). `NDArray.data: DataOf<D>`; `toArray()` gibt `DataOf<D>`
zurück. Alle vier sind ECMAScript-Standard (M5, Z1 unberührt).

**D3 — Erzeugung und Umwandlung.**
- `fromArray(shape, data, opts?: { dtype?: D })`. Ohne `dtype`: aus dem Typed Array abgeleitet
  (`Float32Array` → float32, `Int32Array` → int32, `Uint8Array` → **nicht** bool, sondern ein
  Compile-Fehler ohne explizites `dtype`, weil `Uint8Array` mehrdeutig ist), `number[]` → float64.
  Mit `dtype` bei `number[]`: Werte werden geprüft und konvertiert (int32: ganzzahlig und im
  Bereich, sonst Throw; bool: nur 0/1, sonst Throw).
- `zeros(shape, dtype?)`, `ones(shape, dtype?)` — Default float64.
- `astype<T extends DType>(dtype: T): NDArray<S, T>`. Regeln: → float32 per `Math.fround`;
  float → int32 Richtung null gekürzt, NaN/±Inf/außerhalb des Bereichs → Throw (strenger als
  NumPy, das dort undefiniert ist); → bool `x !== 0` mit NaN → true (wie NumPy); bool → numerisch
  0/1.

**D4 — Promotion (`Promote<A, B>`, Typebene und identische Laufzeit-Tabelle aus EINER Quelle).**

| ⊕ | float64 | float32 | int32 | bool |
|---|---|---|---|---|
| **float64** | float64 | float64 | float64 | — |
| **float32** | float64 | float32 | float64 | — |
| **int32** | float64 | float64 | int32 | — |
| **bool** | — | — | — | — |

„—" = Compile-Fehler am Argument mit Botschaft (E5: `astype` verlangen) und identischer
Laufzeit-Botschaft (M3-Message-Parität). int32 ⊕ float32 → float64 folgt NEP 50.

**D5 — Ergebnis-dtype je Op.**

| Op | Regel |
|---|---|
| `add`/`sub`/`mul` (Array ⊕ Array) | `Promote<A, B>`; int32 ⊕ int32 wickelt modular (Zweierkomplement) |
| `div` | Gleitkomma immer: float32/float32 → float32, sonst float64 (int32/int32 → float64) |
| Skalar-Überladungen | float32/float64 behalten D (Skalar per `fround` bei float32); int32: `add`/`sub`/`mul` behalten int32 (E4), `div` → float64 |
| `matmul`/`dot` | wie Promotion, aber int32 → **float64** (Reduktion, konsequent zu E3) |
| `sum`/`mean` | float32 → float32, float64 → float64, int32/bool → float64 (E3) |
| `norm`/`cosineSimilarity` | Rückgabe `number` wie heute (berechnet in der Präzision des Eingangs, float32 per `fround`) |
| `sqrt` | float32 → float32, float64 → float64, int32 → float64, bool → Compile-Fehler |
| `argmax` | Index als `number` wie heute; Achsen-Variante → `NDArray<…, "int32">` (E6) |
| `topk` | `values: NDArray<…, D>`, `indices: NDArray<…, "int32">` (E6) |
| Vergleiche `gt`/`ge`/`lt`/`le`/`eq`/`ne` (neu) | numerisch ⊕ numerisch (mit Promotion zum Vergleichen) → bool; bool-Operand → Compile-Fehler |
| `where(mask, a, b)` (neu), `any`/`all` (neu) | nur bool-Maske; `where` → `Promote` von a, b |
| `stack` | alle Zeilen gleicher dtype, sonst Compile-Fehler (Promotion über Zeilen bewusst NICHT, Budget) |
| `transpose`/`slice`/`reshape`/`flatten`/`item` | D unverändert; `item` liefert `boolean` bei bool, sonst `number` |
| `toNestedArray`/`toJSON` | `NestedArray<S, D>` bekommt D: bool → `boolean`-Blätter; `toJSON` `data: number[]` bzw. `boolean[]` |

**D6 — Skalar-Regel für int32 (E4) im Detail.** Literal mit Punkt (`2.5`) → Compile-Fehler am
Argument; die Punkt-Form als bewiesen nicht-ganzzahlig ist seit Spike 06 vorhanden
(`literal-arithmetic.ts`). Exponent-Formen und `number` → kein statischer Befund, Laufzeitprüfung
`Number.isInteger` + Bereich, sonst Throw mit derselben Botschaft. Kein wertabhängiger
Ergebnis-dtype — statischer Typ und Laufzeit stimmen immer überein (M2).

**D7 — bool (E5).** Entsteht nur aus Vergleichen, `astype("bool")` und `fromArray` mit
`dtype: "bool"`. Wird verbraucht von `where`, `any`, `all`, `sum`/`mean` (→ float64, zählt bzw.
Anteil) sowie den dtype-neutralen Ops aus D5. Alles andere → Compile-Fehler „use astype".

**D8 — M1-Erweiterung (Entwurf, gilt ab der ersten Kernel-Scheibe je dtype).**
- **float32:** Die TS-Referenz rechnet in f64 und rundet nach JEDER elementaren Operation mit
  `Math.fround`. Für `+ − × ÷ √` ergibt das beweisbar das korrekt gerundete f32-Ergebnis (doppelte
  Rundung ist harmlos, weil 53 ≥ 2·24 + 2; Figueroa 1995 — Baustein 0 prüft die Quelle). Reduktionen
  akkumulieren aufsteigend in f32 mit `fround` pro Schritt. Kernel dürfen dieselbe Reihenfolge nicht
  verlassen (bestehendes Bit-Identity-Law gilt unverändert).
- **int32:** Zweierkomplement-Wrap überall: TS `(a + b) | 0`, `(a - b) | 0`, `Math.imul(a, b)`; Rust
  ausschließlich `wrapping_add`/`wrapping_sub`/`wrapping_mul` (nie `+`/`*` — Debug-Builds paniken
  bei Überlauf). Kein NaN-Thema.
- **bool:** exakte 0/1-Bytes; Vergleiche über promotete Werte, NaN-Vergleiche nach IEEE (`NaN <
  x` false, `NaN != x` true).

**D9 — Backends und Staffelung.** Die TS-Referenz trägt jeden dtype zuerst; WASM-Kernel folgen pro
dtype als eigene Scheiben (float32 zuerst). Bis dahin wirft `backend("wasm").fromArray(…, {
dtype })` für nicht-float64 mit klarer Botschaft **und** der Typ lässt es nicht zu (M2). Paritätslücke
wird in FOLLOWUPS getrackt (M1 v5 erlaubt kernel-lose Referenz-Ops).

**D10 — M3 v8 (Entwurf).** „Klassen-Hover zeigen die aufgelöste Shape als Tupel UND den dtype als
String-Literal (`NDArray<[2, 3], \"float64\">`); ein Default-Typargument darf dabei sichtbar sein."
Wird mit der ersten Umsetzungs-Scheibe eingetragen (Owner-Bestätigung des Wortlauts dann), nicht
mit dieser Scheibe.

**D11 — Prototyp auf `proto/dtype` (nur `NDArray`, nur TS-Referenz).** Real gebaut, mit Tests:
`DType`, `DataOf`, `Promote` (Typ + Laufzeittabelle aus einer Quelle), `fromArray`/`zeros` mit
`dtype`, `astype`, `add` (Array ⊕ Array und Skalar inkl. D6), `sum` (mit Achse, E3), `gt` (erster
Vergleich), `toArray`/`item`/`toNestedArray` dtype-korrekt. Alle übrigen Ops: für D ≠ float64 per
`this`-Typ gesperrt (Compile-Fehler) — so bleibt der Prototyp in sich korrekt (M2).

## Messplan (Prototyp) — vorregistriert

- check:diag Root gegen 227,405 @ 140 (Baseline im frischen Worktree reproduzieren). Dateiset
  möglichst unverändert; kommt eine Datei hinzu, per empty-then-fill dekomponieren.
  **Absolut-Gate für den Prototyp: ≤ +8,000.**
- Aufschlüsselung in Stufen: (1) bloßer Parameter, (2) `DataOf`/Erzeugung/`astype`, (3) `Promote` +
  `add`, (4) Skalar-Regel, (5) `sum`, (6) `gt`, (7) `this`-Sperren der übrigen Ops, (8) Tests/Pins.
- **Hochrechnung** auf die volle Oberfläche aus den gemessenen Stufenkosten, mit offengelegter
  Formel. KEINE vorregistrierte Verdikt-Regel — die Zahl informiert eine Owner-Entscheidung
  (Arbeitsregel 7 greift damit nicht; eine Regel käme nur mit Fuzz-Skript).
- stress/browser, `bench:editor` (Instantiations und Latenz; nicht neu pinnen, nur berichten).
- LSP-Hover (Pflicht, Regel 13): Klasse, `fromArray` mit/ohne dtype, `add` gemischt
  (`float32 ⊕ int32` → `NDArray<…, "float64">`), `sum` über int32, `gt`, Fehlerposition und
  -botschaft bei `bool ⊕ …` und bei `2.5` auf int32.
- Bit-Identität der Prototyp-Laufzeit: float32-`add`/`sum` gegen eine unabhängige Referenz
  (`Math.fround`-Kette über `Float32Array`-Zuweisung), int32-Wrap an den Rändern ±2^31.

## Nicht-Ziele

Veröffentlichung · `WNDArray`/Threaded/Rust in dieser Scheibe · float16/int64/uint8/complex ·
Broadcasting-Änderungen · Promotion über `stack`-Zeilen · Änderungen an COVENANT.md (nur Entwürfe).

## Folge-Scheiben (Vorschlag, je eigene Spec)

dt1 Kern: Parameter, Speicher, Erzeugung, `astype`, Auslesen, M3 v8 · dt2 Elementweise + Promotion
+ Skalar-Regel · dt3 Reduktionen (`sum`/`mean`/`matmul`/`dot`/`norm`) · dt4 Vergleiche, `where`,
`any`/`all` · dt5 übrige Ops (`sqrt`, `argmax`, `topk` mit int32-Indizes, `stack`) · dt6+ `WNDArray` und
Kernel pro dtype (float32 zuerst, dann int32, bool), jeweils mit M1-Erweiterung. Release, sobald
dt1–dt5 auf `NDArray` stehen.
