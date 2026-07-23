# WASM-Parität S3 — `item` + `stack` auf `WNDArray`/threaded (bindende Spec)

Status: **bindend** (Owner-Richtungsabnahme 2026-07-23: WASM-Backend-Parität der W1–W5-Ops
nachziehen; S3 ist die vierte Scheibe der Serie, nach S0/sqrt, S1/Skalar-Overloads und S2/mean).
Version: 2 · Datum: 2026-07-23 · Eskalationsleiter: **Stufe 3** (substanzielle Scheibe, zwei neue
öffentliche API-Member + M1-relevant — voller Verify-Katalog A+B+C). Covenant: v5. **v2:**
Baustein-0-Befunde eingearbeitet (BLOCKER: `ThreadedBackend` hat kein `core`-Feld, der Core lebt auf
`this.pool.core`; MAJOR: die Facaden-Signaturen brauchen eine Typ-Verdrahtung, die D1 nicht nannte —
gelöst über zwei exportierte Signatur-Aliase in `resident.ts` statt neuer Import-Kanten). Kern-Design
D3–D5 bestätigt und empirisch belegt; Details im Addendum am Ende.
Roadmap: vierte Scheibe der WASM-Parität-Serie (S3 von S0–S5: sqrt → Skalar-Overloads → mean →
**item/stack** → argmax → topk).

## Ziel & Warum — und was diese Scheibe NICHT ist

`NDArray.item(...indices)` (W5) und `NDArray.stack(rows)` (W4) existieren heute NUR auf der naiven
JS-Klasse. Wer einen einzelnen Skalar aus einem WASM-residenten Array lesen will, muss heute das
GANZE Array nach JS kopieren (`toArray()`), und wer N residente Zeilen zu einer `[N, D]`-Matrix
zusammensetzen will, muss jede Zeile herauskopieren, in JS stacken und wieder hineinkopieren. Diese
Scheibe schließt beide Lücken.

**Die tragende Beobachtung — beide Ops brauchen KEINEN neuen Kernel (Arbeitsregel 11):**

- **`item`** ist ein reiner strided READ eines einzelnen Elements: Arity-/Integer-/Bounds-Prüfung,
  Offset-Berechnung aus `(strides, offset)` des Handles, dann ein einziger `f64`-Lesezugriff. Es gibt
  keine Arithmetik, die ein Kernel schneller machen könnte, und der Lesepfad existiert bereits —
  `toArray()`s contiguous-Fast-Path liest die WASM-Memory heute schon direkt über eine frisch
  abgeleitete `Float64Array`-View (resident.ts:1466-1469), `dot()`/`norm()` lesen ihr Skalar-Ergebnis
  genauso (resident.ts:1066, :1102). `WNDArray.item` ist damit **kernel-los per Design** — exakt die
  Einordnung, die W5s Spec (M1 v5) für die NDArray-Seite bereits getroffen hat.
- **`stack`** ist reine DATENBEWEGUNG, und der dafür zuständige Kernel existiert bereits und ist
  eingefroren: `nt_materialize` (Kern 03) sammelt eine beliebig strided View in einen contiguous
  row-major Ausgabebereich. `WNDArray.stack` ist genau N solcher Aufrufe, jeder in den `i`-ten
  `D`-Elemente-Slot EINES frisch allozierten Ausgabepuffers. Kein neues Rust, kein neuer Export.

Damit gilt derselbe Gewinn wie bei S2/mean: **kein neuer Kernel ⇒ Freeze-Hash UNVERÄNDERT** (der
Beweis kippt in eine billige, strengere NEGATIVE Assertion: der Hash DARF sich nicht bewegen), kein
Rust, kein ABI-Eintrag, kein `CoreExports`-Member ⇒ **kein `backend-oom`-Stub nötig** (Arbeitsregel
10 geprüft: greift NICHT, weil `CoreExports` unverändert bleibt — `nt_materialize` ist seit Kern 03
Member und im hand-getippten Mock bereits gestubbt; das ist in T2 als Pflicht-Verifikation verankert,
nicht als Annahme).

**Ehrlichkeits-Rahmen (Owner-Vorgabe, wie bei W2/W3/S0/S1/S2):** Diese Scheibe hat **keinen
gemessenen Nutzerbedarf** — niemand ist auf eine WASM-`item`- oder `stack`-Wand gestoßen. Es ist
Vollständigkeits-/Symmetrie-Arbeit. Das Ergebnisse-Doc darf keinen Nutzerbedarf suggerieren, den es
nicht gibt. (Die W4/W5-Wunschlisten-Evidenz stammt aus dem NDArray-Dogfooding, nicht aus residenter
Nutzung.)

Diese Scheibe ist **kein** neuer Kernel, **kein** Threaded-Pool-Kernel (der Pool routet weiterhin nur
matmul; item/stack laufen auf dem residenten Core in beiden Artefakten), **keine** Änderung an
`NDArray.item`/`NDArray.stack`/`itemRuntime`/`stackRuntime` (bereits fertig, bleiben die
Referenz-Orakel), **keine** Verallgemeinerung von `stack` (kein `axis`-Parameter, kein
`concat`/`vstack`/`hstack`, kein höherer Rang — dieselben Nicht-Ziele wie W4), **kein** `item`-Setter
und **keine** andere Op (argmax/topk folgen als S4/S5).

## Die M1-Einordnung: zwei verschiedene Fälle, beide ohne neuen Kern

`item` und `stack` sind M1-technisch NICHT derselbe Fall — das muss die Scheibe sauber trennen:

1. **`WNDArray.stack` — komponierte Op über einen bereits bewiesenen Kern.** `nt_materialize` ist
   seit Kern 03 in Gebrauch (`toArray()` für Views, `contiguous()`) und über die gesamte
   Differential-Suite bit-identisch belegt. Datenbewegung ist bit-erhaltend per WASM-Semantik
   (`f64.load`/`f64.store` sind keine arithmetischen Operationen; Rusts `copy_from_slice` ist ein
   `memcpy`) — dieselbe Begründung, die `transposeRuntime`s NaN-Payload-Fixture (special-values.test.ts,
   Kern 10) bereits empirisch trägt. Bit-Identität zu `stackRuntime` ist damit ein Korollar; der
   Differentialtest (D6) bestätigt sie direkt. Dies ist derselbe „dritte Fall" wie S2/mean
   (WASM-berechnet, komponiert, ohne eigenen Kern) — der bereits offene v6-Kandidat.
2. **`WNDArray.item` — resident, aber vollständig in TS berechnet.** Es wird KEIN WASM-Code
   ausgeführt: die Offset-Arithmetik läuft in JS, der Lesezugriff ist eine `Float64Array`-Indizierung
   auf `core.memory.buffer`. Das ist ein **VIERTER** Fall, den COVENANT v5s M1-Wortlaut nicht benennt
   (er kennt „neuer Kern" — muss bit-identisch sein —, „kernel-lose Referenz-Op" — kein WASM, M1
   bindet nicht — und seit S2 faktisch „komponiert aus bewiesenen Kernen"). Praktisch bleibt die
   Pflicht unverändert und wird hier voll erfüllt: `WNDArray.item` MUSS bit-identisch zu
   `itemRuntime` sein, differentiell bewiesen. Ob M1 dafür eine v6-Formulierung braucht, ist eine
   Owner-Entscheidung nach Baustein C — **nicht still auflösen** (siehe D9).

**Präzisierung zum NaN-Payload bei `item` (ehrliche Grenze, vorab benannt):** beide Flächen geben
einen JS-`number` zurück, gelesen aus einer `Float64Array`. Ob die JS-Engine dabei einen
nicht-kanonischen NaN-Payload erhält, ist eine Engine-Eigenschaft — sie ist auf BEIDEN Flächen
dieselbe Operation, die Differential-Aussage („`WNDArray.item` == `NDArray.item`, bit-identisch")
hält deshalb unabhängig davon. Der Test misst und dokumentiert das tatsächliche Verhalten (D6/T3b),
statt Payload-Erhalt zu behaupten.

## Berührte Covenant-Invarianten

- **M1 (Anker `crates/core/src/`, `spike/src/runtime.ts`): KEIN NEUER KERN. `crates/core/src/`
  UNBERÜHRT.** `spike/src/runtime.ts` wird BERÜHRT, aber ausschließlich **append-only** (zwei neue
  exportierte Validierungs-Helfer, D4) — die bestehenden Referenzfunktionen `itemRuntime`/
  `stackRuntime` bleiben **byte-unberührt**, weil sie in dieser Scheibe das ORAKEL sind (siehe D4 für
  die vollständige Begründung, warum hier bewusst dupliziert statt refaktoriert wird).
- **M2 (Anker `dim.ts`, `literal-arithmetic.ts`, `sym:Guard`, `sym:OkShape`): keine neue
  Degradationskante.** `item` ist die ZWEITE Call-Site des unveränderten `ItemGuard<S, Idx>`
  (vector.ts:649 — bereits shape-generisch, W5/FOLLOWUPS hat das explizit so vorhergesagt), `stack`
  die zweite Call-Site des unveränderten `StackCheck`/`StackShape`. Neu sind ausschließlich die
  beiden `WNDArray`-seitigen Unwrap-Aliase (D5), strukturelle Spiegel von `ndarray.ts`s
  `UnwrapRow`/`RowShapesOf` — inklusive der beiden dort teuer gelernten Fallen (F2:
  Invarianz-Kollaps bei nicht-homomorpher Extraktion; F8: Array-Element-Union-Kollaps ohne frischen
  nackten Typparameter). `vector.ts`/`dim.ts`/`slice.ts`/`reduce.ts` bleiben UNVERÄNDERT.
- **M3 (Anker `sym:Guard`, `sym:ShowShape`): Message-Stämme wortgleich, per Test erzwungen.** Alle
  sechs Runtime-Stämme (drei für `item`, drei für `stack`) müssen zwischen `NDArray`- und
  `WNDArray`-Fläche WORTGLEICH sein. Weil die Validierungslogik dupliziert wird (D4), ist das keine
  Struktur-Garantie, sondern eine **Test-Pflicht** (T4, Cross-Surface-Message-Parität über alle sechs
  Klassen, mit Mutant).
- **M4 (Anker `abi.rs`/`matmul_blocked.rs`/`shape.rs`): UNBERÜHRT, Freeze-Hash UNVERÄNDERT.** Kein
  Rust wird angefasst. Der `check:freeze`-Pin bleibt
  `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176`. **Verifikations-Pflicht
  (negativ):** ein Clean-Rebuild MUSS den unveränderten Hash reproduzieren; täte er es nicht, hätte
  die Scheibe versehentlich Rust berührt (Bug, STOP). `cargo test` bleibt bei 184+1.
- **M5 (Anker `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`): berührt, aber norm-konform.**
  `ndarray.ts` wird NICHT editiert. `threaded.ts` bekommt EINE Facade-Methode (D3) — reine
  Delegation an `WNDArray.stack`, kein neuer Import, kein `node:*`-Zugriff, keine Änderung an der
  Lade-/Env-Logik. Threads bleiben Node-only-Opt-in; threaded-Parität ist automatisch (dasselbe
  Crate, derselbe `nt_materialize`).
- **Z1 (Anker `package.json`): unberührt.** Keine neue Abhängigkeit.
- **Z2 (Anker `package.json`): unberührt.** Alle neuen Tests hängen an BESTEHENDE, registrierte
  Korpus-Dateien an (`resident.test.ts`, `special-values.test.ts`, `resident-lifecycle.test.ts`,
  `backend-api.test.ts` in test:resident; `threaded.test.ts`, `backend-api-threaded.test.ts` in
  test:threaded; `ndarray.test-d.ts`, `backend-api.test-d.ts` im Root-Typkorpus) — **keine neue
  Datei, keine Explizitlisten-Änderung, `test-scripts-guard` bleibt grün, KEIN Order-Noise.**

---

## Design-Entscheidungen

### D1 — Scope (exakte Liste der Änderungen)

Neue Member/Deklarationen, ALLE als Insertion (kein Edit an einem Bestandsmember):

| Datei | Änderung | Art |
| --- | --- | --- |
| `spike/src/runtime.ts` | `itemOffsetStrided(...)`, `stackValidateShapes(...)` | Append (ans Dateiende) |
| `spike/src/wasm/resident.ts` | Import-Erweiterung (`ItemGuard`/`StackCheck`/`StackShape` aus `../vector.ts` — dieselbe Datei, aus der `DotCheck` schon kommt); file-private Typ-Aliase `UnwrapWRow`/`WRowShapesOf`; **exportierte** Signatur-Aliase `StackRowsGuard`/`StackResultOf` (v2, D6); `WNDArray.item(...)`; `static WNDArray.stack(core, rows)` | Insertion |
| `spike/src/wasm/backend-api.ts` | Typ-Import `StackRowsGuard`/`StackResultOf` aus `./resident.ts` (Datei ist dort bereits Wert-Import-Quelle); `WasmBackend.stack(rows)` | Insertion |
| `spike/src/wasm/threaded.ts` | Typ-Import `StackRowsGuard`/`StackResultOf` aus `./resident.ts` (dito); `ThreadedBackend.stack(rows)` — Core über **`this.pool.core`**, NICHT `this.core` (v2/Baustein-0-BLOCKER) | Insertion |

**Keine neue Import-KANTE zwischen Dateien** (v2): beide Facaden importieren ohnehin schon aus
`resident.ts`; `Guard`/`OkShape`/`StackCheck`/`StackShape` bleiben dort, wo sie heute sind. Das ist der
Grund für die zwei exportierten Signatur-Aliase (D6) — ohne sie müssten `backend-api.ts` und
`threaded.ts` je vier neue Typ-Importe aus `ndarray.ts`/`vector.ts` ziehen.

**KEIN** Rust, **KEIN** ABI-Eintrag, **KEIN** `CoreExports`-Member, **KEIN** `backend-oom`-Stub,
**KEIN** Freeze-Re-Pin, **KEIN** Byte an `ndarray.ts`/`vector.ts`/`dim.ts`/`reduce.ts`/`slice.ts`/
`loader.ts`/`itemRuntime`/`stackRuntime`/den Rust-Kernels.

### D2 — Warum `stack` auch auf die beiden Backend-Facaden muss (Erreichbarkeit)

`WNDArray` wird von `spike/src/index.ts` **nicht exportiert** (weder als Wert noch als Typ) —
Konsumenten des Pakets bekommen `WNDArray`-Instanzen ausschließlich über `WasmBackend`/
`ThreadedBackend` (`fromArray`/`zeros`/`ones`). Für die drei bisherigen Kampagnen-Scheiben war das
irrelevant: `sqrt`, die Skalar-Overloads und `mean` sind INSTANZ-Methoden. `stack` ist die **erste
statische** Op der Kampagne — ein `WNDArray.stack` allein wäre für Paket-Konsumenten
**unerreichbar**, die Scheibe würde ihr eigenes Ziel („residente Zeilen stacken können") nicht
liefern.

Deshalb: die Implementierung lebt als `static WNDArray.stack(core, rows)` (die Fläche, die die
Kampagne nachzieht), und beide Facaden bekommen eine Ein-Zeilen-Delegation — jede mit dem Core-Zugriff,
den sie tatsächlich hat (**Baustein-0-BLOCKER, v2**: die beiden Klassen sind hier NICHT strukturgleich):

```ts
// backend-api.ts — WasmBackend hat ein eigenes `private readonly core`
stack<const Rows extends readonly WNDArray<any>[]>(rows: StackRowsGuard<Rows>): StackResultOf<Rows> {
  this.assertLive("stack");
  return WNDArray.stack(this.core, rows);
}

// threaded.ts — ThreadedBackend hat KEIN `core`-Feld; der Core lebt auf dem Pool
// (exakt wie seine bestehenden fromArray/zeros/ones es bereits tun, threaded.ts:812-825)
stack<const Rows extends readonly WNDArray<any>[]>(rows: StackRowsGuard<Rows>): StackResultOf<Rows> {
  this.assertLive("stack");
  return WNDArray.stack(this.pool.core, rows);
}
```

`this.core` in `ThreadedBackend` ist ein TS2339 — von Baustein 0 live reproduziert. **`WNDArray` wird
NICHT neu exportiert** (das wäre eine eigenständige API-Entscheidung des Owners, kein Nebenprodukt
dieser Scheibe).

**`core` als erster Parameter** (statt aus `rows[0]` abgeleitet) ist bewusst: JEDER `WNDArray`-Static
trägt heute `core` als ersten Parameter (`fresh`/`zeros`/`ones`/`fromArray`), die Facade hat ihren
`core` ohnehin, und der leere-`rows`-Fall bleibt wohldefiniert. `stack` prüft dann jede Zeile gegen
DIESEN `core` (statt gegen `rows[0]`s), was die Fehlklasse „Zeile aus einem fremden Core" mit der
bestehenden Meldung abdeckt.

### D3 — Runtime-Semantik `WNDArray.item`

```ts
item<const Idx extends readonly number[]>(...indices: ItemGuard<S, Idx>): number {
  this.assertLive("item");
  const flat = itemOffsetStrided(this.shape, this.strides, this.offset, indices as unknown as readonly number[]);
  // Frisch abgeleitete View (Memory-Regel: nie über eine Aufrufgrenze hinweg cachen —
  // bei Shared Memory schlägt das SILENT fehl). Keine Allokation zwischen Ableitung und Lesen.
  const view = new Float64Array(this.core.memory.buffer, this.buf.ptr, this.buf.lenElems);
  return view[flat] ?? NaN;
}
```

- **Strided, nicht contiguous:** der Offset kommt aus `this.strides` + `this.offset` — NIEMALS aus
  `computeStrides(this.shape)`. Das ist der zentrale Unterschied zu `itemRuntime` (dessen Daten immer
  contiguous sind) und der Grund, warum View-Fälle in D6 Pflicht sind (Arbeitsregel 12). Der
  Pflicht-Mutant M-a zielt exakt darauf.
- **Kein Allokations-, kein Kernel-Aufruf.** `getResidentFreeCount()`-Delta über viele `item`-Aufrufe
  ist exakt `0` — als Assertion getestet (D6), nicht behauptet.
- **`?? NaN`** spiegelt `itemRuntime:1037` wortgleich (strukturell unerreichbar, weil die
  Bounds-Prüfung bereits gehalten hat — dieselbe Klasse wie der unerreichbare Output-Längen-Guard aus
  S0/M6).
- **rank 0:** `item()` ohne Argumente liest `offset` — funktioniert ohne Sonderfall.
- **Ausrichtung:** `this.buf.ptr` ist per Allokator-Konvention 8-Byte-ausgerichtet (abi.rs-Modul-Doc,
  „an 8-byte-alignment convention for every allocation"), `new Float64Array(buffer, ptr, len)` ist
  damit gültig. Für `lenElems === 0` (leerer Puffer, `ptr === 0`) ist die View legal und wird nie
  gelesen — jede Achse mit `d === 0` wird von der Bounds-Prüfung abgelehnt.

### D4 — Die beiden neuen Validierungs-Helfer in `runtime.ts` (und warum dupliziert, nicht refaktoriert)

```ts
export function itemOffsetStrided(
  shape: readonly number[],
  strides: readonly number[],
  base: number,
  indices: readonly number[],
): number;                       // wirft die drei item-Stämme; liefert den flachen Element-Offset

export function stackValidateShapes(
  shapes: readonly (readonly number[])[],
): { n: number; d: number };     // wirft die drei stack-Stämme; liefert Zeilenzahl + Zeilenlänge
```

Beide reproduzieren die Prüf-REIHENFOLGE und die Message-Stämme ihrer Referenz-Gegenstücke exakt:

- `itemOffsetStrided`: Arity (`item: expected N indices (got M)`) → pro Achse Integer
  (`item: index X for axis A is not an integer`) → Negativ-Normalisierung (`i < 0 -> i + d`) →
  Bounds (`item: index X is out of bounds for axis A with dim D`) → `base + Σ i * strides[axis]`.
- `stackValidateShapes`: leer (`stack: expected at least one row`) → pro Zeile Rang
  (`stack: expected 1-D rows (got shape [...] at index i)`) → pro Zeile Länge gegen die ERSTE
  (`stack: row length mismatch (expected D, got X at index i)`), links-nach-rechts, erster Verstoß
  gewinnt.

**Warum nicht `itemRuntime`/`stackRuntime` refaktorieren, sodass sie diese Helfer aufrufen?** Weil
sie in DIESER Scheibe das Korrektheits-Orakel sind. Ein Parität-Beweis, der das Orakel im selben
Commit umbaut, beweist weniger: eine gemeinsame Regression wäre für den Differentialtest unsichtbar.
Dazu kommt die dokumentierte `runtime.ts`-Append-Konvention (FOLLOWUPS: ein Edit an einer
Bestandsfunktion ist eine bewusste Disziplin-Entscheidung, keine Nebensache). **Der Preis ist
Drift-Risiko** — und der wird nicht weggeredet, sondern mechanisch abgesichert: T4 pinnt alle sechs
Stämme cross-surface auf Wortgleichheit, mit Mutant. **Konsolidierung** (`itemRuntime`/
`stackRuntime` auf die Helfer umstellen) geht als eigener FOLLOWUPS-Mini raus, nach der Scheibe.

### D5 — Runtime-Semantik `WNDArray.stack` (N × `nt_materialize` in EINEN Ausgabepuffer)

```ts
static stack<const Rows extends readonly WNDArray<any>[]>(
  core: CoreExports,
  rows: Guard<StackCheck<WRowShapesOf<Rows>>, Rows>,
): WNDArray<OkShape<StackShape<WRowShapesOf<Rows>>>>
```

Reihenfolge der Prüfungen (spiegelt `add`s eigene Reihenfolge: Liveness → Core → Shape):

1. `r.assertLive("stack")` für jede Zeile, links nach rechts.
2. Core-Prüfung jeder Zeile gegen den `core`-Parameter — Meldung wortgleich `assertSameCore`s
   (`WNDArray.stack: operands belong to different WASM core instances`).
3. `stackValidateShapes(rows.map(r => r.shape))` → `{ n, d }` (leeres `rows` wirft hier, bevor je ein
   Puffer angefasst wird).

Datenbewegung:

```ts
const outLen = n * d;
const scratch: ScratchBuf[] = [];
try {
  const outDataBuf = allocBytes(core, outLen * 8);       // fresh, aliast nie eine Zeile
  const shapeBuf = writeU32Array(core, [d]);             // ALLE Zeilen haben dieselbe Shape [d]
  scratch.push(shapeBuf);
  const stridesBuf = allocBytes(core, 4);                // pro Zeile neu beschrieben
  scratch.push(stridesBuf);
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    // Frische View VOR jedem Schreiben (ein vorheriger nt_materialize kann memory.grow ausgelöst
    // und jede vorher abgeleitete View detached haben — Memory-Regel).
    new Uint32Array(core.memory.buffer, stridesBuf.ptr, 1)[0] = r.strides[0] ?? 0;
    const status = core.nt_materialize(
      shapeBuf.ptr, 1, stridesBuf.ptr, r.offset, r.buf.ptr, r.buf.lenElems,
      outDataBuf.ptr + i * d * 8, d,
    );
    if (status !== 0) {
      freeBuf(core, outDataBuf);                          // Ausgabepuffer entkommt nie im Fehlerfall
      throw new Error(`wasm resident nt_materialize (stack): status ${status} for row ${i} with shape [${r.shape.join(",")}]`);
    }
  }
  return WNDArray.fresh(core, [n, d], outDataBuf.ptr, outLen);
} finally {
  for (const buf of scratch) freeBuf(core, buf);
}
```

- **Genau ZWEI Scratch-Allokationen, unabhängig von `n`** (eine Shape-, eine Strides-Zelle), statt
  `2n`. Zulässig, weil alle Zeilen dieselbe Shape `[d]` haben; nur der Stride variiert.
- **`outDataBuf.ptr + i * d * 8`** ist gültig: `validate_region(out_data_ptr, out_len, 8)` prüft die
  Sub-Region, `read_slice_mut` baut den Slice an genau dieser Adresse, und die Ausrichtung bleibt
  8-Byte (Basis 8-ausgerichtet, `i*d*8` ist ein Vielfaches von 8). `memory.grow` verschiebt bestehende
  Allokationen nie — der rohe `ptr` bleibt über den ganzen Loop gültig (nur JS-Views detachen).
- **`d === 0`** (Zeilen der Länge 0, valider W4-Fall: `[[],[]]` → `[2, 0]`): `outLen === 0`,
  `allocBytes(core, 0)` liefert `ptr === 0` ohne Throw, und `nt_materialize` mit `out_len === 0` ist
  per ABI-Kontrakt gültig (`read_slice_mut` hat einen expliziten `len == 0`-Sonderfall,
  abi.rs:141-147; die Modul-Doku benennt zero-length-Regionen ausdrücklich als legal). **Kein
  Sonderfall im Code** — der Loop läuft durch. Das ist zugleich der erste Test des size-0-Output-Pfads
  von `nt_materialize` (FOLLOWUPS-Mini) und wird als solcher explizit getestet. **v2:** Baustein 0 hat
  diesen Pfad am echten Artefakt durchgespielt (`nt_alloc(0)` gibt `ptr = 0` ohne Throw,
  `nt_materialize` mit `out_len = 0` gibt `status = 0`) — das in v1 vorgehaltene `if (d > 0)`-Gate
  entfällt ersatzlos.
- **Kein Aliasing:** der Ausgabepuffer ist frisch; er kann keine Zeile überlappen. Zeilen dürfen
  einander und sich selbst aliasen (`stack([a, a])` ist gültig und getestet).

### D6 — Typ-Ebene

**`item` — zweite Call-Site, NULL neue Typ-Maschinerie:**

```ts
item<const Idx extends readonly number[]>(...indices: ItemGuard<S, Idx>): number
```

Byte-gleich zu `NDArray.item` (ndarray.ts:943). `ItemGuard` ist bereits shape-generisch (vector.ts:649)
— genau wie W5s FOLLOWUPS-Eintrag es vorhergesagt hat. Kein `Guard<>`-Wrapper (F1: TS2370 an dieser
Deklaration), Arity nativ über TS2554 (F3), Spread-Form über `IsDynamicRank<Idx>` (F4).

**`stack` — zweite Call-Site plus zwei gespiegelte Unwrap-Aliase in `resident.ts`:**

```ts
type UnwrapWRow<R> = R extends WNDArray<infer S> ? S : never;
type WRowShapesOf<Rows extends readonly WNDArray<any>[]> = { [I in keyof Rows]: UnwrapWRow<Rows[I]> };
```

Strukturspiegel von `ndarray.ts:150-151`, inklusive beider dort teuer erkaufter Eigenschaften, die
hier GLEICHERMASSEN binden (`WNDArray` trägt denselben `__variance`-Invarianz-Marker wie `NDArray`,
resident.ts:300):

- **homomorph** (`{ [I in keyof Rows]: ... }`), nie `Rows[number] extends WNDArray<infer S>` — die
  nicht-homomorphe Form kollabiert auf heterogenen Tupeln zu `never` (W4-F2, BLOCKER-Klasse);
- **frischer nackter Typparameter** (`UnwrapWRow`) statt inline-Conditional — sonst kollabiert der
  Array-mit-Union-Elementtyp-Fall still zu `never` (W4-F8).

Beide Eigenschaften bekommen je einen Typ-Pin, der bei Verletzung rot wird (T6) — nicht nur einen
Kommentar. `StackCheck`/`StackShape` (vector.ts) bleiben UNVERÄNDERT.

**Zwei exportierte Signatur-Aliase (v2, Antwort auf den Baustein-0-MAJOR):**

```ts
export type StackRowsGuard<Rows extends readonly WNDArray<any>[]> = Guard<StackCheck<WRowShapesOf<Rows>>, Rows>;
export type StackResultOf<Rows extends readonly WNDArray<any>[]> = WNDArray<OkShape<StackShape<WRowShapesOf<Rows>>>>;
```

Sie lösen zwei Probleme auf einmal:

1. **Verdrahtung.** Die Signatur wird an DREI Stellen gebraucht (Static + zwei Facaden). Ohne die
   Aliase müssten `backend-api.ts` und `threaded.ts` je vier Typen aus `ndarray.ts`/`vector.ts` neu
   importieren; mit ihnen importieren beide nur aus `resident.ts`, aus dem sie ohnehin schon
   importieren — keine neue Import-Kante im Graphen.
2. **Budget.** `ndarray.ts:118-125` dokumentiert gemessen, dass TS zwei textgleiche, aber SEPARAT
   geschriebene Mapped-Type-Ausdrücke NICHT dedupliziert (dort: ca. 1.428 statt ca. 801
   Instantiations für zwei Call-Sites). Bei drei Call-Sites wäre das Ausschreiben die teuerste
   erreichbare Variante. Ein benannter Alias wird dedupliziert.

`UnwrapWRow`/`WRowShapesOf` bleiben dagegen **file-privat** — genau wie ihre `ndarray.ts`-Vorbilder
(Baustein-0-Befund: `RowShapesOf` ist nicht exportiert und wird nirgends außerhalb referenziert). Die
Aliase sind resident-interne API und gehen NICHT in `spike/src/index.ts`.

### D7 — Testplan

Alle Anhänge gehen in BESTEHENDE Dateien (Z2, kein Order-Noise).

**T3 — M1-Differential `item` (`resident.test.ts`).** Randomisierte Shapes Rang 0–4 × randomisierte
Index-Tupel, `WNDArray.item(...)` vs `NDArray`-Referenz über `itemRuntime`, `Object.is`-gleich (bzw.
Bit-Vergleich via `bitsOf`). Pflicht-Abdeckung:
- **contiguous** (`fromArray`);
- **VIEWS, explizit gefordert (Arbeitsregel 12):** transponiert, geschnitten (`slice`, nonzero
  offset), offset-verschoben, ZUSAMMENGESETZT (transponiert-dann-geschnitten). Die Referenz für eine
  View ist `itemRuntime(view.shape, view.toArray(), indices)` — `toArray()` liefert die logische
  row-major Kopie, gegen die `itemRuntime`s contiguous-Annahme gilt;
- negative Indizes (jede Achse), gemischt positiv/negativ;
- rank 0 (`item()`);
- `d === 0`-Achse: JEDER Index wirft (Bounds), auf beiden Flächen mit derselben Meldung.

**T3b — Spezialwerte `item`/`stack` (`special-values.test.ts`).** `genDataSpecial`-Raster (ca. 60
Fälle je Op), bit-identisch zur Referenz. Plus eine **nicht-kanonische NaN-Payload-Fixture**: für
`stack` MUSS der Payload byte-exakt überleben (Movement-Op, Muster: die bestehende
transpose-Payload-Fixture); für `item` wird das tatsächliche Verhalten GEMESSEN und dokumentiert
(Cross-Surface-Gleichheit ist die bindende Aussage, nicht Payload-Erhalt).

**T4 — Cross-Surface-Message-Parität (`resident.test.ts`), mit Mutant.** Für alle SECHS Stämme (item:
Arity/Integer/Bounds; stack: leer/Rang/Längen-Mismatch) wird dieselbe fehlerhafte Eingabe auf beiden
Flächen ausgeführt und `err.message` auf **String-Gleichheit** geprüft — nicht auf ein Präfix. Das ist
die mechanische Absicherung gegen das in D4 offengelegte Drift-Risiko. Nicht-Vakuität: Mutant M-c.

**T5 — `stack`-Differential (`resident.test.ts`).** Randomisierte `N`/`D` (inkl. `N=1`, `D=0`,
`D=1`, große `N`), Zeilen gemischt contiguous/View (transponierte Matrixzeile via `slice`,
geschnittene Zeile), Ergebnis-Shape + Daten bit-identisch zu `stackRuntime`. Plus: `stack([a, a])`
(dieselbe Zeile doppelt), Zeilen aus verschiedenen Puffern, und der Fehlerpfad
„Zeile aus fremdem Core".

**T7 — Lebenszyklus (`resident-lifecycle.test.ts`).**
- `item`: `getResidentFreeCount()`-Delta über 500 Aufrufe ist **exakt 0** UND es entsteht keine
  Allokation (kein Puffer-Wachstum) — item ist allokationsfrei.
- `stack`: Free-Count-Delta über viele Aufrufe zeigt exakt die zwei Scratch-Freigaben pro Aufruf und
  kein Netto-Leck; das Ergebnis bleibt nach `dispose()` der Zeilen gültig (unabhängiger Puffer).
- Fehlerpfad: eine mitten im Loop scheiternde Zeile gibt Ausgabepuffer UND Scratch frei (per
  Mock-Core/Status-Injektion nach `backend-oom.test.ts`-Vorbild, falls deterministisch erzwingbar;
  sonst als bewusst ungetestet benennen, nicht behaupten).
- `item`/`stack` auf einem disposed Handle werfen mit `WNDArray.item:`/`WNDArray.stack:`-Präfix.

**T8 — Threaded-Parität (`threaded.test.ts`, `backend-api-threaded.test.ts`).** `item` und `stack`
auf dem THREADED Core bit-identisch zum STABLE Core, je inkl. eines `genDataSpecial`-Falls
(S0/C-2-Lektion) und eines View-Falls; `ThreadedBackend.stack` erreichbar und korrekt.

**T6 — Typ-Pins (`ndarray.test-d.ts`, `backend-api.test-d.ts`).** WNDArray-`item`- und
-`stack`-Gruppe, spiegelt die NDArray-W4/W5-Pins als zweite Call-Site: Rückgabetyp `number`;
Arity-TS2554; OOB-/Dot-Form-Ablehnung am richtigen Argument; Degradation (dynamischer Rang, Union,
Spread) zu no-claim; `stack`-Tupel → `[N, D]`, Array → `[number, D]`, Längen-Mismatch abgelehnt,
leeres Tupel abgelehnt, heterogene Tupel (F2-Pin) und Array-mit-Union-Elementtyp (F8-Pin).
**Budget-Disziplin (W5-D6-Lektion, bindend):** `Equal<...>`-Message-Pins kosten ca. 1.700
Instantiations pro Pin. Es wird EIN konsolidierter Mehr-Positionen-Pin je Fehlerklasse gesetzt, nicht
je Kante einer.

**Pflicht-Mutanten (während Baustein A, je Revert per Backup-Kopie + `diff`-Beweis — NIE
`git checkout`, harte Arbeitsregel 1):**
- **M-a (item, View-Coverage):** `this.strides`/`this.offset` → `computeStrides(this.shape)`/`0`.
  MUSS von den View-Fällen (T3) gefangen werden; wird er von den contiguous-Fällen allein NICHT
  gefangen, ist das der Beweis, dass die View-Abdeckung nicht-vakuös ist.
- **M-b (stack, Slot-Offset):** `outDataBuf.ptr + i * d * 8` → `outDataBuf.ptr`. MUSS vom
  T5-Differential gefangen werden.
- **M-c (Message-Drift):** einen Stamm im neuen Helfer minimal ändern (z. B. `expected` →
  `expecting`). MUSS von T4 gefangen werden — sonst ist T4 vakuös und die D4-Duplikation
  ungesichert.
- **M-d (stale View unter `memory.grow`, v2 — der gefährlichste Mutant der Scheibe):** die
  Strides-View EINMAL vor der Schleife ableiten statt pro Iteration frisch. Baustein 0 hat live
  bewiesen, dass `nt_materialize`s interne `Vec`-Allokation `memory.grow` auslöst (Byte-Länge
  1,1 MB auf 130 MB über 40 Aufrufe), dass ein Wachstum den alten `ArrayBuffer` **detacht**, und
  dass ein Schreibversuch über die veraltete View ein **stiller No-Op** ist — der Kernel meldet
  danach `status === 0` bei komplett falschen Daten. Diese Fehlerklasse hat keinen Statuscode und
  keinen Throw. **Konsequenz für den Testplan (bindend):** T5 MUSS mindestens einen `stack`-Fall
  enthalten, der groß genug ist, um während des Loops tatsächlich `memory.grow` auszulösen —
  sonst ist M-d nicht fangbar und die Disziplin ungesichert. Der Fall wird so dimensioniert, dass
  das Wachstum beobachtet und im Test assertiert wird (`core.memory.buffer.byteLength` vor/nach),
  damit er nicht später still unter die Wachstumsschwelle rutscht.

### D8 — Pins, Budget, Freeze

- **`check:freeze`-Hash UNVERÄNDERT (M4).** Clean-Rebuild reproduziert `8255821b…` exakt;
  `check:freeze` grün OHNE Pin-Änderung; `git status crates/` leer; `cargo test` unverändert 184+1.
  Bewegt sich der Hash: STOP + untersuchen.
- **`check:diag` (Root) — Absolut-Gate ≤ +13.000 (v3, Owner-abgenommen 2026-07-24; v1/v2: ≤ +8.000),
  gestuft gemessen.** **Warum angehoben:** Baustein B fand per LSP-Hover-Sonde, dass die drei
  `stack`-Signaturen mit dem v2-Rückgabetyp-Alias `StackResultOf<Rows>` an der einzigen für
  Paketkonsumenten erreichbaren Fläche als `StackResultOf<readonly [WNDArray<[3]>, …]>` hovern statt
  als sauberes `WNDArray<readonly [2, 3]>` — ein Bruch der Hausregel „hovers must show clean resolved
  tuples" und dem Wortlaut von COVENANT M3 („Klassen-Hover bleiben saubere Tupel") zuwider. Der Fix
  (Alias trägt nur die SHAPE, das Handle wird ausgeschrieben: `WNDArray<StackShapeOf<Rows>>`) ist von
  B mit derselben Sonde als wirksam nachgemessen und kostet **+5.498** Instantiations. Drei
  Alternativen wurden durchgemessen und verworfen: Typ-Pin-Konsolidierung trägt nichts bei (−21),
  eine Zwei-Alias-Aufteilung ist teurer (227.020), und die Test-Aufrufstellen laufen bereits über
  dynamische Shapes. Die Kosten sind strukturell. Der Owner hat die Abwägung mit den gemessenen
  Zahlen entschieden (Hover-Konformität schlägt Budget) und das Scheiben-Gate begründet auf
  ≤ +13.000 gesetzt; 226.690 sind ca. 4,5 % des Instantiation-Budgets. Ursprüngliche Begründung der
  +8.000 (unverändert gültig für den Teil ohne Hover-Fix): (der
  Handoff verlangt ausdrücklich ein eigenes, begründetes Gate statt reflexhaft +6.000): (a) diese
  Scheibe trägt ZWEI Ops, alle bisherigen Kampagnen-Scheiben trugen eine; (b) `item`s NDArray-Original
  (W5) kostete +5.873 und `stack`s (W4) rippelte +845 uniform — hier entfallen zwar die
  Maschinerie-Deklarationskosten (`ItemGuard`/`StackCheck` existieren), es kommen aber zwei neue
  Typ-Aliase mit eigenen Deklarationskosten hinzu (Spike-03/04-Befund: ungenutzte Maschinerie ist
  NICHT gratis) und drei Klassen-Surfaces wachsen (`WNDArray`, `WasmBackend`, `ThreadedBackend`);
  (c) die nach Arbeitsregel 12 verpflichtenden View-Fälle kosteten bei S2 allein +4.189.
  **Gestufte Messung** (Baseline im frischen `git worktree` von HEAD `755d9ac`: check:diag
  213.704 @ 140, stress 107.283 @ 82, browser 2.142 @ 75, Freeze `8255821b…`), Δ nach:
  ① runtime.ts-Helfer, ② `item` + `stack` + Typ-Aliase in resident.ts, ③ Facaden, ④ Test-Anhänge,
  ⑤ Typ-Pins. Überschreitet eine Stufe +3.000, wird VOR dem Weiterbauen konsolidiert (W5-Technik).
  Exit-Code + Fehleranzahl IMMER mitprüfen (Arbeitsregel 6).
- **stress/browser:** Ripple erwarten und deterministisch attribuieren (stress importiert
  `spike/src` direkt → Klassen-Surface-Ripple; browser kompiliert `threaded.ts` nicht). Doppelmessung.
- **`bench:editor` (8 Pins W1–W8):** messen; bei Bewegung uniform neu setzen (Doppelmessung,
  byte-identisch, Dekomposition ins Ergebnisse-Doc). Erwartung: uniforme Bewegung durch das
  `WNDArray`-Klassen-Surface (dritter Mechanismus); `CoreExports` wächst nicht ⇒ vierter Mechanismus
  (keyof) trägt +0.
- **Testzahlen:** `test:resident`, `test:threaded` steigen; `test:core` unverändert (keine
  test:core-Datei berührt); `cargo` unverändert. Exakte Deltas ins Ergebnisse-Doc.

### D9 — Covenant-Kandidaten (nicht still auflösen)

Der v6-Kandidaten-Stapel bekommt zwei Facetten dazu, falls Baustein C sie bestätigt:

- **M1 für resident-aber-vollständig-in-TS berechnete Ops** (`item` — kein WASM-Code läuft, aber die
  Op arbeitet auf WASM-residenten Daten und muss bit-identisch sein). Baustein 0 hat die Analyse
  bestätigt: M1 v5 kennt wörtlich nur „jeder WASM-Kern" und „kernel-lose Referenz-Op" (Op existiert
  NUR in `runtime.ts`); `item` ist keines von beidem. Der S2-Kandidat („komponierte Ops") bleibt
  davon getrennt und wird durch `stack` ein zweites Mal instanziiert.
- **M3 für Cross-Surface-Message-Parität** (v2, Baustein-0-Befund, minor): M3s Wortlaut adressiert
  Typ-vs-Runtime-Konsistenz innerhalb EINER Fläche, vermittelt über `Guard`. Der hier bindende
  T4-Test prüft Runtime-vs-Runtime zwischen ZWEI Flächen — eine andere Achse, die der Vertragstext
  nicht benennt. Der Test bleibt Pflicht (er sichert D4s Duplikations-Drift real ab); nur die
  Zuordnung „das ist M3" ist eine Dehnung.

Beides geht als Owner-Entscheidung in FOLLOWUPS, nicht als stille COVENANT-Änderung.

### D10 — Doc-Platzierung

Ergebnisse-Doc `docs/wasm-parity-item-stack-ergebnisse.md` (volles Narrativ +
Post-Verification-Addendum). FOLLOWUPS: Kampagnen-Zeile (S3 erledigt), das W4/W5-Paritätsitem
(item/stack-Lücken schließen; argmax/topk bleiben → S4/S5), NEU: der D4-Konsolidierungs-Mini und die
v6-Kandidaten. `docs/roadmap.md` WASM-Parität-Sektion. CLAUDE.md „Status" + „Aktuelle Pins & Gates"
(nur Einzeiler + IST-Zahlen). Vollnarrativ an `docs/projekt-log.md`.

### D11 — Sprache

Code/Kommentare/Tests/Commit-Message: Englisch (Hard Constraint). Spec + Ergebnisse-Doc: Deutsch.
„ca." statt Tilde, keine Strikethroughs (GFM-Gate, harte Arbeitsregel 4).

## Akzeptanzkriterien

- **T1:** Alle vier Dateien nur per Insertion/Append geändert (D1-Tabelle); kein Edit an
  `itemRuntime`/`stackRuntime`/einem `WNDArray`-Bestandsmember; kein Rust/ABI/CoreExports/loader.
- **T2 (Freeze + Arbeitsregel 10):** Clean-Rebuild reproduziert `8255821b…`; `check:freeze` grün ohne
  Re-Pin; `cargo test` 184+1 unverändert; **explizit verifiziert**, dass `CoreExports` unverändert
  ist und `backend-oom.test.ts` deshalb keinen neuen Stub braucht (nicht angenommen — geprüft, denn
  ein fehlender Stub würde als TS2739 auftreten und `check:diag` druckte trotzdem eine plausible
  Zahl).
- **T3 (M1 `item`):** Differential grün über contiguous + alle vier View-Klassen + negative Indizes +
  rank 0 + `d === 0`; Spezialwert-Raster grün.
- **T4 (M3):** alle sechs Stämme cross-surface WORTGLEICH, per String-Gleichheit gepinnt; Mutant M-c
  gefangen.
- **T5 (M1 `stack`):** Differential grün inkl. `N=1`, `D=0`, aliasende Zeilen, View-Zeilen; NaN-Payload
  byte-erhalten.
- **T6 (Typ):** Pins für beide Ops auf der WNDArray-Fläche + beiden Facaden, inkl. der F2- und
  F8-Regressions-Pins; konsolidierte Message-Pins (Budget); keine neue konfident-falsche Kante.
- **T7 (Lebenszyklus):** `item` allokationsfrei (Free-Count-Delta exakt 0); `stack` leckfrei inkl.
  Fehlerpfad; disposed-Handle-Throws.
- **T8 (threaded):** item/stack threaded-vs-stable bit-identisch, inkl. Spezialwert- und View-Fall;
  `ThreadedBackend.stack` erreichbar.
- **T9 (Gates/Pins):** Gate-Block grün; `check:diag`-Δ in fünf Stufen dekomponiert und gegatet
  (≤ +13.000, v3); stress/browser attribuiert; bench:editor gemessen und ggf. neu gesetzt.
- **T12 (Hover, v3):** Die drei `stack`-Aufrufformen (statische Methode + beide Facaden) hovern als
  aufgelöstes `WNDArray<readonly [N, D]>`, nicht als Signatur-Alias — per LSP-Sonde gegen den echten
  `tsc --lsp`-Server belegt, mit einer Bestandsmethode (`add`) als Kontrollpunkt.
- **T10 (Mutanten):** M-a, M-b, M-c, M-d je nachweislich gefangen (benannte Assertion) und per
  Backup-Kopie revertiert; für M-d zusätzlich der Nachweis, dass der Testfall `memory.grow`
  wirklich auslöst (sonst wäre das Fangen Zufall).
- **T11 (Docs):** D10 vollständig; v6-Kandidaten in FOLLOWUPS.

## Nicht-Ziele

Kein neuer Kernel (weder `nt_item` noch `nt_stack`), kein Threaded-Pool-Kernel, keine Änderung an
`NDArray.item`/`NDArray.stack`/`itemRuntime`/`stackRuntime`, kein `WNDArray`-Export aus `index.ts`,
kein `stack` mit `axis`/höherem Rang/`concat`/`vstack`/`hstack`, kein `item`-Setter, kein `at`-Alias,
keine `WasmBackend.from(view)`-Ergonomie (bleibt FOLLOWUPS), keine andere Paritäts-Op (argmax/topk
sind S4/S5), keine Behauptung eines gemessenen Nutzerbedarfs, kein Browser-Port des Threads-Pfads,
keine stille COVENANT-Änderung.

## Gate-Block / Definition of Done

`pnpm check` (Dreier-Verbund) · `check:diag`(+`:stress`/`:browser`, Pin-Protokoll D8) ·
`pnpm test:core` · `pnpm test:resident` · `pnpm test:threaded` (baut beide Artefakte) ·
`cargo test --manifest-path crates/core/Cargo.toml` (UNVERÄNDERT) · `pnpm check:freeze` (Hash
UNVERÄNDERT, kein Re-Pin) · `pnpm bench:editor` (8 Pins) · `pnpm test:package` · `pnpm test:example`
(unberührt) · `graph-a-lama query lint` · GFM-Gate auf allen neuen/geänderten `.md`.

## Verify-Plan (Stufe 3)

**Baustein 0 (vor dem Bau, gegen DIESE Spec, adversarial — `brainroute:deep`): DURCHGEFÜHRT
2026-07-23, kein Design-Blocker gegen das Kern-Design — ein BLOCKER und ein MAJOR auf der
Verdrahtungs-Ebene, beide in v2 eingearbeitet. Befunde im Addendum unten.**
Auftrag aus `docs/verify-runde-template.md`, Schwerpunkte:
(a) Ist `nt_materialize` in einen **Sub-Slot** eines größeren Ausgabepuffers (`ptr + i*d*8`, `out_len
= d`) wirklich vertragskonform und in der Praxis korrekt — empirisch am echten Artefakt prüfen, nicht
nur am ABI-Text?
(b) Trägt der `d === 0`-Pfad (`out_len === 0`, `ptr === 0`) wirklich, oder braucht er ein Gate?
(c) Ist die Zwei-Scratch-Puffer-Optimierung (Strides-Zelle pro Zeile neu beschrieben) unter
`memory.grow` sicher — kann ein `nt_materialize`-Aufruf die Memory wachsen lassen, und ist die
frische View pro Iteration ausreichend?
(d) Sind die beiden neuen Helfer WIRKLICH semantik-gleich zu `itemRuntime`/`stackRuntime` (alle
Reihenfolgen, alle Randfälle, `?? 0`/`?? NaN`-Fallbacks) — empirisch gegenrechnen, nicht lesen?
(e) Ist `item` über `this.strides`/`this.offset` für ALLE View-Formen korrekt (transponiert,
geschnitten, komponiert, negative Indizes auf Views)?
(f) Erreichbarkeits-Argument D2: stimmt es, dass `WNDArray` nicht exportiert ist, und sind die
Facaden-Insertionen die kleinste korrekte Antwort — oder gibt es eine bessere?
(g) Typ-Ebene: kollabieren `UnwrapWRow`/`WRowShapesOf` wirklich nicht (F2/F8 gegen `WNDArray`s
Invarianz-Marker EMPIRISCH nachstellen); ist `ItemGuard` an einer `WNDArray`-Methode wirklich
unverändert brauchbar?
(h) Freeze WIRKLICH unverändert; `CoreExports` wirklich unverändert (Arbeitsregel 10).
(i) Covenant: M1-Einordnung beider Fälle, M3-Duplikations-Risiko, M5 (threaded.ts-Edit), Z2.
(j) Test-/Typ-Pin-Lücken; ist das ≤ +8.000-Gate realistisch oder zu lasch/zu streng?
Befunde mergen, Design-Blocker mit dem Owner in die Spec einarbeiten, „Adversariale
Spec-Verifikation (Addendum)" hier anhängen, DANN implementieren.

**Nach der Implementierung:** voller Katalog — **A** (Spec-Konformität pro D, alle Gates frisch,
Freeze-Unveränderlichkeit selbst reproduziert, eigener Mutant), **B** (adversarial: beide Ops gegen
ein UNABHÄNGIGES Orakel — eigener Speicher-Lese-/Gather-Code, nicht `itemRuntime`/`stackRuntime`;
komponierte Views, Spezialwert-Puffer, `d === 0`/rank 0/`N=1`, Lebenszyklus unter Fehlerpfaden,
Mess-/Freeze-Randbedingungen), **C** (`covenant:covenant-verify`: M1-Einordnung beider Fälle,
M3-Wortgleichheit, M4-Hash-Unveränderlichkeit, M5, Z1/Z2; v6-Kandidaten benennen) — parallel,
isoliert (mutierende Verifier je eigener Worktree + Slice-Patch, read-only C im Haupt-Baum). Aufträge
aus `docs/verify-runde-template.md`. Ergebnisse-Doc mit Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-23)

Verifier: `brainroute:deep`, adversarial gegen v1, eigener `git worktree` (HEAD `755d9ac`, danach
per `git worktree remove --force` aufgeräumt; Haupt-Baum nie berührt). **Verdikt: das
kernel-lose Kern-Design (D3/D5) trägt und ist empirisch bestätigt** — die zwei Befunde liegen auf
der Verdrahtungs-Ebene, nicht am Design.

**Der wichtigste Befund ist eine BESTÄTIGUNG mit Zähnen (D5/c):** Die Zwei-Scratch-Puffer-Optimierung
ist nicht nur sicher, ihre Disziplin ist **kritisch**. Drei Proben: (1) `nt_materialize`s interne
`Vec`-Allokation (`materialize.rs:15`) löst `memory.grow` tatsächlich aus (1,1 MB auf 130 MB über 40
Aufrufe); (2) ein Wachstum **detacht** den alten `ArrayBuffer` vollständig (`byteLength` auf 0), und
ein Schreibversuch über die veraltete View ist ein **stiller No-Op**, kein Throw; (3) ein gezielter
Mutant (View einmal vor der Schleife abgeleitet) produziert **`status === 0` bei komplett falschen
Ausgabedaten**. Diese Fehlerklasse hat weder Statuscode noch Exception. Konsequenz in v2: Mutant
**M-d** plus die bindende Auflage, dass T5 einen Fall enthält, der `memory.grow` nachweislich
auslöst.

**BLOCKER (in v2 behoben) — `ThreadedBackend` hat kein `core`-Feld.** v1s D2 behauptete, beide
Facaden bekämen eine „strukturell identische" Delegation über `this.core`. Geprüft an
`threaded.ts:796-825`: die Klasse trägt `readonly pool: ThreadedPool`, der Core lebt auf
`this.pool.core` (`threaded.ts:233`) — die bestehenden `fromArray`/`zeros`/`ones` benutzen bereits
genau das. Der Verifier hat `this.core` dort live als TS2339 reproduziert und die
`this.pool.core`-Variante als sauber kompilierend gegengeprüft. Exakt die Fund-Klasse, die der
Auftrag als Präzedenzfall nennt (Item 10: `WNDArray.strides` ist ein Feld, keine Methode).

**MAJOR (in v2 behoben) — die Facaden-Signaturen brauchten eine Typ-Verdrahtung, die D1 nicht
nannte.** `RowShapesOf`/`UnwrapRow` sind in `ndarray.ts` file-privat (grep-verifiziert: außerhalb
nur zwei Kommentar-Erwähnungen), ein wörtlicher Spiegel wäre es auch — dann können die Facaden ihre
Signatur nicht ausdrücken. v1s Änderungstabelle vermerkte Import-Erweiterungen nur für `resident.ts`.
**Die v2-Lösung geht über den Vorschlag des Verifiers hinaus:** statt vier neuer Typ-Importe je
Facade (und einer Export-Ausnahme für `WRowShapesOf`) exportiert `resident.ts` zwei fertige
Signatur-Aliase (`StackRowsGuard`/`StackResultOf`, D6). Das vermeidet jede neue Import-Kante, hält
`UnwrapWRow`/`WRowShapesOf` file-privat wie ihre Vorbilder — und ist zusätzlich die günstigere
Variante, weil `ndarray.ts:118-125` gemessen dokumentiert, dass TS textgleiche, separat geschriebene
Mapped-Type-Ausdrücke nicht dedupliziert (hier: drei Call-Sites).

**Empirisch bestätigte Annahmen (Auswahl, je mit eigener Probe am echten Artefakt):**

- **(a)** `nt_materialize` in einen Sub-Slot (`outPtr + i*d*8`, `out_len = d`) ist ABI-konform und
  praktisch korrekt: sechs Konfigurationen `n×d` aus `{(1,5),(3,4),(5,1),(7,0),(1,0),(4,8)}`,
  alle Slots byte-exakt, keine Cross-Row-Korruption.
- **(b)** Der `d === 0`-Pfad trägt **ohne** Gate: `nt_alloc(0)` liefert `ptr = 0` ohne Throw,
  `nt_materialize` mit `out_len = 0` liefert `status = 0`. Das `if (d > 0)`-Fallback aus v1 entfällt.
- **(d)** Die zwei neuen Helfer sind semantik-gleich zu den Orakeln: je 20.000 randomisierte Fälle
  (item: Arity/Integer/Bounds; stack: leer/Rang/Längen-Mismatch), 0 Abweichungen in Wurf-Verhalten
  UND Wortlaut.
- **(e)** `item` über `this.strides`/`this.offset` ist für alle View-Formen korrekt: 15.000
  randomisierte Fälle über transponierte, geschnittene (nonzero offset) und komponierte Views inkl.
  negativer Indizes, 0 Abweichungen.
- **(f)** `WNDArray` ist tatsächlich nicht aus `index.ts` exportiert — das Erreichbarkeits-Argument
  aus D2 hält.
- **(g)** `UnwrapWRow`/`WRowShapesOf` kollabieren gegen `WNDArray`s Invarianz-Marker **nicht**
  (heterogenes Tupel bleibt `[[2,3],[4,5]]`, Array-mit-Union-Element bleibt die Union) — und die
  jeweils „falschen" Formen kollabieren nachweislich zu `never`, die Probe ist also diskriminierend,
  nicht vakuös. `ItemGuard<S, Idx>` ist an einer `WNDArray`-Rest-Parameter-Methode unverändert
  drop-in nutzbar (sechs Fälle live kompiliert: TS2554-Arity, OOB, Dot-Form, rank 0, dynamischer
  Rang, Spread).
- **(h)** Clean-Rebuild reproduziert `8255821b…` exakt, `cargo test` 184+1 unverändert,
  `git status crates/` leer; `nt_materialize` ist seit Kern 03 im `backend-oom`-Mock gestubbt —
  Arbeitsregel 10 greift bestätigt nicht.
- Privater Feldzugriff auf FREMDE Instanzen aus einer statischen Methode heraus ist zulässig
  (TS-Privacy ist klassenweit) — Präzedenz `assertSameCore` (`resident.ts:355-359`), zusätzlich per
  eigener Probe bestätigt.

**Zwei Befunde, die bewusst NICHT zu Spec-Änderungen führen:**

- **M3-Zuordnung ist eine Dehnung (minor).** M3s Wortlaut adressiert Typ-vs-Runtime-Konsistenz
  INNERHALB einer Fläche; T4 prüft Runtime-vs-Runtime ZWISCHEN zwei Flächen. Der Test bleibt
  sinnvoll und bindend (er sichert D4s Duplikations-Risiko real ab); die Frage, ob M3 dafür eine
  Präzisierung braucht, geht als Kandidat an Baustein C / den Owner — nicht still aufgelöst
  (D9 erweitert).
- **Das ≤ +8.000-Gate ist „realistisch, aber knapp"** (Verifier-Wortlaut), nicht zu lasch. Der
  eingebaute Circuit-Breaker (+3.000 pro Stufe löst Konsolidierung aus) mindert das Overshoot-Risiko;
  Stufe ④ (Test-Anhänge inkl. View-Coverage) ist die zuerst zu beobachtende Stufe. Unverändert
  übernommen.

## Änderungslog

- **v3 (2026-07-24, Owner-abgenommen):** Nach der Verify-Runde. (1) `check:diag`-Absolut-Gate von
  ≤ +8.000 auf **≤ +13.000** angehoben — Baustein B belegte per LSP-Hover-Sonde, dass der
  v2-Rückgabetyp-Alias `StackResultOf<Rows>` an der Konsumenten-API einen Aliasnamen statt eines
  sauberen Tupels hovert (M3-Wortlaut); der Fix (`StackShapeOf` trägt nur die Shape, das Handle wird
  ausgeschrieben) kostet gemessene +5.498, drei billigere Alternativen wurden gemessen und verworfen.
  Owner-Entscheidung mit den Zahlen vor Augen, nicht still. (2) Neues Akzeptanzkriterium **T12**
  (Hover-Auflösung, per LSP-Sonde belegt). (3) `StackResultOf` heißt jetzt `StackShapeOf` und aliast
  nur noch die Shape. Kern-Design D1–D5 unverändert.
- **v2 (2026-07-23):** Baustein 0 (adversarial gegen v1) bestätigt das kernel-lose Kern-Design
  empirisch (Sub-Slot-`nt_materialize`, `d === 0` ohne Gate, Helfer-Semantik-Gleichheit über 40.000
  Fälle, `item` auf Views über 15.000 Fälle, Typ-Aliase kollabieren nicht) und findet zwei
  Verdrahtungs-Befunde: BLOCKER `ThreadedBackend.this.core` existiert nicht (Fix: `this.pool.core`),
  MAJOR fehlende Typ-Verdrahtung der Facaden (Fix: zwei exportierte Signatur-Aliase in `resident.ts`
  statt neuer Import-Kanten — zugleich die budget-günstigere Variante). Neu: Pflicht-Mutant **M-d**
  (stale View unter `memory.grow` → `status === 0` bei falschen Daten, live nachgewiesen) samt der
  Auflage, dass T5 einen wachstums-auslösenden Fall enthält. Das `if (d > 0)`-Fallback aus D5
  entfällt (empirisch unnötig). Kern-Design D3–D5 unverändert.
- **v1 (2026-07-23):** Erstfassung nach Owner-Richtungsabnahme (WASM-Parität S3 = item + stack).
  Zentrale Design-Wahlen: beide Ops KERNEL-LOS (Arbeitsregel 11) — `item` als reiner strided
  TS-Lesezugriff, `stack` als N × `nt_materialize` in einen Ausgabepuffer; Freeze-Hash UNVERÄNDERT;
  zwei neue `runtime.ts`-Validierungs-Helfer statt eines Refaktorings der Orakel-Funktionen
  (Drift-Risiko per Cross-Surface-Message-Paritätstest abgesichert); `stack` zusätzlich auf beiden
  Backend-Facaden, weil `WNDArray` nicht exportiert ist und ein Static sonst unerreichbar wäre;
  eigenes, begründetes check:diag-Gate ≤ +8.000 statt reflexhaft +6.000.
