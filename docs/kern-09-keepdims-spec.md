# Kern 09 — `keepdims` im Runtime-`sum()` (Binding Spec)

*Phase B, Item 5 — letzter offener Rest. Kleine Scheibe. Datum: 2026-07-12.*

> **Revision 2026-07-12 (vor dem ersten Commit, nach Owner-Review):** (1) D3 ist
> vom Owner **bestätigt** (Review-Finding F1: die Abweichung war zunächst nur
> hier dokumentiert, nicht aktiv vorgelegt — Prozessregel daraus in CLAUDE.md).
> (2) Testplan-Korrektur: die Differentialdatei gehört in die
> **`test:resident`**-Liste (nicht `test:core` wie ursprünglich geschrieben —
> sie braucht `WNDArray`; die Implementierung hatte das bereits richtig, die
> Spec zieht hiermit nach = Review-Finding F4). (3) Owner-mandatierter Nachtrag
> F5: keepdims-Fälle auf **Views** (transponiert, offset-verschoben,
> komponiert; Referenz = dieselbe Kette auf der naiven `NDArray` OHNE keepdims)
> + `WNDArray` zusätzlich im keepdims=false-Paritätsblock.

## Ausgangslage

Die **Typ-Ebene ist fertig und getestet**: `ReduceAxis<S, Axis, KeepDims>` in
`spike/src/reduce.ts` ersetzt (statt entfernt) die reduzierte Achse durch `1`,
wenn `KeepDims = true`; `undefined`-Achse + keepdims ⇒ `AllOnes<S>` (Rang
bleibt, alle Dims `1`). Belegt durch `spike/tests/reduce.test-d.ts` T3
(`ReduceAxisKeepDims<[2,3,4],1> = [2,1,4]`), T10 (`…,-1 = [2,3,1]`), T11
(`ReduceAxisKeepDims<[2,3,4]> = [1,1,1]`), T17 (`number[] ⇒ 1[]`).

**Was fehlt:** der Runtime-Parameter. `NDArray.sum` und `WNDArray.sum` nehmen
heute nur `axis` und geben immer die achsen-*entfernte* Form zurück. Diese
Scheibe verdrahtet `keepdims` durch beide öffentlichen Surfaces auf den bereits
existierenden Typ.

## Kern-Einsicht (warum das klein und risikoarm ist)

`keepdims` ändert **ausschließlich die Shape-Metadaten**, nie die Daten und nie
einen Kernel. Die Reduktion über eine Achse liefert exakt dieselben Werte in
exakt derselben Reihenfolge, egal ob die reduzierte Achse hinterher entfernt
oder als `1` behalten wird — denn eine Achse der Länge `1` trägt nichts zur
row-major-Reihenfolge bei (`product` mit Faktor `1` unverändert, `outSize`
unverändert, `data.length` unverändert). Damit gilt:

> **keepdims-Daten ≡ non-keepdims-Daten**, byte-für-byte — schon bewiesen durch
> die bestehenden Differentialtests (unveränderte Kernel). Das einzig **Neue**
> ist die zurückgegebene **Form**.

Konsequenz: **null Rust-Änderungen, Artefakt-Hash byte-identisch** (starke
Freeze-Form, wie Kern 08). Kein neuer ABI-Einstiegspunkt.

## Binding-Entscheidungen

### D1 — API-Form: zweiter positionaler `keepdims`-Parameter

```ts
sum<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
  axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  keepdims?: KeepDims,
): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>
```

- **`const KeepDims`** ist zwingend: ohne `const` widened `sum(1, true)` das
  Argument zu `boolean`, und `ReduceAxis<S, Axis, boolean>` distribuiert
  (`boolean extends true` → `true | false`) zu einer **Union** aus keep- und
  non-keep-Form. Mit `const` wird das Literal `true`/`false` inferiert. (Genau
  die `const`-Regel aus CLAUDE.md „Key TS limits".)
- **Guard bleibt auf `ReduceAxis<S, Axis>`** (keepdims-frei): der Guard existiert
  nur, um eine bereichsüberschreitende Achse am Argument zu röten. Ob keepdims
  oder nicht ist dafür irrelevant — eine ungültige Achse ist in beiden Fällen
  identisch ungültig (`ReduceAxis<S,Axis,true>` erzeugt den `ShapeError` an
  denselben Stellen wie `…,false`). Kein Grund, `KeepDims` in den Guard zu
  ziehen.
- **Gradual-Kante (dokumentiert, nicht getestet-als-Fehler):** ein *dynamischer*
  `boolean` (Variable statt Literal) als `keepdims` liefert eine Union-Form —
  dieselbe bewusste Degradation wie eine dynamische `number`-Achse zu
  `number[]`. Ein Typtest hält das fest (siehe Testplan T-KD5).

### D2 — Shape-Berechnung: ein appended Helper `keepDimsShape` in `runtime.ts`

`runtime.ts` ist **append-only** (Kern 07/08 beide reine Insertions, git
`--stat` = 0 Deletions). Bestehende Funktionen (`sumRuntime`) werden **nicht**
angefasst. Neu angehängt:

```ts
/** Shape metadata for a keepdims reduction (NumPy `keepdims=True`): the
 * axis-reduced result of `sumRuntime`, but with the reduced axis kept as
 * size-1 instead of removed. The reduction DATA is identical either way —
 * only the reported shape differs — so callers run the normal reduce and
 * swap in this shape. `axis === undefined` ⇒ all-ones of the input rank.
 * `axis` is assumed already range-validated by the caller's reduce path. */
export function keepDimsShape(shape: readonly number[], axis: number | undefined): number[]
```

- `axis === undefined` ⇒ `shape.map(() => 1)` (Länge = Rang, `product = 1`,
  passt zum 1-Element-`data` von `sumRuntime(…,undefined)`).
- sonst normalisiert `axis` (neg. von hinten) und ersetzt genau diese Position
  durch `1`: `shape.map((d, i) => (i === normAxis ? 1 : d))`.
- **Single source of truth**: `NDArray.sum` *und* `WNDArray.sum` rufen denselben
  Helper — damit sind beide Surfaces per Konstruktion form-gleich (die Daten sind
  per Konstruktion schon bit-gleich).

### D3 — Method-Bodies: minimale, verhaltenserhaltende Erweiterung

Die `sum`-Methode auf beiden Klassen (`ndarray.ts`, `wasm/resident.ts`) ist eine
**bestehende** Methode; `keepdims` erweitert sie zwingend (die Typ-Ebene ist
genau darauf ausgelegt). Das ist eine **bewusst offengelegte Abweichung** von
der Insertion-only-Hausregel für TS-Klassenkörper, und sie ist zulässig, weil
die Regel-Begründung hier nicht greift:

1. **Kein Artefakt-Bezug.** Die Insertion-only/Freeze-Disziplin schützt die
   `#[track_caller]`-Panic-Location-Bytes im **Rust**-Artefakt. `ndarray.ts` /
   `resident.ts` sind reines TS, das kein `.wasm` erzeugt — ihre Zeilen können
   den Artefakt-Hash nicht bewegen. (Kern 07 wählte Insertion-only für TS-Bodies
   als *Intent-Analogon*, weil Kern 07 gleichzeitig ABI-Exports *hinzufügte* und
   den Freeze-Beweis sauber halten wollte. Kern 09 fügt keinen Export/Rust hinzu
   → die Kopplung existiert nicht.)
2. **Verhaltenserhaltend für alle Alt-Aufrufer.** `keepdims?` ist optional,
   Default-Verhalten = `false` = exakt das heutige Verhalten. `sum()`, `sum(1)`,
   `sum(-1)` an allen bestehenden Call-Sites bleiben unverändert (Beweis: die
   bestehenden Runtime-/Typtests müssen unverändert grün bleiben).

Der Diff pro Methode ist auf Signatur + eine Shape-Swap-Zeile begrenzt; kein
bestehender Member wird sonst berührt.

**`NDArray.sum` (ndarray.ts):**
```ts
const { shape, data } = sumRuntime(this.shape, this.data, axisNum);
const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
return new NDArray<…>(outShape as …, data);
```

**`WNDArray.sum` (resident.ts):** in beiden Zweigen (axis-undefined /
axis-gegeben) wird `keepdims ? keepDimsShape(this.shape, axisNum) : <bisher>`
als Rückgabe-Shape verwendet; `outLen` (undefined-Zweig: `1`; axis-Zweig:
`product(non-keep-outShape)`) ist unverändert, weil `product` durch die
`1`-Achse nicht wächst. Der Kernel-Call und `WNDArray.fresh(...)`-Länge bleiben
identisch.

### D4 — v1-Backend `wasmSum` (backend.ts): NICHT angefasst

`wasmSum` ist **keine öffentliche Surface** — nur Differential-Twin in Tests
(`sum.test.ts`, `negative-paths`, `backend-oom`) und `demo.ts`. keepdims ändert
keinen Kernel und keine Daten; die einzige neue Größe (die Form) wird bereits
zwischen den **zwei genuinen** öffentlichen Implementierungen (`NDArray` = naive
TS-Referenz vs. `WNDArray` = resident WASM) differentiell geprüft. `wasmSum`
keepdims-fähig zu machen würde nur `keepDimsShape` auf beiden Seiten desselben
Vergleichs duplizieren (schwacher Test) und `backend.ts` unnötig anfassen.
Bleibt frozen.

## Testplan

### Typ-Ebene — `spike/tests/ndarray.test-d.ts` (Insertion, neue `T*`)
- **T-KD1**: `s.sum(1, true)` ⇒ shape `[2, 1, 4]` (`s: NDArray<[2,3,4]>`).
- **T-KD2**: `s.sum(undefined, true)` ⇒ shape `[1, 1, 1]`.
- **T-KD3**: `s.sum(-1, true)` ⇒ shape `[2, 3, 1]`.
- **T-KD4**: `s.sum(1, false)` ⇒ shape `[2, 4]` (expliziter `false` = wie Default).
- **T-KD5** (gradual): `declare const kd: boolean; s.sum(1, kd)` ⇒ Union
  `[2,4] | [2,1,4]` — hält die bewusste Degradation fest.
- **T-KD6** (negativ, unverändert): `s.sum(3, true)` bleibt `@ts-expect-error`
  (Achse out-of-range röten am Argument, keepdims ändert das nicht).

### Typ-Ebene — resident, falls kein eigenes `.test-d.ts`: eine gezielte
Assertion im bestehenden resident-Runtime-Test genügt nicht für *Typen* — daher
mindestens ein `Expect<Equal<...>>` in `ndarray.test-d.ts` gegen den
`WNDArray`-Rückgabetyp (Import ist type-only zulässig), oder Aufnahme in
`slice.test-d.ts`-Stil. Entscheidung bei Implementierung; Minimum: NDArray-Typen
oben sind Pflicht, resident-Typ mind. eine Assertion.

### Runtime-Differential — neue Datei `spike/tests-runtime/keepdims.test.ts`
(explizit in `package.json` **`test:resident`**-Liste eintragen — sie braucht
`WNDArray`; Guard erzwingt die Listung. *Korrigiert per Revision oben, F4*):
Für zufällige Shapes/Achsen (PRNG-Muster wie `sum.test.ts`):
- **Beide Surfaces**: `NDArray.sum(axis, true)` und `WNDArray.sum(axis, true)`.
- **Shape-Assertion**: reduzierte Achse ist `1`, alle anderen Dims gleich dem
  Input; `undefined`-Achse ⇒ all-ones; Rang bleibt (≠ non-keep-Rang−1).
- **Daten-Assertion**: `assertDataBitIdentical` gegen die **non-keepdims**-Summe
  derselben Achse (macht „Daten unverändert" beobachtbar, nicht nur argumentiert)
  — und Referenz ⇄ resident untereinander bit-gleich.
- **`product`-Invariante**: `product(keepShape) === product(nonKeepShape) ===
  data.length`.
- **F5-Nachtrag (Revision, Owner-mandatiert):** keepdims auf **Views** —
  `keepDimsShape` muss die LOGISCHE View-Shape verarbeiten, nie die des
  Buffers. Drei View-Arten: transponiert (invertierte Strides), offset-
  verschobener Slice (`offset != 0`; der `undefined`-Achsen-Fall trifft dort
  `nt_sum_all_strided`), und die Komposition beider. Referenz nicht-zirkulär:
  dieselbe View-Kette auf der naiven `NDArray`, summiert OHNE keepdims.
  Zusätzlich `WNDArray` im keepdims=false-Paritätsblock (beide Surfaces).

### Bestehende Suiten: unverändert grün
`test:core` (817, unverändert — keepdims.test.ts läuft in `test:resident`),
`test:resident` (3279+2 → +Fälle), `pnpm demo` (all-agree), `cargo` (157,
unberührt), `check` (Verbund), Artefakt-Hash **identisch** (`7a65d800…`).

## Gate-Erwartung
- `pnpm check` (Verbund root+stress) clean.
- `check:diag`: Haupt-Pin verschiebt sich minimal (neue Typtests + ein
  `const KeepDims`-Parameter). Falls sich der Pin bewegt: **neu pinnen** und im
  Ergebnisdoc mit Delta ausweisen (kein Gate-Bruch — die Typtests sind
  realistische Sites, keine Digit-Stress-Fälle → bleiben im Haupt-Korpus, nicht
  in `tests-stress`). Absolute Affordability bleibt trivial (< 4 % Budget).
- Artefakt-Hash byte-identisch → per Clean-Rebuild bestätigen.

## Definition of Done
Spec (dies) → Implementierung → **Fresh-Context-Verify** (gegen diese Spec,
coverage-first, eigener Mutant) → `docs/kern-09-keepdims-ergebnisse.md`
(grounded, Honesty-Rule, Post-Verify-Addendum) → **KB-Capture** (falls
allgemeine Lektion) → Commit + Push. Alle Befehle vom Repo-Root.
