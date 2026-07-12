# Kern 10 — Spezialwerte im Differential-Generator (Binding Spec)

*Phase B, Item 6. Datum: 2026-07-12. Belegt den Bit-Identitäts-Anspruch für
IEEE-754-Spezialwertklassen (NaN, ±Inf, ±0, Subnormals), nicht nur für normale
endliche Werte.*

> **Korrektur 2026-07-12 (Implementierungsbefund, offengelegt):** Die
> ±0-Fixture unter D2 stand ursprünglich als `sum([-0,-0,…]) = -0`. Das ist
> **faktisch falsch für diese Codebase**: `sumRuntime` seedet den Akkumulator
> mit `let total = 0` (= `+0`, frozen/pre-existing), und IEEE-754
> round-to-nearest definiert `+0 + (-0) = +0` — ein reiner `-0`-Array summiert
> daher IMMER zu `+0`, `-0` ist durch diese Reduktion prinzipiell unerreichbar
> (kein Kernel-Bug, keine Freeze-Implikation). Der `deep`-Executor hat das per
> `node -e` bewiesen und die Fixture auf den WAHREN Wert (`+0`) korrigiert
> statt der falschen Spec zu folgen — exakt das gewünschte grounded-Verhalten.
> D2-Text unten entsprechend berichtigt.

## Ausgangslage / Problem

Der Anspruch des Projekts ist **bit-identische** Numerik zwischen der naiven
TS-Referenz (`spike/src/runtime.ts`) und den Rust/WASM-Kerneln (v1 copy-based +
resident strided). Belegt wird er durch Differentialtests mit einem
handgeschriebenen seeded PRNG (`spike/tests-runtime/prng.ts`, splitmix64).

**Der Generator erzeugt konstruktionsbedingt keine Spezialwerte:** `nextF64()`
liefert `sign * (intPart 0..999 + 3-Dezimal-Fraktion)` — nie NaN, nie ±Infinity,
nie ±0 (na ja, `+0` bei intPart=0 & frac=0 & sign=+, aber nie `-0`), nie
Subnormals. Damit ist Bit-Identität **nur für normale endliche Werte** belegt
plus eine Handvoll vereinzelter kuratierter Fixtures (div→Inf/NaN/-0 in
elementwise.test.ts, cosine-Nullvektor→NaN in vector.test.ts). Die
KB-Lektion [[bit-identische-differentialtests-zwischen-implementierungen]]
(Punkt 5) benennt genau diese Lücke: **injizieren oder ehrlich als ungedeckt
ausweisen.** Diese Scheibe injiziert.

## Warum das echte Bugs finden KANN (nicht nur Formalie)

- **SIMD-matmul & Subnormals:** der blocked+packed+SIMD128-matmul-Kernel (Kern
  04) ist der Hauptverdächtige. WASM-SIMD128 hat KEIN Flush-to-Zero (anders als
  manche nativen SIMD-Modi) — Subnormals MÜSSEN erhalten bleiben. Falls der
  Kernel (oder eine Toolchain-Stufe) sie doch flusht, weicht er von der
  skalaren TS-Referenz ab. Das ist der wertvollste neue Test.
- **Akkumulations-Ops & Inf/NaN:** `sum`/`dot`/`normSq`/`matmul` akkumulieren.
  `Inf + (-Inf) = NaN`, `Inf * 0 = NaN`, `MAX_VALUE * MAX_VALUE = Inf`. Bit-
  Identität hält nur, wenn die Akkumulationsreihenfolge gespiegelt ist (per
  Konstruktion, Kern 01/04) — der Test macht das an Spezialwerten beobachtbar.
- **±0-Diskriminierung:** `Object.is` unterscheidet ±0; `sum([+0,-0]) = +0`,
  `-0 * positive = -0`. Ein Kernel, der ±0 verwechselt, würde auffallen.
- **NaN-Payload / Movement:** reine Datenbewegung (transpose/reshape/slice/
  fromArray→toArray) kopiert Bytes → NaN-Payloads bleiben byte-exakt erhalten.
  Arithmetik-Ergebnisse dagegen haben in WASM implementierungsdefinierte
  NaN-Payloads (Spec erlaubt Nichtdeterminismus). Diese Asymmetrie wird ehrlich
  behandelt (D3).

## Binding-Entscheidungen

### D1 — Spezialwert-Generator, append-only in `prng.ts`

`prng.ts` wird **nur erweitert** (append-only-Disziplin wie runtime.ts;
`nextF64`/`genData`/`makeRng`/`genBroadcastShapes` bleiben BYTE-IDENTISCH, damit
kein einziger bestehender seeded Testfall seine Werte ändert). Neu angehängt:

1. **`SPECIAL_VALUES: readonly number[]`** — die Menü-Klassen, jede mind. einmal:
   `NaN`, `Infinity`, `-Infinity`, `0` (`+0`), `-0`, `Number.MIN_VALUE`
   (kleinstes positives Subnormal), `-Number.MIN_VALUE`, ein weiteres Subnormal
   (`Number.MIN_VALUE * 4` o. ä., immer noch < `Number.MIN_NORMAL`),
   `Number.MAX_VALUE`, `-Number.MAX_VALUE`. (`Number.MIN_NORMAL` gibt es nicht
   als Konstante — Grenze ist `2.2250738585072014e-308`; Subnormals liegen
   darunter.)
2. **`nextF64Special(rng, specialProb = 0.35): number`** — mit Wahrscheinlichkeit
   `specialProb` ein gleichverteilt gezogener Wert aus `SPECIAL_VALUES`, sonst
   `rng.nextF64()`. Deterministisch (nutzt ausschließlich `rng`-Draws, kein
   `Math.random`). Die Auswahl-Draws kommen aus demselben `rng` — Reproduzier-
   barkeit bleibt.
3. **`genDataSpecial(rng, shape, specialProb = 0.35): Float64Array`** — wie
   `genData`, aber pro Element `nextF64Special`.

**Nicht-Vakuität des Generators (Pflicht):** ein Test muss beweisen, dass
`genDataSpecial` über hinreichend viele Draws JEDE Klasse aus `SPECIAL_VALUES`
tatsächlich mindestens einmal produziert (inkl. `-0`, per `Object.is`
unterschieden, und mind. ein Subnormal). Sonst könnte ein Generator-Bug die
ganze Suite still vakuum machen (KB-Lektion: Generator-Abdeckung ist Beweislast,
[[property-gesetz-beweist-nur-was-der-generator-erreicht]]).

### D2 — Neue Differentialdatei `spike/tests-runtime/special-values.test.ts`

Eine kohärente neue Datei (nicht die bestehenden Differentialdateien
aufbohren — niedrigere Blast-Radius, append-only-freundlich, Präzedenz
keepdims.test.ts). In der **`test:resident`**-Liste (braucht `WNDArray`; Guard
erzwingt Listung). Für jede Op: PRNG-Muster wie sum.test.ts (eigene Seeds),
`genDataSpecial` für die Payloads, Bit-Identität via `assertDataBitIdentical`.

**Op-Abdeckung** (drei-Wege wo v1-Kernel existiert, sonst Referenz⇄resident):

| Op | Referenz | v1 (backend.ts) | resident | Besonderer Fokus |
|---|---|---|---|---|
| add | `elementwiseBinary +` | `wasmAdd` | `.add` | Inf±Inf, NaN-Propagation |
| sub | `elementwiseBinary -` | — | `.sub` | Inf−Inf=NaN |
| mul | `elementwiseBinary *` | — | `.mul` | Inf*0=NaN, MAX*MAX=Inf, ±0 |
| div | `elementwiseBinary /` | — | `.div` | x/0=±Inf, 0/0=NaN, ±0 |
| sum (all+axis) | `sumRuntime` | `wasmSum` | `.sum` | Akkumulation Inf/NaN, ±0 |
| matmul | `matmulRuntime` | `wasmMatmul` | `.matmul` | **SIMD-Subnormals**, Akkumulation |
| dot | `dotRuntime` | — | `.dot` (scalar) | Akkumulation; Skalar-Comparator |
| normSq/norm | `normSqRuntime` | — | `.norm`/scalar | MAX²→Inf; sqrt(NaN)=NaN |
| transpose | `transposeRuntime` | `wasmTranspose` | `.transpose` | **Movement: Payload-Erhalt (D3)** |

- **matmul-Größen:** ein Spread, der sowohl den skalaren als auch den
  blocked+SIMD-Pfad trifft (z. B. n ∈ {2, 3, 8, 16, 32}). Der Executor
  verifiziert/dokumentiert, welche Größe welchen Pfad nimmt (ggf. via
  Kernel-Routing-Schwelle in resident.ts). Subnormal-Erhalt ist HIER am
  wichtigsten.
- **Skalar-Ops (dot, norm, cosine):** liefern `number` — Vergleich via
  `assertScalarBitIdentical` (existiert in vector.test.ts; falls nicht
  wiederverwendbar, ein `Object.is`-Einzelvergleich mit Payload-Ehrlichkeit).

**Kuratierte Fixtures (deterministisch, zusätzlich zum Zufallspass):**
- **Subnormal-Erhalt durch SIMD-matmul (Kern-Fixture):** eine matmul, deren
  Operanden Subnormals und normale Werte enthalten und deren Ergebnisse
  Subnormals bleiben (klein genug gewählt, keine NaN/Inf-Kontamination) —
  beweist, dass der SIMD-Pfad Subnormals NICHT flusht. Referenz⇄v1⇄resident
  byte-identisch.
- **±0-Akkumulation:** `sum([-0, -0, ...]) = +0` (NICHT `-0` — siehe Korrektur
  oben: `sumRuntime`-Seed ist `+0`, `+0 + -0 = +0`, `-0` durch diese Reduktion
  unerreichbar), `sum([+0, -0]) = +0`, byte-identisch auf allen Surfaces via
  `Object.is`. Der Fixture-Claim ist Referenz⇄v1⇄resident-Übereinstimmung auf
  dem WAHREN Wert, nicht dass die Reduktion `-0` erzeugt.
- **Inf/NaN-Propagation:** `[Inf, -Inf] .add` → `[Inf, -Inf]`; `[Inf] + [-Inf]`
  → `NaN`; `MAX .mul MAX` → `Inf`.
- **Movement-Payload-Fixture (D3):** siehe D3.

### D3 — NaN-Payload-Ehrlichkeit (präziser Claim + Movement-Verschärfung)

- **`assertDataBitIdentical` bleibt `Object.is`-basiert** (payload-INSENSITIV).
  Der Doc-Kommentar des Comparators (assert-helpers.ts) wird um die NaN-Payload-
  Klarstellung ergänzt (append/inline, keine Verhaltensänderung): „`Object.is`
  behandelt jedes NaN als gleich, unabhängig vom Payload; die WASM-Spec erlaubt
  nichtdeterministische NaN-Payloads für Arithmetik-Ergebnisse — der belegte
  Claim ist daher: bit-identisch für ALLE Nicht-NaN-Werte (±0 unterschieden),
  NaN gleich als Werteklasse."
- **Movement-Verschärfung (kuratierte Fixture, KEIN neuer Random-Pass):** für
  eine reine Bewegungs-Op (transpose gewählt, hat v1+resident-Kernel) beweist
  EINE Fixture den STÄRKEREN Claim — Payload byte-exakt erhalten: ein NaN mit
  NICHT-kanonischem Payload (via `BigUint64Array` konstruiert, z. B.
  `0x7ff8_0000_dead_beefn`) wird eingespeist, das Bitmuster des transponierten
  Ergebnisses per `bitsOf`-Vergleich (nicht `Object.is`) gegen die Referenz
  geprüft. Falls das auf dieser Engine/Build fehlschlägt (Load/Store
  kanonisiert wider Erwarten), ist das ein FUND (dokumentieren, nicht
  verstecken) — nicht per se ein Bug, aber eine Claim-Grenze. Erwartung: hält
  (WASM f64.load/store kanonisieren nicht).

### D4 — Erwartung: NULL Rust-Änderungen, Artefakt-Hash byte-identisch

Diese Scheibe ist test-only. Die Kernel handhaben IEEE-754 bereits; wir TESTEN
sie nur mit neuen Inputs. Erwartung: `cargo` unverändert 157, Artefakt-Hash
byte-identisch `7a65d800…`. **Falls ein Differentialtest eine echte Divergenz
aufdeckt** (z. B. SIMD flusht Subnormals, oder ein Kernel behandelt ±0/NaN
anders als die Referenz): das ist ein ESKALATIONS-FUND, kein stiller Patch —
reproduzieren, Root Cause beweisen, dem Owner vorlegen (eine Kernel-Änderung
berührt den Freeze und ist eine eigene Entscheidung). Der Test darf NICHT
abgeschwächt werden, um Grün zu erzwingen.

### D5 — Scope: transzendentenfrei

Nur die IEEE-exakte Whitelist (`+ - * / sqrt`). Overflow-zu-Inf via `*`/matmul
von `MAX_VALUE` ist IN Scope (weiterhin IEEE-exakt). Transzendente Ops
(exp/sin/…) existieren nicht und bleiben out of scope (dokumentierte
Determinismus-Entscheidung — libm-Differenzen brechen Bit-Parität). Keine neuen
Produkt-Ops. Keine Änderung an `nextF64`/bestehenden Generatoren.

## Testplan (Zusammenfassung)

1. **Generator-Nicht-Vakuität** (D1): jede `SPECIAL_VALUES`-Klasse erscheint;
   `-0` und Subnormal per `Object.is`/Bitmuster bestätigt.
2. **Zufallspass pro Op** (D2): N Fälle je Op mit `genDataSpecial`,
   Bit-Identität Referenz⇄v1⇄resident (bzw. Referenz⇄resident).
3. **Kuratierte Fixtures** (D2/D3): Subnormal-SIMD-matmul, ±0-Akkumulation,
   Inf/NaN-Propagation, Movement-Payload-Erhalt.
4. **Bestehende Suiten unverändert grün**; Artefakt-Hash identisch.

## Gate-Erwartung

- `pnpm check` (Verbund) clean.
- `pnpm test:core` 817 unverändert; `pnpm test:resident` 3646+2 → +Fälle.
- `cargo` 157 unverändert; Artefakt-Hash `7a65d800…` byte-identisch.
- `check:diag` (Haupt 174.213 / Stress 94.597): eine test-only `.ts`-Datei ohne
  neue Typmaschinerie bewegt den Pin höchstens minimal (die Datei ist reiner
  Wertcode, keine generischen Aliase) — bei Bewegung neu pinnen + Delta
  ausweisen. Messung NUR nach der Mess-Hausregel (frischer Zustand, Exit-Code,
  Files-Count-Beweis).

## Definition of Done

Spec (dies) → Implementierung (`deep`) → **Zwei-Verifier-Runde** (Spec +
adversarial, docs/verify-runde-template.md) → docs/kern-10-special-values-
ergebnisse.md (grounded, Honesty-Rule, beide Verdikte im Addendum) → KB-Upsert
(die bit-identische-Differentialtests-Notiz revidieren: Punkt 5 „injiziert",
Beleg Kern 10) → Commit + Push. Alle Befehle vom Repo-Root.
