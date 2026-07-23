# WASM-Parität S0 — `sqrt` auf `WNDArray`/threaded (bindende Spec)

Status: **bindend** (Owner-Richtungsabnahme 2026-07-23: WASM-Backend-Parität der W1–W5-Ops
nachziehen; `sqrt` ist der Pilot / die dünne vertikale Scheibe, die die komplette „neuer
Kernel"-Pipeline etabliert, bevor die aufwändigeren Ops folgen).
Version: 3 · Datum: 2026-07-23 · Eskalationsleiter: **Stufe 3** (substanzielle Scheibe, M1- UND
M4-Anker berührt — voller Verify-Katalog A+B+C). Covenant: v5. **v3:** Owner-bestätigte
Scope-Erweiterung (D10) — der `Omit<ThreadedCoreExports, "memory">`-Cast in threaded.ts wird durch
einen direkten Cast ersetzt (laufzeit-identisch); das beseitigt eine während der Umsetzung entdeckte,
gemessene `keyof`-getriebene Instantiation-Ripple (+7 pro `CoreExports`-Member) AN DER WURZEL — für
`sqrt` UND jede Folge-Scheibe. Statt Δ0 auf bench:editor/stress verzeichnet der Slice jetzt ein
uniformes −159 (Details in D10 + D7). **v2:** Baustein-0-Befunde
eingearbeitet (C1: D7-Budgetbegründung korrigiert; C2: `genDataSpecial`-Raster-Block in D6 ergänzt;
M-1: D4-Merge-Block präzisiert; N2 in D3) — Details im Addendum am Ende. Zentraler Gewinn: Baustein 0
hat die M1-Bitparität `f64::sqrt` ↔ `Math.sqrt` VOR dem Bau EMPIRISCH bewiesen (30.028 Fälle,
0 Mismatches), nicht mehr nur a priori plausibel.
Roadmap: erste Scheibe der WASM-Parität-Serie (S0 von S0–S5: sqrt → Skalar-Overloads → mean →
item/stack → argmax → topk).

## Ziel & Warum — und was diese Scheibe NICHT ist

`NDArray.sqrt()` (W3, ndarray.ts) existiert heute NUR auf der naiven JS-Klasse — `WNDArray` (der
WASM-residente Zwilling) und der threaded-Pfad kennen die Op nicht (Surface-Asymmetrie, in W3s
Doc-Kommentar und FOLLOWUPS:73 offengelegt). Diese Scheibe schließt die Lücke für `sqrt`: ein
Rust/WASM-Kernel plus die `WNDArray.sqrt()`-Methode, sodass residente Daten in-WASM
wurzelgezogen werden, ohne pro Op nach JS zu kopieren.

**Warum sqrt zuerst (Pilot-Begründung):** `sqrt` ist die einfachste der sieben Paritäts-Ops —
niladisch (kein Argument, kein Guard), shape-erhaltend, elementweise-unär, und IEEE-754-korrekt
gerundet (also a-priori bitparitätsfähig, anders als transzendente `Math.*`). Sie etabliert die
komplette Pipeline (neues Kernel-File → ABI-Eintrag → `CoreExports` → `WNDArray`-Methode →
M1-Differentialtest → Freeze-Re-Pin → threaded-Parität) bei minimaler Typ-Komplexität. Jede
folgende Paritäts-Scheibe wiederverwendet dieses Muster.

**Ehrlichkeits-Rahmen (Owner-Vorgabe, wie bei W3 und der topk-Selektion):** Diese Scheibe hat
**keinen gemessenen Nutzerbedarf** — niemand ist auf eine WASM-`sqrt`-Wand gestoßen. Es ist
Vollständigkeits-/Symmetrie-Arbeit (beide Motoren sollen dasselbe können). Das Ergebnisse-Doc
darf keinen Nutzerbedarf suggerieren, den es nicht gibt. „Wir lassen die Op JS-only" bliebe ein
zulässiger Zustand; die Scheibe existiert, weil der Owner die Parität als OSS-Vollständigkeitsziel
entschieden hat, nicht weil ein Ruckler gemeldet wurde.

Diese Scheibe ist **kein** Threaded-Pool-Kernel (der Pool routet weiterhin NUR matmul — sqrt
läuft auf dem residenten Core, im stable WIE im threaded Artefakt, da beide dasselbe Crate sind),
**keine** Änderung an `NDArray.sqrt`/`sqrtRuntime` (bereits fertig, bleibt die Referenz),
**keine** andere Op (Skalar/mean/item/stack/argmax/topk folgen als eigene Scheiben).

## Berührte Covenant-Invarianten

- **M1 (Anker `crates/core/src/`, `spike/src/runtime.ts` — COVENANT.md:11-17): BINDET JETZT.** Der
  v5-Zusatz „M1 bindet in dem Moment, in dem ein Kernel für die Op entsteht" greift hier: diese
  Scheibe ERZEUGT den `sqrt`-Kernel, also wird Bit-Identität zu `sqrtRuntime` von einer eigenen
  Scheiben-Anforderung zur harten Covenant-Pflicht. Der Kernel muss bit-identisch zur naiven
  Referenz sein, auch für IEEE-Spezialwerte, unter dem Bit-Identity-Law (keine Vektorisierung
  außer QUER zu Output-Elementen — bei einer elementweisen Op trivial erfüllt; kein FMA, kein
  relaxed-simd). `f64::sqrt` ist wie `Math.sqrt` die IEEE-754-korrekt-gerundete Wurzel (ECMA-262
  `sec-math.sqrt` bzw. IEEE 754 §5.4.1) — Bit-Parität ist a priori plausibel und wird empirisch
  im M1-Differentialtest (D6) BEWIESEN, nicht angenommen. Die FOLLOWUPS:73-Paritätslücke für
  `sqrt` schließt mit dieser Scheibe.
- **M4 (Anker `abi.rs`/`matmul_blocked.rs`/`shape.rs`): berührt (abi.rs), append-only gewahrt.**
  Der `check:freeze`-Artefakt-Hash ÄNDERT sich legitim (neuer Kernel im Binary — Kern-07/11-
  Präzedenz: eine Scheibe, die echte Exports/Kernels HINZUFÜGT, ändert den Hash). Der Freeze-Beweis
  dekomponiert (D7): (1) Pre-Edit-Clean-Rebuild reproduziert den alten Hash `0b9df4f1…` exakt;
  (2) der neue Code ist rein ADDITIV — neues File `kernels/sqrt.rs`, `nt_sqrt_strided` strikt ans
  ENDE von `abi.rs` angehängt, `pub mod sqrt;` ans Ende von `kernels/mod.rs`, ZERO Edits an
  bestehenden Funktionskörpern/Zeilenpositionen (kein Zeilenshift in `abi.rs`/`matmul_blocked.rs`/
  `shape.rs`, also keine `#[track_caller]`-Panic-Location-Verschiebung); (3) behaviorale Pins grün;
  (4) Mutations-Beweis; (5) neuer Hash wird der Pin. `matmul_blocked.rs`/`shape.rs` byte-unberührt.
- **M5 (Anker `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`): threaded.ts in D10 TYP-ONLY
  berührt, M5-Eigenschaft unberührt.** Kein eager `node:*`-Import (der D10-Cast `Omit<…>` → direkter
  Cast ändert keinen Import und keinen Laufzeit-Pfad); der Kernel ist WASM, `WNDArray.sqrt` reine
  residente-Memory-Arbeit, der Default-`NDArray`-Pfad bleibt unangefasst. Threaded bleibt
  Node-only-Opt-in hinter `backend("threaded")`. Baustein C bestätigt, dass der laufzeit-identische
  Cast die Browser-Sicherheit nicht antastet.
- **Z1 (Anker `package.json`): unberührt.** Der Kernel ist from-scratch Rust; keine neue
  Abhängigkeit, kein `dependencies`-Feld.
- **Z2 (Anker `package.json`): unberührt.** Alle neuen Tests hängen an BESTEHENDE Korpus-Dateien
  an (`elementwise.test.ts`/`special-values.test.ts` in `test:resident`, `threaded.test.ts` in
  `test:threaded`, Rust-Tests im neuen Kernel-File in `cargo test`) — keine neue Datei, keine
  Explizitlisten-Änderung in package.json, `test-scripts-guard` bleibt grün.
- **M2/M3: unberührt.** `WNDArray.sqrt` ist niladisch und guard-los (kein `Guard`/`OkShape`,
  keine Fehler-Message, keine Shape-Typ-Maschinerie berührt) — exakt wie `NDArray.sqrt(): NDArray<S>`.

## Die tragende Beobachtung: sqrt ist bitparitätsfähig, weil es KEIN transzendentes Op ist

`sqrt` ist von NumTypes Transzendenten-Nicht-Ziel („Keine transzendenten Ops ohne eigene
Determinismus-Entscheidung — brechen Bit-Parität", COVENANT.md) ausgenommen, weil sowohl
ECMA-262 (`Math.sqrt`) als auch IEEE 754 (`f64::sqrt`) die **korrekt gerundete** Wurzel verlangen
— es gibt genau EIN zulässiges Ergebnis pro Eingabe, unabhängig von der Implementierung. Damit
MÜSSEN `Math.sqrt(x)` und `f64::sqrt(x)` für jedes endliche `x` bit-identisch sein (dasselbe
Argument, das `NDArray.sqrt`s Ausnahme vom Nicht-Ziel in W3 trägt und das `norm()` bereits nutzt).
Für die IEEE-Spezialkanten ist das Verhalten ebenfalls normiert und in beiden Sprachen gleich:
`sqrt(+0)=+0`, `sqrt(-0)=-0`, `sqrt(+Inf)=+Inf`, `sqrt(x<0)=NaN`, `sqrt(NaN)=NaN`. Einzige
zulässige Nicht-Bit-Gleichheit: die genaue NaN-**Payload** eines `sqrt(NaN)`/`sqrt(neg)` — M1s
eigener Vorbehalt erlaubt implementierungsdefinierte NaN-Payloads für ARITHMETIK-Ergebnisse (und
`sqrt` ist Arithmetik); der M1-Test prüft NaN deshalb auf Wert-KLASSE (`Number.isNaN`/`is_nan`),
finite Werte und `±0`/`±Inf` auf exakte Bits — exakt die Disziplin der bestehenden `div`-Kernel-
Tests (`assertDataBitIdentical`/`to_bits`, kernels/elementwise.rs). Diese Beobachtung ist das
Fundament; **Baustein 0 hat sie VOR dem Bau empirisch bewiesen** (30.028 Fälle inkl. aller
Spezialkanten + volles Exponentenspektrum, wasm32-Zielbuild mit UND ohne `+simd128`, gegen dieselbe
V8-Engine — 0 Mismatches, 0 unerlaubte NaN-Payload-Abweichungen; Addendum unten), und D6 bestätigt
sie am tatsächlichen Kernel erneut.

---

## Design-Entscheidungen

- **D1 — Scope.** Genau zwei neue Bausteine plus ihre Verdrahtung: (a) der Rust-Kernel
  `sqrt_strided` (+ privater `unary_strided`-Kern) in einem NEUEN File `crates/core/src/kernels/
  sqrt.rs`; (b) die `WNDArray.sqrt(): WNDArray<S>`-Methode in `spike/src/wasm/resident.ts`.
  Verdrahtung: `nt_sqrt_strided`-ABI-Eintrag (abi.rs), `CoreExports`-Deklaration (loader.ts),
  `pub mod sqrt;` (kernels/mod.rs). Threaded-Parität ist AUTOMATISCH (dasselbe Crate → der Kernel
  ist in beiden Artefakten; `WNDArray.sqrt` läuft auf jedem Core) und wird NUR getestet, nicht
  eigens verdrahtet (D6). Kein Byte an `NDArray`/`sqrtRuntime`/`vector.ts`/`reduce.ts`/`dim.ts`.

- **D2 — Rust-Kernel (`kernels/sqrt.rs`, neues File — freeze-sauber).** Ein privater generischer
  Kern `unary_strided<F: Fn(f64) -> f64>`, STRUKTURELL das unäre Gegenstück zu
  `binary_strided` (kernels/elementwise.rs) — dieselbe Validierungs-Reihenfolge, dieselbe
  Iterations-Ordnung, dieselbe Offset-Algebra, nur EIN Operand:
  1. `checked_element_count(shape)?`; `validate_strided_bounds(shape, strides, offset, data.len())?`.
  2. Output-Shape = Input-Shape (shape-erhaltend, kein Broadcast); `size = checked_element_count(shape)`.
  3. Contiguous fast path (Kern-11-Muster, nach ALLEN Validierungen): `offset == 0 && strides ==
     compute_strides(shape)` → `for i in 0..n { out[i] = op(data[i]) }`.
  4. General path (Views): `unravel` + `aligned_effective_strides`, `out[flat] = op(data[eff_off])`.
  `pub fn sqrt_strided(shape, strides, offset, data) = unary_strided(..., |x| x.sqrt())`. `f64::sqrt`
  ist die IEEE-korrekt-gerundete Wurzel (M1). Cargo-Tests im File-eigenen `#[cfg(test)] mod tests`
  nach dem Muster von elementwise.rs: same-shape, transposed view, offset window, size-0-Array,
  rank-too-large, contiguous-fast-path-vs-general-path-Äquivalenz (transponierte View), und
  Spezialwerte (`sqrt(-0).to_bits() == (-0.0).to_bits()`, `sqrt(4.0)==2.0`, `sqrt(-1.0).is_nan()`,
  `sqrt(INFINITY)==INFINITY`, `sqrt(NAN).is_nan()`, ein Subnormal).

- **D3 — ABI-Eintrag (`nt_sqrt_strided`, strikt ans ENDE von abi.rs angehängt — M4 append-only).**
  Signatur = die UNÄRE Teilmenge von `nt_add_strided`s Konvention (identisch zur bereits
  existierenden 8-Parameter-Form von `nt_materialize`, loader.ts:111-120 — das nächste
  Parameterform-Vorbild, N2):
  `nt_sqrt_strided(shape_ptr, rank, strides_ptr, offset, data_ptr, data_len, out_ptr, out_len) ->
  i32` (Status). Liest die Slices via `read_slice`/`validate_rank` (dieselben Helfer wie
  `nt_add_strided`), ruft `sqrt_strided`, schreibt das Ergebnis in `out`, gibt `status_of`/
  `first_error` zurück — exakt die Struktur der bestehenden strided-Einträge. Kein Zeilenshift an
  bestehendem abi.rs-Inhalt.

- **D4 — `CoreExports` (loader.ts).** `nt_sqrt_strided` wird als **NEUER, dritter
  `export interface CoreExports {}`-Merge-Block am echten Dateiende** deklariert (TS-Interface-Merging;
  **v2-Präzisierung, Baustein-0-Befund M-1:** NICHT als Edit in den bestehenden Kern-07-Block Z. 182
  ff. — die Datei-Konvention, Kommentar Z. 175-181, verlangt „ohne eine einzige vorbestehende Zeile
  anzufassen", und der zweite Block wurde in Kern 07 selbst so ans Ende gehängt). Unäre
  Argument-Konvention. `ThreadedCoreExports extends CoreExports` (threaded.ts:165) erbt die
  Deklaration automatisch; der Loader spreadet die echten Instance-Exports (threaded.ts:413), es gibt
  KEINE manuelle Namens-Allowlist, die den neuen Export ignorieren könnte.

- **D5 — `WNDArray.sqrt(): WNDArray<S>` (resident.ts, Klassenkörper-Insertion).** Unäre Methode,
  die `add`s Mechanik spiegelt, aber mit EINEM Operanden und `outShape === this.shape`:
  `assertLive("sqrt")` → `outLen = product(this.shape)` → Scratch-Liste: `writeU32Array(shape)`,
  `writeU32Array(strides)` → `allocBytes(outLen*8)` → `nt_sqrt_strided(shapeBuf.ptr, rank,
  stridesBuf.ptr, this.offset, this.buf.ptr, this.buf.lenElems, outDataBuf.ptr, outLen)` → bei
  `status !== 0`: `freeBuf(outDataBuf)` + `throw` (Message-Stamm analog `add`s
  `wasm resident nt_sqrt_strided: status ${status} for shape [...]`) → `WNDArray.fresh(core,
  this.shape, outDataBuf.ptr, outLen)` → `finally { for (buf of scratch) freeBuf(...) }`. Niladisch,
  guard-los: Rückgabetyp `WNDArray<S>` (zweite Call-Site der bereits shape-generischen niladischen
  Form von `NDArray.sqrt(): NDArray<S>` — KEINE neue Typ-Maschinerie). **Disziplin:** resident.ts
  ist ein TS-Klassenkörper mit privatem Konstruktor → INSERTION-ONLY (neuer Member, zero Edits an
  bestehenden Membern), NICHT append-only-Datei-Disziplin. Platzierung des Members ist frei, solange
  kein bestehender Member editiert wird; empfohlen bei den anderen elementweisen Ops.

- **D6 — M1-Differentialtest + threaded-Parität + Mutant.**
  - **Residenter M1-Test (an `spike/tests-runtime/elementwise.test.ts` angehängt — bestehende
    Datei, keine Listenänderung):** `WNDArray.sqrt()` gegen `sqrtRuntime` (bzw. `NDArray.sqrt`) über
    (i) contiguous, (ii) transponierte View, (iii) geslicte View, (iv) Offset-Fenster, (v) rank-0,
    (vi) size-0-Dim. Referenz für Views: `sqrtRuntime` über die logisch-geordneten Daten der View
    (unabhängig via `.toArray()` VOR dem sqrt bzw. `sliceRuntime`/`transposeRuntime` gewonnen —
    dieselbe Technik wie die bestehenden strided-View-Tests). Bit-Vergleich: `assertDataBitIdentical`
    (Object.is — finite/`±0`/`±Inf` exakt, NaN als Klasse). **Pflicht:** mindestens ein VIEW-Fall
    (beweist den strided general path).
  - **Spezialwerte (an `special-values.test.ts` angehängt) — ZWEI Teile:**
    (a) **Kuratierte Fixtures:** `sqrt(-0)===-0` (Object.is), `sqrt(neg)` → NaN, `sqrt(NaN)` → NaN,
    `sqrt(+Inf)`, `sqrt(+0)`, ein Subnormal, `MAX_VALUE` — jeweils WNDArray-vs-`sqrtRuntime`
    bit-identisch (NaN als Klasse).
    (b) **v2-Ergänzung (Baustein-0-Befund C2): randomisierter `genDataSpecial`-Raster-Block** nach der
    Datei-eigenen Konvention (Header Z. 8-9: „injects SPECIAL_VALUES into every op that has a resident
    kernel"; strukturell am nächsten der `transpose`-Block Z. 627-655): ca. 60 Fälle,
    `genShape(rng, 0, 4)` (Rang 0-4) + `genDataSpecial`, WNDArray.sqrt-vs-`sqrtRuntime` bit-identisch.
    Unär → nur EIN `genDataSpecial`-Aufruf, kein Broadcast-Pairing. **KEIN** byte-exakter-NaN-Payload-
    Fixture (anders als `transpose` Z. 657-673): `sqrt` IST Arithmetik, M1s Vorbehalt erlaubt hier
    Payload-Abweichung — NaN bleibt Klassen-Gleichheit.
  - **Threaded-Parität (an `spike/tests-runtime/threaded.test.ts` angehängt):** `WNDArray.sqrt` auf
    dem THREADED Core bit-identisch zum STABLE Core (das etablierte threaded-vs-stable-Differential
    dieser Datei um `sqrt` erweitern; deckt die automatische Parität ab). Mindestens ein View-Fall.
  - **Pflicht-Mutant (T5, während Baustein A):** eine gezielte Kernel-Mutation, die den
    M1-Differentialtest kippt (Kandidat: `|x| x.sqrt()` → `|x| x` oder `|x| -x.sqrt()`; oder die
    fast-path-Bedingung invertiert). Nachweislich gefangen (benannte Assertion), Revert per
    Backup-Kopie (NIE `git checkout` auf uncommittete Arbeit), `git status`-Beweis.

- **D7 — Pins, Budget, Freeze.**
  - **`check:freeze`-Hash RE-PINS.** Freeze-Beweis vollständig (M4 oben): Pre-Edit-Clean-Rebuild ==
    `0b9df4f1…`, additive-only-Diff, behaviorale Pins grün, Mutations-Beweis, neuer Hash = Pin
    (plattform-gelabelt, `scripts/check-freeze-hash.mjs`). BEIDE Artefakte re-pinnen (stable +
    threads); `test:threaded` beweist ihre bit-Identität zueinander weiter.
  - **`check:diag`:** `WNDArray.sqrt` fügt einen niladischen, guard-losen Member hinzu — zweite
    Call-Site einer bestehenden Typ-Form, KEINE neue Maschinerie. Erwartung: kleiner echter
    Typkosten-Δ, KEIN Klassen-Surface-Ripple (niladische Member rippeln nicht, W-Serie-Befund).
    **v2-Korrektur (Baustein-0-Befund C1):** Das neue Rust-File ist nicht im tsc-Korpus, ABER die
    Test-Anhänge (`elementwise.test.ts`/`special-values.test.ts`/`threaded.test.ts` unter
    `spike/tests-runtime/`) SIND es — `tsconfig.json` includet `spike` und excludet nur
    stress/browser/consumer/scale-workloads, NICHT `tests-runtime`; `check:diag` läuft ohne `-p` gegen
    genau diese tsconfig. Ihr Testinhalt trägt reale Typkosten bei (exakt der Mechanismus, den die
    topk-Umsetzung dokumentierte: „Δ+53 … plus der Testdatei-Erweiterung"). Es entsteht KEIN
    Order-Noise (keine neue Datei, Dateiset unverändert), der gemessene Δ ist also rein Typkosten aus
    `WNDArray.sqrt` PLUS dem neuen Testinhalt. Absolut-Gate: **≤ +1.000** auf diesem GESAMTEN
    dekomponierten Δ (nicht nur auf dem Methodenteil). **v3-Ist-Werte (nach dem D10-Omit-Fix):**
    check:diag Root **206.850 @ 140** (Δ−4 gg. 206.854 — die Omit-Ersparnis überwiegt die
    sqrt-Typkosten leicht), `check:diag:stress` **106.239 @ 82** (Δ−159), `check:diag:browser`
    **2.142 @ 75** (Δ0 — der Browser-Korpus kompiliert threaded.ts nicht, M5). `bench:editor`
    (8 Pins) **uniform −159**, neu gesetzt auf `{w1 27745, w2 29554, w3 60694, w4 27908, w5 33199,
    w6 34369, w7 26917, w8 34784}`, Hard-Gate PASS. Ohne den D10-Fix wäre es ein uniformes +7
    gewesen (die ursprüngliche Δ0-Erwartung übersah die `keyof`-Ripple; s. D10).
  - **Test-/cargo-Zahlen:** `test:resident`, `test:threaded`, `cargo` steigen um die neuen Fälle;
    exakte Deltas ins Ergebnisse-Doc.
  - Gestufte Messung: Baseline im frischen Worktree, dann Δ.

- **D8 — Doc-Platzierung.** Ergebnisse-Doc `docs/wasm-parity-sqrt-ergebnisse.md` (volles Narrativ +
  Post-Verification-Addendum). FOLLOWUPS:73 W3-Nachtrag: sqrt-WASM-Parität ERLEDIGT, M1 bindet
  jetzt, Kernel `nt_sqrt_strided` gespiegelt. `docs/roadmap.md` Post-Roadmap: neue
  „WASM-Parität"-Sektion mit S0 erledigt. CLAUDE.md „Status" + „Pins & Gates": neuer Artefakt-Hash,
  test:resident/threaded/cargo-Zahlen, WASM-Parität-Serie gestartet (nur Einzeiler + IST-Zahlen).

- **D9 — Sprache.** Code/Kommentare/Tests/Commit-Message: Englisch (Hard Constraint). Spec +
  Ergebnisse-Doc: Deutsch. „ca." statt Tilde, keine Strikethroughs (GFM-Gate, harte Arbeitsregel 4).

- **D10 — `Omit`-Fix in threaded.ts (Scope-Erweiterung, owner-bestätigt 2026-07-23, v3).** Während
  der Umsetzung fiel bench:editor mit uniformem +7 (statt der in v1/v2 erwarteten Δ0). Eine
  Mechanismus-Untersuchung (isolierter Worktree, `tsc --generateTrace`, kontrollierte Varianten —
  Rohdaten im Post-Verification-Addendum) hat es an EINE Zeile zurückgeführt: threaded.ts:413,
  `const core = { ...(instance.exports as unknown as Omit<ThreadedCoreExports, "memory">), memory }`.
  `Omit<T,K> = Pick<T, Exclude<keyof T, K>>` löst `keyof ThreadedCoreExports` an dieser Stelle frisch
  auf; jedes zusätzliche `CoreExports`-Mitglied fügt der Union einen Zweig hinzu, den `Exclude`
  (distributiv, eine Instanziierung/Zweig) und `Pick` (homomorpher Mapped-Type, eine Auflösung/Key)
  neu durchlaufen → **+7 Instantiations pro Member, in JEDER Kompilation, fix (arity-unabhängig,
  kumulativ: 2 Member = +14).** Über die WASM-Parität-Kampagne (~6 Ops) wären das ca. +42 allein
  aus diesem Mechanismus. Es ist der **vierte, jetzt benannte Kostenmechanismus** neben Order-Noise /
  Klassen-Surface-Ripple / echten Typkosten: **`keyof`-getriebene Generic-Alias-Neuauflösung.**
  **Fix (gemessen sicher):** `Omit<ThreadedCoreExports, "memory">` → direkter Cast auf
  `ThreadedCoreExports`. LAUFZEIT-IDENTISCH (beide sind `as unknown as` — reine Typ-Assertion; das
  trailing `memory` im Objekt-Literal überschreibt ohnehin jedes gespreadete `memory`), typ-legal,
  und beseitigt die +7 pro Member VOLLSTÄNDIG (für sqrt UND jede Folge-Scheibe) plus einen stehenden
  Omit-Fixkostenblock (−159/Workload). Ein Begründungs-Kommentar an der Zeile verhindert einen
  „Hygiene"-Rückbau. **Covenant:** threaded.ts ist ein M5-Anker (Browser-Sicherheit); der Cast ist
  typ-only, ändert KEINEN eager `node:*`-Import und lässt Threads Node-only-Opt-in — M5 unberührt,
  von Baustein C abzudecken. Kein Rust, kein wasm-Rebuild → `check:freeze` unberührt. test:threaded
  beweist die Laufzeit-Gleichheit (der threaded Core verhält sich identisch).

## Akzeptanzkriterien

- **T1:** `kernels/sqrt.rs` liefert `sqrt_strided` (+ `unary_strided`) mit dem contiguous-fast-path/
  strided-general-path-Aufbau; alle File-eigenen cargo-Tests grün, inkl. der Fast-vs-General-
  Äquivalenz auf einer View und der Spezialwert-Kanten.
- **T2:** `nt_sqrt_strided` ans Ende von abi.rs angehängt, `CoreExports` erweitert, `pub mod sqrt;`
  angehängt — kein Zeilenshift an bestehendem abi.rs/matmul_blocked.rs/shape.rs-Inhalt.
- **T3:** `WNDArray.sqrt(): WNDArray<S>` als Klassenkörper-Insertion, Mechanik gespiegelt von `add`
  (validate-before-alloc, Scratch-`finally`, fresh output, free-on-fail); Signatur niladisch/
  guard-los, keine bestehende Member editiert.
- **T4 (M1):** Residenter Differentialtest deckt contiguous + mindestens eine View + alle
  Spezialwert-Kanten ab, bit-identisch (NaN als Klasse); threaded-vs-stable-Parität für sqrt grün.
- **T5 (Mutant):** mindestens ein Kernel-Mutant nachweislich vom M1-Test gefangen, danach revertiert
  (Backup-Kopie, `git status`-Beweis).
- **T6 (Freeze):** `check:freeze` re-gepinnt mit vollständigem Dekompositions-Beweis (Pre-Edit-Hash
  reproduziert, additive-only, neuer Hash); beide Artefakte.
- **T7 (Gates/Pins):** Gate-Block grün; `check:diag`-Δ dekomponiert + gegatet; die Pin-Bewegungen aus
  dem D10-Omit-Fix (bench:editor/stress −159, Root −4, browser Δ0) gemessen, neu gesetzt und im
  Ergebnisse-Doc dekomponiert (nicht mehr die ursprüngliche Δ0-Erwartung — s. D10).
- **T8 (Docs):** Doc-Platzierung (D8) vollständig.

## Nicht-Ziele

Kein Threaded-Pool-Kernel für sqrt (der Pool bleibt matmul-only; sqrt läuft auf dem residenten
Core in beiden Artefakten), keine Änderung an `NDArray.sqrt`/`sqrtRuntime`/`NDArrayView`, keine
andere Paritäts-Op (Skalar/mean/item/stack/argmax/topk sind eigene Scheiben), keine Änderung an
`sqrt`s öffentlicher Signatur/Semantik auf der NDArray-Seite, keine Behauptung eines gemessenen
Nutzerbedarfs, kein Browser-Port des Threads-Pfads.

## Gate-Block / Definition of Done

`pnpm check` (Dreier-Verbund) · `check:diag`(+stress/browser, Pin-Protokoll D7) · `pnpm test:core` ·
`pnpm test:resident` (inkl. neuer sqrt-M1-Tests) · `pnpm test:threaded` (baut beide Artefakte,
sqrt-Parität) · `cargo test --manifest-path crates/core/Cargo.toml` (inkl. neuer sqrt-Kernel-Tests) ·
`pnpm check:freeze` (NEUER Hash-Pin, beide Artefakte) · `pnpm bench:editor` (8 Pins Δ0) ·
`pnpm test:example` (unberührt) · `graph-a-lama query lint` · GFM-Gate auf allen neuen/geänderten
`.md`.

## Verify-Plan (Stufe 3)

**Baustein 0 (vor dem Bau, gegen DIESE Spec, adversarial — `brainroute:deep`):** Code-Annahmen der
Spec am echten Code prüfen (existiert `unary_strided` NICHT und ist `binary_strided` wirklich das
richtige Vorbild? Ist die abi.rs-Einreihung wirklich append-only-verträglich? Ist `CoreExports` in
loader.ts der einzige Deklarationsort, oder gibt es einen zweiten, der driften würde? Erbt
`ThreadedCoreExports` wirklich automatisch? Ist `outShape === this.shape` für Views korrekt, oder
braucht der general path etwas, das die Spec übersieht?); M1-Bindung + Freeze-Behauptung härten
(ändert der neue Kernel WIRKLICH nur additiv, oder zwingt irgendein `mod`/`use` einen Zeilenshift in
einer frozen Datei?); Testplan-Lücken (fehlt eine Spezialwert-Kante? deckt der threaded-Test die
Parität nicht-vakuär ab?); Covenant-Abgleich (M1 korrekt als bindend eingeordnet? M4-Freeze-Beweis
tragfähig?). Befunde mergen, Design-Blocker mit dem Owner in die Spec einarbeiten, „Adversariale
Spec-Verifikation (Addendum)" hier anhängen, DANN Implementierung.

**Nach der Implementierung:** voller Katalog — **A** (Spec-Konformität pro D, alle Gates frisch,
eigener Mutant), **B** (adversarial: Grenzfälle jenseits der Spec — sqrt auf mehrfach-komponierten
Views, ausschließlich-Spezialwert-Buffer, size-0/rank-0, breite Kernel-Mutanten; Mess-/Freeze-
Randbedingungen angreifen), **C** (`covenant:covenant-verify`: M1 jetzt bindend korrekt? M4-Hash-Re-Pin
sauber? M5/Z1/Z2 unberührt?) — parallel, isoliert (mutierende Verifier je eigener Worktree +
Slice-Patch, read-only C im Haupt-Baum). Aufträge aus docs/verify-runde-template.md. Ergebnisse-Doc
mit Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-23)

Verifier: `brainroute:deep`, adversarial gegen v1 dieser Spec, read-only im Haupt-Baum, empirische
Probe in eigenem Scratch. **Verdikt: kein Blocker gegen den Bau.** Alle geprüften Code-Annahmen der
Spec HALTEN (an Datei:Zeile verankert): `binary_strided` (elementwise.rs:21-85) ist das richtige
Vorbild und es existiert kein unary/sqrt-Kernel; `CoreExports`-Interface-Merging ist real
(loader.ts:16 + 182, Merge-Kommentar 175-181); `ThreadedCoreExports` erbt automatisch (der Loader
spreadet die echten Instance-Exports, threaded.ts:413, keine Namens-Allowlist); `WNDArray.fresh`/
`this.strides`/`offset`/`buf` existieren wie behauptet (resident.ts:304-333, 397-400); der general
path liest Views korrekt in logischer Ordnung, `outShape === this.shape` ist richtig; `sqrtRuntime`
(runtime.ts:911-915) hat die behauptete Signatur/Semantik; die threaded-Paritätstechnik
(`makeOperand(pool.core, …)`) ist im Ziel-File bereits etabliert und nicht-vakuär.

**Der zentrale Gewinn — M1 empirisch bewiesen (VOR dem Bau):** Eine Scratch-Probe (`sqrt_probe.rs`,
kein Teil der Crate, `rustc --target wasm32-unknown-unknown`, mit UND ohne `+simd128` aus dem
repo-eigenen `.cargo/config.toml`) verglich `f64::sqrt` gegen `Math.sqrt` (dieselbe V8-Engine) über
**30.028 Eingaben**: alle Spezialkanten (±0, ±Inf, NaN inkl. abweichender Payloads, negativ,
MAX_VALUE, kleinstes Subnormal, kleinster Normalwert) + 20.000 rohe f64-Bitmuster über das GESAMTE
Exponentenspektrum + 10.000 normale Zufallswerte. **0 Mismatches** in beiden Rustflag-Varianten;
finite/`±0`/`±Inf` byte-exakt, NaN nur als Klasse (0 verbotene Payload-Diffs). Verifiziert, dass die
elementweisen Kernel KEINE SIMD-Intrinsics nutzen (nur `matmul_blocked.rs`) — die Probe ist damit
direkt repräsentativ für den tatsächlichen `sqrt_strided`-Codegen. M1-Bitparität ist damit gemessen,
nicht angenommen.

**Freeze/M4 bestätigt tragfähig:** `abi.rs` endet auf einem `#[cfg(test)]`-Block; „Produktionscode
strikt danach anhängen" ist das bereits gelebte Muster (der komplette Kern-07-Block wurde selbst so
angehängt), `kernels/mod.rs` endet auf `pub mod vector;`. `#[track_caller]` ist in der Crate aktuell
nirgends verwendet — der Zeilenshift-Mechanismus ist derzeit nicht aktiv scharf, die Append-Disziplin
bleibt trotzdem Hygiene. Der Dekompositions-Beweis (Pre-Edit-Rebuild reproduziert `0b9df4f1…`,
additive-only, neuer Pin) ist der zweimal (Kern 07/11) erprobte Mechanismus.

**Zwei Major-Befunde (in v2 eingearbeitet):**
- **C1 — D7s check:diag-Budgetbegründung war sachlich falsch.** v1 behauptete, die Test-Anhänge seien
  nicht im tsc-Korpus; sie SIND es (`tsconfig.json` includet `spike`, excludet `tests-runtime` NICHT),
  ihr Testinhalt trägt reale, mitzuzählende Typkosten. Kein Order-Noise (keine neue Datei), aber der
  Δ ist Methodenteil + Testinhalt. D7 in v2 korrigiert; das ≤+1.000-Gate gilt für den gesamten Δ.
- **C2 — fehlender `genDataSpecial`-Raster-Block.** Die Datei-Konvention von `special-values.test.ts`
  (jede Op mit residentem Kernel bekommt einen randomisierten SPECIAL_VALUES-Raster, vgl. der
  `transpose`-Block) war in D6 nur durch kuratierte Einzel-Fixtures abgedeckt. In v2 als D6(b)
  ergänzt (ca. 60 Fälle nach dem `transpose`-Muster), ausdrücklich OHNE byte-exakten NaN-Payload-Test
  (sqrt ist Arithmetik).

**Minor / Nits (in v2 berücksichtigt):** M-1 — D4 präzisiert auf einen NEUEN dritten Merge-Block statt
Edit in den bestehenden (loader.ts-Konvention). N2 — `nt_materialize` als nächstes ABI-Parameterform-
Vorbild in D3 ergänzt. N1 (D5 näher an `sum()`s achsenlosem Zweig als an `add`) und N3
(`#[track_caller]` nicht aktiv) sind rein dokumentarisch, ohne Plan-Wirkung — hier festgehalten,
keine Textänderung nötig.

## Änderungslog

- **v3 (2026-07-23):** Owner-bestätigte Scope-Erweiterung nach einem Umsetzungs-Befund. bench:editor
  fiel mit uniformem +7 (statt der erwarteten Δ0); eine Mechanismus-Untersuchung führte es an EINE
  Zeile zurück (threaded.ts:413s `Omit<ThreadedCoreExports, "memory">`, `keyof`-getriebene
  Generic-Neuauflösung, +7 fix pro `CoreExports`-Member, kumulativ über die Kampagne). Neue
  Entscheidung D10: `Omit` → direkter Cast (laufzeit-identisch, gemessen sicher), beseitigt die Ripple
  an der Wurzel für sqrt UND jede Folge-Scheibe plus einen stehenden −159-Fixblock. Ist-Werte in D7
  aktualisiert (Root −4, stress/bench:editor −159, browser Δ0), M5-Covenant-Eintrag auf „threaded.ts
  typ-only berührt" präzisiert, T7 auf die Pin-Bewegungen umgestellt. Owner-Entscheidung „Quelle jetzt
  fixen" (gegen „+7 akzeptieren, Quelle später").
- **v2 (2026-07-23):** Baustein 0 (adversarial gegen v1) fand keinen Blocker und bewies die
  M1-Bitparität `f64::sqrt` ↔ `Math.sqrt` empirisch vor dem Bau (30.028 Fälle, 0 Mismatches, beide
  Rustflag-Varianten). Zwei Major-Befunde eingearbeitet: C1 (D7-check:diag-Budgetbegründung korrigiert
  — Test-Anhänge SIND im tsc-Korpus), C2 (randomisierter `genDataSpecial`-Raster-Block in D6(b)
  ergänzt). M-1 (D4 auf neuen dritten Merge-Block präzisiert), N2 (`nt_materialize`-Vorbild in D3).
  Addendum-Abschnitt dokumentiert alle Befunde + die empirische Probe.
- **v1 (2026-07-23):** Erstfassung nach Owner-Richtungsabnahme (WASM-Parität-Pilot sqrt).
