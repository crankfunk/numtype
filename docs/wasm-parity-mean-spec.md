# WASM-Parität S2 — `mean` auf `WNDArray`/threaded (bindende Spec)

Status: **bindend** (Owner-Richtungsabnahme 2026-07-23: WASM-Backend-Parität der W1–W5-Ops
nachziehen; S2 ist die dritte Scheibe der Serie, nach S0/sqrt und S1/Skalar-Overloads).
Version: 2 · Datum: 2026-07-23 · Eskalationsleiter: **Stufe 3** (substanzielle Scheibe, neue
öffentliche API + M1-relevant — voller Verify-Katalog A+B+C). Covenant: v5. **v2:** Baustein-0-Befunde
eingearbeitet (F1: Differential-Methodik für keepdims korrigiert — `meanRuntime` hat keinen
keepdims-Parameter, Shape gegen `keepDimsShape` statt direkt; M3-Throw-Charakterisierung korrigiert —
empirisch der `reduce:`-Stem, nicht eine WASM-Status-Meldung). Kern-Design D1–D4 bestätigt: die
Komposition ist über 1.746 Fälle bit-identisch (0 Mismatches), Lebenszyklus leak-frei. Details im
Addendum am Ende.
Roadmap: dritte Scheibe der WASM-Parität-Serie (S2 von S0–S5: sqrt → Skalar-Overloads → **mean** →
item/stack → argmax → topk).

## Ziel & Warum — und was diese Scheibe NICHT ist

`NDArray.mean` (W2) existiert heute NUR auf der naiven JS-Klasse; `WNDArray` (der WASM-residente
Zwilling) hat zwar `sum` (routet durch die `nt_sum_*_strided`-Kernel), aber KEIN `mean` — wer den
Mittelwert eines residenten Arrays will, muss heute selbst summieren und teilen oder nach JS
kopieren. Diese Scheibe schließt die Lücke: eine `WNDArray.mean`-Methode, sodass residente Daten
in-WASM gemittelt werden.

**Die tragende Beobachtung — mean braucht KEINEN neuen Kernel (Komposition):** `mean` ist per
Definition „Summe geteilt durch die Elementzahl". Beide Bausteine existieren bereits als
bit-identisch bewiesene WASM-Kernel: der `sum`-Kernel (v1, `resident.test.ts` beweist
`WNDArray.sum` == `sumRuntime` bit-identisch) und der in **S1** gebaute `scalar_div`-Kernel
(`data[i] / s`). `WNDArray.mean` ist damit exakt `this.sum(axis, keepdims).div(n)` — die
Komposition zweier bestehender residenter Ops, ohne eine Zeile neues Rust. Das ist die
kleinstmögliche Parität-Scheibe: eine reine TS-Methode auf `WNDArray`.

**Warum mean als S2 (easy-first-Begründung):** Nach S1 (`scalar_div`) ist der Divisions-Baustein
vorhanden; mean fällt als Komposition fast gratis mit, genau wie auf der NDArray-Seite (`meanRuntime`
= `sumRuntime` + eine Division). Kein neuer Kernel ⇒ **kein Freeze-Hash-Wechsel** (anders als S0/S1)
⇒ kein Rust/ABI/CoreExports/backend-oom-Stub. Die einzige neue Zutat ist die `mean`-Methode selbst
plus ihre Tests.

**Ehrlichkeits-Rahmen (Owner-Vorgabe, wie bei W2/W3/S0/S1):** Diese Scheibe hat **keinen gemessenen
Nutzerbedarf** — niemand ist auf eine WASM-`mean`-Wand gestoßen. Es ist Vollständigkeits-/
Symmetrie-Arbeit. Das Ergebnisse-Doc darf keinen Nutzerbedarf suggerieren, den es nicht gibt.

Diese Scheibe ist **kein** neuer Kernel (mean komponiert `sum` + `scalar_div`), **kein** Threaded-
Pool-Kernel (der Pool routet weiterhin nur matmul; mean läuft auf dem residenten Core in beiden
Artefakten), **keine** Änderung an `NDArray.mean`/`meanRuntime`/`sumRuntime`/`scalarElementwise
Runtime` (bereits fertig, bleiben die Referenzen), **keine** andere Op (item/stack/argmax/topk folgen
als S3–S5).

## Die M1-Beobachtung: bit-Identität ist ein KOROLLAR zweier bereits bewiesener Kernel

`WNDArray.mean` erzeugt KEINEN neuen WASM-Kern; es komponiert zwei, die einzeln schon
M1-bit-identisch sind. Sein Ergebnis muss bit-identisch zu `meanRuntime` sein, und das folgt aus:

1. **`WNDArray.sum` == `sumRuntime`** — bit-identisch, direkt getestet (`resident.test.ts`,
   `WNDArray.fromArray(...).sum()` vs `sumRuntime` via `assertDataBitIdentical`, contiguous + axis +
   negative axis). Der `sum`-Kernel akkumuliert in derselben logischen Reihenfolge wie `sumRuntime`
   (Kernel-Kontrakt; f64-Addition ist nicht assoziativ, die Reihenfolge ist deshalb tragend und seit
   v1 gepinnt).
2. **`WNDArray.div(scalar)` == `data[i] / s`** — bit-identisch, in **S1** dreifach belegt
   (`nt_scalar_div_strided`, Differentialtest + unabhängiges Oracle + threaded-Parität).
3. **`meanRuntime` = `sumRuntime` dann `out[i] = summed[i] / n`** pro Element (runtime.ts:859-878),
   `n = product(shape)` (undefined) bzw. `shape[normAxis]` (Achse). Die Determinismus-Entscheidung
   D5 der W2-Spec ist `sum/n` pro Element, NIE `sum*(1/n)`.
4. **`n` wird identisch berechnet** (D2 unten spiegelt `meanRuntime`s eigene Formel).

Da `scalar_div` genau `sum[i] / n` rechnet (nicht `sum[i] * (1/n)`), erfüllt die Komposition die
D5-Determinismus-Entscheidung **von selbst**. Bit-Identität `WNDArray.mean` == `meanRuntime` ist
damit ein Korollar von (1)×(2)×(3)×(4) und wird durch einen direkten Differentialtest (D5) bestätigt
— es entsteht KEIN neuer Kernel, der eigens bewiesen werden müsste, sondern eine Kompositions-
Verpflichtung, die der Differentialtest abdeckt.

## Berührte Covenant-Invarianten

- **M1 (Anker `crates/core/src/`, `spike/src/runtime.ts` — COVENANT.md:11-17): KEIN NEUER KERN,
  Anker UNBERÜHRT.** mean fügt keinen WASM-Kern hinzu und ändert `runtime.ts` nicht — es komponiert
  zwei bereits M1-konforme Kernel (`sum`, `scalar_div`). Bit-Identität zu `meanRuntime` folgt per
  Komposition (s. o.) und wird durch den D5-Differentialtest bewiesen. **Novum / Auslegungsfrage für
  Baustein C:** M1s v5-Wortlaut adressiert entweder einen NEUEN Kern (der bit-identisch sein muss)
  oder eine KERNEL-LOSE Referenz-Op (kein WASM, M1 bindet nicht). mean ist ein DRITTER Fall — eine
  WASM-berechnete **komponierte** Op ohne eigenen Kern. Sie ist per Komposition konform; ob M1s
  Wortlaut dafür eine Präzisierung braucht (v6-Kandidat, analog der v5-„kernel-lose-Referenz-Op"-
  Klarstellung), entscheidet der Owner nach Baustein C — nicht still auflösen. Praktisch bleibt die
  Pflicht dieselbe: `WNDArray.mean` bit-identisch zu `meanRuntime`, differentiell bewiesen.
- **M4 (Anker `abi.rs`/`matmul_blocked.rs`/`shape.rs`): UNBERÜHRT, Freeze-Hash UNVERÄNDERT.** Kein
  Rust wird angefasst; kein neuer Kernel, kein neuer Export. Der `check:freeze`-Artefakt-Hash bleibt
  der S1-Pin `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176`. **Verifikations-
  Pflicht (negativ):** ein Clean-Rebuild MUSS den unveränderten Hash reproduzieren; täte er es nicht,
  hätte die Scheibe versehentlich Rust berührt (Bug). `check:freeze` läuft grün ohne Pin-Änderung.
- **M5 (Anker `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`): UNBERÜHRT.** Kein Rust-Import in
  `ndarray.ts`, kein eager `node:*`-Import; `mean` ist reine residente-Memory-Arbeit (delegiert an
  `sum`/`div`). `threaded.ts` wird NICHT editiert (mean fügt keinen `CoreExports`-Member hinzu → keine
  Vererbungs-Verdrahtung nötig). Threaded bleibt Node-only-Opt-in; threaded-Parität ist automatisch
  (dasselbe Crate, dieselben zwei Kernel).
- **Z1 (Anker `package.json`): unberührt.** Keine neue Abhängigkeit, kein Rust.
- **Z2 (Anker `package.json`): unberührt.** Alle neuen Tests hängen an BESTEHENDE Korpus-Dateien an
  (`resident.test.ts` in test:resident, `special-values.test.ts` in test:resident, `threaded.test.ts`
  in test:threaded, `ndarray.test-d.ts` im Root-Typkorpus) — **keine neue Datei, keine Explizitlisten-
  Änderung, `test-scripts-guard` bleibt grün, KEIN Order-Noise.**
- **M2/M3: unberührt.** `mean` ist die dritte Call-Site der bereits bewiesenen `sum`-Typ-Maschinerie
  (`ReduceAxis`/`OkShape`/`Guard` — dieselben Overloads 0/1/2 wie `sum`); es entsteht KEINE neue
  Fehler-Fläche, KEINE neue Degradationskante. Die Achsen-Validierung wird vollständig an `this.sum`
  delegiert — mean's Runtime-Throw ist WORTGLEICH `WNDArray.sum`s Throw (per Delegation), exakt wie
  `meanRuntime` seinen Throw an `sumRuntime` delegiert. **Baustein-0-Befund (empirisch):** diese
  Meldung ist der `reduce:`-Stem (z. B. `reduce: axis 5 is out of range for shape [3,4] (rank 2)`),
  WORTGLEICH zwischen `WNDArray.mean`, `WNDArray.sum` und `meanRuntime`/`sumRuntime` — auf beiden
  Flächen dieselbe, nicht eine flächenspezifische WASM-Status-Meldung; M3 damit sauber konsistent.

---

## Design-Entscheidungen

- **D1 — Scope.** Genau ein neuer Klassenmember: `WNDArray.mean` (Overloads 0/1/2, spiegelt `sum`),
  Body = `this.sum(axis, keepdims).div(n)` mit Freigabe des Zwischenergebnisses. **KEIN** Rust,
  **KEIN** ABI-Eintrag, **KEIN** `CoreExports`-Member, **KEIN** `backend-oom.test.ts`-Stub, **KEIN**
  Freeze-Re-Pin. Wiederverwendung des bestehenden `sum`-Kernels + des S1-`scalar_div`-Kernels über
  die öffentlichen `WNDArray`-Methoden `this.sum(...)` und `.div(n)`. Kein Byte an `NDArray`/
  `meanRuntime`/`sumRuntime`/`vector.ts`/`reduce.ts`/`dim.ts`/`loader.ts`/`threaded.ts`/den Rust-
  Kernels. Keine andere Op.

- **D2 — Komposition + `n`-Berechnung (spiegelt `meanRuntime` exakt).** In `WNDArray.mean(axis?,
  keepdims?)`:
  ```ts
  const summed = this.sum(axis, keepdims);   // WNDArray<reduced+keepdims shape>, fresh resident buffer
  let n: number;
  if (axis === undefined) {
    n = product(this.shape);
  } else {
    const rank = this.shape.length;
    const normAxis = axis < 0 ? rank + axis : axis;
    n = this.shape[normAxis] ?? 1;           // wortgleich meanRuntime:865-872
  }
  ```
  `n` wird aus `this.shape` (INPUT-Shape) berechnet, NICHT aus der reduzierten Output-Shape — mit
  keepdims wird die Achse im Output size-1, aber der Divisor bleibt die ORIGINAL-Achsengröße (exakt
  `meanRuntime`). Die Achsen-Normalisierung (negative Achse) ist byte-gleich `meanRuntime`s eigener.
  `this.sum(axis, keepdims)` validiert die Achse vor der `n`-Berechnung (wirft bei Ungültigkeit) —
  `n`s Normalisierung läuft also nur auf einer bereits geprüften Achse, wie in `meanRuntime`.

- **D3 — Zwischenergebnis-Lebenszyklus.** Das `summed`-WNDArray ist ein frischer residenter Buffer,
  der nach der Division freigegeben werden MUSS (sonst Leak). `.div(n)` liest `summed`s Buffer und
  allokiert ein FRISCHES, unabhängiges Ergebnis (Skalar-Ops aliasen nie einen Operanden, S1). Der
  Zwischen-Buffer wird per `summed.dispose()` in einem `finally` freigegeben, nachdem `.div` sein
  Ergebnis produziert hat:
  ```ts
  const summed = this.sum(axis, keepdims);
  try {
    const n = /* s. D2 */;
    return summed.div(n);                     // fresh result, unabhängig von summed
  } finally {
    summed.dispose();                         // idempotent (resident.ts:375), gibt den Zwischen-Buffer frei
  }
  ```
  `return summed.div(n)` wird VOLLSTÄNDIG ausgewertet (Ergebnis-WNDArray steht), DANN läuft `finally`
  (dispose), DANN kehrt die Funktion zurück — das Ergebnis ist buffer-unabhängig, dispose danach ist
  sicher. Ein `resident-lifecycle`-Leak-Check (D5) beweist die Nicht-Vakuität (kein Netto-Leak über
  viele mean-Aufrufe; das Zwischen-`summed` wird jedes Mal exakt einmal freigegeben).

- **D4 — Typ-Ebene (Overloads 0/1/2, spiegeln `sum` — dritte Call-Site derselben Maschinerie).**
  ```ts
  mean(): WNDArray<OkShape<ReduceAxis<S, undefined, false>>>;
  mean<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
  ): WNDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  mean<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims: KeepDims,
  ): WNDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  mean<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims?: KeepDims,
  ): WNDArray<any> { /* D2/D3 */ }
  ```
  Byte-gleich zu `WNDArray.sum`s Signaturen (resident.ts:855-866) und zu `NDArray.mean` (ndarray.ts:
  846-857) — KEINE neue Typ-Maschinerie, `ReduceAxis`/`OkShape`/`Guard`/`reduce.ts`/`dim.ts`
  unverändert. Rückgabetyp der Impl-Signatur `WNDArray<any>` (wie `sum`), der Body reicht die
  überladene Achse/keepdims an `this.sum` durch (`this.sum(axis as any, keepdims as any)` bzw. die
  vom Bestand etablierte Form) und dividiert. Reine Klassenkörper-Insertion — kein Edit an einem
  Bestandsmember (mean ist NEU; anders als S1, wo add/sub/mul/div für Overloads editiert werden
  MUSSTEN, gibt es hier keine erzwungene Overload-Umbau-Ausnahme).

- **D5 — M1-Differentialtest + Determinismus-Pin + threaded-Parität + Mutant.**
  - **Residenter M1-Test (an `spike/tests-runtime/resident.test.ts` angehängt — bestehende Datei,
    keine Listenänderung; dort steht schon der `WNDArray.sum`-vs-`sumRuntime`-Differential, mean
    spiegelt ihn):** `WNDArray.mean()` / `.mean(axis)` / `.mean(axis, keepdims)` über randomisierte
    Shapes (Rang 0–4, positive + negative Achsen, keepdims true/false). **Vergleichs-Methodik
    (Baustein-0-Befund F1, KRITISCH):** `meanRuntime(shape, data, axis)` hat KEINEN keepdims-Parameter
    und gibt stets die REDUZIERTE (nicht-keepdims) Shape zurück — ein direkter `assertShapeEqual`
    gegen `meanRuntime(...).shape` scheitert bei JEDEM keepdims-true-Fall. Darum: **Daten** gegen
    `meanRuntime(this.shape, data, axis).data` (`assertDataBitIdentical` — keepdims-invariant, eine
    size-1-Achse ändert die Elementzahl nicht); **Shape** gegen `keepdims ? keepDimsShape(this.shape,
    axis) : meanRuntime(...).shape` (`assertShapeEqual`) — NICHT direkt gegen `meanRuntime(...).shape`
    für keepdims=true. Exakt die Methodik, die `NDArray.mean`s eigene Suite bereits nutzt
    (`spike/tests-runtime/scalar-mean.test.ts:285-292`, `:461-466`).
  - **Determinismus-Pin (D5 der W2-Spec, NICHT-VAKUÖS):** ein Handbeispiel, dessen `sum/n ≠
    sum*(1/n)` in f64 (W2-Präzedenz: `n=49, sum=5` → `5/49 = 0.10204081632653061` vs. `5*(1/49) =
    0.1020408163265306`). Precondition-Assertion beweist die Divergenz der beiden Formeln, dann wird
    assertiert, dass `WNDArray.mean` exakt `sum/n` liefert (per `Object.is`) — beweist, dass die
    Komposition `.div(n)` (also `/n`) nutzt, nicht `*(1/n)`. Voll- UND Achsen-Fall.
  - **size-0 → NaN:** leerer Empfänger UND size-0-Achse: `sum = 0`, `n = 0`, `0/0 = NaN` — wie
    `meanRuntime`, KEIN Throw. Beide Reduktionspfade explizit getestet.
  - **Spezialwerte (an `special-values.test.ts` angehängt):** ein randomisierter `genDataSpecial`-
    Raster-Block (ca. 60 Fälle, Rang 0–4, Achse + niladisch), `WNDArray.mean` vs `meanRuntime`
    bit-identisch (NaN als Klasse — mean ist Arithmetik).
  - **Threaded-Parität (an `threaded.test.ts` angehängt):** `WNDArray.mean` auf dem THREADED Core
    bit-identisch zum STABLE Core (das etablierte threaded-vs-stable-Differential um mean erweitern),
    inkl. mindestens ein `genDataSpecial`-Fall (S0/C-2-Lektion) und ein Achsen-Fall.
  - **Leak-Nicht-Vakuität (an `resident-lifecycle.test.ts` oder `resident-gc.test.ts` angehängt, wo
    das Free-Count-Plateau schon geprüft wird):** viele `mean`-Aufrufe hinterlassen kein Netto-Leak
    (der Zwischen-`summed` wird jedes Mal freigegeben) — beweist D3.
  - **Pflicht-Mutant (T5, während Baustein A) — zielt auf die KOMPOSITION (kein neuer Kernel):**
    Kandidat (a) Determinismus: `.div(n)` → `.mul(1 / n)` (also `sum*(1/n)`) — muss vom nicht-vakuösen
    Determinismus-Pin gefangen werden; oder (b) `n`-Berechnung: `n = this.shape[normAxis]` →
    `this.shape[normAxis] + 1` — muss vom M1-Differential gefangen werden. Nachweislich gefangen
    (benannte Assertion), Revert per Backup-Kopie (NIE `git checkout` — harte Arbeitsregel 1).
  - **Typ-Pins (an `spike/tests/ndarray.test-d.ts` angehängt):** WNDArray-`mean`-Gruppe, spiegelt die
    NDArray-W2-`mean`-Pins als zweite Call-Site: niladisch → rang-reduzierte Shape wie `sum`, positive
    Achse, keepdims, negative Achse; Degradations-Wiring (dynamische Achse → `readonly number[]`,
    Union-Achse, Mixed-Rank) nach argmax/sum-Präzedenz; ein `@ts-expect-error`-OOB-Pin (`mean(5)` auf
    Rang 2 → `Guard`-`__shapeError`, wortgleich zu `sum`s reduce-Stem). Re-litigiert NICHT die
    Union-Achsen-Pin-Familie der `sum`-Maschinerie — nur Wiring-Pins, die die dritte Call-Site
    beweisen.

- **D6 — Pins, Budget, Freeze.**
  - **`check:freeze`-Hash UNVERÄNDERT (M4).** Kein Rust, kein Rebuild-Delta erwartet. Verifikation:
    Clean-Rebuild reproduziert `8255821b…` exakt; `check:freeze` grün OHNE Pin-Änderung. Falls der
    Hash sich bewegt, hat die Scheibe versehentlich Rust berührt — STOP + untersuchen.
  - **`check:diag`:** Beitrag ausschließlich aus (1) der neuen `mean`-Methode (Klassen-Surface-
    Wachstum, dritte Call-Site der `ReduceAxis`-Maschinerie — dritter Mechanismus, generische Member
    rippeln) und (2) den Test-Anhängen (reale Typkosten; KEIN Order-Noise, keine neue Datei, keine
    neuen `CoreExports`-Member → auch KEIN `keyof`-Mechanismus). Absolut-Gate: **Haupt-check:diag-
    Wachstum ≤ +6.000** (mean ist kleiner als S1 — 3 Overloads statt 4+Helper, kein neuer
    CoreExports-Member; W2s NDArray-`mean` trug einen Teil der +5.762). Gestufte Messung: Baseline im
    frischen Worktree (HEAD `9763981`: check:diag 208.015 @ 140, stress 106.960, browser 2.142,
    Freeze `8255821b…`), dann Δ nach (①) `mean`-Methode, (②) Test-Anhänge, (③) Typ-Pins. Exit-Code +
    Fehleranzahl IMMER mitprüfen (Arbeitsregel 6). **stress/browser:** Ripple erwarten und
    deterministisch attribuieren (stress importiert `spike/src` direkt → WNDArray-Klassen-Surface-
    Ripple; browser messen). Doppelmessung. `bench:editor` (8 Pins): messen; bei Bewegung neu setzen
    (Doppelmessung, Ergebnisse-Doc dekomponiert).
  - **Test-Zahlen:** `test:resident`, `test:threaded` steigen um die neuen Fälle; `cargo` UNVERÄNDERT
    (kein Rust). Exakte Deltas ins Ergebnisse-Doc.

- **D7 — Doc-Platzierung.** Ergebnisse-Doc `docs/wasm-parity-mean-ergebnisse.md` (volles Narrativ +
  Post-Verification-Addendum). FOLLOWUPS: die Kampagnen-Zeile (S2 ERLEDIGT), das W2-Paritätsitem (die
  `mean`-Lücke schließt; item/stack/argmax/topk bleiben → S3–S5). `docs/roadmap.md` WASM-Parität-
  Sektion: S2 erledigt. CLAUDE.md „Status" + „Aktuelle Pins & Gates" (check:diag/bench:editor-Pins,
  test:resident/threaded-Zahlen, Freeze-Hash UNVERÄNDERT explizit vermerkt, S2 erledigt). Vollnarrativ
  an `docs/projekt-log.md`. Falls Baustein C eine M1-v6-Präzisierung empfiehlt: als v6-Kandidat in
  FOLLOWUPS, nicht still in COVENANT.md ändern (Owner-Entscheidung + Version-Bump).

- **D8 — Sprache.** Code/Kommentare/Tests/Commit-Message: Englisch (Hard Constraint). Spec +
  Ergebnisse-Doc: Deutsch. „ca." statt Tilde, keine Strikethroughs (GFM-Gate, harte Arbeitsregel 4).

## Akzeptanzkriterien

- **T1:** `WNDArray.mean` als Klassenkörper-Insertion (Overloads 0/1/2 byte-gleich zu `sum`), Body =
  `this.sum(axis, keepdims).div(n)` mit `n` nach D2 und `summed.dispose()` im `finally` (D3); keine
  bestehende Member editiert; kein Rust/ABI/CoreExports/loader/threaded berührt.
- **T2 (Freeze UNVERÄNDERT):** Clean-Rebuild reproduziert `8255821b…`; `check:freeze` grün OHNE
  Pin-Änderung; `cargo test` unverändert.
- **T3 (M1):** Residenter Differentialtest deckt niladisch + Achse + negative Achse + keepdims +
  size-0 ab: Daten bit-identisch zu `meanRuntime(...).data` (NaN als Klasse), Shape gegen
  `keepdims ? keepDimsShape : meanRuntime(...).shape` (F1 — NICHT direkt gegen `meanRuntime`s Shape
  für keepdims=true); Spezialwert-Raster; threaded-vs-stable-Parität grün.
- **T4 (Determinismus, NICHT-VAKUÖS):** der `sum/n`-vs-`sum*(1/n)`-Pin beweist zuerst die
  Formel-Divergenz (Precondition), dann `WNDArray.mean == sum/n` (Voll- + Achsen-Fall).
- **T5 (Lifecycle + Mutant):** Leak-Nicht-Vakuität (kein Netto-Leak über viele mean-Aufrufe); ein
  Kompositions-Mutant (Determinismus- ODER `n`-Mutation) nachweislich gefangen + per Backup-Kopie
  revertiert.
- **T6 (Typ):** WNDArray-`mean`-Typ-Pins (niladisch/Achse/keepdims/negative + Degradations-Wiring +
  OOB-`@ts-expect-error`), spiegeln `sum`/NDArray-`mean`; keine neue konfident-falsche Kante.
- **T7 (Gates/Pins):** Gate-Block grün; `check:diag`-Δ dekomponiert + gegatet (≤ +6.000);
  stress/browser deterministisch attribuiert; bench:editor gemessen und (falls bewegt) neu gesetzt;
  Freeze-Hash unverändert bestätigt.
- **T8 (Docs):** Doc-Platzierung (D7) vollständig; M1-v6-Kandidat (falls Baustein C ihn empfiehlt)
  in FOLLOWUPS.

## Nicht-Ziele

Kein neuer Kernel für mean (Komposition `sum` + `scalar_div`), kein Threaded-Pool-Kernel, keine
Änderung an `NDArray.mean`/`meanRuntime`/`sumRuntime`/`scalar_div`/den Rust-Kernels, keine andere
Paritäts-Op (item/stack/argmax/topk sind S3–S5), keine Änderung an `mean`s öffentlicher Signatur/
Semantik auf der NDArray-Seite, keine `var`/`std`, keine Behauptung eines gemessenen Nutzerbedarfs,
kein Browser-Port des Threads-Pfads, keine stille COVENANT-Änderung.

## Gate-Block / Definition of Done

`pnpm check` (Dreier-Verbund) · `check:diag`(+stress/browser, Pin-Protokoll D6) · `pnpm test:core` ·
`pnpm test:resident` (inkl. neuer mean-M1-/Lifecycle-Tests) · `pnpm test:threaded` (baut beide
Artefakte, mean-Parität) · `cargo test --manifest-path crates/core/Cargo.toml` (UNVERÄNDERT) ·
`pnpm check:freeze` (Hash UNVERÄNDERT, kein Re-Pin) · `pnpm bench:editor` (8 Pins, ggf. neu gesetzt) ·
`pnpm test:example` (unberührt) · `graph-a-lama query lint` · GFM-Gate auf allen neuen/geänderten
`.md`.

## Verify-Plan (Stufe 3)

**Baustein 0 (vor dem Bau, gegen DIESE Spec, adversarial — `brainroute:deep`): DURCHGEFÜHRT
2026-07-23, kein Design-Blocker — Befunde im Addendum unten, v2-Merge oben.** Code-Annahmen prüfen
— insbesondere: (a) ist `mean = this.sum(axis, keepdims).div(n)` WIRKLICH für ALLE Fälle äquivalent zu
`meanRuntime` (voll/Achse/keepdims/negative Achse/size-0-Empfänger/size-0-Achse/rank-0)? Empirisch
gegenrechnen. (b) Ist die `n`-Berechnung byte-gleich `meanRuntime` (Input-Shape, nicht Output; negative
Normalisierung; `?? 1` nur auf validierter Achse)? (c) Ist der Zwischen-`summed`-Lebenszyklus korrekt
(genau einmal freigegeben, Ergebnis buffer-unabhängig; kein Use-after-free, kein Doppel-Free, kein
Leak)? (d) Wirft `this.sum(badAxis)` VOR der Division, und ist die Message konsistent? (e) Bindet der
Determinismus-Pin nicht-vakuär (Formel-Divergenz konstruierbar)? (f) Ist die M1-„komponierte-Op"-
Einordnung tragfähig — genügt der Differentialtest, oder verlangt die Kompositions-Verpflichtung mehr
(z. B. Beweis, dass `sum` und `div` NICHT durch eine Zwischen-Rundung interferieren)? (g) Ist der
Freeze WIRKLICH unverändert (kein `mod`/`use`, das ein Rebuild-Delta zwingt)? (h) Covenant: die
M1-v5-Wortlaut-Frage für komponierte Ops (v6-Kandidat?), M4 negativ (Hash unverändert), M5/Z1/Z2. (i)
Testplan-/Typ-Pin-Lücken. Befunde mergen, Design-Blocker mit dem Owner in die Spec einarbeiten,
„Adversariale Spec-Verifikation (Addendum)" hier anhängen, DANN Implementierung.

**Nach der Implementierung:** voller Katalog — **A** (Spec-Konformität pro D, alle Gates frisch,
Freeze-Unveränderlichkeit selbst reproduziert, eigener Mutant), **B** (adversarial: mean gegen ein
UNABHÄNGIGES Orakel — eigene Summe-dann-Teilen-Referenz, nicht `meanRuntime`; komponierte Views,
ausschließlich-Spezialwert-Buffer, size-0/rank-0, keepdims-Kanten, Determinismus, Lebenszyklus unter
Fehlerpfaden/Doppel-mean; Mess-/Freeze-Randbedingungen), **C** (`covenant:covenant-verify`: die
M1-komponierte-Op-Einordnung, M4-Hash-Unveränderlichkeit, M5/Z1/Z2; v6-Kandidat benennen) — parallel,
isoliert (mutierende Verifier je eigener Worktree + Slice-Patch, read-only C im Haupt-Baum). Aufträge
aus docs/verify-runde-template.md. Ergebnisse-Doc mit Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-23)

Verifier: `brainroute:deep`, adversarial gegen v1, read-only im Haupt-Baum, empirische Proben in
eigenem `git worktree` (HEAD `9763981`, node_modules symlinkt, WASM-Artefakt aus dem Haupt-Baum
kopiert + hash-verifiziert; Haupt-Tree nie angefasst). **Verdikt: kein Blocker gegen das Kern-Design
(D1–D4).**

**Der zentrale Beleg — Kompositions-Äquivalenz empirisch bewiesen:** `this.sum(axis, keepdims).div(n)`
ist über **1.746 Fälle** (1.500 randomisiert Rang 0–4 alle Achsen/keepdims-Kombis, 36 size-0-Empfänger,
10 size-0-Achse, 200 `genDataSpecial`) **bit-identisch** zur korrekten Referenz — 0 Mismatches. Der
Lebenszyklus (D3) ist leak-frei (500 Iterationen, Free-Count-Delta exakt `3×iters`, kein Drift); das
Ergebnis wird buffer-unabhängig produziert, bevor `finally` das Zwischen-`summed` freigibt. Der
Determinismus-Pin (D5) ist nicht-vakuär (`5/49` = `0x3fba1f58d0fac688` vs `5*(1/49)` =
`0x…687`; `scalar_div_strided` ist `x / s`, scalar.rs:46-48). `this.sum(badAxis)` wirft VOR der
Division, Message wortgleich `meanRuntime`s `reduce:`-Stem. `n`-Berechnung byte-gleich `meanRuntime`
(runtime.ts:865-872). Freeze-Baseline `8255821b…` reproduziert. Alle vier Test-Anhang-Ziele sind
bestehende, registrierte Dateien (Z2 sauber). Die M1-„komponierte-Op"-Covenant-Frage ist korrekt
gescoped (M1 bindet an „jeder WASM-Kern"; mean fügt keinen hinzu → keine Text-Pflicht, der
Differentialtest liefert die praktische Garantie) — kein Blocker, an Baustein C/Owner delegiert.

**Ein MAJOR-Befund (in v2 eingearbeitet):**
- **F1 — D5-Differential-Methodik falsch für keepdims=true.** `meanRuntime` hat keinen keepdims-
  Parameter und gibt stets die reduzierte Shape zurück; ein direkter `assertShapeEqual` gegen
  `meanRuntime(...).shape` scheiterte bei 523/1.746 Fällen (alle keepdims=true, z. B. `[5,2,2]
  axis=-2 keepdims=true`: erwartet `[5,2]`, tatsächlich `[5,1,2]`). Fix: Daten gegen `meanRuntime.data`
  (keepdims-invariant), Shape gegen `keepdims ? keepDimsShape : meanRuntime.shape` — exakt wie
  `NDArray.mean`s eigene Suite (scalar-mean.test.ts:285-292, 461-466). Unter der korrigierten Methodik:
  0/1.746 Mismatches. D5-erster-Bullet + T3 in v2 präzisiert. **KEINE Design-Änderung** (die Komposition
  ist korrekt; nur die Test-Vergleichsvorschrift war falsch).

**Zwei NITs (dokumentarisch, keine Design-Risiken — schließt die Post-Impl-Verify-Runde):**
- F2 — Typ-Guard-Äquivalenz per Code-/Präzedenz-Lesart bestätigt (NDArray-`mean` hat den exakten Pin
  `MEAN_AXIS_OOB_MSG`), nicht per frischer tsc-Fixture; Baustein A schließt das.
- F3 — threaded-Parität per Design-Argument bestätigt (mean = zwei sequentielle Aufrufe schon-sicherer
  `sum`/`div`, keine neue View-Caching-Grenze), nicht durch einen threaded-Artefakt-Lauf; Baustein A/B
  (test:threaded) schließt das.

**Korrektur meiner M3-Aussage (in v2):** meine v1-Behauptung „auf der WNDArray-Fläche ist das die
WASM-Status-Meldung" war falsch — die Meldung ist der `reduce:`-Stem, wortgleich zwischen beiden
Flächen. In v2 korrigiert.

## Änderungslog

- **v2 (2026-07-23):** Baustein 0 (adversarial gegen v1) fand keinen Design-Blocker und bewies die
  Kompositions-Äquivalenz empirisch (1.746 Fälle, 0 Mismatches; leak-freier Lebenszyklus; nicht-vakuärer
  Determinismus-Pin; Freeze-Baseline reproduziert). Ein MAJOR (F1: keepdims-Differential-Methodik —
  `meanRuntime` hat keinen keepdims-Parameter, Shape gegen `keepDimsShape` statt direkt) eingearbeitet;
  M3-Throw-Charakterisierung korrigiert (`reduce:`-Stem, nicht WASM-Status). Zwei NITs (tsc-Fixture,
  threaded-Lauf) an die Post-Impl-Verify delegiert. Keine Design-Änderung (D1–D4 bestätigt).
- **v1 (2026-07-23):** Erstfassung nach Owner-Richtungsabnahme (WASM-Parität S2 = mean). Zentrale
  Design-Wahl: mean als Komposition `this.sum(axis, keepdims).div(n)` — KEIN neuer Kernel, Freeze-Hash
  UNVERÄNDERT, M1 als Korollar zweier bereits bewiesener Kernel (sum aus v1, scalar_div aus S1). Die
  D5-Determinismus-Entscheidung (`sum/n`) fällt gratis, weil `scalar_div` genau `x/n` rechnet. Neue
  Covenant-Auslegungsfrage benannt (M1-Wortlaut für komponierte Ops, v6-Kandidat) — für Baustein C /
  Owner, nicht still aufgelöst.
