# WASM-Parität S2 (`mean`): Umsetzungsergebnisse

Spec: [docs/wasm-parity-mean-spec.md](wasm-parity-mean-spec.md) v2 · Datum: 2026-07-23 ·
Status: **Umsetzung abgeschlossen, dreifach verifiziert** (Verify-Runde A+B+C — A CONFIRMED,
B HÄLT, C kein Verstoß; Post-Verification-Addendum am Ende). Ein MINOR-Coverage-Befund von B
(mean-M1-Tests deckten keine Views) wurde nach Owner-Entscheidung VOR dem Commit geschlossen
(26 neue View-Fälle); die Zahlen unten sind der Stand NACH diesem Nachtrag.

**Ehrlichkeitsregel:** Jede Zahl stammt aus einem Kommando mit geprüftem Exit-Code. Was nicht
verifiziert ist, steht als solches da. Diese Scheibe hat **keinen gemessenen Nutzerbedarf** — sie
ist Vollständigkeits-/Symmetrie-Arbeit (die dritte Scheibe der WASM-Parität-Serie, nach S0/sqrt und
S1/Skalar-Overloads), wie in der Spec verankert.

## Was umgesetzt wurde

Dritte Scheibe der WASM-Parität-Serie: `WNDArray.mean(axis?, keepdims?)`, sodass residente Daten
in-WASM gemittelt werden können, ohne selbst zu summieren/teilen oder nach JS zu kopieren.

**Die tragende Eigenschaft dieser Scheibe: KEIN neuer Kernel, Freeze-Hash UNVERÄNDERT.** `mean` ist
per Definition „Summe geteilt durch die Elementzahl" — beide Bausteine existieren bereits als
bit-identisch bewiesene WASM-Kernel (der v1-`sum`-Kernel und der S1-`scalar_div`-Kernel). Die
gesamte Implementierung ist eine reine TS-Klassenkörper-Insertion auf `resident.ts`:

```ts
mean<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
  axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  keepdims?: KeepDims,
): WNDArray<any> {
  const axisNum = axis as unknown as Axis | undefined;
  const summed = this.sum(axis as any, keepdims as any);
  try {
    let n: number;
    if (axisNum === undefined) {
      n = product(this.shape);
    } else {
      const rank = this.shape.length;
      const normAxis = axisNum < 0 ? rank + axisNum : axisNum;
      n = this.shape[normAxis] ?? 1;
    }
    return summed.div(n);
  } finally {
    summed.dispose();
  }
}
```

- **Overloads 0/1/2** byte-gleich zu `WNDArray.sum`s eigenen Signaturen (resident.ts, direkt nach
  `sum()` eingefügt) — dritte Call-Site der bestehenden `ReduceAxis`/`Guard`/`OkShape`-Maschinerie,
  keine neue Typ-Maschinerie, `reduce.ts` unverändert.
- **`n`-Berechnung** wortgleich `meanRuntime`s eigener Formel (runtime.ts:865-872): `product(this
  .shape)` für die volle Reduktion, sonst `this.shape[normAxis]` auf der INPUT-Shape (nicht der
  reduzierten Output-Shape — mit keepdims bleibt der Divisor die Original-Achsengröße), negative
  Achsen-Normalisierung identisch. Die Achsen-Validierung läuft vollständig über `this.sum` — ein
  ungültiger Achsen-Wert wirft VOR der `n`-Berechnung, mit demselben `reduce: axis …`-Stamm wie
  `sum` (M3, per Delegation wortgleich, kein separater Check).
- **Lebenszyklus (D3):** `summed` ist ein frischer residenter Puffer; `summed.div(n)` alloziert
  ein UNABHÄNGIGES frisches Ergebnis (Skalar-Ops aliasen nie einen Operanden, S1-Kontrakt) BEVOR
  `finally` läuft — `summed.dispose()` danach kann das bereits produzierte Ergebnis nicht mehr
  berühren. `dispose()` ist idempotent (resident.ts:375).
- **Zero Rust, zero ABI, zero `CoreExports`-Member, zero `backend-oom.test.ts`-Stub, zero
  Freeze-Re-Pin.** `threaded.ts` gar nicht editiert — Threaded-Parität ist automatisch (dasselbe
  Crate, dieselben zwei bestehenden Kernel `sum`/`scalar_div`).

## M1: Bit-Identität — Korollar zweier bereits bewiesener Kernel, direkt getestet

`WNDArray.mean` erzeugt keinen neuen WASM-Kern; sein Ergebnis ist ein Korollar von (1)
`WNDArray.sum == sumRuntime` (bit-identisch, v1) und (2) `WNDArray.div(scalar) == data[i]/s`
(bit-identisch, S1-dreifach belegt) und (3) `scalar_div` rechnet exakt `sum[i]/n`, nie
`sum[i]*(1/n)` — die D5-Determinismus-Entscheidung fällt damit aus der Komposition heraus, ohne
eigenen Beweis nötig zu sein. Trotzdem NICHT nur behauptet, sondern direkt getestet:

- **`resident.test.ts` (M1-Differential, F1-Methodik):** 120 randomisierte `mean()`-Fälle (Rang
  0-4, keepdims true/false, gegen `meanRuntime(shape, data, undefined)`) + 120 randomisierte
  `mean(axis[, keepdims])`-Fälle (Rang 1-4, positive/negative Achse, keepdims true/false, gegen
  `meanRuntime(shape, data, axis)`). **F1-Methodik (Baustein-0-Befund der Spec, kritisch):**
  `meanRuntime` kennt keinen `keepdims`-Parameter und liefert stets die reduzierte Shape — Daten
  werden gegen `meanRuntime(...).data` verglichen (keepdims-invariant), Shape gegen `keepdims ?
  keepDimsShape(shape, axis) : ref.shape` — NIE direkt gegen `meanRuntime(...).shape` für
  keepdims=true. Plus zwei nicht-vakuöse Determinismus-Pins (`sum/n` vs `sum*(1/n)`, `n=49, sum=5`,
  voll + Achse) und zwei size-0-Pins (leerer Empfänger, size-0-Achse — beide `NaN`, kein Throw).
- **`special-values.test.ts` (60 randomisierte `genDataSpecial`-Fälle, Rang 0-4, Achse + niladisch):**
  `WNDArray.mean` vs `meanRuntime` bit-identisch, NaN als Wert-Klasse (mean ist Arithmetik), F1-
  Methodik angewandt.
- **`threaded.test.ts` (10 Fälle):** `WNDArray.mean` auf dem THREADED Core bit-identisch zum
  STABLE Core (contiguous/transponierte View, rank-0, size-0-Dim, Achsen-Fall, keepdims-Fall,
  Spezialwerte niladisch + Achse — S0/C-2-Lektion: mindestens ein Spezialwert-Fall direkt auf dem
  Threads-Artefakt, nicht nur per Crate-Argument).

**Gesamt: 250 (M1-Differential) + 60 (Spezialwerte) + 10 (threaded) = 320 direkte
Bit-Identitäts-Assertionen für `mean`, alle grün, 0 Abweichungen.**

## Gate-Block

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund) | sauber | 0 |
| `pnpm check:diag` | **213.704 @ 140** (Δ+4.189 gg. 209.515, Δ+5.689 gg. der S2-Vor-Baseline 208.015) | 0 |
| `pnpm check:diag:stress` | **107.283 @ 82** (Δ0 gg. 209.515-Stand — resident.test.ts liegt außerhalb des Stress-Korpus) | 0 |
| `pnpm check:diag:browser` | **2.142 @ 75** (Δ0) | 0 |
| `pnpm test:core` | 1591 / 1591 (unverändert — kein test:core-File berührt) | 0 |
| `pnpm test:resident` | **5048 pass, 2 skipped** (Δ+26 gg. 5022+2 — View-Coverage-Nachtrag, Verify-B-Befund) | 0 |
| `pnpm test:threaded` | **101 pass** (91 + 10 neue mean-Paritäts-Fälle, unverändert seit S2 — der Nachtrag berührt keine threaded-Tests) | 0 |
| `cargo test` | **184 + 1 = 185, UNVERÄNDERT** (kein Rust berührt, `git status crates/` leer) | 0 |
| `pnpm check:freeze` | Pin **UNVERÄNDERT** `8255821b…` (Clean-Rebuild reproduziert ihn exakt) | 0 |
| `pnpm bench:editor` | 8 Pins **UNVERÄNDERT** (identisch `{w1 28.789, w2 30.598, w3 61.738, w4 28.952, w5 34.243, w6 35.413, w7 27.961, w8 35.828}` gg. dem S2-Stand — der Nachtrag berührt nur `spike/tests-runtime`, außerhalb der bench:editor-Workloads), Hard-Gate PASS | 0 |
| `pnpm test:example` | unberührt grün (registry-konsumierendes Beispiel, spike/-unabhängig) | 0 |
| `graph-a-lama query lint` | 0 Befunde (frischer Graph, 1093 Dateien) | — |
| GFM-Gate | 0 Strikethrough-Marker in allen neuen/geänderten `.md` (manuelle Prüfung — kein automatisiertes GFM-Skript im Repo gefunden) | — |

## Pins (Δ-Zerlegung)

Baseline im frischen Worktree @ `9763981` (HEAD, S1-Commit) reproduziert: Freeze-Hash `8255821b…`,
check:diag **208.015 @ 140**, stress **106.960 @ 82**, browser **2.142 @ 75**, test:resident
**4719 total / 4717 pass / 2 skip**, test:threaded **91 pass**, cargo **184 + 1 = 185** — alle
exakt reproduziert, Exit 0, keine Fehler (`grep -c "error TS"` = 0 auf allen drei Korpora).

Gestufte Messung des Root-Korpus im selben Worktree (Dateien schrittweise aus dem Haupt-Baum
hineinkopiert — kein Rust/WASM-Rebuild nötig, `check:diag` ist reine TS-Typprüfung):

| Stufe | Inhalt | check:diag root | Δ (Stufe) | stress | browser |
|---|---|---|---|---|---|
| 0 | Baseline @ 9763981 | 208.015 @ 140 | — | 106.960 @ 82 | 2.142 @ 75 |
| 1 | + `resident.ts` (nur die `mean`-Methode) | 208.348 @ 140 | **+333** | 107.283 @ 82 (**+323**) | 2.142 @ 75 (**Δ0**) |
| 2 | + Test-Anhänge (resident/special-values/threaded/resident-lifecycle) | 209.233 @ 140 | **+885** | 107.283 @ 82 (Δ0) | 2.142 @ 75 (Δ0) |
| 3 | + Typ-Pins (ndarray.test-d.ts) | 209.515 @ 140 | **+282** | 107.283 @ 82 (Δ0) | 2.142 @ 75 (Δ0) |
| 4 | + View-Coverage-Nachtrag (Verify-B-Befund: 26 `mean`-Fälle auf transponierten/geschnittenen/offset-verschobenen/zusammengesetzten Empfängern in resident.test.ts) | 213.704 @ 140 | **+4.189** | 107.283 @ 82 (Δ0) | 2.142 @ 75 (Δ0) |

Stufe 4 ist die einzige Nachtrags-Stufe dieses Addendums (angehängt NACH dem ursprünglichen
Drei-Stufen-Bau der Scheibe, ausgelöst vom adversarialen Verifier — s. „Was NICHT getan wurde"
unten für den Kontext) — 26 neue Testfälle auf vier View-Arten (Transpose, Step-Slice,
Offset-Fenster, zusammengesetzte Transpose+Slice-View) je niladisch/positive-/negative-Achse ×
`keepdims` true/false, plus ein `assertMeanViewMatches`-Helper. Dateiset bleibt bei 140 (kein
neues File — reiner Append an eine registrierte Bestandsdatei), stress/browser bei Δ0 (dieselbe
Begründung wie Stufe 2/3: `resident.test.ts` liegt außerhalb ihrer `include`-Globs).
Gesamtdelta gegen die S2-Vor-Baseline 208.015: **+5.689**, Absolut-Gate ≤ +6.000 weiterhin
eingehalten.

Stufe 3 repliziert exakt den Haupt-Baum-Wert VOR dem Nachtrag (209.515 @ 140 / 107.283 @ 82 /
2.142 @ 75, per
`diff -rq` gegen `spike/` bestätigt — einzige Abweichungen waren erwartete gitignorete Artefakte:
`bench-dx/scale-workloads`, `bench-dx/workloads`, `numtype_core_threads.wasm`). Dateiset
UNVERÄNDERT 140/82/75 in JEDER Stufe — **kein Order-Noise** (keine neue Datei, alle vier
Test-Anhänge gehen an registrierte Bestandsdateien, kein neuer `CoreExports`-Member also auch kein
`keyof`-Mechanismus). Stufe 1 bestätigt: die `mean`-Methode selbst trägt +333 auf dem Root-Korpus
und +323 auf stress (dritte Call-Site der `ReduceAxis`-Maschinerie — dritter Kostenmechanismus,
generische Member rippeln über die Klassen-Surface, nicht der `CoreExports`/`keyof`-Mechanismus,
der bei S0/S1 zum Tragen kam, da `mean` keinen `CoreExports`-Member hinzufügt). stress und browser
bewegen sich NUR in Stufe 1 (Δ0 in Stufe 2/3, da die Test-Runtime-Dateien und die Typ-Pin-Datei
für stress/browser über deren eigene `include`-Globs gar nicht erreichbar sind — nur `spike/src`
wird transitiv importiert). **Absolut-Gate ≤ +6.000: EINGEHALTEN** (+1.500, ≈25 % des Budgets, und
kleiner als S1s +1.165 wie in der Spec erwartet — mean hat nur 3 Overloads statt 4 Overload-Umbauten
+ Helper, keinen neuen `CoreExports`-Member).

`check:diag:stress` und `check:diag:browser` wurden je zweimal gemessen (identische Werte beide
Male) — deterministisch attribuiert: stress importiert `spike/src` direkt und trägt darum denselben
Klassen-Surface-Ripple wie der Root-Korpus (kleinerer Absolutwert, da der Stress-Korpus weniger vom
betroffenen Code berührt); browser rührt `resident.ts`s `WNDArray`-Klasse offenbar nicht in einer
instanziierungs-relevanten Weise an — Δ0, wie schon bei S0/S1 beobachtet.

`bench:editor`s 8 Pins bewegten sich UNIFORM um +323 (zweimal gemessen, byte-identisch beide Male)
— dieselbe `WNDArray`-Klassen-Surface-Wachstumsursache wie der Root-/Stress-Korpus (jeder Workload
instanziiert `WNDArray` mindestens einmal). Neu gesetzt in `spike/bench-dx/editor-latency.ts`:
`{w1 28789, w2 30598, w3 61738, w4 28952, w5 34243, w6 35413, w7 27961, w8 35828}`. Korrektheits-
und Latenz-Gates unverändert PASS (Hover-Median 0,05-0,10 ms, weit unter der 100 ms/200 ms-Grenze).

## Freeze-Beweis (M4) — UNVERÄNDERT bestätigt

Diese Scheibe fügt KEINEN neuen Kernel hinzu — der Freeze-Beweis ist hier eine NEGATIVE
Verifikation (der Hash darf sich NICHT bewegen), nicht eine Dekomposition eines legitimen Changes:

1. `git status crates/` ist LEER — kein einziges Rust-File im Diff.
2. Clean-Rebuild (`cargo clean` + `pnpm build:wasm`) reproduziert exakt den bestehenden S1-Pin:
   `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176`.
3. `pnpm check:freeze` grün OHNE Pin-Änderung (Exit 0), zweimal bestätigt (vor und nach dem
   Mutanten-Zyklus unten, exakt derselbe Hash beide Male).
4. `cargo test`: **184 unittests + 1 zero_alloc-Integrationstest = 185, exakt UNVERÄNDERT** gegen
   die im selben Baseline-Worktree gemessene Kontrollzahl (identisch: 184 + 1, 0 Fehler).
5. Threads-Artefakt gebaut (`pnpm test:threaded`, Exit 0) — bewusst KEIN persistierter Pin
   (S0/S1-Präzedenz: die 10 neuen threaded-vs-stable-`mean`-Tests beweisen die BEHAVIORALE
   Bit-Identität der beiden Artefakte direkt, nicht die Datei-Bytes, die durch unterschiedliche
   Build-Flags/Shared-Memory-Support strukturell divergieren).

**Ergebnis: Freeze-Hash bleibt `8255821b…` — bestätigt UNVERÄNDERT, kein Re-Pin nötig.**

## Pflicht-Mutant (T5)

Kandidat (a) aus der Spec: `.div(n)` → `.mul(1/n)` (Determinismus-Mutation, `sum*(1/n)` statt
`sum/n`). Backup-Kopie VOR der Mutation angelegt (`cp` nach
`/private/tmp/…/scratchpad/mutant-backup/resident.ts.orig`, SHA-256 `c3fe41a5…` vor der Mutation).

- **Mutation angewandt** direkt an `resident.ts`s `mean()`-Methode (`return summed.mul(1 / n);`
  statt `return summed.div(n);`), `pnpm run test:resident` erneut ausgeführt.
- **71 benannte Fehlschläge** (von 5024 Tests): **27/120** `mean_all`-Fälle, **39/120**
  `mean_axis`-Fälle, **3/60** `mean special`-Fälle (special-values.test.ts), plus BEIDE
  dedizierten Determinismus-Pins namentlich:
  - `resident mean: sum/n vs sum*(1/n) discriminator, full reduction (n=49, sum=5)`
  - `resident mean: sum/n vs sum*(1/n) discriminator, axis form (n=49, sum=5)`

  Dass nicht ALLE 320 `mean`-Assertionen fehlschlagen, ist erwartet und spec-konform (W2-
  Präzedenz „nicht jedes Beispiel diskriminiert" — für viele zufällige f64-Wertepaare gilt
  zufällig `sum/n == sum*(1/n)` exakt bitweise; die Leak-Non-Vakuität und die
  size-0/rank-0-Randfälle diskriminieren strukturell gar nicht, da `n=1` bzw. beide Formeln bei
  `0/0` gleichermaßen `NaN` liefern). Die beiden DEDIZIERTEN, nicht-vakuösen Determinismus-Pins
  sind exakt für diesen Zweck gebaut und fangen ihn zuverlässig.
- **Revert:** `cp` aus der Backup-Kopie zurück, `diff`-Beweis (Exit 0, byte-identisch) UND
  SHA-256-Beweis (`c3fe41a5…` identisch vor Mutation und nach Revert). KEIN `git checkout`/
  `git restore` verwendet. `pnpm run test:resident` danach: **5022/5024 pass, 2 skipped, Exit 0**
  (vollständig grün). `pnpm check` (Dreier-Verbund) danach: Exit 0. `git status --short` zeigt nur
  die 13 beabsichtigten Datei-Änderungen dieser Scheibe, keinen Mutanten-Rest. Freeze-Hash nach
  dem gesamten Mutanten-Zyklus erneut `8255821b…` (unverändert, wie erwartet — der Mutant war
  reiner TS-Code, kein Rust).

## Leak-Non-Vakuität (D3)

`resident-lifecycle.test.ts`: ein neuer Test ruft `mean(1)` **N=500 mal auf EINEM persistenten
Empfänger** auf, disposed jedes Ergebnis, und misst `getResidentFreeCount()` als exakte Delta
(nicht nur eine Plateau-Beobachtung) — der Empfänger selbst wird bewusst AUSSERHALB des gemessenen
Fensters alloziert/disponiert, damit sein eigener Lebenszyklus die D3-spezifische Aussage nicht
verwässert.

- **Erwartete Delta: 2N = 1000** (der Zwischen-`summed`-Puffer, freigegeben in `mean`s eigenem
  `finally`, PLUS der finale `.div(n)`-Ergebnis-Puffer, freigegeben vom Test selbst) — **gemessen:
  exakt 1000, Assertion grün.** Weder ein Leak (Delta zu klein) noch ein Doppel-Free (Delta zu
  groß oder ein korrupter Allokator bei der Folge-Allokation) ist beobachtbar.
- **Unabhängige Zweitbestätigung** im selben Test: nach 20 Aufwärm-Zyklen (frischer Empfänger je
  Zyklus, `mean(1)` + dispose beider Puffer) bleibt `core.memory.buffer.byteLength` über 500
  weitere Zyklen exakt auf einem Plateau — kein Wachstum, dieselbe Disziplin wie die bestehende
  „leak plateau"-Kontrolle in derselben Datei.

Beide Assertionen grün, `pnpm run test:resident` Exit 0.

## Was NICHT getan wurde / offen blieb

- **Verify-Runde A+B+C:** abgeschlossen (A CONFIRMED, B HÄLT, C kein Verstoß — Post-Verification-
  Addendum unten). B's MINOR-View-Coverage-Befund wurde geschlossen, A's NIT (Datei-Zahl) korrigiert.
- **Kein neuer Nutzerbedarf behauptet** — wie in der Spec verankert, bleibt dies Symmetrie-/
  Vollständigkeitsarbeit.
- **item/stack/argmax/topk bleiben offen** (S3-S5) — diese Scheibe schließt nur `mean`.
- Nichts an `NDArray`, `meanRuntime`, `sumRuntime`, `scalar_div`, den Rust-Kerneln,
  `threaded.ts` oder `loader.ts`/`CoreExports` wurde berührt (Nicht-Ziele der Spec, per
  `git status`/Diff bestätigt — `resident.ts` zeigt exakt 65 Insertionen, 0 Deletionen/Edits an
  Bestandsmembern).
- **COVENANT M1-v6-Frage** (komponierte Op ohne eigenen Kernel) bleibt eine offene, in FOLLOWUPS
  getrackte Owner-Entscheidung — nicht in dieser Scheibe still aufgelöst (siehe Spec-Addendum und
  FOLLOWUPS.md).
- Kein automatisiertes GFM-`<del>`-Check-Skript im Repo gefunden (weder in `scripts/` noch in
  `.github/workflows/ci.yml`) — die GFM-Prüfung dieser Scheibe ist eine manuelle Tilde-Grep-Probe
  über die neuen/geänderten `.md`-Dateien, kein mechanisches Gate.

## Post-Verification-Addendum (2026-07-23)

Verify-Runde Stufe 3, drei Fresh-Context-Verifier (A/B je isolierter Worktree + Slice-Patch, C
read-only im Haupt-Baum). **Alle drei grün, kein Blocker/Major.** Ein MINOR-Coverage-Befund von B
wurde nach Owner-Entscheidung noch VOR dem Commit geschlossen (s. u.).

- **Baustein A (Spec-Konformität + alle Gates frisch + eigener Mutant) — CONFIRMED.** D1–D8 einzeln
  konform; ALLE Gates unabhängig mit exakt den berichteten Zahlen reproduziert. Der Freeze-Beweis
  wurde als ECHTER Clean-Rebuild nachgebaut (`crates/core/target` gelöscht + `cargo clean` +
  `pnpm build:wasm`) → Hash `8255821b…` unverändert, keine Artefakt-Wiederverwendung. `resident.ts`
  ist eine reine Insertion (0 Deletions); der einzige `-`/`+`-Hunk außerhalb neuer Blöcke ist eine
  KOMMENTAR-Korrektur in `ndarray.test-d.ts` (die veraltete Notiz „WNDArray hat weder add/sub/mul/div
  noch mean" — seit S1/S2 falsch), ohne Assertion-/Pin-Logik zu berühren. Eigener Mutant (n-Off-by-one
  im Achsen-Zweig, anders als der `.mul(1/n)`-Mutant der Umsetzung): 141 benannte Fehlschläge, exakt
  auf den Achsen-Zweig isoliert (niladische Fälle korrekt unberührt) — per Backup-Kopie revertiert,
  SHA-256 vor/nach identisch. Zwei NITs: die „neun Datei-Änderungen"-Zahl im Mutant-Abschnitt (real
  13) — in dieser Runde korrigiert; und der bekannte, vorbestehende Umstand, dass kein automatisiertes
  GFM-Skript existiert (ehrlich offengelegt, nicht von dieser Scheibe eingeführt).
- **Baustein B (adversarial) — HÄLT, kein Blocker/Major.** M1 gegen ein UNABHÄNGIGES Orakel mit
  strukturell anderem Algorithmus (Odometer-Walk über Input-Koordinaten statt `sumRuntime`s
  „Output iterieren, Achse einwärts laufen"; `meanRuntime` nie importiert): **1.256 Fälle, 0
  Mismatches** über contiguous/transponierte/geslicte/offset/komponierte Views + rank-0/size-0/
  size-1-Achse/all-NaN. **Neun Mutanten, alle gefangen** — u. a. fehlende Negativ-Achsen-
  Normalisierung (61 Fehler), `n` aus der Output- statt Input-Shape (116), `dispose()` entfernt
  (Leak — gefangen NUR vom Leak-Test, s. u.), `dispose()` vor `.div()` (use-after-free, 305),
  keepdims zwangsweise false (144 — belegt, dass die F1-Methodik NICHT vakuös ist), Achsen-
  Durchreichung gebrochen (96). Lebenszyklus unabhängig bestätigt, auch auf einem STRIDED-VIEW-
  Empfänger (1.000 mean-Aufrufe, Free-Count exakt vorhergesagt) und auf dem Fehlerpfad (bad axis
  wirft, BEVOR ein Zwischen-Buffer allokiert wird — kein verwaistes Intermediate). Typ-Pin per
  TS2578-Gegenprobe nicht-vakuär. B widerlegte zudem die Spec-Nit F3: der threaded-Test ist
  nicht bloß Infrastruktur-Verdrahtung, sondern fängt echte Logikfehler (Mutant 9).
- **Baustein C (covenant-verify) — kein Verstoß.** M1 (kein neuer Kern, `crates/` nicht im Diff;
  Komposition per 320 Bit-Identitäts-Assertionen belegt), M4 (Freeze-Hash unverändert, kein Re-Pin,
  abi.rs/matmul_blocked.rs/shape.rs nicht im Diff), M5 (`threaded.ts` nicht im Diff, kein neuer
  `node:*`-Import), Z1/Z2 (keine Dependency, keine neue Datei, package.json unberührt), M2/M3
  (mean-Overloads byte-gleich zu `sum`, dritte Call-Site, `reduce:`-Stem delegiert), S1 (lint 0/0).

**Adressierte Befunde:**
- **B/MINOR (View-Coverage) — GESCHLOSSEN in dieser Runde (Owner-Entscheidung).** B fand, dass die
  committeten residenten mean-M1-Tests `mean` ausschließlich über `WNDArray.fromArray` (contiguous,
  Offset 0, natürliche Strides) prüften und NIE auf einer transponierten/geslicten/offset/komponierten
  View — obwohl die Schwester-Scheibe S1 View-Fälle bewusst mitgetestet hatte (eine Lücke der S2-Spec
  D5, nicht der Umsetzung). B's eigener 650-Fälle-View-Sweep fand 0 Mismatches, es war also eine
  Coverage-Claim-Lücke, kein Live-Bug. **Geschlossen:** 26 neue View-Fälle in `resident.test.ts`
  (transponiert/reversed strides · `slice({step:2})`/non-natural strides · `slice({start:2})`/Offset 6 ·
  komponierte Transpose-of-Slice), je über Achsen × keepdims, mit derselben F1-Methodik (Daten gegen
  `meanRuntime(viewShape, view.toArray(), axis).data`, Shape gegen `keepdims ? keepDimsShape : ref.shape`).
  Alle 26 im ERSTEN Lauf grün — exakt wie B's Orakel vorhersagte. Kosten: check:diag +4.189 (Stufe 4
  der Δ-Zerlegung), test:resident +26; stress/browser/bench:editor/cargo/Freeze-Hash ALLE unverändert.
- **B/informativ (Single Point of Coverage):** der Leak-Nicht-Vakuitäts-Test ist der EINZIGE Detektor
  der gesamten „dispose vergessen"-Mutationsklasse (Mutant 3 brach sonst nichts). Festgehalten, damit
  klar ist, dass dieser eine Test nicht beiläufig entschärft werden darf.
- **C (M1-Wortlaut für komponierte Ops) → FOLLOWUPS (v6-Kandidat):** mean ist ein dritter, textlich
  nicht benannter Fall (WASM-berechnet, aber ohne eigenen Kernel). Praktisch ist M1 erfüllt und
  belegt; die Lücke betrifft nur den Vertragstext. Owner-Entscheidung, bewusst nicht still aufgelöst.

**Finale Zahlen nach dem View-Coverage-Nachtrag** (check:diag vom Orchestrator unabhängig
nachgemessen, Exit 0 / 0 Fehlerzeilen): check:diag **213.704 @ 140** (Gesamt-Δ **+5.689** gegen die
Baseline 208.015 — das Absolut-Gate ≤ +6.000 hält, mit 311 Marge; die Enge ist der Preis der
View-Coverage und hier offengelegt) · stress **107.283 @ 82** · browser **2.142 @ 75** · test:core
**1591** · test:resident **5048+2** · test:threaded **101** · cargo **185 (unverändert)** ·
`check:freeze` **`8255821b…` UNVERÄNDERT** · bench:editor 8 Pins unverändert, Hard-Gate PASS.
