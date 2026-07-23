# WASM-Parität S1 (Skalar-Overloads `add`/`sub`/`mul`/`div`): Umsetzungsergebnisse

Spec: [docs/wasm-parity-scalar-spec.md](wasm-parity-scalar-spec.md) v2 · Datum: 2026-07-23 ·
Status: **Umsetzung abgeschlossen, dreifach verifiziert** (Verify-Runde A+B+C — A CONFIRMED,
B HÄLT, C kein Verstoß; Post-Verification-Addendum unten). Zwei nicht-funktionale A-Befunde in der
Runde behoben (test:resident-Delta-Arithmetik + dangling GFM-Verweis).

**Ehrlichkeitsregel:** Jede Zahl stammt aus einem Kommando mit geprüftem Exit-Code. Was nicht
verifiziert ist, steht als solches da. Diese Scheibe hat **keinen gemessenen Nutzerbedarf** — sie
ist Vollständigkeits-/Symmetrie-Arbeit (die zweite Scheibe der WASM-Parität-Serie, nach dem
S0/sqrt-Pilot), wie in der Spec verankert.

## Was umgesetzt wurde

Zweite Scheibe der WASM-Parität-Serie: `WNDArray.add/sub/mul/div` bekommen einen Skalar-Overload
(`w.div(2)`), sodass residente Daten `data[i] op s` in-WASM verrechnet werden, ohne pro Op nach JS
zu kopieren und ohne den `[1]`-Broadcast-Umweg.

- **Kernel** `crates/core/src/kernels/scalar.rs` (neues File): vier `pub fn scalar_{add,sub,mul,
  div}_strided`, jede ein Einzeiler über den S0-`unary_strided`-Kern (`kernels::sqrt`) —
  `unary_strided` wurde dafür von `fn` auf `pub(crate) fn` erweitert (rein additive
  Sichtbarkeitserweiterung, siehe M4-Abschnitt unten). Operandenordnung gepinnt: `data[i] − s`
  (nicht `s − data[i]`), `data[i] / s` (nicht `s / data[i]`). 15 Cargo-Tests (geteilte
  strided-Maschinerie via `add`, op-spezifische Arithmetik/Ordnungsbeweise, Spezialwert-Kanten je
  Op).
- **ABI** vier `nt_scalar_{add,sub,mul,div}_strided` (9-Parameter: `nt_sqrt_strided`s 8-Parameter-
  Form PLUS `scalar: f64` zwischen `data_len` und `out_data_ptr`, `nt_fill`s eigener
  `value: f64`-Parameter als Präzedenz) strikt ans Ende von `abi.rs` angehängt; `pub mod scalar;`
  ans Ende von `kernels/mod.rs`.
- **Schnittstelle** `CoreExports` um die vier Member erweitert (neuer vierter Merge-Block in
  `loader.ts`, keine bestehende Zeile berührt); `ThreadedCoreExports` erbt automatisch. Vier
  `notImplemented(...)`-Stubs im hand-getippten `CoreExports`-Mock in `backend-oom.test.ts`
  ergänzt (F2-Pflicht-Verdrahtung aus der Spec — ohne sie scheitert `check:diag` mit TS2739).
- **`WNDArray.{add,sub,mul,div}`** (resident.ts): jede der vier Bestandsmethoden bekam einen
  Skalar-Overload (Reihenfolge Skalar-ZUERST, generischer Guard-Träger ZULETZT — D2-v3/W2-F1-
  Lektion, proaktiv eingebaut), der bestehende Array-Array-Körper wurde BYTE-IDENTISCH in den
  else-Zweig der neuen Implementierungssignatur verschoben (am `git diff` bewiesen — die
  verschobenen Zeilen erscheinen als reine Kontextzeilen, kein einziges Zeichen im Körper selbst
  geändert). Neuer privater `scalarOp`-Helfer (reine Klassenkörper-Insertion vor `sqrt()`):
  marshalt einmal, wählt den Kernel-Einstiegspunkt per 4-Wege-`switch`.
- **Threaded-Parität automatisch:** derselbe Kernel in beiden Artefakten; kein Pool-Kernel (der
  Pool routet weiterhin nur matmul).

## M1: Bit-Identität — Korollar-Argument, dreifach am echten Code belegt

Anders als `sqrt` (S0, das eine 30k-Fälle-Vorab-Probe brauchte, weil `f64::sqrt`-Bit-Parität ein
NEUER empirischer Claim war) ist Skalar-Bit-Parität ein **Korollar** bereits eingefrorener Fakten:
`x op s` ist der Spezialfall `y := s` (konstant) derselben binären IEEE-Operation, deren
Bit-Identität zur naiven JS-Referenz die Kern-07-Kernel (`nt_add_strided` & Co.) schon committed
beweisen. Trotzdem dreifach belegt, nicht nur behauptet:

1. **Baustein 0 (vor dem Bau):** verifizierte das Korollar-Argument selbst (sind die binären
   Kernel wirklich als bit-identisch committed?) und fuhr zusätzlich einen 36.324-Fälle-
   JS-vs-WASM-Differential am echten `nt_scalar_add_strided` (alle SPECIAL_VALUES-Paare + 2.000
   Zufalls-f64) — 0 Abweichungen. Keine neue Vorab-Probe war nötig, aber diese existiert bereits
   als Beleg.
2. **Der committete Differentialtest** (D6): `WNDArray.op(s)` gegen `scalarElementwiseRuntime`
   über contiguous + transponierte/geslicte View + Offset-Fenster + rank-0 + size-0-Dim, je Op
   (`elementwise.test.ts`); kuratierte Spezialwert-Fixture + 60 randomisierte
   `genDataSpecial`-Fälle je Op inkl. Spezialwert-SKALAR (`nextF64Special`), plus kuratierte
   `div(0)`/`div(-0)`/`sub`-Ordnungs-Fixtures (F3, `special-values.test.ts`); threaded-vs-stable
   inkl. Spezialwerte je Op (`threaded.test.ts`). NaN als Wert-KLASSE (Skalar-Ops sind Arithmetik;
   finite/±0/±Inf byte-exakt).
3. **Der Pflicht-Mutant** (siehe unten): eine gezielte Kernel-Mutation kippt sowohl die Rust- als
   auch die committete JS-Differentialtests mit benannten Assertions — der Test-Katalog erkennt
   eine echte Regression, nicht nur einen aufgeräumten Erfolgsfall.

## Der `[1]`-Broadcast-Äquivalenz-Beweis

Zusätzlich zum reinen Referenz-Vergleich beweist ein 100-Fälle-Differential (`elementwise.test.ts`),
dass `w.op(s)` byte-identisch zu `w.op(WNDArray.fromArray([1], [s]))` ist — dem VORBESTEHENDEN
Broadcast-Pfad — für Rang ≥ 1. Das ist ein echter Äquivalenzbeweis (zwei unabhängige Code-Pfade,
nicht derselbe Code doppelt aufgerufen), nicht bloß ein zweiter Aufruf desselben Mechanismus.
Rang-0 bleibt bewusst ausgenommen: dort würde der `[1]`-Broadcast `[]` fälschlich in `[1]`
verwandeln — genau die Divergenz, derentwegen die shape-erhaltende Skalar-Semantik existiert.

## Gate-Block

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund) | sauber | 0 |
| `pnpm check:diag` | **208.015 @ 140** (Δ+1.165 gg. 206.850) | 0 |
| `pnpm check:diag:stress` | **106.960 @ 82** (Δ+721 gg. 106.239) | 0 |
| `pnpm check:diag:browser` | **2.142 @ 75** (Δ0) | 0 |
| `pnpm test:core` | 1591 / 1591 (unverändert — kein test:core-File berührt) | 0 |
| `pnpm test:resident` | 4717 pass, 2 skipped (Δ+372 gg. 4345+2) | 0 |
| `pnpm test:threaded` | **91 pass** (75 + 16 neue Skalar-Paritäts-Fälle) | 0 |
| `cargo test` | 184 passed (169 + 15) | 0 |
| `pnpm check:freeze` | neuer Pin `8255821b…` (stable) | 0 |
| `pnpm bench:editor` | 8 Pins uniform +721, Hard-Gate PASS | 0 |
| `pnpm test:example` | unberührt grün (registry-konsumierendes Beispiel, NDArray-only) | 0 |
| `graph-a-lama query lint` | 0 Befunde (frischer Graph) | — |
| GFM-Gate | 0 `<del>` in allen neuen/geänderten `.md` (manuelle Tilde-Prüfung) | — |

## Pins (Δ-Zerlegung)

Baseline im frischen Worktree @ 3f3744b (HEAD, ein reiner CLAUDE.md-Docs-Commit über 9893d70/S0
hinaus — check:diag/bench:editor unverändert davon) reproduziert (Freeze-Hash `24a048c7…`,
check:diag **206.850 @ 140**, stress **106.239 @ 82**, browser **2.142 @ 75**, bench:editor
`{w1 27745, w2 29554, w3 60694, w4 27908, w5 33199, w6 34369, w7 26917, w8 34784}` — alle exakt
reproduziert, exit 0, keine Fehler). Gestufte Messung des Root-Korpus in einem zweiten,
separaten Scratch-Worktree (Dateien schrittweise aus dem Haupt-Baum hineinkopiert, KEIN Rust/WASM
nötig — `check:diag` ist reine TS-Typprüfung):

| Stufe | Inhalt | check:diag root | Δ (Stufe) | Δ (kumulativ) |
|---|---|---|---|---|
| 0 | Baseline | 206.850 @ 140 | — | — |
| 1 | + `CoreExports` (4 Member) + `backend-oom.test.ts`-Stubs | 206.850 @ 140 | **0** | 0 |
| 2 | + `resident.ts` (4 Overload-Umbauten + `scalarOp`) | 207.580 @ 140 | **+730** | +730 |
| 3 | + Test-Anhänge (elementwise/special-values/threaded) + Typ-Pins (ndarray.test-d) | 208.015 @ 140 | **+435** | **+1.165** |

Stufe 3 repliziert exakt den Haupt-Baum-Wert (208.015 @ 140, per `diff -rq` gegen `spike/` bestätigt
— einzige Abweichungen waren erwartete gitignorete Artefakte: `.DS_Store`, `numtype_core*.wasm`,
`bench-dx`-generierte Workload-Verzeichnisse). Dateiset unverändert 140 in JEDER Stufe — **kein
Order-Noise** (keine neue TS-Datei; `scalar.rs` ist Rust, `wasm-parity-scalar-spec.md` ist Markdown,
beide außerhalb des TS-Korpus). Stufe 1 bestätigt den S0/D10-Omit-Fix bei n=4 exakt wie in
Baustein 0 vorab gemessen (**+0/Member**, nicht +7 — der `Omit<ThreadedCoreExports,"memory">`→
direkter-Cast-Fix in threaded.ts trägt kampagnenweit). Der WNDArray-Klassen-Surface-Umbau (Stufe 2)
ist mit +730 die dominante Quelle, nahe an Baustein 0s grober Vorab-Schätzung (≈+820 für vier
Overloads + `scalarOp` + Testinhalt + Typ-Pins). **Absolut-Gate ≤ +6.000: EINGEHALTEN** (+1.165,
≈19 % des Budgets).

`check:diag:stress` (Δ+721, gleiches Dateiset 82) und `check:diag:browser` (Δ0) wurden je zweimal
gemessen (Doppelmessung, identische Werte beide Male) — deterministisch attribuiert: stress
importiert `spike/src` direkt und trägt darum denselben Klassen-Surface-Ripple wie der
Root-Korpus (kleinerer Wert, da der Stress-Korpus weniger vom betroffenen Code berührt);
browser kompiliert `threaded.ts` nicht und rührt `resident.ts`s WNDArray-Klasse offenbar nicht in
einer instanziierungs-relevanten Weise an — Δ0, wie schon bei S0/W2 beobachtet.

`bench:editor`s 8 Pins bewegten sich UNIFORM um +721 (zweimal gemessen, byte-identisch) — dieselbe
WNDArray-Klassen-Surface-Wachstumsursache wie der Root-Korpus, nicht workload-spezifisch (jeder
Workload instanziiert `WNDArray` mindestens einmal). Neu gesetzt in
`spike/bench-dx/editor-latency.ts`: `{w1 28466, w2 30275, w3 61415, w4 28629, w5 33920, w6 35090,
w7 27638, w8 35505}`.

## Freeze-Beweis (M4)

Vollständige Dekomposition, alle Schritte mit geprüftem Exit-Code:

1. **Pre-Edit-Clean-Rebuild reproduziert den Pin exakt:** vor jeder Rust-Änderung, in einem
   frischen Worktree @ HEAD, `pnpm build:wasm` → `24a048c767f3949ad0a8747cecccc0e25e25bdad859c5deb45e218a39d70cea2`
   — exakt der bestehende Pin.
2. **Additive-only-Diff:** neues File `crates/core/src/kernels/scalar.rs` (komplett neu, kein
   Bestand berührt); vier `nt_scalar_*_strided`-Funktionen strikt ans ENDE von `abi.rs` angehängt
   (nach `nt_sqrt_strided`, dem bis dahin letzten Item); `pub mod scalar;` ans Ende von
   `kernels/mod.rs`; EIN Ein-Token-Edit an `kernels/sqrt.rs` (`fn unary_strided` →
   `pub(crate) fn unary_strided`) — eine reine Sichtbarkeitserweiterung, kein Zeilenshift, keine
   Verhaltensänderung (`sqrt.rs` ist kein Frozen-Append-only-File — es ist ein S0-File ohne
   `#[track_caller]`, die Append-only-Disziplin gilt für die DREI genannten Frozen-Dateien
   `abi.rs`/`matmul_blocked.rs`/`shape.rs`). `matmul_blocked.rs`/`shape.rs` byte-unberührt (keine
   Zeile editiert).
3. **Behaviorale Pins grün:** alle 15 neuen `scalar.rs`-Cargo-Tests grün, alle 8 vorbestehenden
   `sqrt.rs`-Cargo-Tests weiterhin grün (unverändert — belegt, dass die `unary_strided`-Wiederver-
   wendung `sqrt_strided`s Verhalten nicht berührt hat), voller `cargo test` 184/184.
4. **Mutations-Beweis:** siehe eigener Abschnitt unten.
5. **Neuer Hash wird der Pin:** `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176`
   (stable), eingetragen in `scripts/check-freeze-hash.mjs` (ersetzt, nicht angehängt — Pin-Set-
   Disziplin D4). `pnpm check:freeze` grün auf dem neuen Pin (Exit 0).

Threads-Artefakt gebaut (`pnpm test:threaded`, Exit 0) mit Hash
`046262911b869351fbd747f6273a837fbf7610d2d83930f6c7405f614ac8f3d8` — bewusst KEIN persistierter
Pin (S0-Präzedenz: `test:threaded` beweist die BEHAVIORALE Bit-Identität der beiden Artefakte
über 16 neue Skalar-Paritäts-Tests, nicht die Datei-Bytes, die durch unterschiedliche
Build-Flags/Shared-Memory-Support strukturell divergieren).

## Pflicht-Mutant (T5)

Mutation in `crates/core/src/kernels/scalar.rs`: `scalar_add_strided`s Closure `|x| x + s` →
`|x| x - s` (Operanden-Ordnungsfehler). Backup-Kopie VOR der Mutation angelegt
(`cp` nach `/private/tmp/…/mutant-backup/scalar.rs.orig`, SHA-256 vor/nach der Mutation
verglichen).

- **Cargo:** 5 benannte Fehlschläge (`add_arithmetic`, `add_offset_window`, `add_same_shape`,
  `add_special_value_edges`, `add_transposed_view_operand`), 10/15 scalar-Tests weiterhin grün
  (sub/mul/div-Tests unberührt, wie erwartet für eine reine add-Mutation).
- **Committeter JS-M1-Differential** (`elementwise.test.ts`, nach `pnpm build:wasm` mit der
  mutierten WASM): 30 benannte Fehlschläge — alle sechs `add(s): …`-View-/Rank-Fälle
  (contiguous/transponiert/geslict/Offset-Fenster/rank-0; size-0 diskriminiert erwartungsgemäß
  NICHT, da leere Arrays keine Operanden-Ordnung offenbaren) plus 25 der 100
  `[1]`-Broadcast-Äquivalenz-Fälle mit `op=add`.
- **Revert:** `cp` aus der Backup-Kopie, `diff`-Beweis (Exit 0, byte-identisch) UND SHA-256-Beweis
  (identisch vor/nach Mutation und nach Revert: `d7ab95da…`). KEIN `git checkout`/`git restore`
  verwendet. `pnpm build:wasm` danach reproduzierte erneut exakt `8255821b…` — der Revert ist
  vollständig und deterministisch.

## T4b/F1: Nicht-Vakuität des Diagnose-Qualitätstests

Der reale-tsc-Diagnose-Test in `special-values.test.ts` (mirrors `scalar-mean.test.ts`s F1-Pin)
wurde selbst per Reihenfolgen-Flip auf Nicht-Vakuität geprüft, wie die Spec verlangt:

- Backup-Kopie von `resident.ts` angelegt (SHA-256 `b79fe1a2…`).
- `add`s Overload-Reihenfolge in `resident.ts` VERTAUSCHT (Guard-Träger zuerst, Skalar zuletzt —
  die exakte W2-F1-Regressionsklasse).
- Diagnose-Test lief erneut: **FEHLSCHLAG**, wie vorhergesagt — die Fehlermeldung kollabierte von
  `cannot broadcast shapes [2,3] and [4]: …` zu `Argument of type 'WNDArray<[4]>' is not
  assignable to parameter of type 'number'` (der Skalar-Decoy). 927/928 special-values-Tests
  grün, genau der Diagnose-Test rot — benannte Assertion, kein False-Positive an anderer Stelle.
- Revert per `cp` aus der Backup-Kopie, `diff`-Beweis (Exit 0) UND SHA-256-Beweis (`b79fe1a2…`
  identisch vor Flip/nach Revert). Anschließend `pnpm check` (Exit 0) und der Diagnose-Test erneut
  GRÜN (928/928) bestätigt.

Damit ist bewiesen: der Test erkennt die W2-F1-Regressionsklasse tatsächlich auf der
WNDArray-Seite — er ist nicht bloß eine kosmetische Kopie des NDArray-Pins.

## Was NICHT getan wurde / offen blieb

- **Verify-Runde A+B+C:** abgeschlossen (A CONFIRMED, B HÄLT, C kein Verstoß — Post-Verification-
  Addendum unten). Zwei nicht-funktionale A-Befunde behoben, B/M4-Coverage-Grenze in FOLLOWUPS.
- **Kein neuer Nutzerbedarf behauptet** — wie in der Spec verankert, bleibt dies Symmetrie-/
  Vollständigkeitsarbeit.
- **`mean` bleibt offen** (S2) — diese Scheibe schließt nur add/sub/mul/div.
- Nichts an `NDArray`, `scalarElementwiseRuntime`, `matmul_blocked.rs`, `shape.rs` wurde berührt
  (Nicht-Ziele der Spec, per `git status`/Diff bestätigt).

## Post-Verification-Addendum (2026-07-23)

Verify-Runde Stufe 3, drei Fresh-Context-Verifier (A/B je isolierter Worktree + Slice-Patch, C
read-only im Haupt-Baum). **Alle drei grün, kein Blocker/Major-Codebefund.**

- **Baustein A (Spec-Konformität + alle Gates frisch + eigener Mutant) — CONFIRMED.** D1–D9 einzeln
  gegen den echten Diff konform; ALLE Gates unabhängig mit exakt den obigen Zahlen reproduziert
  (check:diag 208.015 @ 140, stress 106.960, browser 2.142, test:core 1591, test:resident 4717+2,
  test:threaded 91, cargo 184, Freeze-Pin `8255821b…`, bench:editor 8 Pins exakt). Freeze-
  Dekomposition end-to-end selbst nachgebaut: Pre-Edit-Clean-Rebuild == `24a048c7…`, Post-Slice-
  Clean-Rebuild == `8255821b…`, sqrt.rs' 8 Bestandstests isoliert grün (`pub(crate)` verhaltensneutral
  belegt). Byte-Erhaltung der vier Body-Verschiebungen am Diff bewiesen (resident.ts: 8 Deletions,
  ALLE Doc-/Signaturzeilen, NULL innerhalb eines Methoden-Bodies). Eigener Mutant (`scalar_div`
  `/`→`*`, anders als der add-Flip der Umsetzung) gefangen (cargo 2 benannte + JS 92 auf `div`
  isoliert) + per Backup-Kopie revertiert. **Zwei Befunde, beide nicht-funktional, in dieser Runde
  BEHOBEN:** (1) MAJOR — die `test:resident`-Delta-Angabe war arithmetisch falsch („Δ+248 gg.
  4469+2"; 4469 war der Baseline+elementwise-Zwischenstand, 248 nur das special-values-Subtotal),
  korrekt ist **Baseline 4345+2 → Δ+372** (24+100+4+240+3+1); in diesem Doc UND CLAUDE.md korrigiert.
  (2) MINOR — dangling „GFM-Gate | siehe unten"-Querverweis ohne Sektion; die Gate-Tabellen-Zelle
  nennt jetzt das Ergebnis direkt.
- **Baustein B (adversarial) — HÄLT, kein Blocker.** M1 gegen ein UNABHÄNGIGES Orakel (eigener Code,
  liest WASM-Linearspeicher direkt, kein `scalarElementwiseRuntime`/`toArray`-Reuse): **4.472 Fälle,
  0 Mismatches** über contiguous / transponierte+geslicte Views / 300 mehrfach-komponierte Views /
  rank-0 / size-0 / volles 14×14-Spezialwert-Kartesisch je Op. Operandenordnung unabhängig als
  asymmetrisch bewiesen. 8 Mutanten gefahren, alle verhaltensändernden gefangen; Typ-Pin-Nicht-
  Vakuität beidseitig (Suppression entfernt → echter TS2769; Argument kompatibel gemacht → TS2578
  „unused directive"). Freeze-Hash + check:diag-Dekomposition selbst reproduziert (CoreExports-4-
  Member = Δ0 empirisch bestätigt, nicht nur behauptet; +1.165/+721 exakt). **Ein niedriger Befund
  (M4) → FOLLOWUPS:** die cargo-Suite ruft die Kernel-Funktionen direkt und übt die abi.rs-Dispatch-
  Verdrahtung NICHT — ein ABI-Dispatch-Vertauscher (z. B. `nt_scalar_mul` → `scalar_add`) wird nur
  vom TS-Differential gefangen (0/184 cargo). Kein Blocker (das TS-Differential ist Teil des Pflicht-
  Gate-Blocks und fängt es zuverlässig), aber eine kampagnen-weite Coverage-Grenze für S2–S5.
  NIT: `dispose()` + Skalar-Op ist namentlich ungetestet, aber verhaltenskorrekt (B hat es geprobt —
  wirft via `assertLive` wie jede Schwestermethode).
- **Baustein C (covenant-verify) — kein Verstoß.** Alle berührten Invarianten am Diff eingehalten:
  M1 (Kernel reine `unary_strided`-Aufrufe, Operandenordnung == `scalarElementwiseRuntime`, kein
  FMA/SIMD; Korollar der eingefrorenen binären Kernel), M4 (abi.rs strikt append-only, matmul_blocked/
  shape byte-unberührt, `pub(crate)` additiv, Freeze-Pin sauber ERSETZT), M5 (threaded.ts gar nicht
  im Diff, kein neuer `node:*`-Import, Threads bleiben Node-only-Opt-in), Z1/Z2 (keine Dependency,
  keine neue Datei/Listen-Änderung, backend-oom-Edit reiner Inhalt), M2/M3 (keine neue konfident-
  falsche Kante, UW1–UW4 unangetastet, D2-v3-Reihenfolge bestätigt, T4b-Diagnose-Test neu). Lint
  0/0 auf frischem Graph.

**Prozess-Lehre (kampagnen-weit, → FOLLOWUPS):** Die B/M4-Beobachtung ist strukturell dieselbe Klasse
wie Baustein-0/F1 dieser Scheibe (ein Test, der die zu prüfende Verdrahtung gar nicht durchläuft,
schützt nicht) — hier auf der Rust-Seite: cargo prüft Kernel-Funktionen, nicht die ABI-Dispatch-
Schicht. Für S2–S5 als benannte Coverage-Grenze getrackt; die Deckung existiert über das TS-
Differential, ist aber nicht auf der cargo-Ebene lokalisiert.
