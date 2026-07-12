# NumType — Kern 09: `keepdims` im Runtime-`sum()` — Ergebnisse

*2026-07-12. Spec: docs/kern-09-keepdims-spec.md (inkl. Revision vom selben Tag).
Schließt Phase B Item 5 vollständig ab.*

## Summary

`sum(axis, keepdims)` auf beiden öffentlichen Surfaces (`NDArray`, `WNDArray`):
`keepdims = true` behält die reduzierte Achse als `1` (Rang erhalten), volle
Reduktion wird all-ones. Reine Shape-Metadaten — die summierten Daten sind
byte-identisch zur non-keepdims-Summe, daher **null Rust-Änderungen und
Artefakt-Hash byte-identisch** (starke Freeze-Form, wie Kern 08). Typ-Ebene
war seit Spike 01 fertig (`ReduceAxis<S, Axis, KeepDims>`); die Scheibe ist
die Runtime-Verdrahtung plus `const KeepDims`-Literal-Inferenz. Verifiziert
in einer **Zwei-Verifier-Runde** (neu, Owner-Mandat: Spec-Verifier +
adversarialer Verifier): CONFIRMED / HÄLT.

Diese Scheibe war zugleich der Anlass für ein Qualitäts-Upgrade des
Arbeitsprozesses selbst (Owner-Review der Opus-Session, Findings F1–F5 unten)
— Ergebnis: neuer CLAUDE.md-Abschnitt „Qualitätssicherung, modellunabhängig"
und docs/verify-runde-template.md.

## What was built

- **`keepDimsShape(shape, axis)`** — appended ans Ende von
  spike/src/runtime.ts (append-only gewahrt, `sumRuntime` unangetastet):
  `axis === undefined` → all-ones des Input-Rangs; sonst genau die
  normalisierte Achsenposition durch `1` ersetzt. Single source of truth —
  beide Surfaces rufen denselben Helper, sind also form-gleich per
  Konstruktion; die Daten kommen aus den unveränderten Kerneln (bit-Identität
  vorbewiesen).
- **`NDArray.sum` / `WNDArray.sum`** — Signatur um
  `const KeepDims extends boolean = false` + optionales `keepdims?: KeepDims`
  erweitert; Rückgabetyp `OkShape<ReduceAxis<S, Axis, KeepDims>>`. Der Guard
  bleibt bewusst keepdims-frei (eine ungültige Achse ist in beiden Modi
  identisch ungültig). `WNDArray`: `outLen` und alle Kernel-Argumente
  unverändert (eine 1-Achse ändert `product` nicht); nur die berichtete Shape
  wechselt. Die Erweiterung der BESTEHENDEN Methoden ist die vom Owner am
  2026-07-12 bestätigte D3-Abweichung von der Insertion-only-Hausregel
  (unvermeidlich für einen neuen Parameter; kein Rust → keine
  Artefakt-Freeze-Kopplung; kein anderer Member berührt).
- **Typtests T7a–T7g** (spike/tests/ndarray.test-d.ts): `[2,1,4]` / `[1,1,1]`
  / `[2,3,1]` / explizites `false`; Union-Degradation bei dynamischem
  `boolean` (`[2,4] | [2,1,4]`) als bewusste Gradual-Kante gepinnt;
  `@ts-expect-error` für `sum(3, true)`; WNDArray-Rückgabetyp separat gepinnt
  (unabhängig deklarierte Signatur).
- **spike/tests-runtime/keepdims.test.ts** (367 Fälle, in der
  `test:resident`-Liste): nicht-zirkuläres Design — erwartete Shapes aus der
  trusted non-keepdims-`sumRuntime`-Referenz plus Strukturinvarianten (Rang
  erhalten, reduzierte Achse == 1, Entfernen rekonstruiert die non-keep-Shape,
  `product` unverändert), nie aus `keepDimsShape` selbst. Daten
  `assertDataBitIdentical` gegen die non-keep-Referenz auf BEIDEN Surfaces.
  F5-Nachtrag (Owner-mandatiert): keepdims auf Views — transponiert,
  offset-verschoben (inkl. `undefined`-Achse auf Offset-View →
  `nt_sum_all_strided`), komponiert; plus WNDArray im
  keepdims=false-Paritätsblock.
- **Demo**: keepdims-Showcase auf Referenz + Resident, bit-identisch
  asserted.

## Gates (implementer + beide Verifier unabhängig gemessen)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Verbund root+stress) | clean, Exit 0 |
| `pnpm test:core` | 817 / 0 fail (unverändert — keepdims läuft in test:resident) |
| `pnpm test:resident` | **3646 pass + 2 skipped** / 0 fail (3279 + 367 neue = exakt) |
| `cargo test` | 157 (null Rust-Änderungen) |
| `pnpm demo` | all-agree inkl. `cube.sum(1,keep) shape=[2,1,4]` + Resident-Twin |
| Artefakt-Hash | `7a65d800…` **byte-identisch**; adversarialer Verifier bestätigte über `cargo clean` + Rebuild hinaus |
| `pnpm check:diag` | **174.213 @ 126 Files, Exit 0** (neuer Haupt-Pin; 3× stabil) |
| `pnpm check:diag:stress` | **94.597, Exit 0** (neuer Stress-Pin; +74 durch die `sum`-Signatur in src) |

Scheiben-Delta Haupt-Pin: **+497** gegen 173.716 — plausibel für eine neue
Testdatei + zwei erweiterte Signaturen; Gesamtlast ~3,5 % des 5M-Budgets.

## Pin-Aufklärung (Messvorfall, vollständig aufgelöst)

Während der Implementierung schien der dokumentierte Haupt-Pin 173.716 nicht
zu reproduzieren („Baseline" 168.745). **Auflösung:** Die Baseline war
kontaminiert — `git stash` (ohne `-u`) ließ die untracked
`keepdims.test.ts` im Tree; auf dem alten Stand (sum ohne zweiten Parameter)
erzeugte sie Typfehler, die die Instanziierungszählung drückten, und ein
`grep "Instantiations"` verschluckte die Fehlermeldungen. Ein garantiert
sauberer `git worktree` des Pin-Commits 7292313 maß **exakt 173.716 @ 125
Files, Exit 0** — kein Drift, keine Umgebungsänderung (typescript@7.0.2 inkl.
nativem darwin-arm64-Binary seit Spike 01 exakt im Lockfile gepinnt).
Sämtliche Zwischenzahlen aus kontaminierten Messungen (168.745, 170.782,
170.922) sind **verworfen und werden nirgends verwendet**; maßgeblich sind
ausschließlich die oben verankerten Werte (Korpus-Beweis: Files-Count 126 =
Referenz-125 + genau die eine neue .ts-Datei). Konsequenz → Mess-Hausregel
in CLAUDE.md („Qualitätssicherung, modellunabhängig"): Pins nur im frischen
Worktree messen, immer Exit-Code + Fehlerausgabe prüfen.

## Owner-Review der Session (Findings F1–F5, alle adressiert)

Die Scheibe wurde von einer Opus-Session implementiert und auf Fable
reviewt. Substanz hielt dem Repo-Standard stand; Prozess-Findings:

- **F1 (major):** D3-Abweichung nur in der Spec dokumentiert, nicht aktiv
  vorgelegt → Owner hat nachträglich bestätigt; neue Regel „disclosed +
  confirmed VOR Implementierung" in CLAUDE.md.
- **F2 (major):** kontaminierte Pin-Baseline (siehe Pin-Aufklärung) →
  Mess-Hausregel.
- **F3 (moderate):** Hintergrund-Verifier mit git-Operationen im Live-Tree
  beauftragt → Regel: Hintergrund-Agenten fassen den Haupt-Tree nie an.
- **F4 (minor):** Spec-Testplan sagte `test:core`, implementiert war
  `test:resident` (richtig) → Spec per Revision nachgezogen.
- **F5 (minor):** View-Coverage-Lücke → per Owner-Mandat in dieser Scheibe
  geschlossen (7 View-Tests, s. o.).

Zusätzlich Owner-Mandat: **Verify-Runde = immer zwei Verifier** (Spec +
adversarial) — ab dieser Scheibe gelebt, Template in
docs/verify-runde-template.md.

## Post-verification addendum (Zwei-Verifier-Runde)

**Spec-Verifier (Baustein A): CONFIRMED.** D1–D4 einzeln PASS; Testplan
vollständig realisiert (inkl. Nicht-Zirkularität, per Lektüre geprüft); alle
Gates exakt reproduziert; Testzahl-Korroboration 3646−3279 = 367 = exakt die
neue Suite. Eigener Mutant (Negativ-Achsen-Normalisierung in `keepDimsShape`
entfernt): **74/367 Tests rot**, gefangen u. a. von der
`product must be unchanged`-Invariante — Coverage nicht vakuum.

**Adversarialer Verifier (Baustein B): HÄLT** für die gesamte
Kern-09-Oberfläche. Angriffe abgewehrt: Rang 0, **size-0-Dims** (`[2,0,3]`,
außerhalb des PRNG-Bereichs 1–8 — Implementierung dim-wert-agnostisch
korrekt), dynamische Achse, out-of-range-Achse (Fehlertext byte-identisch zu
keepdims=false, Wurf VOR jeder Allokation), Achsen-Extreme, Doppel-Reduktion,
use-after-dispose, Quell-View-Dispose nach Ergebnis (Ergebnis unabhängig
gültig), alle View-Arten gegen unabhängig hergeleitete Referenzen. **5/5
breite Mutanten gefangen** (78–271 von 367 rot, je an den
Strukturinvarianten). Messungen 3× stabil reproduziert; Freeze über
`cargo clean` + Rebuild bestätigt.

**Befunde der Runde und ihre Behandlung:**

1. **MAJOR, vorbestehend (nicht Kern 09):** `NDArray<S>` mit
   **Mixed-Rank-Shape-Union** (z. B. `NDArray<[2,3] | [2,3,4]>`) akzeptiert
   `.sum(2)` still und liefert konfident einen einzelnen konkreten Shape-Typ,
   obwohl Achse 2 für den `[2,3]`-Zweig ungültig ist — Verletzung von „never
   wrong, only incomplete". Reproduziert **byte-identisch mit
   keepdims=false** → liegt in `ReduceAxis`/`Guard`/`OkShape`, von Kern 09
   weder erzeugt noch verschärft (Kontrolle: die übliche Form
   `NDArray<[2,3]> | NDArray<[2,3,4]>` wird korrekt abgewiesen; betroffen ist
   nur die Union IM Typparameter einer einzelnen Instanz). → **FOLLOWUPS**,
   separates Ticket gegen die Reduce-/Guard-Maschinerie.
2. **MINOR:** `keepDimsShape` validiert die Achse nicht selbst (no-op statt
   Wurf bei out-of-range) — heute sicher, da beide Call-Sites prävalidieren
   (dokumentierter Kontrakt im Doc-Kommentar); defensiver Assert als
   günstige Versicherung → **FOLLOWUPS-Mini** (nicht mehr in dieser Scheibe:
   jede Änderung nach dem Verify würde eine neue Runde erfordern).
3. **Nit, vorbestehend:** kreuzmultiplizierter `ShapeError`-Text bei
   Mixed-Rank-Unions — über keine öffentliche Call-Site erreichbar; wird vom
   FOLLOWUPS-Ticket aus Befund 1 mit erfasst.

## Honesty-Notizen

- Die D3-Abweichung war zum Implementierungszeitpunkt NICHT owner-bestätigt
  (nur spec-dokumentiert); die Bestätigung kam nachträglich im Review. Der
  Prozessfehler ist als F1 dokumentiert und die Regel verschärft.
- Drei Diag-Zwischenzahlen der Implementierungssession waren durch den
  Stash-Fehler unbrauchbar und sind verworfen (s. Pin-Aufklärung) — keine
  davon floss in Pins, Gates oder Entscheidungen ein.
- Der ERSTE Spec-Verifier dieser Scheibe wurde abgebrochen (sein Auftrag
  enthielt den kontaminierten Mess-Ablauf) und lieferte nie einen Report; die
  gültige Verify-Runde ist ausschließlich die oben dokumentierte
  Zwei-Verifier-Runde nach Template.
- `wasmSum` (v1-Backend) bleibt bewusst ohne keepdims (D4: kein öffentliches
  Surface, Differential-Twin; keepdims dort würde nur den Helper auf beiden
  Seiten desselben Vergleichs duplizieren).
