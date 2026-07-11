# NumType — Infra 01: Stress-Typtests in eine separat gemessene Messstrecke (Spec + Ergebnisse)

Date: 2026-07-11 · Status: binding (Owner-Entscheidung Marvin, 2026-07-11: „Extremtests separat
messen" — Auflösung des Kern-08-G2-Trend-Flags) · Ergebnisse werden nach Implementierung unten
ergänzt.

## Entscheidung & Warum

Die check:diag-Pin-Serie (133 656 → 188 378 → 200 714 → 243 446) näherte sich der gewählten
250k-Affordability-Linie, wobei 78 % des letzten Deltas mandatierte Cap-/16-Stellen-Stressfälle im
Type-Test-Korpus waren, nicht Produktkosten. Statt die Linie zu verschieben, wandern die Extremtests
in eine EIGENE, separat gemessene tsconfig-Strecke (bench-dx-Muster: isolierte Projekte mit eigenem
`--extendedDiagnostics`-Zähler). Der Haupt-Pin misst künftig Produkt + realistische/semantische Pins;
die Stress-Strecke bekommt einen eigenen Pin. Beides bleibt bei jedem `pnpm check` MITGEPRÜFT —
getrennt wird nur die MESSUNG, nie die Prüfung.

## Was ist ein „Extremtest" (bindende Abgrenzung)

Digit-Arithmetik-Probefälle mit Operanden ≥ 13 Stellen oder an der MAX_SAFE_INTEGER-Kappe, sowie
mandatierte Stress-Reihen dieser Art. NICHT gemeint: semantische Degrade-Pins (union/never/negativ/
dot-form/exponent — billig, tragen die never-wrong-Semantik), realistische Headline-Fälle
(`[1024, 1024] → [1048576]`), die Spike-01-Rang-/Ketten-Stressfälle in limits.test-d.ts (seit jeher
in jeder Pin-Basis, keine Digit-Arithmetik — Verschieben würde die Pin-Serien-Vergleichbarkeit
brechen).

## Verbindliche Umzugsliste

- `product.test-d.ts` → Stress: **P5** (2⁵²·2⁵²·0, transient 32-stellig), **P9** (7×7-stellig),
  **P10** (exakt AN der Kappe), **P11** (eins drüber). Bleiben: P1–P4, P6–P8, P12–P20.
- `reshape.test-d.ts` → Stress: **RG17** (over-cap im Guard), **T7/T8** (Kappen-Fälle auf
  Methoden-Ebene, inkl. der zugehörigen `declare const`-Zeilen). Bleiben: T6 (1024×1024), RG18
  (Exponent-Form) und alles Übrige.
- `slice.test-d.ts`/`vector.test-d.ts`: nach demselben Kriterium prüfen (Literale ≥ 13 Stellen /
  Kappen-Probefälle) und NUR eindeutige Treffer verschieben; jede Verschiebung im Ergebnisteil
  auflisten. Im Zweifel bleiben lassen.
- Verschobene Fälle sind REINE Relocations: Erwartungen zeichengleich, keine semantische Änderung;
  Querverweis-Kommentar in beiden Richtungen.

## Infrastruktur (bindend)

1. Neues Verzeichnis `spike/tests-stress/` mit `product-stress.test-d.ts` /
   `reshape-stress.test-d.ts` (+ ggf. weitere je Umzugsliste) und eigener **standalone**
   `tsconfig.json` — Compiler-Optionen INLINE gespiegelt, kein `extends` (das bench-dx-dokumentierte
   include/files-Vererbungs-Footgun).
2. Root-`tsconfig.json`: `spike/tests-stress` in `exclude` (Muster: bench-dx/workloads).
3. `package.json`: `check` wird Verbund — `tsc --noEmit && tsc --noEmit -p spike/tests-stress`
   (nichts kann still verrotten); `check:diag` bleibt Root-only (= Haupt-Pin); NEU
   `check:diag:stress` (= Stress-Pin).
4. Nicht-Vakuität der neuen Strecke: eine verschobene Erwartung in einer Scratch-Kopie absichtlich
   verfälschen → `check` MUSS fehlschlagen; Revert grün. (Beweist, dass die Stress-tsconfig die
   Dateien wirklich prüft.)

## Gates (vorregistriert, absolut)

- **G1 (hart):** `pnpm check` (Verbund!) clean, Wall ≤ 1,0 s.
- **G2 (hart):** Haupt-`check:diag` ≤ 250 000 (erwartet deutlich unter dem alten 243 446 — Messwert
  wird Pin); Stress-`check:diag:stress` bekommt KEIN Gate in dieser Scheibe, nur einen Pin (die
  Strecke existiert, DAMIT dort Stress wachsen darf; ein Gate folgt, falls je nötig, aus einem
  künftigen Spec).
- **G3 (hart):** Suiten unverändert grün (core 817 · resident 3279+2 · cargo 157 · threaded 65 ·
  demo · bench:editor hard gate) — reine Typ-Test-Relocation darf NICHTS davon bewegen; Artefakt-Hash
  unverändert `7a65d800…` (null Rust-/Runtime-Berührung).

## Ergebnisse (nach Implementierung ergänzt)

Umgesetzt 2026-07-11. Neue Strecke `spike/tests-stress/` (eigene standalone
`tsconfig.json`, kein `extends`, Compiler-Optionen inline gespiegelt vom Root) mit
`product-stress.test-d.ts` (P5, P9, P10, P11 aus `product.test-d.ts`) und
`reshape-stress.test-d.ts` (RG17, T7, T8 inkl. `atCapArr`/`overCapArr` aus
`reshape.test-d.ts`). Reine Relocation — Erwartungen zeichengleich, nur Importpfade
um eine Ebene angepasst; Querverweis-Kommentare an beiden Enden. `slice.test-d.ts`/
`vector.test-d.ts` geprüft: einzige Treffer für „≥13 Stellen / Kappe" sind
Exponentform-Fälle (`1e21`), die laut Abgrenzung NICHT gemeint sind (billige
Degrade-Semantik, keine Digit-Arithmetik) — nichts verschoben.

**Gates:**
- G1: `pnpm check` (Verbund) clean, Wall **0.529s–0.725s** (mehrere Läufe, alle ≤ 1.0s).
- G2: Haupt-`check:diag` **173,716** Instantiations (Pin; deutlich unter dem alten
  243,446 und unter der 250k-Linie). Stress-`check:diag:stress` **94,523**
  Instantiations (neuer Pin, kein Gate).
- G3: `test:core` 817/817 grün · `test:resident` 3279 pass + 2 skipped grün ·
  `cargo test` 157/157 grün · `test:threaded` 65/65 grün · `demo` alle drei Backends
  bit-identisch · `bench:editor` hard gate PASS (M1–M4). Artefakt-Hash nach
  `pnpm build:wasm` (via `pnpm demo`) unverändert `7a65d80062865a5e88952ce3cfbdd974b642f6d3f4b293e3f3b39afad16885d8`.

**Nicht-Vakuitäts-Beweis:** `P9`s Erwartung in `product-stress.test-d.ts` auf einen
falschen Wert (`1000006000008` statt `1000006000009`) geändert → `pnpm check` schlug
mit `spike/tests-stress/product-stress.test-d.ts(28,18): error TS2344: Type 'false'
does not satisfy the constraint 'true'.` fehl (Exit 1) — der Fehler kam aus der
Stress-Strecke, der Root-`tsc`-Aufruf allein hätte ihn nie gesehen. Revert
(`1000006000009`) → `pnpm check` wieder clean, Exit 0. Beweist: die Stress-tsconfig
prüft die Dateien wirklich, und der Verbund-`check` fängt eine dort verrottete
Erwartung.

## Post-Verification-Addendum (Fresh-Context-Pass, CONFIRMED)

Der Verifier hat unabhängig reproduziert: Relocation byte-identisch und vollständig (Diff jeder
verschobenen Zeile gegen den HEAD-Stand), tsconfig-Spiegelung feldgenau, `--listFiles` = exakt die
zwei Stress-Dateien als Roots, beide diag-Pins 2× exakt (173 716 / 94 523), alle Suiten, Demo,
bench:editor, Artefakt-Hash aus wirklich sauberem Rebuild (`cargo clean`). Dazu ZWEI eigene
Korruptions-Tests in einer Scratch-Kopie: (a) verrotteter Stress-Fall → Verbund-`check` fällt,
Root-`tsc` allein bleibt blind (genau die beabsichtigte Asymmetrie); (b) verrotteter VERBLIEBENER
Haupt-Fall → Root-`tsc` fällt (das Hauptprogramm prüft product/reshape.test-d.ts weiterhin
vollständig). Bypass-Suche: keine weitere tsc-Invocation im Repo, die die Stress-Strecke umgehen
könnte (kein CI, keine aktiven Hooks; `check:diag` ist Root-only BY DESIGN, `check:diag:bench` ist
das vorbestehende separate bench-Projekt).

Ein informativer Befund, übernommen: der Haupt-Pin fiel um **69 730** (243 446 → 173 716) — MEHR als
die Summe der früher pro Fall bisektierten Kosten der verschobenen Fälle. Das ist richtungs- und
größenordnungs-plausibel (TS7-Instantiation-Kosten sind nicht-linear, Cache-/Interaktionseffekte
beim gemeinsamen Entfernen mehrerer großer Digit-String-Fälle) und per Korruptions-Test (b) ist
bewiesen, dass NICHTS anderes aus dem Hauptprogramm gefallen ist — aber eine exakte
Delta-Attribution pro Fall existiert nicht (bewusst nicht nachgemessen; der Beweis „nichts fehlt"
trägt, die Attribution wäre reine Neugier). Hinweis fürs Lesen der Pin-Serie: 173 716 ist eine NEUE
Basis (anderer Messkorpus als die Serie bis 243 446) — künftige Deltas vergleichen gegen 173 716
Haupt bzw. 94 523 Stress, nie quer über den Schnitt.
