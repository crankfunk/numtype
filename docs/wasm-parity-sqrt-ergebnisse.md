# WASM-Parität S0 (`sqrt`): Umsetzungsergebnisse

Spec: [docs/wasm-parity-sqrt-spec.md](wasm-parity-sqrt-spec.md) v3 · Datum: 2026-07-23 ·
Status: **Umsetzung abgeschlossen**, Verify-Runde A+B+C durchgeführt (alle drei grün, kein
Blocker/Major).

**Ehrlichkeitsregel:** Jede Zahl stammt aus einem Kommando mit geprüftem Exit-Code. Was nicht
verifiziert ist, steht als solches da. Diese Scheibe hat **keinen gemessenen Nutzerbedarf** — sie
ist Vollständigkeits-/Symmetrie-Arbeit (der Pilot der WASM-Parität-Serie), wie in der Spec
verankert.

## Was umgesetzt wurde

Erste Scheibe der WASM-Parität-Serie: `sqrt` bekommt einen from-scratch Rust/WASM-Kernel und eine
`WNDArray.sqrt()`-Methode, sodass residente Daten in-WASM wurzelgezogen werden (statt pro Op nach
JS zu kopieren).

- **Kernel** `crates/core/src/kernels/sqrt.rs` (neues File): `unary_strided<F>` — das unäre
  Gegenstück zu `binary_strided` (kernels/elementwise.rs), gleiche Validierungsreihenfolge,
  contiguous fast path, strided general path via `unravel`; `sqrt_strided = unary_strided(|x|
  x.sqrt())`. 8 Cargo-Tests.
- **ABI** `nt_sqrt_strided` (8-Parameter-Signatur wie `nt_materialize`) strikt ans Ende von abi.rs
  angehängt; `pub mod sqrt;` ans Ende von kernels/mod.rs.
- **Schnittstelle** `CoreExports` um `nt_sqrt_strided` erweitert (neuer dritter Merge-Block in
  loader.ts, keine bestehende Zeile berührt); `ThreadedCoreExports` erbt automatisch.
- **`WNDArray.sqrt(): WNDArray<S>`** (resident.ts, Klassenkörper-Insertion): niladisch, guard-los,
  Mechanik gespiegelt von `contiguous()` (validate-before-alloc, Scratch-`finally`, fresh output).
- **Threaded-Parität automatisch:** derselbe Kernel in beiden Artefakten; kein Pool-Kernel (der
  Pool routet weiterhin nur matmul).

## M1: Bit-Identität — dreifach belegt

`sqrt` ist vom Transzendenten-Nicht-Ziel ausgenommen, weil `Math.sqrt` (ECMA-262) und `f64::sqrt`
(IEEE 754) beide die **korrekt gerundete** Wurzel verlangen — genau ein zulässiges Ergebnis pro
Eingabe. **M1 bindet mit dieser Scheibe** (erster Kernel für eine der neuen Ops). Die Bit-Identität
ist auf drei unabhängigen Ebenen belegt:

1. **Vor dem Bau (Baustein 0):** eine Scratch-Probe verglich `f64::sqrt` (wasm32-Zielbuild, mit UND
   ohne `+simd128`) gegen `Math.sqrt` über 30.028 Eingaben inkl. aller Spezialkanten + volles
   Exponentenspektrum — 0 Mismatches.
2. **Der committete Differentialtest** (D6): `WNDArray.sqrt` gegen `sqrtRuntime` über contiguous +
   Views (transponiert/geslict/offset) + rank-0/size-0 (elementwise.test.ts), kuratierte Fixtures
   + 60 randomisierte `genDataSpecial`-Fälle (special-values.test.ts), sowie threaded-vs-stable
   inkl. Spezialwerte (threaded.test.ts). NaN als Wert-KLASSE (sqrt ist Arithmetik; finite/±0/±Inf
   byte-exakt).
3. **Nach dem Bau (Baustein B):** ein UNABHÄNGIGER korrekt-gerundeter IEEE-sqrt-Oracle aus exakter
   BigInt-Arithmetik (ruft weder `Math.sqrt` noch `f64::sqrt`), gegen `WNDArray.sqrt` über 4.000
   Fälle / 102.281 Elemente mit 6 gleichzeitigen NaN-Payloads, komponierten Views und
   rank-0-über-Offset-View — 0 Abweichungen.

## Der D10-Befund: der `Omit`-Fix (aus dem Problem wurde ein Kampagnen-Gewinn)

Während der Umsetzung fiel `bench:editor` mit uniformem **+7** statt der erwarteten Δ0. Eine
Mechanismus-Untersuchung (isolierter Worktree, `tsc --generateTrace`, kontrollierte Varianten)
führte es an **eine Zeile** zurück: threaded.ts:413,
`{ ...(instance.exports as unknown as Omit<ThreadedCoreExports, "memory">), memory }`.
`Omit<T,K> = Pick<T, Exclude<keyof T, K>>` löst `keyof ThreadedCoreExports` dort frisch auf; jeder
neue `CoreExports`-Member fügt der Union einen Zweig hinzu, den `Exclude` (distributiv) und `Pick`
(homomorpher Mapped-Type) neu durchlaufen → **+7 Instantiations pro Member, fix (arity-unabhängig),
kumulativ** (2 Member = +14). Über die ~6-Op-Kampagne wären das ca. +42 allein aus diesem
Mechanismus — ein **vierter, jetzt benannter Kostenmechanismus** neben Order-Noise /
Klassen-Surface-Ripple / echten Typkosten: **`keyof`-getriebene Generic-Alias-Neuauflösung.**

**Owner-Entscheidung „Quelle jetzt fixen" (2026-07-23):** `Omit<…, "memory">` → direkter Cast auf
`ThreadedCoreExports`. Laufzeit-identisch (Baustein B vierfach belegt: Typ-Erasure; byte-identisch
nach Node-Type-Stripping; adversariales Verhalten mit kollidierendem `memory`-Property; das echte
Threads-Artefakt exportiert `memory` gar nicht), typ-legal, und **beseitigt die +7 pro Member
vollständig** — für sqrt UND jede Folge-Scheibe — plus einen stehenden Omit-Fixkostenblock
(−159/Workload). Ein Begründungs-Kommentar an der Zeile verhindert einen „Hygiene"-Rückbau.
**Kampagnen-Implikation: jede Folge-Parität-Scheibe kostet auf diesem Mechanismus jetzt +0 statt
+7.**

## Gate-Block

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund) | sauber | 0 |
| `pnpm check:diag` | **206.850 @ 140** (Δ−4 gg. 206.854) | 0 |
| `pnpm check:diag:stress` | **106.239 @ 82** (Δ−159) | 0 |
| `pnpm check:diag:browser` | **2.142 @ 75** (Δ0) | 0 |
| `pnpm test:core` | 1591 / 1591 | 0 |
| `pnpm test:resident` | 4345 pass, 2 skipped (Δ+67 gg. 4278) | 0 |
| `pnpm test:threaded` | **75 pass** (69 + 4 Parität + 2 Spezialwerte aus C-2) | 0 |
| `cargo test` | 169 passed (161 + 8) | 0 |
| `pnpm check:freeze` | neuer Pin `24a048c7…` (stable), threads `9743338d…` | 0 |
| `pnpm bench:editor` | 8 Pins uniform −159, Hard-Gate PASS | 0 |
| `graph-a-lama query lint` | 0 Befunde (frischer Graph) | — |
| `pnpm test:example` | unberührt grün | 0 |

## Pins (Δ-Zerlegung)

Baseline im frischen Worktree @ 4edea57 reproduziert (von Baustein A + B unabhängig): check:diag
**206.854 @ 140**, stress **106.398 @ 82**, browser **2.142 @ 75**, Freeze `0b9df4f1…`. Nach dem
Slice (sqrt-Kernel + `Omit`-Fix):

| Korpus | Baseline | Nachher | Δ | Mechanismus |
|---|---|---|---|---|
| check:diag root | 206.854 @ 140 | **206.850 @ 140** | **−4** | sqrt-Typkosten − Omit-Ersparnis |
| check:diag:stress | 106.398 @ 82 | **106.239 @ 82** | −159 | Omit-Ersparnis (sqrt +0) |
| check:diag:browser | 2.142 @ 75 | 2.142 @ 75 | 0 | kompiliert threaded.ts nicht (M5) |
| bench:editor (8 Pins) | alt | **−159 uniform** | −159 | Omit-Ersparnis (sqrt-Member +0) |

Neuer Root-Pin: **206.850 @ 140**. Neue bench:editor-Pins: `{w1 27745, w2 29554, w3 60694,
w4 27908, w5 33199, w6 34369, w7 26917, w8 34784}`. Neuer Freeze-Hash (stable):
`24a048c767f3949ad0a8747cecccc0e25e25bdad859c5deb45e218a39d70cea2`. Der Freeze-Beweis
dekomponiert (M4): Pre-Edit-Clean-Rebuild reproduziert `0b9df4f1…` (Baustein A hat das selbst
nachgebaut), der Diff an abi.rs/mod.rs ist rein additiv, matmul_blocked.rs/shape.rs byte-unberührt,
neuer Hash = Pin. Threads-Hash `9743338d…` ist bewusst KEIN persistierter Pin (test:threaded
beweist seine Bit-Identität zum stable Core).

## Pflicht-Mutant

`|x| x.sqrt()` → `|x| x` im Kernel → cargo 5/8 rot + der committete JS-Differentialtest rot →
Revert per Backup-Kopie (SHA-Beweis), erneuter Lauf grün. Baustein A fing zusätzlich einen eigenen
Mutant (fast-path-Guard `offset==0` → `offset!=0`, gefangen von `sqrt_offset_window`), Baustein B
fuhr sechs Mutantenklassen (alle verhaltensändernden gefangen; M4 äquivalent, M6 unerreichbar).

## Post-Verification-Addendum

Verify-Runde Stufe 3, drei Fresh-Context-Verifier (A/B je isolierter Worktree + Slice-Patch, C
read-only). **Alle drei grün, kein Blocker/Major.**

- **Baustein A (Spec + alle Gates frisch + eigener Mutant) — CONFIRMED.** D1–D10 einzeln gegen den
  echten Code konform; alle Gates mit exakt den obigen Zahlen selbst reproduziert; Freeze-Beweis
  selbst nachgebaut (Build @ 4edea57 → `0b9df4f1…`); eigener Mutant gefangen + revertiert; zwei
  adversariale Extras (komponierte View, reiner Spezialwert-Buffer) bit-identisch. Befunde nur
  minor/nit: D8-Doc-Platzierung stand aus (dieses Doc), Graph-Staleness (behoben — neu gebaut,
  Lint 0 auf dem frischen Graph).
- **Baustein B (adversarial) — kein Blocker.** M1 gegen einen unabhängigen BigInt-Oracle über
  102.281 Elemente 0 Abweichungen; D10-`Omit`-Fix vierfach als laufzeit-identisch belegt;
  −159-Re-Pin selbst reproduziert (Freeze-Hash stabil über zwei Clean-Rebuilds); Typ-Fläche
  unberührt. Ein niedriger Befund (M6, s. u.).
- **Baustein C (covenant-verify) — keine Verstöße.** M1 korrekt als bindend eingeordnet (Kernel
  rein skalar, kein FMA/SIMD), M4 (additiv, byte-unberührte Rust-Freeze-Dateien, Re-Pin
  strukturell sauber), M5 (threaded.ts-Cast laufzeit-identisch, ndarray.ts unberührt, kein neuer
  node-Import), S1/M2/M3/Z1/Z2 sauber. Zwei niedrige Befunde (C-1, C-2, s. u.).

**Adressierte Befunde:**
- **C-2 (threaded-Spezialwert-Abdeckung) — GESCHLOSSEN in dieser Scheibe:** die threaded-sqrt-
  Paritätstests nutzten nur reguläre Zufallswerte; ergänzt um zwei `genDataSpecial`-Fälle
  (contiguous + View) → die „bit-identisch inkl. IEEE-Spezialwerte"-Aussage ist jetzt direkt auf dem
  threaded Artefakt belegt (test:threaded 73 → 75, check:diag unverändert 206.850). Setzt das
  Muster für die Kampagne.
- **C-1 (M1-NaN-Payload-Vertragslücke) → FOLLOWUPS (v6-Kandidat):** COVENANT.md v5s M1-Wortlaut
  nennt den NaN-als-Klasse-Vorbehalt (auf den sqrt sich beruft) nicht wörtlich — vorbestehend (gilt
  seit Kern 07 für div/mul), aber im v6-Bündel bisher nicht gelistet. Owner-Entscheidung, kein
  Blocker.
- **B/M6 (unerreichbarer abi.rs-Output-Längen-Guard) → FOLLOWUPS-Notiz:** der
  `out_data.len() != out_len`-Zweig in `nt_sqrt_strided` ist über die öffentliche API unerreichbar
  (`outLen` immer korrekt vom Aufrufer berechnet); fail-loud (WASM-Trap) statt stiller Korruption
  falls er je feuerte. Niedrig, kein Blocker.
