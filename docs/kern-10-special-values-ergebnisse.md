# NumType — Kern 10: Spezialwerte im Differential-Generator — Ergebnisse

*2026-07-12. Spec: docs/kern-10-special-values-spec.md (inkl. Korrektur-Block).
Schließt Phase B Item 6 ab. Implementiert von `deep` (Sonnet 5 xhigh), zweifach
verifiziert (Spec + adversarial).*

## Summary

Der Bit-Identitäts-Anspruch zwischen naiver TS-Referenz und WASM-Kerneln war nur
für **normale endliche Werte** belegt (der splitmix64-PRNG erzeugt konstruktiv
keine Spezialwerte). Diese Scheibe injiziert IEEE-754-Spezialwerte (NaN, ±Inf,
±0, Subnormals, ±MAX_VALUE) in den Differential-Generator und belegt Bit-
Identität über alle Whitelist-Ops × drei Surfaces. **Test-only: null Rust-
Änderungen, Artefakt-Hash byte-identisch.** Der Schlüsselbefund — **Subnormal-
Erhalt durch den SIMD-blocked-matmul HÄLT** (kein Flush-to-Zero) — ist per
Mutation direkt als fangbar bewiesen.

## What was built

- **`prng.ts`** (append-only, bestehende Generatoren byte-identisch):
  `SPECIAL_VALUES` (10 Klassen), `nextF64Special(rng, specialProb=0.35)`,
  `genDataSpecial(rng, shape, specialProb)`. Deterministisch (rng-gespeist, kein
  `Math.random`).
- **`assert-helpers.ts`**: `bitsOf` von privat auf `export` gehoben (für die
  Payload-Bit-Fixture nötig — siehe Fund 3) + NaN-Payload-Doc-Klarstellung.
  `assertDataBitIdentical` unverändert (weiter `Object.is`).
- **`special-values.test.ts`** (neu, 619 Fälle, in `test:resident`-Liste): pro
  Op ein Zufallspass mit `genDataSpecial` + kuratierte Fixtures. Abdeckung:
  add (3-Wege), sub/mul/div (2-Wege, kein v1-Kernel), sum all+axis (3-Wege),
  matmul n∈{2,3,8,16,32} (3-Wege), dot/norm/cosine (2-Wege skalar), transpose
  (3-Wege + Movement-Payload-Fixture). Plus 2 Generator-Nicht-Vakuitätstests.

## Der Schlüsselbefund: SIMD-Subnormal-Erhalt hält

WASM-SIMD128 hat kein Flush-to-Zero; der blocked+SIMD-matmul MUSS Subnormals
erhalten. Die kuratierte 3×2 @ 2×3-Fixture ist so dimensioniert, dass `n=3`
sowohl den `f64x2 accumulate_pair`-SIMD-Schritt ALS AUCH den skalaren
`accumulate_single`-Tail im selben Call trifft (per Quelllektüre
matmul_blocked.rs verifiziert, nicht angenommen); alle Ein-/Ausgänge liegen tief
im Subnormalbereich (< 2.2250738585072014e-308). Referenz⇄v1⇄resident
byte-identisch, jedes Subnormal überlebt ungeflusht. **Beide Verifier haben den
Ausfallmodus per Mutation nachgestellt** (SIMD `accumulate_pair` flusht
Subnormal→0) → Fixture + 2 Zufallsfälle werden rot. Der wertvollste neue Test
ist nicht-vakuum.

## Offengelegte Spec-Korrektur (grounded-Verhalten des Executors)

Die ursprüngliche D2-Fixture behauptete `sum([-0,-0,-0]) = -0`. **Faktisch falsch
für diese Codebase**: `sumRuntime` seedet `let total = 0` (`+0`, frozen), IEEE
`+0 + -0 = +0` → ein reiner `-0`-Array summiert IMMER zu `+0`; `-0` ist durch
diese Reduktion unerreichbar. Der `deep`-Executor hat das per `node -e` bewiesen,
die Fixture auf den WAHREN Wert (`+0`) korrigiert und den Spec-Fehler gemeldet
statt der falschen Spec zu folgen — exakt das gewünschte Verhalten. Spec ist
korrigiert (Korrektur-Block). Beide Verifier haben die Herleitung unabhängig
bestätigt.

## Gates (Implementierer + beide Verifier unabhängig gemessen)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Verbund) | clean, Exit 0 |
| `pnpm test:core` | 817 / 0 (unverändert) |
| `pnpm test:resident` | **4265 pass + 2 skipped** / 0 fail (Basis 3646 + 619 neue) |
| `cargo test` | 157 (null Rust-Änderungen) |
| Artefakt-Hash | `7a65d800…` **byte-identisch** (adversarial: auch nach clean `pnpm build:wasm`) |
| `pnpm check:diag` | **174.391 @ 127 Files** (neuer Haupt-Pin) |
| `pnpm check:diag:stress` | 94.597 (unverändert — Datei nicht in dieser tsconfig) |

**Pin-Neusetzung (Spec-Gate-Erwartung):** Haupt-Pin **174.213 → 174.391 = Δ+178**
für EINE test-only-`.ts`-Datei (reiner Wertcode, keine Typmaschinerie). Basis ist
die Kern-09-committete 174.213, NICHT die ältere Infra-01-173.716 (der
adversariale Verifier verglich versehentlich gegen letztere und meldete Δ+675 —
Bookkeeping, kein Korrektheitsproblem). Messung nach der Mess-Hausregel (frisch,
Exit-Code, Files-Count 127 = 126 + die eine neue Datei). ~3,5 % des Budgets.

## Post-verification addendum (Zwei-Verifier-Runde)

**Spec-Verifier (Baustein A): CONFIRMED.** D1–D5 alle PASS. Drei Handrechnungen
unabhängig bestätigt (±0/+0 via `node -e`; Subnormal-matmul-Ausgabe per Hand
nachgerechnet, alle Nicht-Null < 2.2e-308; nicht-kanonischer NaN-Payload
`0x7ff80000deadbeef` ≠ kanonisch, per `bitsOf` geprüft). Routing per Quelllektüre
bestätigt (v1 skalar, resident immer SIMD, keine Schwelle). Alle Gates exakt.
Eigener Mutant (SIMD-Flush in `matmul_blocked.rs`, clean rebuild): 3/619 rot
(Fixture + 2 Zufallsfälle) — direkt der Zielausfallmodus.

**Adversarialer Verifier (Baustein B): HÄLT** mit 5 offengelegten Funden. Zwei
weitere echte Mutanten gefangen (SIMD-Flush: 3 rot; elementwise-±0→+0: 52 rot).
Generator-Nicht-Vakuität am REALEN 0.35-Pfad empirisch bestätigt (jede der 10
Klassen erscheint bei jedem Op zig- bis hundertfach; Subnormals/-0 inklusive) —
nicht nur am `specialProb=1`-Test. Inf/NaN-Propagation nicht-vakuum (34/40
matmul-Fälle erzeugen ≥1 NaN, alle Surfaces bit-gleich). Freeze/Messung
bestätigt.

**Funde und ihre Behandlung:**

1. **Mittel — Coverage-Ehrlichkeit (adressiert, dokumentiert):** Die neuen
   `dot`/`cosine`-Zufallspässe (kleines n ≤24/≤12) haben fast keine Kraft gegen
   **Akkumulationsreihenfolge**-Bugs: bei 35 % Injektion maskiert Inf/NaN-
   Dominanz die rundungssensitiven Normalwerte (rückwärts-Akkumulation → nur
   1/619 rot vs. 134/230 in `vector.test.ts`). **Kein Bug** — Reihenfolge ist
   durch `vector.test.ts` (Normaldaten, größeres n) gedeckt; die neuen Pässe
   belegen **Propagation** (NaN/±Inf/±0), nicht **Ordnung**. Ehrlich benannt:
   in-code-Kommentar an der dot-Sektion + hier; NICHT als „Ordnung abgedeckt"
   verbucht.
2. **Niedrig — Pin-Bookkeeping (adressiert):** Neu gepinnt 174.391 (s. o.),
   CLAUDE.md nachgezogen.
3. **Niedrig — `bitsOf`-Export undokumentiert (adressiert):** Der Export war für
   die D3-Payload-Bit-Fixture nötig (die Bits vergleicht, nicht `Object.is`);
   reine Sichtbarkeitserweiterung, keine Semantikänderung. Hiermit offengelegt.
4. **Niedrig-Mittel — NaN-Payload-Erhalt nur für transpose regressionsgetestet
   (→ FOLLOWUPS):** empirisch hält byte-exakter Payload-Erhalt auch durch
   reshape/slice/fromArray→toArray/wasmAdd auf dieser Engine, ist aber
   scope-konform nur für transpose verankert. Ein künftiger Toolchain-Wechsel,
   der bei einer dieser Ops NaN-Bits kanonisiert, hätte keinen Test. FOLLOWUPS-
   Item.
5. **Sehr niedrig — size-0 + Spezialwert ungetestet (→ FOLLOWUPS-Notiz):**
   größtenteils gegenstandslos (leeres Array trägt keine Werte); nur ein
   Broadcast-Pairing size-0 × nicht-leer-special wäre interessant, wird von
   keinem Generator erzeugt.

## Honesty-Notizen

- Fund 1 ist die wertvollste Einsicht der Runde: die neuen dot/cosine-Pässe sind
  gegen Ordnungs-Bugs schwach (nicht vakuum, aber schwach) — ehrlich benannt
  statt als volle Abdeckung verbucht. Genau der Fall, für den der adversariale
  Verifier existiert.
- Der ±0-Spec-Fehler war MEINER (Orchestrator); der Executor hat ihn gefangen.
  Das ist das Prozessziel (grounded, Spec ist nicht sakrosankt), nicht ein
  Executor-Fehler.
- Der byte-exakte NaN-Payload-Claim gilt nur für Movement-Ops und nur für
  transpose als committeter Test; die WASM-Spec garantiert Payloads für
  Arithmetik NICHT (daher `Object.is`, Werteklasse — siehe
  [[bit-identische-differentialtests-zwischen-implementierungen]]).
