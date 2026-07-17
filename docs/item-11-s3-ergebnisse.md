# Item 11 / S3 — Ergebnisse (Zero-dep-Guard + Paket-Smoke)

**Stand:** erledigt & DREIfach verifiziert (Spec CONFIRMED + adversarial 1 HIGH + N kleinere
Befunde behoben + covenant-verify 5/6 + Z2-Amendment), 2026-07-17. Bindende Spec:
`docs/item-11-api-paket-spec.md` (Abschnitt S3, D-S3.1/2/3). Ausgangscommit 87e6e6b (S2).
ZERO Source-/Rust-Änderungen.

> **Ehrlichkeitsregel:** Jede Zahl in einem gelaufenen Kommando verankert. Der HIGH-Befund
> (Rauchtest prüfte kein WASM) offen berichtet + behoben.

## Was S3 gemacht hat

Macht die Paketierungs-Claims zu geprüften Gates:
- **Zero-dep-Guard** (`spike/tests-runtime/zero-dep-guard.test.ts`, in test:core): prüft, dass
  package.json KEINE `dependencies`/`optionalDependencies`/`peerDependencies` trägt (Z1).
- **Paket-Smoke, drei Tests** (`spike/tests-package/package-smoke.test.ts`, in test:package nach
  build:dist): (1) JS-`NDArray` aus `dist/index.js` rechnet; (2) `backend("wasm")` lädt das
  GEBÜNDELTE `.wasm` über den paket-relativen Loader-Pfad + rechnet; (3) `backend("threaded")`
  ohne Artefakt rejected mit dem gepinnten Stamm.
- **Typ-Smoke** (`spike/tests-package/consumer/consumer.ts` gegen `dist/index.d.ts`,
  Konsumenten-tsconfig skipLibCheck:true): Blocker 1 gefixt + M3 (`@ts-expect-error` am OOB-Arg).
- **Emit-Präzisions-Gate** (`scripts/check-dist-emit.mjs`): unabhängige, kommentar-bewusste
  Prüfung, dass keine `.ts`-Modul-Refs in dist überleben (die EINZIGE Blocker-1-Absicherung —
  der consumer-Smoke fängt das NICHT, S2-Verify A-1).
- **`test:package`-Script** + **test-scripts-guard-Invariante (e)** (tests-package nur in
  test:package).

## Gate-Ergebnisse (final, nach Fix-Einarbeitung)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Dreier) | EXIT 0 (package-smoke im Korpus; nur consumer/ excludiert) |
| `pnpm test:core` | 820 pass (inkl. zero-dep-guard + guard-e) |
| `pnpm test:package` | 3/3 (JS + WASM + threaded-rejection), check-dist-emit OK, consumer-tsc grün |
| `pnpm test:resident`/`cargo`/`demo` | 4280 / 161 / grün (unverändert) |
| Artefakt-Hash | `0b9df4f1…519c7d` byte-identisch |
| `git diff 87e6e6b -- spike/src crates/` | 0 (keine Source/Rust-Änderung) |

**Pin (NEUER Baseline):** `check:diag` **186.691 @ 134 Files** (S2: 179.986 @ 132). Δ+2 Files
(zero-dep-guard + package-smoke, beide im Root-Korpus) → File-Set-Änderung, Order-Noise-behaftet,
NICHT gegen 179.986 verrechenbar (CLAUDE.md-Konvention). `check:diag:stress` unverändert
102.877 @ 82.

## Post-Verification-Addendum (drei Fresh-Context-Verifier, eindeutige PID-Worktrees)

- **Baustein A (Spec): CONFIRMED.** D-S3.1/2/3 konform, alle Gates grün, beide Mutanten
  (dependencies-Injektion → zero-dep-guard rot; `@ts-expect-error`-Ziel entfernt → consumer-tsc
  rot) nicht-vakuös. A-1-Coverage-Konsequenz bestätigt (SOGAR breiter: auch der Laufzeit-Smoke
  fängt eine Blocker-1-Regression nicht — nur check-dist-emit). Drift: D-S3.3 „kein check:diag-
  Move" ist FALSCH (zero-dep-guard liegt in tests-runtime = Root-Korpus) → neuer Pin dokumentiert.
- **Baustein B (adversarial): 1 HIGH + 5 kleinere, alle adressiert.**
- **Baustein C (covenant-verify): 5/6 halten, Z2-Spannung → Owner-Amendment v4.**

### Adressierte Befunde

- **F1 (HIGH, B): Laufzeit-Smoke prüfte den WASM-Ladepfad NICHT.** Der Default-`NDArray` rechnet
  in reinem JS; das gebündelte `.wasm` wird nur über `backend("wasm")`→`initCore()` geladen, das
  kein Test aufrief (B bewies: `.wasm` löschen → Tests grün). **BEHOBEN:** dritter Test
  `backend("wasm")` lädt das gebündelte `.wasm` + rechnet; nicht-vakuös bewiesen (`.wasm` weg →
  dieser Test rot, JS-Pfad grün).
- **F2 (MEDIUM, B): Zero-dep-Guard prüfte nur `dependencies`.** `optionalDependencies` werden
  auto-installiert (Z1-Umgehung). **BEHOBEN:** prüft jetzt dependencies + optionalDependencies +
  peerDependencies; nicht-vakuös bewiesen (optionalDependencies injiziert → rot, benannt).
- **C Z2-Spannung + C-Teil-1 (unnötiger Mit-Ausschluss):** package-smoke.test.ts brauchte dist
  für den Typcheck NICHT (dynamischer Import) → **BEHOBEN:** nur `spike/tests-package/consumer`
  excludiert, package-smoke.test.ts zurück in `pnpm check`. Der consumer-Typ-Smoke bleibt
  strukturell aus check (braucht `dist/index.d.ts`) → **COVENANT v3→v4** (Z2 präzisiert: Build-
  gated Typchecks laufen in test:package; Owner-bestätigt).
- **F5 (LOW, B): check-dist-emit nicht kommentar-fest.** **BEHOBEN:** kommentar-bewusst (eigener
  Block-Tracker, bleibt unabhängig von der Rewriter-Match-Logik).
- **F4 (MEDIUM, false-positive, B): guard-(e) Basename-Vergleich** kollidiert mit ALLEN Korpora
  (nicht nur tests-browser) — sichere Richtung (erzwingt Umbenennung). **DOKUMENTIERT** (House-
  Rule: tests-package-Basenames global eindeutig; darum `package-smoke.test.ts`).
- **F3 (MEDIUM, latent, B): check-dist-emit ist zeilenweise** → multi-line-Import würde unentdeckt.
  tsc emittiert single-line → strukturell unmöglich im Emit. **DOKUMENTIERT** (latente Grenze).
- **A-Drift (minor): check:diag-Move** → neuer Pin 186.691 @ 134 dokumentiert.
- **Eigener Fund beim Bauen:** Basename-Kollision `smoke.test.ts` (tests-browser vs. tests-package)
  vom Guard gefangen → Umbenennung `package-smoke.test.ts`.

## FOLLOWUPS (neu)

- **F6 (LOW, pre-existing): `checkThreadedEnv()` file-I/O ohne harten Test-Timeout** — `test:package`
  erbt das (Item-10-FOLLOWUP „test:resident ohne Timeout"); kein S3-Regress, gebündelt mit dem
  CI-Timeout-Item (Item 12).
- **F3 (latent): check-dist-emit multi-line** — falls tsc je multi-line emittiert, den Scan
  text-weit (nicht zeilenweise) machen. Heute gegenstandslos.

## Nächste Schritte

**Item 11 (API-Schnitt + Paketierung) ist mit S3 KOMPLETT** (S1 Typ-Vorarbeiten, S2 Emit-/Paket-
Pipeline, S3 Zero-dep-Guard + Smoke). Nächstes: Item 12 (CI: alle Gates inkl. Freeze-Hash +
bench:editor + Test-Timeouts) und Item 13 (Release-Mechanik: LICENSE-Datei, author-Feld, npm-Name,
`private:false`, README).
