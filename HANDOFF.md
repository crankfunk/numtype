# Handoff — 2026-07-18, Session-Ende (Item 12 KOMPLETT: Qualitäts-Portfolio + CI)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch
Rust/WASM-Kerne). Remote `github.com/crankfunk/numtype`, privat. **Item 12 (Qualitäts-
Portfolio + CI) ist KOMPLETT, dreifach verifiziert, committet + gepusht UND end-to-end in
CI bestätigt** (HEAD = 8b1623e). Es gibt jetzt eine **8-Job-GitHub-Actions-CI**, die alle
lokalen Gates bei jedem Push/PR maschinell reproduziert — das letzte harte Release-Gate der
Roadmap. Covenant-Regime aktiv (COVENANT.md **v4**, unverändert). Alle Gates grün, lokal UND
in CI (dritter Lauf 8/8 success): `pnpm check` (Dreier-Verbund) · test:core **822** ·
test:resident 4280 · test:package 3/3 · test:browser 4 · test:threaded 69 · cargo **161+1**
(inkl. zero_alloc) · demo · covenant-lint 0/0 · Artefakt-Hash `0b9df4f1…519c7d`
**cross-host bestätigt** (Linux == macOS). Pins: check:diag **187.918 @ 135** / stress
102.877 @ 82 / browser 2.142 @ 75.

## In dieser Session erledigt
Item 12 nach vollem Covenant-Prozess (Spec → Baustein 0 → Impl → A+B+C-Verify → Doku → KB →
Commit → Push → 3 CI-Läufe). **Vier Commits:**

- **11beead — Item 12 Impl.** `.github/workflows/ci.yml` (8 Jobs: check/cargo/test-node/
  test-browser/test-threaded/freeze/editor-gate/demo, `on: push:[main]+pull_request`, keine
  Doppelläufe); `rust-toolchain.toml` (rustc **1.95.0** gepinnt → reproduzierbarer Freeze-
  Hash); `.nvmrc` (Node 24); `scripts/check-freeze-hash.mjs` (Freeze-Gate mit Plattform-Hash-
  MENGE, zero-dep, auch lokal als `pnpm check:freeze`); `bench:editor` zu HARTEM Gate gehärtet
  (`enforceHardGate`: Correctness + W1–W7-Instantiation-Pins exakt, Latenz 2x-Ceiling,
  Exit-Code); `--test-timeout=120000` (F6 geschlossen); zero-dep **S1-Import-Guard**
  (`spike/tests-runtime/s1-import-guard.test.ts`, string-aware Text-Scanner). **Kein Vitest**
  (node --test bleibt, Owner-Entscheid).
- **e625593 — CI-Fix 1 (test:threaded).** Erster CI-Lauf 7/8: `test:threaded` baute nur das
  threads-`.wasm`, die Tests brauchen aber auch das STABLE-`.wasm` (Bit-Identitäts-Vergleich).
  Fix: `test:threaded` baut jetzt `build:wasm && build:wasm:threads` (selbst-genügsam), Job
  installiert stable-Toolchain (`rustup show`).
- **d8e76d7 — CI-Fix 2 (zero_alloc).** Zweiter CI-Lauf 7/8: pre-existing Kern-06-Test
  `crates/core/tests/zero_alloc.rs` flaky auf CI (globaler `#[global_allocator]`-Counter zählt
  transiente Harness-Allokationen anderer Threads mit, delta=4 statt 0; lokal deterministisch
  grün). Owner-entschiedener Fix: **min-delta über N=16 Messungen** — inhärente Allokation wäre
  in JEDER Messung, transiente Kontamination nur manchmal (min filtert sie raus); Beweis bleibt
  exakt, wird plattform-robust.
- **8b1623e — finaler Doc.** Dritter CI-Lauf **8/8 grün** → Item 12 end-to-end CI-verifiziert.

**Verify-Runde (Owner-Mandat):** A CONFORM · B fand+behob ZWEI S1-Guard-Bypässe (mehrzeilige
Import-Formen `import(\n"x")` + `from\n"x"` umgingen den zeilenweisen Scanner — meine
Selbst-Checks verpassten das; Fix = string-aware Text-Scanner) · C kein Verstoß. **Freeze-
Hash cross-host GEKLÄRT:** ubuntu-x64 baut byte-identisch zu macOS-arm64 (kein wasm-opt).
KB: 2 neue Notizen (string-aware-Scanner, erster-CI-Lauf-deckt-auf).

## Offen / in Arbeit
Nichts halbfertig — Item 12 vollständig abgeschlossen. Kleine FOLLOWUPS (alle niedrig/trivial,
in FOLLOWUPS.md): CI `paths-ignore` für reine Doku-Commits; Freeze-Hash-Pin-Label auf
`macos-arm64 + linux-x64` erweitern (Wert gilt schon für beide); nightly-Version mechanisch
synchron halten (ci.yml ⇄ build-wasm-threads.sh, nur per Kommentar). Bewusst zurückgestellt
(Item 13): npm-Name `numtype` sichern (Owner-Aktion, irreversibel), `author`-Feld, LICENSE-
Datei, `private:false`, README-Vollüberarbeitung.

## Nächste Schritte
1. **Item 13 — Release-Mechanik:** npm-Name `numtype` sichern (Owner-Aktion, irreversibel;
   Registry frei Stand 2026-07-17), `author`/LICENSE/`private:false`, README mit Install/
   Usage/10-Sekunden-Demo, 0.x-SemVer-Politik, Begleit-Blog-Post; Forschungsnotizen als
   publizierbare Artefakte (USP-Doc §8.3).
2. Danach **Item 14 — v0.1 research preview.**

## Bekannte Probleme / Stolperfallen
- **`pnpm test:threaded` baut jetzt BEIDE Artefakte** (`build:wasm && build:wasm:threads`) —
  braucht stable 1.95.0 UND die pinned nightly-2026-07-09 (+rust-src). Die threaded-Tests
  vergleichen bit-identisch gegen den STABLE-Core, nicht nur den threads-Core.
- **Freeze-Hash ist cross-host byte-stabil** (Linux == macOS, empirisch im ersten CI-Lauf
  bestätigt) — die Plattform-Hash-Menge in `check-freeze-hash.mjs` braucht vorerst nur EINEN
  Eintrag. Falls je wasm-opt/binaryen eingeführt wird, bricht das (dann Menge erweitern).
- **`crates/core/tests/zero_alloc.rs` war CI-flaky** — jetzt min-delta-robust. Wer diesen Test
  anfasst: die N=16-Schleife nicht entfernen (die transiente Harness-Kontamination kommt zurück).
- **CI läuft bei JEDEM Push** (auch reine Doku) — paths-ignore-FOLLOWUP offen. Ein Doku-Commit
  triggert den vollen 8-Job-Lauf inkl. nightly-build-std.
- **`.nvmrc` „24"** löst LOKAL eine harmlose nvm-Warnung aus (kein `24`-Alias installiert); in
  CI korrekt (setup-node löst „24" → neueste 24.x). Nicht beunruhigen lassen.
- **Pins nie über Korpora/File-Set-Änderungen vergleichen**; Datei-ADDITIONEN tragen ±~2.000
  Order-Noise. Messungen nur im frischen `git worktree`, Exit-Code prüfen.
- **Covenant-Regime** (CLAUDE.md „Qualitätssicherung"): substanzielle Scheiben = voller Katalog
  A+B+C parallel + Lint im Gate-Block; Specs benennen berührte Invarianten-IDs; parallele
  Verifier mit EINDEUTIGEN Worktree-Pfaden; Spec/COVENANT-Änderungen nur mit Owner-Bestätigung
  + Version-Bump.

## Wichtige Dateien & Befehle
- **CI/Infra:** `.github/workflows/ci.yml` · `scripts/check-freeze-hash.mjs` ·
  `rust-toolchain.toml` (rustc 1.95.0) · `.nvmrc` (Node 24) ·
  `spike/tests-runtime/s1-import-guard.test.ts` · `crates/core/tests/zero_alloc.rs`.
- **Spec/Prozess:** `COVENANT.md` (v4) · `docs/verify-runde-template.md` · `FOLLOWUPS.md` ·
  `CLAUDE.md` (Pins/Kommandos/QA-Regeln) · `docs/roadmap.md`.
- **Diese Session:** `docs/item-12-ci-spec.md` (bindende Spec) + `docs/item-12-ergebnisse.md`
  (inkl. §8 Verify-Addendum + §9 Erster-CI-Lauf).
- **Kern-Kommandos:** `pnpm check` (Dreier-Verbund) · `pnpm check:diag[:stress|:browser]` ·
  `pnpm check:freeze` (NEU) · `pnpm build:dist` · `pnpm test:core|resident|package|browser|threaded`
  · `cargo test --manifest-path crates/core/Cargo.toml` · `pnpm demo` · `pnpm bench:editor`
  · `graph-a-lama query lint` · CI beobachten: `gh run list` / `gh run view <id> --log-failed`.
