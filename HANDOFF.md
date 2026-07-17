# Handoff — 2026-07-17, Session-Ende (Item 11 KOMPLETT: API-Schnitt + Paketierung)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch
Rust/WASM-Kerne). Remote `github.com/crankfunk/numtype`, privat. **Item 11 (API-Schnitt +
Paketierung) ist mit allen drei Sub-Scheiben KOMPLETT, je dreifach verifiziert, committet
& lokal** (HEAD = 69ab47a; noch NICHT gepusht zum Zeitpunkt des Handoff-Schreibens —
Push-Frage steht am Session-Ende). Aus dem Forschungs-`spike/` wird jetzt ein
publizierbares npm-Paket nach `dist/` gebaut (`pnpm build:dist`), mit geprüften Gates statt
Behauptungen. Covenant-Regime aktiv (COVENANT.md **v4**). Alle Gates grün:
`pnpm check` (Dreier-Verbund) · test:core 820 · test:resident 4280 · test:package 3/3 ·
test:browser 4 · test:threaded 69 · cargo 161 · demo · covenant-lint 0/0 ·
Artefakt-Hash `0b9df4f1…519c7d` byte-identisch. Pins: check:diag **186,691 @ 134** /
stress **102,877 @ 82** / browser **2,142 @ 75**.

## In dieser Session erledigt
Item 11 als drei Sub-Scheiben (Muster Spec → Baustein 0 → Impl → A+B+C-Verify → Doku →
Commit → KB), jede mit voller Drei-Verifier-Runde unter dem Covenant-Regime:

- **S1 (48ee440) — Typ-Vorarbeiten.** `sum`-Overload-Umbau schließt den COVENANT-M2-Verstoß
  (`Literal|undefined` via optionale Parameter), **beide Facetten** (axis→no-claim,
  keepdims→ehrliche Shape-Union) über Overloads nach Argument-Anzahl + `reduce.ts`-KeepDims
  auf `boolean|undefined`; Impl-Rückgabe `NDArray<any>` (owner-bestätigte Abweichung).
  `slice-literal.ts`→`literal-arithmetic.ts` umbenannt. Machbarkeit vorab per Spike geklärt
  (M3 heilig). COVENANT v2→v3. docs/item-11-s1-*.
- **S2 (87e6e6b) — Emit-/Paket-Pipeline.** `tsconfig.build.json` (dist-Emit) + zero-dep
  `scripts/postbuild-dist.mjs` (Post-Emit-Rewrite, fixt die drei TS7-Emit-Blocker; jetzt
  kommentar-bewusst) + `.wasm`-Bundling + package.json-Metadaten (exports/main/module/types/
  files/sideEffects). `pnpm build:dist`. docs/item-11-s2-*.
- **S3 (69ab47a) — Zero-dep-Guard + Paket-Smoke.** Paketierungs-Claims als Gates:
  `zero-dep-guard.test.ts` (Z1, prüft dependencies/optional/peer), `package-smoke.test.ts`
  (Laufzeit gegen `dist/index.js` inkl. `backend("wasm")`-echtem-WASM-Ladepfad),
  `consumer/` (Typ-Smoke gegen `dist/index.d.ts`), `check-dist-emit.mjs` (Emit-Präzisions-
  Gate, die EINZIGE Blocker-1-Absicherung). `pnpm test:package`. COVENANT v3→v4 (Z2
  präzisiert: build-gated Typchecks laufen in test:package). docs/item-11-s3-*.

**Wichtige Verify-Funde (alle behoben):** S3-F1 (HIGH) — der Laufzeit-Smoke prüfte den
WASM-Ladepfad NICHT (JS-`NDArray` rechnet in reinem JS; WASM nur via `backend("wasm")`) →
`backend("wasm")`-Test ergänzt, nicht-vakuös bewiesen. S3-F2 — Zero-dep-Guard erweitert um
optional/peerDependencies. S2 — zwei Verpackungs-Löcher (kein `rm -rf dist`, Prosa-Rewrite)
behoben.

## Offen / in Arbeit
Nichts halbfertig. Bewusst zurückgestellt (in FOLLOWUPS.md, für Item 12/13):
- **npm-Name `numtype` sichern** (Registry frei 2026-07-17), **`author`-Feld**, **LICENSE-
  Datei**, **`private:false`** — alle bewusst Item 13 (Release-Mechanik).
- **README-Vollüberarbeitung** (Install/Usage/Package-Info) — Item 13; nur der eine
  „Planned backend"-Satz wurde beim Handoff korrigiert.
- F3 (check-dist-emit multi-line, latent), F6 (test:package Test-Timeout) — Item 12.

## Nächste Schritte
1. **Item 12 — Qualitäts-Portfolio + CI:** GitHub Actions mit ALLEN Gates (check-Verbund,
   test:core/resident/package/browser/threaded, cargo, **Artefakt-Hash-Freeze-Check**,
   **bench:editor als Gate**, **Test-Timeouts** — FOLLOWUPS F6). Ggf. Vitest-Migration
   (FOLLOWUPS). test:threaded braucht die gepinnte nightly-2026-07-09 (+rust-src).
2. **Item 13 — Release-Mechanik:** npm-Name sichern (Owner-Aktion, irreversibel),
   `author`/LICENSE/`private:false`, README mit Install/Usage/Demo, 0.x-SemVer-Politik.
3. Danach Item 14 (v0.1 research preview).

## Bekannte Probleme / Stolperfallen
- **Pins nie über Korpora/File-Set-Änderungen vergleichen**; Datei-ADDITIONEN tragen
  ±~2,000 Order-Noise. Messungen nur im frischen `git worktree`, Exit-Code prüfen.
- **`pnpm build:dist` braucht keine nightly** (stable-wasm); `dist/` ist gitignored, kommt
  via `files:["dist"]` in den Tarball (npm-Fallstrick verifiziert: greift nicht).
- **Der Default-`NDArray`-Pfad rechnet in reinem JS** — das gebündelte WASM lädt NUR über
  `backend("wasm")`/`"threaded")` (M5-Browser-Sicherheit). Ein Paket-Smoke muss den WASM-
  Pfad explizit ansteuern (S3-F1-Lektion).
- **Parallele Verifier IMMER mit eindeutigen Worktree-Pfaden** (PID/Timestamp) beauftragen —
  generische Pfade führten in der S1-Runde zu Verifier-Kontamination (KB-Lektion, seit S2
  angewandt).
- **Covenant-Regime** (CLAUDE.md „Qualitätssicherung"): substanzielle Scheiben = voller
  Katalog A+B+C parallel + Lint im Gate-Block; Specs/Delegations-Prompts benennen berührte
  Invarianten-IDs; Spec/COVENANT-Änderungen nur mit Owner-Bestätigung + Version-Bump.
- **test-scripts-guard Basename-Kollision** (S3-F4): tests-package-Testdateinamen müssen
  global eindeutig über ALLE Korpora sein (darum `package-smoke.test.ts`, nicht `smoke.`).
- Threads-Artefakt: gepinnte nightly-2026-07-09 (+rust-src); Freeze-Beweis = Hash aus
  Clean-Rebuild; alle Kommandos vom Repo-Root (cargo-Config-Discovery).

## Wichtige Dateien & Befehle
- Spec/Prozess: `COVENANT.md` (v4) · `docs/verify-runde-template.md` (Bausteine 0/A/B/C)
  · `FOLLOWUPS.md` · `CLAUDE.md` (Pins, Kommandos, QA-Regeln) · `docs/roadmap.md`.
- Diese Session: `docs/item-11-api-paket-spec.md` (bindende Spec) +
  `docs/item-11-s{1,2,3}-ergebnisse.md`.
- Paket-Bau: `tsconfig.build.json` · `scripts/postbuild-dist.mjs` · `scripts/check-dist-emit.mjs`
  · `spike/tests-package/` (Smoke + consumer) · `spike/tests-runtime/zero-dep-guard.test.ts`.
- Kern-Kommandos: `pnpm check` (Dreier-Verbund) · `pnpm check:diag[:stress|:browser]` ·
  `pnpm build:dist` · `pnpm test:core|test:resident|test:package|test:browser|test:threaded`
  · `cargo test --manifest-path crates/core/Cargo.toml` · `pnpm demo` · `pnpm bench:editor`
  · `graph-a-lama . --symbols && graph-a-lama query lint`.
