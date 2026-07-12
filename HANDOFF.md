# Handoff — 2026-07-12, Session-Ende (Phase C komplett: Scoping + Item 10)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch
Rust/WASM-Kerne). Remote `github.com/crankfunk/numtype`, privat. **Phasen A, B und C sind
komplett.** Item 10 (Backend-Wahl-API) gerade abgeschlossen, zweifach post-verifiziert,
committet & gepusht (5b0f951). `main` in Sync mit origin; Tree sauber (bis auf diesen
Handoff-Doku-Commit). Alle Gates grün: `pnpm check` (Verbund) exit 0 · `test:core` 817 ·
`test:resident` 4279 (fail 0) · `test:threaded` 69 · `cargo` 161 · `demo` all-agree ·
Artefakt-Hash `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`
byte-identisch · `check:diag` 175.634 @ 132 / stress 103.882 @ 82. **Nächstes: Phase D
(Paketierung/Release).**

## In dieser Session erledigt
- **Phase-C-Scoping** (Commit 40c2bfd): Items 8/9 (Browser-Threads-Port / stable-no_std)
  nach belegter Constraint-Recherche als **Node-only/experimentelles Opt-in zurückgestellt**
  — es gibt HEUTE keinen Weg vom pinned nightly weg (build-std nightly-only, RFCs 3874/3875
  decken atomics-Rebuilds nicht ab, wasm32-wasip1-threads Sackgasse, no_std vermutlich nicht),
  und der Browser-Port ist durch COOP/COEP-Deployment-Friction wertbegrenzt. Erfüllt das
  Release-Gate. docs/phase-c-threads-scoping.md.
- **Neue verbindliche Work-Ethic** (Commit 3bce153): „Spec-Verifikation VOR der
  Implementierung" — ein adversarialer `brainroute:deep`-Verifier gegen jede bindende Spec
  BEVOR Code entsteht (CLAUDE.md QA-Sektion + verify-runde-template.md **Baustein 0**).
- **Item 10 — Backend-Wahl-API** (Commit 5b0f951): `NDArray.backend("wasm"|"threaded")` →
  `WasmBackend`/`ThreadedBackend` als explizites, **browser-sicheres** Opt-in-Backend
  (empirisch bewiesen via `process.moduleLoadList`-Trace: der JS-`NDArray`-Default zieht
  `threaded.ts`s statische node-Imports nie eager); JS-`NDArray` bleibt Default; **null Rust,
  Hash byte-identisch**. Zweifach post-verifiziert (Baustein A + B). docs/item-10-backend-api-*.
- **Handoff-Doku** (dieser Commit): README/roadmap/CLAUDE.md auf „Phase C komplett, Item 10
  done"; Pins nachgezogen.

## Offen / in Arbeit
Nichts halbfertig — Item 10 ist vollständig (DoD durch inkl. KB-Capture). Bewusst deferred in
`FOLLOWUPS.md`: no_std-30-Min-Experiment, Browser-Threads-Port (nur bei realer Nachfrage),
Browser-Smoke-Test des Standard-Surface (Phase D), `NDArrayView`-Konformität auf WNDArray
(Spike-05-Followup, mit dem konkreten `strides`-Feld-vs-Methode-Blocker), `Backend.from`-Kür,
`test:resident`-Test-Timeout, `unravel_into`-Generalisierung (größter offener Perf-Hebel),
Union-Guard-Soundness-Gap (MAJOR).

## Nächste Schritte
1. **Phase D — Paketierung & Release** (roadmap Items 11–14): aus `spike/` ein Paket mit EINEM
   öffentlichen Surface (`exports`-Map, `.wasm`-Bundling, `d.ts`-Hover-Qualität prüfen — die
   Hovers sind Teil des Produkts). Dazu der **Browser-Smoke-Test** des Standard-Surface
   (COOP-frei, unabhängig von der Threads-Frage) und **CI** mit allen Gates inkl.
   Freeze-Hash-Check + `bench:editor`-Latenz-Gate.
2. Alternativ vorab: der **Union-Guard-Soundness-Gap** (MAJOR, vorbestehend — `NDArray<[2,3] |
   [2,3,4]>.sum(2)` typt still & falsch; eigene Scheibe) oder der Perf-Hebel
   **`unravel_into`-Generalisierung** (eigene Scheibe MIT eigener Messung vor der
   Freeze-Zeremonie).

## Bekannte Probleme / Stolperfallen
- **Spec-Verify VOR Impl ist jetzt Pflicht** (Baustein 0) — hat sich in Item 10 sofort
  bezahlt gemacht (3 Blocker vor dem Bau, u. a. `WNDArray.strides` ist ein FELD, keine
  Methode). **Zwei-Verifier-Regel** (Spec + adversarial) gilt weiter Post-Impl.
- **Delegierte Agenten enden oft auf `git status` statt dem Report** — im Delegations-Prompt
  „Der Report ist deine ALLERLETZTE Nachricht, git-status DAVOR" explizit vorgeben (verifiziert
  wirksam). **Subagenten spawnen KEINE Subagenten** — nur der Haupt-Loop fannt aus.
- **Freeze** = Ganz-Artefakt-Clean-Rebuild-Hash. Item 10 war null Rust → Hash unverändert
  (`0b9df4f1…`).
- **check:diag ist check-order-abhängig** (±~2.000 Rauschen bei Datei-Adds); Attribution via
  empty-then-fill / Ablation. Item-10-Befund: `check:diag:stress`-Anstieg dominant durch
  `threaded.ts`-Generics (D1 zieht Backend-Typen in jede NDArray-importierende Datei), nicht
  `ambient.d.ts`.
- **Mess-Hausregel**: Baselines/Pins nur im frischen `git worktree` des Zielcommits, immer
  Exit-Code prüfen (zsh: `${pipestatus}`, nicht `${PIPESTATUS}`). Hintergrund-Agenten fassen
  den Haupt-Tree nie an.
- **Threads-Build**: pinned nightly-2026-07-09 + `-Z build-std` + rust-src (`test:threaded`,
  `scripts/build-wasm-threads.sh`); env-RUSTFLAGS ersetzt config (`+simd128` mitführen).
- **Kommunikation mit dem Owner auf Deutsch**; Code/README/Spec-Docs bleiben Englisch.

## Wichtige Dateien & Befehle
- **Item 10:** `spike/src/wasm/backend-api.ts` (`WasmBackend`, `checkThreadedEnv`),
  `spike/src/wasm/threaded.ts` (`ThreadedBackend`), `spike/src/ndarray.ts` (`static backend`),
  `spike/src/index.ts` (Exports). Tests: `spike/tests-runtime/backend-api.test.ts` (test:resident,
  inkl. M3 D2-ordering Browser-Sicherheit), `…/backend-api-threaded.test.ts` (test:threaded,
  inkl. M6 dispose-Plateau). Docs: `docs/item-10-backend-api-{spec,ergebnisse}.md`.
- **Prozess:** `docs/verify-runde-template.md` (Bausteine 0/A/B) · `docs/roadmap.md`
  (Phasen; C komplett) · `docs/phase-c-threads-scoping.md` · `FOLLOWUPS.md` (Backlog).
- **Befehle (alle vom Repo-Root):** `pnpm check` (Verbund root+stress) · `check:diag` 175.634
  @ 132 / `check:diag:stress` 103.882 @ 82 · `test:core` 817 · `test:resident` 4279
  (+`:gc`) · `test:threaded` 69 (nightly) · `demo` · `cargo test --manifest-path
  crates/core/Cargo.toml` (161). Freeze-Check: `shasum -a 256 spike/src/wasm/numtype_core.wasm`
  = `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`.
