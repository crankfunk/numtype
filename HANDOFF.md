# Handoff — 2026-07-12, Session-Ende (Kern 09 + Kern 10 + Kern 11, Phase B komplett)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Remote `github.com/crankfunk/numtype`, privat. **Phase B (Minimum Viable Op-Surface) ist KOMPLETT** — alle Items 1–7 erledigt, jede Scheibe zweifach verifiziert (Spec + adversarial), committet und gepusht. Tree sauber, `main` in Sync mit origin. Gates grün: `pnpm check` (Verbund) exit 0 · `test:core` 817 · `test:resident` 4265+2 · `cargo` 161 · `demo` all-agree · Artefakt-Pin `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` · `check:diag` 172 392 @ 128 / stress 94 597.

## In dieser Session erledigt
- **Kern 09 — Runtime `keepdims` auf `sum()`** (Commit 1c046cd): beide Surfaces, ein appended Shape-Helper `keepDimsShape` (runtime.ts, append-only), `const KeepDims`-Typparam speist das vorhandene `ReduceAxis<S,Axis,KeepDims>`; Daten byte-identisch zu non-keepdims (ZERO Rust, Hash byte-identisch); 367 nicht-zirkuläre Differentialtests inkl. Views; owner-bestätigte D3-Abweichung (bestehende `sum`-Methode erweitern). docs/kern-09-*.
- **Kern 10 — Spezialwerte im Differential-Generator** (Commit e64aed6): `SPECIAL_VALUES`/`nextF64Special`/`genDataSpecial` append-only in prng.ts, neue special-values.test.ts (619 Fälle über alle Whitelist-Ops × 3 Surfaces); **Schlüsselbefund: SIMD-matmul erhält Subnormals** (per Mutation als fangbar bewiesen); TEST-ONLY, Hash byte-identisch. docs/kern-10-*.
- **Kern 11 — Contiguous elementwise Fast-Path** (Commit d02f06f): `add_strided`+`binary_strided` überspringen die per-Element-`unravel`-Allokation bei gleicher Shape/offset 0/natural strides → **13–17×**, bit-identisch (mathematisch + committete `.to_bits()`-Äquivalenztests). MESSGETRIEBEN: SIMD elementwise + Packing-A gemessen NO-GO. Neuer Artefakt-Pin `0b9df4f1…`. docs/kern-11-*.
- **check:diag-Anomalie GEPINNT** (Commit 94493dc): der Rückgang beim Datei-Hinzufügen ist **check-order-abhängig** (leere `export {}`-Datei reproduziert ihn; umbenennen ändert den Betrag; nicht-monoton) — Zerlegung −2 043 Reihenfolge / +44 echte Kosten; erklärt rückwirkend Infra 01. Konsequenz in Doku + coding-kb.
- **Prozess-Upgrade** (in der Kern-09-Session verankert): Zwei-Verifier-Regel (docs/verify-runde-template.md), CLAUDE.md-Sektion „Qualitätssicherung, modellunabhängig" (fable-doctrine laden, Graph vor Read, Worktree-Messregel, Abweichungs-Eskalation, kein Live-Tree-Griff für Agenten).
- **Doku (diese Handoff-Runde):** README-Status auf Kern 09/10/11 + „Phase B complete"; roadmap-Phase-B-Block auf KOMPLETT.

## Offen / in Arbeit
Nichts halbfertig. Alle Slices sind abgeschlossen. Offene Punkte leben bewusst deferred in FOLLOWUPS.md.

## Nächste Schritte (priorisiert)
1. **Owner-Entscheidung: Phase C vs. Phase D** (roadmap.md). Phase C = Plattform-Entscheidungen (Browser-Port des Threads-Pfads / no_std-stable, dann Backend-Wahl-API); Phase D = Paketierung/Release. Richtungsentscheidung, kein Automatismus.
2. **Größter offener Perf-Hebel: `unravel_into`-Generalisierung** (FOLLOWUPS) — der breitere Verwandte von Kern 11: `unravel` (shape.rs:105, allozierend) durch das nicht-allozierende Muster über ALLE strided Kernel ersetzen. Eigene Scheibe MIT eigener Messung VOR der Freeze-Zeremonie (general-Case-Payoff ist hergeleitet, nicht gemessen). Freeze-Warnung: die `#[cfg(atomics)]`-gegateten `_into`-Twins NICHT entgatern (Präsenz verschiebt Bytes).
3. Kür/klein aus FOLLOWUPS: NaN-Payload-Erhalt für reshape/slice/fromArray regressionstesten (Kern-10-Befund); `keepDimsShape` defensiver Achsen-Assert; die vorbestehende Union-Guard-Latenz (drei Facetten, eigene Scheibe).

## Bekannte Probleme / Stolperfallen
- **check:diag ist check-order-abhängig** (diese Session gepinnt): jede datei-hinzufügende/-entfernende Scheibe trägt einen Reihenfolge-Rauschterm bis ~±2 000, der NICHT die echten Typ-Kosten sind. Pin nur bei festem Datei-Set exakt; saubere Attribution = erst LEER hinzufügen + messen, dann füllen + messen. Ein echter Regress bleibt sichtbar (≫2 000, monoton).
- **Freeze-Zeremonie**: der Beweis ist der Ganz-Artefakt-Clean-Rebuild-Hash (nicht Per-Funktion-Bytes — jede Quelländerung verschiebt crate-weit `i32.const`-Panic-Location-Zeiger, WAT-Diff zeigt null Opcodes). Vor jeder Kernel-Änderung: Pre-Edit-Clean-Rebuild muss den alten Pin reproduzieren. `#[cfg(test)]`-Tests landen NICHT im Release-wasm.
- **Zwei-Verifier-Regel gilt** ab jetzt für jede substanzielle Scheibe (Aufträge aus docs/verify-runde-template.md, nicht frei formulieren).
- **Mess-Hausregel**: Baselines/Pins nur in frischem `git worktree` des Zielcommits (`git stash` lässt untracked Dateien liegen → kontaminiert); immer Exit-Code prüfen, nie nur die Kennzahl greppen. Hintergrund-Agenten fassen den Haupt-Tree nie an.
- **Vorbestehender Soundness-Gap** (FOLLOWUPS, MAJOR): mixed-rank Shape-Union IM Typparameter einer Instanz (`NDArray<[2,3]|[2,3,4]>`) akzeptiert `.sum(2)` still & typt konfident falsch — liegt in `ReduceAxis`/`Guard`/`OkShape`, eigene Scheibe.
- **Unverändert gültig**: Threads-Build-Regeln (nightly-2026-07-09 + build-std, env-RUSTFLAGS ersetzt config), cargo-Config CWD-basiert (alle Befehle vom Repo-Root), Test-Explizitlisten in package.json mit Guard, shared-Validator-Blind-Spots.

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse:** `docs/{spike-01..06,kern-01..11}-*.md` · `docs/roadmap.md` (Phase-Blöcke) · `docs/verify-runde-template.md` · `docs/infra-01-stress-split.md` · Backlog `FOLLOWUPS.md`.
- **Code:** Typ-Ebene `spike/src/` (`reduce.ts`, `reshape.ts`, `vector.ts`, `slice-literal.ts` = Digit-Arithmetik, `ndarray.ts`/`wasm/resident.ts` = Klassen); Kernel `crates/core/src/kernels/{add,elementwise,vector,matmul_blocked}.rs`; `crates/core/src/shape.rs` (unravel/compute_strides/validate_strided_bounds + die gegateten `_into`-Twins); Test-Infra `spike/tests-runtime/{prng,assert-helpers}.ts`.
- **Befehle (alle vom Repo-Root):** `pnpm check` (Verbund root+stress) · `check:diag` (Pin 172 392 @ 128) / `check:diag:stress` (94 597) · `pnpm test:core` (817) · `pnpm test:resident` (4265+2; +`:gc`) · `pnpm test:threaded` (65, nightly) · `pnpm demo` · `pnpm bench:{scaling,chain,strided,blocked,slice,threaded,crossover,editor,elementwise}` · `cargo test --manifest-path crates/core/Cargo.toml` (161). Artefakt-Freeze-Check: `shasum -a 256 spike/src/wasm/numtype_core.wasm` = `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`.
