# Handoff — 2026-07-11, Session-Ende (Kern 07 + Kern 08: Phase B Items 1 + 5-Rest)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Remote: `github.com/crankfunk/numtype`, privat. Phase A komplett; **in dieser Session Phase B Items 1 und 5-Rest gelandet, beide unabhängig verifiziert (CONFIRMED), committet und gepusht:**

- **Kern 07 — Elementwise + Vektor-Ops** (Commit 9ab262b): `sub`/`mul`/`div` + `dot`/`norm`/`cosineSimilarity` auf NDArray + WNDArray; ein generischer strided Elementwise-Kernel + `dot_strided`/`norm_sq_strided` hinter fünf appendeten ABI-Einstiegspunkten; norm/cosine OHNE eigene Kernel (gepinnte TS-Kompositionen — sqrt/*// sind IEEE-exakt); Skalar-Ops geben plain `number` (dokumentierte Asymmetrie zu `sum()`); `DotCheck`-Guard mit Union-Dim-no-claim; +772 Differentialtests, +48 cargo; drei Mutations-Beweise; **neuer Artefakt-Pin `7a65d800…`** (Export-addierende Phase → vierteiliger Freeze-Beweis, Regel in CLAUDE.md); Abweichung bestätigt: TS-Klassenkörper = insertion-only. docs/kern-07-*.
- **Kern 08 — Runtime `reshape`/`flatten` + Messobligation** (dieser Commit): konsumieren `LiteralShapeProduct` (Spike 04) — `flatten()` hovert als berechnetes Literal (`NDArray<[1048576]>`), Produkt-Mismatch = Compile-Fehler am Argument, Wortlaut wortgleich zum Runtime-Throw (Message-Tabelle in kern-08-ergebnisse.md, Dim-Validität vor Produkt); Stretch `LiteralReshapeDimInvalid` gehalten (negativ/Dot-Form geliftet, Exponent/0/Union no-claim; nackte Verdikte in slice-literal.ts, Messages in reshape.ts — hält slice-literal append-only); WNDArray: View-wenn-contiguous (isContiguous verlangt offset 0 — konservativ, nie unsound), sonst nt_materialize; **NULL Rust-Änderungen, Artefakt-Hash byte-identisch** (starke Freeze-Form, beidseitig per Clean-Rebuild reproduziert). **Beide Spike-04-Obligationen geschlossen:** Guard-Wortlaut + Editor-Hover-Messung (neuer bench:editor-Workload W6: Hover-Mediane 0,06 ms inkl. Big-Dim-Flatten, Toggle 3,3 ms, 23 100 Instantiations isoliert = in-family, ~halb W3; per-Site-Guard-Kosten ~6k bei Huge-Dim-Sites, Deklaration allein billig). `-1`-Inferenz bewusst deferred (FOLLOWUPS). docs/kern-08-*.

Gate-Stand (Kern 08, implementiert + vom Verifier unabhängig reproduziert): `pnpm check` clean 0,42–0,64 s · `check:diag` **243 446** (neuer Pin, deterministisch 2×2 Läufe) · `test:core` 817 · `test:resident` **3279+2** (+187) · `cargo` 157 · `test:threaded` 65 · `pnpm demo` all-agree inkl. reshape-Sektion mit inline-bewiesener View-Route · `bench:editor` hard gate PASS inkl. W6, W1–W5 unregressiert · Artefakt-Hash `7a65d800…` IDENTISCH.

## In dieser Session erledigt
- Kern 07 komplett (Spec → deep → Verify CONFIRMED mit eigenem Mutanten → Ergebnisdoc → KB → Commit 9ab262b → Push).
- Kern 08 komplett (Spec → deep → Verify CONFIRMED → Ergebnisdoc → KB → Commit/Push, dieser Stand).
- KB: 3 Notizen revidiert für Kern 07 (Bit-Identität: Skalar-Kompositionen + NaN-Payload-Ehrlichkeit; Runtime-Fehler-Heben: Union-of-Shapes-Facette; Zeilenshift: vierteiliger Export-Beweis + Insertion-only-Regel) + 1 für Kern 08 (Budget-Gates: Produkt- vs. Stress-Test-Kosten bisektieren); Graph gebaut, Kanten verifiziert.
- FOLLOWUPS: Spike-04-Messobligation ausgetragen [x]; neu: `-1`-Inferenz, size-0-Materialize-Mini, Union-Guard-Zwei-Facetten-Item (Kern 07).

## Offen / in Arbeit
Nichts halbfertig.

## Nächste Schritte (priorisiert, = Roadmap Phase B Rest)
1. **`keepdims` im Runtime-`sum()`** — Typ-Ebene (`ReduceAxisKeepDims`) existiert und ist getestet; Runtime-Parameter fehlt. Kleine Scheibe.
2. **Spezialwerte im Differential-Generator** (NaN/±Inf/±0/Subnormals) — Anspruch „bit-identisch" ist nur für normale endliche Werte + gezielte Fixtures belegt. Comparator-Standard: `Object.is` = NaN-payload-insensitiv (ehrlich dokumentiert; WASM-NaN-Payload-Nichtdeterminismus).
3. **VOR dem nächsten Spec entscheiden: check:diag-Budget-Linie.** Pin-Serie 133 656 → 188 378 → 200 714 → 243 446 nähert sich der gewählten 250k-Linie (5 % des ~5M-Budgets); 78 % des letzten Deltas sind mandatierte Type-Test-Stressfälle, Produkt-Kosten nur +9,5k. Optionen: Linie bewusst neu setzen ODER Stressfälle in eine separat gemessene tsconfig verschieben (bench-dx-Muster). Nicht still driften lassen.
4. Perf-Kür nur nach Messung; Kür/klein: non-integer start/stop-Guard, Deep-readonly-shape, Union-Guard-Scheibe (beide Facetten).

## Bekannte Probleme / Stolperfallen
- **Neu (Kern 08):** `ReshapeCheck` macht DREI Typ-Berechnungen pro Site (zwei Produkte + Dim-Walk) — Huge-Dim-Sites ~6k Instantiations; im Editor trotzdem flach (W6 0,06 ms). isContiguous() verlangt offset 0 → offset-verschobene contiguous-förmige Views materialisieren konservativ. size-0-Materialize-Pfad weiterhin ungetestet (FOLLOWUPS-Mini).
- **Neu (Kern 07):** Export-addierende Phasen → vierteiliger Freeze-Beweis; TS-Klassenkörper insertion-only; `backend-oom.test.ts` braucht pro neuer ABI-Signatur einen Mock-Stub; Union-Guard-Latenz zwei Facetten (FOLLOWUPS).
- **Unverändert gültig:** alle Einträge des vorigen Handoffs (Typ-Ebenen-Regeln Spikes 03–06, Threads-Build-Regeln, cargo-Config CWD, Test-Explizitlisten mit Guard, TS7-Ein-Diagnose-Verhalten, shared-Validator-Blind-Spots: `normalizeSliceSpecs`, `assertVectorPair`, jetzt auch `assertReshapeArgs`).

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse:** `docs/{spike-01..06,kern-01..08}-*.md` · `docs/roadmap.md` (Statusblöcke A + B) · Backlog `FOLLOWUPS.md`.
- **Code:** Typ-Ebene `spike/src/` (`reshape.ts` NEU, `vector.ts`, `slice-literal.ts` = Digit-Arithmetik + Klassifikatoren, `ndarray.ts`/`wasm/resident.ts` = Klassen); Kernel `crates/core/src/kernels/{elementwise,vector}.rs`; DX-Harness `spike/bench-dx/` (Workloads W1–W6).
- **Befehle (alle vom Repo-Root):** `pnpm check`/`check:diag` (Pin: 243 446) · `pnpm test:core` (817) · `pnpm test:resident` (3279+2; +`:gc`) · `pnpm test:threaded` (65, nightly!) · `pnpm demo` · `pnpm bench:{scaling,chain,strided,blocked,slice,threaded,crossover,editor}` · `cargo test --manifest-path crates/core/Cargo.toml` (157).
