# Handoff — 2026-07-13, Session-Ende (Phase-D-Vorarbeiten + Union-Axis + Covenant)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch
Rust/WASM-Kerne). Remote `github.com/crankfunk/numtype`, privat. **Phasen A–C komplett;
die Phase-D-VORARBEITEN (V1–V3) und die Union-Axis-Mini-Scheibe sind erledigt, dreifach
verifiziert, committet & gepusht** (HEAD = origin/main = 22aaabd, Tree sauber). Seit dieser
Session gilt das **Covenant-Regime**: COVENANT.md (v2) ist der stehende Produkt-Vertrag
(S1 mechanisch via `graph-a-lama query lint`, M1–M5/Z1–Z2 per covenant-verify-Agent =
Baustein C, Eskalationsleiter in CLAUDE.md „Qualitätssicherung"). Alle Gates grün:
`pnpm check` (DREIER-Verbund) · test:core 818 · test:resident 4280 · test:browser 4/4 ·
test:threaded 69 · cargo 161 · demo · covenant-lint 0/0 · bench:editor PASS (7 Workloads) ·
Artefakt-Hash `0b9df4f1…519c7d` byte-identisch. Pins: check:diag **178'865 @ 132** /
stress **102'182 @ 82** / browser **2'142 @ 75**. **Nächstes: Item 11 (API-Schnitt/
Paketierung) — die größte Phase-D-Scheibe.**

## In dieser Session erledigt
- **Phase-D-Vorarbeiten-Spec** (8b39c15, docs/phase-d-vorarbeiten-spec.md): drei Scheiben
  gebunden, Baustein-0-verifiziert (fing den Facette-(b)-Blocker VOR dem Bau: die
  `NDArray<A>|NDArray<B>`-Form wird von TS-Inferenz selbst abgelehnt).
- **V3 Browser-Smoke** (e0feee9): erster Real-Browser-Beweis des Standard-Surface
  (Playwright/Chromium devDep, tsc-Emission, node:http+wasm-MIME, byte-exakte
  Differential-Matrix, COOP-frei). Verify-Schließungen: mtime-Freshness-Guard (nie
  `playwright test` direkt!), Streaming-Pfad-Spy, playwright.config typgeprüft.
- **V1 Union-Guard-Fix** (d0cfa78): never-wrong hält für Union-Dims (inkl. matmuls
  konfident-falscher ABLEHNUNG), Mixed-Rank-Shape-Unions (`RankUnknowable` an 7 Gates,
  uniforme Degradation per Owner-Entscheid), tuple-wrapped `Guard` (uniforme Fehler-Union
  → EINE kombinierte Message). Kern-07-Diskrepanz ehrlich aufgeklärt (a914654:
  Compiler nachweislich identisch → Addendum-Prosa beschrieb die Probe falsch).
- **Covenant v1→v2** (4db74e0, 1e8261a, 4f4414e, 22aaabd): Produkt-Vertrag + Baustein C im
  Verify-Template + Eskalationsleiter (nie vorsichtshalber den vollen Katalog) + M2-Notiz
  „bekannter Verstoß" (s. u.).
- **V2 Harmonisierung** (cf414d6): strides = readonly-Property überall (Methode hatte 0
  Aufrufer), `WNDArray implements NDArrayView<S>` (Threaded gratis), deep-readonly `shape`
  via `Readonly<S>` (TS2636-Probe: homomorphe Mapped Types passieren den out-Check).
  UNGEPLANTER Befund: die gemessene NDArray-Invarianz war ein AllOnes-Zufall und fiel durch
  Readonly<S> → Owner-Entscheid RE-Invariantierung per explizitem property-style
  `__variance`-Marker (Method-Shorthand wäre bivariant!); Marker senkte den Pin um −10'308.
- **Union-Axis-Mini** (22aaabd): `IsUnion<Axis>`-Filter in ReduceAxis VOR dem naked Check
  (Position load-bearing, Mutant-bewiesen); jede Achsen-Union degradiert wie die dynamische
  Achse. Baustein 0 fing den Blocker: `Literal|undefined` ist über OPTIONALE Parameter
  strukturell unerreichbar (TS streift undefined bei der Inferenz) → Owner-Scope-Reduktion,
  Familie als bekannter M2-Verstoß in COVENANT.md v2 mit UA_GAP-Sentinel dokumentiert.
  Neuer bench:editor-Workload W7 (Union-Achsen-Hover).

## Offen / in Arbeit
- Nichts halbfertig. Alle bewusst zurückgestellten Punkte stehen in FOLLOWUPS.md
  (Faktenstand-Hauptbuch); release-relevant darunter: **Literal|undefined-Familie**
  (Overload-Split-Kandidat, Item-11-Entscheidung, COVENANT-M2-Notiz), Zero-dep-Guard-Test,
  `slice-literal.ts`-Umbenennung, npm-Name erneut prüfen/sichern, WebKit/Firefox-Smoke,
  COVENANT-M3-Wortlaut-Präzisierung (Klassen- vs. Member-Hover, v3-Kandidat).

## Nächste Schritte
1. **Item 11: API-Konsolidierung + Paketschnitt** — bindende Spec nach Haus-Muster (Spec →
   Baustein 0 → Impl → A+B+C). In die Spec gehören die Item-11-FOLLOWUPS: Overload-Split
   für `sum`s optionale Parameter, slice-literal.ts-Umbenennung, `exports`-Map +
   `.wasm`-Bundling, d.ts-Hover-Qualität, npm-Name, Zero-dep-Guard-Test.
2. Danach Item 12 (CI: alle Gates inkl. Freeze-Hash, bench:editor als Gate, Test-Timeouts)
   und Item 13 (README/Release-Mechanik).

## Bekannte Probleme / Stolperfallen
- **Pins nie über Korpora oder File-Set-Änderungen hinweg vergleichen**; Datei-ADDITIONEN
  tragen ±~2'000 Order-Noise, EDITs nicht (Kommentar-Kontrollproben-Methode). Messungen nur
  im frischen `git worktree`, Exit-Code prüfen.
- **`pnpm test:browser` nie als direktes `playwright test` aufrufen** — Freshness-Guard
  wirft sonst (bewusst; stale `.emit` war ein bewiesener False-Pass-Vektor).
- **Covenant-Regime** (CLAUDE.md „Qualitätssicherung"): substanzielle Scheiben = voller
  Katalog A+B+C parallel + Lint im Gate-Block; Eskalationsleiter für Kleineres. Specs und
  Delegations-Prompts benennen berührte Invarianten-IDs. Spec-Änderungen nur mit
  Owner-Bestätigung + Version-Bump.
- Threads-Artefakt braucht die gepinnte nightly-2026-07-09 (+rust-src); alle Kommandos vom
  Repo-Root (cargo-Config-Discovery); Freeze-Beweis = Artefakt-Hash aus CLEAN-Rebuild.
- `NDArray`/`WNDArray` sind BEWUSST invariant (expliziter `__variance`-Marker) —
  property-style ist Pflicht, ein `out` ist unmöglich (TS2636 auf Transpose-Rückgaben);
  `NDArrayView` ist die einzige enforced-kovariante Surface.
- Optionale Parameter streifen `undefined` aus Inferenz-Unions (UA_GAP-Sentinel wacht);
  Workaround explizites Typ-Argument.

## Wichtige Dateien & Befehle
- Vertrag/Prozess: `COVENANT.md` (v2) · `docs/verify-runde-template.md` (Bausteine 0/A/B/C)
  · `FOLLOWUPS.md` · `CLAUDE.md` (Pins, Kommandos, QA-Regeln).
- Diese Session: `docs/phase-d-vorarbeiten-spec.md` + `-v{1,2,3}-ergebnisse.md`,
  `docs/union-axis-mini-{spec,ergebnisse}.md`.
- Kern-Kommandos: `pnpm check` (Dreier-Verbund) · `pnpm check:diag[:stress|:browser]` ·
  `pnpm test:core|test:resident|test:browser|test:threaded` · `cargo test --manifest-path
  crates/core/Cargo.toml` · `pnpm demo` · `pnpm bench:editor` · `graph-a-lama . --symbols
  && graph-a-lama query lint`.
