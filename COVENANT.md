# Covenant — NumType
<!-- covenant:version 2 -->

## Invarianten

### Strukturell (mechanisch geprüft)
- **S1** · Runtime-Quellcode importiert nie aus Test-, Bench- oder Demo-Verzeichnissen.
  → Regel `covenant-s1` · Anker: `spike/src/`

### Semantisch (geprüft per Verify)
- **M1** · Jeder WASM-Kern ist bit-identisch zur naiven TS-Referenz (`runtime.ts`), auch für
  IEEE-Spezialwerte; Optimierungen nur unter dem Bit-Identity-Law (Vektorisierung nur QUER zu
  Output-Elementen, aufsteigende k-Akkumulation, kein FMA/relaxed-simd).
  Anker: `crates/core/src/`, `spike/src/runtime.ts`
- **M2** · Typ-Ebene „never wrong, only incomplete": Compile-Ablehnung nur für garantierte
  Runtime-Throws; wide/Union/dynamischer Rang degradieren zu no-claim — nie ein
  konfident-falscher Claim.
  Anker: `spike/src/dim.ts`, `spike/src/slice-literal.ts`, `sym:Guard`, `sym:OkShape`
  · Bekannter offener Verstoß (Owner-entschieden 2026-07-13, Frist Item 11):
  `Literal|undefined` durch OPTIONALE Parameter (`sum`s `axis?`/`keepdims?`) — TS streift
  `undefined` bei der Inferenz, der Filter ist strukturell unerreichbar; als
  `UA_GAP`-Sentinel-Pin beobachtbar gemacht (spike/tests/ndarray.test-d.ts), FOLLOWUPS
  „Literal|undefined durch optionale Parameter"; Fix-Kandidat Overload-Split beim
  Item-11-API-Schnitt.
- **M3** · Shape-Fehler erscheinen AM fehlerhaften Argument, Message-Stamm wortgleich zum
  Runtime-Throw; Klassen-Hover bleiben saubere Tupel (`NDArray<[2, 3]>`).
  Anker: `sym:Guard`, `sym:ShowShape`
- **M4** · Frozen Baseline: v1-Kerne/-Einstiegspunkte bleiben byte-unberührt; der bindende
  Freeze-Beweis ist der Artefakt-Hash aus einem Clean-Rebuild; abi.rs/matmul_blocked.rs/shape.rs
  nur append-only.
  Anker: `crates/core/src/abi.rs`, `crates/core/src/kernels/matmul_blocked.rs`, `crates/core/src/shape.rs`
- **M5** · Der Default-`NDArray`-Pfad ist browser-sicher: kein eager Laden von node:*-Builtins;
  Threads ausschließlich als explizites Node-only-Opt-in hinter `backend("threaded")`
  (type-only-Imports und dynamisches `import()` nach Env-Check sind erlaubt).
  Anker: `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`

## Zusagen (Verhalten, das erhalten bleibt)
- **Z1** · Zero-Dependency-Runtime: das Paket bekommt nie ein `dependencies`-Feld; Kernels und
  Typ-Maschinerie bleiben from scratch (Dev-Tooling ist erlaubt).
  Anker: `package.json`
- **Z2** · `pnpm check` bleibt der Verbund ALLER Typ-Korpora (root + stress + browser +
  künftige) — kein Korpus rottet ungeprüft.
  Anker: `package.json`

## Nicht-Ziele
- Kein NumPy-Vollklon (keine 400 Ops), kein GPU/autograd, keine DataFrames.
- Kein Per-Call-Routing zwischen Backend-Cores (dokumentierte Sackgasse, Kern 06).
- Kein Browser-Port des Threads-Pfads in v0 (COOP/COEP-gated; Owner-Option 1).
- Keine transzendenten Ops ohne eigene Determinismus-Entscheidung (brechen Bit-Parität).

## Änderungslog
- v2 (2026-07-13) · M2: bekannter offener Verstoß dokumentiert (Literal|undefined via
  optionale Parameter, UA_GAP-Sentinel, Item-11-Frist) — Norm unverändert; Anlass:
  covenant-verify-Befund der Union-Axis-Mini-Scheibe, Owner-Entscheidung „dokumentieren
  statt Norm konditionieren".
- v1 (2026-07-13) · Erstfassung (Phase-D-Vorarbeiten, vor V2/Item 11).
