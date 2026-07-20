# Covenant — NumType
<!-- covenant:version 5 -->

## Invarianten

### Strukturell (mechanisch geprüft)
- **S1** · Runtime-Quellcode importiert nie aus Test-, Bench- oder Demo-Verzeichnissen.
  → Regel `covenant-s1` · Anker: `spike/src/`

### Semantisch (geprüft per Verify)
- **M1** · Jeder WASM-Kern ist bit-identisch zur naiven TS-Referenz (`runtime.ts`), auch für
  IEEE-Spezialwerte; Optimierungen nur unter dem Bit-Identity-Law (Vektorisierung nur QUER zu
  Output-Elementen, aufsteigende k-Akkumulation, kein FMA/relaxed-simd). Kernel-lose
  Referenz-Ops (Referenzfunktion in `runtime.ts` ohne WASM-Kern-Gegenstück) sind zulässig,
  solange die Paritätslücke in FOLLOWUPS getrackt ist; M1 bindet in dem Moment, in dem ein
  Kernel für die Op entsteht. (Präzisierung v5 — Präzedenzfall W1 argmax/topk.)
  Anker: `crates/core/src/`, `spike/src/runtime.ts`
- **M2** · Typ-Ebene „never wrong, only incomplete": Compile-Ablehnung nur für garantierte
  Runtime-Throws; wide/Union/dynamischer Rang degradieren zu no-claim — nie ein
  konfident-falscher Claim.
  Anker: `spike/src/dim.ts`, `spike/src/literal-arithmetic.ts`, `sym:Guard`, `sym:OkShape`
  · GESCHLOSSEN in Item 11 / S1 (2026-07-17): der `Literal|undefined`-Verstoß durch OPTIONALE
  Parameter (`sum`s `axis`/`keepdims`) ist behoben. Der `sum`-Overload-Umbau (Overloads nach
  Argument-Anzahl 0/1/2 — keine optionalen Parameter mehr in der Mehr-Argument-Form — plus
  `reduce.ts`-`KeepDims`-Erweiterung auf `boolean | undefined`) verhindert das
  `undefined`-Stripping: `a.sum(u)`/`u:0|undefined` degradiert jetzt zu no-claim
  (`readonly number[]`), `a.sum(0,kd)`/`kd:true|undefined` zu einer ehrlichen Shape-Union
  (`readonly [3] | readonly [1,3]`). Der frühere `UA_GAP`-Sentinel-Pin ist umgekehrt
  (`UA_AXIS_CLOSED` + `UA_KEEP_CLOSED`/`WUA_*`, spike/tests/ndarray.test-d.ts) und bewacht
  künftig die Schließung. Dreifach verifiziert (Spec CONFIRMED + adversarial HÄLT +
  covenant-verify kein Verstoß, 2026-07-17).
- **M3** · Shape-Fehler erscheinen AM fehlerhaften Argument, Message-Stamm wortgleich zum
  Runtime-Throw; Klassen-Hover bleiben saubere Tupel (`NDArray<[2, 3]>`). „Klassen-Hover"
  meint die Typ-Parameter-Anzeige der Klasse; Member-Hover (z. B. `.shape`) dürfen
  readonly-Modifier tragen. (Präzisierung v5 — schließt die seit D-V2 offene
  Auslegungsfrage.)
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
- **Z2** · `pnpm check` bleibt der Verbund aller QUELLTEXT-Typ-Korpora (root + stress +
  browser + künftige) — kein Quelltext-Korpus rottet ungeprüft. Typchecks gegen ein
  BAUERGEBNIS (`dist/`, das beim reinen `noEmit`-Check nicht existiert — z.B. der Item-11/S3
  Konsumenten-Smoke `spike/tests-package/consumer/`) laufen stattdessen im Paket-Testlauf
  (`test:package`, nach `build:dist`) — auch sie rotten nicht, liegen aber bewusst in einem
  ANDEREN, build-abhängigen Gate. Reine Quelltext-Smokes bleiben in `pnpm check` (z.B. der
  Laufzeit-Smoke `package-smoke.test.ts`, dessen dist-Import dynamisch/untypisiert ist).
  Anker: `package.json`

## Nicht-Ziele
- Kein NumPy-Vollklon (keine 400 Ops), kein GPU/autograd, keine DataFrames.
- Kein Per-Call-Routing zwischen Backend-Cores (dokumentierte Sackgasse, Kern 06).
- Kein Browser-Port des Threads-Pfads in v0 (COOP/COEP-gated; Owner-Option 1).
- Keine transzendenten Ops ohne eigene Determinismus-Entscheidung (brechen Bit-Parität).

## Änderungslog
- v5 (2026-07-20) · M1: präzisiert — kernel-lose Referenz-Ops sind zulässig (Paritätslücke
  in FOLLOWUPS getrackt; M1 bindet ab Kernel-Existenz). Anlass: covenant-verify-Empfehlung
  der Op-Scheibe W1 (argmax/topk = erster Präzedenzfall), VOR den Folge-Scheiben W2–W5
  geklärt. M3: präzisiert — „Klassen-Hover" = Typ-Parameter-Anzeige; Member-Hover dürfen
  readonly-Modifier tragen (schließt den seit D-V2/2026-07-13 offenen Auslegungs-Mini,
  Baustein-C-Befund V2). Beide Normen inhaltlich unverändert (Klarstellung gelebter
  Praxis); Owner-bestätigt 2026-07-20.
- v4 (2026-07-17) · Z2: präzisiert — der Verbund `pnpm check` deckt alle QUELLTEXT-Typ-Korpora;
  Typchecks gegen ein Bauergebnis (`dist/`, existiert beim noEmit-Check nicht) laufen bewusst
  im Paket-Testlauf (`test:package`), nicht in `pnpm check`. Anlass: covenant-verify-Befund
  der Item-11/S3-Scheibe (der Konsumenten-Typ-Smoke braucht das gebaute `dist/index.d.ts`);
  Owner-entschieden „Norm präzisieren statt Korpus erzwingen". Norm-Absicht unverändert (kein
  Korpus rottet ungeprüft).
- v3 (2026-07-17) · M2: der unter v2 dokumentierte offene Verstoß (Literal|undefined via
  OPTIONALE Parameter, `sum`s `axis`/`keepdims`) ist GESCHLOSSEN — Item 11 / S1
  (`sum`-Overload-Umbau nach Argument-Anzahl + `reduce.ts`-`KeepDims`-Erweiterung, KD-2),
  dreifach verifiziert; M2-Anker `slice-literal.ts` → `literal-arithmetic.ts` (Datei in T1b
  umbenannt). Norm unverändert (der Verstoß war stets ein Norm-Bruch, jetzt behoben);
  Owner-bestätigt.
- v2 (2026-07-13) · M2: bekannter offener Verstoß dokumentiert (Literal|undefined via
  optionale Parameter, UA_GAP-Sentinel, Item-11-Frist) — Norm unverändert; Anlass:
  covenant-verify-Befund der Union-Axis-Mini-Scheibe, Owner-Entscheidung „dokumentieren
  statt Norm konditionieren".
- v1 (2026-07-13) · Erstfassung (Phase-D-Vorarbeiten, vor V2/Item 11).
