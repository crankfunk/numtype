# NumType — Roadmap bis zum möglichen OSS-Release

Stand: 2026-07-10 (nach Kern 06 + Auto-Weiche). Internes Planungsdokument — Verhältnis zu
`FOLLOWUPS.md`: die Roadmap ordnet und priorisiert, FOLLOWUPS bleibt das Backlog-Hauptbuch
(eintragen beim Zurückstellen, austragen beim Commit). Bei Widerspruch gewinnt FOLLOWUPS als
Faktenstand, die Roadmap wird nachgezogen.

## Ausgangslage (Kurzfassung, Details in den Phasen-Docs)

Beide Schichten der Projektthese tragen:

- **Typ-Ebene (der USP):** Broadcasting/matmul/Reduktion als Typen, graduell (Literal-Dims
  statisch, `number`-Dims → Runtime-Checks), Fehler am fehlerhaften Argument, saubere Hovers
  (Spike 01). Seit Kern 05 zusätzlich Arithmetik über große Literal-Dims per Digit-String-Typen
  (statisch berechnete Slice-Shapes, 1,59× Typcheck-Budget bei 3×-Gate).
- **Runtime (from scratch, null Dependencies):** sechs Kerne, jeder bit-identisch zur naiven
  TS-Referenz bewiesen — handgerolltes WASM-ABI (Kern 01), Zero-Copy-Residenz (Kern 02),
  strided Views (Kern 03), blocked+packed+SIMD128-Matmul 2,1–3,25× (Kern 04), O(1)-Slicing
  (Kern 05), Threads mit handgerolltem Shared-Memory-Substrat, ~4× ab n=256/8 Worker,
  inkl. gemessener größenbasierter Auto-Weiche (Kern 06 + Follow-up).
- **Prozess-Substanz** (beim Release selbst ein Asset): Freeze-Disziplin mit
  Artefakt-Hash-Beweis, Differential-/Guard-Tests, Bench-Ehrlichkeitsregeln, per-Phase-Specs
  mit unabhängigen Verifikations-Addenda.

**Ehrlicher Kassensturz:** Op-Surface schmal (add, matmul, sum, transpose, slice, Erzeugung/
Konvertierung); drei getrennte API-Oberflächen (naives `NDArray`, v1-Funktionen,
`WNDArray`/`threadedMatmul`); Threads Node-only auf gepinntem Nightly-Zweitartefakt; Code lebt
in `spike/`, nicht in einem Paket; Bit-Identität bisher nur für normale endliche Werte belegt.

## Was wir releasen

Zwei gekoppelte Deliverables:

**(a) Das npm-Paket `numtype`** (Name frei, geprüft 2026-07-09; Fallback `@numtype/core`).
TypeScript-first-ndarray-Bibliothek, Kern-Story **nicht** „NumPy in JS" (gegen stdlib/numjs
unverteidigbar), sondern: *Shape-Fehler werden Editor-Fehler.* Zero-Dependency, gebündeltes
`.wasm`, kein natives Kompilieren. Nutzung:

```ts
import { NDArray } from "numtype";

const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]); // NDArray<[2, 3]>
const c = a.matmul(NDArray.zeros([3, 4]));               // NDArray<[2, 4]> — Hover zeigt es
a.matmul(NDArray.zeros([5, 4]));                         // ❌ Editor-Squiggle AM Argument
const row = c.slice({ stop: 1 });                        // NDArray<[1, 4]> statisch berechnet
const emb: NDArray<[number, 1536]> = await load();       // graduell: Runtime-Checks
```

Der „Editor-Moment" ist das Produkt; WASM-Perf ist Glaubwürdigkeits-Feature, nicht Kaufgrund.
Zielnutzer (Wettbewerbsanalyse §7): Embedding-/RAG-Pipelines in TS, ML-Pre-/Postprocessing im
Browser, Node-Backends ohne Python-Sidecar, Audio/Signal mit festen Fenstergrößen, Lehre.

**(b) Die Forschungs-Story.** „Wie weit trägt TSs Typsystem Tensor-Shapes?" als Blog-Serie/
Talk — publikationsfähig unabhängig von Adoption; die dokumentierten Grenzen sind selbst ein
Beitrag. Release-Positionierung: **v0.1 research preview** mit den ehrlichen Qualifikationen
aus dem USP-Doc (§5).

**Nicht-Ziele:** kein NumPy-Vollklon (keine 400 Ops), kein GPU/autograd, keine DataFrames.

## Phasen (priorisiert)

Logik der Reihenfolge: A killt das einzige Risiko, das das Projekt obsolet machen könnte;
B macht es für die Ziel-Use-Cases nützlich; C entscheidet, wie groß v0 sein muss (und darf
kürzen); D ist Fleißarbeit, die erst lohnt, wenn A–C stehen. Phasen-Disziplin unverändert:
Spec → Implementierung → Fresh-Context-Verify → Ergebnisdoc → KB-Capture → Commit.

### Phase A — USP absichern (zuerst: das strukturelle Killer-Risiko)

> **Status 2026-07-11: Phase A KOMPLETT, inkl. Kür.** A1 = Spike 02 (Gate PASS), A2 = Spikes
> 03+04 (Bounds-Checks; Shape-Produkte via Digit-Multiplikation, GO als offengelegte
> Gate-Abweichung), A3 = Spike 05 (`NDArrayView<out S>`, NDArray bleibt invariant), A4/Kür =
> Spike 06 (negative literale start/stop + literale steps, Step-Guard). Details in den
> jeweiligen docs/spike-0X-*-Paaren; Faktenstand wie immer in FOLLOWUPS.md.

1. **Editor-Latenz real messen** (VS Code tsserver, nicht nur der `tsc
   --extendedDiagnostics`-Proxy). Der USP *ist* das Editor-Erlebnis; wird Hover/Squiggle bei
   realistischen Op-Ketten träge, muss das Typ-Design reagieren, bevor irgendetwas anderes
   gebaut wird. (FOLLOWUPS-Item, offen.)
2. **Typ-Ebenen-Ausbau aus Kern 05:** Index-Bounds-Checks literaler Indizes (Entscheidung
   Compile-Fehler vs. Runtime-only) und reshape/flatten-Produkte (O(Stellen²),
   Budget-Entscheidung via `check:diag`).
3. **Varianz-Design entscheiden** (`NDArrayView<out S>` vs. Spike-Lösung `AnyNDArray`) —
   API-prägend, muss vor jedem API-Freeze fallen (two-of-three rule,
   docs/spike-01-ergebnisse.md Addendum).
4. Kür: negative literale start/stop, literale steps ≠ 1 (vorzeichenbehaftete Digit-Addition
   bzw. ceil-Division).

### Phase B — Minimum Viable Op-Surface (an den Use Cases entlang, nicht an NumPy)

> **Status 2026-07-12: Phase B KOMPLETT (Items 5–7 alle erledigt & zweifach verifiziert).**
> Item 5: Kern 07 (sub/mul/div + dot/norm/cosineSimilarity, docs/kern-07-*), Kern 08
> (Runtime-`reshape`/`flatten` konsumieren `LiteralShapeProduct`, docs/kern-08-*), Kern 09
> (Runtime-`keepdims` auf `sum()` beider Surfaces, `ReduceAxis<S,Axis,KeepDims>`, docs/kern-09-*).
> Item 6: **Kern 10** (IEEE-754-Spezialwerte NaN/±Inf/±0/Subnormals in den Differential-
> generator injiziert — Bit-Identität auch dort belegt, SIMD-matmul erhält Subnormals nachweislich;
> docs/kern-10-*). Item 7: **Kern 11** (messgetriebener Contiguous-elementwise-Fast-Path,
> 13–17× durch Überspringen der per-Element-`unravel`-Allokation, bit-identisch; SIMD elementwise
> + Packing-Reuse-A gemessen NO-GO; docs/kern-11-*). Ab Kern 09 gilt die Zwei-Verifier-Regel
> (Spec + adversarial, docs/verify-runde-template.md). Pins seit Kern 11: Haupt-`check:diag`
> **172 392 @ 128** (der Rückgang ggü. Kern 10 ist reihenfolge-abhängiges Mess-Rauschen, gepinnt —
> docs/kern-11-*), Stress **94 597**, Artefakt-Hash `0b9df4f1…`. Das Kern-08-G2-Trend-Flag ist
> AUFGELÖST (Infra 01, docs/infra-01-stress-split.md): Stress-Typtests separat gemessen, `check`
> = Verbund. **Nächstes: Phase C oder D.**

5. Elementwise-Familie (sub/mul/div), `dot`/Norm/Cosine-Similarity (der Embedding-Use-Case
   braucht genau das), Runtime-`reshape`/`flatten`, `keepdims` (Typ-Ebene existiert und ist
   getestet).
6. **Spezialwerte im Differential-Generator** (NaN/±Inf/±0/Subnormals) — der Anspruch
   „bit-identisch" muss auch dort belegt sein. Transzendente Ops (exp/…) nur nach der
   dokumentierten Determinismus-Entscheidung (brechen Bit-Parität zur JS-Referenz,
   libm-Differenzen); notfalls v0 ohne sie.
7. Perf-Kür nur nach Messung: Packing-Buffer-Reuse im blocked matmul (hebt vermutlich auch die
   8-Worker-Effizienz), SIMD elementwise (memory-bound — erst messen).

### Phase C — Plattform-Entscheidungen (können den v0-Scope beschneiden → vor der Paketierung)

> **Status 2026-07-12: Items 8 & 9 entschieden (Owner) — ZURÜCKGESTELLT.** Nach einem
> Scoping-Slice (Constraint-Recherche, kein Bau; docs/phase-c-threads-scoping.md) fiel die
> Entscheidung auf **Option 1: Threads bleiben für v0 ein explizit experimentelles,
> Node-only-Opt-in** — weder Browser-Port (8) noch stable/no_std-Pfad (9) werden jetzt
> gebaut. Kernbefunde: (9) es gibt HEUTE keinen Weg weg vom pinned nightly — build-std bleibt
> nightly-only und die 2026-RFCs 3874/3875 decken atomics-target-feature-Rebuilds NICHT ab,
> `wasm32-wasip1-threads` ist eine Sackgasse (withdrawn Proposal, thread-spawn auf stable
> kaputt, kein Browser-WASI), no_std entkommt vermutlich nicht (offen, 30-Min-Experiment →
> FOLLOWUPS); unser Ansatz IST der Ökosystem-Standard (wasm-bindgen-rayon identisch), datierte
> Nightlies reproduzierbar. (8) Der Port ist machbar/abgegrenzt, aber sein Wert ist durch
> COOP/COEP-Deployment-Friction begrenzt (Header nur von der konsumierenden App setzbar; ein
> großer Teil der Zielnutzer — GitHub Pages/Sandboxes/CDNs — kann nicht), und alle
> Vergleichsprojekte liefern Threading als feature-detektiertes Opt-in, nie als Default.
> Erfüllt das Release-Gate ("Threads sauber als Node-only/experimentell abgegrenzt").
>
> **Item 10 (Backend-Wahl-API) ERLEDIGT & zweifach post-verifiziert (2026-07-12,
> Commit 5b0f951):** `NDArray.backend('wasm'|'threaded')` exponiert WASM/Threads als
> explizites, browser-sicheres (empirisch bewiesenes) Opt-in-Backend; das reine-JS
> `NDArray` bleibt der Default; null Rust, Artefakt-Hash byte-identisch; neue Pins
> check:diag 175.634 @ 132 / stress 103.882 @ 82 (docs/item-10-backend-api-*). Diese
> Scheibe pilotierte die neue Work-Ethic-Regel „Spec-Verifikation VOR der
> Implementierung" (Baustein 0) — die Pre-Build-Spec-Review fing DREI Blocker vor dem
> Bau. **Damit ist Phase C inhaltlich abgeschlossen; als Nächstes stünde Phase D
> (Paketierung/Release) an.**

8. **Browser-Port des Threads-Pfads** (COOP/COEP, `crossOriginIsolated`, async Dispatch statt
   blockierendem `Atomics.wait` auf Main) — *oder* bewusste Entscheidung: Threads bleiben in
   v0 ein Node-only-Opt-in (vertretbar; das Standard-Artefakt baut heute schon auf stable).
   **→ ZURÜCKGESTELLT 2026-07-12** (Option 1): Node-only-Opt-in gewählt; Port nur bei realer
   Nachfrage COOP/COEP-fähiger Konsumenten, FOLLOWUPS.
9. no_std/stable-Pfad fürs Threads-Artefakt — der Release soll nicht an einem gepinnten
   Nightly hängen; Alternative: Threads als explizit experimentelles Add-on kennzeichnen.
   **→ ZURÜCKGESTELLT 2026-07-12** (Option 1): als experimentelles Add-on gekennzeichnet;
   kein Stable-Weg existiert heute (s. Scoping-Doc), das no_std-30-Min-Experiment steht in
   FOLLOWUPS. Die nightly-Abhängigkeit ist Build-/Publish-Zeit, keine Endnutzer-Laufzeit.
10. **Backend-Wahl-API** (**ERLEDIGT 2026-07-12**, Commit 5b0f951 — docs/item-10-backend-api-*): ein `NDArray`-Surface, Backend-Wahl bei der
    Erzeugung (Datenplatzierung, primär Umgebungskriterien) — durch die 8/9-Entscheidung
    VEREINFACHT (stable-Backends bleiben synchron, kein async-Umbau): stable-Artefakt als
    Default überall, Threads als umgebungs-detektiertes Node-only-Opt-in. Per-Call-Routing
    zwischen Cores bleibt dokumentierte Sackgasse (FOLLOWUPS 2026-07-10, kern-06-Addendum).
    Beginnt mit Spec/Design; die offene Design-Gabel (ein auto-wählendes Surface vs. explizite
    Backends) geht VOR dem Spec-Freeze an den Owner.

### Phase D — Paketierung & Release

> **Status 2026-07-13: Vorarbeiten KOMPLETT, Item 11 ist der nächste Schritt.** Vor Item 11
> wurden per Owner-Entscheid drei Vorab-Scheiben + eine Mini-Scheibe gezogen (bindende Spec
> docs/phase-d-vorarbeiten-spec.md, Baustein-0-verifiziert; alle Scheiben dreifach verifiziert,
> ab V2 unter dem vollen Covenant-Regime): **V3** Browser-Smoke-Test (Playwright/Chromium als
> devDep, erster Real-Browser-Beweis des Standard-Surface, `pnpm test:browser`,
> docs/phase-d-vorarbeiten-v3-*); **V1** Union-Guard-Fix (never-wrong hält wieder für
> Union-Dims und Mixed-Rank-Shape-Unions; tuple-wrapped `Guard`, `RankUnknowable` an sieben
> Gates, docs/phase-d-vorarbeiten-v1-*); **V2** strides→readonly-Property, `WNDArray
> implements NDArrayView`, deep-readonly `shape` via `Readonly<S>`, plus Owner-entschiedene
> RE-Invariantierung per explizitem `__variance`-Marker nach ungeplantem Varianz-Befund
> (docs/phase-d-vorarbeiten-v2-*); **Union-Axis-Mini** (`IsUnion<Axis>`-Filter in ReduceAxis;
> Restlücke `Literal|undefined` via optionale Parameter als bekannter M2-Verstoß in
> COVENANT.md v2 dokumentiert, UA_GAP-Sentinel, docs/union-axis-mini-*). Außerdem seit dieser
> Runde: **COVENANT.md** (v1→v2) als stehender Produkt-Vertrag mit mechanischem S1-Lint,
> covenant-verify als Baustein C und Eskalationsleiter (CLAUDE.md „Qualitätssicherung").
> **Item 11 übernimmt zusätzlich:** Overload-Split-Entscheidung für die
> Optional-Parameter-Familie, `slice-literal.ts`-Umbenennung, npm-Namen sichern,
> Zero-dep-Guard-Test (alle in FOLLOWUPS).

> **Status 2026-07-17: Item 11 KOMPLETT** (S1+S2+S3, je dreifach verifiziert & committet:
> S1 48ee440, S2 87e6e6b, S3 69ab47a). **S1** sum-Overload-Umbau (COVENANT-M2-Verstoß
> geschlossen, beide Facetten) + `slice-literal.ts`→`literal-arithmetic.ts` (docs/item-11-s1-*).
> **S2** Emit-/Paket-Pipeline: `tsconfig.build.json` + zero-dep Post-Emit-Rewrite
> (`scripts/postbuild-dist.mjs`, fixt die drei TS7-Emit-Blocker) + `.wasm`-Bundling +
> package.json-Metadaten (`pnpm build:dist`; docs/item-11-s2-*). **S3** Zero-dep-Guard +
> Paket-Smoke als geprüfte Gates (`pnpm test:package`: Laufzeit-Smoke gegen `dist/index.js`
> inkl. `backend("wasm")`-WASM-Ladepfad, Konsumenten-Typ-Smoke, Emit-Präzisions-Gate
> `check-dist-emit.mjs`; docs/item-11-s3-*). COVENANT v2→v4 (M2 geschlossen, Z2 präzisiert).
> npm-Name-Sicherung + author-Feld + LICENSE-Datei bewusst nach Item 13 verschoben.

11. **API-Konsolidierung + Paketschnitt:** aus `spike/` ein Paket mit einem öffentlichen
    Surface; `.wasm`-Bundling, `exports`-Map, Hover-Qualität der `d.ts` prüfen (die Hovers
    sind Teil des Produkts). **✅ ERLEDIGT (2026-07-17).**
12. Qualitäts-Portfolio + CI: GitHub Actions mit allen Gates inkl. Artefakt-Hash-Freeze-Check.
    **✅ ERLEDIGT (2026-07-18).** 8-Job-CI (check/cargo/test-node/test-browser/test-threaded/
    freeze/editor-gate/demo) auf ubuntu-latest, Trigger `push:[main]` + `pull_request` (keine
    Doppelläufe); rustc 1.95.0 gepinnt (`rust-toolchain.toml`); Freeze-Hash-Gate mit
    Plattform-Hash-Menge (`scripts/check-freeze-hash.mjs`); `bench:editor` zu hartem Gate
    gehärtet (Correctness+Instantiation-Pins hart, Latenz 2x-Ceiling); `--test-timeout` (F6);
    zero-dep S1-CI-Guard (string-aware Import-Scanner). **Kein Vitest** — `node --test` bleibt
    (Owner-Entscheid, zero-dep). Dreifach verifiziert (A CONFORM + B fand+behob zwei
    S1-Guard-Multi-Line-Bypässe + C kein Verstoß); der erste echte CI-Lauf klärt den
    Linux-Freeze-Hash (D4). docs/item-12-ci-spec.md + docs/item-12-ergebnisse.md.
13. Release-Mechanik: npm-Namen sichern, Lizenz, 0.x-SemVer-Politik, README mit
    10-Sekunden-Demo-GIF, Begleit-Blog-Post; Forschungsnotizen als veröffentlichbare
    Artefakte aufbereiten (USP-Doc §8.3).
    **🔶 UNTERWEGS (2026-07-19).** Metadaten/Doku fertig + committet + gepusht:
    **Apache-2.0** (LICENSE + NOTICE + package.json; Owner-Entscheidung — Patent-Grant statt MIT,
    Commit 3d7e7ba), **README-Vollüberarbeitung** (ANSI-Shadow-figlet-Banner als Signature-Move +
    typecheck-verifizierte Beispiele + §5-Qualifikationen wörtlich + Zero-Dep-Abschnitt, bfdb01b/
    dd6012b), **`engines` node >=20** (2e33a65), **`author` crankfunk** (8e3c495). Privacy-Audit vor
    OSS gemacht (Repo sauber).
    **✅ ERLEDIGT (2026-07-19): numtype@0.1.0 ist LIVE auf npm, das Repo ist PUBLIC.**
    Release-Session: Pre-Flight verifiziert (Name frei, Tarball via `npm pack --dry-run` sauber,
    CI grün, keine CI-Secrets/Rulesets, Commits durchgehend noreply-Identität); drei Lücken
    vor dem Publish geschlossen (**NOTICE fehlte im Tarball** — npm nimmt LICENSE/README
    automatisch mit, die Apache-2.0-NOTICE nicht → in `files`; Version 0.0.0→**0.1.0**;
    **`prepublishOnly` = `pnpm test:package`** als Publish-Airbag); README um „Why NumType
    exists" (Motivation: fehlendes NumPy-Pendant im TS-Ökosystem) + Editor-out-of-the-box-
    Absatz ergänzt. Erst-Publish manuell mit 2FA, Registry-verifiziert (41 Files / ~435 kB
    unpacked); Tag `v0.1.0` (37335d0); GitHub-Metadaten gesetzt (Description, Homepage→npm,
    11 Topics). Fakten-Korrektur (npm-Doku): Trusted Publishing geht auch aus privaten Repos
    (Paket muss existieren, Konfiguration in den npm-Paket-Settings + `id-token: write`);
    nur das Provenance-BADGE braucht ein public Repo — Setup als optionales FOLLOWUP.
    Bewusst nach Item 14 verschoben: Demo-GIF, Blog-Post, Forschungsnotizen-Aufbereitung
    (FOLLOWUPS). Befund: das Threads-`.wasm` ist NICHT im Tarball (Checkout-only für v0).
14. **v0.1 research preview.**
    **✅ ERLEDIGT (2026-07-19) — die Roadmap ist damit durchgespielt.** Alle vier Bausteine:
    (a) Demo-GIF in der README (drei verifizierte Szenen, AppleScript-getippter Take),
    (d) README-Sektion „Versioning: what to expect before 1.0" (0.x-SemVer, inferierte Typen
    als Teil der API), (c) docs/README.md als englischer Reading Guide über die
    Forschungsnotizen, (b) Launch-Blog-Post „Teaching the type checker arithmetic"
    (https://marvinmuegge.com/notes/teaching-the-checker-arithmetic/; Code-Beispiel gegen das
    veröffentlichte Paket typecheck-verifiziert). Dazu im Zuge: Prior-Art-Credit
    (ts-arithmetic) in „The core idea".

## Release-Gates (Definition „release-fähig")

- Editor-Latenz bei realistischen Op-Ketten gemessen und akzeptabel (A1) — hartes Gate.
- Ein konsolidiertes API-Surface; alle Beispiele im README laufen wörtlich.
- Differentialtests inkl. Spezialwerte grün; CI reproduziert alle Gates inkl. Freeze-Hash.
- Threads entweder browser-fähig oder sauber als Node-only/experimentell abgegrenzt.
- USP-Qualifikationen (Geltungsbereich, „im Maßstab unbewiesen") stehen wörtlich in README/Docs.

## Post-Roadmap: Op-Wunschliste (OSS-Wachstumskurs, seit 2026-07-20)

Die Roadmap (Items 1–14) ist durchgespielt; neue Ops entstehen jetzt evidenzbasiert aus der
Dogfooding-Scheibe (docs/dogfooding-rag-ergebnisse.md, Wunschliste W1–W5). **W1 (argmax/topk):
implementiert 2026-07-20** (docs/op-w1-argmax-topk-spec.md/-ergebnisse.md, NDArray-only, kein
WASM-Kernel). **W2 (Skalar-Overloads add/sub/mul/div + mean): implementiert 2026-07-21**
(docs/op-w2-scalar-mean-spec.md/-ergebnisse.md, D6-v2-Overload-Umbau der vier Bestandsmethoden +
neue `mean`-Methode nach sum-Muster, NDArray-only, kein WASM-Kernel). **W3 (sqrt): implementiert
2026-07-21** (docs/op-w3-sqrt-spec.md/-ergebnisse.md, niladische, shape-erhaltende Klassenkörper-
Insertion, NDArray-only, kein WASM-Kernel — F1-Schließung der RAG-Demo-L2-Normalisierungskette
byte-identisch bewiesen). **W4 (stack): implementiert 2026-07-21** (docs/op-w4-stack-spec.md v2 +
Baustein-0-Addendum F1-F8/-ergebnisse.md, statische `NDArray.stack(rows)`-Methode nach
`fromArray`, `StackCheck`/`StackShape` in vector.ts, `RowShapesOf` in ndarray.ts, NDArray-only,
kein WASM-Kernel — F5-Schließung der `embedMatrix`-Zeilen-Flatten-Friction byte-identisch
bewiesen). **W5 (item): implementiert 2026-07-21** (docs/op-w5-item-spec.md v2 +
Baustein-0-Addendum F1-F8/-ergebnisse.md, `NDArray.item(...indices)` — voller Skalar-Read,
`ItemGuard`/`ItemMark`/`ItemFoldAcc` in vector.ts, `itemRuntime` in runtime.ts, NDArray-only,
M1 v5: kernel-los per Design (reiner strided Read, kein Kernel zu parallelisieren). **Damit ist
die komplette Op-Wunschliste W1–W5 abgearbeitet.**

## Post-Roadmap: Scale-Probe (ERLEDIGT 2026-07-21)

Der letzte Punkt der Owner-Reihenfolge vom 2026-07-20 („unproven at scale" in gemessenes
Terrain überführen) ist abgeschlossen — bindende Spec docs/scale-probe-spec.md v2, Ergebnisse
docs/scale-probe-ergebnisse.md, Vorab-Scheibe V0 (Mess-Basis-Reparatur), voller Verify-Katalog
A+B+C plus Baustein 0 und eine Frontier-Zweitmeinung. README und USP-Doc tragen jetzt die
gemessene Aussage statt des Vorbehalts, ausdrücklich auf die Konsumenten-Skala gescoped; die
API-Flächen-Skala bleibt als benannte offene Frage (FOLLOWUPS). Neuer Dauer-Wächter: Workload
`w8` im harten `bench:editor`-Gate. Damit sind alle drei Punkte der Owner-Reihenfolge erledigt
(Launch-Post streuen bleibt Owner-Aktion, Dogfooding-Scheibe und Scale-Probe sind durch).

## Post-Roadmap: topk-Selektion (Messung 2026-07-22, Umsetzung ERLEDIGT 2026-07-23)

Der in Verify-B der W1-Scheibe (F5) dokumentierte algorithmische Defekt — `topkRuntime` sortierte
das GESAMTE Array (O(n log n)), um die k größten Elemente zu finden — ist behoben. Phase 1
(bindende Mess-Spec docs/op-topk-selection-spec.md v6, vier Reparaturrunden der Entscheidungsregel
+ eine Frontier-Zweitmeinung) hat über ein 92-Zellen-Raster gemessen und mechanisch das Verdikt
**reiner Heap** (t* = 1,0) berechnet. Phase 2 (docs/op-topk-selection-ergebnisse.md,
Phase-2-Abschnitt) hat `topkRuntime` in-place durch den größenbeschränkten Max-Heap (O(n log k))
ersetzt — bit-identisch zur alten Full-Sort (konstruktiv per Totalordnung + empirisch per
Orakel-Differentialtest über 300+ Fälle inkl. exakter nicht-kanonischer NaN-Payload), voller
Verify-Katalog A+B+C alle grün. Größenordnung: `n = 1e6, k = 1` von 280 ms auf 3,8 ms (Faktor 74);
ehrliche Kehrseite sieben Zellen ab `k/n = 0,85` absolut langsamer (max. +13,95 ms bei `k = n`,
relativ 5 %, dual bewusst akzeptiert). NDArray-only, kein WASM-Kernel (M1 bindet nicht); ein
künftiger `nt_topk`-Kernel sollte den Heap spiegeln (FOLLOWUPS). Neuer Root-Pin 206.854 @ 140
(Δ+53, reine Typkosten). Der wertvollere Ertrag der Scheibe war methodisch: wie brüchig
vorregistrierte Entscheidungsregeln sind, wenn man sie nicht als Skript gegen synthetische Raster
durchspielt (die Regel wurde viermal gebrochen, bevor sie messen durfte), und dass die auslösende
informelle Sondage um mehr als eine Größenordnung danebenlag.

## Post-Roadmap: WASM-Parität-Kampagne S0–S5 (Owner-entschieden 2026-07-23)

Die W1–W5-Ops sind bewusst NDArray-only gelandet; diese Serie zieht `WNDArray`/threaded nach, damit
beide Backends dasselbe können (Vollständigkeit/Symmetrie, kein gemessener Nutzerbedarf — ehrlich so
benannt). Reihenfolge easy-first, dünne vertikale Scheibe zuerst: **S0 sqrt → S1 Skalar-Overloads →
S2 mean → S3 item/stack → S4 argmax → S5 topk**. **S0 (sqrt): ERLEDIGT 2026-07-23**
(docs/wasm-parity-sqrt-spec.md v3 /-ergebnisse.md): Rust/WASM-Kernel `nt_sqrt_strided` + niladische
`WNDArray.sqrt()`, threaded-Parität automatisch (dasselbe Crate). M1 bindet erstmals für eine der
neuen Ops und ist dreifach belegt (Baustein-0-Vorab-Probe 30.028 Fälle, committeter Differentialtest,
Baustein-B-BigInt-Oracle über 102.281 Elemente — je 0 Abweichungen). Voller Verify-Katalog A+B+C alle
grün, netto −4 check:diag, neuer Freeze-Hash. **Kampagnen-Gewinn (aus einem Umsetzungs-Befund):** ein
`Omit<ThreadedCoreExports,"memory">`-Cast in threaded.ts verursachte pro `CoreExports`-Member +7
Instantiations (`keyof`-getriebene Generic-Neuauflösung, kumulativ); der Fix (direkter Cast,
laufzeit-identisch, vierfach belegt) beseitigt das an der Wurzel — jede Folge-Scheibe kostet auf
diesem Mechanismus +0. S1–S5 folgen dem etablierten Workflow (Spec → Baustein 0 → Impl → Verify
A+B+C → Freeze-Re-Pin).
