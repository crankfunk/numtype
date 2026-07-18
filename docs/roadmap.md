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
14. **v0.1 research preview.**

## Release-Gates (Definition „release-fähig")

- Editor-Latenz bei realistischen Op-Ketten gemessen und akzeptabel (A1) — hartes Gate.
- Ein konsolidiertes API-Surface; alle Beispiele im README laufen wörtlich.
- Differentialtests inkl. Spezialwerte grün; CI reproduziert alle Gates inkl. Freeze-Hash.
- Threads entweder browser-fähig oder sauber als Node-only/experimentell abgegrenzt.
- USP-Qualifikationen (Geltungsbereich, „im Maßstab unbewiesen") stehen wörtlich in README/Docs.
