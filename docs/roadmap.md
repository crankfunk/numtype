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

> **Status 2026-07-11: Item 5 zur Hälfte gelandet — Kern 07** (sub/mul/div + dot/norm/
> cosineSimilarity auf beiden Surfaces, bit-identisch differential-getestet, unabhängig
> verifiziert CONFIRMED; docs/kern-07-*). Offen aus Item 5: Runtime-`reshape`/`flatten`
> (+ FOLLOWUPS-Messobligation Editor-Hover) und `keepdims`. Items 6/7 unverändert offen.

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

8. **Browser-Port des Threads-Pfads** (COOP/COEP, `crossOriginIsolated`, async Dispatch statt
   blockierendem `Atomics.wait` auf Main) — *oder* bewusste Entscheidung: Threads bleiben in
   v0 ein Node-only-Opt-in (vertretbar; das Standard-Artefakt baut heute schon auf stable).
9. no_std/stable-Pfad fürs Threads-Artefakt — der Release soll nicht an einem gepinnten
   Nightly hängen; Alternative: Threads als explizit experimentelles Add-on kennzeichnen.
10. **Backend-Wahl-API**: ein `NDArray`-Surface, Backend-Wahl bei der Erzeugung
    (Datenplatzierung, primär Umgebungskriterien) — erst nach 8/9; Per-Call-Routing zwischen
    Cores ist dokumentierte Sackgasse (FOLLOWUPS 2026-07-10, kern-06-Addendum).

### Phase D — Paketierung & Release

11. **API-Konsolidierung + Paketschnitt:** aus `spike/` ein Paket mit einem öffentlichen
    Surface; `.wasm`-Bundling, `exports`-Map, Hover-Qualität der `d.ts` prüfen (die Hovers
    sind Teil des Produkts).
12. Qualitäts-Portfolio + CI: Vitest-Migration (FOLLOWUPS), GitHub Actions mit allen Gates
    inkl. Artefakt-Hash-Freeze-Check.
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
