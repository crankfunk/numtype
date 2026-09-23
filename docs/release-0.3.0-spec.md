# Release 0.3.0 „Parity + Polish" — Kurz-Spec (Stufe 3a)

**Version:** v1.1 (2026-09-23) · **Status:** Owner-abgenommen 2026-09-23 (Teilung 0a/0b, `toJSON` als `{ shape, data }`, check:diag-Gate ≤ +2.000) → implementierungsreif
**Stufe:** 3a Routine-Scheibe (CLAUDE.md, Eskalationsleiter) — kein Baustein 0, ein
kombinierter Verifier (Baustein R). **Hochstufen auf 3b**, falls der Freeze-Hash sich bewegt
oder ein Blocker auftaucht.
**Anlass:** Stand-Review 2026-09-23, Befunde F1–F7 (Roadmap „ab 2026-09-23", Phase 0a).

## Ziel

`main` als 0.3.0 veröffentlichen — die fertige WASM-Parität S0–S5 erreicht npm — und die vier
Konsumenten-Befunde schließen, die einen ersten Nutzer am härtesten treffen, ohne neue
Typ-Maschinerie.

## Umfang (D1–D6)

- **D1 — `WNDArray` als Typ exportieren (F2).** `export type { WNDArray }` aus
  `spike/src/index.ts`. Nur Typ-Export — der Wert bleibt über die Backend-Fabriken erreichbar
  (Konstruktor ist privat). M3: Hover `WNDArray<[2, 3]>` per LSP messen (Regel 13).
- **D2 — keine `node:`-Typimporte in der von `index.d.ts` erreichbaren Deklarationsmenge (F1).**
  Ursache: `threaded.d.ts` importiert `Worker` aus `node:worker_threads`, weil ein exportierter
  Typ ihn referenziert. Fix: an der Deklarationsgrenze einen lokalen strukturellen Typ statt
  `Worker` verwenden (Laufzeit unverändert, der dynamische Import bleibt). **Kein Subpath-Umbau,
  kein Breaking Change** — `export type { ThreadedBackend }` bleibt. Anker: M5 (Browser-Sicherheit,
  hier auf Typebene). Neues Gate: ein ZWEITER Konsumenten-Typ-Smoke mit `skipLibCheck: false`
  und ohne `@types/node` in `test:package` (D-S2.3 (a) wird damit verschärft, nicht ersetzt — der
  bestehende `skipLibCheck: true`-Smoke bleibt).
- **D3 — `toJSON()` und `inspect` auf `NDArray` und `WNDArray` (F7).**
  - `toJSON(): { shape: number[]; data: number[] }` — verlustfreier Round-Trip über
    `NDArray.fromArray(json.shape, json.data)` (logische Reihenfolge, Views korrekt). Offengelegte
    Grenze: `JSON.stringify` schreibt `NaN`/`±Infinity` als `null` — Standardverhalten, dokumentiert,
    nicht umgangen.
  - `[Symbol.for("nodejs.util.inspect.custom")]` → `NDArray<[2, 3]> [[1, 2, 3], [4, 5, 6]]`
    (browser-sicher, `Symbol.for` braucht keinen Node-Import). `toString()` liefert dieselbe Form.
  - Auf `WNDArray` werfen alle drei nach `dispose()` die benannte Disposed-Meldung. Nicht auf
    `NDArrayView` (kein neuer Member auf der kovarianten Fläche).
  - TS-Klassenkörper insertion-only.
- **D4 — totes v1-`backend.ts` aus dem Paket (F3).** `tsconfig.build.json` schließt
  `spike/src/wasm/backend.ts` aus; die Datei selbst bleibt (eingefrorene Performance-Baseline,
  demo/bench/tests nutzen sie). Gate: `dist/` enthält kein `backend.{js,d.ts}`, `test:package` grün.
- **D5 — README (F4/F5, Regel 19).** „New in 0.2.0" bleibt nur für das, was 0.2.0 enthielt; die
  S0–S5-Parität und D1–D3 wandern unter „New in 0.3.0". Statuszeile auf v0.3.
  `grep -n "TypeScript-runtime only\|no WASM kernel" README.md` gegen `spike/src/index.ts` prüfen.
- **D6 — Release-Mechanik.** `package.json` 0.3.0 · Release-Notes (Englisch) · Tag `v0.3.0`.
  **Reihenfolge (v1.1):** `examples/rag-demo` installiert aus der REGISTRY — der Bump auf `^0.3.0`
  + Lockfile + Demo-Lauf geht erst NACH `npm publish`, als eigener Folge-Commit.
  **`npm publish`, `git push` und das Tag-Push sind nach außen gerichtet → erst nach
  ausdrücklichem Owner-OK im Chat.**

## Nicht-Ziele

Typisiertes `toNestedArray` (Phase 0b, 3b) · neue Ops · Kernel-Änderungen · Subpath-Exports ·
Strukturumbau `spike/` → `src/` · Trusted Publishing (bleibt optional in FOLLOWUPS).

## Gates (Absolut-Grenzen)

- **Freeze-Hash unverändert** `2a54d9fd…` (kein Rust berührt) — bewegt er sich: Blocker, 3b.
- `pnpm check` Exit 0 · check:diag Root **≤ +2.000** gegen 225.983 @ 140 (neue niladische Member
  ripplen laut Mechanismus 3 kaum; Dateiset bleibt 140) · stress/browser berichten.
- test:core / test:resident / test:threaded / cargo zahlengleich oder nur um die neuen Tests
  gewachsen · `test:package` grün inkl. neuem `skipLibCheck: false`-Smoke · `bench:editor`:
  Pins nur bei uniformer, erklärter Verschiebung neu setzen · Lint (`graph-a-lama query lint`).

## Testplan

- D1: Typ-Pin `WNDArray<[2,3]>` aus `"numtype"` importierbar (Konsumenten-Smoke) + LSP-Hover.
- D2: neuer Smoke schlägt auf dem Stand VOR dem Fix fehl (TS2591) und ist danach grün —
  Nicht-Vakuität per Vorher/Nachher belegt.
- D3: Round-Trip `fromArray(toJSON())` für Rang 0, size-0, contiguous und jede View-Klasse
  (Klasse per Regel 12 assertiert); `inspect`-Ausgabe gepinnt für Rang 0/1/2 und size-0;
  Disposed-Throw auf `WNDArray`; NaN→`null` als dokumentierter Fall gepinnt.
- D4: Emit-Gate prüft die Abwesenheit von `backend.*` in `dist/`.
- Verifier R: zwei Mutanten (Pflicht), Stufen-Check.
