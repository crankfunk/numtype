# dt5 — `sqrt`, `argmax`, `topk`, `stack` auf allen dtypes — bindende Spec (Stufe 3b)

**Version:** v1 (2026-10-14) · **Status:** Owner-abgenommen (A1–A4 je nach Empfehlung, 2026-10-14) → Baustein 0
**Stufe:** 3b — neue Typ-Maschinerie (`StackDTypeCheck`, ein Fold über die Zeilen-dtypes),
Covenant-Änderung (v13), Verhaltensänderung bestehender Ops (E6, A3 i).
**Berührte Invarianten:** M2 (Fold mit Union-Gate, niladische bool-Sperre von `sqrt()`, Ende der
übergangsweisen Sperren aus v8), M3 (`stack`-Ausnahme aus v8 entfällt; Meldungen am Argument),
M1 (v6 Datenbewegung: `topk`-Werte bit-exakt inkl. NaN-Payload; neue kernel-lose Referenzen), Z1.
**Grundlage:** docs/dtype-design-spec.md v2.1 (E5, E6, D5 Zeilen `sqrt`/`argmax`/`topk`/`stack`, D8) ·
dt1–dt4 und infra-02 auf `main` · KB-Lehre `float32array-zuweisung-macht-snan-ruhig` · Covenant v10
(niladische bool-Sperre, Präzedenz `norm()`).

## Ziel

Die letzten vier für Nicht-float64 gesperrten Ops rechnen auf allen dtypes. Danach gibt es keine
übergangsweise dtype-Sperre mehr: `DTypeLock` und die Laufzeit-Sperrhelfer entfallen, M2 v8 und die
`stack`-Ausnahme aus M3 v8 sind erledigt. dt5 schließt die dtype-Umsetzung auf `NDArray` ab; danach
folgt das Release 0.4.0 (eigener Schritt, README erst dann, Regel 19).

## Fakten am Code (2026-10-14, `main` = `66d69da`)

- `DTypeLock` (ndarray.ts:612) hat noch genau zwei Nutzer: `argmax` (3 Achsen-Überladungen,
  :1848/1851/1855) und `topk` (:1903). `assertFloat64Locked` sperrt zur Laufzeit `argmax`, `topk`,
  `sqrt` (:2002) und `stack` (:1094); `lockedOpMessage`/`assertFloat64Locked` (runtime.ts:1468/1472)
  werden nicht aus `index.ts` exportiert.
- `argmax()` liefert `number`, die Achsen-Formen `NDArray<…>` mit **float64**-Indizes;
  `topk` liefert `values` und `indices` beide als float64 (`topkRuntime` kopiert Werte
  `Float64Array` → `Float64Array`, NaN-Payload erhalten). `WNDArray.argmax`/`.topk` liefern
  ebenfalls float64-Indizes und bleiben bis dt6 float64-only (D9).
- `sqrt()` ist niladisch, Rückgabe `NDArray<S>` (float64).
- `stack` nimmt `readonly NDArray<any>[]` — das füllt `D` mit dem Default float64, deshalb lehnt der
  Editor eine Nicht-float64-Zeile heute mit der nativen TS2322 über den privaten `__varianceD`-Marker
  ab (M3-v8-Ausnahme, „bis dt5").
- Hausregel (b) greift: bestehende Member werden geändert; bestehende Tests nur nach A2.

## Umfang

- **F1 `argmax`** auf allen dtypes, inkl. bool (A3 iii). `argmax()` bleibt `number`. Achsen-Formen
  liefern `NDArray<…, "int32">` (E6, A3 i). Laufzeit: Vergleich als JS-Zahlen (exakt für alle
  dtypes), NaN-maximal und erster Index gewinnt wie heute; die bestehende float64-Referenz
  `argmaxRuntime` bleibt byte-gleich und ist das Orakel für den Index-Wert.
- **F2 `topk`** auf allen dtypes, inkl. bool. `values: NDArray<…, D>`, `indices: NDArray<…, "int32">`
  (E6). Werte sind reine Datenbewegung (M1 v6): float32 über eine `Uint32Array`-Sicht, float64 über
  `BigUint64Array` (Muster `whereTyped`, dt4) — NaN-Payloads inkl. signalisierender NaNs bleiben
  erhalten; per Test mit `0x7fa00001`. Totalordnung wie heute (`topkRuntime` bleibt Orakel).
- **F3 `sqrt`**: float64 → float64 (bestehende Referenz), float32 → float32 (`Math.fround(Math.sqrt(x))`,
  korrekt gerundet nach Figueroa, D8), int32 → float64. **bool:** niladisch, also keine
  Compile-Sperre möglich → dauerhafte Laufzeit-Sperre mit `BOOL_ARITHMETIC_MESSAGE` (Präzedenz
  `norm()`, M2 v10); Rückgabe-dtype für bool: A3 ii.
- **F4 `stack`**: alle Zeilen gleicher dtype → Ergebnis `NDArray<…, D>`; verschiedene dtypes → eigene
  Meldung am Argument, wortgleich zur Laufzeit (keine Promotion über Zeilen, D5). Neuer Fold
  `StackDTypeCheck` über die Zeilen-dtypes: `IsAnyDType`/`IsUnion` JE ZEILE zuerst (eine Zeile mit
  Union- oder `any`-dtype → kein Anspruch, Laufzeit prüft; Arbeitsregel 3), dann Gleichheit. Werte
  sind Datenbewegung: bit-exakt je dtype (Ganzzahl-Sicht für float32/float64). Die M3-v8-Ausnahme
  (native `__varianceD`-Meldung) entfällt.
- **F5 Aufräumen:** `DTypeLock` gelöscht (null Nutzer). `lockedOpMessage`/`assertFloat64Locked`
  gelöscht (null Aufrufer, nicht öffentlich; A1). Doc-Kommentare der vier Ops aktualisiert.
- **F6 Tests und Pins** — Laufzeittests an `scalar-mean.test.ts` (bzw. `argmax-topk.test.ts`, falls
  sinnvoller; Dateiset unverändert), **Typ-Pins in `spike/tests-dtype/dtype.test-d.ts`** (infra-02),
  Pins auf Alias-Ebene (`StackDTypeCheck` direkt) plus Aufruf-Pins JE Op und JE Überladung, inkl.
  Ergebnis-Shape mit Broadcast/Achse und Union-Operanden (Lehren dt3/dt4). Laufzeit: Index-Werte
  gegen die float64-Orakel auf konvertierten Daten, `Int32Array`-Klasse der Indizes, `topk`/`stack`
  bit-exakt mit sNaN-Mustern in float32 und float64, `sqrt` float32 gegen eine unabhängige
  `Float32Array`-Referenz (inkl. −0, Subnormale, negative Werte → NaN), int32 → float64, bool-Sperre
  (auch size-0), `stack` mit gemischten dtypes (Laufzeit-Meldung), Union-Zeile. Diagnose-INHALTE per
  tsc-Fixture (Regel 2), Hovers per LSP (Regel 13). Mutanten auf Kopien außerhalb des Repos, Typ-
  Mutanten gegen `pnpm check` (Verify-Template Regel 8).
- **F7 Covenant v13** (A4), FOLLOWUPS (neue kernel-lose Referenzen; dt1-K7-Eintrag schließen;
  `WNDArray`-Asymmetrie der Index-dtypes bis dt6).

**Nicht in dt5:** `WNDArray`/WASM (dt6+) · Promotion über `stack`-Zeilen · Release 0.4.0 (eigener
Schritt) · die vorbestehenden FOLLOWUPS-Befunde aus dt4 (Shape-Meldung, sNaN in transpose/slice/
astype, Achse 1.5).

## Owner-Abnahmen vor Baustein 0

**A1 — Methodenliste (Hausregel b):** Signaturen und Körper von `argmax` (4 Überladungen), `topk`,
`sqrt`, `stack`. Modulebene: neu `StackDTypeCheck` (+ Hilfstypen für den Fold), gelöscht `DTypeLock`.
`runtime.ts`: neue Funktionen nur angehängt; gelöscht werden `lockedOpMessage` und
`assertFloat64Locked` (dt1-Helfer, nicht öffentlich, danach ohne Aufrufer) — die einzige Löschung in
`runtime.ts`, bestehende Referenzfunktionen bleiben byte-gleich.

**A2 — bestehende Tests, die sich zwangsläufig ändern** (Baustein 0 belegt die Liste empirisch):
1. die Sperr-Tests für `argmax`/`topk`, `sqrt` und `stack` in `scalar-mean.test.ts` → durch Tests der
   neuen Semantik ersetzt (bool bei `sqrt` bleibt als Sperr-Test);
2. der Cross-Layer-Test (Editor zeigt `lockedOpMessage` wortgleich), seit dt3 auf `argmax(0)` → entfällt
   ersatzlos (es gibt keine übergangsweise Sperre mehr);
3. **zusätzlich wegen E6:** jeder bestehende Test, der für `NDArray.argmax(axis)`/`topk` float64-
   Indizes (`Float64Array`, `dtype "float64"`) festhält — erwartet in `argmax-topk.test.ts` und
   gegebenenfalls in Differential-Tests gegen `WNDArray` (`resident.test.ts`). Baustein 0 zählt sie.

**A3 — vier Design-Entscheidungen (ENTSCHIEDEN 2026-10-14: alle nach Empfehlung):**
- **(i) E6 jetzt auch für float64:** `argmax(axis)` und `topk().indices` liefern int32 für JEDEN
  Empfänger-dtype, auch float64 — eine Verhaltensänderung gegenüber 0.3.0 (Klasse `Int32Array` statt
  `Float64Array`, Werte gleich) und eine Asymmetrie zu `WNDArray` (float64-Indizes bis dt6).
  **Empfehlung: ja** — E6 ist entschieden, 0.4.0 ist ohnehin das dtype-Release, und ein Index-dtype,
  der vom Werte-dtype abhängt, wäre schlechter. Alternative: float64-Empfänger behalten float64-
  Indizes bis dt6 (konsistent zu `WNDArray`, inkonsistent innerhalb von `NDArray`).
- **(ii) `sqrt()` auf bool:** Laufzeit-Sperre wie `norm()` (v10). Rückgabe-dtype für bool:
  **Empfehlung `never`** (`NDArray<S, never>` — ehrlich: es entsteht nie ein Wert; gleiches Bild wie
  abgelehnte Aufrufe seit dt2). Alternative `"float64"` (eine Behauptung, die nie eintritt). Baustein 0
  prüft, ob `never` Folgefehler in Ketten erzeugt.
- **(iii) `argmax`/`topk` auf bool:** **Empfehlung erlauben** — D5 nennt für beide keine Sperre, beide
  vergleichen nur (keine Arithmetik, E5 unberührt), NumPy erlaubt es.
- **(iv) `stack` mit verschiedenen dtypes:** **Empfehlung: eigene Meldung am Argument**
  (z. B. „stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first"),
  wortgleich zur Laufzeit; eine Zeile mit Union-/`any`-dtype → kein Anspruch.

**A4 — Covenant v13 (Wortlaut abgenommen 2026-10-14):**

> **M2 · Präzisierung v13 — Ende der übergangsweisen Sperren (Owner-entschieden 2026-10-14):** seit dt5
> rechnet jede Op auf `NDArray` auf jedem dtype; die übergangsweisen Sperren aus v8 und `DTypeLock`
> entfallen. Dauerhaft nur zur Laufzeit gesperrt bleiben niladische Ops auf bool nach v10 — jetzt
> `norm()` und `sqrt()`. `stack` prüft die Gleichheit der Zeilen-dtypes über `StackDTypeCheck`
> (dtype-Maschinerie nach v8–v12: eine Zeile mit `any`- oder Union-dtype → kein Anspruch). Anker:
> `sym:DTypeLock` entfernt, `sym:StackDTypeCheck` ergänzt.
>
> **M3 · v13:** die befristete Ausnahme `stack` (v8) entfällt — eine dtype-Ablehnung von `stack`
> zeigt die eigene Meldung am Argument, wortgleich zur Laufzeit.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit, jeder mit Stopp;
der Stopp gilt für die ERSTE vollständige Messung, Optimieren erst nach Rückmeldung)

1. **Commit A:** F1 + F2 + F3 (ohne Tests). Messen (Root + dtype). **Stopp** bei Root-Δ > **+2,500**.
2. **Commit B:** F4 + F5. Messen. **Stopp** bei Root-Δ kumuliert > **+4,000**.
3. **Commit C1:** Laufzeittests + A2-Änderungen. Messen. **Stopp** bei Root-Δ kumuliert > **+5,500**.
4. **Commit C2:** Typ-Pins (dtype-Korpus) + FOLLOWUPS; Covenant-Text vorab als eigener Commit.
5. **Commit D:** bench:editor-Re-Pin, falls die Klassen-Surface die Pins verschiebt (zweimal gemessen).

**Tests und Pins werden NIE gestrichen, um unter eine Grenze zu kommen.**

## Gates (vorregistriert, Absolutwerte)

- **Root** `check:diag` **≤ +5,500** gegen 246,763 @ 140 (seit infra-02 = Produkt + Laufzeittests,
  ohne Typ-Pins). Herleitung, offen: dt4-Implementierung +3,821 für 6 Ops + `where` + `any`/`all`;
  dt5 ändert 4 Ops und fügt einen Fold hinzu, entfernt `DTypeLock` aus 4 Signaturen; Laufzeittests
  kosteten dt3 +1,503, dt4 +971. Baustein 0 misst einen Entwurf und kann das Gate VOR der Umsetzung
  neu begründen.
- **dtype-Korpus** `check:diag:dtype`: nur Pin, kein Gate (infra-02 O3); Abdeckung per Mutanten.
- stress/browser berichten · bench:editor nur mit erklärter Verschiebung neu pinnen · Freeze-Hash
  unverändert · alle übrigen Tests unverändert grün (außer A2) · Lint · Autorenprüfung vor dem Push.

## Verify

Baustein 0 (`brainroute:deep`, eigener Worktree) vor dem Code: A2 empirisch (inkl. E6-Folgen in
`argmax-topk.test.ts` und Differential-Tests gegen `WNDArray`), Entwurfsmessung, `StackDTypeCheck`
ohne Loch (Union-Zeile, `any`-Zeile, leere Zeilenliste, Rang-Mischung), `never` als Rückgabe-dtype
für `sqrt()` auf bool im Editor (LSP) und in Ketten, sNaN-Erhalt in `topk`/`stack`. Danach A + B + C
parallel (A auf deep). Jeder Agent-Auftrag trägt Arbeitsregel 22, den sicheren `node_modules`-Link
(`[ -e node_modules ] || ln -sn …` nur im Worktree-Root, Abschlussprüfung auf Selbst-Link im
Haupt-Tree) und die Anweisung, den Worktree zuerst per `git merge --ff-only` auf den Zielstand zu
bringen.
