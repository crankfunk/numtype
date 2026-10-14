# dt5 — `sqrt`, `argmax`, `topk`, `stack` auf allen dtypes — bindende Spec (Stufe 3b)

**Version:** v1.1 (2026-10-14) · **Status:** Owner-abgenommen (A1–A4, Nachträge A5–A8 nach Baustein 0),
Baustein 0 gelaufen (ein Blocker, eingearbeitet) → **implementierungsreif**
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
  `norm()`, M2 v10). Rückgabe-dtype `SqrtDType<D>` = **`IsAnyDType` zuerst** (Baustein-0-Blocker D-1:
  ohne das Gate behauptet `AnyNDArray.sqrt()` falsch `never`) → `IsUnion` → bool `never`, sonst wie
  `ReduceDType` (float32 → float32, sonst float64). Offengelegt und per Pin festgehalten (A6): auf
  einem `never`-Ergebnis zeigen drei Folgefälle eine native Meldung über den privaten
  `__varianceD`-Marker (Zuweisung an `AnyNDArray`, `stack` mit dieser Zeile, Skalar-`add`) — nur in
  Code, der beim `sqrt()` ohnehin wirft.
- **F4 `stack`**: alle Zeilen gleicher dtype → Ergebnis `NDArray<…, D>`; verschiedene dtypes → eigene
  Meldung am Argument, wortgleich zur Laufzeit (keine Promotion über Zeilen, D5). Neuer Fold
  `StackDTypeCheck` über die Zeilen-dtypes: `IsAnyDType`/`IsUnion` JE ZEILE zuerst; eine Zeile mit
  Union- oder `any`-dtype wird ÜBERSPRUNGEN und weitergefaltet (Baustein 0 D-3: reihenfolge-
  unabhängig, kein Anspruch über diese Zeile, Laufzeit prüft; Arbeitsregel 3), dann Gleichheit der
  übrigen. Festlegungen (Baustein 0 D-4): Shape-Fehler gehen vor dtype-Fehlern (Compile- und
  Laufzeit); die Meldung nennt den dtype der ersten definiten Zeile und den der ersten abweichenden;
  Ergebnis-dtype bei Union-/`any`-Zeilen bzw. Array-Eingaben (`NDArray<…>[]`) die Vereinigung bzw.
  `any`. A1 umfasst dafür auch `UnwrapRow`/`RowShapesOf` (Modulebene; `NDArray<infer S>` füllte `D`
  mit float64). Der Meldungstext existiert als Typ-Template und als Laufzeitfunktion — die
  Wortgleichheit ist per tsc-Fixture PFLICHT (Baustein 0 C-3). Offengelegt: der Fold verteuert die
  quadratische Tupel-Faltung (400 Zeilen: 309k → 483k Instantiations, kein Tiefenfehler). Werte
  sind Datenbewegung: bit-exakt je dtype (Ganzzahl-Sicht für float32/float64). Die M3-v8-Ausnahme
  (native `__varianceD`-Meldung) entfällt.
- **F5 Aufräumen:** `DTypeLock` gelöscht (null Nutzer). `lockedOpMessage`/`assertFloat64Locked`
  gelöscht (null Aufrufer, nicht öffentlich; A1). Doc-Kommentare der vier Ops aktualisiert.
- **F6 Tests und Pins** — (v1.1 ergänzt: Reihenfolge Shape vor dtype bei `stack` per tsc-Fixture mit
  Inhaltsprüfung, da `@ts-expect-error` den Mutanten „dtype vor Shape" nicht tötet (Baustein 0 T-1);
  `sqrt` mit `any`-/`AnyNDArray`-/Union-Empfänger (T-2); float64-sNaN-Tests als „pinnt die Garantie,
  tötet keinen Mutanten ohne Sicht" formuliert (V8, T-3); `topkTyped`/`stackTyped` direkt mit
  `byteOffset ≠ 0` (T-4); bool-Fälle nur mit 0/1 (T-6); die drei `never`-Folgefälle als Pins (A6);
  Known-Limit-Pins für offene Typparameter bei `stack`/`argmax(0)` (D-6).) — Laufzeittests an `scalar-mean.test.ts` (bzw. `argmax-topk.test.ts`, falls
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
3. **zusätzlich wegen E6 (v1.1, empirisch, A7):** `argmax-topk.test.ts` bleibt unberührt (vergleicht
   über `Array.from`). Betroffen ist `resident.test.ts:2452` (topk-Cross-Surface-Pin vergleicht die
   Indizes bitweise als float64 → eine Zeile: Indizes vor dem Vergleich nach float64 wandeln).
4. **wegen F5 (v1.1, A7):** in `scalar-mean.test.ts` der Import von `lockedOpMessage` (sonst lädt die
   ganze Datei nicht), die Hilfsfunktion `lockedMsgRegex` und die dann ungenutzte Konstante
   `NON_FLOAT64`. Kompilier-notwendige A2-Edits liegen im SELBEN Commit wie die Produktänderung, die
   sie erzwingt — kein Commit mit rotem Baum (Baustein 0 F-4).

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

**A5–A8 (nach Baustein 0, ENTSCHIEDEN 2026-10-14):** A5 Root-Gate ≤ +2,500 mit Stopps +500/+1,800/
+2,500 · A6 `never` für `sqrt()` auf bool bleibt (mit `any`-Gate, die drei Folgefälle dokumentiert und
gepinnt) · A7 A1/A2 erweitert (s. o.) · A8 v13-Wortlaut nachgeschärft (s. u., bestätigt).
Offengelegt ohne Entscheidungsbedarf: `argmax`/`topk` akzeptieren bool, Vergleiche lehnen bool ab
(D-5, bewusste Asymmetrie, in v13 benannt); E6 bricht Konsumenten, die `indices` als `NDArray<[k]>`
annotieren (README.md:188, examples/rag-demo — gehört in Release-Notes und README-Schritt von 0.4.0,
D-7); `argmax`/`topk` widen Nicht-float64 auf eine temporäre f64-Kopie (20M Elemente: 51–116 ms, D-8).

**A4 — Covenant v13 (Erstfassung abgenommen 2026-10-14; durch A8 ersetzt):**

> **M2 · Präzisierung v13 — Ende der übergangsweisen Sperren (Owner-entschieden 2026-10-14):** seit dt5
> rechnet jede Op auf `NDArray` auf jedem dtype; die übergangsweisen Sperren aus v8 und `DTypeLock`
> entfallen. Dauerhaft nur zur Laufzeit gesperrt bleiben niladische Ops auf bool nach v10 — jetzt
> `norm()` und `sqrt()`. `stack` prüft die Gleichheit der Zeilen-dtypes über `StackDTypeCheck`
> (dtype-Maschinerie nach v8–v12: eine Zeile mit `any`- oder Union-dtype → kein Anspruch). Anker:
> `sym:DTypeLock` entfernt, `sym:StackDTypeCheck` ergänzt.
>
> **M3 · v13:** die befristete Ausnahme `stack` (v8) entfällt — eine dtype-Ablehnung von `stack`
> zeigt die eigene Meldung am Argument, wortgleich zur Laufzeit.

**A8 — v13 nachgeschärft (Baustein 0 C-1, C-2, D-5) — vom Owner bestätigt 2026-10-14:**

> **M2 · Präzisierung v13 — Ende der übergangsweisen Sperren (Owner-entschieden 2026-10-14):** seit dt5
> ist keine Op auf `NDArray` mehr übergangsweise für einen dtype gesperrt; `DTypeLock` und die
> übergangsweisen Sperren aus v8 entfallen. Dauerhaft abgelehnt bleibt bool-Arithmetik (v9–v11;
> Vergleiche lehnen bool-Operanden ab), während `argmax`/`topk` bool als reine Ordnungsoperationen
> akzeptieren — eine bewusste Asymmetrie. Niladische Ops auf bool lehnen nach v10 nur zur Laufzeit
> ab, jetzt `norm()` und `sqrt()`; `sqrt()` trägt den Rückgabe-dtype `SqrtDType` (bool → `never` als
> Compile-Zeit-Signal, dass kein Wert entsteht; `any` und Unions → `DType`). `stack` prüft die
> Gleichheit der Zeilen-dtypes über `StackDTypeCheck` (dtype-Maschinerie nach v8–v12; Zeilen mit
> `any`- oder Union-dtype werden übersprungen, kein Anspruch über sie). Anker: `sym:DTypeLock`
> entfernt, `sym:StackDTypeCheck` und `sym:SqrtDType` ergänzt.
>
> **M3 · v13:** die befristete Ausnahme `stack` (v8) entfällt — eine dtype-Ablehnung von `stack`
> zeigt die eigene Meldung am Argument, wortgleich zur Laufzeit; Shape-Fehler gehen vor. Benannte,
> hingenommene Folgemeldungen: auf einem Ergebnis von `sqrt()` über bool (dtype `never`, das zur
> Laufzeit ohnehin wirft) können Folgeaufrufe eine native Meldung über den privaten
> `__varianceD`-Marker zeigen.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit, jeder mit Stopp;
der Stopp gilt für die ERSTE vollständige Messung, Optimieren erst nach Rückmeldung)

1. **Commit A:** F1 + F2 + F3 + die dadurch erzwungenen A2-Edits (A2.1 für argmax/topk/sqrt, A2.3). Messen
   (Root + dtype). **Stopp** bei Root-Δ > **+500** (Baustein 0: −573).
2. **Commit B:** F4 + F5 + die dadurch erzwungenen A2-Edits (A2.1 stack, A2.2, A2.4). Messen. **Stopp**
   bei Root-Δ kumuliert > **+1,800** (Baustein 0: +899).
3. **Commit C1:** Laufzeittests. Messen. **Stopp** bei Root-Δ kumuliert > **+2,500** (Baustein 0: +1,031).
4. **Commit C2:** Typ-Pins (dtype-Korpus) + FOLLOWUPS; Covenant-Text vorab als eigener Commit.
5. **Commit D:** bench:editor-Re-Pin, falls die Klassen-Surface die Pins verschiebt (zweimal gemessen).

**Tests und Pins werden NIE gestrichen, um unter eine Grenze zu kommen.**

## Gates (vorregistriert, Absolutwerte)

- **Root** `check:diag` **≤ +2,500** gegen 246,763 @ 140 (v1.1, A5: Baustein 0 maß +1,031; das
  ursprüngliche +5,500 war ~5× zu locker) (seit infra-02 = Produkt + Laufzeittests,
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

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-10-14)

`brainroute:deep`, eigener Worktree auf `296e1f6`, Entwurf F1–F5 plus Laufzeittests und ~120 Zeilen
Typ-Pins (Patch im Scratchpad, nicht Merge-Kandidat). **Ein Blocker (D-1, `any`-Gate in `SqrtDType`),
keine falsche Code-Annahme** (alle Zeilenangaben bestätigt). **A2 empirisch:** mit den Original-Tests
auf dem Entwurf: `scalar-mean.test.ts` lädt ohne `lockedOpMessage` nicht (mit Shim 4 rote Tests,
genau die erwarteten), `resident.test.ts` 20 rote Fälle an einer Zeile, `argmax-topk.test.ts`,
`threaded`, `elementwise`, `special-values`, `backend-api` unberührt; kein Typ-Pin betroffen.
**Messung:** Root −573 (A), +899 (A+B), +1,031 (Endstand mit Laufzeittests); dtype-Korpus +4,423 für
die Pins; bench:editor alle 8 Pins verschoben (w1–w7 −49 = stress-Δ, w8 +198 ungeklärt → in D
erklären). **Semantik** gegen BigInt- und Brute-Force-Referenzen ohne Abweichung (sqrt float32
korrekt gerundet, int32 → float64, argmax/topk alle dtypes, sNaN in topk/stack). **Mutanten:** 11
Laufzeit (3 V8-unbeobachtbar, 1 äquivalent), 12 Typ (überlebt: Guard-Reihenfolge → Fixture-Pflicht;
redundanter Array-Schutz → streichen oder bewusst nicht pinnen). Eingearbeitet: D-1, D-3, D-4, A2,
A1, Commit-Schnitt, Gate, F6-Lücken, v13-Wortlaut.

## Nachträge nach der Umsetzung (2026-10-14)

- **F4 überholt (Owner-Entscheidung nach C1):** `stack` kopiert Zeilen per `TypedArray.prototype.set`
  statt über Ganzzahl-Sichten — ECMA-262 SetTypedArrayFromTypedArray kopiert bei gleichem Elementtyp
  byteweise (normativ; von Baustein A und B an der Quelle bestätigt). `topk`/`where` behalten ihre
  Sichten (Gather über Indizes).
- **Gate:** Root +2,533 gegen +2,500 (Stopp nach C1 ausgelöst), vom Owner akzeptiert.
- Ergebnisse, Verify-Runde und offengelegte Grenzen: docs/dtype-dt5-ergebnisse.md.
