# WASM-Parität S3 — `item` + `stack` auf `WNDArray`/threaded (Ergebnisse)

Spec: [docs/wasm-parity-item-stack-spec.md](wasm-parity-item-stack-spec.md) v2 · Datum: 2026-07-23/24 ·
Eskalationsleiter Stufe 3 (voller Verify-Katalog A+B+C) · Covenant v5.

**Ehrlichkeitsregel dieses Dokuments:** Jede Zahl stammt aus einem tatsächlich gelaufenen Kommando;
was nicht gemessen wurde, steht als nicht gemessen da. Diese Scheibe hat **keinen gemessenen
Nutzerbedarf** — niemand ist auf eine WASM-`item`- oder `stack`-Wand gestoßen. Es ist
Vollständigkeits-/Symmetrie-Arbeit, die vierte Scheibe der Kampagne S0–S5.

## Was die Scheibe liefert

`WNDArray` — der WASM-residente Zwilling von `NDArray` — kann jetzt beides, was ihm aus der
W4/W5-Wunschliste fehlte:

- **`item(...indices)`** liest einen einzelnen Skalar direkt aus dem residenten Speicher, statt dass
  der Aufrufer das ganze Array über `toArray()` nach JS kopieren muss.
- **`WNDArray.stack(core, rows)`** setzt N residente Rang-1-Zeilen zu einer `[N, D]`-Matrix zusammen,
  ohne die Daten je aus dem WASM-Speicher herauszuholen. Erreichbar über beide Backend-Facaden
  (`WasmBackend.stack`, `ThreadedBackend.stack`).

Threaded-Parität fällt automatisch mit (dasselbe Crate, derselbe `nt_materialize`) und ist getestet.

## Der tragende Befund: beide Ops sind kernel-los

Arbeitsregel 11 (aus S2/mean: „vor jedem neuen Kernel prüfen, ob die Op definitorisch eine
Komposition bereits verifizierter Kernel ist") hat hier zum zweiten Mal in Folge gegriffen — und in
zwei verschiedenen Ausprägungen:

- **`item` ist kernel-los per Design.** Ein Einzelwert-Lesezugriff hat keine Arithmetik, die ein
  Kernel beschleunigen könnte: Offset aus `(strides, offset)` des Handles ausrechnen, einen `f64`
  über eine frisch abgeleitete `Float64Array`-View lesen, fertig. Denselben Lesepfad benutzen
  `toArray()`s contiguous-Zweig und die Skalar-Rückgaben von `dot()`/`norm()` seit Kern 02/07.
- **`stack` ist Datenbewegung über einen bereits eingefrorenen Kernel.** `nt_materialize` (Kern 03)
  sammelt eine beliebig strided View contiguous ein. `stack` ist N solcher Aufrufe, jeder in den
  `i`-ten Sub-Slot **eines** frisch allozierten Ausgabepuffers (`outPtr + i*d*8`, `out_len = d`).

Konsequenz, dieselbe wie bei S2: **der Freeze-Hash bleibt unverändert**, und der Beweis kippt von
einer teuren positiven Behauptung („der neue Hash ist legitim, hier die Dekomposition") in eine
billige und strengere negative Assertion — er DARF sich nicht bewegen. Kein Rust, kein ABI-Eintrag,
kein neuer `CoreExports`-Member, damit auch kein `backend-oom`-Stub (Arbeitsregel 10 geprüft, greift
bestätigt nicht).

## Was Baustein 0 vor dem Bau gefangen hat

Der adversariale Vor-Verifier bestätigte das Kern-Design empirisch (Sub-Slot-`nt_materialize` über
sechs Konfigurationen byte-exakt; `d === 0` ohne Gate; die zwei neuen Helfer über je 20.000
Zufallsfälle wortgleich zu den Orakeln; `item` auf Views über 15.000 Fälle abweichungsfrei; die
Typ-Aliase kollabieren nicht) — und fand zwei Verdrahtungs-Befunde, die vor der ersten Codezeile in
die Spec mussten:

1. **BLOCKER: `ThreadedBackend` hat kein `core`-Feld.** Die v1-Spec behauptete, beide Facaden bekämen
   eine „strukturell identische" Delegation über `this.core`. Tatsächlich trägt die Klasse
   `readonly pool: ThreadedPool`, der Core lebt auf `this.pool.core` — die bestehenden
   `fromArray`/`zeros`/`ones` machen es bereits so. Live als TS2339 reproduziert. Exakt die
   Fund-Klasse, für die dieser Prozessschritt existiert (Präzedenz Item 10: `WNDArray.strides` ist
   ein Feld, keine Methode).
2. **MAJOR: die Facaden-Signaturen brauchten eine Typ-Verdrahtung, die die Änderungstabelle nicht
   nannte.** Die `ndarray.ts`-Vorbilder `UnwrapRow`/`RowShapesOf` sind file-privat; ein wörtlicher
   Spiegel wäre es auch, dann können die Facaden ihre Signatur nicht ausdrücken. Die v2-Lösung geht
   über den Vorschlag des Verifiers hinaus: statt vier neuer Typ-Importe je Facade exportiert
   `resident.ts` zwei fertige Signatur-Aliase (`StackRowsGuard`/`StackResultOf`). Das vermeidet jede
   neue Import-Kante, hält die Unwrap-Aliase file-privat wie ihre Vorbilder — und ist zusätzlich die
   günstigere Variante, weil `ndarray.ts:118-125` gemessen dokumentiert, dass TS textgleiche, separat
   geschriebene Mapped-Type-Ausdrücke nicht dedupliziert (hier: drei Call-Sites).

### Der wertvollste Befund war eine Bestätigung mit Zähnen

Die Spec-Auflage „Strides-View pro Iteration frisch ableiten" war als Vorsichtsmaßnahme formuliert.
Sie ist keine. Baustein 0 hat in drei Proben nachgewiesen:

1. `nt_materialize`s interne `Vec`-Allokation löst `memory.grow` tatsächlich aus (Byte-Länge 1,1 MB
   auf 130 MB über 40 Aufrufe);
2. ein Wachstum **detacht** den alten `ArrayBuffer` vollständig (`byteLength` auf 0), und ein
   Schreibversuch über die veraltete View ist ein **stiller No-Op** — kein Throw;
3. ein gezielter Mutant (View einmal vor der Schleife abgeleitet) produziert **`status === 0` bei
   komplett falschen Ausgabedaten**.

Diese Fehlerklasse hat weder Statuscode noch Exception. Sie ist damit die gefährlichste der Scheibe,
und sie ist nur beobachtbar, wenn der Speicher während des Loops wirklich wächst. Konsequenz in v2:
Pflicht-Mutant **M-d** plus die bindende Auflage, dass die Testsuite einen Fall enthält, der das
Wachstum nachweislich auslöst und es assertiert — sonst wäre der Mutant nicht fangbar und die
Disziplin ungesichert.

## Umsetzung

| Datei | Δ | Art |
| --- | --- | --- |
| `spike/src/runtime.ts` | +87 / −0 | Append ans Dateiende: `itemOffsetStrided`, `stackValidateShapes` |
| `spike/src/wasm/resident.ts` | +205 / −2 | Insertion (Typ-Aliase, `stack`, `item`); die 2 Deletions sind die zwei sanktionierten Import-Zeilen-Erweiterungen |
| `spike/src/wasm/backend-api.ts` | +13 / −1 | Insertion `WasmBackend.stack` (`this.core`) |
| `spike/src/wasm/threaded.ts` | +18 / −1 | Insertion `ThreadedBackend.stack` (`this.pool.core`) |
| 6 Testdateien + 2 Typ-Test-Dateien | +1.458 / −7 | Anhänge an bestehende, registrierte Korpus-Dateien |
| `spike/bench-dx/editor-latency.ts` | +33 / −? | 8 Instantiation-Pins neu gesetzt |

Disziplin am Diff nachgeprüft: `runtime.ts` hat genau **einen** Hunk, bei Zeile 1038 — reiner Append
nach `itemRuntime`; `itemRuntime`/`stackRuntime` sind byte-unberührt (sie sind in dieser Scheibe das
Orakel). `resident.ts` hat drei Insertions (Typ-Aliase nach `squeezeMatmulShape`, `stack` nach
`fromArray`, `item` am Klassenende) plus zwei Import-Zeilen; kein Edit an einem Bestandsmember. Alle
Deletions im Quellcode-Teil des Diffs sind Import-Zeilen. `crates/` ist im Diff nicht enthalten.

### Warum die Prüf-Logik dupliziert wird

Die zwei neuen Helfer reproduzieren die Prüf-Reihenfolge und die Message-Stämme von
`itemRuntime`/`stackRuntime`, statt dass diese refaktoriert würden, sie aufzurufen. Grund: die
Referenzfunktionen sind in DIESER Scheibe das Korrektheits-Orakel. Ein Paritäts-Beweis, der das
Orakel im selben Commit umbaut, beweist weniger — eine gemeinsame Regression wäre für den
Differentialtest unsichtbar. Der Preis ist Drift-Risiko, und er wird nicht weggeredet, sondern
mechanisch abgesichert: ein Test nagelt alle sechs Fehlermeldungen cross-surface auf
String-Gleichheit fest, mit Mutant M-c als Nicht-Vakuitäts-Beweis. Die Konsolidierung (Orakel auf die
Helfer umstellen) geht als eigener FOLLOWUPS-Mini raus.

## Messung

Baseline im frischen `git worktree` von HEAD `755d9ac`, alle Exit 0 und exakt die dokumentierten
Pins: `check:diag` 213.704 @ 140, stress 107.283 @ 82, browser 2.142 @ 75, Freeze `8255821b…`,
`bench:editor` 8 Pins exakt mit Hard-CI-Gate PASS.

| Stufe | Wert | Δ |
| --- | --- | --- |
| ① `runtime.ts`-Helfer | 213.704 | **0** (reine Funktionen ohne Typ-Maschinerie) |
| ② `resident.ts` (item + stack + Aliase) | 214.815 | +1.111 |
| ③ Facaden | 216.447 | +1.632 |
| ④ Test-Anhänge (6 Dateien) | 220.281 | +3.834 |
| ⑤ Typ-Pins (2 Dateien) | 221.081 | +800 |
| ⑥ Verify-Runden-Fixes (3 Tests, s. u.) | 221.192 | +111 |
| ⑦ Hover-Fix (Baustein-B-Befund) | **226.690 @ 140** | **+5.498** |
| **Gesamt** | **226.690 @ 140** | **+12.986** (Gate ≤ +13.000 nach Owner-Anhebung, Marge 14) |

Dateiset unverändert 140 — kein Order-Noise, alle Anhänge gingen in bestehende Dateien.
`check:diag:stress` 115.498 @ 82 (Δ+8.215, Klassen-Surface-Ripple, dritter Mechanismus — davon
+5.453 aus dem Hover-Fix). `check:diag:browser` 2.142 @ 75 (Δ0 — kompiliert weder `threaded.ts` noch
die Test-Korpora). `check:freeze` **unverändert** `8255821b…`, `cargo test` unverändert 184+1.

### Die Gate-Anhebung: eine Abwägung, keine Nachlässigkeit

Stufe ⑦ ist der Grund, warum das in v2 vorregistrierte Gate (≤ +8.000) in v3 auf ≤ +13.000 angehoben
wurde. Die Entscheidung liegt beim Owner und wurde mit den gemessenen Zahlen vorgelegt, nicht still
getroffen. Der Sachverhalt:

Baustein B stellte mit einer eigenen LSP-Hover-Sonde gegen den echten `tsc --lsp --stdio`-Server
fest, dass die drei `stack`-Signaturen mit dem v2-Rückgabetyp-Alias `StackResultOf<Rows>` als
`StackResultOf<readonly [WNDArray<[3]>, WNDArray<[3]>]>` hovern — an der **einzigen für
Paketkonsumenten erreichbaren Fläche**, mit `add`s sauberem `WNDArray<[2, 3]>` als Kontrollpunkt in
derselben Messung. Ein Top-Level-Typ-Alias in RÜCKGABE-Position wird von der Quick Info namentlich
erhalten; ein Alias in TYP-ARGUMENT-Position wird aufgelöst. Der Fix ist entsprechend klein
(`StackShapeOf` trägt nur die Shape, `WNDArray<StackShapeOf<Rows>>` schreibt das Handle aus) und von
B mit derselben Sonde als wirksam nachgemessen: alle drei Aufrufformen hovern jetzt
`WNDArray<readonly [2, 3]>`.

Drei billigere Wege wurden gemessen und verworfen, bevor die Anhebung vorgelegt wurde: die
Konsolidierung der Facaden-Typ-Pins trägt **−21** bei (nicht der Treiber); eine Aufteilung auf zwei
Aliase ist mit 227.020 **teurer**; und die Test-Aufrufstellen laufen bereits über dynamische Shapes,
die Stufe-④-Technik greift hier also nicht mehr. Die +5.498 sind strukturell.

Die tragende Begründung ist nicht Bequemlichkeit, sondern der Vertrag: COVENANT M3 verlangt
„Klassen-Hover bleiben saubere Tupel (`NDArray<[2, 3]>`)", und mit dem Alias stand dort keine
Klassen-Typ-Parameter-Anzeige mehr. Das Gate war demgegenüber eine scheiben-lokale Budgetgrenze,
die diese Scheibe sich selbst gesetzt hatte; 226.690 sind ca. 4,5 % des Instantiation-Budgets.

### Ein neuer Kostenmechanismus-Datenpunkt (Stufe ④)

Stufe ④ maß zunächst **+6.705**, mehr als doppelt über der in D8 vorregistrierten
Konsolidierungs-Schwelle von +3.000. Per Leave-one-out- und Bisektions-Messung isoliert: ein
`WNDArray.stack(core, [a, b])`-Aufruf, dessen Empfänger eine konkrete literale Shape trägt, zahlt die
volle `StackFold`/`WRowShapesOf`-Maschinerie **pro Aufrufstelle**; derselbe Aufruf über ein Array mit
dynamischer Shape nimmt den billigen No-Claim-Pfad. Nach Umstellung von ca. 15 Laufzeit-Testaufrufen
auf dynamische Shapes („widen past the guard", im Repo etablierte Technik) fiel die Stufe auf +3.834.

**Offengelegte Abweichung:** Die Konsolidierung erfolgte NACH Stufe ⑤, nicht — wie D8 verlangt — vor
dem Weiterbauen; und die Stufe liegt mit +3.834 weiterhin über der Schwelle. Das Gesamt-Gate hält mit
Marge. Der Verlust an Typ-Ebenen-Exposition in den Laufzeittests ist folgenlos, weil die Typ-Ebene
ausschließlich in den `test-d.ts`-Pins geprüft wird — das ist eine Behauptung, die Baustein A gezielt
gegenprüft hat (Verdikt im Addendum).

### `bench:editor`

Die acht Pins wurden in dieser Scheibe **zweimal** gesetzt. Erste Runde: Hard-Gate-FAIL bei
fast-uniformer Bewegung (+2.722 bis +2.761, Spanne 39), neu gepinnt auf
`{w1 31.542, w2 33.351, w3 64.493, w4 31.697, w5 36.996, w6 38.166, w7 30.722, w8 38.550}`. Zweite
Runde nach dem Hover-Fix: erneut FAIL, Bewegung +5.361 bis +5.500 (Spanne 139), Endstand
`{w1 37.018, w2 38.851, w3 69.993, w4 37.173, w5 42.472, w6 43.666, w7 36.222, w8 43.911}`, danach
PASS. Beide Runden zweifach gemessen, je byte-identisch.

Die Bewegung ist der dritte Mechanismus (Klassen-Surface-Ripple), diesmal über **drei** Klassen statt
einer — `WNDArray`, `WasmBackend`, `ThreadedBackend`. Anders als bei S0–S2 ist sie **nicht perfekt
uniform**: w8 ist der einzige Workload mit einem eigenen `stack`-Aufruf und zahlt daher eine andere
Grenzkosten-Mischung als die übrigen sieben, die ausschließlich für das Klassen-Surface zahlen, nie
für eine Aufrufstelle. Baustein A hat die Richtung des Effekts als unbegründet moniert (w8 bewegt
sich am WENIGSTEN, nicht am meisten) — das bleibt offen: belegt ist, dass w8 der einzige Workload mit
Aufrufstelle ist, hergeleitet ist nicht, warum das die Grenzkosten senkt statt sie zu erhöhen. Es ist
ein Genauigkeitsmangel der Begründung, kein Korrektheitsproblem, weil der Gate-Wert ein exakter
Wertvergleich ist.

## Tests

| Suite | vorher | nachher | Δ |
| --- | --- | --- | --- |
| `test:core` | 1.591 | 1.591 | 0 |
| `test:resident` | 5.048+2 | 5.497+2 | +449 |
| `test:threaded` | 101 | 114 | +13 |
| `cargo` | 184+1 | 184+1 | 0 |
| `test:package` | 3 | 3 | 0 |

Abdeckung nach Arbeitsregel 12 (residente Op-Tests müssen VIEWS treffen, nicht nur
`fromArray`-contiguous): `item` wird über alle vier View-Klassen geprüft — transponiert, geschnitten
mit nonzero offset, offset-verschoben, zusammengesetzt — plus rank 0, negative Indizes und
size-0-Achsen; `stack` über gemischt contiguous/View-Zeilen, `N=1`, `D=0`, `D=1`, große `N`,
aliasende Zeilen, Zeilen aus fremden Puffern und den Cross-Core-Fehlerpfad.

### Die vier Pflicht-Mutanten

| Mutant | Änderung | Gefangen von | Nicht-Vakuität |
| --- | --- | --- | --- |
| **M-a** | `item`: `this.strides`/`this.offset` → `computeStrides(shape)`/`0` | 95 Fails, **alle** in den vier View-Blöcken | **0** Fails im contiguous-Grid, rank 0, size-0, Message-Parität, `stack` — der Beweis, dass die View-Abdeckung trägt und contiguous allein nicht genügt |
| **M-b** | `stack`: Slot-Offset `+ i*d*8` entfernt | 58 Fails, alle Mehrzeilen-Fälle | 0 Fails bei `N=1`/`D=0`/`item` (bei `N=1` ist Offset 0 ohnehin korrekt) |
| **M-c** | Message-Stamm „expected" → „expecting" | genau **1** Fail: der Arity-Wortgleichheits-Test | keine falsch-positiven — der Paritätstest ist nicht-vakuös |
| **M-d** | Strides-View einmal vor der Schleife statt pro Iteration | genau **1** Fail: der dedizierte `memory.grow`-Test | das 60-Fälle-Zufallsraster fängt ihn **nicht** — der Beweis, dass der großskalige Fall nötig ist, exakt wie D8 verlangt |

Jeder Revert per Backup-Kopie mit `diff`-Beweis auf Byte-Identität (nie `git checkout` — der
Haupt-Baum trug uncommittete Arbeit); nach jedem Revert lief die betroffene Suite wieder vollständig
grün.

## Befund am Testplan der Spec: `getResidentFreeCount` sieht das Scratch nicht

D7/T7 verlangte für `stack` einen Free-Count-Delta, der „exakt die zwei Scratch-Freigaben pro Aufruf"
zeigt. Am Quellcode geprüft ist das mechanisch unmöglich: `getResidentFreeCount()` inkrementiert
ausschließlich in `releaseBuffer()` — also wenn ein refgezählter `ResidentBuffer` eines
`WNDArray`-Handles auf 0 fällt —, nie über den rohen `nt_free`-Pfad, den `allocBytes`/`freeBuf` für
ephemeres Scratch nutzen. Anders als `mean` (S2) hat `stack` kein internes `WNDArray`-Zwischenhandle;
nur der Ausgabepuffer wird je Aufruf zu einem echten `ResidentBuffer`.

Umgesetzt wurde deshalb der tatsächlich beobachtbare Mechanismus: ein Free-Count-Test über die
Handles (Delta `3×N` aus zwei Input-Zeilen plus einem Ergebnis je Iteration, was zugleich die
Gesundheit des Zählers belegt) plus ein separater `byteLength`-Plateau-Test speziell für die zwei
Scratch-Puffer — dieselbe Technik, die die bestehenden „failing ops leak nothing"-Tests schon nutzen.
Das substanzielle Ziel („kein Netto-Leck") ist damit abgedeckt, über das verfügbare Signal statt über
einen Zähler, der es nicht sieht. Die Spec-Formulierung war an dieser Stelle schlicht falsch; sie
wurde nicht still „repariert", sondern als Befund dokumentiert.

Der in der Spec als optional markierte Mid-Loop-Fehlerpfad („falls deterministisch erzwingbar, sonst
als bewusst ungetestet benennen") war deterministisch erzwingbar — per Mock-Core nach dem Vorbild von
`backend-oom.test.ts` — und ist getestet.

## Covenant

Berührt: M1, M2, M3, M4, M5, Z1, Z2 (S1 mechanisch per `graph-a-lama query lint`: 0 errors, 0
warnings nach Graph-Rebuild). M4 ist die zentrale negative Assertion und hält: der Freeze-Hash ist
unverändert, `crates/` steht nicht im Diff.

Drei Auslegungsfragen gehen als Owner-Entscheidung an FOLLOWUPS, keine still aufgelöst:

1. **M1 für `item`** — ein vierter, im Vertragstext unbenannter Fall: kein WASM-Code läuft, aber die
   Op arbeitet auf WASM-residenten Daten und ist bit-identisch getestet. M1 v5 kennt wörtlich nur
   „jeder WASM-Kern" und „kernel-lose Referenz-Op" (Op existiert NUR in `runtime.ts`).
2. **M1 für `stack`** — zweite Instanz des schon offenen S2-Kandidaten (komponierte Op ohne eigenen
   Kern).
3. **M3 und die Cross-Surface-Message-Parität** — M3s Wortlaut adressiert Typ-vs-Runtime innerhalb
   EINER Fläche; der hier bindende Test prüft Runtime-vs-Runtime zwischen ZWEI Flächen. Der Test
   bleibt Pflicht, nur die Zuordnung ist eine Dehnung.

## Post-Verification-Addendum

Drei frische Kontexte, parallel, mit disjunkten Fragen. Alle drei haben Befunde geliefert; **fünf
davon führten zu Änderungen am Code oder an den Tests, einer zu einer Owner-Entscheidung.**

### Baustein A (Spec-Konformität) — CONFIRMED mit einer Auflage

Patch im eigenen Worktree angewendet, alle Gates frisch gefahren, jede behauptete Zahl exakt
reproduziert (Freeze-Hash, check:diag 221.081 @ 140 im damaligen Stand, stress, browser, die acht
bench:editor-Pins, test:core/resident/threaded/package, cargo). D1–D11 und T1–T11 einzeln geprüft und
bestätigt. Der Disziplin-Check am Diff bestätigt: `runtime.ts` nur append, keine Bestandsmember
editiert, kein Rust.

Die drei vom Implementierer offengelegten Abweichungen wurden gezielt geprüft:

- **Stufe-④-Konsolidierung: kein Coverage-Verlust.** A hat belegt, dass Node die `.ts`-Testdateien
  per nativem Type-Stripping ausführt — Typannotationen sind zur Laufzeit vollständig gelöscht, eine
  dynamisch typisierte Shape erzeugt denselben JS-Wert wie ein Tupel-Literal, und die Assertions
  prüfen reale Laufzeitwerte. Zusätzlich belegt: die Technik ist nicht neu erfunden, sondern stand
  schon vor S3 in `resident-lifecycle.test.ts`.
- **`getResidentFreeCount`-Diskrepanz: mechanisch bestätigt** (der Zähler kann ephemeres Scratch
  strukturell nicht sehen) — siehe aber die Auflage.
- **`bench:editor`-Spanne:** reproduziert; die Einzigartigkeit von w8 ist per grep belegt, die
  Richtung des Effekts nicht hergeleitet (minor, offen dokumentiert).

**Die Auflage (major, empirisch belegt):** A's eigener Pflicht-Mutant entfernte die beiden
`scratch.push(...)`-Aufrufe in `stack` — ein echtes, dauerhaftes Scratch-Leck. Der eigens dafür
geschriebene Leck-Test blieb **grün**: das Leck beträgt 8 Byte pro Aufruf, 500 Iterationen ergeben
ca. 4 KB, und WASM-Speicher wächst in 64-KiB-Seiten — die Plateau-Prüfung war für diese Größenordnung
vakuös. Gefangen wurde der Mutant nur von 1 der 5.496 Tests, dem Mock-basierten Fehlerpfad-Test, der
zählt statt misst.

**Geschlossen:** Die Plateau-Prüfung wurde durch eine exakte Alloc/Free-Bilanz über denselben
Zähl-Mock ersetzt, der bisher nur den Fehlerpfad abdeckte — jede Allokation muss genau eine passende
Freigabe haben. Nicht-Vakuität mit exakt A's Mutant bewiesen: der neue Test fällt und benennt die
beiden fehlenden 4-Byte-Blöcke mit Adresse (`allocs` 5 Einträge, `frees` 3). Revert per Backup-Kopie
mit SHA-256-Abgleich, nie `git checkout`.

### Baustein B (adversarial) — HÄLT mit Befunden

Das stärkste Ergebnis ist ein Negativbefund: B's **eigene, unabhängige Orakel** — eigener
Speicher-Lese-Code und eigene Transpose/Slice-Simulation statt `itemRuntime`/`stackRuntime`/
`toArray()` — fanden über **6.000 `item`-Fälle**, **600 `stack`-Fälle** und **837 threaded-Fälle**
**0 Abweichungen**. Dazu ein selbst gebauter `memory.grow`-Stresstest (N=40, D=60.000): der Speicher
wuchs während des `stack`-Aufrufs live von 30,2 MB auf 49,9 MB, das Ergebnis blieb über 2,4 Millionen
Werte bit-identisch — und gegen B's eigenen Stale-View-Mutanten produzierte derselbe Aufbau
1.199.980 Abweichungen ohne einen einzigen Throw. Die Gefahrenklasse ist real, die Disziplin trägt sie.

Drei MAJOR-Befunde, **alle drei geschlossen:**

1. **Liveness-Prüfung nur für Zeile 0 getestet.** B's Mutant (`assertLive` nur für `rows[0]` statt
   Schleife) lief durch alle 1.280 Tests grün. Die echte Implementierung ist korrekt; getestet war
   sie nicht. Live-Probe gegen den Mutanten: eine disponierte Zeile 1 liefert stille Fehldaten aus
   freigegebenem Speicher, kein Diagnose-Signal — genau die Fehlerklasse, die diese Scheibe als ihre
   gefährlichste behandelt. **Geschlossen:** ein Test mit disponierter MITTLERER und disponierter
   LETZTER Zeile; unter B's Mutant fällt er, nach Revert ist er grün.
2. **Core-Prüfung gegen den `core`-Parameter ungetestet.** Der vorhandene Cross-Core-Test hatte
   `rows[0]` immer im selben Core wie den Parameter — für ihn sind die beiden Designs
   ununterscheidbar. B's Mutant (`r.core !== rows[0].core`) lief durch alle 1.261 Tests grün.
   **Geschlossen:** ein Test, in dem ALLE Zeilen untereinander übereinstimmen und vom Parameter
   abweichen; unter B's Mutant fällt er.
3. **Hover-Regression an der Konsumenten-API** — siehe die Gate-Anhebung oben. Der wertvollste Fund
   der Runde, weil er nicht die Implementierung traf, sondern eine Design-Entscheidung der Spec v2.

Zwei MINOR-Befunde bleiben dokumentiert, nicht geschlossen: die M3-Zuordnung der
Cross-Surface-Paritätstests (bereits als v6-Kandidat in FOLLOWUPS) und eine ungetestete, aber korrekt
gehandhabte Degradations-Kombination (`readonly [2, number]` bei gleicher nicht-literaler Zeilenlänge).

B benennt selbst, was es NICHT geprüft hat: Nebenläufigkeit zwischen laufendem `threadedMatmul` und
gleichzeitigem `stack`/`item` (Nicht-Ziel der Spec, aber nicht empirisch nachgestellt), Subnormal-
Randfälle bei der Integer-Prüfung, die Instantiation-Zahlen (Baustein-A-Aufgabe), und ein eigener
Mutant auf dem Mid-Loop-Fehlerpfad.

### Baustein C (Covenant) — kein Verstoß

Erster Lauf: kein Verstoß gegen S1, M1, M2, M3, M4, M5, Z1, Z2 und kein implementiertes Nicht-Ziel.
`crates/` per `git diff --stat`/`git status` als leer bestätigt; alle Anker existieren weiter; die
einzige `covenant-*`-Lint-Regel ist mit S1 verknüpft und ohne Waise in beide Richtungen. C hat
außerdem eine Präzisierung beigetragen, die keiner der anderen sah: der eigentliche M3-Gehalt für
`WNDArray` (Guard-Ablehnungsstamm gleich eigener Runtime-Throw) ist nur **transitiv** erfüllt —
unveränderte Guards plus wortgleiche Helfer —, nicht als ein direkter Pin geführt.

Die drei Auslegungsfragen wurden bestätigt und mit Empfehlungen versehen; alle drei sind als
v6-Kandidaten in FOLLOWUPS eingetragen (M1 vierter Fall für `item`, M1 komponierte Ops durch `stack`
ein zweites Mal instanziiert, M3 Cross-Surface-Achse).

### Baustein C, zweiter Lauf: M3 mit der Hover-Evidenz

Der erste C-Lauf hatte B's LSP-Messung noch nicht. Nachgereicht und gezielt beurteilt:

- **M3 im finalen Diff: eingehalten.** `WNDArray<StackShapeOf<Rows>>` ist Typ-Argument-Position und
  damit strukturell identisch zum bereits etablierten `add`-Präzedenzfall
  (`WNDArray<OkShape<Broadcast<S, B>>>`), den C im Diff selbst verifiziert hat. C hat die LSP-Sonde
  nicht nachgestellt (read-only-Auftrag, kein Build) und übernimmt B's Messung als Fakt, gestützt auf
  diese strukturelle Übereinstimmung — das ist so offengelegt, nicht als eigene Messung ausgegeben.
- **Der Zwischenstand WAR ein echter M3-Verstoß.** C bestätigt das ausdrücklich: M3 nennt als
  Exemplar wörtlich `NDArray<[2, 3]>` — Klassenname plus aufgelöstes Shape-Tupel. Die alte Fassung
  zeigte an der einzigen konsumentenseitig erreichbaren Fläche weder den Klassennamen noch ein
  aufgelöstes Shape, sondern einen Aliasnamen mit den ROW-Typen als Parameter. „Keine Grauzone,
  sondern das exakte Muster, das M3 verhindern soll." **Der Fix war damit Pflicht, nicht Kür.**
- **Selbstkorrektur von C, unaufgefordert:** Im ersten Lauf hatte C genau diese Fassung gelesen und
  als M3-konform durchgehen lassen — geprüft worden waren nur die M2-Struktureigenschaften, nicht das
  Hover-Verhalten der Rückgabeposition. Die Lücke wurde erst durch eine echte LSP-Messung sichtbar,
  nicht durch Diff-Lesen. Das ist der belastbarste Prozess-Befund dieser Runde (siehe unten).
- **Die Gate-Anhebung berührt keine Invariante.** `bench:editor`/`INSTANTIATION_PINS` kommt in
  COVENANT.md nicht vor, weder als Anker noch als Zahl; M2 regelt Ablehnungs-Korrektheit, nicht
  Instantiation-Kosten; Z2 betrifft Korpus-Vollständigkeit, nicht Schwellwerte.
- **Die drei neuen Tests öffnen keine neue Vertragsfläche.** Erwähnenswert nur: der
  `core`-Parameter-vs-`rows[0]`-Test stärkt nebenbei die Beweislage für das Nicht-Ziel „kein
  Per-Call-Routing zwischen Backend-Cores", weil er ausschließt, dass eine Verwechslung
  Cross-Core-Mischung durchrutschen lässt.

### Prozess-Lehre dieser Runde

**Eine Vertragsnorm über Editor-Verhalten lässt sich nicht aus dem Diff verifizieren.** Drei
Instanzen lasen denselben Quelltext: der Implementierer, Baustein A (Spec-Konformität, alle Gates)
und Baustein C (Vertrag) — alle drei sahen `StackResultOf<Rows>` als Rückgabetyp und keiner
beanstandete ihn. Gefunden hat es nur die Instanz, die den Editor **gestartet** hat. C hat seinen
eigenen Fehlschluss anschließend unaufgefordert benannt. Für künftige Scheiben, die eine Hover-
oder Diagnose-Norm berühren: die Prüfung braucht eine LSP-Messung mit einer Bestandsmethode als
Kontrollpunkt, kein sorgfältiges Lesen — das Repo hat die Harness dafür seit Spike 02.

Der zweite, gleichrangige Befund: **drei der fünf geschlossenen Lücken waren vakuöse Tests**, und
alle drei wurden ausschließlich durch Mutanten sichtbar, nicht durch Lesen. Ein Test, der grün
bleibt, wenn man das Geprüfte kaputt macht, ist von einem echten Test durch Lektüre nicht zu
unterscheiden.
