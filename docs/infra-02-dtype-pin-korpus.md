# Infra 02 — dtype-Typ-Pins in einen eigenen Messkorpus — Kurz-Spec (Stufe 3a)

**Version:** v1 (2026-10-14) · **Status:** Owner-abgenommen (O1–O4 je nach Empfehlung, 2026-10-14) → Umsetzung
**Stufe:** 3a — Infrastruktur nach etabliertem Muster (Präzedenz infra-01, docs/infra-01-stress-split.md):
keine Typ-Maschinerie, kein Kernel (Freeze-Hash bleibt), keine Covenant-Änderung (Z2 sieht „künftige"
Quelltext-Korpora im Verbund `pnpm check` bereits vor). Verify: EIN kombinierter Verifier (Baustein R).
**Berührt:** Z2 (Verbund), Messregeln (Pin-Basis), Hausregel (b) (Umzug bestehender Pins, O4).

## Warum

Drei dtype-Scheiben in Folge überschritten ihr Root-Gate, jedes Mal ausschließlich durch Typ-Pins im
Root-Korpus: dt2 +720 über Gate, dt3 +1,501, dt4 +4,966 (Gate +9,000, Ist +13,966; die dt4-Pins
allein +9,139). Root misst damit Testkosten statt Produktkosten, und jedes künftige Gate lädt
entweder zum Kürzen der Abdeckung ein (dt3) oder wird routinemäßig überschritten (dt4). infra-01 hat
dasselbe Problem für die Digit-Arithmetik-Stressfälle gelöst: getrennt wird die MESSUNG, nie die
PRÜFUNG.

## Fakten am Code (2026-10-14, `main` = `50d7f0e`)

- Alle dtype-Typ-Pins liegen in `spike/tests/ndarray.test-d.ts` (2,471 Zeilen): dt1 ab Zeile ~1583
  (dazu der K5-Abschnitt um Zeile 230), dt2 ab ~1678, dt3 ab ~1875, dt4 ab ~2110. Keine andere
  `.test-d.ts`-Datei erwähnt dtypes.
- Root-`tsconfig.json` schließt `spike/tests-stress` und `spike/tests-browser` aus; `pnpm check` =
  Root + stress + browser; `check:diag:stress` misst den Stress-Korpus getrennt (standalone
  tsconfig, Optionen inline gespiegelt, kein `extends` — infra-01-Footgun).
- Root heute 264,285 @ 140, stress 123,138 @ 82, browser 2,142 @ 75.

## Umfang

1. Neuer Korpus `spike/tests-dtype/` mit standalone `tsconfig.json` (Muster `spike/tests-stress`) und
   einer Datei `dtype.test-d.ts`.
2. **Reiner Umzug:** die dtype-Abschnitte aus `ndarray.test-d.ts` wandern zeichengleich dorthin;
   was sie an Importen und `declare const`s brauchen, wird im neuen File dupliziert (nicht aus dem
   alten entfernt, wenn andere Pins es nutzen). Querverweis-Kommentar in beide Richtungen. Kein Pin
   wird geändert, gestrichen oder hinzugefügt (Umfang: O1).
3. Root-`tsconfig.json`: `spike/tests-dtype` in `exclude`. `package.json`: `check` wird Vierer-Verbund,
   neu `check:diag:dtype`. CI-Job „check" ruft `pnpm check` (prüfen, ob die Job-Beschreibung
   „triple-verbund" angepasst werden muss).
4. **Nicht-Vakuität:** (a) eine umgezogene Erwartung in einer Scratch-Kopie verfälschen → `pnpm check`
   MUSS fehlschlagen; (b) drei Typ-Mutanten aus dt3/dt4, die heute von einem umgezogenen Pin getötet
   werden, auf einer Kopie von `spike/src` anwenden → sie MÜSSEN weiterhin getötet werden, jetzt
   über den neuen Korpus.
5. **Mutanten-Werkzeuge der Folge-Scheiben** prüfen künftig `pnpm check` (Verbund), nicht nur Root —
   in den Verify-Auftrag (docs/verify-runde-template.md) als Satz aufnehmen.
6. Neue Pins: Root (Erwartung: deutlich unter 264,285), `check:diag:dtype`, stress/browser
   unverändert; bench:editor unverändert (die Workloads hängen nicht an Testdateien).
7. Doku: CLAUDE.md (Pins, Messregel „nie über den Root/Stress/dtype-Split vergleichen"), Projekt-Log,
   FOLLOWUPS-Eintrag schließen, dt5-Spec-Vorlage: neue dtype-Pins gehören in den neuen Korpus.

**Nicht-Ziele:** Laufzeittests umziehen (O1) · Pins inhaltlich ändern · neue Pins · Gate für den
dtype-Korpus festlegen (O3) · Covenant-Änderung.

## Owner-Entscheidungen

- **O1 Umfang:** nur die dtype-`.test-d.ts`-Pins (**Empfehlung**) — oder auch die dtype-Laufzeittests
  aus `scalar-mean.test.ts`? Die kosten im Root ebenfalls (dt3 C1 +1,503, dt4 C1 +971), sind aber in
  den expliziten Test-Skript-Listen registriert und enthalten die tsc-Fixture-Tests; ein Umzug wäre
  eine größere, eigene Operation.
- **O2 Ziel:** eigener Korpus `spike/tests-dtype/` (**Empfehlung**) — oder in den bestehenden
  Stress-Korpus? Dieser ist per infra-01 für Digit-Arithmetik-Extremfälle definiert; ihn zu mischen
  verwischt, was sein Pin misst.
- **O3 Gate-Modell danach:** Root-Gate misst wieder Produkt + Laufzeittests; der dtype-Korpus
  bekommt einen Pin, aber kein Gate (wie stress) — seine Qualität sichert der Mutanten-Nachweis, nicht
  ein Budget (**Empfehlung**). Alternative: eigenes Gate auch für den dtype-Korpus.
- **O4 Hausregel (b):** der Umzug ändert eine bestehende Testdatei (Abschnitte entfernt), ohne einen
  Pin inhaltlich zu ändern — Abweichung von „no pre-existing test may be modified", Präzedenz
  infra-01 („reine Relocations, Erwartungen zeichengleich"). Beweis der Zeichengleichheit per
  Skript (entfernte Zeilen = eingefügte Zeilen, Mengenvergleich). Freigabe nötig.

## Gates (vorregistriert, absolut)

- `pnpm check` (Vierer-Verbund) Exit 0 · Nicht-Vakuität 4(a) und 4(b) belegt · Zeichengleichheit
  des Umzugs per Skript belegt · Root `check:diag` ≤ 264,285 (darf nur sinken; der Messwert wird neuer
  Pin) · stress 123,138 und browser 2,142 unverändert · alle Tests unverändert grün (test:core 1668,
  resident 6155+2, package 3) · bench:editor unverändert grün · Freeze-Hash unverändert · Lint 0 ·
  Autorenprüfung vor dem Push.

## Reihenfolge

1. Commit A: Korpus + tsconfig + Umzug + package.json + CI. Messen (alle vier Korpora).
2. Commit B: Verify-Template-Satz (Punkt 5).
3. Verifier R (eigener Worktree), dann Doku-Commit und Push.

## Ergebnis (2026-10-14)

Umgesetzt in `b30d704` (Korpus, Umzug, Verbund, CI-Jobname) und `d29f64d` (Verify-Template Regel 8).
**Zahlen:** Root 264,285 → **246,763 @ 140** (Δ−17,522), neuer Pin **dtype 67,108 @ 81**, stress
123,138 und browser 2,142 unverändert, bench:editor und Freeze-Hash unverändert, test:core 1668,
resident 6155+2, package 3. **Umzug:** 898 Zeilen (alt 230–236 und 1583–2471) zeichengleich verschoben,
geändert nur zwei Importzeilen. **Verifier R (MERGE):** eigene Zählmethode bestätigt die
Relocation (z. B. `Expect<` 466 = 194 + 272, `@ts-expect-error` 116 = 64 + 52); acht eigene
Typ-Mutanten werden ausschließlich über den dtype-Korpus gefangen, der Root-Check allein übersieht
alle acht; Rulesets verlangen keine Status-Checks, die Umbenennung des CI-Jobs ist unbedenklich.
Nebenbefunde: veraltete Querverweise in Bestandsdateien (FOLLOWUPS); Regel 8 präzisiert
(Laufzeit-Mutanten weiterhin gegen die Suites); der Verbund kostet in Summe mehr (der dtype-Korpus
prüft `src` ein zweites Mal), betrifft nur die Laufzeit des Checks, nicht das Produkt.
