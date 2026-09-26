# dt1 — dtype-Kern auf `NDArray` — Ergebnisse

**Spec:** docs/dtype-dt1-spec.md v2.1 (Stufe 3b) · **Datum:** 2026-09-26 · **Commits:** c4cf84d …
6de528f auf `main` (unveröffentlicht; Release nach dt5) · **Covenant:** v8 (7746a4d).

## Ergebnis

`NDArray<S, D>` mit float64/float32/int32/bool ist auf `main`: Speicher (`DataOf<D>`), Erzeugung
(`fromArray`/`zeros`/`ones` mit dtype), `astype`, Auslesen (`toArray`/`item`/`toNestedArray`/
`toJSON` inkl. bool → `boolean`) und die dtype-neutralen Ops (`transpose`, `slice`, `reshape`,
`flatten`) für jeden dtype; alle rechnenden Ops für D ≠ float64 gesperrt — mit eigener Meldung am
Argument, wo es eine Argumentposition gibt, sonst zur Laufzeit (in FOLLOWUPS getrackt).
Bestehender float64-Code: alle 1,591 Core- und 6,155 Resident-Tests unverändert grün.

## Zahlen

| Gate | vorher | nachher |
|---|---|---|
| check:diag Root | 227,405 @ 140 | **237,098 @ 140** (Δ+9,693, Gate ≤ +12,000 inkl. Tests) |
| Messpunkt nach K1–K5 | — | 234,146 (+6,741) |
| check:diag:stress | 116,279 @ 82 | 118,562 @ 82 |
| check:diag:browser | 2,142 | 2,142 |
| bench:editor | … | `{w1 40,219, w2 42,160, w3 73,097, w4 40,374, w5 45,785, w6 46,736, w7 39,300, w8 47,184}`, zweimal identisch |
| test:core | 1,591 | 1,618 |
| test:resident / threaded / package / browser / cargo | 6,155+2 / 139 / 3 / 4 / 222+1 | unverändert |
| Freeze-Hash | `2a54d9fd…` | unverändert |

Hover (echter LSP, von Umsetzung und Baustein A gemessen): `NDArray<[2, 3], "int32">`,
`transpose()` auf int32 → `NDArray<[3, 2], "int32">`, bool-`toNestedArray()` → `boolean[][]`.
Sperrmeldungen wortgleich zu `lockedOpMessage` (jetzt per Test gepinnt, s. u.).

## Post-Verification-Addendum

- **Baustein 0:** Blocker `data`-Feld fehlte in A1 (Owner nachentschieden); K3-Allokation ohne
  Prototyp-Vorlage (→ Vorprüfung als erster Schritt, bestanden); zwei dt2-Sätze aus v8 genommen,
  befristete `stack`-Ausnahme aufgenommen (Owner).
- **Baustein A** (deep, nach Eskalation des ersten Laufs): alle Zahlen exakt reproduziert, A1/A2
  eingehalten, 2 Mutanten gefangen; MAJOR: keine Typ-Pins für die neue Fläche; MINOR: Budget-Stopp
  konnte nicht greifen (K1–K5 in einem Commit).
- **Baustein B:** MAJOR: ungültiger dtype-String ergab still `data === undefined`; MAJOR:
  `DTypeLockPair`-Reihenfolge ohne Test (Mutant überlebte); MAJOR: Testkommentar behauptete einen
  nicht existierenden `Uint8Array`-Pin; MINOR: `stack`-Doku beschrieb die falsche Meldung.
- **Baustein C:** alle Invarianten halten unter v8; nur die Spec-Aufzählung nannte 0-arg `sum()`
  nicht (v2.1 nachgezogen).
- **Fix-Runde** (F1–F5): erschöpfende dtype-Switches mit `never`-Default, die werfen; Test für die
  Sperr-Reihenfolge inkl. Mutanten-Nachweis und wortgleicher Meldung über beide Ebenen; fehlende und
  neue Typ-Pins; `stack`-Doku korrigiert. **Gegencheck (verify):** alle fünf geschlossen, Zahlen
  exakt; neuer Befund: der Mutanten-Nachweis schreibt eine Kopie von `ndarray.ts` in `spike/src` —
  bei hartem Abbruch hätte sie `check:diag` brechen und ins npm-Paket gelangen können. **Selbst
  abgesichert:** Ausschlussmuster `__*-mutant-*-tmp*.ts` in allen drei tsconfig-Globs über
  `spike/src`, eindeutiger Dateiname; belegt mit einer absichtlich fehlerhaften liegengebliebenen
  Datei (check/check:diag grün, 140 Dateien, nicht im Build).
