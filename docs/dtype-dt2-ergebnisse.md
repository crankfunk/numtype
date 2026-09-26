# dt2 — Promotion und elementweise Arithmetik — Ergebnisse

**Spec:** docs/dtype-dt2-spec.md v1.1 (Stufe 3b) · **Datum:** 2026-09-26 · **Commits:** 42e73a8 …
854dc2c auf `main` (unveröffentlicht; Release nach dt5) · **Covenant:** v9 (inkl. `PromoteDiv`-Ergänzung).

## Ergebnis

`add`, `sub`, `mul`, `div` rechnen für float64/float32/int32 in jeder Kombination, mit `Promote`
(und `PromoteDiv` für die Division, Gleitkomma immer) — zur Compile-Zeit berechnet, zur Laufzeit
identisch, beide Tabellen aus EINER Quelle (`PROMOTE_NUMERIC`, `PROMOTE_DIV`). float32 per
`Math.fround` nach jeder Operation, int32 mit Zweierkomplement-Wrap (`mul` ausschließlich
`Math.imul`). Skalar-Regel D6 mit Bereichsprüfung. bool dauerhaft ohne Arithmetik („use astype()").
Hover: `f32.add(i32)` → `NDArray<[2], "float64">`, `i32.div(i32)` → `NDArray<[2], "float64">`,
`f32.div(f32)` → `NDArray<[2], "float32">`.

## Zahlen

| Gate | vorher | nachher |
|---|---|---|
| check:diag Root | 237,098 @ 140 | **243,818 @ 140** (Δ+6,720; Gate +6,000 um 720 überschritten — Owner hat die Überschreitung akzeptiert, s. u.) |
| Messpunkt Commit A (P1+P2) | — | 238,493 (+1,395; Stopp bei +3,500 nicht ausgelöst) |
| Messpunkt Commit B (P3+P4) | — | 238,242 (−251 gegenüber A) |
| Stand vor Fix-Runde | — | 242,079 (+4,981) |
| check:diag:stress | 118,562 | 119,393 |
| check:diag:browser | 2,142 | 2,142 |
| bench:editor | … | `{41,165; 43,171; 73,928; 41,206; 46,751; 47,567; 40,131; 48,125}` |
| test:core | 1,618 | 1,632 |
| resident / package / Freeze-Hash | 6,155+2 / 3 / `2a54d9fd…` | unverändert |

## Post-Verification-Addendum

- **Baustein 0:** kein Blocker; A2-Liste empirisch vollständig (genau vier Tests); Präzisierungen:
  int32-`mul` nur `Math.imul` (große Produkte), Formulierung zu den float64-Pfaden.
- **Baustein A:** MERGE; alle Zahlen und beide Messpunkte als eigene Commits reproduziert; ein
  gestrichener Prüffall im DTypeLockPair-Test (`add`-Skalar auf gesperrtem int32 — dt2 hebt genau
  diese Sperre auf) als zwingend bewertet; Owner zur Kenntnis gegeben.
- **Baustein B:** MERGE-AFTER-FIXES; Bit-Identität unabhängig über 45,000+ Fälle bestätigt;
  **Regression** `AnyNDArray` im Array-Overload von add/sub/mul (gültiger Code abgelehnt — M2);
  `PromoteDiv`-Union-Gate am Argument ungeschützt (Mutant überlebte); int32-Bereichsgrenze ±1 ungetestet.
- **Baustein C:** `PromoteDiv` im v9-Text nicht genannt und nicht aus einer Quelle; int32-Skalar-
  Meldung nicht wortgleich zwischen Typ und Laufzeit; Ausnahme an `div` nur indirekt dokumentiert.
- **Fix-Runde G1–G6** und **Gegencheck (verify): alle sechs geschlossen.** G1: `IsAnyDType` vor dem
  Union-Gate — das übliche `0 extends 1 & T` versagt auf TS 7.0.2 bei EINGESCHRÄNKTEM Typparameter
  (`A extends DType`) und braucht den `infer`-Umweg; vom Gegencheck unabhängig bestätigt. Kosten +698,
  daher die Überschreitung um 720 — Owner-Entscheidung: akzeptiert. v9 um `PromoteDiv` ergänzt
  (Owner-freigegeben).

## Vorfall: falsche Git-Identität (2026-09-26)

Ab dt1 liefen 31 Commits unter `scratch <scratch@local>` statt unter der Identität des Owners —
ein Agent hatte in einem Worktree `git config user.*` gesetzt, was ohne `--worktree` in die
GEMEINSAME `.git/config` schreibt. 16 davon waren bereits auf GitHub. Bemerkt in einem Agentenbericht,
nicht durch eine eigene Prüfung. Behoben: lokale Überschreibung entfernt, alle betroffenen Commits
per `--reset-author` umgeschrieben (Owner-Entscheidung, Datum wird dabei neu gesetzt), Force-Push
nach Owner-OK. Vorbeugung: Arbeitsregel 22 (Agenten setzen nie `git config`; Autorenprüfung vor jedem
Push), Pflichtregel 7 im Verify-Template.
