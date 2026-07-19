# Handoff — 2026-07-19, Session-Ende (Item 13 UNTERWEGS: Release-Mechanik)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM).
Remote `github.com/crankfunk/numtype`, **noch PRIVAT** (nicht public geschaltet). **Item 12 (CI)
ist komplett + end-to-end CI-verifiziert.** **Item 13 (Release-Mechanik) ist UNTERWEGS** — die
Metadaten-/Doku-Vorbereitung ist erledigt, offen bleiben nur der finale `private`-Schalter, der
npm-Publish und das Public-Schalten (alles bewusst zuletzt). Alle Gates unverändert grün seit
Item 12 (kein Code/Rust berührt — nur Metadaten + Doku). Artefakt-Hash weiter `0b9df4f1…`.

## In dieser Session erledigt
- **Privacy-Audit vor OSS:** Repo (Code + GESAMTE Historie) auf Secrets/Credentials/PII geprüft —
  sauber. Git-Identität komplett pseudonym (`crankfunk` + GitHub-noreply, alle Commits). Einziger
  Fund: der Vorname des Owners in einer Doc-Zeile → entfernt (Commit `359c728`). Historie bewusst
  gelassen (ein Vorname, und Umschreiben würde die vielen SHA-Referenzen der Docs brechen).
  GitHub-Profil vom Owner geprüft (pseudonym; Website/X bewusst verlinkt — kontrollierte, gewollte
  Öffentlichkeit, unabhängig vom Repo).
- **Item 13, Metadaten-/Doku-Vorbereitung (committet + gepusht):**
  - **Lizenz Apache-2.0** (`3d7e7ba`): `LICENSE` (wortgetreuer offizieller Text) + `NOTICE`
    (Copyright 2026 crankfunk) + `package.json`. Owner-Entscheidung: Patent-Grant (schützt die
    from-scratch-Kernel-Algorithmen) + Attribution, passend zur TS/Rust/WASM-Familie (statt MIT).
  - **README-Vollüberarbeitung** (`bfdb01b`): **ANSI-Shadow-figlet-Banner** als wiederkehrender
    Signature-Move (`pnpm dlx figlet -f "ANSI Shadow" "<Name>"` → Plain-Codeblock oben); Usage mit
    **typecheck-verifizierten** Beispielen (Release-Gate „Beispiele laufen wörtlich" belegt: temp.
    Prüfdatei gegen die echte API, jede Shape exakt, jedes ❌ ein echter Compile-Fehler); die drei
    §5-Qualifikationen wörtlich; kompaktes „What's implemented" (statt Wall-of-Text).
  - **Zero-Dep-Abschnitt** (`dd6012b`): zero-dep/from-scratch jetzt als eigenständiges Argument
    (keine `dependencies`, per CI-Guard erzwungen; kein wasm-bindgen/BLAS/ndarray). Plus
    Backends-Klarstellung: `backend("threaded")` ist NICHT im Paket (Checkout-only).
  - **`engines`** node `>=20` (`2e33a65`) · **`author`** `crankfunk` (dieser Commit).
- **Wichtiger Verify-Befund** (in FOLLOWUPS): das **Threads-`.wasm` ist NICHT im npm-Tarball**
  (bewiesen via `npm pack --dry-run`: nur `dist/wasm/numtype_core.wasm`, 71.6 kB, 40 Files, 136 kB;
  `threaded.js` drin, sein `.wasm` fehlt — braucht die nightly + build-std). In der README
  klargestellt (Checkout-only). Item-13-Entscheidung offen; Empfehlung: für v0 so lassen.

## Offen / in Arbeit
Item 13 fast fertig — bewusst zuletzt (kein versehentlicher Publish):
- **`"private": false`** — der FINALE Publish-Schalter; erst unmittelbar vor `npm publish` umlegen.
- **npm-Account + `npm publish`** — Owner-Aktion. In 2026: 2FA Pflicht (Authenticator-App), first
  publish per 2FA-Prompt; unscoped `numtype` = automatisch public; VOR jedem Publish
  `npm pack --dry-run` gemeinsam reviewen. Der Name `numtype` wird mit dem ersten Publish besetzt
  (kein separates Reservieren). Für spätere Releases: **Trusted Publishing** via GitHub Actions
  (OIDC, token-frei, Provenance) — geht aber nur aus einem PUBLIC Repo.
- **Repo public schalten** (GitHub Settings → Visibility) — Owner-Aktion. Vorbereitung steht
  (LICENSE + README da; interne `docs/` teils Deutsch = bewusst „internal research notes",
  Owner-ok). Danach ist auch Trusted Publishing möglich.

## Nächste Schritte
1. **Item 13 abschließen:** npm-Account/2FA (Owner) → `npm pack --dry-run`-Review gemeinsam →
   `"private": false` setzen → `npm publish` (Owner, 2FA). Optional gleich Repo public schalten,
   dann später auf Trusted Publishing umstellen.
2. **Item 14 — v0.1 research preview.**
3. threaded-`.wasm`-Bundling-Entscheidung (FOLLOWUPS) — Empfehlung: für v0 Checkout-only lassen.

## Bekannte Probleme / Stolperfallen
- **`"private": true` blockt `npm publish` komplett** — das ist der bewusste letzte Schalter.
- **Threads-`.wasm` nicht im Tarball** (s. o.) — `backend("threaded")` ist im publizierten Paket
  nicht funktionsfähig (Checkout-only), in der README so dokumentiert.
- **`pnpm test:threaded` baut BEIDE Artefakte** (stable + threads) — braucht stable 1.95.0 UND die
  pinned nightly-2026-07-09 (+rust-src).
- **Freeze-Hash cross-host byte-stabil** (Linux == macOS, im CI bestätigt) — Pin-Menge braucht nur
  einen Eintrag.
- **CI läuft bei JEDEM Push** (auch Doku) — paths-ignore-FOLLOWUP offen.
- **`.nvmrc` „24"** löst lokal eine harmlose nvm-Warnung aus; in CI korrekt.
- **Pins nie über File-Set-Änderungen vergleichen**; Messungen nur im frischen `git worktree`.
- **Covenant-Regime** (CLAUDE.md „Qualitätssicherung"): substanzielle Scheiben = A+B+C-Verify;
  Metadaten-/Doku-Minis wie diese Session sind Stufe 0–1 (Lint reicht, kein Voll-Verify).

## Wichtige Dateien & Befehle
- **Release-Metadaten:** `package.json` (license `Apache-2.0` / author `crankfunk` / engines `>=20`
  gesetzt; `private` noch `true`), `LICENSE`, `NOTICE`, `README.md` (Banner-Move + verifizierte
  Beispiele).
- **Paket prüfen:** `pnpm build:dist` · `npm pack --dry-run` (Tarball-Inhalt) · `pnpm test:package`.
- **Prozess/Doku:** `COVENANT.md` (v4) · `FOLLOWUPS.md` · `docs/roadmap.md` · `CLAUDE.md` ·
  `docs/verify-runde-template.md`.
- **Kern-Kommandos:** `pnpm check` · `pnpm test:core|resident|package|browser|threaded` · `cargo test
  --manifest-path crates/core/Cargo.toml` · `pnpm demo` · `pnpm bench:editor` · `git`/`gh` fürs CI.
