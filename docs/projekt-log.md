# NumType — Projekt-Log (historisches Narrativ)

Dieses Dokument ist das vollständige Scheiben-für-Scheiben-Narrativ des Projekts, am
2026-07-19 **wörtlich** aus der CLAUDE.md hierher verschoben (einzige Änderung: Circa-Tilden
`~` → `≈`, weil GitHubs Markdown einzelne Tilden als Strikethrough paart). CLAUDE.md trägt
seitdem nur noch Regeln + aktuellen Stand; dieses Log wächst am Ende pro Scheibe weiter
(Hausregel in CLAUDE.md, „Obligatory workflow" Punkt 5). Detailtiefe pro Scheibe: die
jeweiligen `docs/*-spec.md` + `docs/*-ergebnisse.md` bleiben die Primärquellen.

## Verlauf bis v0.1.0 (Stand 2026-07-19, ursprünglich „Current phase")

Working the release roadmap (docs/roadmap.md, committed 2026-07-10). Done since: the Kern-06 follow-up auto-routing slice (threshold 262_144 = 64³, measured via `bench:crossover`), **Spike 02** (editor latency: headless LSP harness against the native TS7 server — the hard roadmap-A1 gate PASSES with ≈3 orders of magnitude headroom; docs/spike-02-*), and **Spike 03** (compile-time bounds checks for literal integer `slice()` indices — `LiteralIndexBounds` in slice-literal.ts, error at the offending argument mirroring the runtime throw verbatim; never-wrong-only-incomplete: wide/dynamic/non-plain-digit/mixed-union forms pass to the runtime backstop; negative literals work via `-${Abs}` strip + comparison only; 174/174 generated parity grid vs the runtime rule; machinery budget 1.036× — docs/spike-03-*), and **Spike 04** (type-level shape products: `LiteralShapeProduct<S>` via schoolbook digit-string multiplication in slice-literal.ts — 177/177 BigInt parity grid; never-wrong via boundary filters (union/never dims degrade — a product verdict is an unbounded value, Spike 03's subset-check pattern does NOT transfer) + MAX_SAFE_INTEGER cap (bigger digit strings would double-round through float64 into a WRONG literal); budget measured: +2,186 instantiations for unused machinery = pure declaration cost, ≈1,249/site typical shapes; decision GO as a DISCLOSED DEVIATION from the pre-registered gate rule (G1 1.0321× vs ≤1.02× and G2 mean missed as written; absolute + editor gates PASS; independent verify: "defensible, NO-GO not required") — docs/spike-04-*). Also done: **Spike 05 / roadmap A3** (variance design: `NDArray<S>` stays invariant per the two-of-three rule; new minimal covariant read view `NDArrayView<out S>` in ndarray.ts — exactly shape/strides()/toNestedArray(), no data (backend portability), no computed-shape members (TS2636: the `out` annotation check is ABSTRACT, factual monotonicity of `Transpose` doesn't count); class `implements NDArrayView<S>`; `AnyNDArray` stays as the documented both-ways-unsafe escape hatch; demo printArray migrated; check:diag pin now 133,727 — docs/spike-05-*), and **Spike 06 / roadmap A4** (range slices: negative literal start/stop + literal steps ≥ 2 compute statically — no signed addition needed, compare+subtract+clamp; new schoolbook long division `DivCeil`; provably-invalid literal steps (0, negative, dot-form ±1.5) are compile errors at the argument mirroring the runtime throw verbatim; 280/280 grid against the IMPORTED real `normalizeSliceSpecs` as ground truth; union boundary filters retrofitted to start/stop/step; check:diag pin now 188,378 = 3.77% of budget — docs/spike-06-*). **Phase A is complete, including the Kür.** **Kern 07 / Phase B item 1** is done and independently verified (2026-07-11): elementwise `sub`/`mul`/`div` + `dot`/`norm`/`cosineSimilarity` on NDArray + WNDArray — one generic strided elementwise kernel (kernels/elementwise.rs, mirrors add_strided line-for-line) + two reduction kernels (kernels/vector.rs: dot_strided single ascending accumulator, norm_sq_strided logical row-major) behind five appended ABI entry points; `norm`/`cosineSimilarity` have NO own kernels — pinned TS-side scalar compositions, since sqrt/*// are IEEE-exact JS ⇄ WASM; scalar consumer ops return plain `number` (documented asymmetry with `sum()`); `DotCheck` guard (spike/src/vector.ts) with verbatim runtime⇄compile message stems and union-dim no-claim; +772 differential tests, +48 cargo tests, three mutation proofs incl. the verifier's own; new plain-artifact pin `7a65d800…` (a phase adding exports legitimately changes the hash — freeze claim is source-level, see docs/kern-07-ergebnisse.md); disclosed+confirmed deviation: TS class bodies take insertion-only diffs, not literal EOF appends (private constructors) — docs/kern-07-*. **Kern 08 / Phase B item 5 remainder** is done and independently verified (2026-07-11): runtime `reshape`/`flatten` on both surfaces consuming `LiteralShapeProduct` — flatten hovers as a computed literal (`NDArray<[1048576]>`), reshape rejects provable product mismatches at the argument (message verbatim to the runtime throw: `reshape: cannot reshape array of size ${size} into shape [${ns}]`; dim-validity checked first: `reshape: invalid dimension ${d} …`); `ReshapeCheck` in spike/src/reshape.ts (wide → IsUnion → literal equality), stretch `LiteralReshapeDimInvalid` (negative/dot-form dims lifted; exponent/0/union no-claim; returns bare verdicts — messages built in reshape.ts, keeping slice-literal.ts append-only); WNDArray routes view-if-contiguous (isContiguous requires offset 0 — offset-shifted contiguous-shaped views conservatively materialize, never unsound), else nt_materialize; ZERO Rust changes, artifact hash byte-identical (strong freeze form); both Spike-04 obligations closed — guard wording fixed + editor-hover cost of real sites measured (bench:editor W6: 0.06 ms hover medians incl. big-dim flatten, in-family, ≈half of the W3 digit-stress; per-site guard computation ≈6k instantiations for huge-dim sites, declaration alone cheap); -1 inference deferred (FOLLOWUPS) — docs/kern-08-*. **Kern 09 / Phase B item 5 final rest** is done and TWICE-verified (spec + adversarial, 2026-07-12): runtime `keepdims` on `sum()` for both surfaces — one appended shape-metadata helper `keepDimsShape` (runtime.ts, append-only), `const KeepDims extends boolean = false` second type param feeding the pre-existing `ReduceAxis<S, Axis, KeepDims>`; guard stays keepdims-free; data byte-identical to non-keepdims (ZERO Rust changes, artifact hash byte-identical, adversarial verifier confirmed through cargo clean + rebuild); 367 new differential tests (non-circular: structural invariants + trusted non-keepdims reference), incl. views (transposed/offset/composed) and WNDArray false-parity; 5/5 broad mutants + 1 spec-verifier mutant all caught; new pins main 174,213 @ 126 files / stress 94,597; owner-confirmed D3 deviation (extending the existing sum methods — a new param cannot be insertion-only); adversarial finding: mixed-rank shape-union in ONE instance's type param (`NDArray<[2,3]|[2,3,4]>`) confidently mis-types `.sum()` — PRE-EXISTING facet (c) of the FOLLOWUPS union-guard item, not keepdims-related — docs/kern-09-*. This slice also triggered the process upgrade: "Qualitätssicherung, modellunabhängig" section below + docs/verify-runde-template.md (two-verifier rule). **Kern 10 / Phase B item 6** is done and twice-verified (spec CONFIRMED + adversarial HÄLT, 2026-07-12): IEEE-754 special values (NaN/±Inf/±0/subnormals/±MAX) injected into the differential generator — `SPECIAL_VALUES`/`nextF64Special`/`genDataSpecial` appended to prng.ts (existing generators byte-identical), new spike/tests-runtime/special-values.test.ts (619 cases) proves reference⇄v1⇄resident bit-identity (value-class for NaN, byte-exact otherwise) across add/sub/mul/div/sum/matmul/dot/norm/cosine/transpose; TEST-ONLY, zero Rust changes, artifact hash byte-identical; **key finding: SIMD-blocked matmul preserves subnormals (no flush) — proven catchable, both verifiers built the SIMD-flush mutant → fixture+random cases go red**; 3 real mutants caught total; new pins main 174,391 @ 127 files (Δ+178 from Kern 09's 174,213) / stress 94,597; disclosed spec fix (my `sum([-0,-0,-0])=-0` was wrong: sumRuntime seeds +0, so it's +0 — executor proved it via node -e and corrected the fixture); adversarial finding: the new dot/cosine random passes are weak against accumulation-ORDER bugs (Inf/NaN dominance at 35% injection masks rounding — order-sensitivity stays covered by vector.test.ts; the special passes cover PROPAGATION not order — documented in-code, not booked as order coverage); NaN-payload byte-exact preservation regression-tested only for transpose (reshape/slice/fromArray hold empirically → FOLLOWUPS) — docs/kern-10-*. **Kern 11 / Phase B item 7 (perf Kür, focused scope)** is done and twice-verified (spec CONFIRMED + adversarial HÄLT, 2026-07-12): a contiguous fast path in `add_strided` (add.rs) + `binary_strided` (elementwise.rs) skips the per-element `unravel` heap allocation (shape.rs:105) when both operands share the same shape, offset 0, and natural strides (`flat == a_off == b_off` ⇒ `out[i]=op(a[i],b[i])`) — **13–17× on the contiguous hot path**, bit-identical to the general path (mathematically airtight: natural strides ⇒ `idx·strides == flat` for every flat; empirically two committed `.to_bits()` equivalence tests per file). MEASUREMENT-DRIVEN: SIMD elementwise measured NO-GO (memory-bound), packing-reuse facet A measured 3.3% NO-GO — the `unravel` allocation was the real lever (docs/kern-11-*, "Messgrundlage"). Freeze: pre-edit clean rebuild reproduced the old pin `7a65d800…`, new pin **`0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`**; used the ALLOCATING `compute_strides` (never the `#[cfg(atomics)]`-gated `_into` twins — their mere presence shifts artifact bytes, shape.rs:208-220). Freeze honesty (adversarial WAT-diff): "v1 add untouched" is a SOURCE/LOGIC claim, not artifact-byte — a compiler-generated `memcmp` helper for the new slice `==` shifts `i32.const` panic-location pointers crate-wide by ≈96 B in EVERY function (zero opcodes), which is exactly why the freeze proof is a whole-artifact hash. Two adversarial coverage gaps closed IN-slice (test-only, hash unchanged): guard-canonicality (`a_strides==b_strides` mutant survived cargo, only TS caught it) + size-0/size-1-interleaved fast-path. cargo 161, test:resident 4265+2, artifact-hash re-pinned. **check:diag DROPPED to 172,392 @ 128 files** (from Kern 10's 174,391 @ 127) from adding ONE strictly-typed value-only bench file (`spike/bench-core/elementwise.ts`) — reproduced in isolated worktrees by both verifiers, budget/coverage-neutral; **mechanism pinned 2026-07-12 (owner-requested bisection): CHECK-ORDER-dependent counting** — an empty `export {}` file reproduces ≈2,043 of the drop, magnitude varies with the file's sort-position (name), non-monotonic; global instantiation memoization means "who checks a shared recursive type first" decides fresh-vs-cached, and a file addition reshuffles that. Retroactively explains Infra-01's super-additive removal. Consequence: file-adding slices carry ±≈2,000 order-noise in the pin (not real type-cost) — decompose via empty-then-fill for clean attribution. Deferred (FOLLOWUPS): `unravel_into` general-case generalization (bigger lever, payoff inferred not measured — own measured slice), packing-reuse facet B (unmeasured) — docs/kern-11-*. Next: Phase B is COMPLETE (items 1–7 done). **Phase C is COMPLETE (2026-07-12):** Items 8/9 (browser threads port / stable-no_std) deliberately DEFERRED after a scoping pass — threads stay Node-only/experimental for v0 (no way off the pinned nightly today: build-std nightly-only + the 2026 RFCs 3874/3875 don't cover atomics rebuilds, wasm32-wasip1-threads is a dead end, no_std likely doesn't escape either; browser port gated on COOP/COEP headers a library can't set; docs/phase-c-threads-scoping.md). **Item 10 (backend-choice API) DONE + twice post-verified (Commit 5b0f951):** `NDArray.backend('wasm'|'threaded')` exposes the WASM/threads backends as an explicit, browser-safe opt-in (proven empirically via `process.moduleLoadList` trace — the JS `NDArray` default never eagerly pulls threaded.ts's static node imports), JS `NDArray` stays the default, ZERO Rust / hash byte-identical, new pins check:diag 175,634 @ 132 / stress 103,882 @ 82; docs/item-10-backend-api-*. This slice pioneered **"spec-verify BEFORE implementation"** (QA section below + verify-runde-template.md Baustein 0): the pre-build adversarial spec review caught 3 blockers incl. a false code-assumption (WNDArray.strides is a FIELD not a method → NDArrayView conformance deferred to the Spike-05 followup). **Phase D pre-work is UNDERWAY (binding spec docs/phase-d-vorarbeiten-spec.md, Commit 8b39c15, Baustein-0-verified with one blocker merged pre-build — facet (b) of the union item repros ONLY as a shape-union in ONE instance's type param, the `NDArray<A>|NDArray<B>` argument form is rejected by TS inference itself; owner-decided forks: strides→readonly property, mixed-rank unions degrade uniformly; execution order V3→V1→V2). V3 (browser smoke test) DONE + twice-verified + in-slice closures (2026-07-12):** first real-browser proof of the standard surface — Playwright/Chromium as devDep (runtime stays zero-dep), tsc emission via `rewriteRelativeImportExtensions` + `node:http` static server (`application/wasm` MIME is load-bearing: the loader's browser branch has NO instantiateStreaming fallback), COOP-free proven in-page (`crossOriginIsolated===false`), byte-exact differential matrix JS-`NDArray` ⇄ `backend("wasm")` incl. views/both reshape branches/special values, `backend("threaded")` rejects in-browser with the pinned stem; the verify round caught + closed IN-slice (test-only, hash untouched): stale-`.emit` false-pass on DIRECT playwright invocation (mtime-witness freshness guard — never run `playwright test` directly, always `pnpm test:browser`), streaming-path-TAKEN now asserted via a counting `addInitScript` wrapper (existence+MIME checks alone provably miss an `if(false)` loader mutant), `playwright.config.ts` typechecked in the browser leg; the Δ+78 main-pin move is REAL type-cost of the guard-test extension, bisected by verifier A incl. a same-length comment-control probe (Δ+0) — NOT order noise, i.e. a counterexample to blanket order-noise attribution for file EDITS (the ±2,000 order-noise rule below is about file ADDITIONS); docs/phase-d-vorarbeiten-v3-ergebnisse.md. **V1 (union-guard fix) DONE + twice-verified + in-slice closures (2026-07-13):** the three union facets are closed — (a) `IsUnion` filters first in `CompatDim`/`DimEq` (union dim → no-claim/wide; the pre-fix bug on matmul's contraction axis was a confidently-wrong REJECTION, a never-wrong violation); (c) new `RankUnknowable<S>` (dynamic rank OR union `S["length"]`) at all seven rank gates (Broadcast/MatMul/ReduceAxis/Transpose/SliceShape/SliceSpecsGuard/DotCheck), uniform degradation to `readonly Dim[]` per owner decision (deliberate precision loss on Transpose/SliceShape/dot where distribution was already correct); bonus real fix: `SliceSpecsGuard` arity leak on mixed-rank receivers; (b) tuple-wrapped `Guard` (`[Result] extends [ShapeError<infer M>]`) — uniform-error union rejects with ONE combined message, mixed accepts gradually; the Kern-07 claim that `NDArray<A>|NDArray<B>` arguments bypass the Guard does NOT reproduce on TS 7.0.2 (TS inference itself rejects that form — honest discrepancy note in FOLLOWUPS; the real leak form is the shape-union in ONE instance's param). Method: repro-first (20-assertion pre-fix RED proof in fresh worktree, byte-identically reproduced by verifier A), pins only in EXISTING test-d files, ZERO edits to slice-literal.ts/runtime.ts/resident.ts, zero re-expressed old pins, ZERO Rust/hash byte-identical; both surfaces pinned (WNDArray incl. the UW4 combined-message pin from the closure round — Guard-revert mutant now caught by 2 pins); over-degradation direction protected by 163 old-corpus pins (adversarial mutant 3); budget isolated: +1,060 src-only (matches the Baustein-0 estimate) + ≈+4,000 for the new union-heavy pins themselves; `never`-dim verdicts, union-AXIS-param (`sum(0 as 0|2)` — confidently wrong, RELEASE-RELEVANT own mini-slice before Item 11) and specs-tuple-union message cosmetics booked as FOLLOWUPS (pre-existing, out of V1 scope); docs/phase-d-vorarbeiten-v1-ergebnisse.md. **V2 (strides/NDArrayView/readonly-shape) DONE + THREE-way verified (spec CONFIRMED + adversarial HÄLT + covenant-verify: no invariant violation — first slice under the full covenant regime) + in-slice closures (2026-07-13):** strides harmonized to a readonly PROPERTY (owner fork; the NDArray method had ZERO call sites repo-wide, WNDArray field conforms as-is, the 8 field assertions stayed byte-identical), `WNDArray<S> implements NDArrayView<S>` (ThreadedBackend products covered automatically; covariance pins for both backends; NOTE: the implements keyword carries no downstream pins — structural typing — its value is the declaration-site self-check, and it CANNOT catch return-type narrowing of `toNestedArray(): unknown` → closed via `Equal<ReturnType<…>, unknown>` pins), deep-readonly `shape` via `Readonly<S>` on view + both classes (TS2636 pre-flight probe: homomorphic mapped types PASS the abstract out-check, unlike Transpose; `shape[0] = 2` rejection pins — the `= 99` form is CONFOUNDED by literal-type narrowing, only same-value assignment isolates readonly; 56 pins re-expressed `[…]` → `readonly […]`; class hovers stay `NDArray<[2, 3]>`, LSP-verified — covenant M3). **Unplanned finding + owner decision:** Readonly<S> incidentally LIFTED NDArray's measured invariance (the old block was an ACCIDENT of keepdims' `AllOnes<S>` return type, verbatim-confirmed in the baseline tsc error) → owner decided RE-invariantization via explicit `private declare readonly __variance: (s: S) => S` marker on NDArray AND WNDArray (property-style is mandatory — method shorthand is bivariant and inert; an `out` annotation is impossible: Transpose return types fire TS2636); NDArrayView stays the ONE enforced-covariant surface; the marker DROPPED the main pin by −10,308 (the old successful widening check fully resolved the AllOnes chain; the marker check fails early). Adversarial coverage closures: NDArray.strides getter VALUE pinned with hard literals (a reversed-strides mutant previously survived ALL suites), toNestedArray-narrowing pins, the three shape[0] pins delivered (A-Auflage — results doc had claimed them falsely, honesty-corrected). ZERO Rust / hash byte-identical (clean rebuild); docs/phase-d-vorarbeiten-v2-ergebnisse.md. **Phase-D-Vorarbeiten (V1–V3) are COMPLETE.** **Union-AXIS mini-slice DONE + three-way verified (spec CONFIRMED w/ condition + adversarial HÄLT + covenant-verify, 2026-07-13):** `IsUnion<Axis>` filter branch in `ReduceAxis` placed BEFORE the naked `Axis extends number` (position is load-bearing — distribution starts there; position mutant flips 16 pins), `IsUnion` exported from dim.ts (Δ0); every axis union degrades to `readonly Dim[]` like the dynamic axis (incl. all-invalid — no-claim like union dims, deliberate divergence from the Guard shape-level uniform-error rejection), runtime backstop proven in real Node with the verbatim message stem; KeepDims-`boolean` deliberately unfiltered (distribution there is correct-on-all-paths, pinned already-safe). Baustein 0 caught a REAL blocker pre-build: the `0|undefined` CALL form is structurally unreachable — TS strips `undefined` from inferred unions at OPTIONAL parameters (2×2 cross-probe; same root cause hits `keepdims?`) → owner scope reduction; the `Literal|undefined`-via-optional-params family is a documented KNOWN M2 VIOLATION (COVENANT.md v2, dated note under M2, norm unchanged) with the `UA_GAP` sentinel pin (proven a real observer via required-param mutant) + explicit-type-arg workaround pinned; fix candidate = overload split at the Item-11 API cut (FOLLOWUPS). New bench:editor workload W7 (union-axis hover, 0.06 ms, correctness-gated on `NDArray<readonly number[]>`). Pins 178,865 @ 132 / 102,182 @ 82 / browser 2,142 @ 75; ZERO Rust/hash byte-identical; docs/union-axis-mini-*. **Item 11 (API-Schnitt + Paketierung) is COMPLETE** (S1+S2+S3, 2026-07-17, each 3-way verified & committed — S1 48ee440 / S2 87e6e6b / S3 69ab47a; docs/item-11-s{1,2,3}-ergebnisse.md + docs/item-11-api-paket-spec.md): **S1** sum-overload umbau (COVENANT-M2 violation closed, both axis+keepdims facets via arg-count overloads + `reduce.ts` KeepDims→`boolean|undefined`; `NDArray<any>` impl-return owner-confirmed) + `slice-literal.ts`→`literal-arithmetic.ts` rename. **S2** emit/package pipeline: `tsconfig.build.json` + zero-dep `scripts/postbuild-dist.mjs` post-emit rewrite (fixes the three TS7 emit blockers — `.d.ts` `.ts`-endings, `new URL` worker path, `node:worker_threads` in threaded.d.ts via skipLibCheck) + `.wasm` bundling + package.json exports/main/module/types/files/sideEffects (`pnpm build:dist`). **S3** zero-dep-guard + package-smoke as CHECKED GATES (`pnpm test:package`: runtime smoke against `dist/index.js` incl. `backend("wasm")` real-.wasm-load, consumer type-smoke against `dist/index.d.ts`, independent comment-aware `check-dist-emit.mjs` precision gate — the ONLY Blocker-1 guard). COVENANT v2→v4 (M2 closed in v3; Z2 refined in v4: build-gated typechecks run in test:package, not `pnpm check`). npm-name/author-field/LICENSE deliberately deferred to Item 13. **Item 12 (Qualitäts-Portfolio + CI) is COMPLETE** (2026-07-18, three-way verified): an 8-job GitHub-Actions CI on ubuntu-latest (check / cargo / test-node / test-browser / test-threaded / freeze / editor-gate / demo), trigger `push:[main]` + `pull_request` (no double runs — Baustein-0 D-2); **rustc 1.95.0 pinned** via `rust-toolchain.toml` for a reproducible freeze-hash; the freeze-hash gate uses a platform-labelled pin SET (`scripts/check-freeze-hash.mjs`, D4 — a wasm artifact is not cross-host byte-stable, so the first ubuntu run empirically clarifies whether the Linux hash differs from the macOS pin, then it's added as a second entry); `bench:editor` hardened from report to a real gate (`enforceHardGate`: correctness + W1–W7 instantiation pins exact-match hard, latency at the 2x ceiling, `process.exitCode`); `--test-timeout=120000` in test:core/resident/package/threaded (F6 closed); a zero-dep CI-runnable S1 guard (`spike/tests-runtime/s1-import-guard.test.ts`) mirroring the `covenant-s1` rule with a **string-aware text scanner** — the initial line-by-line version was a deceptively-green gate (Verify-B found two multi-line import bypasses — `import(\n"x")` AND `from\n"x"`; the repo has no formatter forcing single-line), fixed in-slice by a comment-stripping state machine + text-wide scan, non-vacuity self-test pins both forms. **No Vitest** — `node --test` stays (owner decision, zero-dep). Three-way verified: A CONFORM (all gates fresh-green), B found+fixed the two S1-guard bypasses, C no covenant violation. NO Rust touched → artifact hash byte-identical `0b9df4f1…`; COVENANT unchanged (v4). npm-name/author/LICENSE stay deferred to Item 13. **Item 13 (release mechanics) is COMPLETE — numtype@0.1.0 is LIVE on npm and the repo is PUBLIC (2026-07-19).** The metadata/docs prep (earlier commits) — Apache-2.0 license (LICENSE + NOTICE + package.json; owner decision, the patent grant over MIT protects the from-scratch kernel algorithms), README rewrite (ANSI-Shadow figlet **signature banner** + typecheck-verified usage examples [the "examples run verbatim" release gate, proven via a temp check file against the real API] + the three §5 qualifications verbatim + a dedicated zero-dep section), `engines` node >=20, `author` crankfunk; a pre-OSS **privacy audit** ran clean (repo + full history — only the owner's first name removed from one doc). Release session (2026-07-19): pre-flight verified (name `numtype` free on the registry, tarball clean via `npm pack --dry-run`, CI green, no CI secrets, no rulesets, all commits on the GitHub-noreply identity); three gaps found + closed pre-publish (**NOTICE was missing from the tarball** — npm auto-includes LICENSE/README but NOT the Apache-2.0 NOTICE → added to `files`; version 0.0.0→**0.1.0**; **`prepublishOnly` = `pnpm test:package`** as publish airbag so a publish can never ship a stale/missing dist); README gained **"Why NumType exists"** (the original motivation: the TS ecosystem's missing NumPy counterpart, minimum-viable framing) + an **editor-support-out-of-the-box** paragraph (no extension/plugin/codegen; verified on TS 7.x, older majors untested → FOLLOWUPS TS-5.x consumer smoke); HANDOFF.md untracked (gitignored, stays local) + contributor note atop CLAUDE.md. First publish manual with 2FA, registry-verified (41 files / ≈435 kB unpacked); tag `v0.1.0` (commit 37335d0); repo flipped public; GitHub metadata set (description, homepage→npm page, 11 topics). **Corrected fact (verified against npm docs):** Trusted Publishing is configured per-package on npmjs.com (+ `id-token: write` in the workflow) and works from private repos too — only the provenance BADGE requires a public repo; setup is an optional FOLLOWUP. Item-13 leftovers deliberately deferred to Item 14: demo GIF, launch blog post, research-notes curation (FOLLOWUPS). Finding: the threads `.wasm` is NOT in the npm tarball (only the stable `numtype_core.wasm`; the threads artifact needs the pinned nightly + build-std) — `backend("threaded")` is checkout-only, documented in the README + FOLLOWUPS (recommend keeping it checkout-only for v0). Next: **Item 14** (v0.1 research preview — incl. the deferred Item-13 leftovers: demo GIF, launch blog post). See HANDOFF.md. The check:diag trend question is RESOLVED (owner decision 2026-07-11, implemented + verified as **Infra 01**, docs/infra-01-stress-split.md): digit-arithmetic stress cases (≥13-digit operands / MAX_SAFE_INTEGER cap probes) live in the separately-measured `spike/tests-stress/` tsconfig; `pnpm check` is a COMPOUND (root + stress — nothing rots; non-vacuity proven in both directions via corruption tests), `check:diag` stays root-only (main pin **173,716**, a NEW baseline — never compare across the split), new `check:diag:stress` (stress pin **94,523**, ungated by design). Realistic/semantic pins (incl. the 1024×1024 headline case and all degrade rows) stay in the main corpus. Done, verified, and committed: Spike 01 (type layer — docs/spike-01-*), Kern 01 (from-scratch kernels behind a hand-rolled `extern "C"` ABI, bit-identical to the naive TS reference — docs/kern-01-*), Kern 02 (zero-copy residency incl. fromArray Float64Array overload — docs/kern-02-*), Kern 03 (strided views: O(1) transpose, refcounted buffers, strided ABI entry points + status 4, `contiguous()` — docs/kern-03-*), Kern 04 (blocked+packed+SIMD128 matmul, bit-identical under the "bit-identity law": vectorize only ACROSS output elements, ascending-k single accumulator chain, no FMA/relaxed-simd — docs/kern-04-*; 2.1–3.25× over the Kern-03 scalar kernel, Kern-03's view-matmul penalty erased by packing), Kern 05 (slicing: O(1) slice views with the first nonzero offsets, zero Rust changes; type layer = gradual core rules + statically computed slice dims via from-scratch digit-string arithmetic in spike/src/slice-literal.ts — docs/kern-05-*), the three small hardenings (v1 OOM path, test-list guard, ABI prevalidation under scope (a)), Kern 06 (threads: hand-rolled substrate, no wasm-bindgen — separate shared-memory artifact on pinned nightly-2026-07-09 + -Zbuild-std, allocation-free `nt_matmul_blocked_partial` split across output rows only ("parallel bit-identity law": bit-identical by construction for any worker count/split), persistent worker_threads pool with per-worker shadow stacks + Atomics handshake, poisoned-pool error semantics with frees deferred past worker.terminate(); 2.86–4.42× at n=512/1024 with 8 workers, threads lose at small n — docs/kern-06-*). The naive TS runtime remains the correctness reference; the v1 copy-based backend remains the frozen performance baseline (its kernels/entry points stay byte-for-byte untouched). Build note: wasm builds need the repo-root `.cargo/config.toml` simd128 rustflag, and cargo config discovery is CWD-based — run all commands from the repo root (a compile_error! guard fires if the flag is lost). Every phase follows: binding spec doc → implementation → fresh-context verification → results doc with post-verification addendum → KB capture → commit.

## Historische Commands-Sektion (mit Pin-Historie, Stand 2026-07-19)

`pnpm check` (types; COMPOUND seit Phase-D-V3 DREIfach: root + `spike/tests-stress` + `spike/tests-browser`-tsconfig — Stress- und Browser-Strecke sind Teil jedes checks, nur die MESSUNG ist getrennt; Nicht-Vakuität aller Legs per Korruptions-Tests bewiesen) · `pnpm check:diag` (Haupt-Pin **187,918 @ 135** Files seit Item 12 (Δ+1 File `s1-import-guard.test.ts` im Root-Korpus → File-Set-Änderung, Order-Noise-behaftet, NICHT gegen 186,691 verrechenbar; neue `check:freeze`-Skript + ci.yml sind Infra, nicht im spike-Korpus); davor **186,691 @ 134** Files seit Item 11 / S3 (Δ+2 Files: zero-dep-guard.test.ts + package-smoke.test.ts im Root-Korpus → File-Set-Änderung, Order-Noise-behaftet, NICHT gegen 179,986 verrechenbar; A-Verifier maß den zero-dep-guard-Teileffekt isoliert als +53 @ 133; DREIfach verifiziert, docs/item-11-s3-ergebnisse.md); davor **179,986 @ 132** Files seit Item 11 / S1 (Δ+1,121 ggü. 178,865 — aber T1b RENAMT eine Root-Korpus-Datei (slice-literal.ts→literal-arithmetic.ts) → Sort-Position-Order-Noise bis ±≈2,000, NICHT als reine Typkost gegen 178,865 verrechenbar; DREIfach verifiziert, docs/item-11-s1-ergebnisse.md); davor 178,865 @ 132 seit der Union-Axis-Mini-Scheibe (Δ+653 echte Typkosten: Filter-Zweig + 22 union-lastige Pins, Dateizahl konstant); davor 178,212 @ 132 seit Phase-D-V2 — NETTO-Rückgang trotz neuer Pins: der `__variance`-Marker allein bringt Δ−10,380 (der alte erfolgreiche Widening-Check löste die teure AllOnes-Kette voll auf, der Marker-Check scheitert früh — dekomponiert im Scratch-Worktree), die V2-Substanz davor war +7,726 (Readonly<S>-Re-Expressionen + Kovarianz-Pins); davor 180,794 @ 132 seit V1 (Δ+5,082 ggü. V3s 175,712, isoliert: +1,060 reine src-Kosten der Union-Filter/RankUnknowable/Guard-Härtung + Rest die neuen union-lastigen test-d-Pins selbst, Dateizahl konstant = echte Typkosten), 175,712 @ 132 seit V3 (Δ+78 = ECHTE Typcheck-Kost der test-scripts-guard-Erweiterung, per Bisektion + gleichlanger Kommentar-Kontrollprobe (Δ+0) belegt, KEIN Order-Noise), 175,634 @ 132 seit Item 10 (172,392 @ 128 war Kern 11) — Δ from Kern 10's 174,391 @ 127 is a real, reproducible DECREASE from adding one bench file; MECHANISM PINNED (2026-07-12): the counter is CHECK-ORDER-dependent — a *empty* `export {}` file at the same path drops it ≈2,043, and the drop varies with the file's NAME/sort-position (`aaaa…` −2,034 vs `zzzz…` −304), non-monotonic — so a file addition carries an order-noise term up to ≈±2,000 that is NOT the change's real type-cost; see docs/kern-11-elementwise-fastpath-ergebnisse.md "Mechanismus GEPINNT") / `check:diag:stress` (Stress-Pin **102,877 @ 82** seit Item 11 / S1 (Δ+695, Dateizahl konstant 82, nicht bisektiert); 102,182 @ 82 seit der Union-Axis-Mini-Scheibe (Δ+86); 102,096 seit Phase-D-V2 (103,511 seit V1 (Δ−371, dim.ts-Content-Effekt, stabil, nicht bisektiert), 103,882 seit Item 10, 94,597 seit Kern 09)) / `check:diag:browser` (Browser-Korpus-Pin 2,142 @ 75 seit Phase-D-V3, ungated by design) · `pnpm test:core` (v1 differential + meta, 818 seit Phase-D-V3: +1 Guard-Invariante (d)) · `pnpm test:resident` (4278+2 = 4280 seit Phase-D-V2 (+1 NDArray-strides-Getter-Wert-Test aus der Schließungsrunde); 4279 seit Item 10; +`:gc` with --expose-gc) · `pnpm test:threaded` (69 seit Item 10; seit Item 12 baut es BEIDE Artefakte — `build:wasm && build:wasm:threads` — weil die threaded-Tests bit-identisch gegen den STABLE-Core vergleichen (numtype_core.wasm), nicht nur gegen den threads-Core; braucht also stable 1.95.0 UND die pinned nightly-2026-07-09 toolchain with rust-src, install command in scripts/build-wasm-threads.sh) · `pnpm test:browser` (Phase-D-V3: Playwright/Chromium-Smoke, 4 Tests, ≈3 s; der Wrapper emittiert IMMER frisch — NIE `playwright test` direkt aufrufen, ein mtime-Freshness-Guard wirft sonst bei stale/fehlendem `.emit`; Erstinstallation: `pnpm exec playwright install chromium`) · `pnpm test:package` (Item 11 / S3: baut zuerst `build:dist`, dann Emit-Präzisions-Gate `scripts/check-dist-emit.mjs` + Laufzeit-Smoke `spike/tests-package/package-smoke.test.ts` gegen `dist/index.js` (3 Tests: JS-`NDArray`, `backend("wasm")`-WASM-Ladepfad, `backend("threaded")`-Rejection) + Konsumenten-Typ-Smoke `spike/tests-package/consumer/` gegen `dist/index.d.ts`; braucht KEIN nightly, nur die stable-wasm) · `pnpm build:dist` (Item 11 / S2: emittiert das publizierbare Paket nach `dist/` — `build:wasm && rm -rf dist && tsc -p tsconfig.build.json && node scripts/postbuild-dist.mjs && .wasm-Kopie`; `dist/` ist gitignored, kommt via `files:["dist"]` in den Tarball; Konsument nutzt `skipLibCheck:true`-Default) · `pnpm demo` (all three backends, asserted equal) · `pnpm bench:scaling` / `bench:chain` / `bench:strided` / `bench:blocked` / `bench:slice` / `bench:threaded` / `bench:crossover` (kalibriert die Auto-Weiche, nightly) / `bench:editor` (Editor-Latenz via LSP-Harness, ≈1,2 s; seit der Union-Axis-Scheibe 7 Workloads — W7 gated den degradierten Union-Achsen-Hover) / `bench:elementwise` (Kern 11: contiguous fast-path vs. general-path add/sub/mul/div, ≈12-18x internal win at n=1024) · `cargo test --manifest-path crates/core/Cargo.toml` (161 seit Kern 11: 157 Kern-10-Basis + 2 Pfad-Äquivalenz + 2 adversarial-Follow-up-Coverage; Achtung: die zuvor dokumentierte 110 war Doku-Drift, tatsächliche Baseline war 109). Note: test scripts use EXPLICIT file lists in package.json — new test files must be added there manually; test-scripts-guard.test.ts (part of test:core) fails if a file is unlisted, double-listed across test:core/test:resident/test:threaded, or missing on disk; seit Phase-D-V3 deckt Invariante (d) auch spike/tests-browser ab (Browser-Testdateien müssen in test:browser registriert sein und dürfen in keiner node-Liste stehen).

## Item 14 — v0.1 research preview (2026-07-19, Abschluss der Roadmap)

Der Release-Tag selbst fiel unter Item 13 (numtype@0.1.0 auf npm, Repo public, Tag v0.1.0);
Item 14 war die Launch-/Sichtbarkeits-Schicht in vier Bausteinen, alle am selben Tag:
**(d)** README-Sektion „Versioning: what to expect before 1.0" — 0.x-SemVer geschärft für eine
Bibliothek, deren inferierte Typen Teil der API sind (jede Typ-Änderung = Minor, nie Patch;
die CI-gegateten Covenant-Zusagen als 0.x-stabil benannt; Typ-Präzisierung als erwartbare
Richtung). **(a)** Demo-GIF (docs/assets/numtype-demo.gif, ≈2,7 MB, absolute
raw.githubusercontent-URL): drei Szenen — matmul-Hover `NDArray<[2, 4]>`, Slice-Arithmetik
`NDArray<[900]>`, Shape-Fehler am Argument — AppleScript-getippt in VS Code (TS 6.0.3, die
gebündelte Version; TS 7 hat keinen tsserver, die klassische Integration kann ihn nicht laden —
alle drei Szenen wurden vorab empirisch gegen 6.0.3 UND 7.0.2 verifiziert, erster Datenpunkt
zur offenen Alt-Major-Frage). Fünf Anläufe; die Fehlschläge waren lehrreich genug für eine
eigene KB-Notiz (Kern: globale Hotkeys kollidieren mit synthetischem Tippen — QWERTZ-Klammern
sind Option-Kombos, `[1024]` = Doppel-Option-Tap → löste Claudes Quick-Entry aus und schickte
einen halben Prompt ab; Lösung: usage-first-Choreografie gegen Unused-Diagnosen, dediziertes
Einzel-Keybinding statt Cmd+K-Chord, expliziter Modifier-Down/Up, Schritt-Log, absolute
Go-to-Line-Navigation). Produktions-Setup in ~/Documents/CODE/numtype-demo-gif (außerhalb des
Repos), als erster echter Konsument des veröffentlichten Pakets. **(c)** docs/README.md als
englischer Reading Guide (Konventionen + Ehrlichkeitsregel, begründeter Sprachhinweis, drei
Leser-Pfade, „If you only read three documents"; 44 Links verifiziert) — bewusst keine
Massenübersetzung. **(b)** Launch-Blog-Post „Teaching the type checker arithmetic"
(https://marvinmuegge.com/notes/teaching-the-checker-arithmetic/), in eigener Session nach
Prompt-Vorgabe verfasst, Owner-publiziert; das Code-Beispiel nachträglich gegen das
veröffentlichte Paket verifiziert (Rang-1-matmul: `inner dimensions 900 and 5 do not match`,
echt). Dazu im Zuge: Prior-Art-Credit für die Digit-String-Repräsentation (ts-arithmetic) in
„The core idea" — Beitrag ist die Anwendung, nicht der Trick. Offener Mini-Befund: der Post
verlinkt Repo/npm nicht (liegt auf der Website). Die Roadmap ist damit vollständig
durchgespielt; weiter geht es post-Roadmap (FOLLOWUPS-Minis, optional Trusted Publishing).

## Dogfooding-Scheibe — RAG-Demo auf dem veröffentlichten Paket (2026-07-20, post-Roadmap)

Erste Scheibe des OSS-Wachstumskurses (Owner-Reihenfolge Punkt 2): eine echte
Konsumenten-Anwendung auf numtype@0.1.1, WIE SIE EIN NUTZER ERLEBT — `examples/rag-demo`
installiert das veröffentlichte Paket aus der npm-Registry (eigenes Install-Root, committetes
Lockfile), kein Workspace-Link auf den lokalen Stand. Deterministische RAG-Retrieval-Demo:
16 Dokumente, 8 Queries, from-scratch gehashte Zeichen-Trigramm-TF-Embeddings (djb2, D=256),
L2-Normalisierung als Matrix-Ausdruck, EIN matmul `[8,256] @ [256,16]` für die gesamte
Ähnlichkeitsmatrix, Ranking + Margin-Assertions (Schwelle 0.03, knappste echte Margin 0.0778),
Mean-Pooling-Sektion, zwei `@ts-expect-error`-Shape-Pins gegen die dist-Typen. Kern-Deliverable
war nicht die Demo, sondern der Friction-Log (F1–F6) und die kuratierte Op-Wunschliste W1–W5
(docs/dogfooding-rag-ergebnisse.md): **argmax/topk (P1, doppelt aufgetreten, null Ersatz) >
Skalar-Overloads für add/sub/mul/div (P2, macht mean fast gratis) > elementweises sqrt als
benannte exakte Unary-Op (P3, IEEE-korrekt gerundet = determinismus-sicher, KEIN Transzendenten-
Gate-Fall) > stack/fromRows (P4) > item/at (P5)**. Ehrliche Kalibrierung gegen die
HANDOFF-Erwartung: argmax/stack bestätigt, mean granularer als erwartet (die echte Lücke ist die
Skalar-Division), concat trat NIE auf (bleibt evidenzlos ohne Listenplatz), sqrt war der
unerwartete Doppel-Fund; Transzendente wurden im gesamten Workload nicht gebraucht.

Prozess: bindende Spec (docs/dogfooding-rag-spec.md, v1→v3) mit Baustein-0-Spec-Verify VOR der
Implementierung — der Verifier fing einen echten Blocker: pnpm-11-Default `minimumReleaseAge`
(≈24h) hätte das CI-Gate nach JEDEM künftigen Release gebrochen (`ERR_PNPM_MINIMUM_RELEASE_AGE_
VIOLATION`, dreifach reproduziert inkl. Cold-HOME); Mitigation = committetes pnpm-workspace.yaml
mit `minimumReleaseAge: 0` im Example. Verify-Runde A+B+C parallel: A CONFIRMED (alle Gates
frisch, check:diag-Pin 187,918 @ 135 exakt, Pflicht-Mutant beißt; klärte nebenbei die
test:core-Zahl: 822 auch am Vor-Scheiben-HEAD — die CLAUDE.md-„818" war vorbestehende Drift).
B HÄLT-mit-Befunden, MAJOR-Fund: „kein Root-pnpm-install" im CI-Job war empirisch falsch (pnpm
installiert beim Aufruf JEDES Root-Scripts auf kaltem Runner implizit die Root-devDeps) →
CI-Job auf direkte `-C`-Steps umgestellt (Spec-v3-Nachtrag); dazu engines-Feld, F-Nummern-
Angleich, `≈`-Fix. B bewies zudem Nicht-Vakuität breit (Margin-Mutant, Hash-Korruptions-Mutant,
@ts-expect-error-Positionsproben, Cold-Install-Repro des Baustein-0-Blockers). C (covenant-
verify): Z1/S1/M1–M5/Nicht-Ziele halten; EIN mittlerer Z2-Befund, unverdünnt: test:package prüft
ein Bauergebnis des AKTUELLEN Commits, test:example ein eingefrorenes Registry-Artefakt eines
VERGANGENEN — der Korpus rottet „still zwischen Releases". Owner-Entscheidung offen (FOLLOWUPS:
Covenant-v5-Präzisierung vs. mechanischer Registry-Tripwire), keine stille Auflösung.

## Op-Scheibe W1 — `argmax`/`topk` auf `NDArray` (2026-07-20, post-Roadmap)

Erste konkrete Op aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md, W1/F4 —
argmax trat zweimal auf, null Ersatz in der Surface). Bindende Spec
(docs/op-w1-argmax-topk-spec.md, v2 nach Baustein-0-Addendum): D1 grenzt bewusst auf die naive
JS-Klasse ein (kein WASM-Kernel, keine `WNDArray`/Threaded-Parität — FOLLOWUPS-Eintrag). `argmax`
übernimmt exakt `sum`s Arity-0/1/2-Overload-Muster (`ReduceAxis`/`Guard`/`OkShape` unverändert
wiederverwendet), mit EINER bewussten Abweichung: die niladische Form gibt `number` zurück
(Scalar-Consumer-Präzedenz von `dot`/`norm`/`cosineSimilarity`), nicht `NDArray<[]>`. `topk`
(Rang-1-only, `{values, indices}` im `torch.topk`-Stil) brauchte neue Typ-Maschinerie —
`TopkCheck<S,K>`/`TopkShape<S,K>` in vector.ts, appended, wiederverwendet die bestehende
Digit-String-Arithmetik aus literal-arithmetic.ts (`Compare`+`NonNegDigits`, minimal exportiert —
vorher unexportiert, Baustein-0-Blocker, Owner-Entscheidung „Exporte ergänzen"), NICHT
`LiteralIndexBounds` (dessen Index-Semantik `k=D` fälschlich als „out" und negatives `k`
fälschlich als „in" klassifiziert hätte — empirisch bewiesen, als verbindliche Warnung in die
Spec eingearbeitet). Datei-Disziplin D5 v2: runtime.ts/vector.ts zeigen im Diff AUSSCHLIESSLICH
Additionen nach dem letzten Bestandscode (vector.ts brauchte dafür einen zweiten, eigenständigen
`import`-Block statt die bestehende Zeile zu erweitern); ndarray.ts-Klassenkörper insertion-only,
Import-Zeilen am Dateikopf erweitert (Präzedenz: `sum`s eigene Importzeile wuchs genauso über
mehrere Kerne).

Zwei echte Befunde während der Implementierung, beide gefangen und gefixt VOR dem Commit: (1)
`argmax(undefined, true)` fiel erst fälschlich in den niladischen `number`-Zweig (Check war
`axisNum === undefined` statt `arguments.length === 0` — TS unterscheidet die Overloads nach
Argument-ANZAHL an der Call-Site, nicht nach Wert; ein 2-Arg-Aufruf mit Achsenwert `undefined`
verwarf so still `keepdims`), reproduzierbar rot vor dem Fix, grün danach. (2) Ein Test mit
handkonstruierter NICHT-kanonischer NaN-Payload (`0x7FF800000000DEAD`) zeigte gelegentlich die
kanonische statt die echte Payload — bisektiert auf eine V8-JIT-Tier-Eigenheit des bestehenden
`bitsOf`-Helfers (`new Float64Array([x])`-Array-Literal-Konstruktion), NICHT auf `topkRuntime`
selbst (separat per direktem `DataView`-Buffer-Read 5/5-mal als korrekt bewiesen); Test auf einen
lokalen `bitsAt`-Helfer umgestellt (Buffer-Read statt Array-Literal), seither deterministisch
grün über mehrere volle Testfile-Läufe.

Pin-Protokoll (D7 v2, gestufte Attribution, alle Zwischenpunkte gemessen): Baseline im frischen
Worktree exakt reproduziert (187,918 @ 135 · stress 102,877 @ 82 · browser 2,142 @ 75). ①
runtime.ts+ndarray.ts(nur argmax)+literal-arithmetic-Exporte: 188,383 (+465). ② +vector.ts-
Maschinerie+ndarray.ts-topk: 188,726 (+343). ③a neues Testfile LEER registriert: 179,186 @ 136
(−9,540 — Order-Noise, Datei-Hinzufügen reshuffelt die Fresh-vs-Cached-Instantiation-Partition,
deutlich über dem „±≈2,000"-Präzedenzfall, aber dieselbe dokumentierte Mechanik). ③b Testfile
GEFÜLLT: 182,249 (+3,063 echte Testkosten). final +test-d.ts-Pins: **184,225 @ 136 (+1,976)**.
Gesamtwachstum ggü. Baseline: **−3,693 — eine NETTO-ABNAHME**, weit innerhalb des
Absolut-Gates ≤+12,000 (Order-Noise dominiert die echten Neukosten). Zweimal gemessen,
byte-identisch. `bench:editor` W1–W7 verschoben sich UNIFORM um +804 (D7-explizit erlaubt,
Latenz/Correctness-Gate unverändert PASS) — Pins in editor-latency.ts aktualisiert.

**Ein offener Befund, ehrlich berichtet statt stillschweigend hingenommen:** `check:diag:stress`
verschob sich um +842 (102,877→103,719 @ 82, zweimal deterministisch reproduziert) — eine
Abweichung von T2s „stress/browser EXAKT unverändert"-Anforderung (`check:diag:browser` hielt
exakt, 2,142 @ 75). Bisektiert (temporärer Revert + Re-Messung, danach exakt wiederhergestellt):
`argmax` allein +469, `topk`s inkrementeller Beitrag +373 — Ursache ist NDArrays gewachsene
Klassen-Member-Fläche (zwei neue überladene generische Methoden), die JEDE `NDArray<S>`-
Instanziierung im stress-Korpus (der viele große literale Shapes für Digit-Arithmetik-
Grenzfälle instanziiert) marginal mehr Auflösungsarbeit kostet — dieselbe Ripple-Klasse wie bei
bench:editor, nur dass D7 diese Verschiebung dort EXPLIZIT erlaubt, für stress/browser aber
straffer formulierte, als die Realität hergab. NICHT ins Gate optimiert (Code nicht verkleinert,
um den alten Pin zu erzwingen); Owner-/Verify-Entscheidung, ob der Pin analog zu bench:editor
mitgezogen wird, steht aus.

Tests: `spike/tests-runtime/argmax-topk.test.ts` (30 Fälle, in test:core registriert — kein
WASM-Gegenstück existiert für D1, also kein klassischer Differential-Partner: Coverage kombiniert
mutations-scharfe Fixtures, eine selbstverifizierende Wort-für-Wort-Stem-Gleichheitsprobe gegen
`sumRuntime`s eigenen Throw, unabhängig geschriebene Brute-Force-Referenzen über ≈150
Zufallsfälle je Op inkl. NaN-Injektion, strukturelle keepdims-Invarianten und transponierte/
gesliceste Empfänger mit unabhängig hergeleiteten Erwartungswerten) + 37 neue Typ-Pins in
ndarray.test-d.ts (exakte Tupel, alle Degradationskanten, `@ts-expect-error` AM `k`-/Achsen-
Argument mit Mutationsprobe, vier Message-Gleichheits-Pins via `Guard<TopkCheck<…>,…>`+`Equal<>`,
MAX_SAFE_INTEGER-Kante). test:core 822→**852**. Alle D8-Gates frisch grün: `pnpm check`
(Dreier-Verbund), test:resident 4278+2 unverändert, cargo 161 unverändert (kein Rust berührt),
`check:freeze`-Hash byte-identisch, `graph-a-lama query lint` 0/0, `pnpm test:example` weiterhin
auf numtype@0.1.1. README: neuer eigenständiger Satz im „What's implemented"-Abschnitt (NICHT im
bit-for-bit-Usage-Block, der für argmax/topk falsch wäre) mit explizitem
„TypeScript-runtime surface only (no WASM kernel yet)"-Caveat. Vollständige Zahlen, Pin-Tabelle
und der offene stress-Befund: docs/op-w1-argmax-topk-ergebnisse.md. Post-Verification-Addendum
(Verify-Runde A+B+C) folgt.

### W1-Nachtrag: Verify-Runde & Abschluss (2026-07-20)

Verify-Runde A+B+C parallel: **A CONFIRMED** (beide Pflicht-Mutanten beißen exakt, alle Gates
doppelt deterministisch, Attributions-Tabelle nachgerechnet), **B HÄLT-mit-Befunden** (eigener
220-Shape-Differential 0 Abweichungen; Friction-Rückprobe: der F4-Workaround der RAG-Demo ist
real durch topk(2) ersetzt; Befunde F1 vorbestehender non-integer-Achsen-Fallback → FOLLOWUPS,
F2 RankUnknowable-Kante → Spec v4 folgt der D-V1.3-Hauspolitik statt der Spec-v1-Formulierung,
Policy-Pin ergänzt), **C kein Verstoß** (M1-Auslegung: kernel-lose Referenz-Ops nicht verboten;
Owner-Empfehlung M1-Präzisierung vor W2–W5 → FOLLOWUPS). Finale Pins: Haupt 184,330 @ 136
(Netto −3,588 zur Vor-W1-Baseline, Order-Noise-dominiert; echte W1-Maschinerie-Kosten in der
Attributions-Tabelle des Ergebnisse-Docs), stress 103,719 @ 82 (akzeptierter, attribuierter
Klassen-Surface-Ripple +842), browser 2,142 @ 75 (exakt). test:core 852. bench:editor-Pins
W1–W7 +804 uniform, doppelt gemessen. Prozess-Notiz: Baustein 0 fing den Export-Blocker VOR
dem Bau (Owner-Entscheid „Exporte ergänzen"), die MAJOR-Warnung vor LiteralIndexBounds hat die
Implementierung nachweislich befolgt — zweite Scheibe in Folge, in der der Pre-Impl-Verifier
den teuersten Fehler abfing.

### W2: Skalar-Overloads (add/sub/mul/div) + `mean` (2026-07-21)

Zweite Op-Scheibe aus der Wunschliste (docs/dogfooding-rag-ergebnisse.md W2/F2 — der
`fromArray([1],[2])`-Skalar-Wrap-Workaround, HANDOFF-Erwartung „mean" als granularere Lücke),
docs/op-w2-scalar-mean-spec.md (Version 2 nach Baustein-0-Addendum). Baustein 0 fing VOR dem Bau
einen echten BLOCKER: die v1-Annahme „Skalar-Overloads = reine Insertion" war falsch — TS2394
verbietet Overload-Signaturen vor einer body-tragenden Deklaration, die vier Bestandsmethoden
add/sub/mul/div MÜSSEN ediert werden (kein Richtungs-Spielraum, Alternativen brechen D1/D2) →
als erzwungene D6-v2-Ausnahme in die Spec gearbeitet (bodylose Overload-Signatur + neue
Skalar-Overload-Signatur + neue union-typisierte Implementierungssignatur, deren Rumpf die
ORIGINALE Logik byte-identisch in den `else`-Zweig verschiebt). Umgesetzt: `runtime.ts`
(Append) bekommt `scalarElementwiseRuntime(op, data, s)` (String-Dispatcher, elementweise
`data[i] op s`) und `meanRuntime` (`sumRuntime` + GENAU EINE Division pro Output-Element durch
`n` — bewusst NICHT `sum*(1/n)`, andere f64-Rundung); `ndarray.ts` konvertiert die vier
Bestandsmethoden nach D6-v2 (git diff bestätigt: die drei Rumpfzeilen jeder Methode erscheinen
als reine Kontextzeilen, kein `+`/`-` — byte-identisch verschoben, nicht neu geschrieben) und
bekommt eine neue `mean`-Methode (Overloads 0/1/2 exakt nach `sum`-Muster, KEIN
`arguments.length`-Sonderfall nötig — anders als `argmax` geben alle `mean`-Overloads
`NDArray<...>` zurück, die W1-Verwechslungsgefahr entfällt strukturell). D2 v2: ein
UNION-Argument über die Overload-Grenze (`number | NDArray<B>`) wird von TS als Ganzes
abgelehnt (TS2769) — dieselbe Kante, die `NDArray.backend(kind)` schon trägt, inkl.
funktionierendem `typeof`-Narrowing-Workaround, per Mutationsprobe (Direktive entfernt → echter
TS2769/TS2345 mit der vorhergesagten Message) non-vakuös bewiesen.

Der bindende Determinismus-Punkt D5 (`sum/n`, nie `sum*(1/n)`) ist zweifach nicht-vakuös
konstruiert: volle Reduktion (n=49, sum=5 → `5/49 = 0.10204081632653061` vs.
`5*(1/49) = 0.1020408163265306`, verschiedene letzte Nachkommastelle) UND ein Achsen-Fall
(shape=[4,49], Zeilensummen [5,9,1,2] — genau 2 von 4 Zeilen diskriminieren, die Spec-Warnung
„nicht jedes Beispiel diskriminiert" ist damit selbst bewiesen, nicht nur zitiert). Neues Testfile
`spike/tests-runtime/scalar-mean.test.ts` (482 Tests, in test:core registriert): expliziter
Op×Rang(0/1/2)×Spezialwert-Katalog gegen den nativen IEEE-Operator selbst, 160 randomisierte
`[1]`-Wrap-Byte-Äquivalenz-Fälle, ein Rang-0-Kontrast-Test gegen den ALTEN `[1]`-Wrap-Workaround
(beweist D2s Motivation als echten Lauf), 300 randomisierte `mean`-Cross-Checks gegen eine
unabhängig geschriebene Brute-Force-Referenz, `mean`-von-empty → NaN auf beiden
Reduktionspfaden (Kontrast zu `argmax`, das dort wirft), Stem-Wortgleichheit, `mean(undefined,
true)`. Typ-Pins (+19: 17 benannte `Expect<Equal<...>>` + 2 `@ts-expect-error`) in
ndarray.test-d.ts: `div(2)`-Shape-Erhalt exakt (Rang 0/2/wide/Readonly-S), Union-über-Grenze +
Narrowing-Workaround, `mean`-Wiring nach argmax-Präzedenz (niladisch, positive/negative Achse,
keepdims, plus die vier von der Spec namentlich verlangten Degradationsfacetten dyn-axis/
union-axis/mixed-rank/keepdims-union, plus OOB-Message-Pin) — bewusst mehr als die „≈4-6"-
Schätzung des Spec-Addendums (10 statt 4–6 in der `mean`-Gruppe), weil D7 den niladischen Pin
separat verlangt und „argmax-Muster" selbst ≈10 Pins trägt; ehrlich als Abweichung im
Ergebnisse-Doc vermerkt statt stillschweigend übernommen.

Pin-Protokoll (gestufte Attribution, empty-then-fill dekomponiert): Baseline im frischen
Worktree exakt reproduziert (184,330 @ 136 · stress 103,719 @ 82 · browser 2,142 @ 75). ①
runtime.ts+ndarray.ts (D6-v2 + mean): 185,204 (+874, Klassen-Surface-Wachstum). ②a neues
Testfile LEER registriert: 187,404 @ 137 (+2,200, Order-Noise). ②b Testfile GEFÜLLT: 189,368
(+1,964, echte Testkosten). final +test-d.ts-Pins: **190,092 @ 137 (+724)**. Gesamtwachstum
ggü. Baseline **+5,762**, deutlich innerhalb des Absolut-Gates ≤+10,000 — zweimal gemessen,
byte-identisch. stress: 103,719→**104,900 @ 82 (Δ+1,181)**, derselbe Klassen-Surface-
Ripple-Mechanismus wie W1s +842 (Datei-Anzahl unverändert, `spike/tests-runtime` ist nicht Teil
dieses Korpus), zweimal deterministisch reproduziert. browser: unverändert 2,142 @ 75, exakt.
`bench:editor` W1–W7 verschoben sich UNIFORM um +1,181 (w1/w2/w3/w5/w6/w7); w4 (die
Fehler-Datei mit den zwei absichtlichen Typfehlern) verschob sich um +1,220 — eine echte,
zweifach reproduzierte, attribuierte Abweichung (die `ShapeError`/`Guard`-Diagnosepfade lösen
sich gegen das größere Overload-Set anders auf), nicht weiter root-caused (Diagnosewert, kein
Korrektheitsrisiko). Pins in `editor-latency.ts` aktualisiert, `check:diag`-neutral verifiziert
(190,092 unverändert vor/nach dem reinen Daten-/Kommentar-Edit). Alle D8-Gates frisch grün:
`pnpm check` (Dreier-Verbund), test:core 852→**1,334** (482 neu), test:resident 4278+2
unverändert, cargo 161 unverändert (kein Rust berührt), `check:freeze`-Hash byte-identisch,
`bench:editor` Hard-Gate PASS nach Pin-Update, `graph-a-lama query lint` 0/0, `pnpm test:example`
weiterhin auf numtype@0.1.1, `pnpm demo` PASS (zusätzliche Absicherung). README: neuer
eigenständiger Absatz im „What's implemented"-Abschnitt direkt nach der W1-Notiz (die
bit-for-bit-Zeile im Usage-Codeblock bleibt unangetastet — der Block ruft `.mul()`/`.div()` nur
mit NDArray-Argumenten auf). FOLLOWUPS: das W1-Paritätsitem um einen W2-Nachtrag erweitert
(Skalar-Overload + `mean` fehlen auch auf `WNDArray`). Vollständige Zahlen, Diskriminator-Beispiele
und der Byte-Erhaltungs-Nachweis: docs/op-w2-scalar-mean-ergebnisse.md. Post-Verification-Addendum
(Verify-Runde A+B+C, Stufe 3) steht noch aus.

### W2-Nachtrag: Verify-Runde, F1-Fix & Recovery (2026-07-21)

Verify-Runde A+B+C: **A CONFIRMED** (drei Mutanten beißen, Byte-Erhaltung am Diff),
**B HÄLT-mit-Befunden** mit EINEM echten MAJOR: Der Overload-Umbau ließ die Shape-Message
des häufigsten Fehlerfalls (simpler Broadcast-Mismatch) hinter dem number-Decoy
verschwinden — TS meldet den Fehler des LETZTEN Overload-Kandidaten, kein bestehender Pin
sah Message-INHALT. Fix: Deklarations-Reihenfolge getauscht (Skalar zuerst, Guard-Träger
zuletzt, jetzt bindend in Spec v3) + neuer Diagnose-Qualitäts-Pin (echter tsc-Lauf auf
Außer-Repo-Fixture, assertiert den Broadcast-Stem; Nicht-Vakuität per Reihenfolgen-Mutant
bewiesen; drei schmale ambient.d.ts-Shims). **C kein Verstoß** — erste Anwendung des
M1-v5-Wortlauts (Paritätslücken-Bedingung erfüllt); ein M2-Wortlaut-Grenzfall
(Union-über-Overload-Grenze) als v6-Kandidat nach FOLLOWUPS. **Prozess-Zwischenfall,
offengelegt:** ein versehentliches `git checkout --` beim Mutanten-Revert warf die
uncommittete ndarray.ts auf HEAD zurück (exakt der Fall der Template-Regel „Mutanten als
revertierter Edit, nie checkout"); Recovery byte-genau durch den Implementierungs-Agenten
aus dessen Kontext, erneut am Diff verifiziert. Finale Zahlen: Haupt-Pin 188,563 @ 137
(+4,233 zur W1-Baseline), stress 104,900 @ 82, browser 2,142, test:core 1,335,
bench:editor PASS (w4 26453), Hash byte-identisch. Damit sind die Wunschlisten-Plätze
1 UND 2 geschlossen: `x.div(2)` liest sich als durch-2-teilen, `mean` existiert in
allen drei Formen — der F2-Workaround der RAG-Demo ist obsolet (Rückprobe bit-identisch).

### W3: `sqrt` — dritte Op-Scheibe der Dogfooding-Wunschliste (2026-07-21)

Wunschlisten-Platz 3 (docs/dogfooding-rag-ergebnisse.md W3/F1 — zweifacher Bruch der
natürlichen L2-Normalisierungs-Kette `mul→sum(axis)→sqrt→reshape→div` in der RAG-Demo, weil
kein `.sqrt()` existierte) ist geschlossen: `NDArray.sqrt(): NDArray<S>`, shape-erhaltend
bei jedem Rang inkl. Rang 0, niladisch (kein Guard, wie `norm()`/`flatten()`). Baustein-0
(brainroute:deep, kompakt) fand keinen Blocker — primärquellen-verankerte IEEE-Begründung
(ECMA-262 `sec-math.sqrt`: exakte 𝔽-Rundung, im Gegensatz zu jeder transzendenten
`Math.*`-Methode, die die Spec wörtlich „implementation-approximated" nennt), gemessener
Typ-Anteil +24 (Probe-Worktree), keine Symbolkollisionen. Umsetzung: `sqrtRuntime` als
reiner Append in runtime.ts (elementweise `Math.sqrt`, frisches Array), `sqrt()` als reine
Klassenkörper-Insertion in ndarray.ts nach `mean` (kein Bestandsmember editiert — anders als
W2s D6-v2-Overload-Umbau, hier reicht eine reine Append, da `sqrt` kein Overload-Partner-
Problem hat), W3-Testblock (227 Tests) an das bestehende `scalar-mean.test.ts` angehängt
(kein neues File), 6 neue Typ-Pins (5 `Equal` + 1 `@ts-expect-error`) an `ndarray.test-d.ts`.
D1 bewusst NDArray-only, kein WASM-Kernel (dieselbe COVENANT-v5-gedeckte Surface-Asymmetrie
wie W1/W2) — FOLLOWUPS-Paritätsitem um einen W3-Nachtrag erweitert. Die F1-Schließung ist
ZWEIFACH bewiesen, byte-identisch gegen die alte Hand-Loop-Formulierung aus
`examples/rag-demo/main.ts`: die Teilkette `m.mul(m).sum(1).sqrt()` UND die volle
L2-Normalisierung `m.div(m.mul(m).sum(1).sqrt().reshape([N,1]))`. Finale Zahlen: Haupt-Pin
190,636 @ 137 (+2,073 zur W2-Baseline, Absolut-Gate ≤ +3,000 eingehalten), stress 104,900 @
82 (Δ 0 — anders als W1/W2 diesmal KEIN Klassen-Surface-Ripple), browser 2,142 (Δ 0),
test:core 1,562 (+227), test:resident 4,278+2 unverändert, cargo 161 unverändert (kein Rust
berührt), `check:freeze`-Hash byte-identisch, `bench:editor` Hard-Gate PASS ohne
Pin-Abweichung, `graph-a-lama query lint` 0/0, `pnpm test:example` weiterhin auf
numtype@0.1.1. README: die W1/W2-Op-Notiz im „What's implemented"-Abschnitt um `sqrt`
ergänzt (bit-for-bit-Zeile bleibt wahr). Vollständige Zahlen und der F1-Schließungs-Beweis:
docs/op-w3-sqrt-ergebnisse.md. Post-Verification-Addendum (Verify-Runde, Stufe 3) steht noch
aus.

### W3-Nachtrag: Verify-Runde (2026-07-21)

A CONFIRMED (Mutant 219/227 rot — die 8 grünen sind exakt die abs≡sqrt-Fälle) ·
B HÄLT-mit-Befunden (zwei kleine Coverage-Lücken in-slice geschlossen:
Aliasing-Isolations-Test + größter-Subnormal-Pin; NaN-Payload-Kanonisierungs-Detail
dokumentiert) · C NULL Befunde mit eigenständigem Doppel-Urteil (sqrt ist algebraisch,
nicht transzendent; IEEE-Pflichtrundung — ECMA-262-Primärquelle seit Baustein 0).
Final: 190,640 @ 137 (+2,077), stress Δ0 (niladischer Member rippelt nicht — Kontrast
zu W1/W2 dokumentiert den Mechanismus weiter), test:core 1,564. Damit ist auch
Wunschlisten-Platz 3 geschlossen: die L2-Normalisierung der RAG-Demo läuft komplett
in numtype, byte-identisch zur alten Hand-Loop-Formulierung bewiesen.

### W4: `stack` — vierte Op-Scheibe der Dogfooding-Wunschliste (2026-07-21)

Wunschlisten-Platz 4 (docs/dogfooding-rag-ergebnisse.md W4/F5 — der selbstgebaute
`embedMatrix`-Zeilen-Flatten-Helper in examples/rag-demo/embedding.ts, `np.stack`-Reflex)
ist geschlossen: `NDArray.stack(rows)` — nur Rang-1-Zeilen gleicher Länge → Rang-2 `[N, D]`.
Baustein 0 (brainroute:deep, frischer Scratch-Worktree mit kompilierender Skizze) fand acht
verbindliche Typ-Formen VOR dem Bau (F1-F8, Spec-Addendum): Schichtung auf `readonly
Shape[]` statt `NDArray` in vector.ts (Zyklus-Vermeidung); homomorpher Mapped Type
`RowShapesOf<Rows>` statt der invarianz-kollabierenden `Rows[number]`-Extraktion (F2,
BLOCKER); ein `Shapes["length"] extends 0`-Gate vor jeder Element-Extraktion (F3, die
Leer-Tupel-Falle); Tupel-Wrapped-Akkumulator-Narrowing im Fold (F4); ein eigener
Array-Pfad via `number extends Shapes["length"]` (F5, Tupel-Rekursion matcht Arrays nie);
Wide-Sentinel-Dim-Merge nach CompatDim-Präzedenz (F6); Ablehnung eines Arrays mit
uniform beweisbar falschem literalen Rang (F7, sound weil auch das leere Array wirft);
IsUnion-Filter für Array-Union-Elementtypen (F8).

Umsetzung: `StackCheck`/`StackShape` als APPEND in vector.ts (eigener Import-Block, drei
gepinnte Message-Templates); `RowShapesOf`/`UnwrapRow` + die statische `stack`-Methode als
Insertion in ndarray.ts NACH `fromArray` (Baustein-0-Empfehlung: stack ist konzeptionell
ein Konstruktor); `stackRuntime` als APPEND in runtime.ts (ein Links-nach-rechts-Durchlauf,
dieselbe Reihenfolge wie der Typ-Fold, dann `Float64Array#set`-Zeilenkopie — exakt
`embedMatrix`s Algorithmus). Ein eigener Scratch-Probe (isolierter `tsc`-Lauf gegen einen
Symlink auf spike/src, außerhalb des Repos) fing WÄHREND der eigenen Verifikation einen
echten, von der Baustein-0-Skizze nicht abgedeckten Bug: `RowShapesOf`s naiv inline
geschriebener homomorpher Mapped Type kollabierte für ein Array mit UNION-Elementtyp
(der F8-Testfall) zu `readonly [number, never]` statt `[number, number]` — derselbe
Invarianz-Kollaps-Mechanismus wie F2, aber innerhalb der Array-Element-Auswertung der
Mapped-Type-Maschinerie selbst (TS wertet den Element-Typ-Ausdruck für ein Array EINMAL
non-distributiv gegen den — hier: Union — Elementtyp aus). Fix: `UnwrapRow<R>` als eigene
Generic mit eigenem naked Type-Parameter (derselbe „extra Generic erzwingt Distribution"-
Kunstgriff wie `ArrayRowD` in vector.ts) — nach dem Fix liefert der Probe korrekt
`[number, number]`, verifiziert am Typ-Pin `STACK_ARRAY_UNION`.

W4-Testblock (8 neue Tests) an scalar-mean.test.ts angehängt (kein neues File):
Stem-Pins über `stackRuntime` DIREKT und über die öffentliche API via dynamischer-Rang-
Zeilen (dieselbe „widen-past-the-guard"-Technik wie `mean(5)`s Achsen-Pin, kein unsicherer
Cast nötig); 1/2/3-Zeilen; D=0; ein Byte-exakter NaN-Payload-Test (`bitsOf`, mirroring
special-values.test.ts's Transpose-Fixture); die F5-Rückprobe (`embedMatrix`s Algorithmus
LOKAL nachgebaut, nicht importiert — das Beispielpaket bleibt bewusst außerhalb des
spike/-Kompilationsgraphen, ein Import hätte check:diags Dateizahl kontaminiert);
Large-N-Smoke (5.000×8); Aliasing-Isolation (W3-Lektion). 19 neue Typ-Pins in
ndarray.test-d.ts decken jede D2-Kante inkl. Message-Equality-Pins am Argument.

Finale Zahlen: Haupt-Pin 194,545 @ 137 (+3,905 zur W3-Baseline, Absolut-Gate ≤ +8,000
mit deutlichem Spielraum eingehalten), stress 105,752 @ 82 (+852, Klassen-Surface-Ripple
wie W1/W2 — ein neuer statischer Member rippelt über jede `NDArray<S>`-Instantiierung im
Korpus), browser 2,142 @ 75 (Δ0), test:core 1,572 (+8), test:resident 4,278+2 unverändert,
cargo 161 unverändert (kein Rust berührt), `check:freeze`-Hash byte-identisch,
`bench:editor` zunächst FAIL (uniform +845 auf allen sieben Workloads, 2× deterministisch
reproduziert — anders als W2s Verify-B-Fund differenziert dieser Ripple NICHT zwischen der
Fehler-Workload w4 und den übrigen, da `stack` keinem der add/sub/mul/div/mean-Overloads
hinzufügt), nach Pin-Update PASS; `graph-a-lama query lint` 0/0; `pnpm test:example`
weiterhin auf numtype@0.1.1; `pnpm test:package` PASS. README: neuer eigenständiger Absatz
nach der sqrt-Notiz. FOLLOWUPS: das Paritätsitem um einen W4-Nachtrag erweitert
(WNDArray/Rust-Kernel-Parität fehlt auch für `stack`). Vollständige Zahlen, der
F5-Schließungsbeweis und der Baustein-0-Fund im Detail: docs/op-w4-stack-ergebnisse.md.
Post-Verification-Addendum (Verify-Runde, Stufe 3) steht noch aus.

### W4-Nachtrag: Verify-Runde-Fix (Baustein B, BLOCKER-Klasse, 2026-07-21)

Baustein B fand einen zweiten, von der eigenen Umsetzungs-Verifikation nicht gefangenen
echten M2-Verstoß: `NDArray.stack([fixed, row])` mit `row: NDArray<[3]>|NDArray<[4]>` — eine
GEWÖHNLICHE Union über einen Ternary, keine `stack`-spezifische Konstruktion — kompilierte
konfident als `readonly [2, 3]` und warf zur Laufzeit. Root Cause: `UnwrapRow`s (bewusst für
F8s Array-Pfad) erzwungene Distribution über ein naked `R` distribuiert AUCH an einer
TUPEL-Position, deren eigener Zeilen-Typ zufällig eine Union ist; `StackFold`s naked
`Head extends readonly [infer D]`-Check distribuiert weiter, gabelt den Fold in parallele
Fortsetzungen mit unterschiedlichen Verdikten — Ergebnis eine gemischte Union
`Dim | ShapeError<...>`, die `Guard`s uniform-error-only-Ablehnung passieren lässt und deren
`ShapeError`-Zweig `StackShape`s `Extract<..., Dim>` still wegwirft. Dieselbe Fehlerform wie
`reduce.ts`s eigene `ReduceAxis`-Lektion (Union-Axis-Mini-Scheibe D-A.2) — dieselbe Lösung:
ein `IsUnion<Head>`-Gate VOR dem naked Match, Position load-bearing (dokumentiert im
StackFold-Kommentar mit explizitem `ReduceAxis`-Verweis). Sechs neue Pins in
ndarray.test-d.ts (Bs Repro beide Reihenfolgen, Doppel-Mismatch-Union, direkte
`StackDimMerge`-Wide-Abdeckung beide Reihenfolgen — F-ADV-2-Schließung, Array-Element-Union
verschiedener Ränge — Verify-C-Lücke, empirisch per Scratch-Probe bestätigt). Nicht-Vakuität
per Backup-Kopie-Mutant bewiesen: den `IsUnion<Head>`-Zweig entfernt → `pnpm tsc --noEmit`
schlägt mit exakt 4 Fehlern fehl (beide Repro-Pins + der Doppel-Mismatch-Aufruf selbst +
dessen Equal-Check), kein anderer Pin betroffen; Restore aus der Kopie, `diff` UND MD5 vorher/
nachher identisch. Finale Zahlen (2× je Messpunkt): Haupt-Pin **195.481 @ 137** (+4.841 zur
W3-Baseline, davon +936 allein der Fix), stress **105.758 @ 82** (+858, davon +6 der Fix),
browser 2.142 @ 75 unverändert, test:core weiterhin 1.572 (Fix ist rein typseitig),
`bench:editor` erneut uniform +6 verschoben und neu gepinnt, Hash weiterhin byte-identisch,
`graph-a-lama query lint` weiterhin 0/0. Pin-Zählungsfehler im Ergebnisse-Doc nebenbei
korrigiert (F-ADV-3: tatsächlich 16 Pins aus der Erst-Umsetzung, nicht 19 — jetzt 22 gesamt
mit den sechs neuen). Vollständiger Befund + Fix-Beweis: docs/op-w4-stack-ergebnisse.md,
Abschnitt „F-ADV-1".

### W4-Nachtrag: Verify-Runde mit In-Slice-Blocker-Fix (2026-07-21)

Die Verify-Runde zahlte sich bei W4 am deutlichsten aus: A CONFIRMED (inkl. unabhängiger
Reproduktion des selbstgefixten F8-Bugs — mit dokumentierter Warnung, dass
Standalone-tsconfig-Proben für distributive Conditional-Fragen unzuverlässig sind),
C ohne Verstöße, aber **B fand einen echten M2-BLOCKER**: Union-Row-Typen aus
gewöhnlichem Branching kompilierten mit konfidentem Literal-Claim und warfen zur
Laufzeit — die erzwungene Distribution des F8-Fixes leckte an Tupel-Positionen, und
Guard-Tuple-Wrap + Extract verschluckten den Error-Zweig der geforkten Fold-Union.
Fix nach Hauspolitik (IsUnion-Gate vor dem naked Destructure), 6 neue Pins (inkl.
der von B und C benannten Coverage-Lücken), Nicht-Vakuität per Mutant, von B
re-verifiziert (Typ ehrlich [2, number], Runtime wirft weiter). Lehre: „distribuiert
natürlich" ist nie eine sichere Scope-Annahme — Misch-Verdikt-Unions müssen VOR der
Destrukturierung gegated werden. Final: 195,481 @ 137 (+4,841), stress 105,758,
test:core 1,572, 22 Typ-Pins. Wunschlisten-Platz 4 geschlossen: embedMatrix ist
durch NDArray.stack ersetzt (byte-identische Rückprobe).

## Op-Scheibe W5: `item` (2026-07-21)

Fünfte und letzte Op-Scheibe der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md
W5/F3 — ein Skalar-Read aus der Score-Matrix): `NDArray.item(...indices)`, NumPys direkter
Skalar-Accessor. Spec docs/op-w5-item-spec.md Version 2 nach Baustein-0-Addendum (brainroute:deep,
Scratch-Worktree, kompilierte Form GELIEFERT UND GEMESSEN vor der Umsetzung — F1-F8):

- **F1 (BLOCKER, vorab gefangen):** `Guard<>` auf dem Rest-Parameter kollabiert zu TS2370 an der
  Deklaration — Rest-Parameter müssen array-artig bleiben. `ItemGuard<S, Idx>` folgt stattdessen
  der `SliceSpecsGuard`-Präzedenz: Tupel-geformt in jeder Verzweigung, nur einzelne Positionen
  werden zu `{__shapeError}`-Objekten retypisiert.
- **F2:** Der Fold ist S-GETRIEBEN (nicht Idx-getrieben wie `SliceSpecsGuard` — dort ist
  Under-Arity gewollt/Partial-Indexing, hier verboten/volle Indizierung).
- **F3 (Spec-Korrektur, erzwungene Mechanik):** Arity-Verstöße sind natives TS2554, nicht eine
  Custom-Message — für ein FEHLENDES Argument existiert architektonisch keine Position, an die
  eine Message gehängt werden könnte. `itemRuntime` trägt einen eigenen, runtime-only
  Arity-Stem.
- **F4 (Regression gefunden + gefixt):** Ohne `IsDynamicRank<Idx>`-Gate bricht ein Spread-Aufruf
  (`item(...arr)`) mit TS2556 — dasselbe Gate wie `SliceSpecsGuard`s `IsDynamicLength`, hier
  direkt aus dim.ts wiederverwendet (`Idx` ist strukturell ein `Shape`).
- **F5:** Dot-Form-Ablehnung ist NICHT in `LiteralIndexBounds` (dort silent-pass zu "unknown")
  — braucht `IsDotFormStep`s Export (ein `export`-Präfix, dieselbe Owner-gedeckte Edit-Klasse
  wie frühere Ein-Wort-Exports).
- **F6:** `LiteralIndexBounds`s Union-Verhalten ist konservativer als sein eigener
  Doc-Kommentar — der `IsUnion`-Pre-Gate in `ItemMark` ist doppelt begründet.
- **F7/F8:** TS7s Ein-Diagnose-pro-Call-Regel reproduziert; Stil `I extends number` statt
  `I & (string|number)`.

Umsetzung folgte der verifizierten Skizze 1:1 (kein zweiter Design-Fund während der eigenen
Implementierung). D1: VOLLE Indizierung (ein Index je Achse), Rang 0 = `item()` ohne Argumente,
kein Setter/Partial-Indexing/`at`-Alias. D3 (Runtime, `itemRuntime` APPENDED in runtime.ts):
Arity-Check, NumPy-Negativ-Normalisierung + Bounds-Check pro Achse (Stems wortgleich zu den
Typ-Stems, siehe unten), Offset-Summe über `computeStrides` — ein reiner strided Read, KEIN
Kernel (M1 v5: kernel-lose Referenz-Ops zulässig, solange die Paritätslücke in FOLLOWUPS
getrackt wird). D4: `item` als Klassenkörper-Append nach `sqrt`.

**D6-Kosten-Befund (der eigentliche Fund dieser Scheibe):** Die Erst-Umsetzung folgte der
Addendum-Skizze wörtlich — 5 separate `Expect<Equal<ItemGuard<...>, HandType>>`-Message-
Equality-Pins in ndarray.test-d.ts — und maß ein Gesamt-Delta von **+11.563**, fast das
Doppelte des ≤ +6.000-Gates. Bisektion (additive Entfernung via Backup-Kopie, keine
Mutanten-Notwendigkeit) zerlegte das: Quellcode allein +623 (nahe an der Baustein-0-Messung
+712), die restlichen +10.940 fast vollständig aus den Testdateien — davon +9.066 allein aus
`ndarray.test-d.ts`, und davon wiederum **~5.020 aus den 5 `ItemGuard`-Message-Pins**. Isolierte
Messung: EIN einzelner `Equal<ItemGuard<...>, T>`-Vergleich gegen einen strukturell ähnlichen
Handtyp kostet ≈1.700-1.750 Instantiations — eine Größenordnung über einer bloßen
`ItemGuard`-Referenz (≈80), einem Self-Compare (≈100-110) oder einem Vergleich gegen `unknown`
(≈100, da `tsc`s Assignability-Check dort die Quelle nicht voll normalisieren muss). Reaktion:
Pin-Konsolidierung — EIN kombinierter Zwei-Fehlerpositionen-Pin (`ItemGuard<[2,3], [0.5, 3]>`,
beweist Dot-Form UND Out-of-Bounds gleichzeitig, TS7s Ein-Diagnose-pro-Call-Fakt F7 ausnutzend)
statt fünf Einzel-Pins, Mixed-Rank-S über die bloße `ItemGuard`-Typebene statt eine
Klasseninstanzen-Union getestet, das `@ts-expect-error`-Real-Call-Trio auf drei statt vier
Fälle reduziert. Finale Zahlen (2× je Messpunkt, deterministisch): Haupt-Pin **201.354 @ 137**
(+5.873 zur W4-Baseline, Gate ≤ +6.000 eingehalten mit 127 Spielraum), stress **106.398 @ 82**
(+640, ausschließlich aus dem geteilten Quellcode — kein stress-eigenes File berührt), browser
2.142 @ 75 unverändert, test:core **1.588** (+16), `bench:editor` einmalig neu gesetzt
(uniform +628 über alle 7 Workloads, zwei Durchläufe grün), Hash weiterhin byte-identisch (keine
Rust-Änderung), `graph-a-lama query lint` weiterhin 0/0. Coverage-Auswirkung der Konsolidierung:
KEIN D2-Kanten-Verlust — jede Kante (Arity beide Richtungen, OOB positiv/negativ, Dot-Form,
gültiges negatives Literal, wide Rang, Union-Index, Mixed-Rank-S, dynamischer Spread) trägt
weiterhin mindestens einen Pin, nur die redundanten Mehrfach-Belege pro Kante wurden dedupliziert.

FOLLOWUPS trägt zwei neue Einträge: den D6-Kostenmechanismus selbst (offene Frage, ob
`SliceSpecsGuard` und andere bestehende Message-Pins denselben Kostenfaktor tragen — eine
Stichprobe deutete auf ≈1.049 für ein vergleichbares `SliceSpecsGuard`-Pin, günstiger als
`ItemGuard`s ≈1.700, aber noch immer weit über einem bloßen Referenzzugriff) und das
Aufsplitten von scalar-mean.test.ts (jetzt W2-W5-Sammelbecken, D6-Mandat der Spec). Mit W5 ist
die komplette Dogfooding-Wunschliste (W1-W5) abgearbeitet. Vollständiger Befund:
docs/op-w5-item-ergebnisse.md.

## Op-Scheibe W5 (item) + Wunschlisten-Abschluss (2026-07-21)

`item(...indices): number` — voller Skalar-Read mit NumPy-Negativ-Normalisierung,
Spike-03-Bounds-Reuse (LiteralIndexBounds ist hier die RICHTIGE Semantik — der
W1-Warnhinweis betraf topks andere Index-Semantik). Baustein 0 fing den fünften
Vor-Bau-Blocker der Serie (Guard<> auf Rest-Params = TS2370; slice()s Fold-Form ist
das Muster) und lieferte die verbindliche ItemGuard-Form gemessen (+712-Skizze).
Implementierung mit offengelegter Budget-Konsolidierung der Pins (Messbefund: ein
ItemGuard-Equal-Pin ≈1,700 Instantiations). Verify: A CONFIRMED (Kanten selbst
enumeriert — Konsolidierung verlustfrei), B widerlegte zwei META-Behauptungen
(IsUnion-Pre-Gate coverage-tot; F6-Prämisse falsch — LiteralIndexBounds' Union-
Disziplin trägt wie in Spike 03 dokumentiert) → Pre-Gate als reduce.ts-Policy-
Angleichung dokumentiert und per neuem Policy-Pin load-bearing gemacht (Mutant:
exakt 1 Zeile rot), C fand das M3-Wiederholungsmuster (native Diagnosen) → v6-
Kandidat in FOLLOWUPS. Final: 201,455 @ 137 (+5,974, Gate ≤ +6,000 — knappster
Lauf der Serie), stress 106,398, test:core 1,588.

**Damit ist die komplette Dogfooding-Wunschliste W1–W5 geschlossen** — jede Op
evidenzbasiert, jede mit Baustein-0-Fang vor dem Bau (5/5!), dreifach verifiziert,
zwei echte Verify-B-Blocker (W2-Diagnose-Verlust, W4-M2-Loch) in-Slice gefixt.
Nächster Schritt: 0.2.0-Bündel-Release (Owner-Publish), Example-Umstellung auf die
neuen Ops als Vorher/Nachher-Showcase, dann Scale-Probe.

## Release 0.2.0 — „the wishlist release" (2026-07-21)

numtype@0.2.0 publiziert (Owner, 2FA; ein abgelaufener npm-Login als einzige Hürde —
E404 beim Publish ist npm-Sprech für „nicht eingeloggt"). Inhalt: die komplette
W1–W5-Serie. Release-Mechanik wie designed: Version-Bump + Example-Dep-Bump im selben
Commit (fb28417) hielt den Registry-Tripwire grün; das UNVERÄNDERTE 0.1.1-Example lief
vor der Umstellung grün gegen 0.2.0 (Drop-in-Kompatibilität asserted — die erste echte
Bewährungsprobe der 0.x-SemVer-Politik). Danach die Showcase-Umstellung (dd60692):
alle fünf Workarounds durch die Ops ersetzt, JEDER gepinnte Score byte-identisch
(die W-Scheiben-Beweise, sichtbar im Konsumenten), FRICTION→RESOLVED-Kommentare
in-place + F→W-Tabelle in der Example-README, dritter @ts-expect-error-Pin (item-OOB).
Tag v0.2.0 gepusht (Ruleset-geschützt). Damit ist der Bogen geschlossen: Demo →
Friction-Log → Wunschliste → fünf verifizierte Op-Scheiben → Release → dieselbe Demo
auf den eigenen Ops. Nächster Owner-Reihenfolge-Punkt: Scale-Probe.

## Scale-Probe — „unproven at scale" fällt (2026-07-21)

Der dritte und letzte Punkt der Owner-Reihenfolge vom 2026-07-20. Ergebnis in einem Satz:
Die interaktive Latenz hält über alle 34 messbaren Sweep-Punkte (warmer Hover-Median
0,04–0,11 ms), die Kosten der Skala landen ausschließlich auf dem Kaltstart, und die einzige
harte Wand sitzt bei Rang 1024 — praktisch unerreichbar, aber real. Vollständige Zahlen und
Methodik: docs/scale-probe-ergebnisse.md.

**Der Prozess war hier wertvoller als das Ergebnis.** Vier Befunde, die ohne die vorgelagerte
Prüfung publiziert worden wären:

1. **Die Frontier-Zweitmeinung fing einen Blocker in der SPEC**, den der adversariale
   Spec-Verifier übersehen hatte: Achse (a) hätte ohne bindende Shape-Diversitäts-Vorgabe
   Cache-Treffer statt Skalierung gemessen. Gemessen liegt zwischen wiederholten und
   verschiedenen Shapes ein Faktor 19 — die bequeme Konstruktion hätte eine flache Kurve und
   den Satz „skaliert mühelos" produziert, ohne dass jemand falsch gemessen hätte. Dieselbe
   Falle steckte in Achse (b), wo die naheliegende Konstruktion „konstant" ergibt und die
   ehrliche „linear, 265 Instantiations pro Kettenglied".
2. **Beide Spec-Prüfer fanden gemeinsam einen Defekt in der bestehenden Mess-BASIS**: die
   generierten Workload-tsconfigs führten `spike/src/ambient.d.ts` nicht, weshalb ALLE sieben
   Editor-Workloads (nicht nur das absichtlich kaputte w4) mit sieben TS2591-Diagnosen liefen —
   unsichtbar, weil `enforceHardGate` das `hadTypeErrors`-Flag nirgends liest. Owner entschied
   „vorher reparieren" statt „erben": Vorab-Scheibe **V0** (c18aa7f), uniform +135 auf allen
   sieben Pins, Latenzwerte und `check:diag` unberührt, die publizierte Hover-Aussage hält auf
   der sauberen Basis.
3. **Verify-B fand ein VAKUÖSES Korrektheits-Gate** in der frischen Implementierung: Weil `tsc`
   lange Tupel in der Hover-Anzeige kürzt, war der Vergleich auf sechs Dimensionen verkürzt
   worden — bedingungslos für JEDEN Rang, auch die ungekürzten. Der Verifier extrahierte die
   echte Prüffunktion und bewies mechanisch, dass eine ab Position 6 durchgehend falsche
   Anzeige durchrutscht. Behoben durch eine elisions-bewusste Prüfung (Präfix + Suffix +
   Rekonstruktion der Gesamtlänge aus dem „N more"-Vermerk, Kürzungsfenster aus dem Text gelesen
   statt hartkodiert); acht Mutationen belegen die Wirksamkeit, zwei Kontrollläufe die
   Symmetrie.
4. **Zwei Verifier widerlegten unabhängig die Charakterisierung des Implementierers**, die
   Datei-Achse wachse „deutlich überproportional". Die Marginalkosten je Datei sind flach bis
   leicht fallend (4.868 → 3.798) — linear mit fester Grundlast. Echt überproportional ist nur
   die Rang-Achse (144 → 2.608 je Rang). Die falsche Beschreibung war nie committet, wäre aber
   in die publizierte Aussage gewandert.

Dazu ein Angriff, den Verify-B sich selbst stellte und ausräumte: In der distinct-Konstruktion
wachsen mit dem Dateiindex auch die Zahlenwerte, die Kurve könnte also Magnituden statt Vielfalt
messen. Eine Gegen-Konstruktion mit fest begrenzten Größen ergab 3.812/3.711 je Datei gegen
3.798–3.961 — der Confound existiert, verfälscht aber nichts.

**Owner-Entscheidungen dieser Scheibe:** alle vier Achsen (statt einer Teilmenge) · Sweep
on-demand plus EIN gepinnter Sentinel (statt gar keiner oder voller CI) · synthetisch mit
rag-demo-Eichung · Claim-Scope = Konsumenten-Skala mit ausdrücklich offener API-Flächen-Frage
(statt einer Extrapolation aus den W-Serien-Ripple-Zahlen) · ambient-Fix als Vorab-Scheibe ·
Z2-Abweichung durch Vertragspräzisierung in v6 auflösen statt dauerhaft dulden.

Covenant: keine Verletzung; zwei Textlücken (Z2 on-demand-Korpora, M2 Rang-Cliff als
Falsch-Ablehnungs-Grenze) als v6-Kandidaten dokumentiert. Das v6-Bündel steht damit bei vier
und ist reif für eine eigene kleine Vertrags-Scheibe.

## topk-Selektion, Phase 1 — die Messung (2026-07-22)

Anlass: ein beim WNDArray-Paritäts-Gespräch aufgefallener algorithmischer Fehler —
`topkRuntime` sortiert das GESAMTE Array, um die k größten zu finden. Ergebnis der Messung:
**reiner Heap**, mechanisch aus der Regel berechnet, beide Läufe einig, null duale
Verletzungen im 92-Zellen-Raster, 57 Gewinn-Zellen. Bei einer Million Elementen und k=1 fällt
der Aufruf von 280 ms auf 3,8 ms — Faktor 74. Vollständige Zahlen und Methodik:
docs/op-topk-selection-ergebnisse.md. **Die Umsetzung ist bewusst NICHT Teil dieser Phase**
und steht als eigene Scheibe mit eigener Verify-Runde aus.

**Der eigentliche Ertrag liegt woanders — in zwei Befunden über das eigene Vorgehen.**

**Erstens: Die Vorab-Sondage, die diese ganze Scheibe ausgelöst hat, war um mehr als eine
Größenordnung daneben.** Sie hatte bei `k` nahe `n` einen Faktor 0,60 nahegelegt (Heap ca. 67 %
langsamer) und damit die gesamte Hybrid-/Schwellen-Debatte begründet. Die disziplinierte
Messung sagt 5,0 %. Bei `k = n/2` kippt sogar das Vorzeichen: Sondage „6 % langsamer",
gemessen „24 % schneller". Das alte Sondage-Skript wurde live auf derselben Maschine erneut
ausgeführt und reproduziert seine eigenen Zahlen exakt — es lag also nicht an der Umgebung,
sondern an der Methode: zwei Aufwärm-Aufrufe statt adaptivem Warmup, 20.000
Korrektheits-Fuzz-Fälle unmittelbar davor im selben Prozess (JIT-Kontamination), gewöhnliche
JS-Arrays statt typisierter. Derselbe Mechanismus wie in Kern 06. Die unbequeme Lehre: Eine
schnelle Sondage kann eine umfangreiche, sorgfältige Folgearbeit auf eine Zahl gründen, die
einer sauberen Messung nicht standhält — und je aufwendiger die Folgearbeit wird, desto
weniger denkt jemand an die Ausgangszahl zurück.

**Zweitens: Die vorregistrierte Entscheidungsregel wurde VIERMAL gebrochen, bevor sie messen
durfte** — jedes Mal von einem unabhängigen Fresh-Context-Verifier, jedes Mal an einer anderen
Stelle, und zwei der gebrochenen Fassungen stammten vom Orchestrator selbst. v2: Die
Zulässigkeitsprüfung ignorierte strukturell genau die Größen, an denen der Gewinn erwartet
wurde; eine Klausel widersprach sich im selben Satz; ein undefinierter Fall ließ die Regel
abbrechen. v3: kein Verdikt in 6,5 % der gefuzzten Raster, zwei Verdikte für dieselben Zahlen
in 43,6 %, und „reiner Heap" trotz null gemessenem Gewinn. v4: der „hohle Hybrid" — Gewinn und
Schwellenbildung entkoppelt, in 29,4 % der Hybrid-Verdikte eine Netto-Regression gegenüber
Nichtstun. v5: zwei mandatierte Messläufe, keiner als verdikt-tragend benannt (39,6 %
divergierende Lauf-Paare), plus rein relative Schwellen, die eine Sub-Mikrosekunden-Zelle über
50-fach-Gewinne entscheiden ließen.

Jeder dieser Fehler hätte nach der Messung ein mechanisch berechnetes, eindeutig aussehendes
Verdikt geliefert — also als „die Zahlen sagen es doch" durchgehen können. **Die Regel selbst
hätte nie signalisiert, dass sie falsch ist.** Gefunden wurden sie ausnahmslos dadurch, dass
Verifier die Regel als Skript nachbauten und gegen tausende synthetische Raster laufen ließen,
statt sie zu lesen. Eine gelesene Regel wirkt eindeutig; eine durchgespielte verrät ihre
Lücken.

Owner-Entscheidungen dieser Scheibe: erst messen statt Algorithmus vorab wählen · duales
Kriterium (relativ UND absolut) für beide Gates, nachdem die Zweitmeinung die Rausch-Fragilität
belegte · Gate gilt für den dekomponierten Pin-Anteil, nicht für die Gesamtverschiebung ·
Schnitt nach der Messung, Umsetzung als eigene Sitzung.

Nebenbefund mit Reichweite über die Scheibe hinaus: Eine einzige LEERE Datei verschob den
Root-Instantiation-Zähler um **+6.611** — das in CLAUDE.md dokumentierte Order-Noise-Band von
„±≈2.000" ist damit zweifach reproduziert widerlegt und auf „±≈7.000" korrigiert. Das betrifft
jede künftige dateihinzufügende Scheibe.

## WASM-Parität S1 — Skalar-Overloads `add`/`sub`/`mul`/`div` auf `WNDArray` (2026-07-23)

Zweite Scheibe der WASM-Parität-Serie (nach S0/sqrt): die vier Skalar-Overloads, die W2 der
naiven `NDArray`-Klasse gegeben hat (`x.div(2)`, shape-erhaltend, kein `[1]`-Broadcast-Umweg),
bekommen jetzt ihr resident-WASM-Gegenstück. Vollständigkeits-/Symmetriearbeit, kein gemessener
Nutzerbedarf — genau wie S0.

**Umsetzung folgt der etablierten Pipeline strukturgleich zu S0:** neues Kernel-File
(`crates/core/src/kernels/scalar.rs`) mit vier `pub fn scalar_{add,sub,mul,div}_strided`, jede
ein Einzeiler über den S0-`unary_strided`-Kern — kein neuer Iterationskern, reine Wiederver-
wendung. Das erforderte, `unary_strided` von `fn` auf `pub(crate) fn` zu erweitern: eine reine
Sichtbarkeitserweiterung, deren Codegen-Neutralität ein Clean-Rebuild-Hash-Vergleich vor dem Bau
bestätigte. ABI: vier `nt_scalar_*_strided`-Einträge (9-Parameter-Form, `scalar: f64` zwischen
Daten- und Output-Block, `nt_fill`s eigener `value: f64`-Parameter als Präzedenz) strikt ans
Ende von `abi.rs` angehängt. `CoreExports` bekam einen vierten Merge-Block in `loader.ts`;
`ThreadedCoreExports` erbte automatisch, ohne eine einzige `threaded.ts`-Zeile zu berühren — der
S0/D10-Omit-Fix (direkter Cast statt `Omit<ThreadedCoreExports,"memory">`) trug hier zum ersten
Mal über eine ganze Vier-Member-Scheibe hinweg und bestätigte sein eigenes Versprechen
(„+0 statt +7 pro Member") empirisch am echten check:diag.

**Die WNDArray-Seite:** die vier Bestandsmethoden `add`/`sub`/`mul`/`div` wurden zu Overload-Sets
umgebaut — Skalar-Overload ZUERST deklariert, generischer `Guard`-Träger ZULETZT (die W2-F1-
Lektion, hier proaktiv statt reaktiv angewandt), der komplette Array-Array-Körper jeder Methode
BYTE-IDENTISCH in den else-Zweig der neuen Implementierungssignatur verschoben — am `git diff`
bewiesen, die verschobenen Zeilen erscheinen als reine Kontextzeilen. Ein neuer privater
`scalarOp`-Helfer (Klassenkörper-Insertion, kein Bestandsmember editiert außer den vier
erzwungenen Overload-Umbauten) marshalt einmal statt vierfach dupliziert.

**M1 als Korollar, nicht als neuer Claim:** anders als `sqrt`, das eine 30k-Fälle-Vorab-Probe
brauchte, weil `f64::sqrt`-Bit-Parität ein neuer empirischer Claim war, ist Skalar-Bit-Parität
ein Korollar der längst eingefrorenen binären Kernel (Kern 07) — `x op s` ist derselbe IEEE-Op
mit konstantem zweiten Operanden. Trotzdem dreifach belegt: Baustein 0 verifizierte das
Korollar-Argument selbst und fuhr zusätzlich einen 36.324-Fälle-Differential gegen den echten
Kernel (0 Abweichungen); der committete M1-Test deckt contiguous/View/Offset-Fenster/rank-0/
size-0/Spezialwert-Raster/kuratierte `div(0)`-`div(-0)`-`sub`-Ordnungs-Fixtures je Op sowie eine
`[1]`-Broadcast-Äquivalenzprobe über 100 Fälle; der Pflicht-Mutant bewies, dass der Katalog eine
echte Regression fängt.

**Zwei Backup-Kopie-Beweise statt `git checkout`:** die Kernel-Mutation (`|x| x + s` →
`|x| x - s` im add-Kernel) kippte 5 benannte cargo-Tests und 30 benannte JS-Differential-Fälle
— Revert per `cp` aus einer vorab angelegten Backup-Kopie, SHA-256-Beweis vor/nach identisch.
Zusätzlich wurde der T4b-Diagnose-Qualitätstest (der reale-tsc-Test, der beweist, dass die
Broadcast-Shape-Message durch das Overload-Set überlebt) selbst per Reihenfolgen-Flip auf
Nicht-Vakuität geprüft: mit vertauschter Deklarationsreihenfolge kollabierte die Meldung exakt
wie bei der W2-F1-Regression vorhergesagt — vom benannten shape-Text zum Skalar-Decoy „not
assignable to type 'number'". Auch dieser Flip wurde per Backup-Kopie revertiert, nicht per
`git checkout`.

**Zahlen:** check:diag Root von 206.850 auf 208.015 @ 140 (Δ+1.165, klar unter dem +6.000-Gate),
dreistufig dekomponiert — die vier `CoreExports`-Member kosten wie erwartet **0**, der
WNDArray-Klassen-Surface-Umbau **+730**, die Test-/Typ-Pin-Anhänge **+435**. stress +721, browser
unverändert. `bench:editor`s acht Pins bewegten sich uniform um +721 (WNDArray wird in jedem
Workload instanziiert) — zweimal gemessen, neu gesetzt. Neuer Freeze-Hash `8255821b…`, ersetzt
den S0-Pin `24a048c7…`. Test-Wachstum: cargo 169→184, test:resident 4471→4719 (248 neue Fälle),
test:threaded 75→91.

Details: docs/wasm-parity-scalar-spec.md v2, docs/wasm-parity-scalar-ergebnisse.md. Die
Verify-Runde A+B+C dieser Scheibe steht zum Zeitpunkt dieses Eintrags noch aus.

## WASM-Parität S2 — `mean` auf `WNDArray` (2026-07-23)

Dritte Scheibe der Kampagne, und die erste, deren gesamte tragende Eigenschaft der Verzicht auf
neuen Rust-Code ist: `mean` ist per Definition „Summe geteilt durch die Elementzahl", und beide
Bausteine existierten bereits als bit-identisch bewiesene Kernel — der v1-`sum`-Kernel und der
gerade in S1 gebaute `scalar_div`-Kernel. `WNDArray.mean(axis?, keepdims?)` ist damit eine reine
TS-Klassenkörper-Insertion: `this.sum(axis, keepdims).div(n)`, `n` wortgleich `meanRuntime`s
eigener Formel (Input-Shape, nicht reduzierte Output-Shape — mit keepdims bleibt der Divisor die
Original-Achsengröße), das Zwischenergebnis `summed` in einem `finally` disponiert, nachdem
`.div(n)` sein eigenes, unabhängiges Ergebnis bereits produziert hat. Kein neuer Kernel, kein ABI-
Eintrag, kein `CoreExports`-Member, kein Freeze-Re-Pin — der Freeze-Hash bleibt wortwörtlich
derselbe wie nach S1.

**M1 als doppeltes Korollar:** anders als S0 (neuer empirischer Claim, 30k-Fälle-Vorab-Probe
nötig) und anders als S1 (Korollar EINES eingefrorenen binären Kernels) ist `mean`s Bit-Identität
ein Korollar ZWEIER bereits bewiesener Kernel gleichzeitig — `sum` UND `scalar_div`. Weil
`scalar_div` exakt `x/n` rechnet, nie `x*(1/n)`, fällt die W2-Determinismus-Entscheidung
(„sum/n, nicht sum*(1/n)") aus der Komposition heraus, ohne einen eigenen Beweis zu brauchen.
Trotzdem direkt getestet, nicht nur behauptet: 320 Bit-Identitäts-Assertionen (250 M1-
Differential-Fälle über resident.test.ts inkl. der F1-keepdims-Methodik, 60 randomisierte
Spezialwert-Fälle, 10 threaded-vs-stable-Paritäts-Fälle), 0 Abweichungen.

**Der F1-Befund aus Baustein 0 der Spec verdient eine eigene Erwähnung:** `meanRuntime` hat
keinen `keepdims`-Parameter und gibt IMMER die reduzierte Shape zurück. Ein naiver
`assertShapeEqual`-Vergleich gegen `meanRuntime(...).shape` scheitert deshalb bei JEDEM
keepdims=true-Fall — in der Baustein-0-Probe waren das 523 von 1.746 Fällen. Der Fix ist einfach
(Daten gegen `meanRuntime.data` — keepdims-invariant, eine size-1-Achse ändert die Elementzahl
nicht; Shape gegen `keepdims ? keepDimsShape(...) : meanRuntime.shape`), aber die Lektion trägt
über die Scheibe hinaus: ein Vergleichs-Helper, der für den EINEN Aufrufer (`sum` ohne keepdims)
richtig war, wird beim zweiten Aufrufer (`mean` MIT keepdims) leise falsch, wenn niemand die
Methodik neu prüft. Dieselbe Klasse Fehler wie ein Test, der die zu prüfende Verdrahtung gar
nicht durchläuft — hier eine Test-METHODIK, die für den neuen Anwendungsfall strukturell blind
war, bis Baustein 0 sie vorab gegenrechnete.

**Backup-Kopie statt `git checkout`:** der Pflicht-Mutant (`.div(n)` → `.mul(1/n)`, D5-
Determinismus-Kandidat) kippte 71 benannte Testfälle — 27/120 `mean_all`, 39/120 `mean_axis`,
3/60 `mean special`, plus BEIDE dedizierten Determinismus-Pins namentlich. Dass nicht alle 320
`mean`-Assertionen kippen, ist erwartet (dieselbe „nicht jedes Beispiel diskriminiert"-Lektion
wie bei W2: für viele zufällige f64-Paare gilt `sum/n == sum*(1/n)` zufällig exakt bitweise) —
die zwei dedizierten, nicht-vakuösen Determinismus-Pins sind genau für diesen Zweck gebaut und
fingen ihn zuverlässig. Revert per `cp` aus einer vorab angelegten Backup-Kopie, SHA-256-Beweis
vor/nach identisch, `test:resident` danach wieder vollständig grün (5022/5024, 2 skip).

**Leak-Non-Vakuität exakt, nicht nur als Plateau:** ein neuer Test in resident-lifecycle.test.ts
ruft `mean(1)` 500-mal auf EINEM persistenten Empfänger auf (dessen eigener Lebenszyklus bewusst
AUSSERHALB des gemessenen Fensters bleibt) und misst `getResidentFreeCount()` als exakte Delta —
erwartet und gemessen: exakt `2N = 1000` (der Zwischen-`summed`-Puffer, freigegeben in `mean`s
eigenem `finally`, plus der finale `.div(n)`-Ergebnis-Puffer, freigegeben vom Test selbst). Eine
zusätzliche `byteLength`-Plateau-Kontrolle bestätigt dasselbe unabhängig.

**Zahlen:** check:diag Root von 208.015 auf 209.515 @ 140 (Δ+1.500, klar unter dem +6.000-Gate,
kleiner als S1s +1.165 wie erwartet), dreistufig dekomponiert — die `mean`-Methode selbst (dritte
Call-Site der `ReduceAxis`-Maschinerie) **+333**, die vier Test-Anhänge **+885**, die Typ-Pins
**+282**. stress +323 (dieselbe Klassen-Surface-Ripple, kleinerer Absolutwert), browser
unverändert. `bench:editor`s acht Pins bewegten sich uniform um +323 (zweimal gemessen,
byte-identisch) — neu gesetzt. Freeze-Hash bestätigt UNVERÄNDERT `8255821b…` (Clean-Rebuild
reproduziert ihn exakt, `git status crates/` leer, `cargo test` unverändert 184+1=185).
Test-Wachstum: test:resident 4719→5024 (305 neue Fälle), test:threaded 91→101.

Details: docs/wasm-parity-mean-spec.md v2, docs/wasm-parity-mean-ergebnisse.md. Die Verify-Runde
A+B+C dieser Scheibe steht zum Zeitpunkt dieses Eintrags noch aus.

## WASM-Parität S3 — `item` + `stack` (2026-07-24, dreifach verifiziert A+B+C)

Vierte Scheibe der Kampagne. `NDArray.item(...indices)` (W5) und `NDArray.stack(rows)` (W4) waren
NDArray-only; `WNDArray` kann jetzt beides, threaded-Parität fällt automatisch mit.

**Arbeitsregel 11 griff zum zweiten Mal in Folge, in zwei verschiedenen Ausprägungen.** `item` ist
kernel-los per Design — ein Einzelwert-Lesezugriff hat keine Arithmetik, die ein Kernel beschleunigen
könnte; Offset aus `(strides, offset)`, dann ein `f64`-Lesezugriff über eine frisch abgeleitete View,
denselben Pfad benutzen `toArray()`s contiguous-Zweig und die Skalar-Rückgaben von `dot()`/`norm()`
seit Kern 02/07. `stack` ist reine Datenbewegung über den seit Kern 03 eingefrorenen
`nt_materialize`: N Aufrufe, jeder in den `i`-ten Sub-Slot EINES frisch allozierten Ausgabepuffers,
zwei Scratch-Puffer insgesamt statt `2n`. Folge: **Freeze-Hash unverändert**, kein Rust, kein
ABI-Eintrag, kein `CoreExports`-Member — Arbeitsregel 10 greift bestätigt nicht.

**Baustein 0 fing zwei Verdrahtungs-Befunde vor der ersten Codezeile** und bestätigte das Kern-Design
empirisch (Sub-Slot-`nt_materialize` über sechs Konfigurationen byte-exakt, `d === 0` ohne Gate, die
zwei neuen Validierungs-Helfer über je 20.000 Zufallsfälle wortgleich zu den Orakeln, `item` auf Views
über 15.000 Fälle abweichungsfrei). BLOCKER: `ThreadedBackend` hat kein `core`-Feld, der Core lebt auf
`this.pool.core` — live als TS2339 reproduziert, exakt die Item-10-Fundklasse. MAJOR: die
Facaden-Signaturen brauchten eine Typ-Verdrahtung, die die Änderungstabelle nicht nannte.

Der wertvollste Baustein-0-Befund war eine **Bestätigung mit Zähnen**: `nt_materialize`s interne
`Vec`-Allokation löst `memory.grow` wirklich aus (1,1 MB auf 130 MB über 40 Aufrufe), ein Wachstum
detacht den alten `ArrayBuffer`, und ein Schreibversuch über die veraltete View ist ein stiller No-Op
— der Kernel meldet danach `status === 0` bei komplett falschen Daten. Weder Statuscode noch
Exception. Daraus wurde Pflicht-Mutant M-d plus die Auflage, dass ein Testfall das Wachstum
nachweislich auslöst; das 60-Fälle-Zufallsraster fängt ihn nicht.

**Die Verify-Runde schloss fünf Lücken, drei davon vakuöse Tests — alle drei nur per Mutant
sichtbar.** Baustein A's eigener Mutant entfernte die beiden `scratch.push`-Aufrufe (echtes
8-Byte-pro-Aufruf-Leck): der eigens dafür geschriebene Leck-Test blieb grün, weil 500 Iterationen
ca. 4 KB ergeben und WASM in 64-KiB-Seiten wächst — ersetzt durch eine exakte Alloc/Free-Bilanz über
den Zähl-Mock. Baustein B's Mutanten zeigten, dass die Liveness-Prüfung nur für Zeile 0 und die
Core-Prüfung nur gegen `rows[0]` getestet waren; beide Lücken mit je einem Test geschlossen, je unter
dem Original-Mutanten fallend. B's unabhängige Orakel fanden über 6.000 `item`-, 600 `stack`- und 837
threaded-Fälle **0 Abweichungen**, und ein selbst gebauter `memory.grow`-Stresstest (30,2 MB auf
49,9 MB während des Aufrufs) blieb über 2,4 Millionen Werte bit-identisch — gegen den Stale-View-
Mutanten dagegen 1.199.980 Abweichungen ohne einen einzigen Throw.

**Der teuerste Befund war eine M3-Verletzung, die drei Leser übersahen.** Der v2-Rückgabetyp-Alias
`StackResultOf<Rows>` ließ `backend.stack([a, b])` — die einzige für Paketkonsumenten erreichbare
Fläche — als `StackResultOf<readonly [WNDArray<[3]>, WNDArray<[3]>]>` hovern statt als sauberes
Tupel. Ein Top-Level-Typ-Alias in RÜCKGABE-Position wird von der Quick Info namentlich erhalten; ein
Alias in TYP-ARGUMENT-Position wird aufgelöst (`add`s `WNDArray<OkShape<Broadcast<S, B>>>` hovert
sauber). Implementierer, Baustein A (alle Gates frisch) und Baustein C (erster Lauf) lasen denselben
Quelltext und beanstandeten nichts; gefunden hat es nur der Verifier, der den echten
`tsc --lsp --stdio` startete. C hat seinen Fehlschluss im zweiten Lauf unaufgefordert benannt und den
Zwischenstand ausdrücklich als echten M3-Verstoß eingestuft — der Fix war Pflicht, nicht Kür. Daraus
wurde Arbeitsregel 13.

Der Fix (Alias trägt nur die SHAPE, das Handle wird ausgeschrieben) kostet +5.498 Instantiations.
Drei billigere Wege wurden gemessen und verworfen: Typ-Pin-Konsolidierung trägt −21 bei, eine
Zwei-Alias-Aufteilung ist teurer (227.020), und die Test-Aufrufstellen laufen bereits über dynamische
Shapes. Die Abwägung ging mit den Zahlen an den Owner: Hover-Konformität schlägt Budget, das
Scheiben-Gate wurde begründet von +8.000 auf +13.000 angehoben (Spec v3). Endstand check:diag
226.690 @ 140 (Δ+12.986, Marge 14), stress 115.498 @ 82, browser 2.142 @ 75 unverändert,
bench:editor in dieser Scheibe zweimal neu gesetzt, test:resident 5497+2, test:threaded 114, cargo
und Freeze-Hash unverändert.

Details: docs/wasm-parity-item-stack-spec.md v3, docs/wasm-parity-item-stack-ergebnisse.md.

---

## WASM-Parität S4: `argmax` (2026-07-24)

Die vierte Scheibe der Kampagne — und die erste seit S1, die einen **echten neuen Kernel**
hinzufügt. S2 (`mean`) und S3 (`item`/`stack`) waren kernel-los; die Pflicht-Vorabfrage aus
Arbeitsregel 11 lautet daher jedes Mal aufs Neue, ob die Op eine Komposition bereits verifizierter
Kernel ist. Für `argmax` ist sie es nicht: der Bestand enthält keine Vergleichs-, Maximum- oder
Index-produzierende Operation. Die Spec schrieb die kernel-lose TS-Alternative (das S3-`item`-Muster,
eine Schleife über `core.memory.buffer`) ausdrücklich als angreifbare Alternative mit auf — Baustein 0
hat sie geprüft und nicht gekippt: `item` war ein O(1)-Skalar-Read, `argmax` ist eine O(N)-Reduktion
wie `dot` und `norm_sq`, die beide einen Kernel haben. Damit bindet M1 neu und der Freeze-Hash
bewegt sich zum ersten Mal seit S1.

**Baustein 0 fand zwei Dinge vor der ersten Codezeile, und das teurere war eine falsche HAUSREGEL.**
Arbeitsregel 10 verlangt für jeden neuen `CoreExports`-Member einen `notImplemented`-Stub im
hand-getippten Mock und benannte `backend-oom.test.ts` als „das EINZIGE strukturell getippte
`CoreExports`-Literal im Repo". Das war falsch — `resident-lifecycle.test.ts:601` ist ein zweites,
mit eigenem lokalem Helfer. Baustein 0 hat es nicht erschlossen, sondern reproduziert: zwei
Dummy-Member ins Interface, und beide Dateien werfen TS2739 bei Exit 1, während `check:diag`
weiterhin eine plausible `Instantiations`-Zeile druckt. Das ist die Arbeitsregel-6-Falle in
Reinform — ein Gate, das die Gesundheit seines eigenen Messlaufs nicht mitprüft. Der zweite Befund
war eine unentschiedene Frage, die das Repo seit zwei Scheiben mit sich herumtrug: für View- und
keepdims-Tests existierten **zwei unvereinbare Präzedenzfälle** (S2/`mean` benutzt `keepDimsShape`
als Shape-Orakel; `NDArray.argmax`s eigener Test lehnt genau das als zirkulär ab). Beide haben
recht, für verschiedene Fragen — aufgelöst als dreiteilige bindende Regel: Daten gegen
`argmaxRuntime` mit vorab gelesenem `view.toArray()`, keepdims-Shape über strukturelle Invarianten,
plus ein orakelfreier Cross-Surface-Shape-Pin.

Baustein 0 korrigierte außerdem die Gate-Begründung. Die Spec v1 hatte ≤+6.000 vorregistriert und
sich dabei auf S2s *erstgemessene* +1.500 gestützt — S2s **realisierte** Gesamtkosten waren aber
+5.689, weil der View-Coverage-Nachtrag allein +4.189 kostete. S4 fordert diese View-Fälle von
Anfang an und hat zusätzlich eine Rust/ABI-Schicht. Das Gate wurde daraufhin **vor jeder Messung**
auf ≤+8.000 angehoben, mit offengelegter Herleitung und unveränderter STOPP-Klausel — ausdrücklich
etwas anderes als S3, wo ein gerissenes Gate nachträglich angehoben werden musste. Realisiert
wurden +3.138; die alte Registrierung hätte also auch gehalten, was die v1-Begründung nicht
weniger falsch macht.

**Der tragende Semantik-Punkt der Op** ist, dass `nt_argmax_all_strided` den Index in die *logische*
row-major-Abflachung der View liefern muss, nie den berechneten Speicher-Offset — auf transponierten
oder geslicten Views fallen beide auseinander. Das ist dieselbe Falle, die `sum_all_strided`s
Doc-Kommentar seit Kern 03 für die Akkumulations*reihenfolge* beschreibt, hier in der Index-Domäne.
Baustein A hat sie zur wertvollsten Einzelmessung der Runde gemacht: statt die Totalordnung
anzugreifen (die der Implementierer schon mutiert hatte), setzte A `max_idx` auf den Speicher-Offset.
Es fielen 3 cargo-Tests — darunter exakt die dafür geschriebene Nicht-Vakuitäts-Assertion — und
**40 von 1607 JS-Tests, ausschließlich die vier View-Klassen**. Kein einziger contiguous-Fall fiel,
was mathematisch erwartbar ist (bei natürlichen Strides und Offset 0 gilt `off == flat`). Damit ist
Arbeitsregel 12 nicht mehr nur begründet, sondern gemessen.

**Verify-B fand eine reale Lücke, die alle Gates grün gelassen hatte.** `WNDArray.argmax` gibt bei
`status != 0` den frisch allozierten Ausgabepuffer frei, bevor es wirft — eine Garantie, die der
Doc-Kommentar des Moduls ausdrücklich behauptet. B entfernte den `freeBuf`-Aufruf und maß: 0 von
334 Op-Tests, 0 von 7 Lifecycle-Tests, 0 von 1 threaded-Test fielen. Mit einem eigenen zählenden
Core-Wrapper, der Status 1 erzwingt, wies B ein reales 16-Byte-Leck nach. Kein committeter Test
erzwang je einen Kernel-Fehlschlag für diese Op, obwohl S3 für `stack` genau so einen Test hat.
Geschlossen wurde die Lücke mit zwei Tests, je einem pro Einstiegspunkt, und dem entscheidenden
Beweis-Detail: **jeder Mutant fällte genau einen Test, der jeweils andere Zweig blieb grün** — ein
schlampiger Test hätte beide oder keinen gefällt.

Der zweite B-Befund war unangenehmer, weil er eine bereits gelernte Lektion war. Der Leck-Test
behauptete in seinem Kommentar, 500 Aufrufe ohne Speicherwachstum seien eine „unabhängige
Bestätigung", weil ein geleaktes 8-Byte-Ergebnis irgendwann eine Seitengrenze reißen würde. Die
Rechnung: 500 × 8 = 4.000 Byte gegen 64-KiB-Seiten, es bräuchte 8.192 Iterationen. Die Assertion
war für genau das Leck, das sie adressierte, arithmetisch unfähig — dieselbe Fehlerklasse, die S3
bereits gelernt und in der Wissensbasis abgelegt hatte, reproduziert ausgerechnet in dem Test, der
sie verhindern sollte. Der KB-Capture bestätigte das unabhängig: zwei der drei Lektionen dieser
Scheibe waren exakte Wiederholungen bestehender Notizen, mit identischen Zahlen, und wurden
revidiert statt dupliziert.

**Zur Tie-Blindheit verschärfte B die Diagnose des Implementierers.** Dessen Pflicht-Mutant (`>` →
`>=` in der Totalordnung) war von 0 der 240 randomisierten Differentialfälle gefangen worden, mit
der Erklärung „Zufalls-Floats erzeugen praktisch nie Gleichstände". B präzisierte: die Daten stammen
aus einer *stetigen* Verteilung, ein Gleichstand ist dort ein **Maß-Null-Ereignis** — die 0/240 sind
eine mathematische Gewissheit der Testkonstruktion, kein Pech. Und das Spezialwert-Raster fängt es
nur *zufällig* (5/60), weil sein Generator mit 35 % pro Element aus einem kleinen diskreten Pool
zieht; ob eine Kollision entsteht, hängt am Seed statt an einer Konstruktion. Verlässlich fangen es
nur die absichtlich konstruierten Rust-Unit-Tests. Die Lücke sitzt damit genau in der
Cross-Language-Schicht, in der M1s Vertrag lebt.

**Arbeitsregel 13 wurde diesmal vierfach erfüllt statt einmal.** S3 hatte gelehrt, dass eine
Hover-Norm gemessen und nicht gelesen werden muss — dort übersahen drei Leser des Quelltexts eine
M3-Verletzung. In dieser Scheibe fuhren der Implementierer, Baustein A, Baustein B und Baustein C je
unabhängig eine echte `tsc --lsp --stdio`-Messung mit `sum` als Kontrollpunkt. Alle vier: saubere
aufgelöste Tupel, keine Alias-Leckage. Die S3-Fehlerklasse ist hier by construction abwesend, weil
D2 den Rückgabetyp ausschreibt statt einen Top-Level-Alias in Rückgabeposition einzuführen.

Baustein C fand keine Vertragsverstöße und keine neue Auslegungsfrage, präzisierte aber einen
bestehenden v6-Kandidaten in Richtung S5: der NaN-Payload-Vorbehalt an M1 kann für `argmax`
strukturell nicht greifen, weil die Ausgabe stets ein ganzzahliger Index ist und nie ein kopierter
Datenwert. Bei `topk` wird er relevant, weil dort `values[i] === data[indices[i]]` byte-exakt gilt.

Ehrlich offengelegt bleibt eine benennbare Einschränkung der M1-Behauptung: eine nicht-ganzzahlige
Achse (nur über `as unknown as number` erreichbar) divergiert zwischen den Flächen, weil JS den
Float bis in `Array.prototype.slice` mitschleppt, während die WASM-ABI `ToInt32` vor Rusts
Negativ-Achsen-Arithmetik anwendet. B hat live reproduziert, dass derselbe Mechanismus für
`WNDArray.sum` seit S1 gilt und die Wurzel als FOLLOWUPS-Item seit W1 offen ist — die Scheibe erbt
die Lücke korrekt, statt sie zu erzeugen, aber die Behauptung ist an dieser Eingabe nicht wörtlich
unqualifiziert wahr.

Endstand: Freeze-Hash `8255821b…` → `eba6ba7ac85d15a814fd027392a81c7450d885d2048d7efd6694b7e8370988bb`
(dreiteilig bewiesen, von A und B je unabhängig in beide Richtungen nachgestellt), check:diag
229.828 @ 140 (Δ+3.138, null Order-Noise), stress 115.934 @ 82 (Δ+436), browser 2.142 @ 75
unverändert, bench:editor uniform +436 — diesmal perfekt uniform, weil kein Workload eine
`argmax`-Aufrufstelle hat. test:resident 5866+2, test:threaded 127, cargo 204+1.

Details: docs/wasm-parity-argmax-spec.md v3, docs/wasm-parity-argmax-ergebnisse.md.

---

## WASM-Parität S5: `topk` — und der Abschluss der Kampagne (2026-07-25)

Die fünfte und letzte Scheibe. Sie hat den Kernel `nt_topk_strided` gebracht, aber ihr
eigentlicher Ertrag war eine Korrektur an dem, was das Projekt über sie zu wissen glaubte.

**Zwei Sätze in FOLLOWUPS und CLAUDE.md waren falsch.** Sie standen dort seit der
topk-Selektions-Scheibe: ein künftiger `nt_topk`-Kernel „sollte diesen Heap-Algorithmus
spiegeln", und S5 sei „die härteste M1 der Kampagne". Beim Lesen von `topkRuntime` vor dem
Spec-Schreiben zeigte sich, dass die Funktion das Gegenargument längst im eigenen
Doc-Kommentar trug: `topkCompareValues` plus der Indextiebreak `|| (idxA - idxB)` ist eine
strikte Totalordnung auf den paarweise verschiedenen Indizes `0..n-1`. Es gibt daher genau
eine korrekte top-`k`-Menge in genau einer Reihenfolge — Heap und Vollsortierung müssen
übereinstimmen. Damit hängt Bit-Identität nicht am Algorithmus, sondern nur an der exakten
Transliteration des Ordnungs-Prädikats und daran, dass `values[i] = data[indices[i]]` ein
reiner Element-Kopiervorgang bleibt. Bei `topk` findet überhaupt keine
Gleitkomma-Arithmetik statt — kategorial anders als bei `sum`, wo die Akkumulationsreihenfolge
die Bits ändert und der Kernel die Schleifenordnung spiegeln muss.

Weil dieses Argument die gesamte M1-Begründung trug, bekam Baustein 0 den ausdrücklichen
Auftrag, es zu brechen. Es hielt: der Verifier leitete die Ordnung unabhängig neu her
(Trichotomie für jedes Paar, Transitivität aus „totale Präordnung + strikter Tiebreak",
IEEE-754-Übereinstimmung der Vergleichsoperatoren zwischen JS und Rust/WASM). Baustein B
bestätigte es später empirisch und fügte ein nützliches Korollar hinzu: an jeder
Heap-Vergleichsstelle sind die verglichenen Indizes strukturell verschieden, also kann der
Komparator dort nie exakt 0 liefern — womit `<`-vs-`<=`-Mutationen an genau diesen Stellen
beweisbar äquivalente Mutanten sind und keine Testlücke anzeigen. Das erspart künftigen
Verifiern die Jagd auf einen nicht existierenden Bug. Die praktische Konsequenz stand als
bindende Festlegung in der Spec: die Tests dürfen sich nicht auf Heap-Interna festlegen,
sonst pinnen sie eine Implementierung statt eines Vertrags.

**Der echte M1-Risikopunkt lag woanders, und die Vorgängerscheibe hatte ihn präzise
vorhergesagt.** Baustein C hatte in S4 festgestellt, dass der NaN-Payload-Vorbehalt für
`argmax` strukturell nicht greifen kann (die Ausgabe ist immer ein Index) und erst bei
`topk` beißt, wo echte Datenwerte durch den Kernel wandern. Der Beweis lief in drei Stufen:
Baustein 0 schloss vorab eine Lücke im Repo-Präzedenzfall (die bestehende Fixture deckt nur
2D-Transpose ab; er schickte zusätzlich eine Rang-1, gestridete, Offset-Sicht mit
nicht-kanonischem NaN durch `nt_materialize` — genau das Lesemuster des künftigen Kernels)
und grenzte das ehrlich als Analogie-Evidenz ab; die Implementierung ersetzte sie durch
direkte Evidenz an `nt_topk_strided` (zwei verschiedene Bitmuster, roh per `DataView`
gesetzt, im WASM-Speicher verifiziert, contiguous und gestridet, stable und über alle vier
threaded-Pools, plus ein cargo-Test).

**Baustein 0 fand außerdem einen MAJOR an der Spec und entlarvte eine meiner
Vorsichtsmaßnahmen als überstreng.** Der Append-Punkt: v1 nannte die letzte reale Funktion
als „letztes Item" von `abi.rs` — tatsächlich ist es ein Testmodul, und die Datei hat ein
durchgängiges, nie gebrochenes Muster (Realcode nach dem jeweils aktuellen Dateiende, dann
das eigene Testmodul). Wörtlich gelesen hätte v1 erstmals Realcode vor ein bestehendes
Testmodul gespleißt; für den Freeze-Hash folgenlos, aber musterbrechend. Und die
Budget-Disziplin: ich hatte aus der S3-Lektion (+6.705 pro literaler Aufrufstelle) abgeleitet,
Differentialtests müssten `k` dynamisch verwenden. Der Verifier maß es statt es zu glauben —
der Aufschlag liegt im niedrigen dreistelligen Bereich. Die Restriktion wurde auf eine
Vorsicht mit Messpflicht zurückgestuft.

**Die Zahl selbst blieb am Ende ungeklärt, und das steht so im Ergebnis-Doc.** Sie wurde
dreimal gemessen und ergab dreimal etwas anderes: ≈122 (Proxy), ≈149 (echter Code), ≈95
(eigene Sonde des Spec-Verifiers). Alle in derselben Größenordnung, alle meilenweit von
S3s +6.705, aber die Streuung ist selbst der Befund — der Instantiation-Zähler ist sichtbar
sensitiv gegenüber dem Boilerplate der Messsonde. Dokumentiert ist deshalb die Spanne mit
dem benannten Mechanismus statt einer der drei Zahlen als Wahrheit.

**Das Gate hielt mit der engsten Marge der Kampagne: +7.551 von ≤+8.000, 449 übrig.** Der
Implementierer hat korrekt nicht ins Gate optimiert, sondern bisektiert — und dabei den
verwertbarsten Nebenbefund der Scheibe gefunden: der Kostentreiber ist nicht `topk` (+130),
sondern der Vier-View-Klassen-Block, den Arbeitsregel 12 verlangt, mit +3.926 = 52 % der
ganzen Scheibe. Getrieben nicht von der Op, sondern von den literal-argumentigen
`.slice()`-Aufrufstellen darin, die je die volle `SliceSpecsGuard`/`SliceShape`-Maschinerie
zahlen. Für Laufzeittests kauft ein literales Slice-Spec nichts. Die Zahl ist dreifach
unabhängig durch Ausbau des Blocks bestätigt (Implementierer, Baustein A, Baustein B — alle
exakt 233.453 @ 140 ohne ihn) und wird als Hausregel-Kandidat weitergetragen.
> **Korrektur 2026-07-25 (Gegenprüfung am zweiten Korpus):** der Betrag +3.926 hält (ein
> viertes Mal reproduziert), die beiden Schlussfolgerungen dieses Absatzes nicht. Nur 2.232
> davon (57 %) sind slice-getrieben, und der strukturell identische S4/argmax-View-Block
> kostet lediglich 734 — die Zahl ist ein Ausreißer, keine Eigenschaft der Blockklasse.
> Details im eigenen Abschnitt weiter unten und in docs/slice-literal-budget-ergebnisse.md.

**Baustein B fand einen Testkommentar, der mehr behauptete als sein Test beweist** — zum
dritten Mal in dieser Kampagne dieselbe Klasse. Der cargo-Test zum zweiten ABI-Regionscheck
rief mit beiden Pointern auf 0 und demselben `out_len` auf, sodass der erste Check immer
zuerst griff; entfernt man den zweiten Check, bleibt der Test grün. B konstruierte einen
diskriminierenden Fall und bewies es per Mutant. Nach S3 (Leck-Test drei Größenordnungen zu
grobkörnig) und S4 (eine „unabhängige Bestätigung", die 500 × 8 Byte gegen 64-KiB-Seiten
hielt) ist das der dritte Fund — Bewusstsein allein verhindert die Klasse offenbar nicht, es
braucht die mechanische Probe. Geschlossen mit dem diskriminierenden Test; der überclaimende
Kommentar wurde auf das zurückgezogen, was er wirklich prüft.

B hat außerdem die Tie-Blindheits-Lektion aus S4 präzisiert und damit korrigiert: blind sind
nicht „randomisierte" Tests, sondern Ziehungen aus einer **stetigen** Verteilung. Diskrete
Spezialwert-Generatoren treffen Gleichstände gelegentlich (gemessen: 0 von 140
stetig-zufälligen Fällen fingen den Tie-Mutanten, aber 6 von 66 Spezialwert-Fällen, und alle
11 von 11 konstruierten).

**Baustein C korrigierte eine Fragestellung des Orchestrators.** Ich hatte gefragt, ob die
Scheibe den NaN-Payload-v6-Kandidaten schließt. C: nein — sie liefert den fehlenden
Gegenfall, der die Scope-Grenze des Vorbehalts erst sichtbar macht, denn der Kandidatentext
spricht nur von Arithmetik-Ergebnissen, und `topk` liegt auf der anderen Seite. Ohne diese
Grenze läse ein künftiger v6-Leser „NaN nur als Wert-Klasse" für alle Kernel. C fand
außerdem eine Inkonsistenz in der Projekt-Historie: der Wortlaut von M3 deckt
Methoden-Rückgabetypen nicht, aber S3 hat ein strukturell identisches Problem als echte
M3-Verletzung gewertet, für +5.498 gefixt und das Gate dafür eigens angehoben. Entweder war
S3 zu weit gefasst oder M3 muss den Fall decken — Owner-Entscheidung, Wortlaut liegt vor.
Das v6-Bündel steht damit bei zehn Kandidaten.

Arbeitsregel 13 wurde diesmal dreifach unabhängig erfüllt: Implementierer, A und B fuhren je
eine echte `tsc --lsp --stdio`-Messung mit Kontrollpunkt. Alle drei: der Rückgabetyp hovert
als aufgelöstes Objekt-Literal, kein Alias-Leck.

Endstand: Freeze-Hash `146afdf629694318a5dcca87c5bb980ae6280e875ae5d9b5ddef045a00c0c324`,
check:diag 237.379 @ 140, stress 116.053 @ 82, browser 2.142 @ 75 unverändert, bench:editor
uniform +119, test:resident 6122+2, test:threaded 139, cargo 222+1.

**Damit ist die Kampagne S0–S5 abgeschlossen.** `WNDArray` und der threaded-Pfad tragen alle
fünf Dogfooding-Ops; die README trägt keine „TypeScript-runtime only"-Ausnahme mehr, empirisch
gegen `spike/src/index.ts` verifiziert.

Details: docs/wasm-parity-topk-spec.md v2, docs/wasm-parity-topk-ergebnisse.md.

### Nachtrag zu S5: der CI-Fund, den kein lokales Gate sehen konnte (2026-07-25)

Der committete Stand `6fc2d47` war in acht von neun CI-Jobs grün und rot auf `freeze` —
lokal hatte jedes Gate gehalten. Die Ursache: `order.sort_by(...)` war der erste Gebrauch
von `core::slice::sort` im ganzen Crate. Damit landeten erstmals die Panic-Sites von Rusts
Sortier-Maschinerie im Artefakt, und eine davon war nicht auf `/rustc/<hash>/…` remapped,
sondern zeigte in den lokal installierten `rust-src`-Baum — inklusive Benutzername und
Host-Triple. Linux baute deshalb deterministisch einen anderen Hash (Rerun bestätigte
denselben Wert), und `build:dist` hätte den Pfad in den npm-Tarball getragen.

Vor jeder Entscheidung wurde geprüft statt angenommen, ob je etwas geleakt ist: das
publizierte `numtype@0.2.0` frisch aus der Registry gezogen (null Treffer auf `/Users/`
oder `rustup`, drin nur der remappte `/rustc/…/raw_vec/mod.rs`), und die gesamte
git-Historie — es wurde nie ein `.wasm` committet, `.gitignore` greift, und `git grep` über
`git rev-list --all` findet in keiner getrackten Datei einen Host-Pfad. Der Leak existierte
ausschließlich in der lokalen Binärdatei.

Die bequeme Auflösung wäre gewesen, den Linux-Hash als zweiten Plattform-Pin einzutragen —
was `check-freeze-hash.mjs` in seiner Fehlermeldung sogar ausdrücklich anbietet. Das wäre
falsch gewesen: der Hinweis ist für eine echte Erstplattform gedacht, hier baute Linux
vorher nachweislich identisch, also war die Abweichung ein Befund und kein Pin-Anlass —
und ein zweiter Pin hätte ein Artefakt festgeschrieben, das nicht reproduzierbar ist und
einen Home-Pfad trägt.

Stattdessen wurde die Ursache entfernt: ein selbst geschriebener In-Place-Heapsort auf dem
ohnehin vorhandenen Max-Heap ersetzt die std-Sortierung. O(k log k) bleibt erhalten (eine
Insertion Sort war ausdrücklich ausgeschlossen — bei `k = n`, einem getesteten Aufruf, wäre
der Kernel quadratisch geworden), der temporäre Vec entfällt, das Artefakt schrumpft um
11,5 %. Dass Heapsort instabil ist, spielt nachweislich keine Rolle — genau diese Scheibe
hatte bewiesen, dass die Ordnung eine strikte Totalordnung auf paarweise verschiedenen
Indizes ist, es also keine Gleichstände gibt, zwischen denen Stabilität entscheiden könnte.
Der tragende Befund der Spec zahlte sich unmittelbar aus.

Die Testinhalte wurden nicht angefasst — sie sind der Beweis und blieben unverändert grün.
Der Mutanten-Beweis ist ungewöhnlich scharf: die Vergleichsrichtung in `sift_down` gedreht
fällte unter anderem `topk_k_equals_n_is_the_whole_vector_sorted`, und bei `k = n` treten
null Evictions auf — dieser Test durchläuft also ausschließlich den neuen Sortierpfad.

Die Lehre steht als Arbeitsregel 14 in CLAUDE.md. Sie hat zwei Hälften: ein neu benutztes
std-Generikum kann Host-Pfade einschleppen, und lokal ist das unsichtbar. Gefunden hat es
ausschließlich der cross-host laufende CI-`freeze`-Job, dessen Plattform-Unabhängigkeit bis
dahin ein unbemerkter Nebeneffekt war — hier erwies sie sich als load-bearing. Neuer
Freeze-Pin `2a54d9fdba55e4e88a9d54cb3b01e111c2717abf13017f778b90accd5cff87e4`.

---

## COVENANT v6: elf Auslegungsfragen in einem Zug (2026-07-25)

Eine reine Vertrags-Scheibe, ohne eine Zeile Produktivcode. Über die Kampagnen W1–W5 und
S0–S5 waren Stellen aufgelaufen, an denen der Vertragstext die gelebte Praxis nicht abdeckte
— sämtlich aus `covenant-verify`- oder Baustein-0-Befunden, sämtlich in FOLLOWUPS getrackt,
keine davon ein offener Normbruch. Sie in Einzelscheiben abzuarbeiten wäre teurer gewesen als
der Nutzen; gebündelt liest sich der Vertrag danach als ein Stück.

**Zwei Punkte waren echte Owner-Entscheidungen, der Rest Konsolidierung.** Die erste betraf
M3 und war unangenehm, weil sie rückwirkend war: der Wortlaut deckte nur Klassen-Hover, aber
S3 hatte einen Methoden-Rückgabetyp als echte M3-Verletzung gewertet, dafür +5.498
Instantiations bezahlt und das Scheiben-Gate eigens von +8.000 auf +13.000 angehoben — mit
Owner-Abnahme. Entweder war die Einstufung zu weit, oder der Vertrag beschrieb die gehaltene
Norm nicht. Entscheidung: M3 erweitern, S3 war richtig. Damit beschreibt der Vertrag, was das
Projekt nachweislich hält — es hat zweimal danach gehandelt und einmal echtes Budget gezahlt.
Preis: die LSP-Messung wird für solche Flächen Pflicht statt Kür, was angesichts der
S3-Erfahrung (drei Leser übersahen die Verletzung, nur der Messende fand sie) folgerichtig ist.

Die zweite ging andersherum aus: die „insertion-only"-Disziplin für TS-Klassenkörper bleibt
bewusst Hausregel und wandert NICHT in den Vertrag. M4s Begründung ist ein
Artefakt-Byte-Argument — eine Zeilenverschiebung ändert `#[track_caller]`-Metadaten und damit
die kompilierten Bytes unberührter Funktionen —, und dieser Mechanismus hat in TypeScript
keinen Gegenpart. v6 sagt das jetzt ausdrücklich, statt die Frage jede Scheibe neu aufkommen
zu lassen.

**Die Verifikation lief mit umgedrehter Frage.** Der übliche `covenant-verify` prüft „hält der
Code den Vertrag" — hier wäre das zirkulär gewesen, denn der Vertrag war der Diff. Die
relevante Frage lautete: stimmt jede neue Klausel faktisch? Ein Vertragstext, der etwas
Falsches über den Code behauptet, vergiftet ab sofort jede Scheiben-Prüfung, weil er
maschinell gelesen und angewendet wird. Der Verifier prüfte deshalb jede Klausel einzeln am
Code — Op-Zuordnung der NaN-Payload-Klassen, Existenz der behaupteten Tests, die
Rang-Cliff-Zahlen, ob `w8` wirklich im `editor-gate`-Job mitläuft — und reproduzierte den
W2-Overload-Grenzfall sogar empirisch mit einer eigenen `tsc`-Probe.

Ergebnis: keine einzige falsche Tatsachenbehauptung, aber drei Präzisionslücken, alle vor dem
Commit korrigiert. Die wichtigste betraf meinen eigenen Änderungslog: ich hatte geschrieben,
alle Normen blieben inhaltlich unverändert „mit einer Ausnahme". Der Verifier zeigte, dass es
**zwei** sind — die Z2-Erweiterung legalisiert eine Praxis, die vom bisherigen Wortlaut
wörtlich abwich, was das Scale-Probe-Ergebnisdoc selbst so schreibt. Dazu: eine Zeitangabe war
als Spanne formuliert, obwohl nur ein Einzelwert gemessen ist (68,51 s), und `transpose` war
zweideutig — der Rust-Kern `nt_transpose` kopiert wirklich, `WNDArray.transpose()` ist eine
kernel-lose O(1)-View.

**Eine Prozess-Lehre fiel nebenbei ab.** Nach dem Schließen der zehn Kandidaten lief ein
Kontroll-Grep — und fand einen elften COVENANT-Eintrag, der derselben Sache galt, aber
„v6-Präzisierungs-Kandidat" statt „v6-Kandidat" hieß und deshalb durch meine Extraktion
gefallen war. Der Verifier hatte denselben blinden Fleck, weil er dieselbe Suche benutzte:
er bestätigte ausdrücklich „alle zehn mappen sauber". Ein Verifier, der die Suchmethode des
Geprüften erbt, erbt auch deren Lücken — die Gegenmaßnahme war nicht mehr Sorgfalt beim
Suchen, sondern eine Kontrolle über eine ANDERE Achse (Zählung aller offenen COVENANT-Einträge
statt Suche nach dem erwarteten Stichwort).

Ein zwölfter Kandidat fiel beim Schreiben auf und wurde bewusst NICHT eigenmächtig
eingebaut, sondern dem Owner vorgelegt: nach dem
Host-Pfad-Fund im S5-Abschluss ist an M4 offen, WESSEN Clean-Rebuild den bindenden
Freeze-Beweis liefert, seit das Artefakt host-abhängig werden kann. Ein Vertrag darf nicht
dadurch wachsen, dass dem Schreibenden beim Schreiben noch etwas einfällt. **Der Owner hat
ihn aufgenommen** — v6 enthält ihn als zweite M4-Präzisierung: „Clean-Rebuild" heißt
host-unabhängiger Clean-Rebuild, und eine Abweichung auf einer vorher identisch bauenden
Plattform ist ein Befund, kein Pin-Anlass. Die Nuance, die dabei benannt gehört: die
EIGENSCHAFT galt schon (der CI-Job prüfte seit Item 12 Linux gegen einen macOS-Pin), nur die
daraus folgende REGEL stand nirgends.

## `slice`-Literal-Budget: eine Gegenprüfung, die ihren eigenen Anlass widerlegte (2026-07-25)

Die Scheibe begann als FOLLOWUPS-Eintrag aus WASM-Parität S5: der Vier-View-Klassen-Block in
`resident.test.ts` koste +3.926 Instantiations = 52 % der Scheibe, getrieben von
literal-argumentigen `.slice()`-Aufrufstellen — Kandidat für eine Hausregel. Der Owner hat
nicht die Regel bestellt, sondern die **Gegenprüfung an einem zweiten Korpus**. Das war die
richtige Entscheidung, und sie hat sich sofort ausgezahlt.

**Der Betrag hielt, die Schlüsse nicht.** Der Ausbau des Blocks reproduzierte 233.453 @ 140 auf
die Ziffer — ein viertes Mal nach den drei Messungen in S5. Aber der strukturell identische
S4/argmax-View-Block, gleicher Aufbau, gleiche vier View-Klassen, gleiche drei Slice-Formen,
kostet nur **734**. Faktor 5,3. Aus einer Einzelbeobachtung war eine Aussage über eine
Codeklasse geworden, und die hielt nicht. Zweitens: von den +3.926 sind nur **2.232 = 57 %**
slice-getrieben, der Rest sind gewöhnliche Testinhalts-Kosten. Und drittens war die abgeleitete
Marge falsch — „hätte 449 auf ≈4.400 gehoben" verwechselte Ausbau des Blocks mit Widen seiner
Specs; richtig sind ≈2.681.

**Der eigentlich neue Befund ist methodisch: diese Kosten sind stark super-additiv.** Drei
strukturgleiche Blöcke einzeln gemessen ergeben 549 + 579 + 549; zusammen gemessen **3.034**.
Alle 32 Stellen über sieben Dateien zusammen **11.780**, rund 3.200 über der Teilsummen-
Erwartung. Und eine einzelne Stelle isoliert zu widen kann den Zähler sogar **erhöhen** (+207
an Zeile 2087, zweimal reproduziert). Damit ist die in der Notiz implizit verwendete
Pro-Aufrufstelle-Rechnung nicht bloß ungenau, sondern **undefiniert** — es ist derselbe
Fresh-vs-Cached-Partitionsmechanismus wie beim Order-Noise, nur innerhalb eines fixen
File-Sets, ohne dass eine Datei dazukommt. Das erklärt nachträglich einen Befund aus S5, der
dort als „eine Zahl bleibt ungeklärt" stehen geblieben war: die Literal-Mehrkosten *einer*
Aufrufstelle waren dreimal gemessen worden und hatten drei verschiedene Werte ergeben
(≈122/≈149/≈95). Kein Messfehler — eine solche Zahl existiert nicht.

**Die Prämisse hielt trotzdem, und daraus wurde die Scheibe.** Die sieben betroffenen Dateien
enthalten null Typ-Ebenen-Assertionen, und `slice.test-d.ts` sagt in seinem eigenen Dateikopf,
dass `WNDArray.slice` über das NDArray-Pinning mitabgedeckt ist. Die literalen Specs kaufen
dort also nachweislich nichts. Die Schreibweise wurde **gemessen statt gewählt**: drei
Kandidaten, der geteilte `wideSpecs(...)`-Helfer gewinnt gleichzeitig auf Kosten (225.599),
Kürze und Lesbarkeit.

**Baustein 0 fing zwei Dinge vor der ersten Codezeile.** Erstens fehlte D1s Kriterium die
**Empfänger-Bedingung**: `RankUnknowable` greift in `slice.ts` schon VOR jeder Spec-Betrachtung,
also kosten literale Specs auf dynamisch getippten Empfängern gar nichts (gemessen Δ−35 über
drei solche Stellen). Die 32er-Liste war richtig, die aufgeschriebene Herleitungsregel nicht —
und genau die wäre über D6 als Dauerregel ins Projekt gewandert. Zweitens war die
Nicht-Vakuitäts-Pflicht **ortsabhängig** formuliert und hätte mit 41 % Wahrscheinlichkeit einen
Scheinbeweis geliefert.

**Daraus wurde D7, und das ist der teuerste Fund der Scheibe — obwohl er nicht zu ihr gehört.**
An 13 der 32 Stellen ist das Test-Orakel selbstreferentiell: `assertMeanViewMatches` und
Geschwister bilden ihre Referenz aus dem `.toArray()` des **bereits konstruierten** Views. Wird
der Slice-Spec verfälscht, entsteht ein anderer View, aber Referenz und Kandidat rechnen beide
konsistent darüber weiter. Selbst nachgestellt: `resident.test.ts:501` von `{step:2}` auf
`{step:1}` gedreht — was aus dem gestrideten View einen contiguous macht und damit genau die
Eigenschaft zerstört, für die der Block existiert — ergibt **6122 pass, 0 fail**. Die Tests
sind nicht falsch; verloren ist die *Coverage-Aussage*, dass der Empfänger die View-Klasse ist,
die der Testname behauptet. Ausgerechnet **Arbeitsregel 12** stützt sich auf diese Blöcke. Der
S5/topk-Block ist die Ausnahme, weil sein Autor Precondition-Assertions gesetzt hat. Der Befund
ist vorbestehend, wird durch die Migration weder verursacht noch verschlimmert, und wurde
deshalb bewusst out of scope gehalten — er steht als eigener FOLLOWUPS-Eintrag.

**Baustein B fand einen MAJOR, den die Scheibe selbst erzeugt hatte.** 32 Aufrufstellen durch
einen geteilten Helfer zu führen macht den Code lesbarer und billiger — und erzeugt eine neue
Single-Point-of-Failure-Fläche, gegen die **22 von 32** Stellen blind sind. Neun davon durch
einen Mechanismus, den D7 nicht abdeckt: an ihnen bauen die naive Referenz UND der residente
Kandidat ihren View über denselben Helfer mit denselben Argumenten, ein Helfer-Bug hebt sich
also symmetrisch auf. Selbst nachgeprüft: `return specs` → `return []`, der destruktivste
denkbare Helfer-Bug, schlägt sich in nur 23 von über 7.700 Fällen nieder. Geschlossen mit zwei
direkten Tests für `wideSpecs` (Nicht-Vakuität am subtileren Mutanten `return specs.slice(0,
-1)` bewiesen), bewusst ans Dateiende angehängt, damit die Zeilennummern der D1-Tabelle gültig
bleiben.

**Endstand:** `check:diag` **225.671 @ 140**, von 237.379 — **Δ−11.708 = −4,93 %**, der erste
Rückgang dieser Größenordnung im Projekt. Zweistufig und getrennt ausgewiesen, damit die
Vorregistrierung nachprüfbar bleibt: die Umsetzung traf den vorab im Worktree gemessenen Wert
**225.599 exakt**, die Verify-Runde legte +72 für die zwei neuen Helfer-Tests drauf. Alle
Nebengates Δ0 (stress, browser, alle acht Editor-Pins, Freeze-Hash, cargo), alle übrigen
Testzahlen zahlengleich. Daraus **Arbeitsregel 16** — inklusive der Empfänger-Bedingung und der
Warnung, dass Einzelsite-Messungen in diesem Zähler nicht existieren.

## View-Preconditions: ein Testname ist kein Beleg (2026-07-25/26)

Aus D7 der Budget-Scheibe wurde eine eigene Scheibe. Der Befund dort lautete „13 Stellen mit
selbstreferentiellem Orakel"; real waren es **304 Testfälle in 13 Blöcken**, und die
**transpose**-Untervarianten fehlten in der Ursprungsnotiz vollständig — sie waren nur deshalb
nicht aufgefallen, weil die Vorgänger-Scheibe zufällig ausschließlich Slice-Zeilen angefasst
hatte.

Klassifiziert wurde nach Arbeitsregel 15 über eine orthogonale Achse: statt 60 Aufrufstellen
einzeln zu prüfen, wurden die View-Primitive selbst mutiert (`slice` ignoriert seine Specs;
`transpose` ist die Identität). Die Kontrollgruppe fiel dabei so sauber aus, wie man es sich
nur wünschen kann: der `topk`-Vier-View-Block ist der einzige mit Precondition-Assertions und
der einzige, der beide Mutanten fängt. Der Fix war damit nie eine Erfindung, sondern lag
bereits im Repo.

**Baustein 0 fing zwei Blocker vor der ersten Codezeile, beide meine Fehler und beide in der
BEWEISFÜHRUNG statt im Design.** Erstens war der Geltungsbereich zu klein: `elementwise.test.ts`
fehlte, obwohl FOLLOWUPS die Stellen namentlich nannte — Auslöser war eine Formulierung, die
aus dem korrekten Teilbefund „`sqrt` fängt" den falschen Gesamteindruck „elementwise ist
abgedeckt" machte. Das Signal lag sogar in meiner eigenen Messung (`n=6, gefallen=2`), ich hatte
den Mischwert bemerkt und nicht verfolgt. Zweitens waren die Kontrollgruppen-Zahlen um Faktor 2
zu hoch, weil der Test-Reporter Fehlschläge zweimal druckt und `grep -c` sie doppelt zählt —
ausgerechnet in der Tabelle, die die Spec selbst „der tragende Teil des Beweises" nannte.

Der Fix sind 17 rein additive Einfügestellen: exakte Pins auf Shape, vollständigen
Strides-Vektor und Offset, direkt nach der View-Konstruktion. Ungleichungen wären zu schwach
gewesen (sie fangen keinen Off-by-eine-Konstante), und ein geteilter Assertions-Helfer wurde
bewusst NICHT gebaut — die Vorgänger-Scheibe musste gerade erst reparieren, dass genau so eine
Single-Point-of-Failure-Fläche entsteht. Alle 17 Sollwerte stimmten im ersten Lauf.

**Der teuerste Ertrag war ein Prozess-Befund.** Ich hatte Baustein A und B parallel dispatcht
und beiden erlaubt, Mutanten im HAUPT-Working-Tree anzuwenden. Beide taten es gleichzeitig. B
bemerkte ein fremdes `// MUTANT S` in `resident.ts`, verwarf seine Messung und baute sie in
isolierten Worktrees neu auf. Bei A war die Rückwirkung größer: er hatte eine unerklärte
Diskrepanz (zwei identische Läufe, 669 gegen 654 Fehlschläge, Differenz genau ein Block, den
der Mutant strukturell nicht treffen kann) als „nicht-deterministische Test-Harness-
Kontamination" verbucht — also als vorbestehenden Infrastruktur-Mangel. Auf Nachfrage zog er
die Deutung zurück. **Ohne die Aufdeckung wäre ein Phantom-Befund über die Test-Infrastruktur
ins Projekt gewandert.** Die KB trägt zu dieser Falle bereits eine Notiz; ich habe sie vor dem
Dispatch nicht konsultiert. Beide Verifier hielten sich übrigens korrekt an „Mutant nur als
sofort revertierter Edit mit Backup-Beweis" — die Regel ist für einen Schreiber richtig und für
zwei wertlos, weil ein Backup nur belegt, dass man auf den eigenen Schnappschuss zurückgesetzt
hat, nicht dass der Schnappschuss unverfälscht war.

Alle entscheidenden Zahlen wurden danach in einem Zustand neu erhoben, in dem nur der
Orchestrator schreibt, mit einer Vorbedingungsprüfung vor jedem Mutanten. Dabei wurde auch die
letzte offene Lücke geschlossen: B hatte angemerkt, dass sein Off-by-one-Mutant in 11 Blöcken
nur deshalb fiel, weil er den Lesezugriff aus dem Puffer schob (`.toArray()` warf, bevor eine
Assertion lief), sodass eine **kleine, in-bounds** Korruption ungeprüft blieb. Ein eigens
konstruierter Mutant O (`offset += Math.max(0, spec.start * stride - 1)`, garantiert in-bounds)
löste **93 Fehlschläge aus, die namentlich `precondition — EXACT offset` nennen** — der Pin
leistet die Arbeit selbst.

**Endstand:** `check:diag` **225.983 @ 140**, Δ+312 gegen ≤+4.000, dekomponiert in +4 für die
Preconditions (sie benutzen ausschließlich bereits instanziierte Typen) und +308 für den
D3-Umbau. Alle Nebengates Δ0, alle Testzahlen zahlengleich. Daraus die Verschärfung von
Arbeitsregel 12: der View-Fall muss seine Klasse EXPLIZIT assertieren, nicht nur im Testnamen
behaupten.

## Stand-Review, Prozess-Kalibrierung und Release-Scheibe 0a (2026-09-23)

**Anlass:** nach ≈2 Monaten Pause ein Stand-Review im Auftrag des Owners. Zwei parallele
Agenten: Gate-Gesundheit am HEAD (alle Pins exakt reproduziert, CI grün) und ein
Produkt-Review aus Konsumentensicht (gepackter Build + echtes npm-0.2.0-Tarball + Playground).
Der tragende Befund war kein Code-Fehler, sondern eine Lücke zwischen Arbeit und Nutzer: die
fertige WASM-Parität S0–S5 lag unveröffentlicht auf `main`, während die README sie bereits als
verfügbar bewarb (npm 0.2.0: 0 Treffer für `mean`/`argmax`/`topk`/`item` in `resident.d.ts`).
Dazu vier Konsumenten-Papercuts: `WNDArray` ohne exportierten Typnamen; `node:worker_threads`
in der von `index.d.ts` erreichbaren Deklarationsmenge (TS2591 bei `skipLibCheck: false`);
kein `toJSON`/`inspect` (`JSON.stringify` serialisierte `data` als Objekt); totes v1-`backend.ts`
im Paket.

**Owner-Entscheidungen:** Prozess neu kalibriert (CLAUDE.md 789 → 132 Zeilen, Vorfassung
archiviert; neue Eskalationsstufe 3a „Routine-Scheibe" mit einem kombinierten Verifier,
Baustein R); Klassifikations-Scheibe geparkt; Roadmap 0a → 0b → dtype-Design → 2 → 3 → 1;
dtype: Option C (volles dtype).

**0a (erste Scheibe auf Stufe 3a, docs/release-0.3.0-spec.md v1.1):** D1 `export type
{ WNDArray }`; D2 lokales strukturelles `WorkerHandle` statt `Worker` an der
Deklarationsgrenze (Laufzeit unverändert) + neuer Konsumenten-Smoke `consumer-strict`
(`skipLibCheck: false`, ohne `@types/node`); D3 `toJSON()` → `{ shape, data }`, inspect-Symbol
und `toString()` auf `NDArray`/`WNDArray` über einen geteilten `formatNDArrayDisplay`-Helfer
(runtime.ts, append); D4 `backend.ts` aus dem Build + Emit-Gate; D5 README „New in 0.2.0"
auf das tatsächlich Ausgelieferte zurückgeschnitten, „New in 0.3.0" neu.
**Verifikation:** Verifier R (eigener Worktree) — Verdikt MERGE, alle Gates exakt
reproduziert, 4/4 Mutanten gefangen (geteilter Helfer, `WNDArray.toJSON`-Offset, D2-Revert,
D4-Revert), Freeze-Hash nach Clean-Rebuild byte-identisch `2a54d9fd…`, Hover
`WNDArray<readonly [2, 3]>` per echtem `tsc --lsp --stdio` sauber, keine `node:`-Importe in der
gesamten `dist/*.d.ts`-Menge; covenant-verify: S1/M1–M5/Z1/Z2 halten, keine Befunde. Zwei
Verifier-Nits im selben Zug geschlossen (Rang-1-Pin für `WNDArray`-inspect; echter
`fromArray`-Round-Trip in zwei View-Tests, deren Namen ihn behaupteten); zwei als FOLLOWUPS
(Kürzung großer Arrays in inspect; Browser-Smoke für D3).
**Zahlen:** check:diag 226,148 @ 140 (Δ+165 gegen ≤ +2,000) · stress 116,149 (Δ+96) · browser Δ0
· bench:editor uniform +96 (= stress-Δ, unabhängige Bestätigung der Klassen-Surface-Ripple) ·
test:resident 6145+2 · Rest zahlengleich.
**Lehren:** (1) Eine neue Testdatei hätte check:diag allein durch Order-Noise um ≈+4,600
bewegt — der Implementierer hat das selbst gemessen und die Tests in eine bestehende Datei
gelegt (Δ+164). (2) Der Lint-Graph war zwei Monate alt; ein „0 Verstöße" auf einem veralteten
Graph belegt nichts — vor dem Lint neu bauen (`graph-a-lama . --symbols`).

**Nachtrag (selber Tag): inspect-Kürzung vor dem Release** (Owner-Entscheidung, Stufe 2 —
FOLLOWUPS-Mini mit Anker `runtime.ts`, M1 inhaltlich nicht berührt, reine Anzeige).
`formatNDArrayDisplay` fasst ab >1000 Elementen jede Achse >6 auf je 3 Randwerte um `...`
zusammen (NumPy-Defaults); `toJSON`/`toNestedArray` bleiben vollständig. Offengelegt: der
Helfer wurde im Körper geändert statt append-only ergänzt — er entstand in derselben,
unveröffentlichten Release-Scheibe und ist keine Referenzfunktion (Orakel), deren Stabilität
die Append-Konvention schützt. 6 neue Tests (Grenze genau 1000 vs. 1001, beide Achsen in
Rang 2, kurze Achse bleibt ganz, toJSON vollständig, WNDArray-Parität); 2 Mutanten gefangen
(`>=` statt `>`: 1 Test; Rand um eins verkürzt: 4 Tests), je per Backup-`diff` revertiert.
check:diag 226,220 (Δ+72) · stress 116,220 (Δ+71) · bench:editor uniform +71 = stress-Δ,
zweimal bestätigt · test:resident 6151+2 · Lint 0/0 auf frischem Graph.

**Release 0.3.0 (2026-09-24):** `main` gepusht (CI 9/9 grün auf dem Release-Commit, npm-`latest`
da noch 0.2.0), Owner-Publish per `pnpm publish`, danach Registry-Tarball verifiziert (SHA-512 =
Registry-Integrität; neue `WNDArray`-Member, `WNDArray`-Typexport, 0 `node:`-Importe in den
`.d.ts`, kein `backend.*`, Kürzung enthalten), Beispiel auf `^0.3.0` gebumpt — der UNVERÄNDERTE
Beispielcode läuft grün (Drop-in-Kompatibilität 0.2 → 0.3), dann Tag `v0.3.0`.
**Stolperer:** eine Abfrage der Tarball-URL VOR der Propagation erzeugte einen CDN-Negativcache —
der Tarball blieb ≈5 Minuten 404, obwohl die Metadaten 0.3.0 schon zeigten (mit Query-Parameter
sofort 200). Nach einem Publish die Tarball-URL erst abfragen, wenn die Metadaten die Version
zeigen, oder mit Cache-Bust.

## 0b — typisiertes `toNestedArray` (2026-09-24, Stufe 3b)

`toNestedArray()` auf `NDArray`/`WNDArray` liefert einen am Rang berechneten Typ statt
`unknown` (docs/typed-nested-array-spec.md v2 / -ergebnisse.md). **Verlauf:** eine
Wegwerf-Probe VOR der Spec entschied zwei Designfragen (die View kann den Typ wegen TS2636
nicht tragen; der Methoden-Hover löst auf); Baustein 0 fand trotzdem einen Blocker, den die
Probe nicht abgedeckt hatte — die rekursive Zerlegung über `S` liefert für Unions gleichen
Rangs `number[][] | number[][]` (eigene Alias-Metadaten je Zweig, keine Deduplizierung). Fix:
Rang-Akkumulator über `S["length"]` (eine Instanziierung pro Rang, zugleich das tail-rekursive
Muster der Hausregel); Preis: Rang-Grenze 999 statt >1024. Verify A+B+C: A fand einen
fehlenden zugesagten Pin (geschlossen — die erste Fassung war mit `?.` selbst vakuös), B
5/5 Mutanten gefangen, C eine Wortlaut-Spannung in M3 → **COVENANT v7** (rekursive,
exportierte Aliase in Rückgabe-Position konform). check:diag 227,405 (Δ+1,185 gegen ≤ +4,000),
stress Δ+59, bench:editor uniform +60 — die erste Abweichung zwischen beiden, eingegrenzt auf
die verschiedene Korpus-Schließung, nicht isoliert. Unveröffentlicht: nach der SemVer-Policy
ein Minor (die Verengung von `unknown` kann Konsumenten-Casts brechen) → gebündelt mit 0.4.0.
**Lehren:** (1) eine Vorab-Probe entscheidet nur die Fragen, die sie stellt — die Union-Kante
stand in der Spec als Behauptung („ergibt korrekt `number[][]`"), nicht als Probe-Ergebnis;
(2) Typ-Pins über Indexzugriffe sind leicht vakuös (`?.` fügt `undefined` unabhängig vom
geprüften Flag hinzu) — Gegenmutante gehört dazu.

## dtype-Design (2026-09-24/25, Stufe 3b)

Nach der Owner-Wahl „volles dtype" (Option C): Bestandsaufnahme (67 → gezählt 98 öffentliche
Signaturen, 26 f64-Kernel, Shape-Maschinerie vollständig dtype-unabhängig), sieben
Owner-Entscheidungen E1–E7 in einfacher Sprache vorgelegt, Baustein 0 (Blocker: Union-dtype
hebelt die bool-Ablehnung aus → Union-Gate; Major: `this`-Sperren mit unlesbarer Meldung → O2
Guard-Muster), dann ein Prototyp auf einem nie gemergten Branch mit stufenweiser Messung und
Verify A+B+C. **Ergebnis:** tragfähig mit Nachbesserungen — Maschinerie +6,980, Laufzeit in 18
adversarialen Proben fehlerfrei; zwei Defekte für dt1 (`AnyNDArray` = nur float64; unbewachte
Prüfreihenfolge); drei TS-Grenzen. Ein zeitlich begrenzter Versuch, die Skalar-Meldungen über
eine einzige Signatur sichtbar zu machen, war korrekt, aber mit +8,495 teurer als die ganze
Maschinerie → NO-GO, benannte Ausnahme. Nebenbei: die TS-„nur einfügen"-Hausregel auf
gewöhnliche Ops eingegrenzt (Owner, 2026-09-25) — ihr Ursprung ist ein Rust-Byte-Argument, das
es in TS nicht gibt. **Lehren:** (1) Überladungen sind an heißen Aufrufstellen fast kostenlos, eine
bedingte Ein-Signatur-Form dagegen nicht — Diagnose-Qualität kostet dort Budget pro Aufrufstelle;
(2) die Prototyp-Obergrenze hätte vorab sagen müssen, ob Tests mitzählen.
