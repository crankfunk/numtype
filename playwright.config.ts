/**
 * V3 (Browser-Smoke-Test, docs/phase-d-vorarbeiten-spec.md D-V3.1/D-V3.2):
 * Playwright config for the ONE browser test file
 * (spike/tests-browser/smoke.test.ts), Chromium only (v0 scope — WebKit/
 * Firefox are FOLLOWUPS). Lives at the repo root, deliberately OUTSIDE the
 * `include: ["spike"]` root tsconfig corpus (D-V3.2) — Playwright transpiles
 * this file itself via its own bundler, not via any of this project's three
 * `tsc` legs.
 *
 * No `webServer` entry: the static file server is owned by the test file
 * itself (`spike/tests-browser/server.ts`, started in `test.beforeAll`) —
 * per spec, "statischer Server im Test-Fixture über node:http".
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./spike/tests-browser",
  testIgnore: ["**/.emit/**"],
  timeout: 30_000,
  fullyParallel: true,
  retries: 0,
  reporter: [["list"]],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
