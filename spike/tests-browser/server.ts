/**
 * V3 (Browser-Smoke-Test, docs/phase-d-vorarbeiten-spec.md D-V3.3): a
 * minimal static-file HTTP server for the browser smoke test's own test
 * fixture — zero new runtime dependencies (`node:http`, ambient-declared in
 * spike/src/ambient.d.ts, same "scoped shim" discipline as every other
 * declaration there). Serves the tsc-emitted product tree
 * (spike/tests-browser/.emit/, produced by tsconfig.emit.json + the
 * test:browser script's wasm copy step) on an OS-assigned ephemeral port —
 * `smoke.test.ts` owns the server's lifecycle (start in `test.beforeAll`,
 * close in `test.afterAll`).
 *
 * Deliberately serves NO COOP/COEP headers (no `Cross-Origin-Opener-Policy`/
 * `Cross-Origin-Embedder-Policy`) — that absence is exactly the point of
 * this test (spec scope: "COOP/COEP-frei ausgeliefert", i.e. the standard
 * surface must not *require* cross-origin isolation to load and run).
 *
 * MIME map: `.wasm` -> `application/wasm` (REQUIRED —
 * `WebAssembly.instantiateStreaming` rejects on a mismatched content-type,
 * and spike/src/wasm/loader.ts's browser branch has no fallback on a
 * rejected streaming instantiation, loader.ts:158-160; D-V3.4.3's mutation
 * proof flips the entry below to `application/octet-stream` by hand and
 * observes `pnpm test:browser` go red), `.js` -> `text/javascript`,
 * everything else -> `application/octet-stream`. `/` (and any path with no
 * file extension) serves a tiny synthesized HTML document rather than a
 * checked-in `index.html` — the test only needs a real http(s) page to
 * `page.goto()` into (so `crossOriginIsolated`/`typeof process` assertions
 * and same-origin dynamic `import()` all behave like a real deployed page),
 * not any actual page content.
 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";

const MIME: Record<string, string> = {
  ".wasm": "application/wasm",
  ".js": "text/javascript",
};

const BLANK_HTML = "<!doctype html><html><head><title>numtype browser smoke</title></head><body></body></html>";

function extensionOf(pathname: string): string {
  const dot = pathname.lastIndexOf(".");
  return dot === -1 ? "" : pathname.slice(dot);
}

export interface StaticServer {
  readonly baseUrl: string;
  close(): Promise<void>;
}

/** Serve `rootDir` (a `file://` directory URL) as static files at
 * `http://127.0.0.1:<ephemeral port>/`. */
export function startStaticServer(rootDir: URL): Promise<StaticServer> {
  const server = createServer((req, res) => {
    void (async () => {
      const rawPath = (req.url ?? "/").split("?")[0] ?? "/";
      const ext = extensionOf(rawPath);
      if (ext === "") {
        res.writeHead(200, { "content-type": "text/html" });
        res.end(BLANK_HTML);
        return;
      }
      const fileUrl = new URL("." + rawPath, rootDir);
      try {
        const bytes = await readFile(fileUrl);
        const mime = MIME[ext] ?? "application/octet-stream";
        res.writeHead(200, { "content-type": mime });
        res.end(bytes);
      } catch {
        res.writeHead(404, { "content-type": "text/plain" });
        res.end(`not found: ${rawPath}`);
      }
    })();
  });

  return new Promise((resolve, reject) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null) {
        reject(new Error("startStaticServer: server has no address after listen"));
        return;
      }
      resolve({
        baseUrl: `http://127.0.0.1:${address.port}`,
        close: () =>
          new Promise<void>((res2, rej2) => {
            server.close((err) => (err ? rej2(err) : res2()));
          }),
      });
    });
  });
}
