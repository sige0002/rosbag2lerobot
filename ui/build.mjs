// Build script for the bagel UI.
//
// Steps:
//   1. Typecheck the TypeScript sources (`tsc --noEmit`).
//   2. Bundle src/main.ts -> dist/bundle.js with esbuild.
//   3. Copy index.html -> dist/index.html, rewriting the dev script
//      src ("./src/main.ts") to the bundled "./bundle.js".
//
// Run with: npm run build

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "dist");

// 1. Typecheck. Inherit stdio so tsc errors surface; non-zero exit aborts.
console.log("[build] typecheck (tsc --noEmit)...");
execFileSync("npx", ["tsc", "--noEmit"], { cwd: here, stdio: "inherit" });

// 2. Bundle.
console.log("[build] bundle (esbuild)...");
mkdirSync(dist, { recursive: true });
await esbuild.build({
  entryPoints: [resolve(here, "src/main.ts")],
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
  sourcemap: true,
  outfile: resolve(dist, "bundle.js"),
  logLevel: "info",
});

// 3. Copy index.html, rewriting the script src to the bundle.
console.log("[build] copy index.html -> dist/index.html...");
let html = readFileSync(resolve(here, "index.html"), "utf8");
html = html.replace(
  /<script\s+type="module"\s+src="\.\/src\/main\.ts"><\/script>/,
  '<script type="module" src="./bundle.js"></script>',
);
writeFileSync(resolve(dist, "index.html"), html);

console.log("[build] done -> dist/bundle.js + dist/index.html");
