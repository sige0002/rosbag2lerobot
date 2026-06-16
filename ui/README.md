# bagel UI (frontend)

Presentation-only frontend for the `bagel ui` command. TypeScript + a single
`index.html`, bundled with [esbuild](https://esbuild.github.io/). It holds **no
privilege**: all filesystem access and process execution live in the Python
backend (`bagel ui`), which exposes an allow-listed JSON API. This frontend just
renders that API and shows the equivalent `bagel ...` CLI command for every
action.

**node/npm are required only to *build* the frontend.** Once `ui/dist/` exists,
running `bagel ui` needs only Python — no Node at runtime.

## Build

```bash
# from this directory (ui/)
npm install        # installs esbuild + typescript (devDependencies only)
npm run build      # typecheck (tsc --noEmit) + bundle -> ui/dist/
```

`npm run build` runs `build.mjs`, which:

1. typechecks the sources with `tsc --noEmit`,
2. bundles `src/main.ts` into `dist/bundle.js` (minified, with sourcemap),
3. copies `index.html` to `dist/index.html`, rewriting the dev script tag
   (`./src/main.ts`) to the built bundle (`./bundle.js`).

The output is `ui/dist/` (`index.html` + `bundle.js` + `bundle.js.map`).

## How the backend serves it

`bagel ui` (`src/bagel/cli.py` → `bagel/ui/server.py`) serves `ui/dist/` when
`ui/dist/index.html` exists; otherwise it falls back to a packaged placeholder
page (`src/bagel/ui/static/`) so the command is usable before the frontend is
built. So the workflow is: build once here, then launch the backend:

```bash
# from the repo root, after `npm run build` here
bagel ui --bags-root /path/to/bags --output-root /path/to/output
```

See [`docs/cli_reference.md`](../docs/cli_reference.md#ui) for the `bagel ui`
options, security model (127.0.0.1-only, per-launch session token,
path-traversal-confined roots), and examples.

## Develop / typecheck

```bash
npm run typecheck   # tsc --noEmit only (no bundle)
```

Sources live in `src/` (`main.ts`, `api.ts`, `panels.ts`). `dist/` and
`node_modules/` are gitignored — rebuild with `npm run build` after pulling.
