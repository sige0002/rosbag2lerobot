"""Localhost control UI for the scaffold→convert→quality loop (plan.md D-7).

The UI is split into a privileged Python backend and a presentation-only
TypeScript/HTML frontend (built separately under ``ui/``). This package holds
*only* the backend:

- :mod:`bagel.ui.security` -- pure path-confinement + token helpers (no I/O
  beyond ``Path.resolve``); the single chokepoint for every filesystem access.
- :mod:`bagel.ui.jobs` -- async ``bagel convert`` subprocess tracking, keyed by
  job id, polling ``meta/job_summary.json`` for incremental progress.
- :mod:`bagel.ui.api` -- the JSON allow-list API. Thin wrappers over existing
  ``bagel`` verbs (imported in-process for the fast read-only verbs, shelled
  out for ``convert``). Every response echoes the equivalent ``bagel ...`` CLI
  invocation so the CLI stays the source of truth.
- :mod:`bagel.ui.server` -- a stdlib ``ThreadingHTTPServer`` request handler
  that binds ``127.0.0.1`` only, enforces token auth on ``/api/*`` and serves
  the static frontend bundle on ``/`` and ``/assets/*``.

No web framework and no new dependencies: everything is stdlib
(``http.server`` + ``json`` + ``subprocess`` + ``secrets``).
"""

from __future__ import annotations
