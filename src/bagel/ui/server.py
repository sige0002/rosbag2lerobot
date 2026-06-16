"""Stdlib HTTP server for the ``bagel ui`` backend (plan.md D-7).

A :class:`http.server.ThreadingHTTPServer` bound to ``127.0.0.1`` only, with a
:class:`BaseHTTPRequestHandler` that:

- enforces token auth (header ``X-Bagel-Token``, or — for the preview iframe
  only — a ``?token=`` query param) on every ``/api/*`` route;
- dispatches parsed JSON requests to :class:`bagel.ui.api.Api`;
- serves the static frontend bundle (``ui/dist`` if built, else the placeholder
  ``src/bagel/ui/static/``) on ``/`` and ``/assets/*`` *without* auth.

No web framework — just ``http.server`` + ``json``. Threading lets the FE poll
``/api/convert/{id}`` while other requests are in flight.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bagel.ui.api import Api, ApiError
from bagel.ui.security import token_matches

logger = logging.getLogger(__name__)

# Static MIME types we serve from the frontend bundle. Anything else falls back
# to ``application/octet-stream``.
_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def make_server(
    api: Api,
    static_dir: Path,
    port: int,
) -> ThreadingHTTPServer:
    """Build a configured (but not yet serving) HTTP server.

    Args:
        api: The backend API instance (holds roots, token, job registry).
        static_dir: Directory whose contents are served on ``/`` + ``/assets/*``.
        port: TCP port to bind on ``127.0.0.1``.

    Returns:
        A :class:`ThreadingHTTPServer` ready for ``serve_forever()``.
    """
    handler = _make_handler(api, static_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def _make_handler(api: Api, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    """Return a request-handler class closed over *api* and *static_dir*."""

    class Handler(BaseHTTPRequestHandler):
        """Per-request handler for the bagel UI."""

        server_version = "bagel-ui"

        # Route API logging through the module logger, not stderr.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
            logger.debug("%s - %s", self.address_string(), fmt % args)

        # -- auth --------------------------------------------------------

        def _authed(self, query: dict[str, list[str]], *, allow_query: bool) -> bool:
            """Check the request's token (header always; query iff allowed)."""
            header_token = self.headers.get("X-Bagel-Token")
            if token_matches(api.token, header_token):
                return True
            if allow_query:
                q = query.get("token", [None])[0]
                if token_matches(api.token, q):
                    return True
            return False

        # -- responses ---------------------------------------------------

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, detail: str) -> None:
            self._send_json(status, {"error": True, "detail": detail})

        def _send_html(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ApiError(400, f"Invalid JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise ApiError(400, "JSON body must be an object.")
            return data

        # -- GET ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - http.server contract
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path.startswith("/api/"):
                self._handle_api_get(path, query)
                return
            # Static assets (unauthenticated).
            self._serve_static(path)

        def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
            # Preview accepts the token via header OR ?token= (iframe-friendly).
            allow_query = path == "/api/preview"
            if not self._authed(query, allow_query=allow_query):
                self._send_error_json(401, "Missing or invalid token.")
                return
            try:
                if path == "/api/config":
                    self._send_json(200, api.config())
                elif path == "/api/preview":
                    dataset = query.get("dataset", [""])[0]
                    html = api.preview(dataset)
                    self._send_html(200, html)
                elif path.startswith("/api/convert/"):
                    job_id = path[len("/api/convert/") :]
                    self._send_json(200, api.convert_status(job_id))
                else:
                    self._send_error_json(404, f"No such endpoint: {path}")
            except ApiError as exc:
                self._send_error_json(exc.status, exc.detail)
            except Exception as exc:  # noqa: BLE001 - surface as 500, never crash thread
                logger.exception("Unhandled error on GET %s", path)
                self._send_error_json(500, str(exc))

        # -- POST --------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802 - http.server contract
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if not path.startswith("/api/"):
                self._send_error_json(404, f"No such endpoint: {path}")
                return
            if not self._authed(query, allow_query=False):
                self._send_error_json(401, "Missing or invalid token.")
                return

            try:
                body = self._read_body()
                handler = {
                    "/api/browse": api.browse,
                    "/api/inspect": api.inspect,
                    "/api/scaffold": api.scaffold,
                    "/api/validate-config": api.validate_config,
                    "/api/convert": api.convert,
                    "/api/validate-dataset": api.validate_dataset,
                    "/api/quality-report": api.quality_report,
                }.get(path)
                if handler is None:
                    self._send_error_json(404, f"No such endpoint: {path}")
                    return
                self._send_json(200, handler(body))
            except ApiError as exc:
                self._send_error_json(exc.status, exc.detail)
            except Exception as exc:  # noqa: BLE001 - surface as 500, never crash thread
                logger.exception("Unhandled error on POST %s", path)
                self._send_error_json(500, str(exc))

        # -- static ------------------------------------------------------

        def _serve_static(self, path: str) -> None:
            """Serve a file from *static_dir*; ``/`` maps to ``index.html``.

            Path confinement: the request path is treated as relative, ``..``
            segments are rejected, and the resolved file must stay inside
            *static_dir*.
            """
            rel = path.lstrip("/")
            if rel in ("", "assets", "assets/"):
                rel = "index.html"
            # Confine to static_dir (the resolved file must stay inside it); this
            # is the sole gate — any ``..`` escape fails the containment check.
            target = (static_dir / rel).resolve()
            base = static_dir.resolve()
            if target != base and not target.is_relative_to(base):
                self._send_error_json(403, "Forbidden.")
                return
            if not target.is_file():
                # SPA fallback: unknown non-asset paths render the app shell.
                target = (static_dir / "index.html").resolve()
                if not target.is_file():
                    self._send_error_json(404, "Not found.")
                    return
            self._send_file(target)

        def _send_file(self, target: Path) -> None:
            data = target.read_bytes()
            mime = _MIME_TYPES.get(target.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
