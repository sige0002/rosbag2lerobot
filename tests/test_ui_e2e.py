"""End-to-end test for the ``bagel ui`` backend against a real bag.

Drives the full reproduction loop over HTTP (stdlib :mod:`http.client`) against
a live :class:`http.server.ThreadingHTTPServer`:

    browse -> scaffold -> validate-config -> convert -> poll -> quality-report
    -> validate-dataset -> preview

Uses the real bag ``bagdata/airoa-moma-mcap/235210`` with the checked-in
``configs/hsr.yaml`` (the scaffolded config has empty actions and is only smoke-
checked; conversion uses the known-good hsr config). Skipped when ``bagdata/``
or ``ffmpeg`` is absent.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from bagel.ui.api import Api
from bagel.ui.jobs import JobRegistry
from bagel.ui.security import Root
from bagel.ui.server import make_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAGS_ROOT = PROJECT_ROOT / "bagdata" / "airoa-moma-mcap"
REAL_BAG = BAGS_ROOT / "235210"
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"
TOKEN = "e2e-token-abcdefghijklmnopqrstuvwxyz0123456789"


def _require_real() -> None:
    if not REAL_BAG.is_dir():
        pytest.skip(f"real bag not present: {REAL_BAG}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


@pytest.fixture
def server_ctx(tmp_path: Path):
    _require_real()
    output_root = tmp_path / "out"
    output_root.mkdir()
    roots = [
        Root(id=0, label="bags", path=BAGS_ROOT),
        Root(id=1, label="output", path=output_root),
    ]
    api = Api(roots=roots, token=TOKEN, registry=JobRegistry())
    static_dir = PROJECT_ROOT / "src" / "bagel" / "ui" / "static"
    server = make_server(api, static_dir, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def conn() -> HTTPConnection:
        return HTTPConnection("127.0.0.1", port)

    try:
        yield conn
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(conn_factory, path: str, token: str | None = TOKEN):
    c = conn_factory()
    headers = {"X-Bagel-Token": token} if token is not None else {}
    c.request("GET", path, headers=headers)
    resp = c.getresponse()
    body = resp.read()
    ctype = resp.getheader("Content-Type", "")
    c.close()
    return resp.status, body, ctype


def _post(conn_factory, path: str, payload: dict[str, Any]):
    c = conn_factory()
    headers = {"Content-Type": "application/json", "X-Bagel-Token": TOKEN}
    c.request("POST", path, body=json.dumps(payload), headers=headers)
    resp = c.getresponse()
    data = json.loads(resp.read() or b"{}")
    c.close()
    return resp.status, data


@pytest.mark.integration
def test_ui_full_loop(server_ctx) -> None:
    conn = server_ctx
    bag_abs = str(REAL_BAG)

    # 1. browse the bags root and find the 235210 bag.
    status, data = _post(conn, "/api/browse", {"root_id": 0, "subpath": ""})
    assert status == 200, data
    assert "command" in data
    assert any(b.endswith("/235210") for b in data["bags"]), data["bags"]

    # 2. scaffold a config from the bag (stdout, no file write).
    status, data = _post(
        conn,
        "/api/scaffold",
        {"bags": [bag_abs], "robot_type": "hsr", "task": "demo"},
    )
    assert status == 200, data
    assert "command" in data
    assert "robot_type" in data["yaml"]

    # 3. validate-config: smoke-check the scaffolded YAML against the bag.
    status, data = _post(
        conn,
        "/api/validate-config",
        {"config_yaml": data["yaml"], "bags": [bag_abs]},
    )
    assert status == 200, data
    assert "command" in data
    assert data["report"]["results"]["verdict"] in ("OK", "FAIL")

    # 4. convert with the known-good hsr config (async subprocess).
    hsr_yaml = HSR_CONFIG.read_text()
    status, data = _post(
        conn,
        "/api/convert",
        {
            "config_yaml": hsr_yaml,
            "bags": [bag_abs],
            "output": "ds",
        },
    )
    assert status == 200, data
    assert "command" in data
    job_id = data["job_id"]

    # 5. poll until done.
    final: dict[str, Any] = {}
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        st, body, _ct = _get(conn, f"/api/convert/{job_id}")
        assert st == 200
        final = json.loads(body)
        if final["state"] in ("done", "failed"):
            break
        time.sleep(1.0)
    assert final["state"] == "done", final
    assert final["progress"]["total"] == 1
    assert final["progress"]["done"] == 1
    assert final["summary"]["n_success"] == 1

    # 6. quality-report: expect an OK verdict on a clean single-episode convert.
    status, data = _post(conn, "/api/quality-report", {"dataset": "ds"})
    assert status == 200, data
    assert "command" in data
    assert data["report"]["verdict"] == "OK", data["report"]

    # 7. validate-dataset structural check.
    status, data = _post(conn, "/api/validate-dataset", {"dataset": "ds"})
    assert status == 200, data
    assert data["report"]["verdict"] == "OK", data["report"]

    # 8. preview HTML (header token + iframe-friendly query token).
    status, body, ctype = _get(conn, "/api/preview?dataset=ds")
    assert status == 200, body
    assert "text/html" in ctype
    html = body.decode("utf-8")
    assert "<html" in html.lower()
    # Preview embeds the quality verdict and a recognizable report marker.
    assert "OK" in html

    status, body, ctype = _get(
        conn, f"/api/preview?dataset=ds&token={TOKEN}", token=None
    )
    assert status == 200
    assert "text/html" in ctype
