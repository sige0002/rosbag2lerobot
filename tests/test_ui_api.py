"""HTTP-level tests for the ``bagel ui`` backend API.

Starts a real :class:`http.server.ThreadingHTTPServer` on an ephemeral port in
a background thread and drives it with stdlib :mod:`http.client`. Asserts auth
(missing token -> 401), path confinement (``..`` -> 403), the documented
response keys, and that every endpoint echoes a ``command`` string. A tiny
synthetic dataset (built with :class:`bagel.writer.DatasetWriter`) backs the
dataset-consuming endpoints; ``convert`` is exercised against a stub ``bagel``
subprocess so the unit test stays fast and offline.
"""

from __future__ import annotations

import json
import shutil
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from bagel.ui.api import Api
from bagel.ui.jobs import JobRegistry
from bagel.ui.security import Root
from bagel.ui.server import make_server
from bagel.writer import DatasetWriter

TOKEN = "test-token-abcdefghijklmnopqrstuvwxyz0123456789"


# ---------------------------------------------------------------------------
# Synthetic dataset (mirrors tests/test_validation.py helper style)
# ---------------------------------------------------------------------------


def _write_dataset(out_dir: Path) -> None:
    shape = (32, 32, 3)
    feats: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.images.cam": {
            "dtype": "video",
            "shape": list(shape),
            "names": ["height", "width", "channels"],
        },
    }
    writer = DatasetWriter(
        out_dir,
        {"robot_type": "regression"},
        feats,
        fps=10,
        video_codec="libx264",
    )
    rng = np.random.default_rng(0)
    for ep_len in (5, 7):
        for i in range(ep_len):
            frame: dict[str, Any] = {
                "observation.state": np.array([float(i), 0.0], dtype=np.float32),
                "action": np.array([float(i), 0.0], dtype=np.float32),
                "task": "t",
                "observation.images.cam": Image.fromarray(
                    rng.integers(0, 256, shape, dtype=np.uint8)
                ),
            }
            writer.add_frame(frame)
        writer.save_episode()
    writer.finalize()


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def server_ctx(tmp_path: Path):
    """Start a UI server on an ephemeral port; yield (conn_factory, roots)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    bags_root = tmp_path / "bags"
    output_root = tmp_path / "out"
    bags_root.mkdir()
    output_root.mkdir()
    # A non-bag subdir for browse, and a fake bag dir (mcap marker file).
    (bags_root / "group").mkdir()
    fake_bag = bags_root / "group" / "ep0"
    fake_bag.mkdir()
    (fake_bag / "ep0.mcap").write_bytes(b"\x00")
    (fake_bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")

    # A real synthetic dataset under the output root.
    ds = output_root / "ds"
    _write_dataset(ds)

    roots = [
        Root(id=0, label="bags", path=bags_root),
        Root(id=1, label="output", path=output_root),
    ]
    api = Api(roots=roots, token=TOKEN, registry=JobRegistry())
    static_dir = (
        Path(__file__).resolve().parent.parent / "src" / "bagel" / "ui" / "static"
    )
    server = make_server(api, static_dir, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def conn() -> HTTPConnection:
        return HTTPConnection("127.0.0.1", port)

    try:
        yield conn, roots
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(conn_factory, path: str, token: str | None = TOKEN) -> tuple[int, bytes, str]:
    c = conn_factory()
    headers = {"X-Bagel-Token": token} if token is not None else {}
    c.request("GET", path, headers=headers)
    resp = c.getresponse()
    body = resp.read()
    ctype = resp.getheader("Content-Type", "")
    c.close()
    return resp.status, body, ctype


def _post(
    conn_factory, path: str, payload: dict[str, Any], token: str | None = TOKEN
) -> tuple[int, dict[str, Any]]:
    c = conn_factory()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Bagel-Token"] = token
    c.request("POST", path, body=json.dumps(payload), headers=headers)
    resp = c.getresponse()
    data = json.loads(resp.read() or b"{}")
    c.close()
    return resp.status, data


# ---------------------------------------------------------------------------
# Auth + confinement
# ---------------------------------------------------------------------------


def test_missing_token_401(server_ctx) -> None:
    conn, _roots = server_ctx
    status, _body, _ctype = _get(conn, "/api/config", token=None)
    assert status == 401


def test_wrong_token_401(server_ctx) -> None:
    conn, _roots = server_ctx
    status, _body, _ctype = _get(conn, "/api/config", token="nope")
    assert status == 401


def test_browse_traversal_403(server_ctx) -> None:
    conn, _roots = server_ctx
    status, data = _post(conn, "/api/browse", {"root_id": 0, "subpath": "../etc"})
    assert status == 403
    assert data["error"] is True
    assert "detail" in data


def test_bag_under_output_root_rejected(server_ctx) -> None:
    # A path under --output-root (the dataset root) must not be accepted as a
    # bag input: bag args are confined to the bags roots only.
    conn, roots = server_ctx
    output_root = next(r.path for r in roots if r.label == "output")
    ds_under_output = str(output_root / "ds")
    status, data = _post(conn, "/api/inspect", {"bags": [ds_under_output]})
    assert status == 403, data
    assert data["error"] is True


def test_confine_bags_relative_and_absolute(tmp_path: Path) -> None:
    """_confine_bags accepts root-relative paths (what the FE sends from browse)
    and absolute paths under a bags root, and rejects output-root / traversal."""
    from bagel.ui.api import ApiError

    bags_root = tmp_path / "bags"
    out_root = tmp_path / "out"
    bag = bags_root / "group" / "ep0"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    out_root.mkdir()
    api = Api(
        roots=[
            Root(id=0, label="bags", path=bags_root),
            Root(id=1, label="output", path=out_root),
        ],
        token=TOKEN,
        registry=JobRegistry(),
    )

    # Root-relative path (the exact form the frontend sends) resolves correctly.
    rel = api._confine_bags(["group/ep0"])
    assert [p.resolve() for p in rel] == [bag.resolve()]
    # Absolute path under the bags root also works.
    ab = api._confine_bags([str(bag)])
    assert [p.resolve() for p in ab] == [bag.resolve()]
    # A path under the output root is not a valid bag.
    with pytest.raises(ApiError):
        api._confine_bags([str(out_root / "ds")])
    # Traversal is rejected.
    with pytest.raises(ApiError):
        api._confine_bags(["../escape"])
    # Empty selection is a 400.
    with pytest.raises(ApiError):
        api._confine_bags([])


def test_static_index_unauthenticated(server_ctx) -> None:
    conn, _roots = server_ctx
    status, body, ctype = _get(conn, "/", token=None)
    assert status == 200
    assert "text/html" in ctype
    assert b"bagel ui" in body


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_config(server_ctx) -> None:
    conn, _roots = server_ctx
    status, body, _ctype = _get(conn, "/api/config")
    assert status == 200
    data = json.loads(body)
    assert "command" in data
    assert data["bagel_version"]
    labels = {r["label"] for r in data["roots"]}
    assert labels == {"bags", "output"}


def test_browse(server_ctx) -> None:
    conn, _roots = server_ctx
    status, data = _post(conn, "/api/browse", {"root_id": 0, "subpath": "group"})
    assert status == 200
    assert "command" in data
    names = {e["name"] for e in data["entries"]}
    assert "ep0" in names
    ep0 = next(e for e in data["entries"] if e["name"] == "ep0")
    assert ep0["is_dir"] and ep0["is_bag"]
    assert any(b.endswith("/ep0") for b in data["bags"])


def test_validate_dataset(server_ctx) -> None:
    conn, _roots = server_ctx
    status, data = _post(conn, "/api/validate-dataset", {"dataset": "ds"})
    assert status == 200
    assert "command" in data
    assert data["report"]["verdict"] in ("OK", "FAIL")
    assert "issues" in data["report"]


def test_quality_report(server_ctx) -> None:
    conn, _roots = server_ctx
    status, data = _post(conn, "/api/quality-report", {"dataset": "ds"})
    assert status == 200
    assert "command" in data
    assert "score" in data["report"]
    assert "verdict" in data["report"]


def test_validate_dataset_missing_404(server_ctx) -> None:
    conn, _roots = server_ctx
    status, data = _post(conn, "/api/validate-dataset", {"dataset": "nope"})
    assert status == 404
    assert data["error"] is True


def test_preview_header_token(server_ctx) -> None:
    conn, _roots = server_ctx
    status, body, ctype = _get(conn, "/api/preview?dataset=ds")
    assert status == 200
    assert "text/html" in ctype
    assert b"<html" in body.lower()


def test_preview_query_token_ok(server_ctx) -> None:
    # The FE embeds preview in an <iframe>, which cannot set headers; the token
    # therefore rides in ?token= for this route only.
    conn, _roots = server_ctx
    status, body, ctype = _get(
        conn, f"/api/preview?dataset=ds&token={TOKEN}", token=None
    )
    assert status == 200
    assert "text/html" in ctype
    assert b"<html" in body.lower()


def test_preview_wrong_query_token_401(server_ctx) -> None:
    conn, _roots = server_ctx
    status, _body, _ctype = _get(conn, "/api/preview?dataset=ds&token=bad", token=None)
    assert status == 401


def test_convert_status_unknown_404(server_ctx) -> None:
    conn, _roots = server_ctx
    status, body, _ctype = _get(conn, "/api/convert/does-not-exist")
    assert status == 404
    data = json.loads(body)
    assert data["error"] is True


# ---------------------------------------------------------------------------
# convert (stubbed subprocess) — verifies job lifecycle without a real run
# ---------------------------------------------------------------------------


def test_convert_launch_and_poll(server_ctx, monkeypatch) -> None:
    import sys

    import bagel.ui.api as api_mod

    conn, roots = server_ctx

    # Stub the bagel subprocess to a trivial python one-liner that writes a
    # complete job_summary.json and exits 0 — exercising launch + status without
    # a real conversion.
    output_root = next(r.path for r in roots if r.label == "output")
    bag_path = output_root.parent / "bags" / "group" / "ep0"

    def fake_argv(self, *args: str) -> list[str]:
        # Find the --output dir we were asked to write into.
        out = args[args.index("--output") + 1]
        script = (
            "import json,os;"
            f"d=os.path.join({out!r},'meta');"
            "os.makedirs(d,exist_ok=True);"
            "open(os.path.join(d,'job_summary.json'),'w')"
            ".write(json.dumps({'n_success':1,'n_failed':0,'total_frames':3}))"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(api_mod.Api, "_bagel_argv", fake_argv)

    status, data = _post(
        conn,
        "/api/convert",
        {
            "config_yaml": "robot_type: x\nfps: 10\ntask: t\nobservations: []\nactions: []\n",
            "bags": [str(bag_path)],
            "output": "conv_out",
        },
    )
    assert status == 200, data
    assert "command" in data
    job_id = data["job_id"]

    # Poll until terminal (the stub exits ~immediately).
    final: dict[str, Any] = {}
    for _ in range(200):
        st, body, _ct = _get(conn, f"/api/convert/{job_id}")
        assert st == 200
        final = json.loads(body)
        if final["state"] in ("done", "failed"):
            break
    assert final["state"] == "done", final
    assert final["progress"]["total"] == 1
    assert final["progress"]["done"] == 1
    assert final["summary"]["n_success"] == 1

    # No leftover per-convert config tempfile under the output root, and no
    # leftover stderr temp file, after the job reaches a terminal state.
    output_root = next(r.path for r in roots if r.label == "output")
    assert not list(output_root.glob("bagel_ui_cfg_*")), "config tempfile leaked"
    import tempfile as _tempfile

    leftover_stderr = list(Path(_tempfile.gettempdir()).glob("bagel_ui_stderr_*"))
    assert not leftover_stderr, f"stderr tempfile leaked: {leftover_stderr}"


def test_convert_stderr_not_undrained_pipe(server_ctx, monkeypatch) -> None:
    # Regression for the stderr deadlock: a child that writes a large volume of
    # stderr must still complete (the registry redirects stderr to a file, not
    # an undrained PIPE that would block once the pipe buffer fills).
    import sys

    import bagel.ui.api as api_mod

    conn, roots = server_ctx
    output_root = next(r.path for r in roots if r.label == "output")
    bag_path = output_root.parent / "bags" / "group" / "ep0"

    def fake_argv(self, *args: str) -> list[str]:
        out = args[args.index("--output") + 1]
        # Emit ~1 MiB of stderr (far beyond a 64 KiB pipe buffer) then write a
        # complete summary and exit 0.
        script = (
            "import sys,json,os;"
            "sys.stderr.write('x'*1048576);sys.stderr.flush();"
            f"d=os.path.join({out!r},'meta');"
            "os.makedirs(d,exist_ok=True);"
            "open(os.path.join(d,'job_summary.json'),'w')"
            ".write(json.dumps({'n_success':1,'n_failed':0,'total_frames':3}))"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(api_mod.Api, "_bagel_argv", fake_argv)

    status, data = _post(
        conn,
        "/api/convert",
        {
            "config_yaml": "robot_type: x\nfps: 10\ntask: t\nobservations: []\nactions: []\n",
            "bags": [str(bag_path)],
            "output": "conv_out_stderr",
        },
    )
    assert status == 200, data
    job_id = data["job_id"]

    final: dict[str, Any] = {}
    for _ in range(400):
        st, body, _ct = _get(conn, f"/api/convert/{job_id}")
        assert st == 200
        final = json.loads(body)
        if final["state"] in ("done", "failed"):
            break
    assert final["state"] == "done", final
