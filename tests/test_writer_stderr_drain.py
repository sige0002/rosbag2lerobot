"""Regression tests for the encoder's stderr handling.

The streaming writer opens one long-lived ffmpeg per camera with
``stderr=subprocess.PIPE``. ffmpeg writes to that pipe for its whole life,
so it must be drained continuously: with nobody reading, the OS pipe buffer
(typically 64 KiB) fills and ffmpeg blocks inside ``write()`` forever. The
symptom is a conversion that stops making progress at 0% CPU — easy to
misread as an OOM kill, but really a subprocess pipe deadlock.

These tests swap the ffmpeg argv for a small Python stub, so they exercise
the writer's pipe plumbing against *real* pipes without needing ffmpeg (and
without waiting for a real encode). Driving the encoder through its private
methods follows the precedent set by :mod:`tests.test_audit`: it keeps the
test pointed at the pipe machinery rather than at parquet/stats bookkeeping.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from rosbag2lerobot import writer as writer_mod
from rosbag2lerobot.writer import _STDERR_TAIL_MAXLEN, DatasetWriter, _StderrTail

VKEY = "observation.images.cam"

# 64x64 RGB frames: 12 KiB each. 40 of them is ~480 KiB, comfortably past
# any plausible pipe buffer, so a stub that never reads stdin is guaranteed
# to block the feeder rather than silently absorbing everything.
_FRAME_BYTES = b"\x00" * (64 * 64 * 3)
_N_FRAMES = 40

# Generous: the stub does its work in milliseconds. This bound only has to
# tell "finished" apart from "deadlocked".
_DEADLOCK_TIMEOUT = 30.0

# Writes 4 MiB to stderr *before* reading stdin. Undrained, it blocks on its
# own stderr; the writer's feeder then blocks on a full stdin pipe, and
# neither side can move — the exact deadlock this module guards against.
_FLOODING_STUB = """\
import sys

sys.stderr.buffer.write(b"E" * (4 * 1024 * 1024))
sys.stderr.buffer.flush()
sys.stdin.buffer.read()
sys.stderr.buffer.write(b"stub-done\\n")
sys.stderr.buffer.flush()
"""

# Dies immediately with a diagnostic, like ffmpeg rejecting its arguments.
_FAILING_STUB = """\
import sys

sys.stderr.buffer.write(b"stub failure: unknown encoder\\n")
sys.stderr.buffer.flush()
sys.exit(3)
"""


def _install_stub_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> list[list[str]]:
    """Run ``body`` instead of ffmpeg; return the list of argvs the writer built.

    The stub is launched through the real ``Popen`` with the writer's own
    pipe configuration, so stdin/stdout/stderr behave exactly as they do in
    production.

    The patch lands on the stdlib ``subprocess`` module (writer.py imports the
    module, not the name), so it is visible process-wide for the duration of
    the test. Only ffmpeg invocations are redirected — anything else is passed
    through to the real ``Popen`` untouched, so an unrelated subprocess cannot
    silently end up running the stub.
    """
    stub = tmp_path / "stub_encoder.py"
    stub.write_text(body)
    real_popen = subprocess.Popen
    seen: list[list[str]] = []

    def _fake_popen(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        argv = [cmd] if isinstance(cmd, str) else list(cmd)
        # Match on the basename so resolving ffmpeg to an absolute path in the
        # writer does not silently de-stub this suite.
        if not argv or Path(str(argv[0])).name != "ffmpeg":
            return real_popen(cmd, *args, **kwargs)
        seen.append(argv)
        return real_popen([sys.executable, str(stub)], *args, **kwargs)

    monkeypatch.setattr(writer_mod.subprocess, "Popen", _fake_popen)
    return seen


@pytest.fixture
def features_with_video() -> dict:
    """One camera plus the mandatory numeric columns.

    Duplicated from :mod:`tests.test_writer` (as in :mod:`tests.test_audit`)
    so this module runs standalone.
    """
    return {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": {"axes": ["j1", "j2"]},
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": {"axes": ["j1", "j2"]},
        },
        VKEY: {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
        },
    }


@pytest.fixture
def writer(tmp_path: Path, features_with_video: dict) -> DatasetWriter:
    return DatasetWriter(
        tmp_path / "out", {"robot_type": "cam_robot"}, features_with_video, fps=10
    )


class TestStderrTail:
    """The retained tail is bounded and keeps the *end* of the stream."""

    def test_short_output_is_kept_whole(self) -> None:
        tail = _StderrTail()
        tail.append(b"unknown encoder 'libsvtav1'\n")
        assert tail.text() == "unknown encoder 'libsvtav1'\n"

    def test_empty_tail_is_empty_string(self) -> None:
        assert _StderrTail().text() == ""

    def test_long_output_is_truncated_to_the_last_bytes(self) -> None:
        tail = _StderrTail(maxlen=64)
        tail.append(b"x" * 1000)
        tail.append(b"the last line")
        text = tail.text()
        assert len(text) == 64
        assert text.endswith("the last line")

    def test_default_bound_holds_across_many_appends(self) -> None:
        """A multi-hour, multi-episode run must not grow this buffer."""
        tail = _StderrTail()
        for _ in range(200):
            tail.append(b"y" * 4096)  # 800 KiB total
        assert len(tail.text()) == _STDERR_TAIL_MAXLEN


class TestEncoderCommand:
    def test_ffmpeg_is_started_quietly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer: DatasetWriter
    ) -> None:
        """Banner and per-frame progress are suppressed at the source.

        Draining makes the deadlock impossible; these flags keep the volume
        that has to be drained near zero in the first place.
        """
        seen = _install_stub_encoder(monkeypatch, tmp_path, _FAILING_STUB)
        writer._ensure_encoder(VKEY, 64, 64)
        try:
            (cmd,) = seen
            assert cmd[0] == "ffmpeg"
            assert "-hide_banner" in cmd
            assert "-nostats" in cmd
            assert cmd[cmd.index("-loglevel") + 1] == "warning"
        finally:
            with pytest.raises(RuntimeError):
                writer._close_video_encoder(VKEY)


class TestStderrDeadlock:
    def test_encoder_survives_an_encoder_that_floods_stderr(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer: DatasetWriter
    ) -> None:
        """4 MiB of child stderr must not stall the frame feeder.

        Without a drain thread this hangs forever, which is the bug this
        test exists for; it is driven from a worker thread so the failure
        shows up as a timeout instead of a hung test session.
        """
        _install_stub_encoder(monkeypatch, tmp_path, _FLOODING_STUB)

        done = threading.Event()
        failure: list[BaseException] = []

        def _drive() -> None:
            try:
                writer._ensure_encoder(VKEY, 64, 64)
                for _ in range(_N_FRAMES):
                    writer._image_feed_queues[VKEY].put(_FRAME_BYTES)
                writer._drain_video_queue(VKEY)
                writer._close_video_encoder(VKEY)
            except BaseException as exc:  # noqa: BLE001 - reported by the assert below
                failure.append(exc)
            finally:
                done.set()

        threading.Thread(target=_drive, name="drive-encoder", daemon=True).start()
        try:
            finished = done.wait(timeout=_DEADLOCK_TIMEOUT)
            assert finished, (
                "the encoder deadlocked: its stderr is not being drained, so it "
                "blocked in write() once the pipe buffer filled"
            )
            assert not failure, f"driving the encoder failed: {failure[0]!r}"
        finally:
            # On a deadlock the stub is still alive holding both pipes; reap it
            # so the failing run does not leave a zombie behind.
            for proc in list(writer._image_encoders.values()):
                proc.kill()
                proc.wait(timeout=10)


class TestFailureDiagnostics:
    """Switching to a drained tail must not cost us the error message."""

    def test_close_reports_exit_code_and_stderr(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer: DatasetWriter
    ) -> None:
        _install_stub_encoder(monkeypatch, tmp_path, _FAILING_STUB)
        writer._ensure_encoder(VKEY, 64, 64)

        with pytest.raises(RuntimeError) as excinfo:
            writer._close_video_encoder(VKEY)
        assert "code 3" in str(excinfo.value)
        assert "stub failure: unknown encoder" in str(excinfo.value)

    def test_drain_reports_stderr_when_the_encoder_dies_mid_episode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer: DatasetWriter
    ) -> None:
        """A dead encoder surfaces as its own stderr, not as ``BrokenPipeError``."""
        _install_stub_encoder(monkeypatch, tmp_path, _FAILING_STUB)
        writer._ensure_encoder(VKEY, 64, 64)
        for _ in range(_N_FRAMES):
            writer._image_feed_queues[VKEY].put(_FRAME_BYTES)

        with pytest.raises(RuntimeError) as excinfo:
            writer._drain_video_queue(VKEY)
        assert "stub failure: unknown encoder" in str(excinfo.value)

    def test_encoder_state_is_cleared_after_a_clean_close(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer: DatasetWriter
    ) -> None:
        """No per-encoder bookkeeping may leak across output files."""
        _install_stub_encoder(monkeypatch, tmp_path, "import sys; sys.stdin.read()")
        writer._ensure_encoder(VKEY, 64, 64)
        writer._close_video_encoder(VKEY)

        assert VKEY not in writer._image_encoders
        assert VKEY not in writer._image_stderr_readers
        assert VKEY not in writer._image_stderr_tails
