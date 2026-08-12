"""Tests for NVENC detection and ``--video-codec auto`` fallback.

``ffmpeg -encoders`` only proves NVENC was *compiled in*. A container
started without the NVIDIA runtime still lists ``h264_nvenc`` and then dies
at the first frame with ``Cannot load libcuda.so.1``, so detection also runs
a one-frame test encode and believes that instead.

Covered here:
    * the encoder listing gates the probe (no NVENC listed -> no probe run);
    * a listed-but-unusable encoder is reported as unavailable, with the
      reason logged once;
    * ffmpeg missing / timing out is swallowed at both steps;
    * the verdict is cached, so the probe runs once per process;
    * ``--video-codec auto`` falls back to libx264 when the probe fails,
      ``--gpu`` fails fast with an actionable message, and ``--no-gpu``
      never probes at all.

All subprocess interaction is mocked — these tests do not require ffmpeg,
let alone a GPU, on the test host.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from click.testing import CliRunner

from rosbag2lerobot.cli import _detect_nvenc, main
from rosbag2lerobot.cli._common import _nvenc_probe_error

from .conftest import tiny_config_yaml

ENCODER_LIST = (
    "Encoders:\n"
    " V..... libx264              libx264 H.264 / AVC\n"
    " V..... h264_nvenc           NVIDIA NVENC H.264 encoder\n"
    " V..... hevc_nvenc           NVIDIA NVENC hevc encoder\n"
)
CPU_ONLY_LIST = "Encoders:\n V..... libx264              libx264 H.264 / AVC\n"

# The failure this whole mechanism exists for, as ffmpeg actually reports it
# inside a container started without --gpus: root cause first, then six lines
# of cascade ending on something useless.
LIBCUDA_ERROR = (
    "[h264_nvenc @ 0x55d0] Cannot load libcuda.so.1\n"
    "[vost#0:0/h264_nvenc @ 0x55d1] Error while opening encoder - maybe "
    "incorrect parameters such as bit_rate, rate, width or height.\n"
    "[vost#0:0/h264_nvenc @ 0x55d1] Could not open encoder before EOF\n"
    "[out#0/null @ 0x55d2] Nothing was written into output file, because at "
    "least one of its streams received no packets.\n"
)


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    """Drop the cached verdict around every test (it is per-process state)."""
    _detect_nvenc.cache_clear()
    yield
    _detect_nvenc.cache_clear()


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _fake_ffmpeg(list_stdout: str, probe: Any):
    """Return a subprocess.run double: encoder listing, then the probe.

    *probe* is either a CompletedProcess to return or an exception to raise.
    """

    def _run(cmd: list[str], **_kwargs: Any) -> Any:
        if "-encoders" in cmd:
            return _result(stdout=list_stdout)
        if isinstance(probe, BaseException):
            raise probe
        return probe

    return _run


class TestDetectNvenc:
    def test_usable_nvenc_is_detected(self) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=_fake_ffmpeg(ENCODER_LIST, _result(returncode=0)),
        ):
            assert _detect_nvenc() is True

    def test_listed_but_unusable_nvenc_is_rejected(self, caplog) -> None:
        """The container case: the encoder exists, the driver does not."""
        with caplog.at_level(logging.WARNING, logger="rosbag2lerobot"):
            with mock.patch(
                "rosbag2lerobot.cli._common.subprocess.run",
                side_effect=_fake_ffmpeg(
                    ENCODER_LIST, _result(returncode=255, stderr=LIBCUDA_ERROR)
                ),
            ):
                assert _detect_nvenc() is False

        # The fallback is explained, in ffmpeg's own words, exactly once — and
        # with the root cause, not the cascade it ends on.
        assert "cannot encode here" in caplog.text
        assert "Cannot load libcuda.so.1" in caplog.text
        assert "Nothing was written into output file" not in caplog.text
        assert caplog.text.count("cannot encode here") == 1

    def test_no_nvenc_listed_skips_the_probe(self) -> None:
        """Machines that never had NVENC must not pay for a test encode."""
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_kwargs: Any) -> Any:
            calls.append(cmd)
            return _result(stdout=CPU_ONLY_LIST)

        with mock.patch("rosbag2lerobot.cli._common.subprocess.run", side_effect=_run):
            assert _detect_nvenc() is False
        assert len(calls) == 1
        assert "-encoders" in calls[0]

    def test_verdict_is_cached(self) -> None:
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_kwargs: Any) -> Any:
            calls.append(cmd)
            if "-encoders" in cmd:
                return _result(stdout=ENCODER_LIST)
            return _result(returncode=0)

        with mock.patch("rosbag2lerobot.cli._common.subprocess.run", side_effect=_run):
            assert _detect_nvenc() is True
            assert _detect_nvenc() is True
            assert _detect_nvenc() is True
        assert len(calls) == 2  # one listing + one probe, not three of each

    def test_returns_false_when_ffmpeg_missing(self) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg not on PATH"),
        ):
            assert _detect_nvenc() is False

    def test_returns_false_when_the_listing_times_out(self) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
        ):
            assert _detect_nvenc() is False

    def test_returns_false_when_the_probe_times_out(self) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=_fake_ffmpeg(
                ENCODER_LIST, subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)
            ),
        ):
            assert _detect_nvenc() is False

    def test_detection_does_not_steal_tty(self) -> None:
        """Neither ffmpeg call may read from the controlling terminal."""
        seen: list[tuple[list[str], Any]] = []

        def _run(cmd: list[str], **kwargs: Any) -> Any:
            seen.append((cmd, kwargs.get("stdin")))
            if "-encoders" in cmd:
                return _result(stdout=ENCODER_LIST)
            return _result(returncode=0)

        with mock.patch("rosbag2lerobot.cli._common.subprocess.run", side_effect=_run):
            _detect_nvenc()

        assert len(seen) == 2
        for cmd, stdin in seen:
            assert "-nostdin" in cmd
            assert stdin is subprocess.DEVNULL


class TestProbeCommand:
    def test_probe_frame_is_large_enough_for_nvenc(self) -> None:
        """NVENC rejects frames below its minimum dimension, so a too-small
        probe would report a healthy GPU as broken (128x128 already does)."""
        captured: list[list[str]] = []

        def _run(cmd: list[str], **_kwargs: Any) -> Any:
            captured.append(cmd)
            return _result(returncode=0)

        with mock.patch("rosbag2lerobot.cli._common.subprocess.run", side_effect=_run):
            assert _nvenc_probe_error() is None

        cmd = captured[0]
        size = next(a for a in cmd if a.startswith("color=")).split("s=")[1]
        width, height = (int(v) for v in size.split(":")[0].split("x"))
        assert width >= 256 and height >= 256
        # One frame, encoded by NVENC, written nowhere.
        assert "h264_nvenc" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "1"
        assert cmd[-3:] == ["-f", "null", "-"]

    def test_probe_reports_the_root_cause_not_the_cascade(self) -> None:
        """ffmpeg prints the real reason first and generic follow-ups after."""
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            return_value=_result(returncode=255, stderr=LIBCUDA_ERROR),
        ):
            reason = _nvenc_probe_error()
        assert reason == "[h264_nvenc @ 0x55d0] Cannot load libcuda.so.1"

    def test_probe_reports_exit_code_when_stderr_is_empty(self) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            return_value=_result(returncode=234, stderr=""),
        ):
            assert _nvenc_probe_error() == "ffmpeg exited 234"


class TestAutoCodecSelection:
    """CLI-level: which codec a run actually ends up using."""

    def _convert(self, tmp_path: Path, tiny_bag, *extra: str):
        tiny_bag(name="bags/ep0")
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")
        return CliRunner().invoke(
            main,
            [
                "convert",
                "--config",
                str(cfg_path),
                "--bags",
                str(tmp_path / "bags"),
                "--output",
                str(tmp_path / "out"),
                *extra,
            ],
        )

    def test_auto_falls_back_to_cpu_when_the_probe_fails(
        self, tmp_path: Path, tiny_bag
    ) -> None:
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=_fake_ffmpeg(
                ENCODER_LIST, _result(returncode=255, stderr=LIBCUDA_ERROR)
            ),
        ):
            result = self._convert(tmp_path, tiny_bag)

        assert result.exit_code == 0, result.output
        log = (tmp_path / "out" / "meta" / "conversion_log.json").read_text()
        assert '"codec": "libx264"' in log

    def test_auto_uses_nvenc_when_the_probe_succeeds(
        self, tmp_path: Path, tiny_bag
    ) -> None:
        """The healthy path is unchanged: a working NVENC is still chosen."""
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=_fake_ffmpeg(ENCODER_LIST, _result(returncode=0)),
        ):
            result = self._convert(tmp_path, tiny_bag)

        assert result.exit_code == 0, result.output
        log = (tmp_path / "out" / "meta" / "conversion_log.json").read_text()
        assert '"codec": "h264_nvenc"' in log

    def test_gpu_flag_fails_fast_when_nvenc_cannot_encode(
        self, tmp_path: Path, tiny_bag
    ) -> None:
        """Better to stop at startup than to die on the first frame."""
        with mock.patch(
            "rosbag2lerobot.cli._common.subprocess.run",
            side_effect=_fake_ffmpeg(
                ENCODER_LIST, _result(returncode=255, stderr=LIBCUDA_ERROR)
            ),
        ):
            result = self._convert(tmp_path, tiny_bag, "--gpu")

        assert result.exit_code != 0
        assert "NVENC cannot encode here" in result.output
        assert "--gpus all" in result.output

    def test_no_gpu_never_probes(self, tmp_path: Path, tiny_bag) -> None:
        """--no-gpu has already decided; probing would only waste a subprocess
        and could warn about a GPU the user just said not to use."""
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_kwargs: Any) -> Any:
            calls.append(cmd)
            return _result(stdout=ENCODER_LIST)

        with mock.patch("rosbag2lerobot.cli._common.subprocess.run", side_effect=_run):
            result = self._convert(tmp_path, tiny_bag, "--no-gpu")

        assert result.exit_code == 0, result.output
        assert not any("-encoders" in c or "lavfi" in c for c in calls)
