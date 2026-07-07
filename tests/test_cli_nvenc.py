"""Skeleton tests for NVENC detection and ``--video-codec auto`` fallback.

This file is a **test scaffold** produced by the cli worker. The
full assertions (e.g. --video-codec auto CLI-level fallback via
``CliRunner``) will be fleshed out by ``worker-test``.

Covered here:
    * ``_detect_nvenc`` → True when ``ffmpeg -encoders`` output
      contains an NVENC encoder name.
    * ``_detect_nvenc`` → False when ``ffmpeg`` binary is absent
      (``FileNotFoundError`` from :func:`subprocess.run`).
    * ``_detect_nvenc`` → False when the ``ffmpeg`` call times out
      (:class:`subprocess.TimeoutExpired`).

All subprocess interaction is mocked — these tests do not require
a real ffmpeg on the test host.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest import mock

import pytest

from rosbag2lerobot.cli import _detect_nvenc


class TestDetectNvenc:
    """Unit tests for :func:`rosbag2lerobot.cli._detect_nvenc`."""

    def test_returns_true_when_nvenc_present(self) -> None:
        """ffmpeg stdout advertises h264_nvenc → detector must return True."""
        fake_stdout = (
            "Encoders:\n"
            " V..... libx264              libx264 H.264 / AVC\n"
            " V..... h264_nvenc           NVIDIA NVENC H.264 encoder\n"
            " V..... hevc_nvenc           NVIDIA NVENC hevc encoder\n"
        )
        fake_result = mock.MagicMock()
        fake_result.stdout = fake_stdout
        with mock.patch(
            "rosbag2lerobot.cli.subprocess.run",
            return_value=fake_result,
        ) as run_mock:
            assert _detect_nvenc() is True
        # subprocess.run is called with ffmpeg -hide_banner -encoders.
        args, _kwargs = run_mock.call_args
        assert args[0][0] == "ffmpeg"
        assert "-encoders" in args[0]

    def test_detection_does_not_steal_tty(self) -> None:
        """ffmpeg must run with ``-nostdin`` and ``stdin=DEVNULL`` so it can
        never read from / block on the controlling terminal (which would
        leave the user's shell unusable)."""
        fake_result = mock.MagicMock()
        fake_result.stdout = "Encoders:\n V..... libx264 libx264 H.264 / AVC\n"
        with mock.patch(
            "rosbag2lerobot.cli.subprocess.run",
            return_value=fake_result,
        ) as run_mock:
            _detect_nvenc()
        args, kwargs = run_mock.call_args
        assert "-nostdin" in args[0]
        assert kwargs["stdin"] is subprocess.DEVNULL

    def test_returns_false_when_ffmpeg_missing(self) -> None:
        """FileNotFoundError from subprocess.run must be swallowed."""

        def _raise_missing(*_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("ffmpeg not on PATH")

        with mock.patch(
            "rosbag2lerobot.cli.subprocess.run",
            side_effect=_raise_missing,
        ):
            assert _detect_nvenc() is False

    def test_returns_false_on_timeout(self) -> None:
        """TimeoutExpired from subprocess.run must be swallowed."""

        def _raise_timeout(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)

        with mock.patch(
            "rosbag2lerobot.cli.subprocess.run",
            side_effect=_raise_timeout,
        ):
            assert _detect_nvenc() is False


# TODO (worker-test):
# - Add CliRunner-based tests for ``--video-codec auto`` that patch
#   ``rosbag2lerobot.cli._detect_nvenc`` to force each branch
#   (auto→h264_nvenc, auto→libx264, --gpu without NVENC raises
#   click.UsageError, --no-gpu with explicit _nvenc codec raises
#   click.UsageError, --gpu with non-NVENC codec logs warning).
# - Once integrated, remove this TODO.
pytest.importorskip("click")  # sanity: CLI tests depend on click
