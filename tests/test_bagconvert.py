"""Tests for the ROS1 .bag -> ROS2 MCAP conversion helpers (rosbag2lerobot.bagconvert).

These cover the pure path/detection helpers without invoking the (slow,
data-dependent) rosbags conversion itself.
"""

from __future__ import annotations

from pathlib import Path

from rosbag2lerobot.bagconvert import (
    discover_ros1_bags,
    is_ros1_bag,
    output_name,
)


def _write_ros1_bag(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#ROSBAG V2.0\n\x00\x01junk")
    return path


class TestIsRos1Bag:
    def test_detects_ros1_magic(self, tmp_path: Path) -> None:
        bag = _write_ros1_bag(tmp_path / "data.bag")
        assert is_ros1_bag(bag) is True

    def test_rejects_wrong_extension(self, tmp_path: Path) -> None:
        f = _write_ros1_bag(tmp_path / "data.mcap")
        assert is_ros1_bag(f) is False

    def test_rejects_non_ros1_content(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bag"
        f.write_bytes(b"not a rosbag")
        assert is_ros1_bag(f) is False

    def test_rejects_directory(self, tmp_path: Path) -> None:
        assert is_ros1_bag(tmp_path) is False


class TestOutputName:
    def test_generic_data_bag_uses_parent(self, tmp_path: Path) -> None:
        src = tmp_path / "235210" / "data.bag"
        assert output_name(src) == "235210"

    def test_named_bag_uses_stem(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings" / "recording_01.bag"
        assert output_name(src) == "recording_01"


class TestDiscoverRos1Bags:
    def test_finds_bags_recursively_and_skips_non_ros1(self, tmp_path: Path) -> None:
        _write_ros1_bag(tmp_path / "a" / "data.bag")
        _write_ros1_bag(tmp_path / "b" / "data.bag")
        # decoy: wrong magic, should be skipped
        decoy = tmp_path / "c" / "data.bag"
        decoy.parent.mkdir(parents=True)
        decoy.write_bytes(b"not a bag")

        found = discover_ros1_bags([tmp_path])
        names = sorted(p.parent.name for p in found)
        assert names == ["a", "b"]

    def test_accepts_explicit_file(self, tmp_path: Path) -> None:
        bag = _write_ros1_bag(tmp_path / "x" / "data.bag")
        found = discover_ros1_bags([bag])
        assert found == [bag]

    def test_deduplicates(self, tmp_path: Path) -> None:
        bag = _write_ros1_bag(tmp_path / "x" / "data.bag")
        found = discover_ros1_bags([bag, tmp_path])
        assert len(found) == 1
