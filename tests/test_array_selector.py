"""Tests for the array-selector bracket notation (``field[idx]``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class TestParseArraySelector:
    """Unit tests for ``_parse_array_selector``."""

    def test_no_brackets_returns_none_index(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        assert _parse_array_selector("pos") == ("pos", None)

    def test_positive_index(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        assert _parse_array_selector("pos[0]") == ("pos", 0)
        assert _parse_array_selector("speed[11]") == ("speed", 11)

    def test_negative_index(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        assert _parse_array_selector("pos[-1]") == ("pos", -1)

    def test_invalid_index_string(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        with pytest.raises(ValueError, match="Invalid array index"):
            _parse_array_selector("pos[abc]")

    def test_missing_closing_bracket(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        with pytest.raises(ValueError, match="missing ']'"):
            _parse_array_selector("pos[0")

    def test_empty_field_name(self) -> None:
        from rosbag2lerobot.decoders.builtin import _parse_array_selector

        with pytest.raises(ValueError, match="empty field name"):
            _parse_array_selector("[0]")


class TestDecodeJointStateBracket:
    """Bracket notation on ``sensor_msgs/msg/JointState``."""

    def _make(self) -> SimpleNamespace:
        return SimpleNamespace(
            name=["shoulder", "elbow", "wrist"],
            position=[0.1, 0.2, 0.3],
            velocity=[1.0, 2.0, 3.0],
            effort=[0.0, 0.0, 0.0],
        )

    def test_bracket_position(self) -> None:
        from rosbag2lerobot.decoders.builtin import decode_joint_state

        result = decode_joint_state(self._make(), ["position[0]"], {})
        np.testing.assert_allclose(result, [0.1])

    def test_bracket_velocity_negative(self) -> None:
        from rosbag2lerobot.decoders.builtin import decode_joint_state

        result = decode_joint_state(self._make(), ["velocity[-1]"], {})
        np.testing.assert_allclose(result, [3.0])

    def test_name_based_still_works(self) -> None:
        from rosbag2lerobot.decoders.builtin import decode_joint_state

        result = decode_joint_state(
            self._make(), ["position.elbow", "position.wrist"], {}
        )
        np.testing.assert_allclose(result, [0.2, 0.3])

    def test_wildcard_still_works(self) -> None:
        from rosbag2lerobot.decoders.builtin import decode_joint_state

        result = decode_joint_state(self._make(), ["position.*"], {})
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    def test_bracket_out_of_bounds_raises(self) -> None:
        from rosbag2lerobot.decoders.builtin import decode_joint_state

        with pytest.raises(ValueError, match="index 99"):
            decode_joint_state(self._make(), ["position[99]"], {})


class TestMsgDecoderBracketSelector:
    """Bracket notation on the generic ``.msg`` decoder path.

    Custom types without a built-in decoder (e.g.
    ``rm_ros_interface/msg/RmPlusState`` with a ``pos`` array) go through
    ``MsgDecoder._get_nested_value``, which must accept ``pos[0]`` identically
    to the built-in decoders while keeping the legacy ``pos.0`` dot form.
    """

    def _make(self) -> SimpleNamespace:
        return SimpleNamespace(
            pos=[10.0, 20.0, 30.0],
            state=SimpleNamespace(pos=[1.0, 2.0, 3.0]),
        )

    def test_bracket_index(self) -> None:
        from rosbag2lerobot.decoders.msg_parser import MsgDecoder

        assert MsgDecoder._get_nested_value(self._make(), "pos[0]") == 10.0
        assert MsgDecoder._get_nested_value(self._make(), "pos[2]") == 30.0

    def test_bracket_negative(self) -> None:
        from rosbag2lerobot.decoders.msg_parser import MsgDecoder

        assert MsgDecoder._get_nested_value(self._make(), "pos[-1]") == 30.0

    def test_dot_index_still_works(self) -> None:
        from rosbag2lerobot.decoders.msg_parser import MsgDecoder

        assert MsgDecoder._get_nested_value(self._make(), "pos.0") == 10.0

    def test_whole_field_unchanged(self) -> None:
        from rosbag2lerobot.decoders.msg_parser import MsgDecoder

        assert MsgDecoder._get_nested_value(self._make(), "pos") == [10.0, 20.0, 30.0]

    def test_nested_then_bracket(self) -> None:
        from rosbag2lerobot.decoders.msg_parser import MsgDecoder

        assert MsgDecoder._get_nested_value(self._make(), "state.pos[1]") == 2.0
