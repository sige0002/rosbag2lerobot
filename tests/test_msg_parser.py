"""Tests for .msg file parsing.

The msg_parser reads ROS2 .msg definition files and produces structured
representations used by the generic decoder fallback.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight msg parser (to be tested)
# ---------------------------------------------------------------------------
# We define a minimal parser here so the tests are self-contained.
# When the real msg_parser module exists in the source tree, these tests
# should be redirected to import from there.


@dataclass
class MsgField:
    """A single field in a .msg definition."""

    type_name: str  # e.g. "float64", "string", "geometry_msgs/Pose"
    field_name: str  # e.g. "joint_positions"
    is_array: bool = False
    array_size: int | None = None  # None for variable-length, int for fixed
    default_value: Any = None
    is_constant: bool = False
    constant_value: Any = None


@dataclass
class MsgDefinition:
    """Parsed .msg file."""

    package: str
    name: str
    fields: list[MsgField] = field(default_factory=list)
    constants: list[MsgField] = field(default_factory=list)


def parse_msg_text(text: str, package: str = "", name: str = "") -> MsgDefinition:
    """Parse a .msg definition from text content.

    Supports:
    - Primitive types: bool, int8, uint8, int16, uint16, int32, uint32,
      int64, uint64, float32, float64, string, byte, char
    - Fixed-length arrays: float64[7]
    - Variable-length arrays: float64[]
    - Constants: uint8 FOO=42
    - Comments: lines starting with # (and inline # comments)
    - Nested types: geometry_msgs/Pose
    """
    fields: list[MsgField] = []
    constants: list[MsgField] = []

    for raw_line in text.strip().splitlines():
        # Strip inline comments
        line = raw_line.split("#")[0].strip()
        if not line:
            continue

        # Check for constant: TYPE NAME=VALUE
        if "=" in line:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            type_name = parts[0]
            name_val = parts[1]
            eq_idx = name_val.index("=")
            const_name = name_val[:eq_idx].strip()
            const_val = name_val[eq_idx + 1 :].strip()
            constants.append(
                MsgField(
                    type_name=type_name,
                    field_name=const_name,
                    is_constant=True,
                    constant_value=const_val,
                )
            )
            continue

        # Regular field: TYPE NAME
        parts = line.split()
        if len(parts) < 2:
            continue
        type_str = parts[0]
        field_name_str = parts[1]

        # Check for arrays
        is_array = False
        array_size: int | None = None
        if "[" in type_str:
            is_array = True
            bracket_start = type_str.index("[")
            bracket_end = type_str.index("]")
            size_str = type_str[bracket_start + 1 : bracket_end]
            if size_str:
                array_size = int(size_str)
            type_str = type_str[:bracket_start]

        fields.append(
            MsgField(
                type_name=type_str,
                field_name=field_name_str,
                is_array=is_array,
                array_size=array_size,
            )
        )

    return MsgDefinition(
        package=package,
        name=name,
        fields=fields,
        constants=constants,
    )


def parse_msg_file(path: str | Path) -> MsgDefinition:
    """Parse a .msg file from disk.

    Infers the message name from the filename (e.g. MyMsg.msg -> MyMsg).
    """
    p = Path(path)
    text = p.read_text()
    msg_name = p.stem
    return parse_msg_text(text, name=msg_name)


# ---------------------------------------------------------------------------
# Tests: simple .msg parsing
# ---------------------------------------------------------------------------


class TestParseSimpleMsg:
    def test_single_field(self) -> None:
        text = "float32 value"
        defn = parse_msg_text(text)
        assert len(defn.fields) == 1
        assert defn.fields[0].type_name == "float32"
        assert defn.fields[0].field_name == "value"
        assert defn.fields[0].is_array is False

    def test_multiple_fields(self) -> None:
        text = textwrap.dedent("""\
            float64 x
            float64 y
            float64 z
        """)
        defn = parse_msg_text(text)
        assert len(defn.fields) == 3
        assert [f.field_name for f in defn.fields] == ["x", "y", "z"]

    def test_string_field(self) -> None:
        text = "string name"
        defn = parse_msg_text(text)
        assert defn.fields[0].type_name == "string"

    def test_bool_field(self) -> None:
        text = "bool is_active"
        defn = parse_msg_text(text)
        assert defn.fields[0].type_name == "bool"

    def test_uint8_field(self) -> None:
        text = "uint8 status"
        defn = parse_msg_text(text)
        assert defn.fields[0].type_name == "uint8"


# ---------------------------------------------------------------------------
# Tests: arrays
# ---------------------------------------------------------------------------


class TestParseArrays:
    def test_fixed_length_array(self) -> None:
        text = "float64[7] joint_positions"
        defn = parse_msg_text(text)
        f = defn.fields[0]
        assert f.is_array is True
        assert f.array_size == 7
        assert f.type_name == "float64"
        assert f.field_name == "joint_positions"

    def test_variable_length_array(self) -> None:
        text = "float64[] values"
        defn = parse_msg_text(text)
        f = defn.fields[0]
        assert f.is_array is True
        assert f.array_size is None
        assert f.type_name == "float64"

    def test_fixed_and_variable_mixed(self) -> None:
        text = textwrap.dedent("""\
            float64[3] position
            float64[] extra_data
        """)
        defn = parse_msg_text(text)
        assert defn.fields[0].array_size == 3
        assert defn.fields[1].array_size is None

    def test_uint8_array(self) -> None:
        text = "uint8[100] data"
        defn = parse_msg_text(text)
        assert defn.fields[0].array_size == 100
        assert defn.fields[0].type_name == "uint8"


# ---------------------------------------------------------------------------
# Tests: nested types
# ---------------------------------------------------------------------------


class TestParseNestedTypes:
    def test_nested_type(self) -> None:
        text = "geometry_msgs/Pose pose"
        defn = parse_msg_text(text)
        assert defn.fields[0].type_name == "geometry_msgs/Pose"
        assert defn.fields[0].field_name == "pose"

    def test_std_msgs_header(self) -> None:
        text = "std_msgs/Header header"
        defn = parse_msg_text(text)
        assert defn.fields[0].type_name == "std_msgs/Header"

    def test_nested_array(self) -> None:
        text = "geometry_msgs/Point[3] vertices"
        defn = parse_msg_text(text)
        f = defn.fields[0]
        assert f.type_name == "geometry_msgs/Point"
        assert f.is_array is True
        assert f.array_size == 3


# ---------------------------------------------------------------------------
# Tests: comments and constants
# ---------------------------------------------------------------------------


class TestParseCommentsAndConstants:
    def test_comment_lines_skipped(self) -> None:
        text = textwrap.dedent("""\
            # This is a comment
            float32 value
            # Another comment
        """)
        defn = parse_msg_text(text)
        assert len(defn.fields) == 1

    def test_inline_comment(self) -> None:
        text = "float32 value  # inline comment"
        defn = parse_msg_text(text)
        assert len(defn.fields) == 1
        assert defn.fields[0].field_name == "value"

    def test_empty_lines_skipped(self) -> None:
        text = "\n\nfloat32 value\n\n"
        defn = parse_msg_text(text)
        assert len(defn.fields) == 1

    def test_constant_definition(self) -> None:
        text = "uint8 STATUS_OK=0"
        defn = parse_msg_text(text)
        assert len(defn.fields) == 0
        assert len(defn.constants) == 1
        assert defn.constants[0].field_name == "STATUS_OK"
        assert defn.constants[0].constant_value == "0"
        assert defn.constants[0].is_constant is True

    def test_constant_with_spaces(self) -> None:
        text = "string FRAME_ID=base_link"
        defn = parse_msg_text(text)
        assert len(defn.constants) == 1
        assert defn.constants[0].constant_value == "base_link"

    def test_mixed_fields_and_constants(self) -> None:
        text = textwrap.dedent("""\
            uint8 MODE_IDLE=0
            uint8 MODE_ACTIVE=1
            uint8 mode
            float32 value
        """)
        defn = parse_msg_text(text)
        assert len(defn.constants) == 2
        assert len(defn.fields) == 2
