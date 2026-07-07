"""Generic ``.msg`` file parser and decoder for ROS2 message types.

Parses ``.msg`` file definitions and creates generic decoders that
extract numeric fields from deserialized rosbags message objects.

Main classes:

- ``MsgParser``      -- Reads and parses ``.msg`` files into
                        ``MsgDefinition`` objects.
- ``MsgDecoder``     -- Uses a ``MsgDefinition`` to decode a deserialized
                        message into a ``numpy`` array.
- ``FieldDefinition``-- Metadata for a single field in a ``.msg`` file.
- ``MsgDefinition``  -- Collection of ``FieldDefinition`` entries for one
                        message type.

``MsgParser.parse_from_type()`` searches configured paths and the
project ``msgs/`` directory for a matching ``.msg`` file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ROS2 primitive types and their numpy equivalents
_PRIMITIVE_TYPES: dict[str, np.dtype] = {
    "bool": np.dtype("bool"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "string": np.dtype("object"),
    "byte": np.dtype("uint8"),
    "char": np.dtype("uint8"),
}

# Default search paths for .msg files (relative to project root)
_MSG_SEARCH_PATHS: list[str] = []


def set_msg_search_paths(paths: list[str]) -> None:
    """Set the search paths for .msg files.

    Args:
        paths: List of directory paths to search for .msg files.
    """
    global _MSG_SEARCH_PATHS
    _MSG_SEARCH_PATHS = list(paths)


def get_msg_search_paths() -> list[str]:
    """Get the current search paths for .msg files."""
    return list(_MSG_SEARCH_PATHS)


@dataclass
class FieldDefinition:
    """A single field in a ``.msg`` definition.

    Attributes:
        name: Field name as written in the ``.msg`` file.
        type: ROS2 type string (e.g. ``"float32"``, ``"geometry_msgs/msg/Point"``).
        is_array: Whether the field is an array (``float32[]`` or ``float32[3]``).
        array_length: Fixed length for bounded arrays, or ``None`` for unbounded.
        is_primitive: ``True`` if ``type`` is a ROS2 primitive type.
    """

    name: str
    type: str
    is_array: bool = False
    array_length: int | None = None  # None for variable-length arrays
    is_primitive: bool = True

    @property
    def numpy_dtype(self) -> np.dtype | None:
        """Return numpy dtype for primitive types, None for complex types."""
        if self.is_primitive:
            return _PRIMITIVE_TYPES.get(self.type)
        return None


@dataclass
class MsgDefinition:
    """Parsed ``.msg`` file definition.

    Attributes:
        msg_type: Fully qualified ROS2 type name, e.g.
            ``"my_package/msg/MyMessage"``.
        fields: Ordered list of field definitions.
    """

    msg_type: str  # e.g. "my_package/msg/MyMessage"
    fields: list[FieldDefinition] = field(default_factory=list)

    @property
    def field_names(self) -> list[str]:
        """Return list of field names."""
        return [f.name for f in self.fields]


class MsgParser:
    """Parser for ROS2 ``.msg`` files.

    Reads ``.msg`` text content and produces ``MsgDefinition`` instances.
    Handles comments, constants, array notation, and primitive/complex
    type detection.
    """

    def parse(self, msg_file_path: str | Path) -> MsgDefinition:
        """Parse a .msg file and return its definition.

        Args:
            msg_file_path: Path to the .msg file.

        Returns:
            MsgDefinition with parsed fields.
        """
        path = Path(msg_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Message file not found: {path}")

        # Derive msg_type from file path
        # Convention: .../pkg_name/msg/TypeName.msg
        msg_type = self._infer_msg_type(path)

        content = path.read_text(encoding="utf-8")
        fields = self._parse_content(content)

        return MsgDefinition(msg_type=msg_type, fields=fields)

    def parse_from_type(self, msg_type: str) -> MsgDefinition | None:
        """Try to find and parse a .msg file for a given message type.

        Searches the configured search paths and project msgs/ directory.

        Args:
            msg_type: ROS2 message type, e.g. "my_package/msg/MyMessage"

        Returns:
            MsgDefinition if found, None otherwise.
        """
        # msg_type format: "package/msg/TypeName"
        parts = msg_type.split("/")
        if len(parts) != 3:
            logger.debug("Cannot parse msg_type format: %s", msg_type)
            return None

        package, _, type_name = parts
        filename = f"{type_name}.msg"

        # Build search paths
        search_dirs: list[Path] = []

        # Add configured search paths
        for sp in _MSG_SEARCH_PATHS:
            search_dirs.append(Path(sp))

        # Add default project msgs directory
        project_msgs = Path(__file__).resolve().parent.parent.parent.parent / "msgs"
        if project_msgs.is_dir():
            search_dirs.append(project_msgs)
            # Also search robot-specific subdirectories
            for subdir in project_msgs.iterdir():
                if subdir.is_dir():
                    search_dirs.append(subdir)

        # Search for the .msg file
        for search_dir in search_dirs:
            # Try direct match
            candidate = search_dir / filename
            if candidate.exists():
                return self.parse(candidate)

            # Try package/msg/TypeName.msg structure
            candidate = search_dir / package / "msg" / filename
            if candidate.exists():
                return self.parse(candidate)

        logger.debug("No .msg file found for type: %s", msg_type)
        return None

    def _parse_content(self, content: str) -> list[FieldDefinition]:
        """Parse ``.msg`` file content into field definitions.

        Skips blank lines, comments (``#``), and constant definitions
        (``TYPE NAME = VALUE``).
        """
        fields: list[FieldDefinition] = []

        for line in content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Remove inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()

            # Skip constants (TYPE NAME = VALUE)
            if "=" in line:
                # Check it's a constant definition (has = after name)
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "=":
                    continue
                # Could also be "TYPE NAME=VALUE" without spaces
                if len(parts) == 2 and "=" in parts[1]:
                    continue

            # Parse field: TYPE NAME
            tokens = line.split()
            if len(tokens) < 2:
                continue

            type_str = tokens[0]
            name = tokens[1]

            # Parse array notation
            is_array = False
            array_length: int | None = None

            if "[" in type_str:
                is_array = True
                bracket_start = type_str.index("[")
                bracket_end = type_str.index("]")
                length_str = type_str[bracket_start + 1 : bracket_end].strip()
                if length_str:
                    array_length = int(length_str)
                type_str = type_str[:bracket_start]

            # Determine if primitive
            is_primitive = type_str in _PRIMITIVE_TYPES

            fields.append(
                FieldDefinition(
                    name=name,
                    type=type_str,
                    is_array=is_array,
                    array_length=array_length,
                    is_primitive=is_primitive,
                )
            )

        return fields

    @staticmethod
    def _infer_msg_type(path: Path) -> str:
        """Infer message type from file path.

        Tries to find package/msg/TypeName.msg pattern in the path.
        Falls back to using just the filename.
        """
        parts = path.parts
        type_name = path.stem

        # Look for /msg/ in path
        for i, part in enumerate(parts):
            if part == "msg" and i > 0:
                package = parts[i - 1]
                return f"{package}/msg/{type_name}"

        # Fallback: use parent directory name as package
        parent = path.parent.name
        return f"{parent}/msg/{type_name}"


class MsgDecoder:
    """Generic decoder that uses a ``MsgDefinition`` to extract fields.

    When no dedicated built-in decoder exists for a message type, this
    class serves as a fallback: it iterates over the parsed field
    definitions and extracts numeric values from the deserialized message.

    Attributes:
        definition: The parsed ``.msg`` definition to decode against.
    """

    def __init__(self, msg_definition: MsgDefinition) -> None:
        self.definition = msg_definition

    def decode(
        self,
        msg: Any,
        selector: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Decode a message using the parsed definition.

        Args:
            msg: Deserialized rosbags message object.
            selector: Optional list of field names/paths to extract.
                Supports dot notation for nested fields.
            config: Optional configuration dict.

        Returns:
            np.ndarray with dtype float32 containing extracted values.
        """
        if config is None:
            config = {}

        if selector is not None:
            return self._decode_selected(msg, selector)

        return self._decode_all(msg)

    def _decode_all(self, msg: Any) -> np.ndarray:
        """Decode all numeric fields from the message."""
        values: list[float] = []

        for field_def in self.definition.fields:
            if field_def.type == "string":
                continue  # Skip string fields in numeric output

            value = getattr(msg, field_def.name, None)
            if value is None:
                continue

            if field_def.is_array:
                arr = np.array(value, dtype=np.float32)
                values.extend(arr.tolist())
            elif field_def.is_primitive:
                values.append(float(value))
            else:
                # Nested type: try to extract numeric sub-fields recursively
                nested = self._extract_nested_numeric(value)
                values.extend(nested)

        return np.array(values, dtype=np.float32)

    def _decode_selected(self, msg: Any, selector: list[str]) -> np.ndarray:
        """Decode only selected fields from the message."""
        values: list[float] = []

        for sel in selector:
            value = self._get_nested_value(msg, sel)
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.array(value, dtype=np.float32)
                values.extend(arr.tolist())
            else:
                values.append(float(value))

        return np.array(values, dtype=np.float32)

    @staticmethod
    def _get_nested_value(obj: Any, dotted_path: str) -> Any:
        """Get a value using dot notation, supporting array indexing.

        Supports paths like:
            "field_name"
            "nested.field"
            "array.0" (index into array)
        """
        current = obj
        for part in dotted_path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    @staticmethod
    def _extract_nested_numeric(obj: Any) -> list[float]:
        """Recursively extract numeric values from a nested object."""
        values: list[float] = []
        # Try common numeric attributes
        for attr_name in ("x", "y", "z", "w"):
            val = getattr(obj, attr_name, None)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))

        # If no standard attributes, try all attributes
        if not values:
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    val = getattr(obj, attr_name)
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                except (AttributeError, TypeError):
                    continue

        return values
