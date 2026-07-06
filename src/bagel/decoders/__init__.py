"""Decoder registry and dispatch for ROS2 message types.

This package provides a decorator-based registration system and a
``decode()`` dispatch function for converting deserialized ROS2 messages
to numpy arrays or PIL Images.

Decoder resolution order:

1. User-defined decoder specified via ``"decoder"`` key in config
   (``"module:function"`` string).
2. Built-in registry (populated by ``@register_decoder`` decorators in
   ``builtin.py`` and ``image.py``).
3. Generic fallback via ``msg_parser.MsgDecoder`` if a matching ``.msg``
   file is found.

Submodules:

- ``builtin``    -- Decoders for standard ROS2 messages (JointState,
                    Twist, Odometry, IMU, etc.).
- ``image``      -- Decoders for ``sensor_msgs/msg/Image`` and
                    ``sensor_msgs/msg/CompressedImage``.
- ``msg_parser`` -- Generic ``.msg`` file parser and field extractor.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global registry: msg_type string -> decoder function
_DECODER_REGISTRY: dict[str, Callable] = {}

# Parallel registry for array-returning image decoders (conversion fast path).
# Functions here return ``np.ndarray`` (H, W, 3) uint8 instead of PIL Images,
# letting the CLI/writer pipeline skip the PIL round trip. Pixel values are
# identical to the PIL decoders in ``_DECODER_REGISTRY``.
_ARRAY_DECODER_REGISTRY: dict[str, Callable] = {}


def register_decoder(msg_type: str) -> Callable:
    """Decorator to register a decoder function for a given ROS2 message type.

    Args:
        msg_type: ROS2 message type string, e.g. "sensor_msgs/msg/JointState".

    Returns:
        Decorator that registers the function and returns it unchanged.
    """

    def decorator(func: Callable) -> Callable:
        if msg_type in _DECODER_REGISTRY:
            logger.warning(
                "Overwriting existing decoder for %s with %s",
                msg_type,
                func.__qualname__,
            )
        _DECODER_REGISTRY[msg_type] = func
        return func

    return decorator


def register_array_decoder(msg_type: str) -> Callable:
    """Decorator to register an array-returning decoder for *msg_type*.

    Array decoders share the ``(msg, selector, config)`` signature with
    regular decoders but return an RGB ``np.ndarray`` instead of a PIL
    Image. They are dispatched via :func:`decode_array`.
    """

    def decorator(func: Callable) -> Callable:
        _ARRAY_DECODER_REGISTRY[msg_type] = func
        return func

    return decorator


def get_registered_types() -> list[str]:
    """Return a list of all registered message types."""
    return list(_DECODER_REGISTRY.keys())


def _resolve_user_decoder(func_path: str) -> Callable:
    """Resolve a user-defined decoder from a 'module:function' string.

    Args:
        func_path: String in the format "my_package.my_module:my_function".

    Returns:
        The resolved callable.

    Raises:
        ValueError: If the format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the function is not found in the module.
    """
    if ":" not in func_path:
        raise ValueError(
            f"User decoder must be in 'module:function' format, got: {func_path!r}"
        )
    module_path, func_name = func_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def decode(
    msg_type: str,
    deserialized_msg: Any,
    selector: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray | Image.Image:
    """Dispatch to the appropriate decoder for the given message type.

    Lookup order:
    1. Check if config specifies a user-defined decoder via "decoder" key
       (a "module:function" string).
    2. Check the built-in registry.
    3. Fall back to the generic msg_parser-based decoder.

    Args:
        msg_type: ROS2 message type string.
        deserialized_msg: Deserialized rosbags message object.
        selector: Optional list of field selectors for extraction.
        config: Optional configuration dict. May contain:
            - "decoder": a "module:function" string for user-defined decoders
            - other decoder-specific settings (e.g. "image_size", "unit_conversion")

    Returns:
        np.ndarray (float32) for numeric data, or PIL.Image.Image for images.

    Raises:
        ValueError: If no decoder can handle the message type.
    """
    if config is None:
        config = {}

    # 1. User-defined decoder override
    user_decoder_path = config.get("decoder")
    if user_decoder_path and isinstance(user_decoder_path, str):
        user_func = _resolve_user_decoder(user_decoder_path)
        return user_func(deserialized_msg, selector, config)

    # 2. Built-in registry
    if msg_type in _DECODER_REGISTRY:
        return _DECODER_REGISTRY[msg_type](deserialized_msg, selector, config)

    # 3. Fallback to msg_parser generic decoder
    from bagel.decoders.msg_parser import MsgDecoder, MsgParser

    # Try to find a .msg file for this type
    parser = MsgParser()
    msg_definition = parser.parse_from_type(msg_type)
    if msg_definition is not None:
        generic_decoder = MsgDecoder(msg_definition)
        return generic_decoder.decode(deserialized_msg, selector, config)

    raise ValueError(
        f"No decoder registered for message type {msg_type!r} "
        f"and no .msg file found for fallback parsing."
    )


def decode_array(
    msg_type: str,
    deserialized_msg: Any,
    selector: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray | Image.Image:
    """Decode an image message, preferring the array fast path.

    For message types with a registered array decoder this skips the PIL
    round trip and returns an RGB ``np.ndarray`` (H, W, 3) uint8 whose pixel
    values are identical to :func:`decode`'s PIL output. Other message types
    fall back to :func:`decode` unchanged (which may return a PIL Image).

    Args:
        msg_type: ROS2 message type string.
        deserialized_msg: Deserialized rosbags message object.
        selector: Optional list of field selectors for extraction.
        config: Optional configuration dict (e.g. "image_size").

    Returns:
        ``np.ndarray`` (fast path) or whatever :func:`decode` returns.
    """
    if config is None:
        config = {}
    if not config.get("decoder"):
        array_decoder = _ARRAY_DECODER_REGISTRY.get(msg_type)
        if array_decoder is not None:
            return array_decoder(deserialized_msg, selector, config)
    return decode(msg_type, deserialized_msg, selector, config)


# Import built-in decoders to trigger registration on module load
import bagel.decoders.builtin  # noqa: E402, F401
import bagel.decoders.image  # noqa: E402, F401
