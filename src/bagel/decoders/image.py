"""Image decoders for ROS2 sensor_msgs image types.

Decodes ``sensor_msgs/msg/Image`` and ``sensor_msgs/msg/CompressedImage``
to ``PIL.Image.Image`` in RGB format.  Supports multiple pixel encodings
(rgb8, bgr8, rgba8, mono8, 16UC1, 32FC1, etc.) and optional resizing via
the ``image_size`` config key.

Compressed images are decoded using OpenCV for JPEG/PNG and fall back to
PIL for other formats.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from bagel.decoders import register_decoder

logger = logging.getLogger(__name__)

# Number of channels and numpy dtype for each ROS image encoding
_ENCODING_INFO: dict[str, tuple[int, np.dtype]] = {
    "rgb8": (3, np.dtype("uint8")),
    "bgr8": (3, np.dtype("uint8")),
    "rgba8": (4, np.dtype("uint8")),
    "bgra8": (4, np.dtype("uint8")),
    "mono8": (1, np.dtype("uint8")),
    "8UC1": (1, np.dtype("uint8")),
    "8UC3": (3, np.dtype("uint8")),
    "8UC4": (4, np.dtype("uint8")),
    "16UC1": (1, np.dtype("uint16")),
    "32FC1": (1, np.dtype("float32")),
}


def _resize_if_needed(img: Image.Image, config: dict[str, Any]) -> Image.Image:
    """Resize image to target size if specified in config.

    Config key: "image_size" as [height, width] or (height, width).
    """
    image_size = config.get("image_size")
    if image_size is None:
        return img
    target_h, target_w = int(image_size[0]), int(image_size[1])
    if img.size == (target_w, target_h):
        return img
    return img.resize((target_w, target_h), Image.LANCZOS)


@register_decoder("sensor_msgs/msg/Image")
def decode_image(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> Image.Image:
    """Decode a raw ROS2 Image message to a PIL RGB Image.

    Supports encodings: rgb8, bgr8, rgba8, bgra8, mono8, 8UC1, 8UC3, 8UC4.

    Args:
        msg: Deserialized sensor_msgs/msg/Image with attributes:
            data, height, width, step, encoding.
        selector: Unused for image decoding.
        config: May contain "image_size" as [height, width].

    Returns:
        PIL.Image.Image in RGB mode.
    """
    encoding = msg.encoding.lower() if hasattr(msg, "encoding") else "rgb8"
    # Normalize encoding lookup (case-sensitive for UC types)
    enc_key = msg.encoding if msg.encoding in _ENCODING_INFO else encoding

    if enc_key not in _ENCODING_INFO:
        raise ValueError(
            f"Unsupported image encoding: {msg.encoding!r}. "
            f"Supported: {list(_ENCODING_INFO.keys())}"
        )

    channels, dtype = _ENCODING_INFO[enc_key]
    height = int(msg.height)
    width = int(msg.width)

    # Convert raw bytes to numpy array
    raw = np.frombuffer(bytes(msg.data), dtype=dtype)

    if channels == 1:
        img_array = raw.reshape((height, width))
    else:
        img_array = raw.reshape((height, width, channels))

    # Convert to RGB
    if enc_key in ("bgr8", "8UC3"):
        # Only convert if it's known BGR; 8UC3 assumed BGR (ROS convention)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    elif enc_key == "bgra8":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    elif enc_key in ("rgba8", "8UC4"):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif enc_key in ("mono8", "8UC1"):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif enc_key == "16UC1":
        # Normalize 16-bit to 8-bit for display
        img_array = (img_array / 256).astype(np.uint8)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif enc_key == "32FC1":
        # Normalize float to 0-255
        min_val, max_val = img_array.min(), img_array.max()
        if max_val > min_val:
            img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(
                np.uint8
            )
        else:
            img_array = np.zeros((height, width), dtype=np.uint8)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    # else: rgb8 is already RGB

    pil_img = Image.fromarray(img_array, mode="RGB")
    return _resize_if_needed(pil_img, config)


@register_decoder("sensor_msgs/msg/CompressedImage")
def decode_compressed_image(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> Image.Image:
    """Decode a compressed ROS2 Image message (JPEG/PNG) to a PIL RGB Image.

    Args:
        msg: Deserialized sensor_msgs/msg/CompressedImage with attributes:
            data, format.
        selector: Unused for image decoding.
        config: May contain "image_size" as [height, width].

    Returns:
        PIL.Image.Image in RGB mode.
    """
    fmt = msg.format.lower() if hasattr(msg, "format") else ""

    raw_data = bytes(msg.data)

    if "jpeg" in fmt or "jpg" in fmt:
        # Use OpenCV for JPEG decompression (handles more edge cases)
        buf = np.frombuffer(raw_data, dtype=np.uint8)
        img_array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Failed to decode JPEG compressed image")
        # OpenCV decodes to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_array, mode="RGB")
    elif "png" in fmt:
        buf = np.frombuffer(raw_data, dtype=np.uint8)
        img_array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Failed to decode PNG compressed image")
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_array, mode="RGB")
    else:
        # Fallback: try PIL directly
        try:
            pil_img = Image.open(io.BytesIO(raw_data))
            pil_img = pil_img.convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot decode compressed image with format {msg.format!r}: {exc}"
            ) from exc

    return _resize_if_needed(pil_img, config)
