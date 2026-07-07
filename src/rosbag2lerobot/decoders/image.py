"""Image decoders for ROS2 sensor_msgs image types.

Decodes ``sensor_msgs/msg/Image`` and ``sensor_msgs/msg/CompressedImage``
to ``PIL.Image.Image`` in RGB format.  Supports multiple pixel encodings
(rgb8, bgr8, rgba8, mono8, 16UC1, 32FC1, etc.) and optional resizing via
the ``image_size`` config key.

Compressed images are decoded using OpenCV for JPEG/PNG and fall back to
PIL for other formats.  ``compressedDepth`` (RVL 圧縮の 16bit 深度) は専用の
純 Python デコーダ（``_decode_rvl``）で復号し、8bit グレースケールへ正規化して
動画特徴として保存する。

変換パイプライン（CLI → writer）は PIL を介さない numpy 配列版デコーダ
（``@register_array_decoder`` で登録、``decoders.decode_array`` から dispatch）
を使う。公開デコーダ（``decode()`` 経由）は従来どおり PIL Image を返す薄い
ラッパで、画素値は配列版と完全に一致する。
"""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from rosbag2lerobot.decoders import register_array_decoder, register_decoder

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
    "mono16": (1, np.dtype("uint16")),  # 16UC1 のエイリアス（sample_bags の深度 raw）
    "32FC1": (1, np.dtype("float32")),
}


def _as_uint8_buffer(data: Any) -> np.ndarray:
    """Return *data* as a 1-D ``uint8`` numpy view without copying.

    rosbags deserializes ``uint8[]`` fields to numpy arrays already; plain
    ``bytes`` (tests, other producers) are wrapped via ``frombuffer``.
    """
    if isinstance(data, np.ndarray):
        return data
    return np.frombuffer(data, dtype=np.uint8)


def _resize_array_if_needed(arr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Resize an RGB array to the target size if specified in config.

    Config key: "image_size" as [height, width] or (height, width).
    Uses PIL LANCZOS so the resized pixels are identical to the historical
    PIL-based path.
    """
    image_size = config.get("image_size")
    if image_size is None:
        return arr
    target_h, target_w = int(image_size[0]), int(image_size[1])
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    img = Image.fromarray(arr, mode="RGB")
    return np.asarray(img.resize((target_w, target_h), Image.LANCZOS))


def decode_image_array(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode a raw ROS2 Image message to an RGB ``uint8`` array ``(H, W, 3)``.

    Supports encodings: rgb8, bgr8, rgba8, bgra8, mono8, 8UC1, 8UC3, 8UC4,
    16UC1, mono16, 32FC1.

    Args:
        msg: Deserialized sensor_msgs/msg/Image with attributes:
            data, height, width, step, encoding.
        selector: Unused for image decoding.
        config: May contain "image_size" as [height, width].

    Returns:
        ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, RGB order.
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

    # View raw bytes as a numpy array (no copy)
    raw = np.frombuffer(_as_uint8_buffer(msg.data), dtype=dtype)

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
    elif enc_key in ("16UC1", "mono16"):
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

    return _resize_array_if_needed(img_array, config)


@register_decoder("sensor_msgs/msg/Image")
def decode_image(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> Image.Image:
    """Decode a raw ROS2 Image message to a PIL RGB Image.

    Thin PIL wrapper around :func:`decode_image_array`; see there for the
    supported encodings and config keys.

    Returns:
        PIL.Image.Image in RGB mode.
    """
    return Image.fromarray(decode_image_array(msg, selector, config), mode="RGB")


def _decode_rvl(buf: bytes, rows: int, cols: int) -> np.ndarray:
    """RVL（Run-length / Variable-length）圧縮された深度ストリームを復号する。

    ROS ``compressed_depth_image_transport`` の RVL アルゴリズム（Wilson 方式）の
    純 Python 実装。3bit 単位の可変長グループを LSB から積み上げて値を復元し、
    「ゼロ連 → 非ゼロ連（ジグザグ差分）」の繰り返しで全画素を再構成する。

    Args:
        buf: RVL ストリーム（``ConfigHeader`` と rows/cols を除いた本体）。
        rows: 画像の行数（高さ）。
        cols: 画像の列数（幅）。

    Returns:
        ``(rows, cols)`` の ``uint16`` 深度配列。
    """
    if rows < 0 or cols < 0:
        raise ValueError(f"corrupt RVL stream: negative dimensions {rows}x{cols}")
    num_pixels = rows * cols
    # RVL ストリームは 32bit ワード列。長さが 4 の倍数でなければ破損とみなす
    # （ROS 実装は常にワード境界で書き出す）。
    if len(buf) % 4 != 0:
        raise ValueError(
            f"corrupt RVL stream: buffer length {len(buf)} is not a multiple of 4"
        )
    # 32bit ワード（リトルエンディアン）を MSB 側から 8 ニブルに展開して走査する。
    words = np.frombuffer(buf, dtype="<u4")
    shifts = np.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=np.uint32)
    nibbles = ((words[:, None] >> shifts) & 0xF).astype(np.uint8).ravel().tolist()
    n_nibbles = len(nibbles)

    out = np.zeros(num_pixels, dtype=np.uint16)
    nidx = 0

    def decode_vle() -> int:
        nonlocal nidx
        value = 0
        shift = 0
        while True:
            if nidx >= n_nibbles:
                raise ValueError("corrupt RVL stream: ran out of nibbles")
            n = nibbles[nidx]
            nidx += 1
            value |= (n & 0x7) << shift
            shift += 3
            if not (n & 0x8):
                return value

    idx = 0
    previous = 0
    remaining = num_pixels
    while remaining > 0:
        zeros = decode_vle()  # ゼロ画素の連長（out は初期化済みなので index 送りのみ）
        idx += zeros
        remaining -= zeros
        nonzeros = decode_vle()
        remaining -= nonzeros
        # 連長が残画素を超えると out への書き込みが配列外になる → 破損。
        if remaining < 0 or idx + nonzeros > num_pixels:
            raise ValueError("corrupt RVL stream: run length exceeds remaining pixels")
        for _ in range(nonzeros):
            positive = decode_vle()
            delta = (positive >> 1) ^ -(positive & 1)  # ジグザグ復号
            previous = (previous + delta) & 0xFFFF
            out[idx] = previous
            idx += 1
    return out.reshape((rows, cols))


def _decode_compressed_depth(data: bytes, fmt: str) -> np.ndarray:
    """``compressedDepth`` ペイロードを ``uint16`` 深度配列へ復号する。

    レイアウト: ``[0:12]`` = ConfigHeader、``[12:16]`` = cols(int32 LE)、
    ``[16:20]`` = rows(int32 LE)、``[20:]`` = RVL ストリーム。実データ
    （HSR）は ``16UC1; compressedDepth rvl`` のみ存在するため RVL のみ対応。

    Args:
        data: CompressedImage の ``data`` バイト列。
        fmt: 小文字化した ``format`` 文字列。

    Returns:
        ``(rows, cols)`` の ``uint16`` 深度配列。

    Raises:
        ValueError: RVL 以外の format、またはヘッダが不足している場合。
    """
    if "rvl" not in fmt:
        raise ValueError(f"Only RVL compressedDepth is supported, got format {fmt!r}")
    if len(data) < 20:
        raise ValueError("compressedDepth payload too short for header")
    cols = int.from_bytes(data[12:16], "little")
    rows = int.from_bytes(data[16:20], "little")
    return _decode_rvl(data[20:], rows, cols)


def _depth16_to_rgb(depth: np.ndarray) -> np.ndarray:
    """16bit 深度を 8bit グレースケール RGB へ正規化する（4a: lossy 動画保存）。

    既存の ``16UC1`` 経路と同じ ``>>8``（``/256``）固定スケールでフレーム間の
    一貫性を保つ。

    Args:
        depth: ``uint16`` の ``(H, W)`` 深度配列。

    Returns:
        ``(H, W, 3)`` の ``uint8`` RGB 配列。
    """
    gray = (depth >> 8).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def decode_compressed_image_array(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode a compressed ROS2 Image message (JPEG/PNG) to an RGB array.

    Args:
        msg: Deserialized sensor_msgs/msg/CompressedImage with attributes:
            data, format.
        selector: Unused for image decoding.
        config: May contain "image_size" as [height, width].

    Returns:
        ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, RGB order.
    """
    fmt = msg.format.lower() if hasattr(msg, "format") else ""

    if "compresseddepth" in fmt:
        raw = msg.data
        raw_bytes = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        depth = _decode_compressed_depth(raw_bytes, fmt)
        img_array = _depth16_to_rgb(depth)
    elif "jpeg" in fmt or "jpg" in fmt:
        # Use OpenCV for JPEG decompression (handles more edge cases)
        buf = _as_uint8_buffer(msg.data)
        img_array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Failed to decode JPEG compressed image")
        # OpenCV decodes to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    elif "png" in fmt:
        buf = _as_uint8_buffer(msg.data)
        img_array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Failed to decode PNG compressed image")
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: try PIL directly
        raw = msg.data
        raw_bytes = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            img_array = np.asarray(pil_img.convert("RGB"))
        except Exception as exc:
            raise ValueError(
                f"Cannot decode compressed image with format {msg.format!r}: {exc}"
            ) from exc

    return _resize_array_if_needed(img_array, config)


@register_decoder("sensor_msgs/msg/CompressedImage")
def decode_compressed_image(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> Image.Image:
    """Decode a compressed ROS2 Image message (JPEG/PNG) to a PIL RGB Image.

    Thin PIL wrapper around :func:`decode_compressed_image_array`.

    Returns:
        PIL.Image.Image in RGB mode.
    """
    return Image.fromarray(
        decode_compressed_image_array(msg, selector, config), mode="RGB"
    )


# Register the array fast path for the conversion pipeline (CLI → writer).
register_array_decoder("sensor_msgs/msg/Image")(decode_image_array)
register_array_decoder("sensor_msgs/msg/CompressedImage")(decode_compressed_image_array)
