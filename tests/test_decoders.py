"""Tests for the decoder modules."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


class TestDecoderRegistry:
    """Tests for the decoder registry in __init__.py."""

    def test_register_and_dispatch(self):
        from bagel.decoders import _DECODER_REGISTRY

        # Built-in decoders should be registered
        assert "sensor_msgs/msg/JointState" in _DECODER_REGISTRY
        assert "sensor_msgs/msg/Image" in _DECODER_REGISTRY
        assert "sensor_msgs/msg/CompressedImage" in _DECODER_REGISTRY

    def test_get_registered_types(self):
        from bagel.decoders import get_registered_types

        types = get_registered_types()
        assert "std_msgs/msg/Float32" in types
        assert "geometry_msgs/msg/Twist" in types

    def test_user_defined_decoder(self):
        from bagel.decoders import decode

        # Create a simple user decoder module path that we can test
        # We'll test the resolution mechanism indirectly
        config = {"decoder": "json:loads"}  # This would fail but tests resolution
        with pytest.raises(Exception):
            decode("fake/msg/Type", {}, config=config)

    def test_unknown_type_no_msg_file(self):
        from bagel.decoders import decode

        with pytest.raises(ValueError, match="No decoder registered"):
            decode("totally_unknown/msg/FakeType", SimpleNamespace())


class TestBuiltinDecoders:
    """Tests for builtin.py decoders."""

    def _make_joint_state(
        self,
        names: list[str],
        position: list[float],
        velocity: list[float] | None = None,
        effort: list[float] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            name=names,
            position=position,
            velocity=velocity or [0.0] * len(names),
            effort=effort or [0.0] * len(names),
        )

    def test_joint_state_no_selector(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = decode_joint_state(msg, None, {})
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float32

    def test_joint_state_with_selector(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(
            ["shoulder", "elbow", "wrist"],
            [0.1, 0.2, 0.3],
        )
        result = decode_joint_state(msg, ["position.elbow", "position.wrist"], {})
        np.testing.assert_allclose(result, [0.2, 0.3])

    def test_joint_state_with_alias(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a", "b"], [10.0, 20.0])
        result = decode_joint_state(msg, ["pos.a"], {})
        np.testing.assert_allclose(result, [10.0])

    def test_joint_state_wildcard(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = decode_joint_state(msg, ["position.*"], {})
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_joint_state_pad_to(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a", "b"], [1.0, 2.0])
        result = decode_joint_state(msg, ["position.a", "position.b"], {"pad_to": 7})
        assert len(result) == 7
        np.testing.assert_allclose(result[:2], [1.0, 2.0])
        np.testing.assert_allclose(result[2:], [0.0] * 5)

    def test_joint_state_rad2deg(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a"], [np.pi])
        result = decode_joint_state(msg, None, {"unit_conversion": "rad2deg"})
        np.testing.assert_allclose(result, [180.0], atol=1e-4)

    def test_joint_state_field_only_selector(self):
        """selector=["position"] should return all position values."""
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = decode_joint_state(msg, ["position"], {})
        assert len(result) == 3
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_joint_state_invalid_field(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a"], [1.0])
        with pytest.raises(ValueError, match="has no field"):
            decode_joint_state(msg, ["nonexistent_field"], {})

    def test_joint_state_unknown_joint(self):
        from bagel.decoders.builtin import decode_joint_state

        msg = self._make_joint_state(["a"], [1.0])
        with pytest.raises(ValueError, match="not found"):
            decode_joint_state(msg, ["position.unknown_joint"], {})

    def test_twist_no_selector(self):
        from bagel.decoders.builtin import decode_twist

        msg = SimpleNamespace(
            linear=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            angular=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        )
        result = decode_twist(msg, None, {})
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_twist_with_selector(self):
        from bagel.decoders.builtin import decode_twist

        msg = SimpleNamespace(
            linear=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            angular=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        )
        result = decode_twist(msg, ["linear.x", "angular.z"], {})
        np.testing.assert_array_equal(result, [1.0, 6.0])

    def test_twist_alias(self):
        from bagel.decoders.builtin import decode_twist

        msg = SimpleNamespace(
            linear=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            angular=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        )
        result = decode_twist(msg, ["lin.x", "ang.z"], {})
        np.testing.assert_array_equal(result, [1.0, 6.0])

    def test_twist_stamped(self):
        from bagel.decoders.builtin import decode_twist_stamped

        inner = SimpleNamespace(
            linear=SimpleNamespace(x=1.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.5),
        )
        msg = SimpleNamespace(header=SimpleNamespace(), twist=inner)
        result = decode_twist_stamped(msg, ["linear.x", "angular.z"], {})
        np.testing.assert_array_equal(result, [1.0, 0.5])

    def test_odometry_no_selector(self):
        from bagel.decoders.builtin import decode_odometry

        msg = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.1, y=0.2, z=0.3),
                    angular=SimpleNamespace(x=0.4, y=0.5, z=0.6),
                )
            ),
        )
        result = decode_odometry(msg, None, {})
        assert len(result) == 13
        assert result.dtype == np.float32

    def test_imu_default(self):
        from bagel.decoders.builtin import decode_imu

        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.81),
        )
        result = decode_imu(msg, None, {})
        assert len(result) == 10
        assert result.dtype == np.float32

    def test_pose_euler_selector(self):
        """selector 'orientation.euler_xyz' returns 3 finite euler floats."""
        from bagel.decoders.builtin import decode_pose
        from bagel.transforms import quat_xyzw_to_euler

        # A non-trivial quaternion (45deg about each axis-ish).
        qx, qy, qz, qw = 0.2706, 0.2706, 0.6533, 0.6533
        msg = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        )
        result = decode_pose(msg, ["orientation.euler_xyz"], {})
        assert len(result) == 3
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        expected = quat_xyzw_to_euler(qx, qy, qz, qw, convention="xyz")
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_pose_euler_with_normal_selectors(self):
        """Mixing a position selector with an euler selector flattens correctly."""
        from bagel.decoders.builtin import decode_pose
        from bagel.transforms import quat_xyzw_to_euler

        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        msg = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        )
        result = decode_pose(
            msg, ["position.x", "position.y", "orientation.euler_xyz"], {}
        )
        # 1 (x) + 1 (y) + 3 (euler) = 5 values
        assert len(result) == 5
        np.testing.assert_allclose(result[:2], [1.0, 2.0])
        expected = quat_xyzw_to_euler(qx, qy, qz, qw, convention="xyz")
        np.testing.assert_allclose(result[2:], expected, atol=1e-5)

    def test_pose_normal_selector_unchanged(self):
        """A non-euler selector still returns one value per entry (no regression)."""
        from bagel.decoders.builtin import decode_pose

        msg = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        result = decode_pose(msg, ["position.x", "orientation.w"], {})
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_imu_euler_selector(self):
        """Imu 'orientation.euler_zyx' returns 3 finite euler floats (zyx order)."""
        from bagel.decoders.builtin import decode_imu
        from bagel.transforms import quat_xyzw_to_euler

        qx, qy, qz, qw = 0.1, 0.2, 0.3, 0.9273
        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.81),
        )
        result = decode_imu(msg, ["orientation.euler_zyx"], {})
        assert len(result) == 3
        assert np.all(np.isfinite(result))
        expected = quat_xyzw_to_euler(qx, qy, qz, qw, convention="zyx")
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_euler_bare_selector(self):
        """A bare 'euler_xyz' selector resolves the message itself as the quat."""
        from bagel.decoders.builtin import _extract_by_selector
        from bagel.transforms import quat_xyzw_to_euler

        quat = SimpleNamespace(x=0.2706, y=0.2706, z=0.6533, w=0.6533)
        result = _extract_by_selector(quat, "euler_xyz")
        assert len(result) == 3
        expected = quat_xyzw_to_euler(0.2706, 0.2706, 0.6533, 0.6533, convention="xyz")
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_joy_default(self):
        from bagel.decoders.builtin import decode_joy

        msg = SimpleNamespace(axes=[0.5, -0.3], buttons=[1, 0, 1])
        result = decode_joy(msg, None, {})
        assert len(result) == 5
        np.testing.assert_allclose(result, [0.5, -0.3, 1.0, 0.0, 1.0])

    def test_joy_selector(self):
        from bagel.decoders.builtin import decode_joy

        msg = SimpleNamespace(axes=[0.5, -0.3, 0.8], buttons=[1, 0])
        result = decode_joy(msg, ["axes.0", "buttons.1"], {})
        np.testing.assert_allclose(result, [0.5, 0.0])

    def test_float32(self):
        from bagel.decoders.builtin import decode_float32

        msg = SimpleNamespace(data=3.14)
        result = decode_float32(msg, None, {})
        np.testing.assert_allclose(result, [3.14], atol=1e-5)

    def test_float64(self):
        from bagel.decoders.builtin import decode_float64

        msg = SimpleNamespace(data=2.718281828)
        result = decode_float64(msg, None, {})
        assert result.dtype == np.float32
        assert len(result) == 1

    def test_float32_multi_array(self):
        from bagel.decoders.builtin import decode_float32_multi_array

        msg = SimpleNamespace(data=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = decode_float32_multi_array(msg, None, {})
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_float32_multi_array_selector(self):
        from bagel.decoders.builtin import decode_float32_multi_array

        msg = SimpleNamespace(data=[10.0, 20.0, 30.0, 40.0])
        result = decode_float32_multi_array(msg, ["0", "3"], {})
        np.testing.assert_array_equal(result, [10.0, 40.0])

    def test_string(self):
        from bagel.decoders.builtin import decode_string

        msg = SimpleNamespace(data="hello world")
        result = decode_string(msg, None, {})
        assert result == "hello world"


def _rvl_encode(depth: np.ndarray) -> bytes:
    """Test-only RVL encoder (inverse of ``image._decode_rvl``) for round-trips."""
    flat = depth.astype(np.uint16).ravel().tolist()
    nibbles: list[int] = []

    def enc_vle(value: int) -> None:
        while True:
            n = value & 0x7
            value >>= 3
            nibbles.append(n | 0x8 if value else n)
            if not value:
                break

    n = len(flat)
    i = 0
    previous = 0
    while i < n:
        z = 0
        while i < n and flat[i] == 0:
            z += 1
            i += 1
        enc_vle(z)
        start = i
        while i < n and flat[i] != 0:
            i += 1
        enc_vle(i - start)
        for j in range(start, i):
            delta = flat[j] - previous
            previous = flat[j]
            enc_vle((delta << 1) ^ (delta >> 31))  # zigzag encode
    while len(nibbles) % 8:
        nibbles.append(0)
    out = bytearray()
    for k in range(0, len(nibbles), 8):
        word = 0
        for j in range(8):
            word |= (nibbles[k + j] & 0xF) << (28 - 4 * j)
        out += int(word).to_bytes(4, "little")
    return bytes(out)


class TestImageDecoders:
    """Tests for image.py decoders."""

    def test_rgb8_image(self):
        from bagel.decoders.image import decode_image

        h, w = 4, 6
        data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8).tobytes()
        msg = SimpleNamespace(data=data, height=h, width=w, step=w * 3, encoding="rgb8")
        result = decode_image(msg, None, {})
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (w, h)

    def test_bgr8_image(self):
        from bagel.decoders.image import decode_image

        h, w = 4, 6
        # Create a known BGR image (pure blue in BGR = [255, 0, 0])
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel
        msg = SimpleNamespace(
            data=bgr.tobytes(), height=h, width=w, step=w * 3, encoding="bgr8"
        )
        result = decode_image(msg, None, {})
        arr = np.array(result)
        # After BGR->RGB conversion, blue channel should be in position 2
        assert arr[0, 0, 2] == 255
        assert arr[0, 0, 0] == 0

    def test_mono8_image(self):
        from bagel.decoders.image import decode_image

        h, w = 4, 6
        data = np.full((h, w), 128, dtype=np.uint8).tobytes()
        msg = SimpleNamespace(data=data, height=h, width=w, step=w, encoding="mono8")
        result = decode_image(msg, None, {})
        assert result.mode == "RGB"

    def test_image_resize(self):
        from bagel.decoders.image import decode_image

        h, w = 100, 200
        data = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
        msg = SimpleNamespace(data=data, height=h, width=w, step=w * 3, encoding="rgb8")
        result = decode_image(msg, None, {"image_size": [50, 100]})
        assert result.size == (100, 50)  # PIL size is (width, height)

    def test_compressed_jpeg(self):
        from bagel.decoders.image import decode_compressed_image
        import cv2

        # Create a small test image and encode as JPEG
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = [100, 150, 200]
        _, buf = cv2.imencode(".jpg", img)
        msg = SimpleNamespace(data=buf.tobytes(), format="jpeg")
        result = decode_compressed_image(msg, None, {})
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_compressed_png(self):
        from bagel.decoders.image import decode_compressed_image
        import cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = [50, 100, 150]
        _, buf = cv2.imencode(".png", img)
        msg = SimpleNamespace(data=buf.tobytes(), format="png")
        result = decode_compressed_image(msg, None, {})
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_mono16_image(self):
        from bagel.decoders.image import decode_image

        h, w = 4, 6
        data = np.full((h, w), 30000, dtype=np.uint16).tobytes()
        msg = SimpleNamespace(
            data=data, height=h, width=w, step=w * 2, encoding="mono16"
        )
        result = decode_image(msg, None, {})
        assert result.mode == "RGB"
        assert result.size == (w, h)

    def test_compressed_depth_rvl_roundtrip(self):
        from bagel.decoders.image import (
            _decode_compressed_depth,
            _decode_rvl,
            decode_compressed_image,
        )

        depth = np.array(
            [
                [0, 0, 100, 101, 0, 0],
                [0, 300, 305, 0, 0, 500],
                [501, 502, 0, 0, 0, 0],
                [0, 1000, 999, 998, 0, 1],
            ],
            dtype=np.uint16,
        )
        rows, cols = depth.shape
        stream = _rvl_encode(depth)

        # 純関数 RVL の round-trip は完全一致でなければならない。
        assert np.array_equal(_decode_rvl(stream, rows, cols), depth)

        # ConfigHeader(12) + cols + rows + RVL を組んでデコーダ経路を検証。
        payload = (
            b"\x00" * 12
            + int(cols).to_bytes(4, "little")
            + int(rows).to_bytes(4, "little")
            + stream
        )
        assert np.array_equal(
            _decode_compressed_depth(payload, "16uc1; compresseddepth rvl"), depth
        )

        msg = SimpleNamespace(data=payload, format="16UC1; compressedDepth rvl")
        pil = decode_compressed_image(msg, None, {})
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"
        assert pil.size == (cols, rows)

    def test_compressed_depth_non_rvl_raises(self):
        from bagel.decoders.image import decode_compressed_image

        msg = SimpleNamespace(data=b"\x00" * 24, format="16UC1; compressedDepth png")
        with pytest.raises(ValueError, match="RVL"):
            decode_compressed_image(msg, None, {})

    def test_rvl_odd_length_raises_valueerror(self):
        from bagel.decoders.image import _decode_rvl

        # Buffer length not a multiple of 4 -> clean ValueError, not the raw
        # ValueError np.frombuffer would otherwise raise.
        with pytest.raises(ValueError, match="multiple of 4"):
            _decode_rvl(b"\x00\x00\x00", 2, 2)

    def test_rvl_truncated_raises_valueerror(self):
        from bagel.decoders.image import _decode_rvl

        depth = np.array(
            [[0, 100, 101, 0], [200, 201, 0, 300], [0, 0, 400, 401]],
            dtype=np.uint16,
        )
        rows, cols = depth.shape
        stream = _rvl_encode(depth)
        assert len(stream) >= 8  # multi-word, so truncation loses real data.
        # Truncate to one word: the decoder runs out of nibbles before all
        # pixels are produced -> ValueError (not IndexError).
        truncated = stream[:4]
        with pytest.raises(ValueError, match="corrupt RVL"):
            _decode_rvl(truncated, rows, cols)

    def test_unsupported_encoding(self):
        from bagel.decoders.image import decode_image

        msg = SimpleNamespace(
            data=b"", height=1, width=1, step=1, encoding="bayer_rggb8"
        )
        with pytest.raises(ValueError, match="Unsupported image encoding"):
            decode_image(msg, None, {})


class TestMsgParser:
    """Tests for msg_parser.py."""

    def test_parse_simple_msg(self, tmp_path):
        from bagel.decoders.msg_parser import MsgParser

        msg_content = """\
# A test message
float32 x
float64 y
int32 count
string label
"""
        msg_file = tmp_path / "test_pkg" / "msg" / "TestMsg.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(msg_content)

        parser = MsgParser()
        definition = parser.parse(msg_file)

        assert len(definition.fields) == 4
        assert definition.fields[0].name == "x"
        assert definition.fields[0].type == "float32"
        assert not definition.fields[0].is_array
        assert definition.fields[1].name == "y"
        assert definition.fields[3].name == "label"
        assert definition.fields[3].type == "string"

    def test_parse_arrays(self, tmp_path):
        from bagel.decoders.msg_parser import MsgParser

        msg_content = """\
float32[3] position
float64[] velocities
uint8[10] buffer
"""
        msg_file = tmp_path / "pkg" / "msg" / "Arrays.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(msg_content)

        parser = MsgParser()
        definition = parser.parse(msg_file)

        assert len(definition.fields) == 3
        assert definition.fields[0].is_array
        assert definition.fields[0].array_length == 3
        assert definition.fields[1].is_array
        assert definition.fields[1].array_length is None  # variable length
        assert definition.fields[2].array_length == 10

    def test_skip_constants(self, tmp_path):
        from bagel.decoders.msg_parser import MsgParser

        msg_content = """\
uint8 STATUS_OK = 0
uint8 STATUS_ERR = 1
float32 value
string name
"""
        msg_file = tmp_path / "pkg" / "msg" / "WithConst.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(msg_content)

        parser = MsgParser()
        definition = parser.parse(msg_file)

        assert len(definition.fields) == 2
        assert definition.fields[0].name == "value"
        assert definition.fields[1].name == "name"

    def test_nested_types(self, tmp_path):
        from bagel.decoders.msg_parser import MsgParser

        msg_content = """\
geometry_msgs/Point position
float32 speed
"""
        msg_file = tmp_path / "pkg" / "msg" / "Nested.msg"
        msg_file.parent.mkdir(parents=True)
        msg_file.write_text(msg_content)

        parser = MsgParser()
        definition = parser.parse(msg_file)

        assert len(definition.fields) == 2
        assert definition.fields[0].name == "position"
        assert not definition.fields[0].is_primitive

    def test_msg_decoder_all_fields(self):
        from bagel.decoders.msg_parser import (
            FieldDefinition,
            MsgDecoder,
            MsgDefinition,
        )

        definition = MsgDefinition(
            msg_type="test/msg/Test",
            fields=[
                FieldDefinition(name="x", type="float32"),
                FieldDefinition(name="y", type="float64"),
                FieldDefinition(
                    name="data", type="float32", is_array=True, array_length=3
                ),
            ],
        )
        msg = SimpleNamespace(x=1.0, y=2.0, data=[3.0, 4.0, 5.0])
        decoder = MsgDecoder(definition)
        result = decoder.decode(msg)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_msg_decoder_selector(self):
        from bagel.decoders.msg_parser import (
            FieldDefinition,
            MsgDecoder,
            MsgDefinition,
        )

        definition = MsgDefinition(
            msg_type="test/msg/Test",
            fields=[
                FieldDefinition(name="a", type="float32"),
                FieldDefinition(name="b", type="float32"),
                FieldDefinition(name="c", type="float32"),
            ],
        )
        msg = SimpleNamespace(a=10.0, b=20.0, c=30.0)
        decoder = MsgDecoder(definition)
        result = decoder.decode(msg, selector=["b", "c"])
        np.testing.assert_allclose(result, [20.0, 30.0])

    def test_msg_decoder_nested_selector(self):
        from bagel.decoders.msg_parser import (
            FieldDefinition,
            MsgDecoder,
            MsgDefinition,
        )

        definition = MsgDefinition(
            msg_type="test/msg/Test",
            fields=[
                FieldDefinition(
                    name="pose", type="geometry_msgs/Pose", is_primitive=False
                ),
            ],
        )
        msg = SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0, z=3.0))
        )
        decoder = MsgDecoder(definition)
        result = decoder.decode(msg, selector=["pose.position.x", "pose.position.z"])
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_parse_from_type_not_found(self):
        from bagel.decoders.msg_parser import MsgParser

        parser = MsgParser()
        result = parser.parse_from_type("nonexistent/msg/FakeType")
        assert result is None


class TestDispatchIntegration:
    """Integration tests for the full decode dispatch path."""

    def test_decode_float32(self):
        from bagel.decoders import decode

        msg = SimpleNamespace(data=42.0)
        result = decode("std_msgs/msg/Float32", msg)
        np.testing.assert_allclose(result, [42.0])

    def test_decode_twist(self):
        from bagel.decoders import decode

        msg = SimpleNamespace(
            linear=SimpleNamespace(x=1.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.5),
        )
        result = decode(
            "geometry_msgs/msg/Twist", msg, selector=["linear.x", "angular.z"]
        )
        np.testing.assert_array_equal(result, [1.0, 0.5])

    def test_decode_image_rgb(self):
        from bagel.decoders import decode

        h, w = 2, 3
        data = np.zeros((h, w, 3), dtype=np.uint8)
        data[0, 0] = [255, 0, 0]
        msg = SimpleNamespace(
            data=data.tobytes(), height=h, width=w, step=w * 3, encoding="rgb8"
        )
        result = decode("sensor_msgs/msg/Image", msg)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
