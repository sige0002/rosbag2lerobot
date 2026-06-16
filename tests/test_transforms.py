"""Unit tests for src/bagel/transforms.py (quaternion math + TransformLookup)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from bagel.transforms import (
    TransformLookup,
    matrix_to_quat,
    quat_to_matrix,
    quat_xyzw_to_euler,
)


def _q_axis(axis: str, angle: float) -> tuple[float, float, float, float]:
    h = angle / 2.0
    s, c = math.sin(h), math.cos(h)
    return {"x": (s, 0, 0, c), "y": (0, s, 0, c), "z": (0, 0, s, c)}[axis]


class TestEuler:
    def test_identity(self):
        assert quat_xyzw_to_euler(0, 0, 0, 1) == pytest.approx((0, 0, 0))

    def test_roll_90(self):
        r, p, y = quat_xyzw_to_euler(*_q_axis("x", math.pi / 2))
        assert (r, p, y) == pytest.approx((math.pi / 2, 0, 0), abs=1e-9)

    def test_pitch_45(self):
        r, p, y = quat_xyzw_to_euler(*_q_axis("y", math.pi / 4))
        assert (r, p, y) == pytest.approx((0, math.pi / 4, 0), abs=1e-9)

    def test_yaw_90(self):
        r, p, y = quat_xyzw_to_euler(*_q_axis("z", math.pi / 2))
        assert (r, p, y) == pytest.approx((0, 0, math.pi / 2), abs=1e-9)

    def test_gimbal_lock_no_nan(self):
        # pitch = +90deg is the singular case; must stay finite + deterministic.
        r, p, y = quat_xyzw_to_euler(*_q_axis("y", math.pi / 2))
        assert all(math.isfinite(a) for a in (r, p, y))
        assert p == pytest.approx(math.pi / 2, abs=1e-6)

    def test_gimbal_lock_yaw_continuous(self):
        # The degenerate (|sinp|>=1) branch must agree with the limit of the
        # non-singular branch as pitch -> +-90deg. Build q = qz(yaw) * qy(pitch)
        # so the non-singular result has a known yaw, then compare the value at
        # the exact singularity (pitch=+-90) against pitch=+-89.999.
        def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return (
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            )

        def _wrap(a: float) -> float:
            return math.atan2(math.sin(a), math.cos(a))

        for pitch_deg in (90.0, -90.0):
            for yaw_deg in (30.0, -30.0, 120.0):
                near = qmul(
                    _q_axis("z", math.radians(yaw_deg)),
                    _q_axis(
                        "y", math.radians(pitch_deg - math.copysign(0.001, pitch_deg))
                    ),
                )
                at = qmul(
                    _q_axis("z", math.radians(yaw_deg)),
                    _q_axis("y", math.radians(pitch_deg)),
                )
                _, p_near, y_near = quat_xyzw_to_euler(*near)
                _, p_at, y_at = quat_xyzw_to_euler(*at)
                # pitch saturates at +-90 in the degenerate branch.
                assert p_at == pytest.approx(
                    math.copysign(math.pi / 2, pitch_deg), abs=1e-6
                )
                # yaw must be continuous across the singularity.
                assert _wrap(y_at - y_near) == pytest.approx(0.0, abs=1e-3)

    def test_zyx_reverses_order(self):
        q = _q_axis("z", math.pi / 2)
        xyz = quat_xyzw_to_euler(*q, convention="xyz")
        zyx = quat_xyzw_to_euler(*q, convention="zyx")
        assert zyx == pytest.approx((xyz[2], xyz[1], xyz[0]))

    def test_bad_convention(self):
        with pytest.raises(ValueError, match="convention"):
            quat_xyzw_to_euler(0, 0, 0, 1, convention="abc")


class TestMatrixRoundtrip:
    def test_quat_matrix_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = rng.normal(size=4)
            v /= np.linalg.norm(v)
            q2 = matrix_to_quat(quat_to_matrix(*v))
            q2 = np.array(q2)
            # quaternion double-cover: q and -q are the same rotation.
            assert np.allclose(q2, v, atol=1e-7) or np.allclose(q2, -v, atol=1e-7)


def _tf(parent, child, sec, nanosec, txyz, qxyzw):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=parent, stamp=SimpleNamespace(sec=sec, nanosec=nanosec)
        ),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=txyz[0], y=txyz[1], z=txyz[2]),
            rotation=SimpleNamespace(x=qxyzw[0], y=qxyzw[1], z=qxyzw[2], w=qxyzw[3]),
        ),
    )


def _msg(*transforms):
    return SimpleNamespace(transforms=list(transforms))


class TestTransformLookup:
    def _build(self) -> TransformLookup:
        lk = TransformLookup()
        lk.add_static(_msg(_tf("A", "B", 0, 0, (1, 0, 0), (0, 0, 0, 1))))
        lk.add_dynamic(_msg(_tf("B", "C", 0, 1000, (0, 1, 0), (0, 0, 0, 1))))
        lk.add_dynamic(_msg(_tf("B", "C", 0, 3000, (0, 2, 0), (0, 0, 0, 1))))
        return lk

    def test_compose_static_dynamic(self):
        lk = self._build()
        out = lk.lookup("A", "C", 1000)  # pose of C in A
        assert out[:3] == pytest.approx([1, 1, 0])
        assert out[3:] == pytest.approx([0, 0, 0, 1])

    def test_inverse_direction(self):
        lk = self._build()
        out = lk.lookup("C", "A", 1000)  # pose of A in C
        assert out[:3] == pytest.approx([-1, -1, 0])

    def test_nearest_in_time(self):
        lk = self._build()
        assert lk.lookup("A", "C", 1400)[:3] == pytest.approx([1, 1, 0])  # ->1000
        assert lk.lookup("A", "C", 2600)[:3] == pytest.approx([1, 2, 0])  # ->3000

    def test_no_path_raises(self):
        lk = self._build()
        with pytest.raises(ValueError):
            lk.lookup("A", "Z", 1000)

    def test_repeated_lookups_identical(self):
        # Caching (adjacency + path + stamp arrays) must not change results:
        # repeated lookups return identical values.
        lk = self._build()
        first = lk.lookup("A", "C", 1400)
        for _ in range(5):
            assert np.array_equal(lk.lookup("A", "C", 1400), first)
        # A different stamp still resolves nearest-in-time correctly after the
        # path/adjacency caches are warm.
        assert lk.lookup("A", "C", 2600)[:3] == pytest.approx([1, 2, 0])

    def test_add_after_lookup_invalidates_cache(self):
        # Warm the caches with a lookup, then extend the tree; the new topology
        # and the new dynamic sample must be reflected (cache invalidation).
        lk = self._build()
        lk.lookup("A", "C", 1400)  # warms adjacency + path caches

        # New static edge C->D extends the path A-B-C-D; must now be reachable.
        lk.add_static(_msg(_tf("C", "D", 0, 0, (0, 0, 1), (0, 0, 0, 1))))
        out = lk.lookup("A", "D", 1400)  # pose of D in A at t->1000
        # A<-B(1,0,0) ∘ B<-C(0,1,0)@1000 ∘ C<-D(0,0,1) = (1,1,1).
        assert out[:3] == pytest.approx([1, 1, 1])

        # A new dynamic sample on B->C nearer to a queried stamp must be picked
        # up too (stamp-array cache rebuilt on the next sorted access).
        lk.add_dynamic(_msg(_tf("B", "C", 0, 2500, (0, 9, 0), (0, 0, 0, 1))))
        assert lk.lookup("A", "C", 2500)[:3] == pytest.approx([1, 9, 0])
