"""座標変換ユーティリティ（純ロジック・I/O 非依存）。

ROS2 の ``/tf`` / ``/tf_static``（``tf2_msgs/msg/TFMessage``）から frame tree を
構築し、任意の 2 フレーム間の相対姿勢を解決する :class:`TransformLookup` と、
quaternion(xyzw)→euler 変換 :func:`quat_xyzw_to_euler` を提供する。

設計方針:
- 依存は numpy のみ（scipy/transforms3d を増やさない）。変換は 4x4 同次行列で
  合成・反転し、出力時に並進+quaternion へ分解する。
- メッセージ I/O からは独立（``add_static`` / ``add_dynamic`` は deserialize 済み
  メッセージを受け取るだけ）なので、合成データで単体テスト可能。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# quaternion / matrix helpers
# ---------------------------------------------------------------------------


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """quaternion(xyzw) を 3x3 回転行列へ変換する（正規化込み）。"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 回転行列を quaternion(xyzw) へ変換する（trace 法）。"""
    m = rot
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def quat_xyzw_to_euler(
    x: float, y: float, z: float, w: float, *, convention: str = "xyz"
) -> tuple[float, float, float]:
    """quaternion(xyzw) を euler 角（ラジアン）へ変換する。

    Args:
        x, y, z, w: quaternion 成分（xyzw 順）。
        convention: ``"xyz"``（roll-x, pitch-y, yaw-z; ROS の RPY 相当）または
            ``"zyx"``（順序を反転して返す）。

    Returns:
        ``(roll, pitch, yaw)`` をラジアンで。``convention="zyx"`` の場合は
        ``(yaw, pitch, roll)`` の順で返す。

    Notes:
        gimbal lock（pitch≈±90°）でも NaN を返さず決定的な値を返す
        （縮退時は roll=0 とし yaw に回転を畳み込む）。
    """
    if convention not in ("xyz", "zyx"):
        raise ValueError(f"Unsupported euler convention: {convention!r}")
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    x, y, z, w = x / n, y / n, z / n, w / n

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0 - 1e-9:  # gimbal lock
        # 縮退時は roll=0 とし yaw に回転を畳み込む。yaw の符号は非縮退側
        # （sinp→±1 の極限）と一致させる必要がある: 非縮退の yaw 公式は
        # pitch→±90° で ``-copysign(2, sinp)*atan2(x, w)`` に収束するため、
        # ここでもその符号を採用して特異点を跨いで連続にする。
        pitch = math.copysign(math.pi / 2.0, sinp)
        roll = 0.0
        yaw = -math.copysign(2.0, sinp) * math.atan2(x, w)
    else:
        pitch = math.asin(sinp)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # 値域を (-pi, pi] に正規化
    roll, pitch, yaw = (
        math.atan2(math.sin(a), math.cos(a)) if i != 1 else a
        for i, a in enumerate((roll, pitch, yaw))
    )
    if convention == "zyx":
        return (yaw, pitch, roll)
    return (roll, pitch, yaw)


def _homogeneous(translation: tuple[float, float, float], quat_xyzw) -> np.ndarray:
    """並進 + quaternion(xyzw) から 4x4 同次変換行列を作る。"""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_matrix(*quat_xyzw)
    mat[:3, 3] = translation
    return mat


# ---------------------------------------------------------------------------
# TransformLookup
# ---------------------------------------------------------------------------


@dataclass
class TransformLookup:
    """``/tf`` / ``/tf_static`` から frame tree を構築し相対姿勢を解決する。

    エッジ ``(parent, child)`` は「child フレームの parent における姿勢」
    （point_parent = T @ point_child）を表す 4x4 行列で保持する。静的エッジは
    単一の行列、動的エッジは ``(stamp_ns, 行列)`` の時刻順リスト。
    """

    _static: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    _dynamic: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = field(
        default_factory=dict
    )
    _dynamic_sorted: bool = field(default=True)

    # ---- ingest -------------------------------------------------------
    def _iter_transforms(self, tf_msg: Any):
        for tr in tf_msg.transforms:
            parent = tr.header.frame_id
            child = tr.child_frame_id
            t = tr.transform.translation
            r = tr.transform.rotation
            stamp = tr.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            mat = _homogeneous((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))
            yield parent, child, stamp_ns, mat

    def add_static(self, tf_msg: Any) -> None:
        """``/tf_static`` メッセージ（複数 transform 可）を取り込む。"""
        for parent, child, _stamp_ns, mat in self._iter_transforms(tf_msg):
            self._static[(parent, child)] = mat

    def add_dynamic(self, tf_msg: Any) -> None:
        """``/tf`` メッセージ（複数 transform 可）を取り込む。"""
        for parent, child, stamp_ns, mat in self._iter_transforms(tf_msg):
            self._dynamic.setdefault((parent, child), []).append((stamp_ns, mat))
            self._dynamic_sorted = False

    def _ensure_sorted(self) -> None:
        if self._dynamic_sorted:
            return
        for timeline in self._dynamic.values():
            timeline.sort(key=lambda kv: kv[0])
        self._dynamic_sorted = True

    # ---- query --------------------------------------------------------
    def _nearest(self, timeline: list[tuple[int, np.ndarray]], stamp_ns: int):
        """時刻 ``stamp_ns`` に最も近い動的変換を返す（nearest-in-time）。"""
        stamps = [s for s, _ in timeline]
        i = bisect.bisect_left(stamps, stamp_ns)
        if i == 0:
            return timeline[0][1]
        if i >= len(timeline):
            return timeline[-1][1]
        before_s, before_m = timeline[i - 1]
        after_s, after_m = timeline[i]
        return before_m if (stamp_ns - before_s) <= (after_s - stamp_ns) else after_m

    def _edge(self, parent: str, child: str, stamp_ns: int) -> np.ndarray | None:
        """point_child→point_parent を写す行列 T_{parent<-child} を返す。"""
        if (parent, child) in self._static:
            return self._static[(parent, child)]
        if (child, parent) in self._static:
            return np.linalg.inv(self._static[(child, parent)])
        if (parent, child) in self._dynamic:
            return self._nearest(self._dynamic[(parent, child)], stamp_ns)
        if (child, parent) in self._dynamic:
            return np.linalg.inv(
                self._nearest(self._dynamic[(child, parent)], stamp_ns)
            )
        return None

    def _neighbors(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {}
        for a, b in list(self._static.keys()) + list(self._dynamic.keys()):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj

    def _find_path(self, frame_from: str, frame_to: str) -> list[str]:
        """frame_from→frame_to の frame 列を BFS で求める（無向）。"""
        if frame_from == frame_to:
            return [frame_from]
        adj = self._neighbors()
        if frame_from not in adj or frame_to not in adj:
            raise ValueError(f"frame not in tf tree: {frame_from!r} / {frame_to!r}")
        queue = [frame_from]
        prev: dict[str, str] = {frame_from: frame_from}
        while queue:
            cur = queue.pop(0)
            if cur == frame_to:
                break
            for nb in adj.get(cur, ()):
                if nb not in prev:
                    prev[nb] = cur
                    queue.append(nb)
        if frame_to not in prev:
            raise ValueError(f"no transform path from {frame_from!r} to {frame_to!r}")
        path = [frame_to]
        while path[-1] != frame_from:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def lookup(self, frame_to: str, frame_from: str, stamp_ns: int) -> np.ndarray:
        """frame_from の姿勢を frame_to で表した ``[tx,ty,tz,qx,qy,qz,qw]`` を返す。

        Args:
            frame_to: 基準フレーム。
            frame_from: 姿勢を求めたいフレーム。
            stamp_ns: 動的 tf の参照時刻（nearest-in-time で選択）。

        Returns:
            7 要素の ``float64`` 配列（並進 3 + quaternion xyzw 4）。

        Raises:
            ValueError: フレームが tree に無い／経路が存在しない場合。
        """
        self._ensure_sorted()
        path = self._find_path(frame_from, frame_to)
        result = np.eye(4, dtype=np.float64)
        for u, v in zip(path[:-1], path[1:]):
            t_v_u = self._edge(v, u, stamp_ns)  # point_u -> point_v
            if t_v_u is None:
                raise ValueError(f"missing tf edge between {u!r} and {v!r}")
            result = t_v_u @ result
        qx, qy, qz, qw = matrix_to_quat(result[:3, :3])
        tx, ty, tz = result[:3, 3]
        return np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float64)
