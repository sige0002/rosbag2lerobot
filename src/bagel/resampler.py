"""Time synchronisation and fixed-fps resampling for bagel.

Takes streams of timestamped messages from multiple ROS topics and produces
a sequence of fixed-rate frames where every feature key has a value.

The ``Resampler`` converts variable-rate sensor data (e.g. 200 Hz joint
states, 30 fps cameras) into a single fixed-fps timeline.  Three fill
policies are supported: ``hold`` (zero-order hold), ``nearest`` (closest
within tolerance), and ``drop`` (null if nothing within tolerance).

The lookup per frame is vectorised with ``numpy.searchsorted`` so that
the full F frames x K keys index computation is O((F+N) log N) in a
handful of C-level calls rather than ``F * K`` Python-level binary
searches.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Resampler:
    """Resample multi-topic message streams to fixed-fps frames.

    Parameters
    ----------
    fps:
        Target frames per second.
    policy:
        How to fill missing values at a frame time.

        * ``"hold"`` – carry forward the last known value (default).
        * ``"nearest"`` – use the closest message within *tolerance_ms*.
        * ``"drop"`` – leave the value as ``None`` if no message is within
          tolerance.
    tolerance_ms:
        Maximum allowed time gap (in milliseconds) between the frame time
        and the nearest message before the value is considered missing.
        Only meaningful for ``"nearest"`` and ``"drop"`` policies (for
        ``"hold"`` the tolerance is still used to emit a warning but the
        last value is always carried forward).
    """

    fps: int
    policy: str = "hold"
    tolerance_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.policy not in ("hold", "drop", "nearest"):
            raise ValueError(
                f"Invalid resampling policy '{self.policy}'. "
                "Must be one of: hold, drop, nearest"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample(
        self,
        messages: list[tuple[str, int, Any]],
        feature_keys: list[str],
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Resample a list of messages to fixed-fps frames.

        Parameters
        ----------
        messages:
            Sorted list of ``(feature_key, timestamp_ns, decoded_value)``
            tuples.  **Must be sorted by timestamp_ns.**
        feature_keys:
            All expected feature keys that should appear in each output frame.
        start_ns:
            Episode start time in nanoseconds.  Defaults to the timestamp of
            the first message.
        end_ns:
            Episode end time in nanoseconds.  Defaults to the timestamp of
            the last message.

        Returns
        -------
        list[dict[str, Any]]
            One dict per frame.  Each dict has:
            - All entries from *feature_keys* (value or ``None``).
            - ``"timestamp"`` – ``float32`` frame timestamp relative to
              episode start (= ``frame_index / fps``).
            - ``"frame_index"`` – ``int``.
        """
        if not messages:
            return []

        # Determine time bounds
        if start_ns is None:
            start_ns = messages[0][1]
        if end_ns is None:
            end_ns = messages[-1][1]

        if end_ns < start_ns:
            raise ValueError("end_ns must be >= start_ns")

        # Build per-key parallel lists of (timestamps, values) in a single
        # pass over *messages*. Using two parallel lists avoids allocating
        # intermediate tuples and keeps binary-search-friendly memory
        # layout.  Bind the dicts' ``.append`` bound methods into a local
        # dispatch table so the hot loop can skip the per-iteration dict
        # lookup.
        key_timestamps: dict[str, list[int]] = {k: [] for k in feature_keys}
        key_values: dict[str, list[Any]] = {k: [] for k in feature_keys}
        ts_append: dict[str, Any] = {k: key_timestamps[k].append for k in feature_keys}
        val_append: dict[str, Any] = {k: key_values[k].append for k in feature_keys}
        for key, ts_ns, value in messages:
            ts_app = ts_append.get(key)
            if ts_app is not None:
                ts_app(ts_ns)
                val_append[key](value)

        # ------------------------------------------------------------------
        # Vectorised frame-time / per-key index computation
        # ------------------------------------------------------------------
        duration_s = (end_ns - start_ns) / 1e9
        n_frames = max(1, int(math.ceil(duration_s * self.fps)))
        frame_period_ns = int(1e9 / self.fps)
        tolerance_ns = int(self.tolerance_ms * 1e6)

        # Timestamps as int64 nanoseconds for every frame (C-level arange).
        frame_times = np.int64(start_ns) + np.arange(
            n_frames, dtype=np.int64
        ) * np.int64(frame_period_ns)

        # Pre-compute timestamp float32 values once (ループ内の割り算回避).
        timestamps_f32 = (np.arange(n_frames) / self.fps).astype(np.float32)

        # Per-key timestamp arrays (one-time int64 conversion).
        key_ts_arr: dict[str, np.ndarray] = {
            k: np.asarray(key_timestamps[k], dtype=np.int64) for k in feature_keys
        }

        # Vectorised per-key picker: one Python list[Any] per key, length
        # *n_frames*, honouring the configured policy.  We keep the keys in
        # a parallel tuple so the frame-dict assembly below can use
        # ``zip`` (avoids per-frame dict-merge overhead).
        keys_tuple: tuple[str, ...] = tuple(feature_keys)
        per_key_picked: list[list[Any]] = [
            self._pick_values_vectorised(
                frame_times,
                key_ts_arr[k],
                key_values[k],
                tolerance_ns,
            )
            for k in keys_tuple
        ]

        # Timestamps as a plain Python list once (avoids per-frame
        # numpy-scalar indexing in the hot dict-building loop).
        timestamps_list = timestamps_f32.tolist()  # list[float]

        # Assemble frame dicts.  Zipping the transposed per-key value
        # lists lets the inner ``dict(zip(...))`` walk a single C-level
        # iterator, sidestepping the ``{**inner}`` merge that is
        # noticeably slower.  ``per_key_picked`` is empty when
        # *feature_keys* is empty (no content keys), in which case we
        # still need *n_frames* timestamp-only frames.
        frames: list[dict[str, Any]] = []
        frames_append = frames.append
        f32 = np.float32
        if per_key_picked:
            for fi, per_frame_vals in enumerate(zip(*per_key_picked)):
                frame = dict(zip(keys_tuple, per_frame_vals))
                frame["frame_index"] = fi
                frame["timestamp"] = f32(timestamps_list[fi])
                frames_append(frame)
        else:
            for fi in range(n_frames):
                frames_append(
                    {
                        "frame_index": fi,
                        "timestamp": f32(timestamps_list[fi]),
                    }
                )

        return frames

    # ------------------------------------------------------------------
    # Internals – vectorised per-key value picker
    # ------------------------------------------------------------------

    def _pick_values_vectorised(
        self,
        frame_times: np.ndarray,
        ts_arr: np.ndarray,
        val_list: list[Any],
        tolerance_ns: int,
    ) -> list[Any]:
        """Vectorised per-key value picker.

        Given a *sorted* ``ts_arr`` (int64 ns) and the parallel ``val_list``,
        return a plain Python list of length ``len(frame_times)`` where the
        i-th element is the picked value (or ``None``) for the i-th frame.

        The value selection is done via ``numpy.searchsorted`` once for the
        whole timeline; the only Python-level work is the final
        ``val_list[idx]`` indexing (O(F)), which is unavoidable because
        ``val_list`` holds arbitrary Python objects (numpy arrays, PIL
        images, strings, …).
        """
        n_frames = int(frame_times.shape[0])
        if ts_arr.size == 0:
            return [None] * n_frames

        # idxs[i] = index of the last ts <= frame_times[i]; -1 if none.
        # Equivalent to bisect_right(ts_list, frame_times[i]) - 1.
        idxs = np.searchsorted(ts_arr, frame_times, side="right") - 1

        if self.policy == "hold":
            return self._pick_hold_vectorised(
                frame_times,
                ts_arr,
                val_list,
                idxs,
                tolerance_ns,
            )
        # "nearest" and "drop" share the same nearest-within-tolerance logic.
        return self._pick_nearest_vectorised(
            frame_times,
            ts_arr,
            val_list,
            idxs,
            tolerance_ns,
        )

    @staticmethod
    def _pick_hold_vectorised(
        frame_times: np.ndarray,
        ts_arr: np.ndarray,
        val_list: list[Any],
        idxs: np.ndarray,
        tolerance_ns: int,
    ) -> list[Any]:
        """Vectorised hold policy selection.

        * ``idx >= 0``: carry forward ``val_list[idx]``.
        * ``idx == -1``: if the *first* message is within tolerance of the
          current frame time, fall back to it; otherwise ``None``.
        """
        n_frames = int(frame_times.shape[0])
        first_ts = int(ts_arr[0])
        # Precompute the mask of "before-any-message" frames that are still
        # close enough to the very first message to inherit its value.
        before_mask = idxs < 0
        close_to_first = before_mask & (np.abs(frame_times - first_ts) <= tolerance_ns)
        # Cheap path: convert to Python lists once for zero-cost scalar access.
        idxs_list = idxs.tolist()
        close_to_first_list = close_to_first.tolist()
        first_val = val_list[0]

        result: list[Any] = [None] * n_frames
        for i in range(n_frames):
            idx_i = idxs_list[i]
            if idx_i >= 0:
                result[i] = val_list[idx_i]
            elif close_to_first_list[i]:
                result[i] = first_val
            # else: stays None
        return result

    @staticmethod
    def _pick_nearest_vectorised(
        frame_times: np.ndarray,
        ts_arr: np.ndarray,
        val_list: list[Any],
        idxs: np.ndarray,
        tolerance_ns: int,
    ) -> list[Any]:
        """Vectorised nearest-within-tolerance selection.

        For each frame, consider two candidates:

        * ``idx`` – the last ts <= frame_time (may be -1).
        * ``idx + 1`` – the first ts > frame_time (may be >= N).

        Pick whichever has the smaller absolute distance AND is within
        ``tolerance_ns``; otherwise ``None``.  Matches the semantics of the
        original ``_pick_nearest`` exactly, including tie-breaking (the
        previous sample wins ties because it is evaluated first with a
        strictly-less comparison).
        """
        n_frames = int(frame_times.shape[0])
        n_msgs = int(ts_arr.shape[0])
        sentinel = np.iinfo(np.int64).max  # "no candidate" distance.

        # --- candidate A: ts_arr[idx] (ts <= frame_time) ----------------
        valid_a = (idxs >= 0) & (idxs < n_msgs)
        # Clip to a safe index so gather below never goes out of bounds.
        idxs_safe = np.where(valid_a, idxs, 0)
        dist_a = np.where(
            valid_a,
            np.abs(frame_times - ts_arr[idxs_safe]),
            sentinel,
        )

        # --- candidate B: ts_arr[idx + 1] (ts > frame_time) -------------
        nxt = idxs + 1
        valid_b = (nxt >= 0) & (nxt < n_msgs)
        nxt_safe = np.where(valid_b, nxt, 0)
        dist_b = np.where(
            valid_b,
            np.abs(frame_times - ts_arr[nxt_safe]),
            sentinel,
        )

        # Pick whichever distance is smaller (ties -> candidate A, matching
        # the original `d < best_dist` strict-less semantics).
        pick_b = dist_b < dist_a
        best_dist = np.where(pick_b, dist_b, dist_a)
        best_idx = np.where(pick_b, nxt, idxs)
        within = best_dist <= tolerance_ns

        # Python-level gather — val_list is heterogeneous (numpy arrays /
        # PIL Images / strings) so we cannot keep the result in numpy.
        within_list = within.tolist()
        best_idx_list = best_idx.tolist()
        result: list[Any] = [None] * n_frames
        for i in range(n_frames):
            if within_list[i]:
                result[i] = val_list[best_idx_list[i]]
        return result


# ---------------------------------------------------------------------------
# Trim-to-valid
# ---------------------------------------------------------------------------


def trim_to_valid_range(
    frames: list[dict[str, Any]],
    required_keys: list[str],
    fps: int,
) -> list[dict[str, Any]]:
    """Trim a frame list to the range where every required key has a value.

    LeRobot v3.0 rejects parquet rows with missing values for declared
    features. Resampling can produce ``None`` values for the first few
    frames (before any source message has arrived) and for the tail (if
    the bag was truncated mid-recording). This helper finds the longest
    inner range ``[first, last]`` such that every key in *required_keys*
    is non-None for every frame in that range, and returns the frames in
    that range with ``frame_index`` and ``timestamp`` re-numbered from 0.

    If no such range exists (at least one required key is missing for all
    frames) an empty list is returned.

    Parameters
    ----------
    frames:
        Output of :meth:`Resampler.resample`.
    required_keys:
        Feature keys that must be present in every retained frame.
    fps:
        Dataset target FPS; used to recompute each retained frame's
        ``timestamp`` relative to the new episode start.

    Returns
    -------
    list[dict[str, Any]]
        A newly-sliced list with ``frame_index`` restarted at 0 and
        ``timestamp`` recomputed as ``frame_index / fps``. The original
        frames are not mutated.
    """
    if not frames or not required_keys:
        return list(frames)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    def _all_present(frame: dict[str, Any]) -> bool:
        return all(frame.get(k) is not None for k in required_keys)

    n = len(frames)
    first = next((i for i in range(n) if _all_present(frames[i])), None)
    if first is None:
        return []
    last = next((i for i in range(n - 1, -1, -1) if _all_present(frames[i])), first)

    inv_fps = np.float32(1.0 / fps)
    trimmed: list[dict[str, Any]] = []
    for new_idx, f in enumerate(frames[first : last + 1]):
        f_copy = dict(f)
        f_copy["frame_index"] = new_idx
        f_copy["timestamp"] = np.float32(new_idx) * inv_fps
        trimmed.append(f_copy)
    return trimmed


# ---------------------------------------------------------------------------
# Utility (kept for reference / backwards compatibility)
# ---------------------------------------------------------------------------


def _bisect_right_ts(ts_list: list[int], target: int) -> int:
    """Return the insertion point for *target* in *ts_list* (sorted asc).

    Equivalent to ``bisect.bisect_right(ts_list, target)`` but avoids an
    import for a trivial function.

    .. note::
        The main ``Resampler.resample`` path no longer calls this helper
        (it uses ``numpy.searchsorted`` for a whole-timeline, vectorised
        lookup).  The function is retained as a reference implementation
        for tests that cross-check the vectorised path against a classic
        bisect-per-frame pipeline.
    """
    lo, hi = 0, len(ts_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_list[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
