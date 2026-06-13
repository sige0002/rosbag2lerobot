"""bagel -- Convert ROS2 rosbag files to LeRobot Dataset v3.0 format.

This package provides a config-driven pipeline for converting ROS2 rosbag
recordings into datasets compatible with the LeRobot v3.0 specification.
It requires no ROS2 installation at runtime, relying on the ``rosbags``
library for bag reading and deserialization.

Key modules:

- ``cli``        -- Command-line interface (convert, inspect, validate-msg).
- ``config``     -- YAML configuration loader and typed dataclasses.
- ``reader``     -- Rosbag reader with custom message type support.
- ``resampler``  -- Multi-rate time synchronization to fixed FPS.
- ``writer``     -- LeRobot v3.0 dataset writer (parquet + video + metadata).
- ``stats``      -- Online per-feature statistics (Welford's algorithm).
- ``decoders``   -- Pluggable ROS2 message decoders (built-in + custom).

Version: 0.1.0
"""

__version__ = "0.1.0"
