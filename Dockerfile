# rosbag2lerobot — ROS2 rosbag (db3/MCAP) to LeRobot Dataset v3.0 converter.
#
# The image carries the CLI and ffmpeg, nothing else: bags, config and output
# are mounted at run time.
#
#   docker build -t rosbag2lerobot .
#   docker run --rm \
#       -v "$PWD/robot_config.yaml:/config.yaml:ro" \
#       -v "$PWD/bags:/bags:ro" \
#       -v "$PWD/out:/out" \
#       rosbag2lerobot convert --config /config.yaml --bags /bags --output /out
FROM python:3.11-slim

# ffmpeg encodes the dataset videos and is probed for NVENC support at startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Unbuffered: without it a conversion's progress lines sit in a pipe buffer and
# `docker logs` looks stalled for minutes at a time.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

# Neutral working directory so relative paths in a mounted config do not
# resolve into the installed source tree.
WORKDIR /work

ENTRYPOINT ["rosbag2lerobot"]
CMD ["--help"]
