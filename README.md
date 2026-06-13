# bagel

Convert ROS2 rosbag files to
[LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot) format.
The tool is fully config-driven: a single `robot_config.yaml` defines
every topic-to-feature mapping, message decoder, and resampling
policy. No ROS2 installation is required at runtime.

日本語版: [README_ja.md](README_ja.md)

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone https://github.com/sige0002/bagel.git
cd bagel

# Create venv and install
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Dev extras (pytest, ruff, etc.)
uv pip install -e ".[dev]"
```

ffmpeg is required at runtime for video encoding:

```bash
sudo apt-get install ffmpeg   # Ubuntu / Debian
brew install ffmpeg           # macOS
```

**NVENC (optional).** The default `ffmpeg` package on Ubuntu 24.04
already ships with NVENC support — nothing else needs to be built.
Verify with `ffmpeg -hide_banner -encoders | grep nvenc`. No extra
Python packages (PyTorch, etc.) are required. See
[`docs/performance.md`](docs/performance.md) for tuning.

## Quick Start

### 1. Inspect a rosbag

```bash
bagel inspect --bags /path/to/my_bag/
```

### 2. Convert a single bag

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/my_bag/ \
  --output /path/to/output_dataset/
```

### 3. Convert multiple bags (one episode per bag)

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /path/to/output_dataset/ \
  --max-episodes 10 \
  --repo-id "myorg/my-dataset"
```

### 4. Dry run (validate without writing)

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /tmp/dry \
  --dry-run
```

### 5. Validate a custom .msg file

```bash
bagel validate-msg --msg msgs/my_robot/MyType.msg
```

## CLI Options

| Option             | Description                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| `--config`         | Path to `robot_config.yaml` (required).                                                              |
| `--bags`           | Path to a bag directory or a parent directory containing bags (required).                            |
| `--output`         | Output directory for the dataset (required).                                                         |
| `--task`           | Override the task name from the config.                                                              |
| `--fps`            | Override the FPS from the config.                                                                    |
| `--max-episodes`   | Limit number of episodes to convert.                                                                 |
| `--workers`        | Number of parallel workers (default: 1).                                                             |
| `--video-codec`    | `auto` (default) picks `h264_nvenc` if NVENC is available, else `libx264`. Accepts `libx264`, `libsvtav1`, `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`. |
| `--gpu / --no-gpu` | Force GPU (NVENC) on/off. Default: auto-detect via `ffmpeg -encoders`.                               |
| `--ffmpeg-preset`  | Override ffmpeg preset. Codec-specific (`veryfast` for libx264, `p4` for NVENC, `8` for libsvtav1).  |
| `--ffmpeg-crf`     | Override quality. Mapped to `-crf` for CPU codecs and `-cq` for NVENC codecs.                        |
| `--dry-run`        | Validate config and bags without writing output.                                                     |
| `--repo-id`        | HuggingFace repo ID for the dataset.                                                                 |
| `-v / --verbose`   | Enable debug logging.                                                                                |

## Example Commands

### NVENC (GPU) auto-detect

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto
```

### Force CPU (libx264, reproducible)

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --no-gpu
```

### Parallel workers (4 episodes at a time)

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto --workers 4
```

### AV1 UHQ on Blackwell / DGX Spark

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec av1_nvenc --ffmpeg-preset p4
```

More tuning knobs and a throughput comparison table are in
[`docs/output_and_performance.md`](docs/output_and_performance.md).

## Documentation

Start with the index: [`docs/README.md`](docs/README.md).

| Topic | Document |
|---|---|
| CLI option reference and worked examples | [`docs/cli_reference.md`](docs/cli_reference.md) |
| YAML config reference (resampling, required-topic alignment, `stamp_source`, stale-message filtering) + supported messages + custom decoders | [`docs/configuration.md`](docs/configuration.md) |
| Pipeline architecture and internals (type registration, decoders, resampler, `trim_to_valid`, writer) | [`docs/architecture.md`](docs/architecture.md) |
| LeRobot v3.0 output format and performance tuning | [`docs/output_and_performance.md`](docs/output_and_performance.md) |
| Adding a new robot + bundled configs summary | [`docs/adding_new_robot.md`](docs/adding_new_robot.md) |
| Development (project tree, tests, skills, troubleshooting) | [`docs/development.md`](docs/development.md) |

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for the full text.

## Contributing

See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md). Please use the
provided [issue templates](.github/ISSUE_TEMPLATE) when reporting bugs
or requesting features.
