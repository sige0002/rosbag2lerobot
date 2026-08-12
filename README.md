# rosbag2lerobot

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
git clone https://github.com/sige0002/rosbag2lerobot.git
cd rosbag2lerobot

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
rosbag2lerobot inspect --bags /path/to/my_bag/
```

### 2. Convert a single bag

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/my_bag/ \
  --output /path/to/output_dataset/
```

### 3. Convert multiple bags (one episode per bag)

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /path/to/output_dataset/ \
  --max-episodes 10 \
  --repo-id "myorg/my-dataset"
```

### 4. Dry run (validate without writing)

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /tmp/dry \
  --dry-run
```

### 5. Validate a custom .msg file

```bash
rosbag2lerobot validate-msg --msg msgs/my_robot/MyType.msg
```

### 6. Scaffold a config for an unknown robot

```bash
rosbag2lerobot scaffold --bags /path/to/all_bags/ -o configs/my_robot.yaml
```

### 7. Validate and score a generated dataset

```bash
rosbag2lerobot validate-dataset         --dataset /path/to/output_dataset/
rosbag2lerobot audit-timestamps         --dataset /path/to/output_dataset/
rosbag2lerobot validate-video-metadata  --dataset /path/to/output_dataset/  # pre-training: mp4 frames ↔ metadata
rosbag2lerobot quality-report           --dataset /path/to/output_dataset/ -o report.json
```

## Commands

| Command | Purpose |
|---|---|
| `inspect` | Show topics, message counts, time ranges (`--fps-stats` / `--suggest-image-size` for diagnostics). |
| `scaffold` | Auto-generate a starter `robot_config.yaml` from an unknown robot's bag(s); auto-runs `validate-config` unless `--no-validate`. |
| `convert` | Convert bag(s) to a LeRobot v3.0 dataset. |
| `validate-config` | Check a YAML config against a bag (topics / msg_type / image size). |
| `validate-dataset` | Verify a generated dataset conforms to LeRobot Dataset v3.0 (files, `info.json`, parquet schemas, cross-checks). |
| `quality-report` | Score data quality (null/NaN, out-of-range, freeze frames, video↔data reconciliation) into a 0..1 report. |
| `audit-timestamps` | Audit timestamp continuity of a generated dataset. |
| `validate-video-metadata` | Reproduce LeRobot's per-row video frame lookup via FFmpeg and check it fits the real mp4 (torch-free pre-training gate; `--strict` validates every data row against per-frame PTS). |
| `validate-msg` | Syntax-check a `.msg` file. |
| `preview` | Write a self-contained static HTML report (summary, quality, sample frames, stats) for a dataset; read-only, no server. |
| `push-to-hub` | Upload a generated dataset to the HuggingFace Hub with an auto-generated dataset card (opt-in; `--dry-run` plans only). |
| `to-mcap` | Convert ROS1 `.bag` recordings to ROS2 MCAP bags. |

Full options for every command are in
[`docs/cli_reference.md`](docs/cli_reference.md).

**`convert` progress and run metadata.** `convert` shows a tqdm ETA progress
bar when stdout is a terminal. `--json` emits the run's `job_summary` to stdout,
`--quiet` suppresses the bar and INFO logs, and `--skip-failed` records a failed
bag and continues (the dataset finalizes from the good episodes) instead of
aborting. Every run also writes two files under `meta/`: `conversion_log.json`
(provenance — input SHA256, per-bag frame counts/timing, codec, config snapshot
+ hash, rosbag2lerobot/ffmpeg versions, run timestamp) and `job_summary.json`
(success/fail counts, throughput, byte sizes, per-worker / per-episode
breakdown). `--manifest-extra FILE` merges your own JSON object (job id,
ticket, operator…) into `conversion_log.json`; keys that collide with the
fields rosbag2lerobot writes itself are ignored, so the manifest cannot be made
to misreport how the dataset was produced.

**Progress without a terminal.** When stdout is not a TTY (a pipe, a log file,
`docker logs`) no bar is drawn — it would only be carriage-return soup — and
plain lines like `episode 3/40: 62% (12000/19500 messages)` are logged instead,
at most every 10% of an episode or every 30 s. In parallel, a running
conversion maintains `meta/progress.json`, rewritten atomically every couple of
seconds:

```json
{ "episode_index": 3, "episode_total": 40, "messages_done": 12000,
  "messages_total": 19500, "updated_at": "2026-08-13T04:05:06.123456+00:00" }
```

`messages_total` comes from the bag's `metadata.yaml` (`null` when the bag does
not report one). With `--workers > 1` the file advances once per completed
episode rather than continuously, because episodes decode in separate
processes. It is transient run state, not part of the dataset: a successful run
deletes it, so a `progress.json` left behind means the run died — and says how
far it got.

**Clock sanity.** If a message's `header.stamp` diverges from its bag receive
time by more than `timestamps.max_header_receive_skew_ms` (default 60 s), the
episode fails with a message naming the topic and the observed skew, instead of
producing a dataset whose timing is silently wrong — the usual cause is an
unsynchronised clock on the recording or publishing host. Raise the threshold
or set it to `null` to disable the check; see
[`docs/configuration.md`](docs/configuration.md).

**`--json` on report commands.** `validate-config`, `validate-dataset`,
`quality-report`, `audit-timestamps`, `validate-video-metadata`, `inspect`,
`validate-msg`, and `to-mcap` all accept `--json` to print their report dict to
stdout (suppressing the human summary). The file flags (`--json-out`,
`-o`/`--report`) still work.

**Config: splits, TF features, typo detection.** A `split:` block in
`robot_config.yaml` sets `train`/`val`/`test` ratios (default `train=1.0`, a
single split that is byte-identical to the legacy output) and `min_length` (drop
episodes shorter than N frames); `info.json["splits"]` then holds contiguous
ranges. A feature can be looked up across the TF tree with `frame_from` /
`frame_to` (set `topic: /tf`, `msg_type: tf2_msgs/msg/TFMessage`), optionally
yielding euler angles via the `orientation.euler_xyz` selector. Unknown keys in
the config now raise an error with a "did you mean: X?" suggestion; run
`validate-config --suggest-fixes` for copy-pasteable `image_size` fixes. See
[`docs/cli_reference.md`](docs/cli_reference.md) for details.

**`convert --resume`** is a safe re-run guard (P0 scope only): converting into
a non-empty `--output` *without* `--resume` now aborts to avoid corrupting an
existing dataset. With `--resume`, a finalized dataset is a no-op and a crashed
(non-finalized) output is wiped and rebuilt. Skipping already-converted
episodes is a planned P1 feature, not yet implemented.

**Depth decoders.** `compressedDepth` (RVL-compressed 16-bit depth, e.g. the
HSR head depth) and raw `sensor_msgs/msg/Image` with `mono16` encoding are
supported. Depth is stored as an 8-bit grayscale video feature with
`video.is_depth_map: true` — note this is lossy 8-bit, so precision is reduced.
`configs/hsr.yaml` has a commented "Depth (optional)" block showing how to
enable it.

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
| `--video-codec`    | `auto` (default) picks `h264_nvenc` if NVENC is *usable*, else `libx264`. Accepts `libx264`, `libsvtav1`, `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`. |
| `--gpu / --no-gpu` | Force GPU (NVENC) on/off. Default: auto-detect. NVENC must be listed by ffmpeg **and** pass a one-frame test encode, so an ffmpeg that lists `h264_nvenc` but cannot reach the driver (a container without `--gpus all`) falls back to `libx264` with a logged reason instead of dying at the first frame. `--gpu` then fails at startup rather than mid-run. |
| `--ffmpeg-preset`  | Override ffmpeg preset. Codec-specific (`veryfast` for libx264, `p4` for NVENC, `8` for libsvtav1).  |
| `--ffmpeg-crf`     | Override quality. Mapped to `-crf` for CPU codecs and `-cq` for NVENC codecs.                        |
| `--dry-run`        | Validate config and bags without writing output.                                                     |
| `--repo-id`        | HuggingFace repo ID for the dataset.                                                                 |
| `--json`           | Emit the run's `job_summary` JSON to stdout (suppresses the human `Done.` logs).                     |
| `--quiet`          | Suppress the progress bar and INFO chatter.                                                          |
| `--skip-failed`    | Record per-episode failures and continue (finalize from good episodes). Default: a failure aborts.   |
| `--manifest-extra` | JSON file whose object is merged into `meta/conversion_log.json`. Keys owned by rosbag2lerobot win.  |
| `-v / --verbose`   | Enable debug logging.                                                                                |

## Example Commands

### NVENC (GPU) auto-detect

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto
```

### Force CPU (libx264, reproducible)

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --no-gpu
```

### Parallel workers (4 episodes at a time)

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto --workers 4
```

### AV1 UHQ on Blackwell / DGX Spark

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec av1_nvenc --ffmpeg-preset p4
```

More tuning knobs and a throughput comparison table are in
[`docs/output_and_performance.md`](docs/output_and_performance.md).

## Docker

The repo ships a [`Dockerfile`](Dockerfile) (python:3.11-slim + ffmpeg + the
CLI). Bags, config, and output are mounted at run time, and the image's
entrypoint is `rosbag2lerobot`, so a container invocation is the CLI invocation
with the paths swapped for mount points:

```bash
docker build -t rosbag2lerobot .

docker run --rm \
  -v "$PWD/robot_config.yaml:/config.yaml:ro" \
  -v "$PWD/bags:/bags:ro" \
  -v "$PWD/out:/out" \
  rosbag2lerobot convert --config /config.yaml --bags /bags --output /out
```

Add `-u "$(id -u):$(id -g)"` to have the dataset written as your own user
rather than root. Any other subcommand works the same way
(`docker run --rm -v "$PWD/bags:/bags:ro" rosbag2lerobot inspect --bags /bags`).
NVENC needs the NVIDIA container runtime (`--gpus all`) and is not part of this
image's promise; without it the codec `auto` falls back to `libx264`. Because
the container has no TTY by default, `convert` logs plain progress lines and
maintains `meta/progress.json` instead of drawing a bar.

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
