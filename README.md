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

### 6. Scaffold a config for an unknown robot

```bash
bagel scaffold --bags /path/to/all_bags/ -o configs/my_robot.yaml
```

### 7. Validate and score a generated dataset

```bash
bagel validate-dataset --dataset /path/to/output_dataset/
bagel quality-report   --dataset /path/to/output_dataset/ -o report.json
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
| `validate-msg` | Syntax-check a `.msg` file. |
| `preview` | Write a self-contained static HTML report (summary, quality, sample frames, stats) for a dataset; read-only, no server. |
| `push-to-hub` | Upload a generated dataset to the HuggingFace Hub with an auto-generated dataset card (opt-in; `--dry-run` plans only). |
| `to-mcap` | Convert ROS1 `.bag` recordings to ROS2 MCAP bags. |
| `ui` | Launch a localhost (127.0.0.1-only) control UI for the browse → scaffold → convert → quality loop. Frontend/backend separated; every action shows its equivalent `bagel ...` CLI command. |

Full options for every command are in
[`docs/cli_reference.md`](docs/cli_reference.md).

**`convert` progress and run metadata.** `convert` shows a tqdm ETA progress
bar by default. `--json` emits the run's `job_summary` to stdout, `--quiet`
suppresses the bar and INFO logs, and `--skip-failed` records a failed bag and
continues (the dataset finalizes from the good episodes) instead of aborting.
Every run also writes two files under `meta/`: `conversion_log.json` (provenance
— input SHA256, per-bag frame counts/timing, codec, config snapshot + hash,
bagel/ffmpeg versions, run timestamp) and `job_summary.json` (success/fail
counts, throughput, byte sizes, per-worker / per-episode breakdown).

**`--json` on report commands.** `validate-config`, `validate-dataset`,
`quality-report`, `audit-timestamps`, `inspect`, `validate-msg`, and `to-mcap`
all accept `--json` to print their report dict to stdout (suppressing the human
summary). The file flags (`--json-out`, `-o`/`--report`) still work.

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

## Local UI

`bagel ui` launches a **localhost (127.0.0.1-only) control UI** that walks the
browse bag → scaffold/edit config → convert (with progress) → quality + preview
loop in the browser. The frontend and backend are **separated**: a Python
backend (stdlib `http.server`, no new dependencies) holds all privilege behind
an allow-listed JSON API, while the TypeScript + HTML frontend (in
[`ui/`](ui/README.md)) is presentation-only. The **CLI is the source of truth** —
every UI action shows (and lets you copy) the equivalent `bagel ...` command.

It runs as a host process (not Docker), so it can reach bags anywhere under the
allow-listed roots. Security: it binds `127.0.0.1` only, mints a per-launch
session token (printed in the URL as `?token=...`), and confines all filesystem
access to the `--bags-root` / `--output-root` directories (path traversal is
blocked). It is not network-exposed.

Build the frontend once, then launch:

```bash
# 1. Build the frontend bundle (needs node/npm; only for building)
cd ui && npm install && npm run build && cd ..

# 2. Launch the UI (Python only at runtime)
bagel ui \
  --bags-root /path/to/bags \
  --output-root /path/to/output
```

Frontend build details are in [`ui/README.md`](ui/README.md); full `bagel ui`
options, the security model, and examples are in
[`docs/cli_reference.md`](docs/cli_reference.md#ui).

### Using the UI

In the browse panel, navigate (click `[dir]`) until you see `[bag]` entries, then
**tick a bag's checkbox to select it** — the **Inspect** / scaffold buttons only
become clickable once at least one bag is selected (entering a folder is not
selecting). Point `--bags-root` at the directory that *directly contains* your
bag folders so they appear as `[bag]` immediately; a folder one level too high
shows only `[dir]` and nothing is selectable until you click in. ROS1 `.bag`
recordings are not shown as bags — convert them with `bagel to-mcap` first.

### Remote access (server on an SSH host)

`bagel ui` binds `127.0.0.1` only, so from another machine reach it through an
SSH port-forward (don't expose it on the network):

```bash
# on the remote (where the bags live): note the printed URL + token
bagel ui --bags-root /abs/bags --output-root /abs/out --port 8765 --no-open

# on your laptop: forward the port (keep this terminal open)
ssh -N -L 8765:127.0.0.1:8765 user@remote

# then open the printed URL in your laptop browser
http://127.0.0.1:8765/?token=XXXX
```

Open it via **`127.0.0.1` or `localhost` only** — a `127.0.0.1` URL points at the
machine you open it on, and requests carrying any other `Host` header are
rejected with `403 Host not allowed.` (DNS-rebinding defence). Copy the full
`?token=...`: the page itself loads without it, but the API calls need it.

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
| `--json`           | Emit the run's `job_summary` JSON to stdout (suppresses the human `Done.` logs).                     |
| `--quiet`          | Suppress the progress bar and INFO chatter.                                                          |
| `--skip-failed`    | Record per-episode failures and continue (finalize from good episodes). Default: a failure aborts.   |
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
