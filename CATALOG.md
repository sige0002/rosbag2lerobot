# CATALOG — 実行カタログ

rosbag2lerobot の実行方法早見表。詳細は [`README.md`](README.md) / [`README_ja.md`](README_ja.md) / [`docs/`](docs/README.md)。

## セットアップ

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"     # dev = pytest, pytest-cov
sudo apt-get install ffmpeg    # 動画エンコードに必須
```

## CLI（`rosbag2lerobot <command>`）

| コマンド | 用途 | 最小実行例 |
|---|---|---|
| `convert` | bag → LeRobot v3.0 データセット | `rosbag2lerobot convert --config configs/hsr.yaml --bags <bags>/ --output <out>/` |
| `inspect` | topic・件数・時間範囲を表示 | `rosbag2lerobot inspect --bags <bags>/` |
| `inspect --fps-stats` | topic ごとの FPS / 先頭末尾ラグ / ギャップ | `rosbag2lerobot inspect --bags <bags>/ --fps-stats` |
| `validate-config` | YAML と bag の整合性検査 | `rosbag2lerobot validate-config --config configs/hsr.yaml --bags <bags>/` |
| `validate-msg` | `.msg` 構文チェック | `rosbag2lerobot validate-msg --msg msgs/my_robot/MyType.msg` |
| `audit-timestamps` | 生成データセットの timestamp 連続性監査 | `rosbag2lerobot audit-timestamps --dataset <out>/` |
| `to-mcap` | ROS1 `.bag` → ROS2 MCAP に事前変換 | `rosbag2lerobot to-mcap <src>.bag -o <out>/` |

### convert の主要オプション

`--task` / `--fps`（config 上書き）、`--max-episodes N`、`--workers N`(並列)、
`--video-codec auto|libx264|libsvtav1|h264_nvenc|hevc_nvenc|av1_nvenc`、
`--gpu/--no-gpu`、`--ffmpeg-preset`、`--ffmpeg-crf`、`--dry-run`、`--repo-id`、`-v`。

```bash
# NVENC 自動検出 + 4 並列
rosbag2lerobot convert --config configs/hsr.yaml --bags <bags>/ --output <out>/ --video-codec auto --workers 4
# CPU 固定（再現性重視）
rosbag2lerobot convert --config configs/hsr.yaml --bags <bags>/ --output <out>/ --no-gpu
# 書き込まず検証のみ
rosbag2lerobot convert --config configs/hsr.yaml --bags <bags>/ --output /tmp/dry --dry-run
```

## テスト / リント

```bash
source .venv/bin/activate
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                       # 全テスト
uv run pytest tests/test_resampler.py -v   # 単一ファイル
uv run pytest -m integration -v        # 実 bag 統合（要 bagdata/）
uv run pytest -m slow / -m nvenc       # 重いテスト / NVENC 必須テスト（opt-in）
```

## データ配置

- 入力 bag: `bagdata/`（gitignore 済み）。`sample_bags/` は合成サンプル。
- 出力データセット: `output/`（gitignore 済み）。LeRobot v3.0 構造（`data/` `videos/` `meta/`）。
- ロボット設定: `configs/{hsr,so101,robot_template}.yaml`。
- カスタムメッセージ: `msgs/<robot>/*.msg`。

## 設定ファイル一覧

| config | ロボット |
|---|---|
| `configs/hsr.yaml` | HSR |
| `configs/so101.yaml` | SO-101 |
| `configs/robot_template.yaml` | 新規ロボット用テンプレート |

### robot config の主な `resampling` キー（詳細: [docs/configuration.md](docs/configuration.md)）

- `default_policy`(`hold`/`nearest`/`drop`) / `tolerance_ms` — 固定 FPS への補間方針。
- `align_to_required`（既定 true）— 全 required トピックがそろう交差区間で start/stop（終了トピックのずれ対策）。
- `max_stamp_delay_ms` — QoS スタール（古い latch 値）を `header.stamp` 遅延で破棄。feature 単位で上書き可。
- 各 feature の `stamp_source`(`header`/`receive`) — 採用タイムスタンプ源。
