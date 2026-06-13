# bagel

ROS2 rosbag ファイルを
[LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot) 形式に変換
するツールです。設定ファイル（`robot_config.yaml`）一つで、トピックと
フィーチャーのマッピング、メッセージのデコード、リサンプリングの挙動を
すべて制御できます。**ROS2 のインストールは不要**です。

English version: [README.md](README.md)

## インストール

Python 3.11+ と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
# リポジトリをクローン
git clone https://github.com/sige0002/bagel.git
cd bagel

# 仮想環境を作成してインストール
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# 開発用依存（pytest、ruff 等）
uv pip install -e ".[dev]"
```

動画エンコードのためランタイムに ffmpeg が必要です:

```bash
sudo apt-get install ffmpeg   # Ubuntu / Debian
brew install ffmpeg           # macOS
```

**NVENC（オプション）.** Ubuntu 24.04 標準の `ffmpeg` パッケージには
NVENC ビルドが含まれているため、ソースからのビルドは不要です。
`ffmpeg -hide_banner -encoders | grep nvenc` で確認できます。追加の
Python パッケージ（PyTorch 等）は一切不要です。詳細は
[`docs/performance.md`](docs/performance.md) を参照してください。

## クイックスタート

### 1. rosbag の確認

```bash
bagel inspect --bags /path/to/my_bag/
```

### 2. 単一 bag の変換

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/my_bag/ \
  --output /path/to/output_dataset/
```

### 3. 複数 bag の変換（bag ごとに1エピソード）

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /path/to/output_dataset/ \
  --max-episodes 10 \
  --repo-id "myorg/my-dataset"
```

### 4. Dry run（書き込みなしで検証）

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /tmp/dry \
  --dry-run
```

### 5. カスタム .msg ファイルの検証

```bash
bagel validate-msg --msg msgs/my_robot/MyType.msg
```

## CLI オプション一覧

| オプション            | 説明                                                                                                          |
|----------------------|---------------------------------------------------------------------------------------------------------------|
| `--config`           | `robot_config.yaml` へのパス（必須）。                                                                         |
| `--bags`             | bag ディレクトリ、または bag 群の親ディレクトリ（必須）。                                                        |
| `--output`           | 出力データセットディレクトリ（必須）。                                                                           |
| `--task`             | config のタスク名をオーバーライド。                                                                             |
| `--fps`              | config の FPS をオーバーライド。                                                                                |
| `--max-episodes`     | 変換するエピソード数の上限。                                                                                    |
| `--workers`          | 並列ワーカー数（デフォルト: 1）。                                                                               |
| `--video-codec`      | デフォルト `auto`（NVENC 検出時 `h264_nvenc`、なければ `libx264`）。他に `libx264` / `libsvtav1` / `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` を指定可。 |
| `--gpu / --no-gpu`   | GPU（NVENC）の強制 ON/OFF。デフォルトは `ffmpeg -encoders` による自動判定。                                    |
| `--ffmpeg-preset`    | ffmpeg の preset 上書き（libx264 は `veryfast`、NVENC は `p4`、libsvtav1 は `8` 等）。                          |
| `--ffmpeg-crf`       | 画質の上書き。CPU コーデックは `-crf`、NVENC は `-cq` にマップ。                                                |
| `--dry-run`          | config と bag の検証のみ（出力なし）。                                                                          |
| `--repo-id`          | HuggingFace リポジトリ ID。                                                                                    |
| `-v / --verbose`     | デバッグログを有効化。                                                                                          |

## サンプルコマンド集

### NVENC（GPU）自動検出

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto
```

### CPU 強制（libx264、再現性重視）

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --no-gpu
```

### 並列ワーカー（4 エピソード同時）

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto --workers 4
```

### Blackwell / DGX Spark での AV1 UHQ

```bash
bagel convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec av1_nvenc --ffmpeg-preset p4
```

チューニング項目とスループット比較の詳細は
[`docs/performance.md`](docs/performance.md) にあります。

## ドキュメント

まず索引: [`docs/README.md`](docs/README.md)

| トピック | ドキュメント |
|---|---|
| CLI オプション全一覧と実例集 | [`docs/cli_reference.md`](docs/cli_reference.md) |
| YAML 設定リファレンス（リサンプリング／必須トピック交差区間アライメント・`stamp_source`・スタール破棄）＋対応メッセージ型＋カスタムデコーダ拡張 | [`docs/configuration.md`](docs/configuration.md) |
| パイプラインと内部処理（型登録 / デコーダ / リサンプラ / `trim_to_valid` / ライター） | [`docs/architecture.md`](docs/architecture.md) |
| 出力フォーマット（LeRobot v3.0）とパフォーマンスチューニング | [`docs/output_and_performance.md`](docs/output_and_performance.md) |
| 新しいロボットの追加手順＋同梱設定一覧 | [`docs/adding_new_robot.md`](docs/adding_new_robot.md) |
| 開発（プロジェクト構成、テスト、スキル、トラブルシューティング） | [`docs/development.md`](docs/development.md) |

## ライセンス

Apache License 2.0 — 全文は [`LICENSE`](LICENSE) を参照してください。

## コントリビューション

[`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) を参照してください。
バグ報告・機能要望には [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE) の
テンプレートをご利用ください。
