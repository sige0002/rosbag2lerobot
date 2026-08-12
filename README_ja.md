# rosbag2lerobot

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
git clone https://github.com/sige0002/rosbag2lerobot.git
cd rosbag2lerobot

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
rosbag2lerobot inspect --bags /path/to/my_bag/
```

### 2. 単一 bag の変換

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/my_bag/ \
  --output /path/to/output_dataset/
```

### 3. 複数 bag の変換（bag ごとに1エピソード）

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /path/to/output_dataset/ \
  --max-episodes 10 \
  --repo-id "myorg/my-dataset"
```

### 4. Dry run（書き込みなしで検証）

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/all_bags/ \
  --output /tmp/dry \
  --dry-run
```

### 5. カスタム .msg ファイルの検証

```bash
rosbag2lerobot validate-msg --msg msgs/my_robot/MyType.msg
```

### 6. 未知ロボットの config を雛形生成

```bash
rosbag2lerobot scaffold --bags /path/to/all_bags/ -o configs/my_robot.yaml
```

### 7. 生成データセットの検証とスコアリング

```bash
rosbag2lerobot validate-dataset         --dataset /path/to/output_dataset/
rosbag2lerobot audit-timestamps         --dataset /path/to/output_dataset/
rosbag2lerobot validate-video-metadata  --dataset /path/to/output_dataset/  # 学習前: mp4 フレーム数 ↔ metadata
rosbag2lerobot quality-report           --dataset /path/to/output_dataset/ -o report.json
```

## コマンド一覧

| コマンド | 用途 |
|---|---|
| `inspect` | トピック一覧・メッセージ数・時間範囲を表示（`--fps-stats` / `--suggest-image-size` で診断）。 |
| `scaffold` | 未知ロボットの bag から `robot_config.yaml` の雛形を自動生成。`--no-validate` 以外では `validate-config` を自動実行。 |
| `convert` | bag を LeRobot v3.0 データセットに変換。 |
| `validate-config` | YAML config と bag の整合性を検証（トピック / msg_type / 画像サイズ）。 |
| `validate-dataset` | 生成データセットが LeRobot Dataset v3.0 構造に準拠するか検証（ファイル / `info.json` / parquet スキーマ / 件数突き合わせ）。 |
| `quality-report` | データ品質をスコアリング（null/NaN・範囲外・フリーズフレーム・動画↔データ整合）し 0..1 のレポートに集約。 |
| `audit-timestamps` | 生成データセットのタイムスタンプ連続性を監査。 |
| `validate-video-metadata` | LeRobot の学習時フレーム参照を FFmpeg ベースで再現し実 mp4 と照合（torch 非依存の学習前チェック。`--strict` で全 data 行を PTS 照合）。 |
| `validate-msg` | `.msg` ファイルの構文チェック。 |
| `preview` | データセットの自己完結型 HTML レポート（サマリ・品質・サンプルフレーム・統計）を生成。読み取り専用・サーバ不要。 |
| `push-to-hub` | 生成データセットを HuggingFace Hub にアップロードし、データセットカードを自動生成（任意機能。`--dry-run` は計画のみ）。 |
| `to-mcap` | ROS1 `.bag` を ROS2 MCAP bag に変換。 |

各コマンドの全オプションは
[`docs/cli_reference.md`](docs/cli_reference.md) を参照してください。

**`convert` の進捗とランメタデータ.** `convert` は stdout が端末のとき tqdm の
ETA 付き進捗バーを表示します。`--json` はそのランの `job_summary` を stdout に
出力し、`--quiet` は進捗バーと INFO ログを抑制、`--skip-failed` は失敗した bag を
記録して処理を継続します（中断せず、成功エピソードからデータセットを確定）。各ランは
`meta/` 配下に 2 ファイルを書き出します: `conversion_log.json`（来歴 — 入力
SHA256・bag 毎のフレーム数/所要時間・コーデック・config スナップショット+ハッシュ・
rosbag2lerobot/ffmpeg バージョン・実行時刻）と `job_summary.json`（成功/失敗件数・スループット・
バイト数・ワーカー別/エピソード別の内訳）。`--manifest-extra FILE` を付けると、
任意の JSON オブジェクト（ジョブ ID・チケット・作業者など）を
`conversion_log.json` にマージできます。rosbag2lerobot 自身が書くキーと衝突した
分は無視されるため、マニフェストが「どう作られたか」を偽ることはできません。

**端末が無いときの進捗.** stdout が TTY でないとき（パイプ・ログファイル・
`docker logs`）は進捗バーを描画せず（キャリッジリターンの塊になるだけなので）、
`episode 3/40: 62% (12000/19500 messages)` のようなプレーンな行をログに出します
（エピソードの 10% ごと、または 30 秒ごとのいずれか早い方）。並行して、実行中の
変換は `meta/progress.json` を数秒おきに atomic に書き換えます:

```json
{ "episode_index": 3, "episode_total": 40, "messages_done": 12000,
  "messages_total": 19500, "updated_at": "2026-08-13T04:05:06.123456+00:00" }
```

`messages_total` は bag の `metadata.yaml` 由来です（取得できない場合は `null`）。
`--workers > 1` ではエピソードが別プロセスでデコードされるため、連続ではなく
エピソード完了ごとの更新になります。このファイルは実行中の一時状態であって
データセットの一部ではありません。成功したランは削除するので、`progress.json`
が残っていればそのランは死んでいて、どこまで進んだかを示しています。

**時計の健全性.** メッセージの `header.stamp` と bag 受信時刻の乖離が
`timestamps.max_header_receive_skew_ms`（既定 60 秒）を超えると、そのエピソードは
トピック名と実測スキューを示して失敗します。タイミングが静かに壊れたデータセット
を作らないためで、原因はたいてい収録ホストや publisher 側の時刻同期漏れです。
しきい値を上げる、または `null` で無効化できます。詳細は
[`docs/configuration.md`](docs/configuration.md) を参照してください。

**レポート系コマンドの `--json`.** `validate-config` / `validate-dataset` /
`quality-report` / `audit-timestamps` / `validate-video-metadata` / `inspect` /
`validate-msg` / `to-mcap` はいずれも `--json` でレポート dict を stdout に出力
できます（人間向け summary は抑制）。従来のファイル出力フラグ（`--json-out`、
`-o`/`--report`）も引き続き使えます。

**config: split・TF 特徴量・タイポ検出.** `robot_config.yaml` の `split:` ブロック
で `train`/`val`/`test` の比率（デフォルト `train=1.0`。単一 split で従来出力と
バイト一致）と `min_length`（N フレーム未満のエピソードを除外）を指定できます。
`info.json["splits"]` は連続レンジの集合になります。特徴量は `frame_from` /
`frame_to`（`topic: /tf`、`msg_type: tf2_msgs/msg/TFMessage` を併記）で TF tree
から姿勢を引け、`orientation.euler_xyz` セレクタで euler 角に変換できます。config
の未知キーは「did you mean: X?」サジェスト付きでエラーになります。`image_size`
不一致の修正例は `validate-config --suggest-fixes` で出力されます。詳細は
[`docs/cli_reference.md`](docs/cli_reference.md) を参照してください。

**`convert --resume`** は安全な再実行ガードです（P0 範囲のみ）。`--resume`
なしで非空の `--output` に変換しようとすると、既存データセット破壊を防ぐため
中断します。`--resume` を付けると、完成済みデータセットは何もせず（no-op）、
途中でクラッシュした（未確定の）出力は削除して再変換します。変換済み
エピソードのスキップは P1 予定の機能で、まだ未実装です。

**深度デコーダ.** `compressedDepth`（RVL 圧縮の 16bit 深度、例: HSR ヘッド
深度）と raw `sensor_msgs/msg/Image` の `mono16` エンコーディングに対応しま
した。深度は `video.is_depth_map: true` を持つ 8bit グレースケール動画として
保存されます（8bit のため精度は低下します）。`configs/hsr.yaml` に有効化方法
を示すコメント付き「Depth (optional)」ブロックがあります。

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
| `--json`             | そのランの `job_summary` JSON を stdout に出力（人間向け `Done.` ログは抑制）。                                  |
| `--quiet`            | 進捗バーと INFO ログを抑制。                                                                                    |
| `--skip-failed`      | エピソード単位の失敗を記録して継続（成功分から確定）。デフォルトは失敗で中断。                                   |
| `--manifest-extra`   | JSON ファイルの内容を `meta/conversion_log.json` にマージ。rosbag2lerobot 所有のキーが優先（衝突分は無視）。      |
| `-v / --verbose`     | デバッグログを有効化。                                                                                          |

## サンプルコマンド集

### NVENC（GPU）自動検出

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto
```

### CPU 強制（libx264、再現性重視）

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --no-gpu
```

### 並列ワーカー（4 エピソード同時）

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec auto --workers 4
```

### Blackwell / DGX Spark での AV1 UHQ

```bash
rosbag2lerobot convert \
  --config configs/hsr.yaml \
  --bags /path/to/bags --output /path/to/out \
  --video-codec av1_nvenc --ffmpeg-preset p4
```

チューニング項目とスループット比較の詳細は
[`docs/performance.md`](docs/performance.md) にあります。

## Docker

リポジトリには [`Dockerfile`](Dockerfile)（python:3.11-slim + ffmpeg + CLI）が
入っています。bag・config・出力先は実行時にマウントし、イメージの entrypoint は
`rosbag2lerobot` なので、パスをマウント先に置き換えるだけで CLI と同じ使い方が
できます:

```bash
docker build -t rosbag2lerobot .

docker run --rm \
  -v "$PWD/robot_config.yaml:/config.yaml:ro" \
  -v "$PWD/bags:/bags:ro" \
  -v "$PWD/out:/out" \
  rosbag2lerobot convert --config /config.yaml --bags /bags --output /out
```

`-u "$(id -u):$(id -g)"` を付けると出力が root ではなく自分の所有になります。
他のサブコマンドも同様です
（`docker run --rm -v "$PWD/bags:/bags:ro" rosbag2lerobot inspect --bags /bags`）。
NVENC は NVIDIA container runtime（`--gpus all`）が必要で、このイメージの保証範囲
外です（無ければ codec `auto` は `libx264` にフォールバックします）。コンテナには
既定で TTY が無いため、`convert` はバーを描画せずプレーンな進捗ログと
`meta/progress.json` を使います。

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
