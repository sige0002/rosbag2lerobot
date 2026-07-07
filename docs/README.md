# ドキュメント索引

`rosbag2lerobot` のドキュメントは以下の 7 本に整理されています。用途から辿ってください。

## 入口

| 目的 | ドキュメント |
|---|---|
| まず動かしたい／CLI の全オプションと実例 | [cli_reference.md](cli_reference.md) |
| YAML 設定を書きたい／オプションの意味を調べたい | [configuration.md](configuration.md) |
| bag ごとに task / subtask を指定したい | [task_json.md](task_json.md) |
| 新しいロボット（bag セット）を追加したい | [adding_new_robot.md](adding_new_robot.md) |

## 仕組みを知る

| 目的 | ドキュメント |
|---|---|
| パイプラインと内部処理（型登録・デコーダ・リサンプラ・`trim_to_valid`・ライター） | [architecture.md](architecture.md) |
| 出力フォーマット（LeRobot v3.0 ディレクトリ構成・メタデータ）とパフォーマンスチューニング | [output_and_performance.md](output_and_performance.md) |

## 開発・トラブルシュート

| 目的 | ドキュメント |
|---|---|
| テストの実行・依存関係・よくあるエラー | [development.md](development.md) |
| mp4 と parquet のフレーム数整合性に関する調査・回帰テスト | [frame_alignment_investigation_ja.md](frame_alignment_investigation_ja.md) |

## 索引（キーワード → 掲載ドキュメント）

| キーワード | 掲載先 |
|---|---|
| `--config`, `--bags`, `--output`, `--dry-run`, `--verbose` 等の CLI フラグ | [cli_reference.md](cli_reference.md) |
| `--video-codec` / `--gpu` / `--ffmpeg-preset` / `--ffmpeg-crf` | [cli_reference.md](cli_reference.md), [output_and_performance.md](output_and_performance.md) |
| `--fps-stats` / `--topics` / `--gap-threshold-ms` / `--head` / `--json-out` | [cli_reference.md](cli_reference.md) |
| `--suggest-image-size` / `--samples` | [cli_reference.md](cli_reference.md) |
| `observations` / `actions` / `selector` / `optional` | [configuration.md](configuration.md) |
| `resampling.default_policy` / `tolerance_ms` / `trim_to_valid` | [configuration.md](configuration.md), [architecture.md](architecture.md) |
| `custom_msgs` / `.msg` ファイル登録 | [configuration.md](configuration.md) |
| カスタムデコーダ追加（`@register_decoder`） | [configuration.md](configuration.md), [architecture.md](architecture.md) |
| LeRobot v3.0 出力レイアウト（`data/`, `videos/`, `meta/`） | [output_and_performance.md](output_and_performance.md) |
| NVENC / DGX Spark / Blackwell | [output_and_performance.md](output_and_performance.md) |
| `inspect`, `validate-msg`, `to-mcap` サブコマンド | [cli_reference.md](cli_reference.md) |
| `scaffold`, `validate-config`, `validate-dataset`, `quality-report`, `audit-timestamps` サブコマンド | [cli_reference.md](cli_reference.md) |
| `preview`, `push-to-hub` サブコマンド | [cli_reference.md](cli_reference.md) |
| `convert --resume`（再実行ガード） | [cli_reference.md](cli_reference.md) |
| `convert --json` / `--quiet` / `--skip-failed`（進捗・失敗継続） | [cli_reference.md](cli_reference.md) |
| `conversion_log.json` / `job_summary.json`（ランメタデータ） | [cli_reference.md](cli_reference.md) |
| レポート系コマンドの `--json`（stdout 出力） | [cli_reference.md](cli_reference.md) |
| `split:` 分割 / `frame_from`・`frame_to`（TF 特徴量）/ `euler_xyz` セレクタ / config タイポ検出 | [cli_reference.md](cli_reference.md), [configuration.md](configuration.md) |
| config 雛形生成 / データセット構造検証 / 品質スコアリング / HTML プレビュー / Hub アップロード | [cli_reference.md](cli_reference.md) |
| FPS 統計 / タイムスタンプ監査 / image_size 提案 / config-bag 整合検証 | [cli_reference.md](cli_reference.md), [architecture.md](architecture.md) |
| テスト実行・`uv sync` | [development.md](development.md) |
| エラー対処（トピック不一致・デシリアライズ失敗・OOM 等） | [development.md](development.md) |

## 同梱設定（参考）

`src/rosbag2lerobot/configs/` 以下にサンプル YAML が置かれています。

- `robot_template.yaml` — コメント入りテンプレート。新規ロボットはこれを雛形に。
- `so101.yaml` — 単腕・標準メッセージのみを使う最小構成の例。
- `hsr.yaml` — 全身移動マニピュレータの例（ROS1 bag は `rosbags-convert` で ROS2 化してから使用）。

以降のドキュメントではサンプル bag を `/path/to/ros_sample_bag/` と表記します。
