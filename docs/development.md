# 開発

テストの回し方、依存関係、リポジトリ構成、Claude Code スキル、よくあるトラブルをまとめます。

## 目次

- [プロジェクト構成](#プロジェクト構成)
- [テスト](#テスト)
- [依存関係](#依存関係)
- [同梱スキル](#同梱スキル)
- [トラブルシューティング](#トラブルシューティング)

## プロジェクト構成

```
rosbag2lerobot/
├── src/rosbag2lerobot/
│   ├── __init__.py
│   ├── cli/                 # Click CLI パッケージ（コマンドごとに分割）
│   │   ├── __init__.py       # 再エクスポート（main / ヘルパ）
│   │   ├── main.py           # Click グループ＋コマンド登録
│   │   ├── _common.py        # 共有ヘルパ（logger, _detect_nvenc, _emit_report ...）
│   │   ├── convert.py        # convert コマンド＋パイプライン
│   │   ├── inspect.py        # inspect コマンド
│   │   ├── scaffold.py       # scaffold コマンド
│   │   ├── validate_config.py / validate_msg.py / validate_dataset.py
│   │   ├── audit_timestamps.py / quality_report.py
│   │   └── preview.py / push_to_hub.py / to_mcap.py
│   ├── config.py            # YAML ローダ＋データクラス
│   ├── reader.py            # rosbag リーダ（rosbags ライブラリ）
│   ├── resampler.py         # 可変レート → 固定 FPS リサンプラ
│   ├── writer.py            # LeRobot v3.0 ライター
│   ├── stats.py             # オンライン統計
│   ├── diagnostics.py       # F1/F3/F4 診断純関数（FPS 統計・image_size 検出・config↔bag 検証）
│   ├── audit.py             # F2 監査純関数（生成データセットのタイムスタンプ連続性検査）
│   ├── configs/             # サンプル設定
│   │   ├── robot_template.yaml
│   │   ├── so101.yaml
│   │   └── hsr.yaml
│   └── decoders/
│       ├── __init__.py      # デコーダレジストリ
│       ├── builtin.py       # 標準 ROS2 メッセージ型
│       ├── image.py         # Image / CompressedImage
│       └── msg_parser.py    # 汎用 .msg フォールバック
├── msgs/                    # カスタム .msg 置き場
├── tests/
└── pyproject.toml
```

## テスト

```bash
source .venv/bin/activate
python -m pytest tests/ --tb=short -q

# カバレッジ
python -m pytest tests/ --cov=rosbag2lerobot --cov-report=term-missing

# 単一ファイル
python -m pytest tests/test_config.py -v

# 統合テスト（実 bag ダウンロードあり）
python -m pytest tests/ -m integration -v
```

### テストレイアウト

| ファイル | スコープ |
|---|---|
| `test_config.py` | YAML ロード＋バリデーション |
| `test_decoders.py` | メッセージ型ごとのデコーダ挙動 |
| `test_msg_parser.py` | 汎用 `.msg` パーサ |
| `test_resampler.py` | リサンプリング＋`trim_to_valid_range` |
| `test_stats.py` | Welford 統計と分位数 |
| `test_writer.py` | LeRobot v3.0 ライター出力 |
| `test_diagnostics.py` | F1/F3/F4 の純関数（`compute_topic_fps_report` / `detect_image_shape` / `validate_config_against_bag`）— 17 ケース |
| `test_audit.py` | F2 の純関数（`audit_episode_timestamps` / `AuditReport` / `BoundaryError`）— 14 ケース |
| `test_e2e.py` | 合成データの E2E |
| `test_e2e_<robot>.py` | ロボット別 E2E |
| `test_integration_real.py` | 実 bag 統合（要データ取得） |

## 依存関係

- [rosbags](https://pypi.org/project/rosbags/) — 純 Python の ROS2 bag リーダ（ROS2 インストール不要）。
- [pyarrow](https://pypi.org/project/pyarrow/) — parquet I/O。
- [numpy](https://pypi.org/project/numpy/) — 配列計算。
- [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) — 画像デコード。
- [Pillow](https://pypi.org/project/Pillow/) — 画像ハンドリング。
- [PyYAML](https://pypi.org/project/PyYAML/) — YAML パース。
- [click](https://pypi.org/project/click/) — CLI フレームワーク。
- [tqdm](https://pypi.org/project/tqdm/) — 進捗バー。
- [ffmpeg](https://ffmpeg.org/) — 動画エンコード（システム依存）。

Python 環境管理は **必ず `uv`** を使う（素の `pip install` は禁止）。

## 同梱スキル

`.claude/skills/` 配下に繰り返しワークフローがスキルとして入っています。Claude Code セッションでスラッシュコマンドとして呼べます。

| スキル | 用途 |
|---|---|
| `/nvenc-smoke` | NVENC の可用性と `h264_nvenc` vs `libx264` のスループットを合成クリップで測定。新マシンで最初に叩くチェック。 |
| `/bench-convert` | `convert` を 3 コーデック（libx264 / h264_nvenc / av1_nvenc + UHQ）で走らせ、壁時計・ピーク RSS・MP4 サイズを比較レポート。 |
| `/verify-dataset` | 生成されたデータセットの構造検査（`.staging` 削除済み、`info.json` のコーデックラベル整合性、parquet 行数、mp4 フレーム数、`from_timestamp`/`to_timestamp` の単調性）。 |
| `/quality-cycle` | `uv run ruff check --fix` → `uv run pytest -q` → E2E の max-RSS 測定を 1 サイクルで実行。 |

各スキルは独自の `SKILL.md`（フロントマター）と `run.sh` / インライン `uv run python` を持ちます。詳しくは `.claude/skills/README.md`。

## トラブルシューティング

### トピックが見つからない

```
ValueError: Topic /right/joint_states not found in bag
```

YAML のトピック名と bag の実トピック名が一致していません。

1. `rosbag2lerobot inspect --bags /path/to/ros_sample_bag/` で実トピック一覧を確認。
2. YAML のトピック名を修正（名前空間は大文字小文字を区別する）。
3. 欠けることがあるトピックは `optional: true` にする。

### カスタムメッセージのデシリアライズに失敗

```
TypeError: Cannot deserialize my_robot_msgs/msg/MyCustomMsg
```

カスタム `.msg` が登録されていない。

1. `msgs/<robot>/` 配下に `.msg` を置く。
2. `custom_msgs` セクションに追加。
3. `rosbag2lerobot validate-msg --msg <path>` で構文確認。

### リサンプリングで null が多量発生

トピックのレートが低く、`tolerance_ms` 以内にメッセージが無い。

1. `tolerance_ms` を増やす（50 → 100 → 500 ms）。
2. `default_policy` を `hold` に切り替える（前値保持）。
3. 出力 FPS を下げる。

### ffmpeg が見つからない

```
FileNotFoundError: ffmpeg not found
```

```bash
sudo apt-get install ffmpeg   # Ubuntu / Debian
brew install ffmpeg           # macOS
```

### OOM（メモリ不足）

bag が長大・カメラが多いと RAM を食い切ります。

1. `--max-episodes` で 1 回あたりのエピソード数を制限。
2. `--workers 1` に落としてピーク RSS を単一エピソード分に抑える。
3. 事前に bag を短く分割する。

メモリ内訳の詳細は [`output_and_performance.md#メモリフットプリント`](output_and_performance.md#メモリフットプリント)。

### 画像サイズ不一致

```
ValueError: Image size mismatch: expected (480, 640), got (720, 1280)
```

YAML の `image_size` を `inspect` で確認した実解像度に合わせる。

### `lerobot-train: command not found`（関連ツール）

`lerobot-train` 等のコマンドは LeRobot 側の venv が activate されていないと PATH に出ません。`uv run lerobot-train ...` で実行するか、`source .venv/bin/activate` してから叩きます。
