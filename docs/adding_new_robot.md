# 新しいロボットの追加手順

新規ロボット（新しい bag セット）を `rosbag2lerobot` に統合する流れを、順に解説します。各ステップは前段を前提にしています。

## 目次

- [同梱設定一覧](#同梱設定一覧)
- [0. 前提条件](#0-前提条件)
- [1. bag の調査](#1-bag-の調査)
- [2. 特徴量レイアウトの決定](#2-特徴量レイアウトの決定)
- [3. カスタムメッセージ型の扱い](#3-カスタムメッセージ型の扱い)
- [4. `configs/<robot>.yaml` の作成](#4-configsrobotyaml-の作成)
- [5. dry-run](#5-dry-run)
- [6. 本番変換](#6-本番変換)
- [7. 出力の検証](#7-出力の検証)
- [8. 回帰テストの追加](#8-回帰テストの追加)
- [トラブルシューティング](#トラブルシューティング)

## 同梱設定一覧

`src/rosbag2lerobot/configs/` にある YAML は、新規設定を書き始めるときの参考になります。

| ファイル | 構成 |
|---|---|
| `robot_template.yaml` | 全フィールドにコメント付きのテンプレート。新規は原則これをコピーして始める。 |
| `so101.yaml` | 単腕・標準メッセージのみの最小構成例（カスタム型なし）。 |
| `hsr.yaml` | 移動ベース＋アーム＋グリッパの全身マニピュレータ例。ROS1 bag は先に `rosbags-convert input.bag --dst output_ros2/` で ROS2 化してから使う。 |

## 0. 前提条件

```bash
cd rosbag2lerobot
uv venv .venv && source .venv/bin/activate
uv sync                       # ランタイム依存
uv sync --extra dev           # + pytest, ruff など
sudo apt-get install ffmpeg   # 動画エンコード用
```

確認:

```bash
rosbag2lerobot --help
rosbag2lerobot inspect --bags /path/to/ros_sample_bag/
```

## 1. bag の調査

まず `inspect` で bag の中身を見ます。設定の情報源はこれ（推測で書かない）。

```bash
rosbag2lerobot inspect --bags /path/to/ros_sample_bag/ \
  | grep -v "(0 msgs)"
```

記録すべき項目:

- **状態ストリーム**（関節位置、EEF 姿勢、力／トルク）。
- **カメラ**とそのパブリッシュレート。
- **アクションストリーム**（コマンド）。
- **カスタムメッセージ型**（`.msg` 登録が要るかもしれないもの）。
- **疎なトピック**（エピソードあたりメッセージ数が少ないもの）— リサンプリング方針を左右する。

## 2. 特徴量レイアウトの決定

出力 LeRobot 特徴量キーを決めます。規則:

- `observation.images.<name>` — カメラ。
- `observation.state` または `observation.state.<part>` — 関節位置 / EEF 姿勢 / 力。
- `action` または `action.<part>` — コマンド。

双腕・複数部位があるなら `<part>` サフィックス（`right_arm`, `left_arm`, `right_gripper` 等）。単腕・単一部位ならサフィックス無しで十分。

## 3. カスタムメッセージ型の扱い

bag に非標準型が混じる場合は 3 通りから最小限のもので対応します。

### A. 何もしない

rosbag2 コネクションごとに埋め込まれた msgdef をリーダーが自動登録します（`reader._override_types_from_bag`）。多くの bag ではこれで十分。

### B. `.msg` ファイルをフォールバックとして登録

`msgs/<robot>/<TypeName>.msg` に置き、YAML に追記:

```yaml
custom_msgs:
  - msg_file: "msgs/my_robot/MyCustomMsg.msg"
    package: "my_robot_msgs"
```

パスは YAML のあるディレクトリからの相対で解決。bag に msgdef が埋め込まれていない古いライターへの保険。

検証:

```bash
rosbag2lerobot validate-msg --msg msgs/my_robot/MyCustomMsg.msg
```

### C. カスタムデコーダを書く

汎用 `MsgParser` で意味のある値が取れない場合（文字列フィールドを含む、フラット化したいネストがある等）、`decoders/builtin.py` に追加:

```python
@register_decoder("my_pkg/msg/MyType")
def decode_my_type(msg, selector, config):
    values = [msg.a, msg.b, *msg.array]
    return _finalize(values, config)
```

詳細は [`configuration.md#カスタムデコーダの追加方法`](configuration.md#カスタムデコーダの追加方法)。

## 4. `configs/<robot>.yaml` の作成

`robot_template.yaml` をコピーして書き換えます。必須トップレベル:

```yaml
robot_type: "my_robot"
fps: 30
task: "このエピソードの内容"
```

observation / action を列挙。bag に存在しないかもしれない or 極端に疎なものは `optional: true`（`trim_to_valid` の制約から外れ、欠損はゼロ埋めされる）。

### 例: 単腕・カメラ 1 台の最小構成

```yaml
robot_type: "my_robot"
fps: 30
task: "pick and place"

observations:
  - key: "observation.images.front"
    topic: "/camera/image_raw/compressed"
    msg_type: "sensor_msgs/msg/CompressedImage"
    dtype: "image"
    image_size: [480, 640, 3]
    stamp_source: "header"

  - key: "observation.state"
    topic: "/joint_states"
    msg_type: "sensor_msgs/msg/JointState"
    selector: "position"
    dtype: "float32"
    stamp_source: "header"

actions:
  - key: "action"
    topic: "/joint_commands"
    msg_type: "sensor_msgs/msg/JointState"
    selector: "position"
    dtype: "float32"
    stamp_source: "header"

resampling:
  default_policy: "hold"
  tolerance_ms: 500
  trim_to_valid: true
  align_to_required: true     # 全 required トピックがそろう区間で start/stop（既定 true）
  # max_stamp_delay_ms: 100   # 任意: QoS スタール（古い latch 値）を破棄するしきい値(ms)
```

`default_policy` / `tolerance_ms` / `align_to_required` / `max_stamp_delay_ms` の選び方は [`configuration.md#リサンプリング`](configuration.md#リサンプリング) を参照。

## 5. dry-run

実書き込みをせず、config と bag の整合性を確認:

```bash
rosbag2lerobot convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /tmp/dry --dry-run
```

observation / action と各 bag のトピック表が並ぶので、ここでトピック名・メッセージ型の不一致を全部潰します。verbose でさらに詳細を出したければ `-v` を先頭に付与:

```bash
rosbag2lerobot -v convert --dry-run ...
```

## 6. 本番変換

```bash
rosbag2lerobot convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag_dir/ \
  --output /path/to/output_dataset/
```

`--bags` が複数 bag ディレクトリの親を指していれば、それぞれが 1 エピソードになります。オプション詳細は [`cli_reference.md`](cli_reference.md)。

## 7. 出力の検証

```python
import json, pyarrow.parquet as pq
info = json.load(open('/path/to/output_dataset/meta/info.json'))
print('episodes:', info['total_episodes'], 'frames:', info['total_frames'])

t = pq.read_table('/path/to/output_dataset/data/chunk-000/file-000.parquet')
print('rows:', t.num_rows, 'cols:', t.column_names)

for c in t.column_names:
    if t[c].null_count:
        print('NULL IN', c, '->', t[c].null_count)
```

期待: null ゼロ、`frame_index` が 0 始まり、特徴量ごとのシェイプが `info.json` と一致。

## 8. 回帰テストの追加

`tests/test_e2e_<robot>.py` を作成して最低限以下をチェック:

```python
from rosbag2lerobot.config import load_config

class TestMyRobotConfig:
    def test_load_yaml(self):
        cfg = load_config("src/rosbag2lerobot/configs/my_robot.yaml")
        assert cfg.fps == 30
        assert "observation.state" in cfg.observation_keys
        assert cfg.resampling.default_policy == "hold"
```

チェックイン済みのサンプル bag があるなら、`tests/test_integration_real.py` を拡張して実変換とメタデータの値を突き合わせます。

## トラブルシューティング

### デシリアライズ中に `ValueError: buffer is smaller than requested size`

ローカル `.msg` が bag 埋め込み msgdef と不一致です。通常は `_override_types_from_bag` が自動解決します。エラーが出続けるなら、`rosbags.Reader` で bag を開いて `conn.msgdef.data` を表示し、bag がそもそも msgdef を埋め込んでいるか確認してください。埋め込みが無いなら、ローカル `.msg` をワイヤフォーマットに合わせて更新。

### 最初の数十〜数百フレームが全部ゼロ

プリロール（センサー起動前に bag 記録開始）。`resampling.trim_to_valid: true`（デフォルト）ならログに `trim_to_valid: dropped N frames` と出て自動的に削られるはず。ログが出ていない場合は `trim_to_valid` が誤って `false` になっていないか確認。

### 疎なアクション列がほぼゼロ

コマンドがエピソードあたり数回しか発行されないトピックでは想定通り。`inspect` で生のメッセージ時刻を確認する。連続値が欲しいなら該当トピックを **非** optional にして、最初のコマンド以降だけにエピソードを切り詰める。

### `stats.json` で特徴量シェイプが揃わない

ライターは最初の非 null フレームから shape を推論します。セレクタが可変長ベクトルを返すならカスタムデコーダで固定長パディングするように直す。

### その他のエラー

依存関係・ffmpeg 未検出・トピック不一致・OOM 等の対処は [`development.md#トラブルシューティング`](development.md#トラブルシューティング) を参照。
