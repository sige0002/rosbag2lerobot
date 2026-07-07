# 設定ファイルリファレンス

変換の挙動はすべて YAML で定義します。完全版テンプレートは `src/rosbag2lerobot/configs/robot_template.yaml`。

## 目次

- [最小例](#最小例)
- [トップレベルフィールド](#トップレベルフィールド)
- [特徴量マッピング (`observations` / `actions`)](#特徴量マッピング-observations--actions)
  - [フィールド一覧](#フィールド一覧)
  - [特徴量キー命名規則](#特徴量キー命名規則)
- [セレクタ](#セレクタ)
- [カメラ](#カメラ)
- [リサンプリング](#リサンプリング)
- [カスタムメッセージ](#カスタムメッセージ)
- [オプショナルトピック](#オプショナルトピック)
- [対応メッセージ型](#対応メッセージ型)
- [カスタムデコーダの追加方法](#カスタムデコーダの追加方法)

## 最小例

```yaml
robot_type: "my_robot"
fps: 30
task: "pick and place"

observations:
  - key: "observation.images.top"
    topic: "/camera/top/image_raw/compressed"
    msg_type: "sensor_msgs/msg/CompressedImage"
    dtype: "image"
    image_size: [480, 640, 3]

  - key: "observation.state"
    topic: "/joint_states"
    msg_type: "sensor_msgs/msg/JointState"
    selector: "position"
    dtype: "float32"

actions:
  - key: "action"
    topic: "/joint_commands"
    msg_type: "sensor_msgs/msg/JointState"
    selector: "position"
    dtype: "float32"
```

## トップレベルフィールド

| フィールド | 必須 | 説明 |
|---|---|---|
| `robot_type` | ✓ | ロボットを識別する一意な文字列。 |
| `fps` | ✓ | 出力データセットの目標フレームレート（Hz）。 |
| `task` | ✓ | タスクの短い説明文。bag ディレクトリに `task.json` があればそちらが優先される（詳細は [`task_json.md`](task_json.md)）。 |
| `repo_id` | — | HuggingFace の repo id（例: `myorg/my-dataset`）。CLI の `--repo-id` で上書き可。 |
| `observations` | ✓ | 観測特徴量のリスト（後述）。 |
| `actions` | ✓ | 行動特徴量のリスト（後述）。 |
| `custom_msgs` | — | カスタム `.msg` ファイル登録（後述）。 |
| `resampling` | — | リサンプリング設定（後述）。 |

## 特徴量マッピング (`observations` / `actions`)

1 エントリ = 1 ROS2 トピック → 1 LeRobot 特徴量キー。

### フィールド一覧

| フィールド | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `key` | ✓ | — | LeRobot 特徴量キー。命名規則は後述。 |
| `topic` | ✓ | — | ROS2 トピック名（例: `/joint_states`）。 |
| `msg_type` | ✓ | — | ROS2 メッセージ型（例: `sensor_msgs/msg/JointState`）。 |
| `selector` | — | `""` | メッセージ内のサブフィールドを指すドット区切りパス。詳細は[セレクタ](#セレクタ)節。 |
| `dtype` | — | `"float32"` | 出力 dtype。`float32` / `float64` / `int32` / `int64` / `uint8` / `bool` / `image` / `string`。 |
| `image_size` | — | なし | `[H, W]` または `[H, W, C]`。画像特徴量では必須。 |
| `stamp_source` | — | `"header"` | タイムスタンプ源。`"header"` = メッセージ `header.stamp`、`"receive"` = bag 受信時刻。リサンプリングで実際に採用される時刻を決める（[リサンプリング](#リサンプリング)参照）。 |
| `max_stamp_delay_ms` | — | なし | スタール（古い latch 値）検出のしきい値（ms）。`stamp_source: header` のとき `abs(受信時刻 − header.stamp)` がこの値を超えるメッセージを 1 件単位で破棄。未指定 = このトピックは判定しない、`0` = 少しでも遅延があれば破棄。`resampling.max_stamp_delay_ms`（全体既定）を上書きする。 |
| `unit_conversion` | — | `1.0` | 数値乗数、または特殊文字列 `"rad2deg"` / `"deg2rad"`。 |
| `optional` | — | `false` | `true` にするとそのトピックが bag に無くても警告だけで続行。`trim_to_valid` の判定からも外れ、欠損はゼロ埋めされる。 |

### 特徴量キー命名規則

- `observation.images.<name>` — カメラ（例: `observation.images.front`）。
- `observation.state` — 単一状態ベクトル。
- `observation.state.<part>` — 部位別（例: `observation.state.right_arm`）。
- `action` — 行動ベクトル。
- `action.<part>` — 部位別（例: `action.right_arm`, `action.left_gripper`）。

## セレクタ

メッセージからサブフィールドを抽出する式です。

```yaml
# JointState の position を全部抜く
selector: "position"

# ネストしたフィールドをカンマ区切りで複数抽出
selector: "pose.position.x,pose.position.y,pose.position.z"

# 特定の関節だけ名前で抽出
selector: "position.joint_1,position.joint_2"

# ワイルドカード
selector: "position.*"
```

セマンティクス（ドットパス走査、`JointState` / `Joy` の特殊処理、フィールドエイリアス等）は [`architecture.md`](architecture.md#デコーダ-decoders) 参照。

## カメラ

画像トピックは `dtype: "image"` と `image_size` を必ず指定します。raw / compressed どちらも対応。

```yaml
# Compressed image (JPEG / PNG)
- key: "observation.images.front"
  topic: "/camera/front/image_raw/compressed"
  msg_type: "sensor_msgs/msg/CompressedImage"
  dtype: "image"
  image_size: [480, 640]

# Raw image
- key: "observation.images.wrist"
  topic: "/camera/wrist/image_raw"
  msg_type: "sensor_msgs/msg/Image"
  dtype: "image"
  image_size: [480, 640, 3]
```

## リサンプリング

複数レートのトピックを目標 FPS に揃える設定です。

```yaml
resampling:
  default_policy: "hold"     # "hold" | "nearest" | "drop"
  tolerance_ms: 500.0        # 許容時間差（ms）
  trim_to_valid: true        # required 特徴量が揃わない先頭／末尾を切る
  align_to_required: true    # 全 required 特徴量がそろう交差区間で start/stop（既定 true）
  max_stamp_delay_ms: null   # スタール検出の全体しきい値（ms）。null=無効, 0=0ms 超で破棄
```

| ポリシー | 挙動 |
|---|---|
| `hold` | ゼロ次ホールド。最後に届いた値を次の更新まで保持。模倣学習の疎なコマンドに向く。 |
| `nearest` | `tolerance_ms` 以内で最も時刻が近いメッセージを採用。なければ null。 |
| `drop` | `tolerance_ms` 以内にメッセージが無ければ null のまま残す。 |

### `trim_to_valid` について

デフォルト `true`。「全 required 特徴量に値がある最長区間」にエピソードを切り詰めます。先頭（センサー起動前のプリロール）と末尾（記録中に bag が切れたテール）を落とすため。`optional: true` の特徴量は trim 判定から除外され、残った欠損はライターがゼロ埋めします。

> 中間欠損の扱い・タイムスタンプ再生成など詳細な挙動は [`architecture.md`](architecture.md#trim_to_valid_range-が何をしているか) 参照。

### `align_to_required` について（トピックの開始・終了ずれ対策）

デフォルト `true`。required（非 optional）特徴量は、トピックごとに配信開始・終了の時刻が微妙にずれます。`true` だと出力グリッドを **全 required 特徴量がそろう交差区間** `[max(各 required の最初の時刻), min(各 required の最後の時刻)]` に合わせます。これにより、早く配信を止めたトピックの最終値が `hold` で末尾まで引き延ばされる（= 終了トピックのずれ）のを防ぎ、区間内は全 required 特徴量が実データを持ちます。`false` にすると従来どおり bag 全体の時間範囲を使います。`trim_to_valid` とは独立で両方適用できます（交差区間で start/stop → さらに残った欠損端を trim）。

> required にメッセージが 1 件も無い、または required 同士が時間的に重ならない場合、そのエピソードは空になります（警告ログ）。

### `max_stamp_delay_ms` について（QoS スタール対策）

ROS の QoS（`TRANSIENT_LOCAL`）や depth などで「前に publish された古い latch メッセージ」が残ると、`header.stamp` が bag 受信時刻より大きく遅れます。`max_stamp_delay_ms`（feature 単位 ＞ `resampling` 全体の順で解決）を設定すると、`abs(受信時刻 − header.stamp)` がしきい値（ms）を超えるメッセージを **1 件単位で破棄**します（デコード前に判定するので無駄なデコードもしません）。`stamp_source: header` で header が取れるトピックにのみ効きます。未指定（`null`）で無効、`0` は「少しでも遅延があれば破棄」。

### `default_policy` と `tolerance_ms` の選び方

| データ特性 | ポリシー | tolerance |
|---|---|---|
| 高レート（≥ fps）センサー、疎なコマンドなし | `nearest` | 50–100 ms |
| 疎なコマンドトピック（`optional: true` のアクション）あり | `hold` | 500 ms |
| クロックスキューのあるマルチソース | `nearest` | `1/fps` の半分 |

模倣学習の安全な既定は `hold + 500 ms`。

## カスタムメッセージ

ROS2 のインストール無しに非標準型をデシリアライズするため、`.msg` ファイルを登録します。

```yaml
custom_msgs:
  - msg_file: "msgs/my_robot/MyCustomMsg.msg"
    package: "my_robot_msgs"
```

- `msg_file` のパスは YAML のあるディレクトリからの相対パスで解決。
- **bag 埋め込み msgdef が優先**。`custom_msgs` はその機能が無い古い bag への保険。詳しくは [`architecture.md`](architecture.md#型システムとmsg登録-readerpy)。
- 登録前に構文チェックしたければ `rosbag2lerobot validate-msg --msg <path>`。

## オプショナルトピック

bag に存在しない可能性があるトピックは `optional: true` を付けます。

```yaml
observations:
  - key: "observation.state.force"
    topic: "/force_sensor"
    msg_type: "my_robot_msgs/msg/Force"
    selector: "fx,fy,fz,mx,my,mz"
    dtype: "float32"
    optional: true
```

オプショナル特徴量は `trim_to_valid` の判定から除外され、残った区間に値が無ければライターがゼロ埋めします（LeRobot スキーマ互換のため）。

## 対応メッセージ型

組み込みデコーダで直接扱えるメッセージ型。

| メッセージ型 | 出力形状 | 備考 |
|---|---|---|
| `sensor_msgs/msg/JointState` | `float32[N]` | selector で position / velocity / effort を選択 |
| `sensor_msgs/msg/Image` | `PIL.Image (RGB)` | rgb8 / bgr8 / rgba8 / mono8 / 16UC1 等 |
| `sensor_msgs/msg/CompressedImage` | `PIL.Image (RGB)` | JPEG / PNG 自動判定 |
| `sensor_msgs/msg/Imu` | `float32[10]` | quaternion + gyro + accel |
| `sensor_msgs/msg/Joy` | `float32[N]` | axes + buttons |
| `geometry_msgs/msg/Twist` | `float32[6]` | linear xyz + angular xyz |
| `geometry_msgs/msg/TwistStamped` | `float32[6]` | Twist デコーダに委譲 |
| `geometry_msgs/msg/PoseStamped` | `float32[7]` | position + quaternion |
| `nav_msgs/msg/Odometry` | `float32[13]` | pose(7) + twist(6) |
| `std_msgs/msg/Float32` | `float32[1]` | スカラー |
| `std_msgs/msg/Float64` | `float32[1]` | float32 にキャスト |
| `std_msgs/msg/Float32MultiArray` | `float32[N]` | index で選択可 |
| `std_msgs/msg/String` | `str` | 文字列 |

未登録のメッセージ型は `msg_parser.MsgDecoder` が自動で数値リーフを抽出します（フォールバック）。

## カスタムデコーダの追加方法

組み込みで対応できない／セレクタで分解できない場合、3 通りの選択肢があります。軽いものから検討してください。

### 1. `.msg` を `custom_msgs` に登録（最も軽い）

汎用 `MsgParser` が自動でフィールドを舐めて数値を抽出します。特別な整形が不要ならこれで十分。

### 2. YAML でデコーダ関数を参照

`module:function` 形式で書きます。

```yaml
- key: "observation.state"
  topic: "/custom_topic"
  msg_type: "my_pkg/msg/CustomState"
  decoder: "my_package.decoders:decode_custom_state"
  dtype: "float32"
```

関数シグネチャ:

```python
from typing import Any
import numpy as np
import PIL.Image

def decode_custom_state(
    msg: Any,
    selector: list[str] | None,
    config: dict[str, Any],
) -> np.ndarray | PIL.Image.Image:
    ...
```

### 3. 組み込みデコーダとして追加（恒久対応）

`src/rosbag2lerobot/decoders/builtin.py` などに追加し `@register_decoder` で登録。

```python
from rosbag2lerobot.decoders import register_decoder

@register_decoder("my_pkg/msg/CustomState")
def decode_custom_state(msg, selector, config):
    values = [msg.a, msg.b, *msg.array]
    return _finalize(values, config)
```

`_finalize` ヘルパーは `float32` へのキャストと `unit_conversion` の適用を共通化しているので、数値デコーダは原則これで締めます。
