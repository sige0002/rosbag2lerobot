# CLI リファレンス

`bagel` は Click ベースの CLI です。サブコマンドは5つ。

| コマンド | 用途 |
|---|---|
| [`inspect`](#inspect) | bag のトピック一覧・メッセージ数・時間範囲を表示（書き込みなし）。`--fps-stats` / `--suggest-image-size` で診断モード。 |
| [`convert`](#convert) | bag を LeRobot v3.0 データセットに変換 |
| [`validate-config`](#validate-config) | YAML config と bag の整合性を検証（トピック / msg_type / 画像サイズ） |
| [`audit-timestamps`](#audit-timestamps) | 生成済みデータセットの `meta/episodes/*.parquet` のタイムスタンプ連続性を監査 |
| [`validate-msg`](#validate-msg) | `.msg` ファイルの構文チェック |

## 目次

- [共通オプション](#共通オプション)
- [`inspect`](#inspect)
  - [基本オプション](#基本オプション)
  - [FPS 統計 (F1)](#fps-統計-f1)
  - [image_size 提案 (F3)](#image_size-提案-f3)
- [`convert`](#convert)
  - [必須オプション](#必須オプション)
  - [config オーバーライド](#config-オーバーライド)
  - [エピソード制御](#エピソード制御)
  - [動画エンコード関連](#動画エンコード関連)
  - [検証モード](#検証モード)
- [`validate-config`](#validate-config)
- [`audit-timestamps`](#audit-timestamps)
- [`validate-msg`](#validate-msg)
- [実例集](#実例集)

## 共通オプション

```bash
bagel [-v|--verbose] <subcommand> [OPTIONS]
```

| オプション | 説明 |
|---|---|
| `-v, --verbose` | ログレベルを DEBUG に引き上げる。通常は INFO。メッセージ型登録の詳細、フレームごとの判定結果、ffmpeg 呼び出し行などが見えるようになる。問題切り分け時に最初につけるフラグ。 |
| `--help` | ヘルプ表示。サブコマンド側にも付与可能（`bagel convert --help`）。 |

> **verbose の使いどころ**：`trim_to_valid: dropped N frames` のような info ログは標準でも出るが、「どの required キーが欠けて trim されたか」「カスタム msg の登録が成功したか」「ffmpeg に渡された実コマンド」は DEBUG でないと見えません。

## `inspect`

bag の中身を覗くだけ。config 不要で走らせられます。追加フラグで FPS 統計 (F1) や image_size 提案 (F3) に切り替わります。

```bash
bagel inspect --bags /path/to/ros_sample_bag/
```

### 基本オプション

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--bags PATH` | ✓ | — | bag ディレクトリ単体、または bag 群を含む親ディレクトリ。 |
| `--config PATH` |  | なし | 省略可。YAML を渡すとトピック解釈のための型登録も行われる。`--suggest-image-size` 時は必須。 |

`--fps-stats` も `--suggest-image-size` も付けない場合は従来通り「Duration / Start / End / Topics（msg_type, msg_count）」の一覧だけを出力（後方互換）。

### FPS 統計 (F1)

`--fps-stats` を付けるとトピック単位の周期統計を計算します。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--fps-stats / --no-fps-stats` | `--no-fps-stats` | トピック毎の FPS / head・tail lag / ギャップ統計を計算する。 |
| `--topics TEXT` | 全トピック | `--fps-stats` 対象のトピックをカンマ区切りで絞り込む（例: `/camera/color/image_raw,/joint_states`）。 |
| `--gap-threshold-ms FLOAT` | `200.0` | この値（ミリ秒）を超えたメッセージ間隔を「ギャップ」としてレポートに列挙する。 |
| `--head INT` | `5` | 先頭サンプルの間隔を何件出力に含めるか。 |
| `--json-out PATH` | なし | 指定すると `--fps-stats` と `--suggest-image-size` の結果を JSON で書き出す（人間向け出力は抑制）。 |

出力は bag ごとに `(topic, msg_count, mean_fps, min, max, head_lag_ms, gaps)` を並べた表。`mean_fps` はトピックの実効レート（`正の間隔数 / 総経過時間`）で、バースト送信でタイムスタンプがほぼ重複しても外れ値に引っ張られません。`min` / `max` は瞬間レート（間隔ごとの `1/dt`）の分布なので、ピークは大きく出ることがあります。純関数本体は `bagel.diagnostics.compute_topic_fps_report` を参照。

### image_size 提案 (F3)

`--suggest-image-size` は `--config` と併用して、YAML の `image_size` と bag から実デコードした画像の shape を突き合わせます。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--suggest-image-size` | off | 各 image 特徴量について先頭数フレームをデコードし、YAML `image_size` との一致を判定する。`--config` 必須。 |
| `--samples INT` | `5` | 1 トピックあたりデコードするサンプルフレーム数。 |
| `--json-out PATH` | なし | 上記 FPS 統計と共通。JSON 出力先。 |

結果は `(key, topic, yaml_image_size, decoded_shape, mismatch)` のリスト。`mismatch=true` のトピックは YAML 側を修正する目安になります。純関数は `bagel.diagnostics.detect_image_shape`。

## `convert`

bag → LeRobot v3.0 データセット本体の変換。1 bag ディレクトリ = 1 エピソード。

### 必須オプション

| オプション | 説明 |
|---|---|
| `--config PATH` | `robot_config.yaml` へのパス。[`configuration.md`](configuration.md) 参照。 |
| `--bags PATH` | bag ディレクトリ、または bag 群の親ディレクトリ。`discover_bags` がソート順に列挙する。 |
| `--output PATH` | 出力データセットディレクトリ。存在しなければ作成される。 |

### config オーバーライド

CLI から YAML の一部を上書きできます。一度きりの実験用。

| オプション | 上書きするフィールド | 用途 |
|---|---|---|
| `--task TEXT` | `config.task` | 同じ bag を別タスクとして扱う試験など。bag ディレクトリに `task.json` があればそちらが優先される（詳細は [`task_json.md`](task_json.md)）。 |
| `--fps INT` | `config.fps` | 一時的に fps を下げて出力サイズを小さくしたい場合。 |
| `--repo-id TEXT` | `config.repo_id` | HuggingFace Hub の repo id（例: `myorg/my-dataset`）。 |

### エピソード制御

| オプション | デフォルト | 説明 |
|---|---|---|
| `--max-episodes INT` | 全件 | 先頭 N 個の bag だけ変換。動作確認や小規模ベンチに。 |
| `--workers INT` | 1 | 並列ワーカー数。2 以上で `ProcessPoolExecutor` が起動し、bag 単位で同時デコードする。ライターは逐次受け取りなので出力順は決定的。メモリは `workers × 1 エピソード分` まで膨らむので注意。 |

### 動画エンコード関連

| オプション | デフォルト | 説明 |
|---|---|---|
| `--video-codec TEXT` | `auto` | 使用する動画コーデック。`auto` は `ffmpeg -encoders` を走査して NVENC があれば `h264_nvenc`、なければ `libx264`。明示指定も可（`libx264` / `libsvtav1` / `h264_nvenc` / `hevc_nvenc` / `av1_nvenc`）。 |
| `--gpu / --no-gpu` | 自動 | NVENC を使うかどうか。`--gpu` 指定時に NVENC が見つからなければエラーで落ちる。`--no-gpu` は CPU コーデックを強制（再現性重視）。`--video-codec` と矛盾すると警告／エラー。 |
| `--ffmpeg-preset TEXT` | コーデック毎の既定値 | ffmpeg の preset。libx264: `veryfast` 等／NVENC: `p1`〜`p7`／libsvtav1: 数値 `0`〜`13`。 |
| `--ffmpeg-crf INT` | コーデック毎の既定値 | 画質指定。CPU コーデックは `-crf`、NVENC は `-cq` にマップ。 |

> コーデックごとのスループット／サイズ比較は [`output_and_performance.md`](output_and_performance.md#スループット比較) を参照。

### 検証モード

| オプション | 説明 |
|---|---|
| `--dry-run` | 実際の書き込みをせず、config と各 bag のマッチングレポートだけ出す。observations / actions の key → topic → msg_type の対応と、bag 側の実トピック一覧を並べて表示。本番変換の前に毎回推奨。 |

## `validate-config`

YAML config と bag の整合性を検証します（F4）。CI で config ドリフトを検知する用途を想定。純関数は `bagel.diagnostics.validate_config_against_bag`。

```bash
bagel validate-config \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--config PATH` | ✓ | — | 検証対象の `robot_config.yaml`。 |
| `--bags PATH` | ✓ | — | bag ディレクトリまたは親ディレクトリ。先頭の 1 bag を代表として検証する。 |
| `--samples INT` |  | `5` | 画像 shape チェック時に 1 トピックあたりデコードするサンプルフレーム数。 |
| `--strict` |  | off | 警告（image shape 不一致 / 未使用トピック）もエラー扱いにして exit code を非ゼロにする。 |
| `--json-out PATH` |  | なし | 指定すると検証レポートを JSON で書き出す。 |
| `--ignore-unused-topics` |  | off | bag に存在するが config から参照されていないトピックの info 表示を抑制する。 |

出力は以下のカテゴリに分けて表示され、終了ステータスは `strict` とエラー件数で決まります。

| カテゴリ | レベル | 内容 |
|---|---|---|
| `missing_required_topics` | ERROR | `optional: false` の特徴量のトピックが bag にない |
| `msg_type_mismatches` | ERROR | YAML の `msg_type` と bag 側 msgdef が不一致 |
| `image_shape_mismatches` | WARN | YAML `image_size` と実デコード shape が不一致 |
| `missing_optional_topics` | INFO | `optional: true` のトピックが bag に存在しない |
| `unused_bag_topics` | INFO | bag 側にはあるが config から参照されていないトピック |

## `audit-timestamps`

生成済み LeRobot v3.0 データセットの `meta/episodes/*.parquet` のタイムスタンプ連続性を監査します（F2）。純関数は `bagel.audit.audit_episode_timestamps`。

```bash
bagel audit-timestamps --dataset /path/to/output_dataset/
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--dataset PATH` | ✓ | — | 監査対象の LeRobot v3.0 データセットルート。 |
| `--max-drift-us FLOAT` |  | `1.0` | 1 行あたりおよび累積で許容する最大ドリフト（マイクロ秒）。これを超える差分はエラー。 |
| `--json-out PATH` |  | なし | 指定すると `AuditReport` を JSON で書き出す（人間向け summary は常に併出）。 |
| `--video-key TEXT` |  | 全 video_key | 特定の `video_key` だけを監査する。省略時は `info.json` に存在する全 video_key。 |

監査内容:

- 同一 mp4 内では `to_timestamp[i] == from_timestamp[i+1]`（許容差 `max_drift_us`）
- `from_timestamp` が `0.0` にリセットされるのは mp4 境界のみ

エラーが 1 件でもあれば exit code を非ゼロで返すので、CI の後段に挟めます。

## `validate-msg`

`.msg` ファイルの構文だけチェックします。`custom_msgs` を書く前に使う。

```bash
bagel validate-msg --msg msgs/my_robot/MyCustomMsg.msg
```

| オプション | 必須 | 説明 |
|---|---|---|
| `--msg PATH` | ✓ | 検証対象の `.msg` ファイル。 |

成功すれば緑で `OK`、失敗すれば赤で `INVALID: <error>` を出して exit 1。

## 実例集

### 1. まずは inspect

```bash
bagel inspect --bags /path/to/ros_sample_bag/
```

### 2. dry-run で config を詰める

```bash
bagel convert \
  --config src/bagel/configs/robot_template.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /tmp/dry --dry-run
```

### 3. 本番変換（auto codec、単一プロセス）

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /path/to/output_dataset/
```

### 4. 複数 bag を 4 並列で変換、上限 10 エピソード

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag_dir/ \
  --output /path/to/output_dataset/ \
  --workers 4 --max-episodes 10 \
  --repo-id "myorg/my-dataset"
```

### 5. CPU 固定（再現性重視）

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /path/to/out \
  --no-gpu
```

### 6. AV1 NVENC + 高品質プリセット

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /path/to/out \
  --video-codec av1_nvenc --ffmpeg-preset p4
```

### 7. デバッグログを出す

```bash
bagel -v convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --output /tmp/out --dry-run
```

### 8. カスタム `.msg` の事前検証

```bash
bagel validate-msg --msg msgs/my_robot/MyCustomMsg.msg
```

### 9. トピック周期を計測（F1）

```bash
bagel inspect \
  --bags /path/to/ros_sample_bag/ \
  --fps-stats \
  --gap-threshold-ms 100 \
  --head 10
```

特定トピックだけ見たい場合:

```bash
bagel inspect \
  --bags /path/to/ros_sample_bag/ \
  --fps-stats \
  --topics /camera/color/image_raw,/joint_states \
  --json-out /tmp/fps_stats.json
```

### 10. YAML の `image_size` を実データで検証（F3）

```bash
bagel inspect \
  --bags /path/to/ros_sample_bag/ \
  --config configs/my_robot.yaml \
  --suggest-image-size \
  --samples 10
```

### 11. config と bag の整合性を CI で検証（F4）

```bash
bagel validate-config \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag/ \
  --strict \
  --json-out /tmp/validate.json
```

### 12. 生成データセットのタイムスタンプ監査（F2）

```bash
bagel audit-timestamps \
  --dataset /path/to/output_dataset/ \
  --max-drift-us 1.0
```

特定 `video_key` だけ絞りたい場合:

```bash
bagel audit-timestamps \
  --dataset /path/to/output_dataset/ \
  --video-key observation.images.cam_high \
  --json-out /tmp/audit.json
```
