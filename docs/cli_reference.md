# CLI リファレンス

`bagel` は Click ベースの CLI です。

| コマンド | 用途 |
|---|---|
| [`inspect`](#inspect) | bag のトピック一覧・メッセージ数・時間範囲を表示（書き込みなし）。`--fps-stats` / `--suggest-image-size` で診断モード。 |
| [`scaffold`](#scaffold) | 未知ロボットの bag から `robot_config.yaml` の雛形を自動生成。`--no-validate` 以外では `validate-config` を自動実行。 |
| [`convert`](#convert) | bag を LeRobot v3.0 データセットに変換 |
| [`validate-config`](#validate-config) | YAML config と bag の整合性を検証（トピック / msg_type / 画像サイズ） |
| [`validate-dataset`](#validate-dataset) | 生成済みデータセットが LeRobot Dataset v3.0 構造に準拠するか検証 |
| [`quality-report`](#quality-report) | 生成済みデータセットのデータ品質をスコアリング |
| [`audit-timestamps`](#audit-timestamps) | 生成済みデータセットの `meta/episodes/*.parquet` のタイムスタンプ連続性を監査 |
| [`validate-msg`](#validate-msg) | `.msg` ファイルの構文チェック |
| [`preview`](#preview) | 生成済みデータセットの自己完結型 HTML プレビューレポートを生成 |
| [`push-to-hub`](#push-to-hub) | 生成済みデータセットを HuggingFace Hub にアップロード（データセットカード付き） |
| [`ui`](#ui) | localhost（127.0.0.1 限定）コントロール UI を起動（閲覧 → scaffold → convert → 品質確認） |

## 目次

- [共通オプション](#共通オプション)
- [レポート系コマンドの `--json`](#レポート系コマンドの---json)
- [`inspect`](#inspect)
  - [基本オプション](#基本オプション)
  - [FPS 統計 (F1)](#fps-統計-f1)
  - [image_size 提案 (F3)](#image_size-提案-f3)
- [`scaffold`](#scaffold)
- [`convert`](#convert)
  - [必須オプション](#必須オプション)
  - [config オーバーライド](#config-オーバーライド)
  - [エピソード制御](#エピソード制御)
  - [動画エンコード関連](#動画エンコード関連)
  - [検証モード](#検証モード)
  - [再実行ガード（--resume）](#再実行ガード--resume)
  - [進捗・JSON・失敗継続](#進捗json失敗継続)
  - [出力されるランメタデータ](#出力されるランメタデータ)
- [config の新キー（split / TF 特徴量 / タイポ検出）](#config-の新キーsplit--tf-特徴量--タイポ検出)
- [`validate-config`](#validate-config)
- [`validate-dataset`](#validate-dataset)
- [`quality-report`](#quality-report)
- [`audit-timestamps`](#audit-timestamps)
- [`validate-msg`](#validate-msg)
- [`preview`](#preview)
- [`push-to-hub`](#push-to-hub)
- [`ui`](#ui)
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

## レポート系コマンドの `--json`

レポートを出力する全コマンドに `--json` フラグがあります。付けるとレポート dict を **stdout に JSON で出力**し、人間向けの summary は抑制されます（CI でのパース用途）。対象コマンド:

| コマンド | `--json` で出力される内容 |
|---|---|
| [`validate-config`](#validate-config) | `{config, bag, results}`（`results` は検証レポート） |
| [`validate-dataset`](#validate-dataset) | `DatasetValidationReport` の dict（issues / verdict / exit_code） |
| [`quality-report`](#quality-report) | 品質レポート dict（per-feature 統計 / score / verdict） |
| [`audit-timestamps`](#audit-timestamps) | `AuditReport` の dict |
| [`inspect`](#inspect) | `--fps-stats` / `--suggest-image-size` のレポート dict |
| [`validate-msg`](#validate-msg) | 検証結果 dict（`{valid, error, ...}`） |
| [`to-mcap`](#to-mcap) | 変換結果 dict |

> **P0 のファイル出力フラグとの関係.** `--json-out PATH`（`inspect` / `validate-config` / `validate-dataset` / `audit-timestamps`）と `-o, --report PATH`（`quality-report`）は後方互換のため残っています。`--json`（stdout）とファイル出力フラグは併用可能で、`--json` は **summary を抑制**する点だけが異なります。

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
| `--json` |  | off | レポート dict を stdout に JSON 出力（人間向け summary は抑制）。`--json-out` と併用可。 |

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

## `scaffold`

未知ロボットの bag から `robot_config.yaml` の雛形を自動生成します。最初に見つかった 1 bag のトピックを走査し、image トピックと、デコーダで読める数値トピックを LeRobot の feature key にマッピングします。デコーダ未対応のトピックや command 系トピックは「候補」としてコメントアウトして出力します。生成後はメモリ上で config を検証し、さらに `--no-validate` を付けない限り、その bag に対して `validate-config` を実行してマッピング保証（往復可能性）を確認します。

```bash
bagel scaffold \
  --bags /path/to/ros_sample_bag/ \
  -o configs/my_robot.yaml
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--bags PATH` | ✓ | — | bag ディレクトリ単体、または bag 群を含む親ディレクトリ。最初に見つかった 1 bag を雛形の元にする。 |
| `-o, --output FILE` |  | stdout | 出力 config パス。省略時は標準出力に表示。 |
| `--fps INTEGER` |  | auto | ターゲット fps。auto は image トピックの最小 fps、なければ state トピックの中央値 fps。 |
| `--robot-type TEXT` |  | `unknown_robot` | 生成 config の `robot_type` 値。 |
| `--task TEXT` |  | `TODO_describe_task` | 生成 config の `task` 記述。 |
| `--min-count INTEGER` |  | `1` | メッセージ数がこの値未満のトピックを除外する。 |
| `--samples INTEGER` |  | `3` | shape 検出のために 1 トピックあたりデコードする画像フレーム数。 |
| `--no-validate` |  | off | 生成 config への自動 `validate-config` をスキップする。 |

マッピング規則:

- image トピック → `observation.images.<slug>`（slug はトピックパス由来。衝突時は曖昧性を解消）。
- デコーダで読める数値トピック → `observation.state`（双腕の場合は `_left` / `_right`）。
- デコーダ未対応トピック → コメントアウトした候補として出力。
- `actions` は `actions: []` のまま、command 系候補をコメントで併記。

終了ステータス: 生成と検証が成功すれば `0`。`--no-validate` を付けない場合、自動 `validate-config` がエラーを返すと非ゼロ。

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

### 再実行ガード（--resume）

`--resume` は **安全な再実行ガード**です（P0 範囲のみ）。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--resume` | off | 非空の `--output` への再変換を許可する。完成済みデータセットなら何もせず終了（no-op）、途中でクラッシュした（未確定の）出力は丸ごと削除して再変換する。 |

- `--resume` **なし**で非空の `--output` に変換しようとすると、既存データセットの破壊を防ぐため **中断**します（書き込みは一切行いません）。
- `--resume` **あり**で出力が **完成済み**なら **no-op**（何もしない）。
- `--resume` **あり**で出力が **未確定（クラッシュ済み）**なら、出力を **クリーンに削除してから再変換**します。

> **注意.** 「変換済みエピソードをスキップする」本来の意味でのレジューム（作業のスキップ）は **P1 予定の機能で、まだ未実装**です。現状の `--resume` は再実行時の安全策のみを提供します。

### 進捗・JSON・失敗継続

`convert` はデフォルトで **tqdm の ETA 付き進捗バー**を表示します。以下のフラグで出力形態と失敗時の挙動を制御できます。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--json` | off | そのランの `job_summary`（後述）を **stdout に JSON 出力**する。人間向けの `Done.` ログは抑制される。 |
| `--quiet` | off | 進捗バーと INFO レベルのログを抑制する（WARNING 以上は残る）。 |
| `--skip-failed` | off | 1 つの bag のデコード/リサンプルが失敗しても、その失敗を **記録してスキップ**し、残りの bag を処理して成功エピソードからデータセットを確定する。**デフォルトは off**で、ワーカーで例外が起きるとラン全体が中断する。 |

> `--skip-failed` で記録された失敗は `job_summary.json` の `episodes` 配列に `success: false` + `error` 文字列として残ります（`n_failed` にも反映）。

### 出力されるランメタデータ

`convert` は変換のたびに `meta/` 配下へ次の 2 ファイルを書き出します（dry-run を除く）。

| ファイル | 内容 |
|---|---|
| `meta/conversion_log.json` | **来歴（provenance）マニフェスト**。各入力 bag の `path` / `sha256`（記録ペイロードのみのハッシュ。`metadata.yaml` は除外）/ `frame_count` / `processing_time_s`、有効なコーデック・`ffmpeg_preset` / `ffmpeg_crf`、`fps`、`total_episodes` / `total_frames`、config の全文スナップショット（`config_snapshot`）とその `config_sha256`、`bagel_version` / `ffmpeg_version`、`run_timestamp`（ISO-8601 UTC）。 |
| `meta/job_summary.json` | **ランの統計**。`n_episodes` / `n_success` / `n_failed`、`total_frames`、`wall_time_s`、スループット `frames_per_min`、`input_bytes` / `output_bytes`、`workers`（ワーカー別の件数/フレーム/時間）、`episodes`（エピソード別の `index` / `bag_path` / `worker` / `success` / `n_frames` / `processing_time_s` / `error`）。 |

`--json` を付けると `job_summary.json` と同じ dict が stdout にも出力されます。

## config の新キー（split / TF 特徴量 / タイポ検出）

`convert` の挙動を変える `robot_config.yaml` 側の新キーです。詳細な設定リファレンスは [`configuration.md`](configuration.md)、ここでは新キーの要点のみ示します。

### `split:`（train/val/test 分割 + 長さフィルタ）

トップレベルに `split:` ブロックを置くと、エピソードを `train` / `val` / `test` に分割し、短いエピソードを除外できます。

```yaml
split:
  train: 0.8
  val: 0.1
  test: 0.1
  min_length: 30   # 30 フレーム未満のエピソードは split 計算前に除外
```

| キー | デフォルト | 説明 |
|---|---|---|
| `train` | `1.0` | `train` split に割り当てる比率（`[0, 1]`）。 |
| `val` | `0.0` | `val` split の比率。 |
| `test` | `0.0` | `test` split の比率。 |
| `min_length` | `0` | このフレーム数未満のエピソードを除外する（`0` は全保持）。 |

- 3 つの比率は合計 `1.0` でなければなりません（許容誤差 `1e-6`）。
- 分割は **決定的かつ連続**で、`[0, total_episodes)` を隙間なく被覆します。件数は `n_train = round(train·N)`、`n_val = round(val·N)`、`n_test = N − n_train − n_val`（**端数は test が吸収**）。幅 0 の split は省略されます。
- デフォルト（`train=1.0`）では `info.json["splits"]` は `{"train": "0:N"}` となり、**従来の単一 split とバイト一致**します。多分割時は `{"train": "0:5", "val": "5:6", "test": "6:7"}` のように複数の連続レンジになります。

### `frame_from` / `frame_to`（TF lookup 特徴量）

特徴量に `frame_from` と `frame_to`（**必ず両方セット**）を指定すると、その特徴量は単一トピックではなく `/tf` + `/tf_static`（`tf2_msgs/msg/TFMessage`）から出力フレームグリッド上でサンプリングされる **TF 特徴量**になります。`topic` / `msg_type` は引き続き必須で、`topic: /tf`、`msg_type: tf2_msgs/msg/TFMessage` を指定します。

```yaml
observations:
  - key: observation.ee_pose
    topic: /tf
    msg_type: tf2_msgs/msg/TFMessage
    frame_from: tool0        # 姿勢を求めたいフレーム
    frame_to: base_link      # 基準フレーム
    # tf_topic: /tf          # デフォルト /tf
    # tf_static_topic: /tf_static   # デフォルト /tf_static
```

| キー | デフォルト | 説明 |
|---|---|---|
| `frame_from` | なし | 姿勢を求めるソースフレーム。`frame_to` とセットで指定。 |
| `frame_to` | なし | 姿勢を表す基準フレーム。`frame_from` とセットで指定。 |
| `tf_topic` | `/tf` | 動的 TF トピック名。 |
| `tf_static_topic` | `/tf_static` | 静的 TF トピック名。 |

- 出力は既定で 7 要素の pose `[tx, ty, tz, qx, qy, qz, qw]`。
- `selector: orientation.euler_xyz`（または `euler_xyz` / `...euler_zyx`）を付けると quaternion を **euler 角（ラジアン）** に置き換え、6 要素 `[tx, ty, tz, roll, pitch, yaw]` を出力します。
- 動的 TF は出力フレームグリッド上で **nearest-in-time**（時刻最近傍）で参照します。

> `euler_xyz` セレクタは TF 特徴量に限らず、quaternion を持つ任意のフィールド（例: `pose.orientation.euler_xyz`）にも使えます。

### タイポ検出（未知キーのエラー化）

`robot_config.yaml` の未知キーは黙って無視されず、**`ValueError` を送出**します。近いキーがあれば `difflib` による `(did you mean: X?)` サジェストが付きます。対象は全セクション（トップレベル / feature mapping / resampling / split / custom_msgs entry）。

```
Unknown feature mapping key: 'topci' (did you mean: topic?)
```

加えて [`validate-config`](#validate-config) の `--suggest-fixes` は、`image_size` 不一致についてコピペ可能な修正スニペットを summary の後に出力します。

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
| `--suggest-fixes` |  | off | summary の後に、image shape 不一致についてコピペ可能な `image_size` 修正スニペットを出力する。 |
| `--json` |  | off | 検証レポート（`{config, bag, results}`）を stdout に JSON 出力（人間向け summary は抑制）。 |

出力は以下のカテゴリに分けて表示され、終了ステータスは `strict` とエラー件数で決まります。

| カテゴリ | レベル | 内容 |
|---|---|---|
| `missing_required_topics` | ERROR | `optional: false` の特徴量のトピックが bag にない |
| `msg_type_mismatches` | ERROR | YAML の `msg_type` と bag 側 msgdef が不一致 |
| `image_shape_mismatches` | WARN | YAML `image_size` と実デコード shape が不一致 |
| `missing_optional_topics` | INFO | `optional: true` のトピックが bag に存在しない |
| `unused_bag_topics` | INFO | bag 側にはあるが config から参照されていないトピック |

## `validate-dataset`

生成済みデータセットが LeRobot Dataset v3.0 の構造に準拠しているか検証します。CI で生成物の破損や仕様逸脱を検知する用途を想定。

```bash
bagel validate-dataset --dataset /path/to/output_dataset/
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--dataset DIRECTORY` | ✓ | — | 検証対象の LeRobot v3.0 データセットルート。 |
| `--strict` |  | off | WARN レベルの問題（余分なカラム）もエラー扱いにして exit code を非ゼロにする。 |
| `--json-out FILE` |  | なし | 指定すると検証レポートを JSON で書き出す。 |
| `--json` |  | off | 検証レポート dict を stdout に JSON 出力（人間向け summary は抑制）。 |

検証内容:

- 必須ファイルの存在。
- `meta/info.json` の必須キー／値（`codebase_version` は `v3.0` であること、`splits` は `"0:{total_episodes}"` であること）。
- `data/*.parquet` のカラムの pyarrow 型。
- `tasks` / `episodes` parquet の存在と整合。
- エピソード数・フレーム合計のクロスチェック。

終了ステータス:

| code | 意味 |
|---|---|
| `0` | エラーなし（`--strict` 時は WARN もなし）。 |
| `1` | バリデーションエラーあり（`--strict` では WARN もエラーに昇格）。 |
| `2` | parquet が読めない等のセットアップエラー。 |

## `quality-report`

生成済みデータセットのデータ品質をスコアリングし、スコアカードを出力します。

```bash
bagel quality-report \
  --dataset /path/to/output_dataset/ \
  -o report.json
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--dataset DIRECTORY` | ✓ | — | 対象の LeRobot v3.0 データセットルート。 |
| `-o, --report FILE` |  | なし | 指定すると品質レポートを JSON で書き出す。 |
| `--freeze-std-eps FLOAT` |  | `0.001` | フリーズフレーム検出の、ペア毎 std しきい値。 |
| `--range-tol FLOAT` |  | `0.0` | 範囲外判定に加える、`stats.json` の min/max への絶対許容値。 |
| `--score-threshold FLOAT` |  | `0.95` | OK 判定とする最小品質スコア。 |
| `--json` |  | off | 品質レポート dict を stdout に JSON 出力（人間向け summary は抑制）。 |

レポート内容:

- 特徴量ごとの null / NaN 率。
- `stats.json` の min/max に対する範囲外率（`--range-tol` を加味）。
- フリーズフレーム（**報告のみ**で、これ単独で fail にはならない）。
- 動画フレーム数 ↔ データ行数の突き合わせ（**不一致は HARD FAIL**）。
- 全体スコア `[0, 1]`（重み: null=0.5 / range=0.3 / freeze=0.2）。

終了ステータス:

| code | 意味 |
|---|---|
| `0` | スコアが `--score-threshold` 以上、かつ動画フレーム不一致なし。 |
| `1` | スコアがしきい値未満、またはいずれかの動画でフレーム不一致。 |
| `2` | メタデータ欠落・読めない等のセットアップエラー。 |

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
| `--json` |  | off | `AuditReport` dict を stdout に JSON 出力（人間向け summary は抑制）。 |
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
| `--json` |  | 結果 dict を stdout に JSON 出力（人間向け summary は抑制）。 |

成功すれば緑で `OK`、失敗すれば赤で `INVALID: <error>` を出して exit 1。

## `preview`

生成済みデータセットの **自己完結型 HTML プレビューレポート**を 1 ファイルで書き出します。サマリ・品質スコアと表・サンプル動画フレームのギャラリー（inline base64）・特徴量ごとの数値統計を、外部アセットなしの単一 HTML にレンダリングします。**読み取り専用**でサーバは不要です。

```bash
bagel preview --dataset /path/to/output_dataset/
# => /path/to/output_dataset/meta/preview.html
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--dataset DIRECTORY` | ✓ | — | 対象の LeRobot v3.0 データセットルート。 |
| `--n-frames INT` |  | `3` | video キーごとに埋め込むサンプルフレーム数。 |
| `-o, --out FILE` |  | `<dataset>/meta/preview.html` | 出力 HTML パス。 |
| `--sample-video / --no-sample-video` |  | `--no-sample-video` | mp4 をデコードしてフリーズフレーム数を数え、品質セクションに含める。 |

終了ステータス:

| code | 意味 |
|---|---|
| `0` | HTML を正常に生成。 |
| 非ゼロ | メタデータ欠落・読めない等のセットアップエラー。 |

### 例

```bash
# 各カメラ 5 フレームを埋め込み、フリーズ検出も実施して出力先を指定
bagel preview \
  --dataset /path/to/output_dataset/ \
  --n-frames 5 \
  --sample-video \
  -o /tmp/preview.html
```

## `push-to-hub`

生成済みデータセットを **HuggingFace Hub にアップロード**し、データセットカードを自動生成します。**任意（opt-in）機能**で、実アップロードには HF 認証が必要です。

> **まず `--dry-run` で確認.** `--dry-run` を付けると **何もアップロードせず**、計画される `repo_id`・対象ファイル数・カードのプレビューだけを出力します（`--card-out` を併用するとカードをファイルにも書き出す）。実アップロード前の確認用にまずこれを実行してください。

```bash
# 計画のみ（アップロードなし）
bagel push-to-hub --dataset /path/to/output_dataset/ --dry-run

# 実アップロード（HF 認証が必要）
bagel push-to-hub --dataset /path/to/output_dataset/ --repo-id myorg/my-dataset
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--dataset DIRECTORY` | ✓ | — | 対象の LeRobot v3.0 データセットルート。 |
| `--repo-id TEXT` |  | `info.json['repo_id']` | HuggingFace データセット repo id。未指定かつ `info.json` にも無ければ exit 2。 |
| `--dry-run` |  | off | アップロードせず計画のみ出力（`repo_id` + ファイル数 + カードプレビュー）。 |
| `--private` |  | off | repo を private で作成（`--dry-run` 時は無視）。 |
| `--token TEXT` |  | 環境ログイン | HuggingFace 認証トークン（未指定時は ambient なログインを使用）。 |
| `--card-out FILE` |  | なし | `--dry-run` 時に、生成したカードをこのパスにも書き出す。 |

- `--repo-id` は未指定なら `info.json['repo_id']` にフォールバックします。どちらも無ければ **exit 2**。
- `--dry-run` **なし**ではデータセットがアップロードされ、カードは repo ルートに配置されます（実行には HF 認証が必須）。

終了ステータス:

| code | 意味 |
|---|---|
| `0` | アップロード（または dry-run の計画出力）が成功。 |
| `2` | `repo_id` が解決できない等のセットアップエラー。 |

### 例

```bash
# dry-run でカードを確認しつつファイルに保存
bagel push-to-hub \
  --dataset /path/to/output_dataset/ \
  --repo-id myorg/my-dataset \
  --dry-run \
  --card-out /tmp/README.md
```

## `to-mcap`

ROS1 `.bag` 録画を ROS2 MCAP bag に変換します。`bagel` 本体は ROS2 bag（mcap/sqlite3）しか読めないため、ROS1 録画（例: airoa raw データセット）を事前変換して `bagel convert` に渡せるようにする用途です。

```bash
bagel to-mcap -o /path/to/out_bags/ /path/to/ros1_bags/
```

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `-o, --output DIRECTORY` | ✓ | — | 出力ベースディレクトリ。各 bag は `<output>/<name>/` に書き出される。 |
| `--overwrite` |  | off | 既存の出力 bag ディレクトリを上書きする。 |
| `--dst-version INT` |  | `9` | 書き出す ROS2 bag フォーマットのバージョン。 |
| `--json` |  | off | 変換結果 dict を stdout に JSON 出力（人間向け summary は抑制）。 |

`SOURCES` は `.bag` ファイルまたはディレクトリ（再帰的に `*.bag` を探索）を指定できます。

## `ui`

bag 閲覧 → config の scaffold/編集 → convert（進捗表示付き）→ 品質確認 + preview のループを回す **localhost コントロール UI** を起動します。`127.0.0.1` のみにバインドし、起動ごとのセッショントークンを発行し、フロントエンドのバンドルと許可リスト式 JSON API を配信します。トークン付き URL を表示し（`--no-open` を付けない限り）ブラウザで開きます。

```bash
bagel ui \
  --bags-root /path/to/ros_sample_bag_dir/ \
  --output-root /path/to/output/
```

### フロントエンド／バックエンド分離

- **バックエンド（Python）.** 標準ライブラリの `http.server` だけで動き、**新規依存はありません**。すべての権限（ファイルアクセス・プロセス実行）は許可リスト式 JSON API の背後に閉じ込められています。`bagel` の `--json` API と `bagel preview` を再利用します。
- **フロントエンド（TypeScript + HTML）.** [`ui/`](../ui/README.md) にあり、esbuild でバンドルされます。**表示専用**で権限を持ちません。各 UI 操作には等価な `bagel ...` CLI コマンドが表示され、コピーできます（**CLI が真実の源**）。
- Docker ではなく **ホストプロセス**として動くため、許可リストのルート配下にある bag にどこからでも到達できます。

### フロントエンドのビルド（前提）

`bagel ui` は `ui/dist/` がビルド済みならそれを配信し、未ビルドなら同梱のプレースホルダページにフォールバックします。実際の UI を使うには、初回に一度フロントエンドをビルドしてください（**node/npm はビルド時のみ必要**。ランタイムは Python のみ）。

```bash
cd ui
npm install        # esbuild + typescript（devDependencies のみ）
npm run build      # typecheck + bundle -> ui/dist/
```

出力は `ui/dist/`（`index.html` + `bundle.js`）。詳細は [`ui/README.md`](../ui/README.md) を参照。

### オプション

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--bags-root DIRECTORY` | ✓ | — | bag を閲覧・読み取りできる許可ルートディレクトリ。**繰り返し指定可**（`--bags-root A --bags-root B`）。 |
| `--output-root DIRECTORY` | ✓ | — | 変換出力 + データセット読み取りの許可ルートディレクトリ。 |
| `--port INTEGER` |  | `8765` | `127.0.0.1` でバインドする TCP ポート。 |
| `--no-open` |  | off | 起動時にトークン付き URL をブラウザで開かない。 |

### セキュリティ

- **`127.0.0.1` 限定バインド.** `--host` オプションはなく、ネットワークには公開されません。
- **起動ごとのセッショントークン.** 起動のたびにトークンを発行し、URL に `?token=...` として表示します。API はこのトークンを要求します。
- **ファイルアクセスの限定.** すべてのファイルアクセスは `--bags-root` / `--output-root` 配下に限定され、**パストラバーサルは遮断**されます。

### 例

```bash
# 複数の bag ルートを許可し、ポートを変えてブラウザは開かない
bagel ui \
  --bags-root /data/hsr_bags/ \
  --bags-root /data/so101_bags/ \
  --output-root /data/datasets/ \
  --port 9000 \
  --no-open
```

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

### 13. 未知ロボットの config を雛形生成

```bash
bagel scaffold \
  --bags /path/to/ros_sample_bag/ \
  -o configs/my_robot.yaml \
  --robot-type my_robot \
  --task "pick and place"
```

生成された YAML のコメント候補（`observation.state` / `actions`）を手で整えてから `convert` に進みます。

### 14. 生成データセットの構造検証

```bash
bagel validate-dataset \
  --dataset /path/to/output_dataset/ \
  --strict \
  --json-out /tmp/validate_dataset.json
```

### 15. 生成データセットの品質スコアリング

```bash
bagel quality-report \
  --dataset /path/to/output_dataset/ \
  -o /tmp/quality.json \
  --score-threshold 0.9
```

### 16. クラッシュ後に安全に再変換（--resume）

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag_dir/ \
  --output /path/to/output_dataset/ \
  --resume
```

### 17. 失敗を許容しつつ JSON サマリを取得（--skip-failed / --json）

```bash
bagel convert \
  --config configs/my_robot.yaml \
  --bags /path/to/ros_sample_bag_dir/ \
  --output /path/to/output_dataset/ \
  --skip-failed --quiet --json > /tmp/job_summary.json
```

### 18. HTML プレビューを生成

```bash
bagel preview \
  --dataset /path/to/output_dataset/ \
  --n-frames 5 \
  --sample-video
```

### 19. Hub へのアップロードを dry-run で確認

```bash
bagel push-to-hub \
  --dataset /path/to/output_dataset/ \
  --repo-id myorg/my-dataset \
  --dry-run \
  --card-out /tmp/README.md
```
