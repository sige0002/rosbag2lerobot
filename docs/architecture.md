# アーキテクチャと内部処理

`rosbag2lerobot` が ROS2 rosbag を LeRobot v3.0 データセットに変換する仕組みを、モジュール単位でまとめます。拡張・デバッグ・内部挙動の理解が必要な方向け。

## 目次

- [パイプライン全体像](#パイプライン全体像)
  - [診断系モジュール](#診断系モジュール)
- [モジュール責務一覧](#モジュール責務一覧)
- [型システムと `.msg` 登録 (`reader.py`)](#型システムとmsg登録-readerpy)
- [デコーダ (`decoders/`)](#デコーダ-decoders)
- [リサンプラ (`resampler.py`)](#リサンプラ-resamplerpy)
- [`trim_to_valid_range` が何をしているか](#trim_to_valid_range-が何をしているか)
- [ライター (`writer.py`)](#ライター-writerpy)
- [統計 (`stats.py`)](#統計-statspy)
- [CLI (`cli/`)](#cli-cli)
- [並行性とメモリプロファイル](#並行性とメモリプロファイル)
- [不変条件](#不変条件)

## パイプライン全体像

```
┌──────────────┐   ┌─────────────┐
│ ROS2 rosbag  │   │ robot_config│
│ (mcap/db3)   │   │   .yaml     │
└──────┬───────┘   └──────┬──────┘
       │                  │
       ▼                  ▼
   config.load_config            YAML のパース＋バリデーション
       │ RobotConfig
       ▼
   reader.BagReader              bag を開き、型を登録
       │ (+ _override_types_from_bag : 埋め込み msgdef 優先)
       ▼
   decoders.decode               msg → np.ndarray / PIL.Image
       │ (key, ts_ns, value)
       ▼
   resampler.Resampler           可変レート → 固定 FPS
       │ (+ trim_to_valid_range : 先頭／末尾の null を削る)
       ▼
   writer.DatasetWriter          parquet + MP4 + メタデータ
       │  (+ stats.StatsComputer)
       ▼
   LeRobot v3.0 データセット
   (data/, videos/, meta/)
```

CLI エントリ (`cli/convert.py::convert`) が各 bag についてこのパイプラインを上から順に走らせます。1 bag ディレクトリ = 1 エピソード。

### 診断系モジュール

変換本線とは独立した診断機能は `diagnostics.py`（F1 FPS 統計 / F3 image_size 提案 / F4 config-bag 整合検証）と `audit.py`（F2 生成データセットのタイムスタンプ監査）に切り出してあり、CLI サブコマンド `inspect` / `validate-config` / `audit-timestamps` が薄いラッパとして呼び出します。純関数設計なので CI から直接 import しても使えます。

## モジュール責務一覧

| モジュール | 役割 |
|---|---|
| `config.py` | YAML → 型付き `RobotConfig` データクラス。下流は `dict` ではなくこのオブジェクトを参照する。`all_topics`, `topic_to_features`, `required_feature_keys` 等の派生ヘルパーを提供。 |
| `reader.py` | `rosbags.rosbag2.Reader` をラップ。カスタム `.msg` を登録した後、bag に埋め込まれた msgdef で上書きして CDR デシリアライズがワイヤフォーマットと必ず一致するようにする。 |
| `decoders/` | `msg_type → callable` のレジストリ。標準型（sensor_msgs / geometry_msgs / nav_msgs / std_msgs）には手書きデコーダ、未登録型には `msg_parser.MsgDecoder` が汎用的に数値を抽出するフォールバック。 |
| `resampler.py` | `(key, ts_ns, value)` タプル列を固定 FPS フレーム列に変換。`hold` / `nearest` / `drop` の3ポリシー。`trim_to_valid_range` で required 特徴量のそろう区間に切り詰め。 |
| `writer.py` | LeRobot 特徴量スキーマを構築し、フレームをカメラごとの常駐 `ffmpeg` エンコーダにストリーミング（各フレームを 1 回だけエンコード、200MB 到達でファイルをローテーション）、parquet / MP4 / `info.json` / `stats.json` / `tasks.parquet` / `episodes/` を出力。 |
| `stats.py` | Welford のオンラインアルゴリズムで min / max / mean / std / 分位数を算出し `meta/stats.json` に書き出す。 |
| `diagnostics.py` | 診断系の純関数群。`compute_topic_fps_report`（F1: トピック周期統計）、`detect_image_shape` / `_normalize_yaml_image_size`（F3: 画像 shape 検出）、`validate_config_against_bag` + `ValidationReport`（F4: config ↔ bag 整合検証）。CLI からも CI スクリプトからも直接呼べる。 |
| `audit.py` | 生成済みデータセット側の監査純関数群（F2）。`audit_episode_timestamps` が `meta/episodes/*.parquet` を走査し、`to_timestamp[i] == from_timestamp[i+1]` と mp4 境界のリセット規則を検証。結果は `AuditReport` / `VideoKeyAuditResult` / `BoundaryError` で構造化。 |
| `cli/` | Click ベース CLI パッケージ。コマンドごとにモジュール分割（`cli/convert.py`, `cli/inspect.py`, `cli/scaffold.py` ほか）。`cli/main.py` が Click グループと登録、`cli/_common.py` が共有ヘルパ、`cli/convert.py` がエピソード単位オーケストレーション (`_process_episode`)。 |

## 型システムと `.msg` 登録 (`reader.py`)

### なぜ二重登録が必要か

`rosbags.typesys.Typestore` は3つの dict（`fielddefs` / `types` / `cache`）を同期保持します。`register(d)` は既に異なるスキーマで同じ型名が登録されていると上書きを拒否するため、「YAML の `custom_msgs` で事前登録 → bag 側の msgdef が実は違うレイアウト」というケースで素朴に動きません。

### 解決策

`reader._force_register` が対象型の 3 dict をクリアしてから `typestore.register(d)` を呼び直します。bag ごとに 1 回 `dict.pop` するだけの軽コストです。

### bag 埋め込み msgdef を優先

rosbag2 ライターは公開トピックごとに正確な `.msg` テキストをメタデータ領域に埋め込みます。つまり「bag に書かれた定義 = 収録時点の真のスキーマ」。ローカルの `.msg` ファイルとズレていた場合はローカル側がドリフト（ファームウェア更新、フォーク等）したということなので、`reader._override_types_from_bag` は毎回 bag 側で上書きします。

### `.msg` ファイルが必要なケース

古い rosbag2 ライターは msgdef を埋め込まないものがあります。そうした bag に対しては YAML の `custom_msgs` に登録したローカル `.msg` がフォールバックとして働きます。

## デコーダ (`decoders/`)

### ディスパッチ

```
decode(msg_type, msg, selector, config)
  ├── msg_type が _DECODER_REGISTRY にある: その関数を呼ぶ
  └── それ以外: MsgDecoder(msg_type).decode(msg, selector, config)
                 └── MsgParser でメッセージを走査し数値リーフを抽出
```

### `_finalize` ヘルパー

全ての数値デコーダは `return _finalize(values, config)` で締めます。共通処理は:

1. `list` / `tuple` / `np.ndarray` を受ける
2. `float32` にキャスト
3. `unit_conversion`（数値乗数または `"rad2deg"` / `"deg2rad"`）を適用

### セレクタ

ドット区切りの属性パス（例: `"pose.position.x"`）。`_get_nested_attr` が `getattr` で走査します。`JointState` / `Joy` の一部メッセージは最初のドットで「field.index」や「field.joint_name」を分割する特殊処理を持ちます。YAML 作成者向けに `"lin.x"` のようなエイリアス (`FIELD_ALIASES`) も用意。

## リサンプラ (`resampler.py`)

### フレーム単位の探索

時刻 `frame_ns` のフレームに対して、各キーの ts_list に対し:

```
idx = np.searchsorted(ts_list, frame_ns, side='right') - 1
  ├── hold   : val_list[idx] を返す（idx < 0 かつ最初のメッセージが tolerance 外なら None）
  ├── nearest: |ts[idx]-frame_ns| と |ts[idx+1]-frame_ns| を比較
  │            tolerance_ns 以内で近い方、なければ None
  └── drop   : nearest と同じ（下流での None 扱いが異なる）
```

`numpy.searchsorted` による一括検索で、F フレーム × K キーを C レベル数回の呼び出しで完了させます。

### `hold` が既定で正しい理由

模倣学習の軌跡にはグリッパー開閉やタスク遷移のような疎なイベントが混じります。イベント間は物理系が最後の指令値を保持するので、`hold` がこのセマンティクスに一致します。`nearest` だとギャップにゼロが入り、ポリシーが「ほとんどの時間何もしない」と誤学習します。

## `trim_to_valid_range` が何をしているか

リサンプリング直後の frame リストは、先頭（センサー起動前のプリロール）や末尾（bag が記録中に切れた場合）に `None` が残ることがあります。LeRobot v3.0 は宣言された特徴量に null を許さないため、この関数で該当区間を落とします。

### アルゴリズム

```
first = 全 required キーが非 None である最初のフレーム
last  = 全 required キーが非 None である最後のフレーム
return frames[first : last+1]
  — frame_index は 0 から振り直し
  — timestamp は frame_index / fps で再計算
```

### required / optional の区別

- **required（`optional: true` でない特徴量）**: trim 判定の対象。どれか 1 つでも None だとそのフレームは先頭／末尾から切り落としの候補になる。
- **optional**: trim 判定には使わない。残った区間に None があってもライター側でゼロベクトルにフォールバックされる（LeRobot は「値があれば 0 でも可」だが null は不可）。

### 中間の欠損はどうなるか

`trim_to_valid_range` は先頭と末尾しか見ません。中間の欠損の扱いはリサンプラポリシー次第です。

- **`hold`**: 一度でもメッセージが届いた後は直前値が保持される → None は発生せず、parquet 書き込みに失敗しない。ただし「映像が途中で落ちた秒数」は直前フレームが固定表示され、タイムスタンプ上は連続した固定 FPS に見える。
- **`nearest` / `drop`**: `tolerance_ms` を超えた区間は None のまま残る → required ならその時点で書き込みが破綻する可能性。中間欠損を気にするパイプラインには不向き。

### タイムスタンプは再生成される

trim 後の `timestamp` は `frame_index / fps` で必ず 0 から振り直されます。元 bag の wall-clock は保持されません。録画の断絶時刻を明示的に残したい場合は、bag を断絶点で分割してそれぞれを別エピソードに変換してください。

### エッジケース

- **required キーが 1 つもない**: trim は何もせず入力をそのまま返す。
- **条件を満たすフレームが皆無**: 空リストを返し、CLI は警告を出してエピソード書き込みをスキップ。
- **境界で 1 特徴量だけ欠損**: その特徴量の制約が `first` を後方にずらす。

## ライター (`writer.py`)

### エピソードのステートマシン

```
DatasetWriter.__init__               ← 実行ごとに 1 回
  ├── 動画キーを検出
  └── data/, meta/episodes/, videos/<k>/ を確保

各エピソードで:
  各フレームで add_frame(dict)       ← _episode_frames に蓄積
  save_episode()                      ← parquet + 動画 + エピソードメタ書き出し
    └── エピソード単位バッファをリセット

finalize()
  ├── meta/tasks.parquet を書き出し
  ├── meta/episodes/chunk-000/file-XXX.parquet を書き出し
  ├── 統計を計算 (StatsComputer) → meta/stats.json
  └── meta/info.json を書き出し
```

### parquet 列の構築

`_write_data_parquet` が `_episode_frames` を 1 回走査して列ごとのリストを作ります。None 値に対してはシェイプに合わせたゼロベクトルにフォールバック（optional 特徴量のみ発生。required は trim で保証されるため null にならない）。画像は parquet 内では PNG バイト、正規の動画ストリームは `videos/<key>/` 配下の MP4。

### 動画エンコード

カメラごとに常駐する `ffmpeg` エンコーダ（`ffmpeg -f rawvideo ... -c:v <codec>`）へ
生 RGB をパイプします。連続するエピソードは同じ出力 mp4 にストリーミングされ、
ファイルサイズが `video_files_size_in_mb` を超えたエピソード境界でローテーション
します（各フレームのエンコードは 1 回のみ）。代表的なデフォルト:

- libsvtav1: `-preset 8 -crf 30`
- H.264 / HEVC NVENC: `-preset p4`
- 共通: `-pix_fmt yuv420p`（幅広い再生互換のため）

`ffmpeg` の非ゼロ終了や stderr は `RuntimeError` で再送出。

### `info.json` の契約

| キー | 意味 |
|---|---|
| `codebase_version` | 常に `"v3.0"` |
| `robot_type` | `RobotConfig.robot_type` |
| `total_episodes`, `total_frames`, `total_tasks`, `total_videos` | 全 finalize 後の累計 |
| `chunks_size` | チャンクあたりエピソード数（デフォルト 1000） |
| `fps` | 目標 FPS |
| `splits` | `{"train": "0:N"}` |
| `features` | `Dict[feature_key, {dtype, shape, names, ...}]` |
| `data_path`, `video_path` | テンプレートパス |

### PR #3239 互換のタイムスタンプ丸め

`meta/episodes/*.parquet` の `from_timestamp` / `to_timestamp` は float32 で、長尺データセットで累積ドリフトすると `FrameTimestampError` を誘発します。rosbag2lerobot は各値を `round(x, 6)` でマイクロ秒丸めし、次エピソードのオフセットにもその丸め値を繰り越すことで累積誤差を 1e-6 s に制限しています（LeRobot 本家 PR #3239 と同挙動）。

## 統計 (`stats.py`)

Welford オンラインアルゴリズムで特徴量ごとに `(n, mean, M2)` を蓄積し、`compute()` 時に:

- `min`, `max`, `mean`, `std` をフルシェイプで出力
- `q01`, `q50`, `q99` をリザーバサンプルから算出（フレームあたり O(1) メモリ）

画像特徴量は `(H, W, C)` → `(C,)` に平均プーリングで縮約し、stats ファイルサイズを抑えます。

## CLI (`cli/`)

### `convert` 内部フロー

```python
cfg = load_config(config_path)
# CLI オーバーライド（--task, --fps, --repo-id）適用
cfg = discover_bags(bags_path)[:max_episodes]
resampler = Resampler(fps=..., policy=..., tolerance_ms=...)

if workers > 1:
    episodes_iter = _iter_episodes_parallel(...)
else:
    episodes_iter = _iter_episodes_serial(...)

write_dataset(episodes=episodes_iter, config=cfg, output_dir=..., ...)
```

### `_process_episode` 内部

```python
with BagReader(bag_path, cfg) as reader:
    messages = []
    for topic, recv_ns, raw in reader.iter_messages(topics=cfg.all_topics):
        header_ns = extract_header_stamp_ns(raw)          # header.stamp（無ければ None）
        for fm in cfg.topic_to_features.get(topic, []):
            # (B) スタール drop: |recv − header| > 閾値 なら decode せず捨てる
            thr = fm.max_stamp_delay_ms or cfg.resampling.max_stamp_delay_ms
            if thr is not None and header_ns is not None \
               and abs(recv_ns - header_ns) > thr * 1e6:
                continue
            # (A) 採用ts: stamp_source=header かつ header あり → header_ns、それ以外 recv_ns
            ts = header_ns if (fm.stamp_source == "header" and header_ns is not None) else recv_ns
            # 画像は _LazyImage で遅延デコード（リサンプルで採用されたフレームのみ
            # 後段で materialize）。それ以外は即時 decode。
            decoded = decode(fm.msg_type, raw,
                             _split_selector(fm.selector),
                             _build_decoder_config(fm))
            messages.append((fm.key, ts, decoded))
    messages.sort(key=lambda m: m[1])

    # (C) start/stop の決定
    if cfg.resampling.align_to_required:
        window = _required_window(messages, cfg.required_feature_keys)
        if window is None:                # required 欠落 or 重なりなし → 空エピソード
            return []
        start_ns, end_ns = window
    else:
        start_ns, end_ns = reader.get_time_range()   # 従来: bag 全体範囲

    frames = resampler.resample(messages,
                                cfg.observation_keys + cfg.action_keys,
                                start_ns, end_ns)

if cfg.resampling.trim_to_valid and frames:
    frames = trim_to_valid_range(frames, cfg.required_feature_keys, cfg.fps)
return frames
```

主な設計判断:

- **(A) `stamp_source`**: `header` 指定かつ `header.stamp` が取れるトピックはその時刻を採用、それ以外は bag 受信時刻。`reader.extract_header_stamp_ns` が header 欠如・未設定（stamp=0）を `None` で返す。
- **(B) スタール drop**: `max_stamp_delay_ms`（feature 単位 ＞ `resampling` 全体）超過のメッセージは **decode 前に** 1 件破棄（`TRANSIENT_LOCAL` の古い latch 値対策。decode は高コストなので前段で弾く）。
- **(C) `align_to_required`**: `_required_window` が required 特徴量の `[max(最初), min(最後)]` 交差区間を返し、これを resample 範囲にすることで `hold` の区間外外挿（終了トピックのずれ）を防ぐ。`False` なら従来の bag 全体範囲。
- デコードは reader のコンテキスト内で完結させる（型登録が必ず通る）。
- `messages.sort(...)` は採用 ts 基準（header 採用で受信順と変わりうるため必須）。
- trim は `with` を抜けてから行う（bag を閉じてから in-memory 操作）。

## 並行性とメモリプロファイル

### 並行性

`--workers N` で `ProcessPoolExecutor` が走ります。bag ごとに独立デコード → `as_completed` で結果を集める → 元の bag インデックス順にソートしてライターに流す、というパイプライン。出力順は決定的です。`ffmpeg` は内部でマルチスレッドなので追加のスレッド管理は不要。

### メモリ

| ステージ | 割り当て |
|---|---|
| reader | bag サイズの約 1/10（rosbags がチャンク単位でストリーミング展開） |
| decoder | O(messages) の numpy 配列 |
| resampler | O(frames × features) の dict |
| writer | O(frames × features) + カメラごとの PIL 画像バッファ（`save_episode` まで保持） |

カメラ 3 台 / 480×640×3 / 30 fps / 30 秒のエピソードで画像バッファだけ約 240 MB（3 × 900 × 480 × 640 × 3 バイト）。OOM するようなら bag を分割して 1 bag = 1 エピソードにしてください。

`--workers N` にすると RAM はおおよそ `N × (1 エピソード分)` になります。

## 不変条件

下流コードが依存しているので、以下は破ってはいけません:

1. 各エピソードの `frame_index` は 0 始まり・連続（`trim_to_valid_range` の事前・事後条件）。
2. `timestamp` は常に `frame_index / fps`（秒、`float32`）。
3. `meta/info.json` で宣言された全特徴量列は `data/.../file-XXX.parquet` の全行に値を持つ（null なし）。
4. parquet 内の `observation.images.*` 列は PNG バイト、正規ビデオは `videos/<feature_key>/chunk-XXX/file-XXX.mp4`。
5. エピソードごとの `dataset_from_index` / `dataset_to_index` がグローバル `index` 列の境界と一致する。
