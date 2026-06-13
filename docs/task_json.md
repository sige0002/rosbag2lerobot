# `task.json` リファレンス

各 rosbag ディレクトリに任意で `task.json` を配置すると、そのエピソード固有の
**タスク文字列** と **時間範囲つきサブタスク** を指定できます。

```
bags/
├── ep0001/
│   ├── metadata.yaml
│   ├── ep0001.db3
│   └── task.json     ← これ
└── ep0002/
    └── ...
```

`task.json` は任意です。存在しなければ従来どおり CLI `--task` または YAML
`config.task` が使われます。

## スキーマ

```json
{
  "task": "Pick up the apple and place it in the basket",
  "subtasks": [
    {"start": 0.0, "end": 1.4, "subtask": "Approach the apple"},
    {"start": 1.4, "end": 2.8, "subtask": "Grasp the apple"},
    {"start": 2.8, "end": 4.0, "subtask": "Lift the apple"},
    {"start": 4.0, "end": 5.5, "subtask": "Move to basket"},
    {"start": 5.5, "end": 6.0, "subtask": "Release the apple"}
  ]
}
```

| フィールド | 必須 | 型 | 説明 |
|---|---|---|---|
| `task` | — | str | エピソードのタスク文字列。省略 / 空文字なら `--task` / YAML の `config.task` にフォールバック。 |
| `subtasks` | — | list[object] | サブタスク span の配列。省略 / 空配列なら当該エピソードにサブタスクなし。 |
| `subtasks[].start` | ✓ | number | 開始秒（エピソード先頭 `timestamp=0.0` からの相対秒）。 |
| `subtasks[].end` | ✓ | number | 終了秒。`start < end` 必須。 |
| `subtasks[].subtask` | ✓ | str | サブタスク名（非空文字列）。 |

未知のトップレベルフィールド（`description`, `tags` など）は無視されます（前方互換）。

## タスク解決の優先順位

最優先から順に:

1. `<bag>/task.json` の `task` フィールド（非空）
2. CLI `--task`
3. YAML `config.task`

## 時間指定の基準

`start` / `end` は LeRobot 出力の `timestamp` と同じ **エピソード先頭を 0.0 とした秒**
で記述してください。rosbag 全体の絶対時刻ではありません。また
`resampling.trim_to_valid = true`（既定）の場合、トリム後のエピソード先頭が
`timestamp=0.0` になる点に注意してください。

## サブタスクの全時間カバー要件

`subtasks` を 1 つでも指定した場合、以下を全て満たす必要があります（違反は
`ValueError` で変換を中断）:

- `subtasks[0].start == 0.0`
- すべての隣接ペアで `subtasks[i].end == subtasks[i+1].start`（ギャップ/オーバーラップ禁止）
- `subtasks[-1].end >= episode_duration = frame_count / fps`

最後の span の `end` は エピソード長を多少上回っていても OK（切り詰めて扱う）ですが、
不足すると末尾フレームが未割当になるためエラーになります。

## 出力への反映

- **すべての bag で `subtasks` が空 / 未指定** のとき: 出力は従来どおり。
  `subtasks.parquet` は生成されず、`data` parquet に `subtask_index` 列は付かない。
- **1 つでも subtasks を持つ bag があるとき**: 以下が追加で出力される。
  - `meta/subtasks.parquet` — `tasks.parquet` と同形式（index=subtask 文字列, col=`subtask_index`）
  - 全 `data/chunk-XXX/file-YYY.parquet` に `subtask_index` 列（int64）
    — その bag に subtasks が無いエピソードのフレームは `-1`
  - 各 `episodes/chunk-XXX/file-YYY.parquet` に `subtasks` 列（list[str]）
  - `info.json` に `total_subtasks` フィールド

## 最小の例（task のみ）

```json
{"task": "fold towel"}
```

## サブタスクなしで task だけ上書き

`task.json` がなくても `config.task` / CLI `--task` で足りるので、
通常は bag ごとに違うタスクを付けたいときだけ配置します。

## エラー例

```json
// NG: ギャップがある (1.0 → 1.5)
{"subtasks": [
  {"start": 0.0, "end": 1.0, "subtask": "a"},
  {"start": 1.5, "end": 5.0, "subtask": "b"}
]}
```

```json
// NG: 末尾が短い (エピソード長 5.0 に対して 3.0 まで)
{"subtasks": [
  {"start": 0.0, "end": 3.0, "subtask": "a"}
]}
```

```json
// NG: start が 0.0 でない
{"subtasks": [
  {"start": 0.5, "end": 5.0, "subtask": "a"}
]}
```
