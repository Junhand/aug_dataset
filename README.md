# LeRobot Similar Instruction Generator

LeRobot形式のデータセットからタスク指示（instruction）を読み込み、vLLMを使用して類似のタスク指示を生成するツールです。

## 概要

ロボット学習データセットの言語指示を多様化するために、既存のタスク指示から意味的に同等な別の表現を生成します。これにより、Vision-Language-Action (VLA) モデルの汎化性能向上が期待できます。

## 要件

- Python 3.12
- vLLM
- CUDA対応GPU

## インストール

```bash
uv sync
```

## 使用方法

### 基本的な使用

```python
from generate_similar_instruction import generate_similar_instructions

instruction = "Navigate to the shelf"
similar = generate_similar_instructions(instruction)
print(similar)  # 例: "Go to the shelf"
```

### データセットから生成

```bash
python aug_instruction.py
```

### 動画拡張

```bash
python aug_movie.py
```


スクリプト内の `dataset_path` を対象のLeRobotデータセットパスに変更してください。

## 設定

### 環境変数

スクリプト冒頭で以下の環境変数を設定しています。必要に応じて変更してください。

| 環境変数 | 説明 |
|----------|------|
| `HF_HOME` | HuggingFaceキャッシュディレクトリ |
| `TRITON_CACHE_DIR` | Tritonキャッシュディレクトリ |
| `VLLM_CACHE_ROOT` | vLLMキャッシュディレクトリ |

### 生成パラメータ

`generate_similar_instructions()` 関数のパラメータ:

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `temperature` | 0.8 | 生成の多様性（高いほど多様） |
| `max_tokens` | 256 | 最大生成トークン数 |

## データセット形式

LeRobot形式の `episodes.jsonl` を読み込みます。

```
dataset_root/
└── meta/
    └── episodes.jsonl
```

`episodes.jsonl` の各行は以下の形式です:

```json
{"episode_index": 0, "tasks": ["Navigate to the shelf"], "short_horizon_task": "...", "primitive_action": "..."}
```

## 出力例

```
データセット読み込み中: /path/to/dataset
=== 単一instruction生成 ===
元のインストラクション：Navigate to the shelf
生成したインストラクション：Go to the shelf
```

## 使用モデル

- `openai/gpt-oss-120b`: タスク指示生成用LLM（https://huggingface.co/openai/gpt-oss-120b）

## ライセンス

MIT License