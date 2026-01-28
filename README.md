# LeRobot Similar Instruction Generator

LeRobot形式のデータセットからタスク指示（instruction）を読み込み、vLLMを使用して類似のタスク指示を生成するツールです。

## 概要

ロボット学習データセットの言語指示を多様化するために、既存のタスク指示から意味的に同等な別の表現を生成します。これにより、Vision-Language-Action (VLA) モデルの汎化性能向上が期待できます。

## 要件

- Python 3.12
- vLLM
- CUDA対応GPU

## データセット拡張の実行環境インストール

```bash
conda create --prefix ./.aug_env python=3.10
conda activate .aug_envの環境先
conda install ffmpeg -c conda-forge
pip install -r requirements.txt
# pip install lerobot==0.4.3
# pip install av==15.1.0 numpy==2.2.6
# pip install requests python-dotenv
# pip install ruff mypy pre-commit
```

## GPT-OSSモデルの環境インストール
```bash
conda create --prefix ./.vllm_env python=3.12
pip install vllm
```

## 画像拡張モデルの環境インストール
Qwen-Image-Edit-2511の利用
```bash
conda create --prefix ./.edit_env python=3.12
conda activate .image_envの環境先

pip install diffusers accelerate transformers
pip install "huggingface_hub[cli]"

# モデルのインストール
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir ./Qwen-Image-Edit-Lightning
pip install torch torchvision peft
pip install uvicorn[standard] fastapi[all] pillow pydantic

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install einops decord pycocotools
```


## エラー対応
- ``Disk quota exceeded``の場合は、``aug_dataset/.tmp``のようにtmp先を変更するか、``pip install lerobot==0.4.3 --no-cache-dir``のように``--no-cache-dir``をつけてください。
- ``prod(-1)``のエラーが出る場合は、``.cpu().prod(-1)``として、CPUで処理してください。


## 使用方法

### vllmサーバの立ち上げ
```
sbatch vllm_server/serve_gpt-oss-120b.sh 
```

下記を自分の環境に合わせてください。
```
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.aug_env
```

以下で、疎通確認をしてください。
```
curl -X POST http://aic-gh2b-310033:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {
        "role": "user",
        "content": "日本の首都は？"
      }
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

以下のような返答があれば、OKです。
```
{"id":"chatcmpl-bcd524af9a884fc88fba39f6b6a94551","object":"chat.completion","created":1769329277,"model":"openai/gpt-oss-120b","choices":[{"index":0,"message":{"role":"assistant","content":"日本の首都は東京（とうきょう）です。","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":"The user asks in Japanese: \"日本の首都は？\" which means \"What is the capital of Japan?\" It's a straightforward factual question. No policy issue. Answer: Tokyo."},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":77,"total_tokens":138,"completion_tokens":61,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```

### qwen-edit-image & sam3サーバの立ち上げ
```bash
sbatch run_augment_lerobot_dataset.sh
```
以下で、疎通確認をしてください。
```
curl -X GET http://aic-gh2b-310034:11303/health
```
以下のような返答があれば、OK。
```
{"status":"healthy","ready":true,"num_gpus":8,"processes_alive":8}
```


### 基本的な使用

```python
cp .env.sample .env
```
.envに、vllmの情報を記入

```
python src/augment_lerobot_dataset.py \
  --src-repo-id hsr/2025-09_task48_absolute \
  --dst-repo-id hsr/2025-09_task48_absolute_aug \
  --api-url http://aic-gh2b-310034:11303 \
  --max-workers 56 \
  --offline \
  --use-batch 
```
スクリプト内の `dataset_path` を対象のLeRobotデータセットパスに変更してください。


### データセットから生成

```bash
python aug_instruction.py
```

### 動画拡張

```bash
python aug_movie.py
```

### 全体拡張
```
sbatch run_augment_lerobot_dataset.sh
```

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

## tips
```
  File "aug_dataset/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1171, in get_image_features
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
                   ^^^^^^^^^^^^^^^^^^^^^^^
```
のエラー時は、
``aug_dataset/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py``
の1171行目を以下に修正する。
```
split_sizes = (image_grid_thw.to("cpu").prod(-1) // self.visual.spatial_merge_size**2).tolist()
```

## 静的解析・コード整形

本プロジェクトでは、以下のツールを用いて **静的解析・コード整形** を行います。

- **Ruff**：コードフォーマット・Lint
- **mypy**：型チェック
- **uv**：依存管理および実行環境
- **Makefile**：コマンドの簡略化
- **pre-commit**：コミット前チェック

解析対象は **`src/` ディレクトリ配下のみ** です。

---

### 静的解析実行方法
- 自動修正（フォーマット + Lint 修正）
  ```
  make fix
  ```

- チェック（修正せず検査のみ）
  ```
  make check
  ```

- コミット前に自動で静的解析（pre-commit）
  ```
  uv run pre-commit install（1回のみ）
  ```