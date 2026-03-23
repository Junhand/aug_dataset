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

pip install diffusers accelerate transformers bitsandbytes
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


# オーギュメント内容
| 軸 | idx | 内容 |
|---|---|---|
| 空間変化 | 0 | 部分変化耐性 |
| トーン明度 | 1-4 | ・白飛び風 ・暗所風 ・HDR風 ・フラット/褪色 |
| 色シフト | 5-7 | ・暖色化 ・寒色化 ・モノクロ+ノイズ |
| テクスチャ | 8-9 | ・強ぼかし ・過剰シャープ+ノイズ |
| スタイル | 10-13 | ・ポスタリゼーション(4bit/3bit) ・ソラリゼーション ・ヒストグラム均一化 |
| 幾何 | 14-16 | ・透視変換 ・アフィン変換 ・弾性変形 |
| 複合 | 17-19 | ・監視カメラ風 ・アクションカム風 ・オクルージョン耐性 |


データ拡張として**視覚的に最大限多様**になるよう、7つの知覚軸に分けて設計。

## 1. **空間変化 (0)** — 画像変換

**プリセット0**: 画像変換。

- SAM3とQwen-Image-editを利用し、背景変換を行う。
- 画像に対する頑健性、多様性を高める。

## 2. **トーン/明度** (1-4) — 明るさ・コントラスト

**プリセット1**: 露出オーバー。

- 明るさを上げてコントラストを下げ、白飛びした写真のような見た目にする。
- 窓際や照明直下での撮影を模擬。

**プリセット2**: 露出アンダー。

- 暗くしてコントラストを強め、さらにノイズを加える。
- 暗い部屋でのカメラノイズを再現。

**プリセット3**: HDR風。

- コントラストと彩度を両方強くする。
- 明暗差がくっきりした鮮やかな映像にする。

**プリセット4**: フラット/褪せた映像。

- コントラストを下げて彩度もほぼゼロにする。
- 古い監視カメラや曇天下のような映像にする。

## 3. **色シフト** (5-7) — 色味のシフト

**プリセット5**: 暖色シフト。

- 色相を黄〜赤方向にずらし彩度を上げる。
- 白熱灯下での撮影を模擬する。

**プリセット6**: 寒色シフト。

- 色相を青方向にずらし暗くする。
- 蛍光灯やLEDの青白い環境を再現する。

**プリセット7**: ほぼグレースケール + 強ノイズ。

- 彩度をほぼゼロにして粗いノイズを加える。
- 安価なモノクロカメラの映像に近くなる。

## 4. **テクスチャ** (8-9) — テクスチャ・質感

**プリセット8**: 強いガウシアンブラー。

- ピンボケや手ブレを模擬。

**プリセット9**: 過度なシャープニング + ノイズ。

- 安価なセンサーがエッジ強調した結果ノイズが目立つ映像を再現。

## 5. **スタイル** (10-13) — 色調マッピング

**プリセット10**: ポスタリゼーション（4bit）。

- 色の階調を16段階に減らす。
- やや平坦な色味にする。

**プリセット11**: 強ポスタリゼーション（3bit）+ 高彩度。

- 色を8段階まで減らしつつ彩度を上げる。
- アニメ/漫画的な見た目にする。

**プリセット12**: ソラリゼーション。

- 閾値0.7以上の明るいピクセルを反転させる。
- 部分的にネガフィルムのような不自然な色とする。

**プリセット13**: ヒストグラム均等化。

- 画像全体のコントラストを自動調整し、暗い部分と明るい部分の差を均一化。
- 局所的なディテールが強調される。

## 6. **幾何** (14-16) — 幾何変形

3つとも**ロボティクス安全**として、空間関係が大きく壊れない程度の軽い変形にとどめている。水平反転は左右が反転するとロボットの行動が破綻するため不使用。

**プリセット14**: 軽い透視変換（distortion_scale=0.15）。

- カメラの角度が少し変わったような歪みを加える。

**プリセット15**: 軽いアフィン変換。

- ±5度回転、±5%移動、±8%スケールを組み合わせる。
- カメラの微小な位置ずれを模擬。

**プリセット16**: 弾性変形。

- 画像を有機的にゆがめる。
- 柔軟な表面やレンズ歪みの再現に近い。

## 7. **複合** (17-19) — 複合

**プリセット17**: 監視カメラ風。

- ブラー + 暗い + グレースケール + 強ノイズの4つを組み合わせる。
- 低品質な監視映像を再現。

**プリセット18**: アクションカメラ風。

- 色ジッター + 透視変換 + シャープニングを組み合わる。
- 動きのある鮮明な映像にする。

**プリセット19**: オクルージョン耐性。

- オートコントラスト + ランダム消去を組み合わせる。
    
    （ランダム消去は、画像の2〜12%をランダムに黒で塗りつぶす）
    
- 物体の一部が隠れている状況に対する頑健性を高める。
