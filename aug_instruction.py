import json
import re
from vllm import LLM, SamplingParams
from pathlib import Path

import os

# スクリプトの冒頭で環境変数を設定
os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
os.environ["TRITON_CACHE_DIR"] = "./.cache"
os.environ["VLLM_CACHE_ROOT"] = "./.cache"

from vllm import LLM, SamplingParams

def extract_similar_task(text: str) -> str | None:
    """
    モデル出力からsimilar_taskを抽出する
    様々な形式に対応
    """
    # パターン1: 正規のJSON形式 {"similar_task": "..."}
    json_pattern = r'\{\s*"similar_task"\s*:\s*"([^"]+)"\s*\}'
    match = re.search(json_pattern, text)
    if match:
        return clean_task_text(match.group(1))
    
    # パターン2: シングルクォートのJSON {'similar_task': '...'}
    json_pattern_single = r"\{\s*'similar_task'\s*:\s*'([^']+)'\s*\}"
    match = re.search(json_pattern_single, text)
    if match:
        return clean_task_text(match.group(1))
    
    # パターン3: similar_task: "..." または similar_task: '...'
    key_value_pattern = r'similar_task["\']?\s*:\s*["\']([^"\']+)["\']'
    match = re.search(key_value_pattern, text)
    if match:
        return clean_task_text(match.group(1))
    
    # パターン4: 最後の有効なJSON部分を探す（後方から検索）
    all_jsons = re.findall(r'\{[^{}]+\}', text)
    for json_str in reversed(all_jsons):
        try:
            parsed = json.loads(json_str)
            if "similar_task" in parsed:
                return clean_task_text(parsed["similar_task"])
        except json.JSONDecodeError:
            continue
    
    return None


def clean_task_text(text: str) -> str:
    """
    タスクテキストから不要な接頭辞を削除する
    例: "1. Go to the shelf" -> "Go to the shelf"
    """
    # 先頭の番号付きリスト形式を削除 (1. 2. など)
    text = re.sub(r'^\s*\d+\.\s*', '', text)
    # 先頭の箇条書き形式を削除 (- * など)
    text = re.sub(r'^\s*[-*•]\s*', '', text)
    # 先頭・末尾の空白を削除
    text = text.strip()
    return text


def load_episode(dataset_path: str) -> dict:
    """
    LeRobot形式データセットのepisodes.jsonlから最初のepisodeを読み込む
    
    Args:
        dataset_path: データセットのルートパス
    
    Returns:
        最初のepisodeの辞書
    """
    jsonl_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    
    if not jsonl_path.exists():
        # meta/がない場合、直接episodes.jsonlを探す
        jsonl_path = Path(dataset_path) / "episodes.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"episodes.jsonlが見つかりません: {jsonl_path}")
    
    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    
    if not episodes:
        raise ValueError("episodes.jsonlが空です")
    
    return episodes

def extract_instruction(episode: dict) -> str:
    """
    episodeからinstructionを抽出する
    
    Args:
        episode: episodeの辞書
    
    Returns:
        instruction文字列
    """
    # LeRobot形式では tasks や language_instruction などのキーを確認
    if "tasks" in episode and episode["tasks"]:
        return episode["tasks"][0], episode["short_horizon_task"], episode["primitive_action"]
    else:
        raise KeyError(f"instructionが見つかりません。利用可能なキー: {list(episode.keys())}")

def generate_similar_instructions(
    input_instruction: str,
    temperature: float = 0.8,
    max_tokens: int = 256
) -> list[str]:
    """
    入力instructionから類似のinstructionを生成する
    
    Args:
        input_instruction: 元となるinstruction
        num_outputs: 生成する類似instructionの数
        temperature: 生成の多様性（高いほど多様）
        max_tokens: 最大トークン数
    
    Returns:
        生成された類似instructionのリスト
    """
    # モデルの初期化（instructionモード）
    llm = LLM(
        model="openai/gpt-oss-120b",
        trust_remote_code=True,
    )
    
    # サンプリングパラメータ設定
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens
    )
    
    # instructionモード用のプロンプト
    prompt = f"""You are an excellent task generation assistant.
Based on the given task, generate a similar task with the same meaning and purpose.

You must output in the following JSON format:
{{
    "similar_task": "<generated similar task>",
}}

### Original task:
{input_instruction}

### Similar task:
"""
    
    # 生成実行
    outputs = llm.generate([prompt], sampling_params)
    
    # 結果をパースして抽出
    result = ""
    output = outputs[0].outputs[0]
    generated = output.text.strip()
    similar_task = extract_similar_task(generated)
    if similar_task:
        result = similar_task
    else:
        # 抽出失敗時は生のテキストを保存（デバッグ用）
        print(f"Warning: Failed to extract similar_task from: {generated[:100]}...")

    return result

if __name__ == "__main__":
    # 使用例
    dataset_path = "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot/hsr/airoa-hsr-all-v1.0-202505-09-success-stat-curation"

    print(f"データセット読み込み中: {dataset_path}")
    episodes = load_episode(dataset_path)
    first_episode = episodes[0]
    instruction, _, _ = extract_instruction(first_episode)

    print("=== 単一instruction生成 ===")
    similar = generate_similar_instructions(instruction)
    print(f"元のインストラクション：{instruction}\n生成したインストラクション：{similar}")