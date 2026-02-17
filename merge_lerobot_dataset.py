import lerobot.datasets.dataset_tools as dt
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    # 1. データセットAの前半分（エピソードのインデックス）を特定
    repo_a = "hsr/2025-09_task48_relative_curated_v2"
    repo_b = "hsr/2025-09_task48_relative_curated_v2_aug_task_and_image3"
    ds_a = LeRobotDataset(repo_a)
    episodes = 100
    episodes_indices = list(range(episodes))

    # 2. 分割（Aの前半分だけを抽出した新しいデータセットを一時的に作成）
    # この関数は内部で動画やParquetを最適化して処理するため、for文より圧倒的に高速です
    dt.split_dataset(
        repo_id=repo_a,
        new_repo_id=f"{repo_a}_0-{episodes}",
        splits={"subset": episodes_indices} # 'subset'という名前の分割を作成
    )

    # 3. マージ（作成された Aの前半分 と B を統合）
    # 作成されたリポジトリ名は「新リポジトリ名_分割名」となります
    dt.merge_datasets(
        repo_ids=[f"{repo_a}_0-{episodes}", repo_b],
        new_repo_id=f"{repo_b}-partial-merge"
    )

if __name__ == "__main__":
    os.environ["HF_LEROBOT_HOME"] = "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot"
    os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
    os.environ.pop("LEROBOT_HOME", None)

    main()