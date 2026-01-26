from __future__ import annotations

import argparse
import os
import logging
import torch
import time
import multiprocessing as mp
import psutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions
from aug_movie import edit_image_gaussian_noise

from tqdm import tqdm
from typing import Optional, Callable

LOG_FORMAT = (
    "\n==================================================\n"
    "%(asctime)s - %(name)s - %(levelname)s\n"
    "%(message)s"
    "\n==================================================\n"
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _process_batch(args):
    """バッチ処理（GPU割り当て）"""
    worker_id, batch, new_task, num_gpus = args

    # GPU割り当て
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    results = []
    for idx, frame_data in batch:
        # GPU に移動
        if num_gpus > 0:
            frame_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in frame_data.items()
            }

        new_frame = process_frame(
            frame_data,
            task_transform=lambda x: new_task,
            image_transform=edit_image_gaussian_noise,
        )

        # CPU に戻す
        new_frame = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in new_frame.items()
        }
        results.append((idx, new_frame))

    return results


def parallel_process_frames(frames, new_task, num_workers=None):
    """並列フレーム処理（順序保証・高速版）"""
    num_gpus = torch.cuda.device_count()
    if num_workers is None:
        num_workers = num_gpus or 1

    total = len(frames)
    print(f"Using {num_workers} workers, {num_gpus} GPUs, processing {total} frames")

    # バッチに分割
    batch_size = max(1, total // num_workers)
    batches = []
    for i in range(num_workers):
        start = i * batch_size
        end = start + batch_size if i < num_workers - 1 else total
        if start < total:
            batches.append((i, frames[start:end], new_task, num_gpus))

    # 並列処理
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        batch_results = list(
            tqdm(
                pool.imap_unordered(_process_batch, batches),
                total=len(batches),
                desc="Processing",
            )
        )

    # 結果をフラット化してソート
    results = [item for batch in batch_results for item in batch]
    return sorted(results, key=lambda x: x[0])


def process_frame(
    frame: dict,
    task_transform: Optional[Callable] = None,
    image_transform: Optional[Callable] = None,
) -> dict:
    """Process frame: remove skip keys and apply transforms.

    Args:
        frame: Original frame dict from LeRobotDataset
        task_transform: Function to transform task value (default: identity)
        image_transform: Function to transform image value (default: permute to HWC)

    Returns:
        Processed frame dict ready for add_frame()
    """
    # metadataを指定
    SKIP_KEYS = {"index", "episode_index", "timestamp", "frame_index", "task_index"}

    # Default transforms
    if task_transform is None:

        def task_transform(x):
            return x

    if image_transform is None:

        def image_transform(x):
            return x.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

    new_frame = {}
    for key, value in frame.items():
        # Skip metadata keys managed by LeRobotDataset
        if key in SKIP_KEYS:
            continue

        # Apply transforms based on key type
        if "task" in key:
            value = task_transform(value)
        elif "observation.image" in key:
            value = image_transform(value)

        new_frame[key] = value
    return new_frame


def augment_dataset(
    src_repo_id: str,
    dst_repo_id: str,
) -> None:
    # Load src dataset
    original_ds = LeRobotDataset(src_repo_id)

    # システム情報取得
    total_gpus = 8
    num_gpus_used = torch.cuda.device_count() or 1
    cpu_count = psutil.cpu_count(logical=False)

    # GPU使用率に応じてリソースを按分
    gpu_ratio = num_gpus_used / total_gpus
    available_cpus = int(cpu_count * gpu_ratio)

    import pdb

    pdb.set_trace()
    # Create new dataset with same features
    dst_ds = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=original_ds.meta.info["fps"],
        features=original_ds.meta.info["features"],
        robot_type=original_ds.meta.info["robot_type"],
        use_videos=True,
        image_writer_processes=available_cpus,
        image_writer_threads=4,
    )
    start = time.time()
    # Copy all episodes
    meta_episodes = original_ds.meta.episodes
    num_episodes = len(meta_episodes["dataset_from_index"])

    # Add more episodes (copy of last)
    logger.info(f"Adding copy of {num_episodes} episodes")
    for ep_idx in tqdm(range(num_episodes), desc="Augment episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])

        # frames = []
        # for idx in tqdm(range(start_idx, end_idx), desc="Loading"):
        #     frame = original_ds[idx]
        #     frames.append((idx, frame))

        # results = parallel_process_frames(frames, new_task)

        # for idx, new_frame in tqdm(results, desc="Adding"):
        #     dst_ds.add_frame(new_frame)
        for i, idx in enumerate(range(start_idx, end_idx)):
            frame = original_ds[idx]
            new_frame = process_frame(
                frame,
                task_transform=lambda x: new_task,
                image_transform=edit_image_gaussian_noise,
            )
            dst_ds.add_frame(new_frame)
        dst_ds.save_episode()

    dst_ds.finalize()
    diff_time = time.time() - start
    logger.info(f"total_time: {diff_time}")
    aug_ds = LeRobotDataset(dst_repo_id)
    merged = merge_datasets(
        [original_ds, aug_ds], output_repo_id=f"{dst_repo_id}_merged"
    )
    merged.finalize()
    logger.info(f"Done! Augmeted Episodes: {num_episodes}, saved to {dst_repo_id}")
    logger.info(
        f"Done! Merged Episodes: {num_episodes} -> {2 * num_episodes}, saved to {dst_repo_id}_merged"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--dst-repo-id", required=True)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    if args.offline:
        os.environ["HF_LEROBOT_HOME"] = (
            "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot"
        )
        os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
        os.environ.pop("LEROBOT_HOME", None)

    augment_dataset(args.src_repo_id, args.dst_repo_id)


if __name__ == "__main__":
    main()
