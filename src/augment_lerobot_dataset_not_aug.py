from __future__ import annotations

import argparse
import os
import logging
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset

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

    # Create new dataset with same features
    dst_ds = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=original_ds.meta.info["fps"],
        features=original_ds.meta.info["features"],
        robot_type=original_ds.meta.info["robot_type"],
        use_videos=True,
        image_writer_processes=64,
        image_writer_threads=2,
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

        for i, idx in enumerate(range(start_idx, end_idx)):
            frame = original_ds[idx]
            new_frame = process_frame(frame)
            dst_ds.add_frame(new_frame)
        dst_ds.save_episode()

    dst_ds.finalize()
    diff_time = time.time() - start
    logger.info(f"total_time: {diff_time}")

    logger.info(f"Done! Augmeted Episodes: {num_episodes}, saved to {dst_repo_id}")


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
