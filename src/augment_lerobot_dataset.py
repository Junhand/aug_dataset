from __future__ import annotations

import argparse
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from aug_instruction import generate_similar_instructions
from aug_movie import edit_image_noise

from tqdm import tqdm
from typing import Optional, Callable


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
    original_meta = LeRobotDatasetMetadata(src_repo_id)
    os.cpu_count()

    # Create new dataset with same features
    import pdb

    pdb.set_trace()
    dst_ds = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=original_meta.fps,
        features=original_meta.features.copy(),
        robot_type=original_meta.robot_type,
        use_videos=True,
        image_writer_processes=32,
        image_writer_threads=8,
    )

    # Copy all episodes
    meta_episodes = original_ds.meta.episodes
    num_episodes = len(meta_episodes["dataset_from_index"])
    for ep_idx in tqdm(range(2), desc="Copying episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]

        for idx in range(start_idx, end_idx):
            frame = original_ds[idx]
            new_frame = process_frame(frame)
            dst_ds.add_frame(new_frame)
        dst_ds.save_episode()

    # Add one more episode (copy of last)
    last_ep_idx = num_episodes - 1
    start_idx = meta_episodes["dataset_from_index"][last_ep_idx]
    end_idx = meta_episodes["dataset_to_index"][last_ep_idx]

    print(
        f"Adding copy of last episode (idx={last_ep_idx}, frames={end_idx - start_idx})"
    )
    i = 0
    new_task = ""
    for idx in tqdm(range(start_idx, end_idx), desc="Adding last episode copy"):
        frame = original_ds[idx]
        if i == 0:
            i += 1
            new_task = generate_similar_instructions(frame["task"])
            import pdb

            pdb.set_trace()

        def task_transform(x):
            return new_task

        new_frame = process_frame(
            frame, task_transform=task_transform, image_transform=edit_image_noise
        )
        dst_ds.add_frame(new_frame)
    dst_ds.save_episode()
    dst_ds.finalize()
    print(
        f"Done! Episodes: {num_episodes} -> {num_episodes + 1}, saved to {dst_repo_id}"
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
