#!/usr/bin/env python3
"""
LeRobot dataset augmentation script.

- Load an existing LeRobot dataset (Hub repo_id or local root)
- For each episode:
    - Augment instruction (optional) using aug_instruction.generate_similar_instructions
    - Augment video/image frames using aug_movie.edit_image_noise (or swap to QwenImageEditor)
    - Write as a new LeRobot dataset (v3 compatible through LeRobotDataset API)

Notes:
- We DO NOT pass 'episode_index', 'task_index', 'index', 'frame_index' to add_frame().
  LeRobotDataset.add_frame() manages frame_index internally and episode_index is stored in episode_buffer.
- Always call dataset.finalize() at the end (required in v3 docs).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from aug_instruction import generate_similar_instructions

# ---- your augmentation modules ----
# place this script so that these imports work (e.g., same folder or PYTHONPATH)
from aug_movie import edit_image_noise  # lightweight augmentation


def _to_hwc_uint8(img: Any) -> np.ndarray:
    """
    Convert various image representations to HWC uint8 numpy array.
    Expected inputs in LeRobot sample dict are often torch.Tensor or np.ndarray.
    """
    # torch is optional here; we avoid importing torch at module level
    if "torch" in sys.modules:
        import torch  # type: ignore

        if isinstance(img, torch.Tensor):
            x = img.detach().cpu().numpy()
        else:
            x = img
    else:
        x = img

    x = np.asarray(x)

    # Common cases:
    # - CHW uint8 (C,H,W) -> HWC
    # - HWC uint8 already
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[2] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))

    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)

    # If grayscale (H,W) -> (H,W,1)
    if x.ndim == 2:
        x = x[..., None]

    return x


def _collect_episode_ranges(src_ds) -> Iterable[tuple[int, int, int]]:
    """
    Return iterable of (episode_index, start_global_idx, end_global_idx_exclusive).

    Prefer src_ds.episode_data_index["from"/"to"] if present.
    Fallback: use src_ds.meta.episodes if shaped similarly.

    This mirrors typical usage in LeRobotDataset internals and downstream projects.
    """
    if hasattr(src_ds, "episode_data_index") and src_ds.episode_data_index is not None:
        epi = src_ds.episode_data_index
        # epi["from"][ep] / epi["to"][ep] are often tensors; convert to int
        for ep in range(src_ds.num_episodes):
            start = int(epi["from"][ep])
            end = int(epi["to"][ep])
            yield ep, start, end
        return

    # Fallback: attempt src_ds.meta.episodes structure
    if hasattr(src_ds, "meta") and hasattr(src_ds.meta, "episodes") and src_ds.meta.episodes is not None:
        episodes = src_ds.meta.episodes
        # try dict-like access
        if isinstance(episodes, dict) and "from" in episodes and "to" in episodes:
            for ep in range(len(episodes["from"])):
                yield ep, int(episodes["from"][ep]), int(episodes["to"][ep])
            return

    raise RuntimeError(
        "Could not determine episode ranges. "
        "Your lerobot version may differ; please inspect src_ds.episode_data_index or src_ds.meta.episodes."
    )


def _resolve_local_dataset_dir(src_repo_id: str, src_root: Path | None) -> Path | None:
    """Resolve the on-disk dataset directory.

    Different lerobot versions interpret `root` differently:
      - Some expect root == dataset directory that contains meta/, data/, videos/
      - Others accept a parent root and derive dataset_dir from repo_id

    We support both by probing the filesystem.
    """
    if src_root is None:
        return None

    src_root = Path(src_root)

    # Case A: src_root itself is the dataset directory
    if (src_root / "meta" / "info.json").exists():
        return src_root

    # Case B: src_root is a parent; append <org>/<name>
    candidate = src_root / src_repo_id
    if (candidate / "meta" / "info.json").exists():
        return candidate

    # Case C: src_root is HF_LEROBOT_HOME-like and datasets live under <root>/lerobot/<org>/<name>
    candidate2 = src_root / "lerobot" / src_repo_id
    if (candidate2 / "meta" / "info.json").exists():
        return candidate2

    return None


def _build_dst_create_kwargs(src_ds, dst_repo_id: str, dst_root: Path) -> dict[str, Any]:
    """
    Prepare args for LeRobotDataset.create(...) by borrowing metadata from the source dataset.
    """
    # fps
    fps = int(getattr(src_ds, "fps", src_ds.meta.info.get("fps")))

    # robot_type and features: most robust is meta.info
    robot_type = None
    features = None
    use_videos = True

    if hasattr(src_ds, "meta") and hasattr(src_ds.meta, "info") and src_ds.meta.info is not None:
        info = src_ds.meta.info
        robot_type = info.get("robot_type") or info.get("robot") or info.get("robotType")
        features = info.get("features")
        # if no explicit features in info, fallback to src_ds.features
        if features is None and hasattr(src_ds, "features"):
            features = dict(src_ds.features)

        # v3 datasets are typically video-backed (mp4); keep it on unless you know otherwise
        use_videos = bool(info.get("video", True)) if isinstance(info, dict) else True

    if features is None and hasattr(src_ds, "features"):
        features = dict(src_ds.features)

    if features is None:
        raise RuntimeError("Could not infer dataset features from source dataset.")

    return {
        "repo_id": dst_repo_id,
        "fps": fps,
        "root": dst_root,
        "robot_type": robot_type,
        "features": features,
        "use_videos": use_videos,
        # speed knobs; safe defaults
        "image_writer_processes": 0,
        "image_writer_threads": 4,
    }


def augment_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    *,
    src_root: Path | None,
    dst_root: Path,
    noise_level: float,
    num_aug_per_episode: int,
    include_original: bool,
    augment_instruction: bool,
    instruction_temperature: float,
    instruction_max_tokens: int,
) -> None:
    # Load src dataset (local-first)
    dataset_dir = _resolve_local_dataset_dir(src_repo_id, src_root)
    import pdb

    pdb.set_trace()
    if dataset_dir is None:
        # Fall back to hub behavior only if user didn't provide src_root, otherwise error loudly.
        if src_root is None:
            src_ds = LeRobotDataset(src_repo_id)
        else:
            raise FileNotFoundError(
                "Could not find local dataset metadata (meta/info.json). Tried:\n"
                f"  - {Path(src_root) / 'meta' / 'info.json'}\n"
                f"  - {Path(src_root) / src_repo_id / 'meta' / 'info.json'}\n"
                f"  - {Path(src_root) / 'lerobot' / src_repo_id / 'meta' / 'info.json'}\n"
                "Fix: pass --src-root as the dataset directory itself, e.g. .../<org>/<name>, or as the parent that contains <org>/<name>."
            )
    else:
        # Many lerobot versions expect root == dataset directory
        src_ds = LeRobotDataset(src_repo_id)

    # Create destination dataset
    create_kwargs = _build_dst_create_kwargs(src_ds, dst_repo_id, dst_root)
    dst_ds = LeRobotDataset.create(**create_kwargs)

    # Identify image/video keys to augment
    # LeRobot uses features with dtype "image" or "video" for visual modalities
    image_keys = []
    for k, ft in dst_ds.features.items():
        if isinstance(ft, dict) and ft.get("dtype") in ("image", "video"):
            image_keys.append(k)

    # Iterate episodes
    for ep, start, end in _collect_episode_ranges(src_ds):
        # Fetch the canonical (original) instruction/task for this episode
        # Typical: dataset[i]["task"] exists when loading (LeRobotDataset.__getitem__ adds it)
        original_task = ""
        try:
            original_task = src_ds[start].get("task", "")
        except Exception:
            original_task = ""

        # Build list of variants for this episode:
        #   - optionally include original (no aug)
        #   - plus N augmented versions
        variants = []
        if include_original:
            variants.append(("orig", original_task))

        for a in range(num_aug_per_episode):
            if augment_instruction and original_task:
                similar = generate_similar_instructions(
                    original_task,
                    temperature=instruction_temperature,
                    max_tokens=instruction_max_tokens,
                )
                task = similar.strip() if isinstance(similar, str) else ""
                if not task:
                    task = original_task  # fallback
            else:
                task = original_task
            variants.append((f"aug{a:02d}", task))

        # For each variant, write a new episode
        for tag, task_text in variants:
            for i in range(start, end):
                item = src_ds[i]  # dict of tensors/arrays

                frame: dict[str, Any] = {}

                # Task: stored as special key "task" (not in features)
                if task_text:
                    frame["task"] = task_text

                # Preserve timestamp if present (optional)
                if "timestamp" in item:
                    frame["timestamp"] = item["timestamp"]

                # Copy all feature keys EXCEPT the ones add_frame/save_episode manage internally
                # - do not pass: index, episode_index, task_index, frame_index
                skip_keys = {"index", "episode_index", "task_index", "frame_index"}

                for k, v in item.items():
                    if k in skip_keys:
                        continue
                    if k == "task":
                        continue  # we already set it
                    if k not in dst_ds.features:
                        # ignore non-feature keys that may exist in item
                        continue

                    # Augment visual keys
                    if k in image_keys and tag != "orig":
                        hwc = _to_hwc_uint8(v)
                        aug_hwc = edit_image_noise(hwc, noise_level=noise_level)
                        frame[k] = aug_hwc  # HWC uint8
                    else:
                        frame[k] = v

                dst_ds.add_frame(frame)

            dst_ds.save_episode()
            print(f"[OK] wrote episode (src_ep={ep}, variant={tag}) frames={end - start}")

    # Finalize (required for v3 per docs)
    if hasattr(dst_ds, "finalize") and callable(dst_ds.finalize):
        dst_ds.finalize()
    print(f"Done. New dataset written to: {dst_root / dst_repo_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-repo-id", required=True, help="Source dataset repo_id (e.g. org/name)")
    p.add_argument("--dst-repo-id", required=True, help="Destination dataset repo_id (new name)")

    p.add_argument("--src-root", default=None, help="Optional local root for source dataset")
    p.add_argument("--dst-root", required=True, help="Root directory where new dataset is created")

    p.add_argument("--noise-level", type=float, default=25.0, help="Gaussian noise stddev for edit_image_noise()")
    p.add_argument("--num-aug-per-episode", type=int, default=1, help="How many augmented variants per episode")
    p.add_argument("--include-original", action="store_true", help="Also copy original episodes (no augmentation)")

    p.add_argument(
        "--augment-instruction", action="store_true", help="Use vLLM to generate similar task text per episode"
    )
    p.add_argument("--instruction-temperature", type=float, default=0.8)
    p.add_argument("--instruction-max-tokens", type=int, default=256)
    p.add_argument("--offline", action="store_true", help="Disable any Hugging Face Hub access (local-only).")

    args = p.parse_args()

    if args.offline:
        os.environ["HF_LEROBOT_HOME"] = "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot"
        os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
        os.environ.pop("LEROBOT_HOME", None)

    src_root = Path(args.src_root) if args.src_root else None
    dst_root = Path(args.dst_root)

    src_repo_id = Path(args.src_repo_id)
    dst_repo_id = Path(args.dst_repo_id)

    augment_dataset(
        src_repo_id=str(src_repo_id),
        dst_repo_id=str(dst_repo_id),
        src_root=src_root,
        dst_root=dst_root,
        noise_level=args.noise_level,
        num_aug_per_episode=args.num_aug_per_episode,
        include_original=args.include_original,
        augment_instruction=args.augment_instruction,
        instruction_temperature=args.instruction_temperature,
        instruction_max_tokens=args.instruction_max_tokens,
    )


if __name__ == "__main__":
    main()
