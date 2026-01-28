from __future__ import annotations

import argparse
import os
import logging
import torch
import time
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions

import requests
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


SEGMENT_TASKS = "shelf, object, pet bottle, container, box"
# SEGMENT_TASKS = "oven, bread, dish, table, plate"


def build_prompt(task: str) -> tuple[str, str]:
    prompt = (
        "Make minimal and subtle changes to only small, task-irrelevant regions of the background. "
        "Keep most of the original background unchanged. "
        "Preserve the same indoor setting, materials, scene context, objects, and overall atmosphere. "
        f"Task: {task}"
    )

    negative_prompt = " "
    return prompt, negative_prompt


class QwenImageEditClient:
    """Client for Qwen-Image-Edit API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 300,
        pool_size: int = 250,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # Increase connection pool size
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @staticmethod
    def tensor_to_base64(tensor: torch.Tensor) -> str:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def base64_to_tensor(b64_str: str) -> torch.Tensor:
        buffer = io.BytesIO(base64.b64decode(b64_str))
        return torch.load(buffer, weights_only=True)

    def edit_image(
        self,
        image_tensor: torch.Tensor,
        task: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """Edit single image. Input/Output: CHW tensor [0,1]."""

        payload = {
            "image_tensor_b64": self.tensor_to_base64(image_tensor),
            "prompt": prompt,
            "task": SEGMENT_TASKS,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        resp = self.session.post(
            f"{self.base_url}/edit", json=payload, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API error: {resp.status_code} - {resp.text}")

        result = resp.json()
        return self.base64_to_tensor(result["image_tensor_b64"])

    def health_check(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200 and resp.json().get("ready", False)
        except requests.RequestException:
            return False


# Global client instance
_api_client: Optional[QwenImageEditClient] = None


def get_api_client(base_url: str = "http://localhost:8000") -> QwenImageEditClient:
    """Get or create API client singleton."""
    global _api_client
    if _api_client is None:
        _api_client = QwenImageEditClient(base_url)
        if not _api_client.health_check():
            raise RuntimeError(f"API server not available at {base_url}")
        logger.info(f"Connected to API server at {base_url}")
    return _api_client


def edit_image_with_api(
    frame: torch.Tensor,
    task: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    入力:
      frame: torch.Tensor (C, H, W), 値域 [0, 1]
      prompt: 編集指示

    処理:
      - APIを呼び出して画像を編集

    出力:
      torch.Tensor (C, H, W), 値域 [0, 1]
    """
    client = get_api_client()
    # API call (CHW -> CHW)
    edited_chw = client.edit_image(
        image_tensor=frame,
        task=task,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )
    return edited_chw


def create_image_transform_with_api(task: str, seed: int = 42):
    """Create image transform function with API call."""
    prompt, negative_prompt = build_prompt(task)

    def transform(frame: torch.Tensor) -> torch.Tensor:
        return edit_image_with_api(frame, task, prompt, negative_prompt, seed)

    return transform


def process_frame(
    frame: dict,
    task_transform: Optional[Callable] = None,
    image_transform: Optional[Callable] = None,
) -> dict:
    """Process frame: remove skip keys and apply transforms."""
    SKIP_KEYS = {"index", "episode_index", "timestamp", "frame_index", "task_index"}

    if task_transform is None:

        def task_transform(x):
            return x

    if image_transform is None:

        def image_transform(x):
            return x.permute(1, 2, 0)

    new_frame = {}
    for key, value in frame.items():
        if key in SKIP_KEYS:
            continue

        if "task" in key:
            value = task_transform(value)
        elif key == "observation.image.hand":
            # 左に90度回転（反時計回り）
            # (C, H, W) -> (C, W, H)
            roted_cwh = torch.rot90(value, k=1, dims=(1, 2))
            # Edit Image
            edited_cwh = image_transform(roted_cwh)
            # (C, W, H) -> (C, H, W)
            edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))

            value = edited_chw.permute(1, 2, 0)  # CHW -> HWC

        elif "observation.image" in key:
            edited_chw = image_transform(value)
            value = edited_chw.permute(1, 2, 0)  # CHW -> HWC

        new_frame[key] = value
    return new_frame


def process_frames_batch_parallel(
    frames: list[dict],
    task: str,
    max_workers: int = 4,
    seed_base: int = 42,
) -> list[dict]:
    """
    複数フレームを並列でAPI処理する。

    Args:
        frames: フレームのリスト
        task: タスク名
        max_workers: 並列ワーカー数
        seed_base: シードのベース値

    Returns:
        処理済みフレームのリスト（順序保証）
    """
    client = get_api_client()

    prompt, negative_prompt = build_prompt(task)

    results: list[dict] = [{} for _ in range(len(frames))]

    def process_single(args):
        idx, frame, seed = args
        # 画像キーを探してAPI処理
        new_frame = {}
        for key, value in frame.items():
            if key in {
                "index",
                "episode_index",
                "timestamp",
                "frame_index",
                "task_index",
            }:
                continue
            if "task" in key:
                new_frame[key] = task

            elif key == "observation.image.hand":
                # 左に90度回転（反時計回り）
                # (C, H, W) -> (C, W, H)
                roted_cwh = torch.rot90(value, k=1, dims=(1, 2))
                # Edit Image
                edited_cwh = client.edit_image(
                    roted_cwh, task, prompt, negative_prompt, seed
                )
                # (C, W, H) -> (C, H, W)
                edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))

                new_frame[key] = edited_chw.permute(1, 2, 0)  # CHW -> HWC

            elif "observation.image" in key:
                # CHW tensor -> API -> HWC tensor
                edited = client.edit_image(value, task, prompt, negative_prompt, seed)
                new_frame[key] = edited.permute(1, 2, 0)
            else:
                new_frame[key] = value
        return idx, new_frame

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, (i, frame, seed_base + i)): i
            for i, frame in enumerate(frames)
        }

        for future in tqdm(
            as_completed(futures), total=len(frames), desc="API Processing"
        ):
            idx, new_frame = future.result()
            results[idx] = new_frame

    return results


def augment_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    api_url: str = "http://localhost:8000",
    max_workers: int = 4,
    use_batch: bool = True,
) -> None:
    """Dataset augmentation using API."""
    # Initialize API client
    global _api_client
    _api_client = QwenImageEditClient(api_url)
    if not _api_client.health_check():
        raise RuntimeError(f"API server not available at {api_url}")

    # Load src dataset
    original_ds = LeRobotDataset(src_repo_id)

    # Create new dataset
    dst_ds = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=original_ds.meta.info["fps"],
        features=original_ds.meta.info["features"],
        robot_type=original_ds.meta.info["robot_type"],
        use_videos=True,
        image_writer_processes=56,
        image_writer_threads=2,
    )

    start = time.time()
    meta_episodes = original_ds.meta.episodes
    num_episodes = len(meta_episodes["dataset_from_index"])

    logger.info(f"Adding copy of {num_episodes} episodes using API at {api_url}")

    for ep_idx in tqdm(range(num_episodes), desc="Augment episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])

        if use_batch:
            # バッチ処理（並列API呼び出し）
            frames = [original_ds[idx] for idx in range(start_idx, end_idx)]
            processed_frames = process_frames_batch_parallel(
                frames, new_task, max_workers=max_workers, seed_base=ep_idx * 10000
            )
            for new_frame in tqdm(processed_frames, desc="Adding frames"):
                dst_ds.add_frame(new_frame)
        else:
            # 逐次処理
            image_transform = create_image_transform_with_api(new_task, seed=ep_idx)
            for idx in tqdm(range(start_idx, end_idx), desc="Processing frames"):
                frame = original_ds[idx]
                new_frame = process_frame(
                    frame,
                    task_transform=lambda x: new_task,
                    image_transform=image_transform,
                )
                dst_ds.add_frame(new_frame)

        dst_ds.save_episode()

    dst_ds.finalize()
    diff_time = time.time() - start
    logger.info(f"total_time: {diff_time:.2f}s")

    aug_ds = LeRobotDataset(dst_repo_id)
    merged = merge_datasets(
        [original_ds, aug_ds], output_repo_id=f"{dst_repo_id}_merged"
    )
    merged.finalize()

    logger.info(f"Done! Augmented Episodes: {num_episodes}, saved to {dst_repo_id}")
    logger.info(
        f"Done! Merged Episodes: {num_episodes} -> {2 * num_episodes}, saved to {dst_repo_id}_merged"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--dst-repo-id", required=True)
    p.add_argument("--api-url", default="http://localhost:11303", help="API server URL")
    p.add_argument(
        "--max-workers", type=int, default=4, help="Parallel workers for API calls"
    )
    p.add_argument("--use-batch", action="store_true", help="Disable batch processing")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    if args.offline:
        os.environ["HF_LEROBOT_HOME"] = (
            "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot"
        )
        os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
        os.environ.pop("LEROBOT_HOME", None)

    augment_dataset(
        args.src_repo_id,
        args.dst_repo_id,
        api_url=args.api_url,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
    )


if __name__ == "__main__":
    main()
