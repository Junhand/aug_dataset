from __future__ import annotations

import argparse
import os
import logging
import torch
import time
import io
import base64
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions

import requests
from tqdm import tqdm
from typing import Optional, Callable, List

LOG_FORMAT = (
    "\n==================================================\n"
    "%(asctime)s - %(name)s - %(levelname)s\n"
    "%(message)s"
    "\n==================================================\n"
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


SEGMENT_TASKS = "shelf, object, pet bottle, container, box"
# SEGMENT_TASKS = "box, oven, microwave oven, object, food, dish, table, plate"


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
    """Client for Qwen-Image-Edit API with multi-node round-robin load balancing (lock-free)."""

    def __init__(
        self,
        base_urls: List[str] | str = "http://localhost:8000",
        timeout: int = 300,
        pool_size: int = 250,
    ):
        """
        Args:
            base_urls: 単一のURL文字列、またはURLのリスト
                       例: "http://localhost:8000"
                       例: ["http://node1:11303", "http://node2:11303", "http://node3:11303"]
            timeout: リクエストタイムアウト（秒）
            pool_size: 各ノードへの接続プールサイズ
        """
        # URLリストの正規化
        if isinstance(base_urls, str):
            self.base_urls = [base_urls.rstrip("/")]
        else:
            self.base_urls = [url.rstrip("/") for url in base_urls]

        self.num_nodes = len(self.base_urls)
        self.timeout = timeout

        # ロックフリーのラウンドロビン（itertools.cycleはスレッドセーフ）
        self._node_cycle = itertools.cycle(range(self.num_nodes))

        # 各ノード用のセッションを作成
        self.sessions: List[requests.Session] = []
        for _ in self.base_urls:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self.sessions.append(session)

        logger.info(
            f"Initialized client with {self.num_nodes} node(s): {self.base_urls}"
        )

    def _get_next_node_idx(self) -> int:
        """ロックフリーでノードインデックスを取得"""
        return next(self._node_cycle)

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
        node_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Edit single image.

        Args:
            image_tensor: Input CHW tensor [0,1]
            task: Task description
            prompt: Edit prompt
            negative_prompt: Negative prompt
            seed: Random seed
            node_idx: 指定されたノードを使用（Noneの場合はラウンドロビン）

        Returns:
            Edited CHW tensor [0,1]
        """
        payload = {
            "image_tensor_b64": self.tensor_to_base64(image_tensor),
            "prompt": prompt,
            "task": SEGMENT_TASKS,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        # ノード選択
        if node_idx is None:
            node_idx = self._get_next_node_idx()

        base_url = self.base_urls[node_idx]
        session = self.sessions[node_idx]

        try:
            resp = session.post(f"{base_url}/edit", json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"API error: {resp.status_code} - {resp.text}")

            result = resp.json()
            return self.base64_to_tensor(result["image_tensor_b64"])

        except Exception as e:
            logger.warning(f"Node {node_idx} ({base_url}) failed: {e}")
            raise

    def edit_image_with_fallback(
        self,
        image_tensor: torch.Tensor,
        task: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Edit single image with fallback to other nodes on failure.
        """
        start_node = self._get_next_node_idx()

        for i in range(self.num_nodes):
            node_idx = (start_node + i) % self.num_nodes
            try:
                return self.edit_image(
                    image_tensor=image_tensor,
                    task=task,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    node_idx=node_idx,
                )
            except Exception as e:
                if i == self.num_nodes - 1:
                    raise RuntimeError(f"All nodes failed. Last error: {e}")
                continue

        raise RuntimeError("All nodes failed")

    def health_check(self) -> dict:
        """
        Check health of all nodes.

        Returns:
            Dict with node status: {"healthy_nodes": [...], "unhealthy_nodes": [...]}
        """
        healthy = []
        unhealthy = []

        for idx, (base_url, session) in enumerate(zip(self.base_urls, self.sessions)):
            try:
                resp = session.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200 and resp.json().get("ready", False):
                    healthy.append(
                        {"index": idx, "url": base_url, "status": resp.json()}
                    )
                else:
                    unhealthy.append(
                        {"index": idx, "url": base_url, "error": "Not ready"}
                    )
            except requests.RequestException as e:
                unhealthy.append({"index": idx, "url": base_url, "error": str(e)})

        return {"healthy_nodes": healthy, "unhealthy_nodes": unhealthy}

    def is_ready(self) -> bool:
        """Check if at least one node is ready."""
        health = self.health_check()
        return len(health["healthy_nodes"]) > 0


# Global client instance
_api_client: Optional[QwenImageEditClient] = None


def get_api_client(
    base_urls: List[str] | str = "http://localhost:8000",
) -> QwenImageEditClient:
    """Get or create API client singleton."""
    global _api_client
    if _api_client is None:
        _api_client = QwenImageEditClient(base_urls)
        health = _api_client.health_check()
        if not health["healthy_nodes"]:
            raise RuntimeError(
                f"No healthy API servers available. Unhealthy: {health['unhealthy_nodes']}"
            )
        logger.info(f"Connected to {len(health['healthy_nodes'])} healthy node(s)")
        for node in health["healthy_nodes"]:
            logger.info(
                f"  - Node {node['index']}: {node['url']} (GPUs: {node['status'].get('num_gpus', 'N/A')})"
            )
    return _api_client


def create_image_transform_with_api(task: str, seed: int = 42):
    """Create image transform function with API call."""
    prompt, negative_prompt = build_prompt(task)

    def transform(frame: torch.Tensor) -> torch.Tensor:
        client = get_api_client()
        return client.edit_image(frame, task, prompt, negative_prompt, seed)

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
            roted_cwh = torch.rot90(value, k=1, dims=(1, 2))
            edited_cwh = image_transform(roted_cwh)
            edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))
            value = edited_chw.permute(1, 2, 0)
        elif "observation.image" in key:
            edited_chw = image_transform(value)
            value = edited_chw.permute(1, 2, 0)

        new_frame[key] = value
    return new_frame


def process_frames_batch_parallel(
    frames: list[dict],
    task: str,
    max_workers: int = 4,
    seed_base: int = 42,
) -> list[dict]:
    """
    複数フレームを並列でAPI処理する（複数ノードにラウンドロビン分散、ロックフリー）。
    """
    client = get_api_client()
    prompt, negative_prompt = build_prompt(task)
    results: list[dict] = [{} for _ in range(len(frames))]

    # 事前にノード割り当てを決定（ロックフリー）
    num_nodes = client.num_nodes

    def process_single(args):
        idx, frame, seed, assigned_node = args
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
                roted_cwh = torch.rot90(value, k=1, dims=(1, 2))
                # 事前に割り当てられたノードを使用
                edited_cwh = client.edit_image(
                    roted_cwh,
                    task,
                    prompt,
                    negative_prompt,
                    seed,
                    node_idx=assigned_node,
                )
                edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))
                new_frame[key] = edited_chw.permute(1, 2, 0)

            elif "observation.image" in key:
                # 事前に割り当てられたノードを使用
                edited = client.edit_image(
                    value, task, prompt, negative_prompt, seed, node_idx=assigned_node
                )
                new_frame[key] = edited.permute(1, 2, 0)
            else:
                new_frame[key] = value

        return idx, new_frame

    # タスクを事前に作成し、ノードを均等に割り当て
    tasks = [(i, frame, seed_base + i, i % num_nodes) for i, frame in enumerate(frames)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, task_args): task_args[0]
            for task_args in tasks
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
    api_urls: List[str] | str = "http://localhost:8000",
    max_workers: int = 4,
    use_batch: bool = True,
) -> None:
    """Dataset augmentation using API with multi-node support."""
    global _api_client
    _api_client = QwenImageEditClient(api_urls)
    health = _api_client.health_check()
    if not health["healthy_nodes"]:
        raise RuntimeError(
            f"No healthy API servers available. Unhealthy: {health['unhealthy_nodes']}"
        )

    logger.info(f"Using {len(health['healthy_nodes'])} healthy node(s) for processing")

    original_ds = LeRobotDataset(src_repo_id)

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

    logger.info(
        f"Adding copy of {num_episodes} episodes using {len(health['healthy_nodes'])} API node(s)"
    )

    for ep_idx in tqdm(range(num_episodes), desc="Augment episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])

        if use_batch:
            frames = [original_ds[idx] for idx in range(start_idx, end_idx)]
            processed_frames = process_frames_batch_parallel(
                frames, new_task, max_workers=max_workers, seed_base=ep_idx * 10000
            )
            for new_frame in tqdm(processed_frames, desc="Adding frames"):
                dst_ds.add_frame(new_frame)
        else:
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


def parse_api_urls(url_string: str) -> List[str]:
    """Parse comma-separated URL string into list."""
    return [url.strip() for url in url_string.split(",") if url.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--dst-repo-id", required=True)
    p.add_argument(
        "--api-urls",
        default="http://localhost:11303",
        help="API server URL(s), comma-separated for multiple nodes. "
        "Example: 'http://node1:11303,http://node2:11303,http://node3:11303'",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Parallel workers for API calls (recommend: 8 * num_nodes)",
    )
    p.add_argument("--use-batch", action="store_true", help="Enable batch processing")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    if args.offline:
        os.environ["HF_LEROBOT_HOME"] = (
            "/home/group_25b505/group_5/.cache/huggingface/lerobot/lerobot"
        )
        os.environ["HF_HOME"] = "/home/group_25b505/group_5/.cache/huggingface"
        os.environ.pop("LEROBOT_HOME", None)

    api_urls = parse_api_urls(args.api_urls)
    logger.info(f"Using API endpoints: {api_urls}")

    augment_dataset(
        args.src_repo_id,
        args.dst_repo_id,
        api_urls=api_urls,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
    )


if __name__ == "__main__":
    main()
