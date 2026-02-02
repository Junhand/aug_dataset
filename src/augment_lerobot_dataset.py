"""
Dataset Augmentation Client with Multi-Node API Support and TorchVision Augmentation
- 1 frame -> 10 frames (1 Qwen edit + 9 torchvision augmentations)
"""

from __future__ import annotations

import argparse
import os
import logging
import torch
import time
import io
import base64
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions

import requests
from tqdm import tqdm
from torchvision.transforms import v2

LOG_FORMAT = (
    "\n==================================================\n"
    "%(asctime)s - %(name)s - %(levelname)s\n"
    "%(message)s"
    "\n==================================================\n"
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


SEGMENT_TASKS = "shelf, object, pet bottle, container, box"  # task05
# SEGMENT_TASKS = "box, oven, microwave oven, object, food, dish, table, plate" #task48

# Augmentation multiplier: 1 original -> N_AUGMENT frames
N_AUGMENT = 10


def build_prompt(task: str) -> tuple[str, str]:
    prompt = (
        "Make minimal and subtle changes to only small, task-irrelevant regions of the background. "
        "Keep most of the original background unchanged. "
        "Preserve the same indoor setting, materials, scene context, objects, and overall atmosphere. "
        f"Task: {task}"
    )
    negative_prompt = " "
    return prompt, negative_prompt


@dataclass
class AugmentationConfig:
    """Configuration for torchvision augmentations."""

    # Color augmentation ranges
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_range: Tuple[float, float] = (-0.1, 0.1)

    # Sharpness
    sharpness_range: Tuple[float, float] = (0.5, 2.0)

    # Gaussian blur
    blur_kernel_size: int = 5
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0)

    # Gaussian noise
    noise_std_range: Tuple[float, float] = (0.01, 0.05)


class TorchVisionAugmentor:
    """
    TorchVision-based image augmentation for robotics datasets.
    Generates multiple augmented versions of a single frame.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

        # Define individual transforms
        self.transforms = {
            "color_jitter_light": v2.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            "color_jitter_medium": v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            "color_jitter_strong": v2.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            "brightness_up": v2.ColorJitter(brightness=(1.1, 1.3)),
            "brightness_down": v2.ColorJitter(brightness=(0.7, 0.9)),
            "contrast_up": v2.ColorJitter(contrast=(1.1, 1.3)),
            "contrast_down": v2.ColorJitter(contrast=(0.7, 0.9)),
            "gaussian_blur": v2.GaussianBlur(
                kernel_size=self.config.blur_kernel_size,
                sigma=self.config.blur_sigma_range,
            ),
            "sharpness": v2.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
        }

        # Predefined augmentation combinations for deterministic behavior
        self.augmentation_presets = [
            # Preset 0: Light color jitter
            ["color_jitter_light"],
            # Preset 1: Medium color jitter
            ["color_jitter_medium"],
            # Preset 2: Strong color jitter
            ["color_jitter_strong"],
            # Preset 3: Brightness up + slight blur
            ["brightness_up", "gaussian_blur"],
            # Preset 4: Brightness down + sharpness
            ["brightness_down", "sharpness"],
            # Preset 5: Contrast up
            ["contrast_up"],
            # Preset 6: Contrast down + color jitter light
            ["contrast_down", "color_jitter_light"],
            # Preset 7: Gaussian blur only
            ["gaussian_blur"],
            # Preset 8: Sharpness + color jitter light
            ["sharpness", "color_jitter_light"],
        ]

    def add_gaussian_noise(self, tensor: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def apply_preset(
        self, tensor: torch.Tensor, preset_idx: int, seed: int
    ) -> torch.Tensor:
        """
        Apply a specific augmentation preset to a tensor.

        Args:
            tensor: CHW tensor [0, 1]
            preset_idx: Index of the preset (0-8 for 9 augmentations)
            seed: Random seed for reproducibility

        Returns:
            Augmented CHW tensor [0, 1]
        """
        torch.manual_seed(seed)
        random.seed(seed)

        if preset_idx >= len(self.augmentation_presets):
            # Fallback: random combination
            preset_idx = preset_idx % len(self.augmentation_presets)

        preset = self.augmentation_presets[preset_idx]
        result = tensor.clone()

        for transform_name in preset:
            if transform_name in self.transforms:
                result = self.transforms[transform_name](result)

        # Add small noise to some presets for variety
        if preset_idx in [1, 3, 5, 7]:
            noise_std = random.uniform(0.01, 0.03)
            result = self.add_gaussian_noise(result, noise_std)

        return result

    def generate_augmentations(
        self, tensor: torch.Tensor, n_augment: int = 9, base_seed: int = 42
    ) -> List[torch.Tensor]:
        """
        Generate multiple augmented versions of a single frame.

        Args:
            tensor: CHW tensor [0, 1]
            n_augment: Number of augmentations to generate
            base_seed: Base seed for reproducibility

        Returns:
            List of augmented CHW tensors [0, 1]
        """
        augmented = []
        for i in range(n_augment):
            aug_tensor = self.apply_preset(tensor, i, base_seed + i * 1000)
            augmented.append(aug_tensor)
        return augmented


class QwenImageEditClient:
    """Client for Qwen-Image-Edit API with multi-node round-robin load balancing."""

    def __init__(
        self,
        base_urls: List[str] | str = "http://localhost:8000",
        timeout: int = 300,
        pool_size: int = 250,
    ):
        if isinstance(base_urls, str):
            self.base_urls = [base_urls.rstrip("/")]
        else:
            self.base_urls = [url.rstrip("/") for url in base_urls]

        self.num_nodes = len(self.base_urls)
        self.timeout = timeout
        self._node_cycle = itertools.cycle(range(self.num_nodes))

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
        """Edit single image using Qwen API."""
        payload = {
            "image_tensor_b64": self.tensor_to_base64(image_tensor),
            "prompt": prompt,
            "task": SEGMENT_TASKS,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

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

    def health_check(self) -> dict:
        """Check health of all nodes."""
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


# Global instances
_api_client: Optional[QwenImageEditClient] = None
_augmentor: Optional[TorchVisionAugmentor] = None


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


def get_augmentor() -> TorchVisionAugmentor:
    """Get or create augmentor singleton."""
    global _augmentor
    if _augmentor is None:
        _augmentor = TorchVisionAugmentor()
    return _augmentor


def process_single_frame_with_augmentation(
    frame: dict,
    task: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    node_idx: int,
    augment_idx: int,  # 0 = Qwen edit, 1-9 = torchvision augmentation
) -> dict:
    """
    Process a single frame with either Qwen edit or torchvision augmentation.

    Args:
        frame: Original frame dict
        task: Task description
        prompt: Qwen edit prompt
        negative_prompt: Qwen negative prompt
        seed: Random seed
        node_idx: API node index for Qwen
        augment_idx: 0 for Qwen edit, 1-9 for torchvision augmentations

    Returns:
        Processed frame dict
    """
    SKIP_KEYS = {"index", "episode_index", "timestamp", "frame_index", "task_index"}

    client = get_api_client()
    augmentor = get_augmentor()

    new_frame = {}

    for key, value in frame.items():
        if key in SKIP_KEYS:
            continue

        if "task" in key:
            new_frame[key] = task

        elif key == "observation.image.hand":
            roted_cwh = torch.rot90(value, k=1, dims=(1, 2))

            if augment_idx == 0:
                # Qwen edit
                edited_cwh = client.edit_image(
                    roted_cwh, task, prompt, negative_prompt, seed, node_idx=node_idx
                )
            else:
                # TorchVision augmentation (augment_idx 1-9 -> preset 0-8)
                edited_cwh = augmentor.apply_preset(
                    roted_cwh, augment_idx - 1, seed + augment_idx * 1000
                )

            edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))
            new_frame[key] = edited_chw.permute(1, 2, 0)

        elif "observation.image" in key:
            if augment_idx == 0:
                # Qwen edit
                edited = client.edit_image(
                    value, task, prompt, negative_prompt, seed, node_idx=node_idx
                )
            else:
                # TorchVision augmentation
                edited = augmentor.apply_preset(
                    value, augment_idx - 1, seed + augment_idx * 1000
                )

            new_frame[key] = edited.permute(1, 2, 0)
        else:
            new_frame[key] = value

    return new_frame


def process_frames_batch_parallel_with_augmentation(
    frames: list[dict],
    task: str,
    n_augment: int = N_AUGMENT,
    max_workers: int = 4,
    seed_base: int = 42,
) -> list[dict]:
    """
    Process frames with both Qwen edits and torchvision augmentations.
    Each input frame produces n_augment output frames.

    Args:
        frames: List of original frames
        task: Task description
        n_augment: Number of augmented versions per frame (default: 10)
        max_workers: Number of parallel workers
        seed_base: Base seed for reproducibility

    Returns:
        List of processed frames (len = len(frames) * n_augment)
    """
    client = get_api_client()
    prompt, negative_prompt = build_prompt(task)

    # Total output frames
    total_output = len(frames) * n_augment
    results: list[dict] = [{}] * total_output

    num_nodes = client.num_nodes

    def process_single(args):
        frame_idx, frame, augment_idx, seed, assigned_node = args

        new_frame = process_single_frame_with_augmentation(
            frame=frame,
            task=task,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            node_idx=assigned_node,
            augment_idx=augment_idx,
        )

        # Calculate output index
        output_idx = frame_idx * n_augment + augment_idx
        return output_idx, new_frame

    # Create tasks: for each frame, create n_augment tasks
    tasks = []
    for frame_idx, frame in enumerate(frames):
        for augment_idx in range(n_augment):
            seed = seed_base + frame_idx * 10000 + augment_idx * 100

            # Only use API nodes for Qwen edits (augment_idx == 0)
            if augment_idx == 0:
                assigned_node = frame_idx % num_nodes
            else:
                assigned_node = 0  # Not used for torchvision augmentation

            tasks.append((frame_idx, frame, augment_idx, seed, assigned_node))

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, task_args): task_args for task_args in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(tasks),
            desc=f"Processing (x{n_augment} augmentation)",
        ):
            output_idx, new_frame = future.result()
            results[output_idx] = new_frame

    return results


def augment_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    api_urls: List[str] | str = "http://localhost:8000",
    n_augment: int = N_AUGMENT,
    max_workers: int = 4,
    use_batch: bool = True,
) -> None:
    """
    Dataset augmentation with Qwen edits and torchvision augmentations.
    Each frame is expanded to n_augment frames.
    """
    global _api_client
    _api_client = QwenImageEditClient(api_urls)
    health = _api_client.health_check()
    if not health["healthy_nodes"]:
        raise RuntimeError(
            f"No healthy API servers available. Unhealthy: {health['unhealthy_nodes']}"
        )

    logger.info(f"Using {len(health['healthy_nodes'])} healthy node(s) for processing")
    logger.info(f"Augmentation multiplier: 1 frame -> {n_augment} frames")
    logger.info("  - 1 Qwen-edited frame")
    logger.info(f"  - {n_augment - 1} torchvision-augmented frames")

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
        f"Processing {num_episodes} episodes -> {num_episodes * n_augment} augmented episodes"
    )

    for ep_idx in tqdm(range(num_episodes), desc="Augment episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])

        if use_batch:
            frames = [original_ds[idx] for idx in range(start_idx, end_idx)]

            # Process all frames with augmentation
            processed_frames = process_frames_batch_parallel_with_augmentation(
                frames,
                new_task,
                n_augment=n_augment,
                max_workers=max_workers,
                seed_base=ep_idx * 100000,
            )

            # Save as n_augment separate episodes
            for aug_idx in range(n_augment):
                # Extract frames for this augmentation variant
                aug_frames = [
                    processed_frames[frame_idx * n_augment + aug_idx]
                    for frame_idx in range(len(frames))
                ]

                for new_frame in aug_frames:
                    dst_ds.add_frame(new_frame)

                dst_ds.save_episode()

        else:
            # Non-batch mode: process frame by frame
            for aug_idx in range(n_augment):
                prompt, negative_prompt = build_prompt(new_task)

                for idx in tqdm(
                    range(start_idx, end_idx), desc=f"Episode {ep_idx} Aug {aug_idx}"
                ):
                    frame = original_ds[idx]
                    new_frame = process_single_frame_with_augmentation(
                        frame=frame,
                        task=new_task,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=ep_idx * 100000 + idx * 100 + aug_idx,
                        node_idx=idx % _api_client.num_nodes,
                        augment_idx=aug_idx,
                    )
                    dst_ds.add_frame(new_frame)

                dst_ds.save_episode()

    dst_ds.finalize()
    diff_time = time.time() - start
    logger.info(f"Total time: {diff_time:.2f}s")

    # Merge with original dataset
    aug_ds = LeRobotDataset(dst_repo_id)
    total_episodes = num_episodes + num_episodes * n_augment
    merged = merge_datasets(
        [original_ds, aug_ds], output_repo_id=f"{dst_repo_id}_merged"
    )
    merged.finalize()

    logger.info(
        f"Done! Original: {num_episodes} episodes, "
        f"Augmented: {num_episodes * n_augment} episodes"
    )
    logger.info(f"Saved to: {dst_repo_id}")
    logger.info(f"Merged: {total_episodes} episodes, saved to: {dst_repo_id}_merged")


def parse_api_urls(url_string: str) -> List[str]:
    """Parse comma-separated URL string into list."""
    return [url.strip() for url in url_string.split(",") if url.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Dataset augmentation with Qwen edits + torchvision augmentations"
    )
    p.add_argument("--src-repo-id", required=True, help="Source dataset repo ID")
    p.add_argument("--dst-repo-id", required=True, help="Destination dataset repo ID")
    p.add_argument(
        "--api-urls",
        default="http://localhost:11303",
        help="API server URL(s), comma-separated for multiple nodes",
    )
    p.add_argument(
        "--n-augment",
        type=int,
        default=N_AUGMENT,
        help=f"Number of augmented frames per original frame (default: {N_AUGMENT})",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Parallel workers for processing",
    )
    p.add_argument("--use-batch", action="store_true", help="Enable batch processing")
    p.add_argument("--offline", action="store_true", help="Use offline cache paths")
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
        n_augment=args.n_augment,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
    )


if __name__ == "__main__":
    main()
