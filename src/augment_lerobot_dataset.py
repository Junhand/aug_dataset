"""
Dataset Augmentation Client with Multi-Node API Support
- Server: SAM3 + Qwen-Image-Edit only (1 frame -> 1 frame)
- Client: TorchVision augmentations (1 frame -> 49 frames)
- Total: 1 frame -> 50 frames
- v8: Memory-efficient processing (aug_idx by aug_idx)
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
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions

import requests
from tqdm import tqdm
from torchvision.transforms import v2


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


LOG_FORMAT = (
    "\n==================================================\n"
    "%(asctime)s - %(name)s - %(levelname)s\n"
    "%(message)s"
    "\n==================================================\n"
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# SEGMENT_TASKS = "shelf, object, pet bottle, container, box"  # task05
SEGMENT_TASKS = "box, oven, microwave oven, object, food, dish, table, plate, toy, stuffed toy"  # task48

# Augmentation multiplier: 1 original -> N_AUGMENT frames
N_AUGMENT = 20


def build_prompt(task: str) -> tuple[str, str]:
    prompt = (
        "Make minimal and subtle changes to only small, task-irrelevant regions of the background. "
        "Keep most of the original background unchanged. "
        "Preserve the same indoor setting, materials, scene context, objects, and overall atmosphere. "
        f"Task: {task}"
    )
    negative_prompt = " "
    return prompt, negative_prompt


# =============================================================================
# TorchVision Augmentor (Client-side)
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for torchvision augmentations."""

    brightness_range: Tuple[float, float] = (0.6, 1.4)
    contrast_range: Tuple[float, float] = (0.6, 1.4)
    saturation_range: Tuple[float, float] = (0.6, 1.4)
    hue_range: Tuple[float, float] = (-0.15, 0.15)
    sharpness_range: Tuple[float, float] = (0.3, 2.5)
    noise_std_range: Tuple[float, float] = (0.005, 0.08)


class TorchVisionAugmentor:
    """
    TorchVision-based image augmentation for robotics datasets.
    Provides 49 augmentation presets.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._build_transforms()
        self._build_presets()

    def _build_transforms(self):
        """Build individual transform functions."""
        self.transforms = {
            # === Color Jitter Variants ===
            "color_jitter_very_light": v2.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01
            ),
            "color_jitter_light": v2.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            "color_jitter_medium": v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            "color_jitter_strong": v2.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            "color_jitter_very_strong": v2.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15
            ),
            # === Brightness Variants ===
            "brightness_up_slight": v2.ColorJitter(brightness=(1.05, 1.15)),
            "brightness_up_medium": v2.ColorJitter(brightness=(1.15, 1.3)),
            "brightness_up_strong": v2.ColorJitter(brightness=(1.3, 1.5)),
            "brightness_down_slight": v2.ColorJitter(brightness=(0.85, 0.95)),
            "brightness_down_medium": v2.ColorJitter(brightness=(0.7, 0.85)),
            "brightness_down_strong": v2.ColorJitter(brightness=(0.5, 0.7)),
            # === Contrast Variants ===
            "contrast_up_slight": v2.ColorJitter(contrast=(1.05, 1.15)),
            "contrast_up_medium": v2.ColorJitter(contrast=(1.15, 1.3)),
            "contrast_up_strong": v2.ColorJitter(contrast=(1.3, 1.5)),
            "contrast_down_slight": v2.ColorJitter(contrast=(0.85, 0.95)),
            "contrast_down_medium": v2.ColorJitter(contrast=(0.7, 0.85)),
            "contrast_down_strong": v2.ColorJitter(contrast=(0.5, 0.7)),
            # === Saturation Variants ===
            "saturation_up_slight": v2.ColorJitter(saturation=(1.1, 1.2)),
            "saturation_up_medium": v2.ColorJitter(saturation=(1.2, 1.4)),
            "saturation_up_strong": v2.ColorJitter(saturation=(1.4, 1.6)),
            "saturation_down_slight": v2.ColorJitter(saturation=(0.8, 0.9)),
            "saturation_down_medium": v2.ColorJitter(saturation=(0.6, 0.8)),
            "saturation_down_strong": v2.ColorJitter(saturation=(0.3, 0.6)),
            "grayscale_partial": v2.ColorJitter(saturation=(0.0, 0.3)),
            # === Hue Variants ===
            "hue_shift_slight_pos": v2.ColorJitter(hue=(0.02, 0.05)),
            "hue_shift_medium_pos": v2.ColorJitter(hue=(0.05, 0.1)),
            "hue_shift_slight_neg": v2.ColorJitter(hue=(-0.05, -0.02)),
            "hue_shift_medium_neg": v2.ColorJitter(hue=(-0.1, -0.05)),
            # === Gaussian Blur Variants ===
            "gaussian_blur_light": v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            "gaussian_blur_medium": v2.GaussianBlur(kernel_size=5, sigma=(0.3, 0.7)),
            "gaussian_blur_strong": v2.GaussianBlur(kernel_size=7, sigma=(0.7, 1.2)),
            # === Sharpness Variants ===
            "sharpness_up_slight": v2.RandomAdjustSharpness(
                sharpness_factor=1.3, p=1.0
            ),
            "sharpness_up_medium": v2.RandomAdjustSharpness(
                sharpness_factor=1.7, p=1.0
            ),
            "sharpness_up_strong": v2.RandomAdjustSharpness(
                sharpness_factor=2.2, p=1.0
            ),
            "sharpness_down": v2.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0),
            # === Autocontrast ===
            "autocontrast": v2.RandomAutocontrast(p=1.0),
        }

    def _build_presets(self):
        """Build 19 augmentation presets (aug_idx 1-19, with aug_idx 0 being Qwen edit)."""
        self.augmentation_presets = [
            # === Single Transform Presets (aug_idx 1-10) ===
            ["color_jitter_light"],  # 0 -> aug_idx 1
            ["color_jitter_medium"],  # 1 -> aug_idx 2
            ["brightness_up_medium"],  # 2 -> aug_idx 3
            ["brightness_down_medium"],  # 3 -> aug_idx 4
            ["contrast_up_medium"],  # 4 -> aug_idx 5
            ["contrast_down_medium"],  # 5 -> aug_idx 6
            ["saturation_up_medium"],  # 6 -> aug_idx 7
            ["saturation_down_medium"],  # 7 -> aug_idx 8
            ["gaussian_blur_light"],  # 8 -> aug_idx 9
            ["gaussian_blur_medium"],  # 9 -> aug_idx 10
            # === Single Transform Presets (aug_idx 11-14) ===
            ["sharpness_up_medium"],  # 10 -> aug_idx 11
            ["sharpness_down"],  # 11 -> aug_idx 12
            ["autocontrast"],  # 12 -> aug_idx 13
            ["grayscale_partial"],  # 13 -> aug_idx 14
            # === Combination Presets (aug_idx 15-19) ===
            ["brightness_up_slight", "color_jitter_light"],  # 14 -> aug_idx 15
            ["brightness_down_slight", "contrast_down_slight"],  # 15 -> aug_idx 16
            ["contrast_up_slight", "saturation_up_medium"],  # 16 -> aug_idx 17
            ["gaussian_blur_light", "color_jitter_light"],  # 17 -> aug_idx 18
            ["sharpness_up_slight", "brightness_up_slight"],  # 18 -> aug_idx 19
        ]

        # Presets that should have noise added (odd indices)
        self.noise_presets = {1, 3, 5, 7, 9, 11, 13, 15, 17}

    def add_gaussian_noise(self, tensor: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def apply_preset(
        self, tensor: torch.Tensor, preset_idx: int, seed: int
    ) -> torch.Tensor:
        """Apply a specific augmentation preset to a tensor."""
        torch.manual_seed(seed)
        random.seed(seed)

        preset_idx = preset_idx % len(self.augmentation_presets)
        preset = self.augmentation_presets[preset_idx]
        result = tensor.clone()

        for transform_name in preset:
            if transform_name in self.transforms:
                try:
                    result = self.transforms[transform_name](result)
                except Exception as e:
                    logger.warning(f"Transform {transform_name} failed: {e}")
                    continue

        if preset_idx in self.noise_presets:
            noise_std = random.uniform(0.01, 0.04)
            result = self.add_gaussian_noise(result, noise_std)

        # Clamp to [0.0, 1.0] to avoid floating point precision issues
        result = torch.clamp(result, 0.0, 1.0)

        return result

    @property
    def num_presets(self) -> int:
        return len(self.augmentation_presets)


# =============================================================================
# Qwen API Client (Server-side: SAM3 + Qwen only)
# =============================================================================


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
        prompt: str,
        task: str,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
        node_idx: Optional[int] = None,
        overlay_alpha: float = 0.0,
    ) -> torch.Tensor:
        """Edit single image using Qwen API (SAM3 + Qwen-Image-Edit)."""
        payload = {
            "image_tensor_b64": self.tensor_to_base64(image_tensor),
            "prompt": prompt,
            "task": task,
            "seed": seed,
            "overlay_alpha": overlay_alpha,
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
        health = self.health_check()
        return len(health["healthy_nodes"]) > 0


# =============================================================================
# Global Instances
# =============================================================================

_api_client: Optional[QwenImageEditClient] = None
_augmentor: Optional[TorchVisionAugmentor] = None


def get_api_client(
    base_urls: List[str] | str = "http://localhost:8000",
) -> QwenImageEditClient:
    global _api_client
    if _api_client is None:
        _api_client = QwenImageEditClient(base_urls)
        health = _api_client.health_check()
        if not health["healthy_nodes"]:
            raise RuntimeError(
                f"No healthy API servers. Unhealthy: {health['unhealthy_nodes']}"
            )
        logger.info(f"Connected to {len(health['healthy_nodes'])} healthy node(s)")
        for node in health["healthy_nodes"]:
            logger.info(
                f"  - Node {node['index']}: {node['url']} "
                f"(GPUs: {node['status'].get('num_gpus', 'N/A')})"
            )
    return _api_client


def get_augmentor() -> TorchVisionAugmentor:
    global _augmentor
    if _augmentor is None:
        _augmentor = TorchVisionAugmentor()
        logger.info(f"Initialized augmentor with {_augmentor.num_presets} presets")
    return _augmentor


# =============================================================================
# Frame Processing (Memory-Efficient: One aug_idx at a time)
# =============================================================================

SKIP_KEYS = {"index", "episode_index", "timestamp", "frame_index", "task_index"}


def process_single_frame_single_aug(
    frame: dict,
    task: str,
    prompt: str,
    negative_prompt: str,
    aug_idx: int,
    seed: int,
    node_idx: int,
    overlay_alpha: float = 0.0,
) -> dict:
    """
    Process a single frame for a single aug_idx.
    Memory efficient: only returns 1 frame dict.
    """
    client = get_api_client()
    augmentor = get_augmentor()

    aug_seed = seed + aug_idx * 1000
    new_frame = {}

    for key, value in frame.items():
        if key in SKIP_KEYS:
            continue

        if "task" in key:
            new_frame[key] = task

        elif "observation.image" in key:
            if not isinstance(value, torch.Tensor):
                new_frame[key] = value
                continue

            try:
                if key == "observation.image.hand":
                    roted_cwh = torch.rot90(value, k=1, dims=(1, 2))

                    if aug_idx == 0:
                        edited_cwh = client.edit_image(
                            image_tensor=roted_cwh,
                            prompt=prompt,
                            task=SEGMENT_TASKS,
                            negative_prompt=negative_prompt,
                            seed=aug_seed,
                            node_idx=node_idx,
                            overlay_alpha=overlay_alpha,
                        )
                    else:
                        edited_cwh = augmentor.apply_preset(
                            roted_cwh, aug_idx - 1, aug_seed
                        )

                    edited_chw = torch.rot90(edited_cwh, k=3, dims=(1, 2))
                    new_frame[key] = edited_chw.permute(1, 2, 0)
                else:
                    if aug_idx == 0:
                        edited = client.edit_image(
                            image_tensor=value,
                            prompt=prompt,
                            task=SEGMENT_TASKS,
                            negative_prompt=negative_prompt,
                            seed=aug_seed,
                            node_idx=node_idx,
                            overlay_alpha=overlay_alpha,
                        )
                    else:
                        edited = augmentor.apply_preset(value, aug_idx - 1, aug_seed)

                    new_frame[key] = edited.permute(1, 2, 0)

            except Exception as e:
                logger.error(f"Error processing {key} aug_idx={aug_idx}: {e}")
                new_frame[key] = value.permute(1, 2, 0) if value.dim() == 3 else value
        else:
            new_frame[key] = value

    return new_frame


def process_episode_for_single_aug(
    frames: List[dict],
    task: str,
    aug_idx: int,
    seed_base: int,
    max_workers: int,
    overlay_alpha: float = 0.0,
) -> List[Optional[dict]]:
    """
    Process all frames of an episode for a single aug_idx in parallel.
    Returns: List of processed frames for this aug_idx only.
    """
    client = get_api_client()
    prompt, negative_prompt = build_prompt(task)
    num_nodes = client.num_nodes

    num_frames = len(frames)
    results: List[Optional[dict]] = [None] * num_frames

    def process_single(args):
        frame_idx, frame, seed, assigned_node = args
        new_frame = process_single_frame_single_aug(
            frame=frame,
            task=task,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aug_idx=aug_idx,
            seed=seed,
            node_idx=assigned_node,
            overlay_alpha=overlay_alpha,
        )
        return frame_idx, new_frame

    tasks = [
        (frame_idx, frame, seed_base + frame_idx * 100000, frame_idx % num_nodes)
        for frame_idx, frame in enumerate(frames)
    ]

    # Use more workers for TorchVision (CPU-only) vs Qwen (GPU)
    actual_workers = (
        min(max_workers, num_frames)
        if aug_idx == 0
        else min(max_workers * 2, num_frames, 64)
    )

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(process_single, t): t for t in tasks}

        for future in as_completed(futures):
            frame_idx, new_frame = future.result()
            results[frame_idx] = new_frame

    return results


def process_episode_all_augs_parallel(
    frames: List[dict],
    task: str,
    n_augment: int,
    seed_base: int,
    max_workers: int,
    overlay_alpha: float = 0.0,
) -> List[List[Optional[dict]]]:
    """
    Process all frames for all aug_idx in parallel (frame-level parallelism).

    Returns: results[aug_idx] = List of processed frames

    Strategy:
    - aug_idx=0 (Qwen): Process all frames in parallel with API calls
    - aug_idx=1-19 (TorchVision): Process all frames in parallel on CPU
    """
    client = get_api_client()
    prompt, negative_prompt = build_prompt(task)
    num_nodes = client.num_nodes
    num_frames = len(frames)

    # Results: [aug_idx][frame_idx] = processed frame
    all_results: List[List[Optional[dict]]] = [
        [None] * num_frames for _ in range(n_augment)
    ]

    def process_frame_aug(args):
        frame_idx, frame, aug_idx, seed, node_idx = args
        new_frame = process_single_frame_single_aug(
            frame=frame,
            task=task,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aug_idx=aug_idx,
            seed=seed,
            node_idx=node_idx,
            overlay_alpha=overlay_alpha,
        )
        return frame_idx, aug_idx, new_frame

    # Process aug_idx=0 (Qwen) first - GPU bound, limited parallelism
    qwen_tasks = [
        (frame_idx, frame, 0, seed_base + frame_idx * 100000, frame_idx % num_nodes)
        for frame_idx, frame in enumerate(frames)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame_aug, t): t for t in qwen_tasks}
        for future in tqdm(
            as_completed(futures), total=len(qwen_tasks), desc="Qwen (aug=0)"
        ):
            frame_idx, aug_idx, new_frame = future.result()
            all_results[aug_idx][frame_idx] = new_frame

    # Process aug_idx=1-19 (TorchVision) - CPU bound, high parallelism
    torchvision_tasks = [
        (frame_idx, frame, aug_idx, seed_base + frame_idx * 100000, 0)
        for aug_idx in range(1, n_augment)
        for frame_idx, frame in enumerate(frames)
    ]

    # Use more workers for CPU-bound TorchVision tasks
    tv_workers = min(max_workers * 4, 64, len(torchvision_tasks))

    with ThreadPoolExecutor(max_workers=tv_workers) as executor:
        futures = {executor.submit(process_frame_aug, t): t for t in torchvision_tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(torchvision_tasks),
            desc=f"TorchVision (aug=1-{n_augment - 1})",
        ):
            frame_idx, aug_idx, new_frame = future.result()
            all_results[aug_idx][frame_idx] = new_frame

    return all_results


# =============================================================================
# Main Dataset Augmentation
# =============================================================================


def augment_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    api_urls: List[str] | str = "http://localhost:8000",
    n_augment: int = N_AUGMENT,
    max_workers: int = 16,
    start_episode: int = 0,
    num_episodes: Optional[int] = None,
    overlay_alpha: float = 0.0,
    resume: bool = False,
) -> None:
    """
    Dataset augmentation (Memory-efficient version):
    - Processes one aug_idx at a time
    - Saves episode immediately after processing
    - Releases memory before moving to next aug_idx

    Args:
        start_episode: Episode index to start from (0-based)
        num_episodes: Number of episodes to process (None = all remaining)
        resume: If True, resume from where it left off (auto-detect start_episode)
    """
    global _api_client
    _api_client = QwenImageEditClient(api_urls)
    health = _api_client.health_check()
    if not health["healthy_nodes"]:
        raise RuntimeError(
            f"No healthy API servers. Unhealthy: {health['unhealthy_nodes']}"
        )

    get_augmentor()

    logger.info(f"Using {len(health['healthy_nodes'])} healthy node(s)")
    logger.info(f"Augmentation: 1 frame -> {n_augment} frames")
    logger.info("  - aug_idx=0: Server (SAM3 + Qwen)")
    logger.info(f"  - aug_idx=1-{n_augment - 1}: Client (TorchVision)")
    logger.info("Memory-efficient mode: processing one aug_idx at a time")

    original_ds = LeRobotDataset(src_repo_id)

    # Check if resuming from existing dataset
    dst_ds = None
    resumed_episodes = 0

    if resume:
        try:
            existing_ds = LeRobotDataset(dst_repo_id)
            existing_episodes = existing_ds.num_episodes
            # Each source episode produces n_augment augmented episodes
            resumed_episodes = existing_episodes // n_augment
            logger.info(f"Found existing dataset with {existing_episodes} episodes")
            logger.info(
                f"Resuming from source episode {resumed_episodes} (skipping {resumed_episodes} already processed)"
            )

            # Override start_episode with resumed position
            start_episode = resumed_episodes

            # Open in append mode
            dst_ds = existing_ds
            dst_ds.start_image_writer(
                num_processes=56,
                num_threads=2,
            )
        except Exception as e:
            logger.info(f"No existing dataset found or error loading: {e}")
            logger.info("Starting from scratch...")
            resume = False

    if dst_ds is None:
        dst_ds = LeRobotDataset.create(
            repo_id=dst_repo_id,
            fps=original_ds.meta.info["fps"],
            features=original_ds.meta.info["features"],
            robot_type=original_ds.meta.info["robot_type"],
            use_videos=True,
            image_writer_processes=16,
            image_writer_threads=2,
        )

    start = time.time()
    meta_episodes = original_ds.meta.episodes
    total_available = len(meta_episodes["dataset_from_index"])

    # Calculate episode range
    ep_start = min(start_episode, total_available)
    if num_episodes is not None:
        ep_end = min(ep_start + num_episodes, total_available)
    else:
        ep_end = total_available

    total_to_process = ep_end - ep_start

    logger.info(
        f"Processing episodes {ep_start} to {ep_end - 1} ({total_to_process} episodes)"
    )
    logger.info(f"Output: {total_to_process * n_augment} augmented episodes")

    for ep_idx in tqdm(range(ep_start, ep_end), desc="Episodes"):
        start_idx = meta_episodes["dataset_from_index"][ep_idx]
        end_idx = meta_episodes["dataset_to_index"][ep_idx]
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])

        # Load frames once per episode
        frames = [original_ds[idx] for idx in range(start_idx, end_idx)]
        logger.info(f"Episode {ep_idx}: {len(frames)} frames")

        # Process all frames for all aug_idx in parallel
        all_results = process_episode_all_augs_parallel(
            frames=frames,
            task=new_task,
            n_augment=n_augment,
            seed_base=ep_idx * 1000000,
            max_workers=max_workers,
            overlay_alpha=overlay_alpha,
        )

        # Save each aug_idx as a separate episode
        for aug_idx in range(n_augment):
            for frame_idx in range(len(frames)):
                dst_ds.add_frame(all_results[aug_idx][frame_idx])
            dst_ds.save_episode()

        # Release memory
        del all_results
        del frames
        gc.collect()

        # Log memory usage
        mem_mb = get_memory_usage_mb()
        logger.info(f"Episode {ep_idx} complete. Memory: {mem_mb:.1f} MB")

        # Finalize and recreate dataset every episode to flush memory
        logger.info(f"Finalizing dataset after episode {ep_idx}...")
        dst_ds.finalize()

        # Recreate dataset in append mode if more episodes to process
        if ep_idx < ep_end - 1:
            dst_ds = LeRobotDataset(dst_repo_id)
            # Re-enable write mode
            dst_ds.start_image_writer(
                num_processes=56,
                num_threads=2,
            )

    diff_time = time.time() - start
    logger.info(f"Total time: {diff_time:.2f}s")

    # Merge (reload the finalized augmented dataset)
    aug_ds = LeRobotDataset(dst_repo_id)
    merged = merge_datasets(
        [original_ds, aug_ds], output_repo_id=f"{dst_repo_id}_merged"
    )
    merged.finalize()

    logger.info(f"Done! Saved to: {dst_repo_id}")


def parse_api_urls(url_string: str) -> List[str]:
    return [url.strip() for url in url_string.split(",") if url.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Dataset augmentation: Server (SAM3+Qwen) + Client (TorchVision) - Memory Efficient"
    )
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--dst-repo-id", required=True)
    p.add_argument("--api-urls", default="http://localhost:11303")
    p.add_argument("--n-augment", type=int, default=N_AUGMENT)
    p.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Parallel workers for Qwen API calls",
    )
    p.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from (0-based)",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to process (None = all remaining)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing dataset (auto-detect start position)",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.0,
        help="Overlay alpha for mask visualization (0.0 = no overlay, 1.0 = full red)",
    )
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
        start_episode=args.start_episode,
        num_episodes=args.num_episodes,
        overlay_alpha=args.overlay_alpha,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
