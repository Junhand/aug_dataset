"""
Dataset Augmentation Client with Multi-Node API Support
- Server: SAM3 + Qwen-Image-Edit only (1 frame -> 1 frame)
- Client: TorchVision augmentations (1 frame -> 19 frames)
- Total: 1 frame -> 20 frames
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
import ctypes
import gc
import psutil
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from aug_instruction import generate_similar_instructions

import requests
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def force_memory_release() -> None:
    """GC + force glibc to return free memory to OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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
# TorchVision Augmentor (Client-side) — 19 Maximally Diverse Presets
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
    TorchVision-based image augmentation with 19 maximally diverse presets.

    Design principles:
      - Each preset produces a visually distinct result from all others.
      - Robotics-safe: no horizontal flip, no large geometric distortion
        (preserves spatial relationships needed for policy learning).
      - Covers different perceptual axes: tone, color, texture, style, geometry, occlusion.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._build_transforms()
        self._build_presets()

    def _build_transforms(self):
        """Build individual transform functions."""
        self.transforms = {
            # === Color / Tone ===
            "color_jitter_strong": v2.ColorJitter(
                brightness=0.35, contrast=0.35, saturation=0.35, hue=0.12
            ),
            "brightness_up_strong": v2.ColorJitter(brightness=(1.35, 1.55)),
            "brightness_down_strong": v2.ColorJitter(brightness=(0.45, 0.65)),
            "contrast_up_strong": v2.ColorJitter(contrast=(1.4, 1.7)),
            "contrast_down_strong": v2.ColorJitter(contrast=(0.4, 0.6)),
            "saturation_up_strong": v2.ColorJitter(saturation=(1.5, 1.8)),
            "near_grayscale": v2.ColorJitter(saturation=(0.0, 0.15)),
            "hue_shift_warm": v2.ColorJitter(hue=(0.06, 0.12)),
            "hue_shift_cool": v2.ColorJitter(hue=(-0.12, -0.06)),
            # === Blur / Sharpness ===
            "gaussian_blur_strong": v2.GaussianBlur(kernel_size=9, sigma=(1.0, 2.0)),
            "sharpness_up_strong": v2.RandomAdjustSharpness(
                sharpness_factor=2.5, p=1.0
            ),
            # === Style / Tone-mapping ===
            "autocontrast": v2.RandomAutocontrast(p=1.0),
            "equalize": v2.RandomEqualize(p=1.0),
            "posterize_4bit": v2.RandomPosterize(bits=4, p=1.0),
            "posterize_3bit": v2.RandomPosterize(bits=3, p=1.0),
            "solarize": v2.RandomSolarize(threshold=0.7, p=1.0),
            # === Geometric (mild, robotics-safe) ===
            "perspective_mild": v2.RandomPerspective(distortion_scale=0.15, p=1.0),
            "affine_mild": v2.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.92, 1.08)
            ),
            "elastic_mild": v2.ElasticTransform(alpha=30.0, sigma=4.0),
            # === Erasing / Occlusion ===
            "random_erasing": v2.RandomErasing(
                p=1.0, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0
            ),
        }

    def _build_presets(self):
        """
        19 maximally diverse presets (aug_idx 1–19, with aug_idx 0 being Qwen edit).

        Perceptual axes covered:
          0-3   : Global tone / brightness
          4-6   : Color shifts (warm, cool, desaturated)
          7-8   : Texture (blur, sharp)
          9-12  : Style (posterize, solarize, equalize)
          13-15 : Geometry (perspective, affine, elastic)
          16-18 : Cross-axis combinations
        """
        self.augmentation_presets = [
            # --- Global tone / brightness (0-3) ---
            {
                "transforms": ["brightness_up_strong", "contrast_down_strong"],
                "noise": False,
            },
            {
                "transforms": ["brightness_down_strong", "contrast_up_strong"],
                "noise": True,
                "noise_std": (0.02, 0.06),
            },
            {
                "transforms": ["contrast_up_strong", "saturation_up_strong"],
                "noise": False,
            },
            {
                "transforms": ["contrast_down_strong", "near_grayscale"],
                "noise": False,
            },
            # --- Color shifts (4-6) ---
            {
                "transforms": ["hue_shift_warm", "saturation_up_strong"],
                "noise": False,
            },
            {
                "transforms": ["hue_shift_cool", "brightness_down_strong"],
                "noise": False,
            },
            {
                "transforms": ["near_grayscale"],
                "noise": True,
                "noise_std": (0.04, 0.08),
            },
            # --- Texture (7-8) ---
            {
                "transforms": ["gaussian_blur_strong"],
                "noise": False,
            },
            {
                "transforms": ["sharpness_up_strong"],
                "noise": True,
                "noise_std": (0.03, 0.06),
            },
            # --- Style / Tone-mapping (9-12) ---
            {
                "transforms": ["posterize_4bit"],
                "noise": False,
            },
            {
                "transforms": ["posterize_3bit", "saturation_up_strong"],
                "noise": False,
            },
            {
                "transforms": ["solarize"],
                "noise": False,
            },
            {
                "transforms": ["equalize"],
                "noise": False,
            },
            # --- Geometry (13-15) ---
            {
                "transforms": ["perspective_mild"],
                "noise": False,
            },
            {
                "transforms": ["affine_mild"],
                "noise": False,
            },
            {
                "transforms": ["elastic_mild"],
                "noise": False,
            },
            # --- Cross-axis combinations (16-18) ---
            {
                "transforms": [
                    "gaussian_blur_strong",
                    "brightness_down_strong",
                    "near_grayscale",
                ],
                "noise": True,
                "noise_std": (0.05, 0.10),
            },
            {
                "transforms": [
                    "color_jitter_strong",
                    "perspective_mild",
                    "sharpness_up_strong",
                ],
                "noise": False,
            },
            {
                "transforms": ["autocontrast", "random_erasing"],
                "noise": False,
            },
        ]

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

        orig_h, orig_w = result.shape[-2], result.shape[-1]

        for transform_name in preset["transforms"]:
            transform = self.transforms.get(transform_name)
            if transform is None:
                continue
            try:
                result = transform(result)
            except Exception as e:
                logger.warning(f"Transform {transform_name} failed: {e}")
                continue

        if result.shape[-2] != orig_h or result.shape[-1] != orig_w:
            result = F.resize(result, [orig_h, orig_w], antialias=True)

        if preset.get("noise", False):
            noise_range = preset.get("noise_std", (0.01, 0.04))
            noise_std = random.uniform(*noise_range)
            result = self.add_gaussian_noise(result, noise_std)

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
# Frame Processing
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


# =============================================================================
# Sequential aug processing (Fix #4: process one aug_idx at a time)
# =============================================================================


def process_episode_all_augs_sequential_save(
    frames: List[dict],
    task: str,
    n_augment: int,
    seed_base: int,
    max_workers: int,
    dst_ds,
    overlay_alpha: float = 0.0,
) -> None:
    """
    Process all frames for all aug_idx and save directly to dataset.

    Memory-efficient: processes one aug_idx at a time, saves immediately,
    then releases memory before moving to the next aug_idx.
    """
    client = get_api_client()
    prompt, negative_prompt = build_prompt(task)
    num_nodes = client.num_nodes
    num_frames = len(frames)

    for aug_idx in range(n_augment):
        aug_results: List[Optional[dict]] = [None] * num_frames

        def process_frame(args):
            frame_idx, frame, seed, node_idx = args
            return frame_idx, process_single_frame_single_aug(
                frame=frame,
                task=task,
                prompt=prompt,
                negative_prompt=negative_prompt,
                aug_idx=aug_idx,
                seed=seed,
                node_idx=node_idx,
                overlay_alpha=overlay_alpha,
            )

        tasks = [
            (fi, f, seed_base + fi * 100000, fi % num_nodes)
            for fi, f in enumerate(frames)
        ]

        if aug_idx == 0:
            actual_workers = min(max_workers, num_frames)
            desc = "Qwen (aug=0)"
        else:
            actual_workers = min(max_workers * 2, num_frames, 64)
            desc = f"TorchVision (aug={aug_idx})"

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(process_frame, t): t for t in tasks}
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc=desc, leave=False
            ):
                fi, new_frame = future.result()
                aug_results[fi] = new_frame
            futures.clear()

        # Save this aug_idx as one episode immediately
        for fi in range(num_frames):
            dst_ds.add_frame(aug_results[fi])
            aug_results[fi] = None  # Release each frame after adding
        dst_ds.save_episode()

        # Release memory for this aug_idx
        del aug_results
        force_memory_release()


# =============================================================================
# Snapshot helpers (Fix #5: rsync for constant memory)
# =============================================================================


def _get_dataset_dir(repo_id: str) -> str:
    """Get the local filesystem path for a dataset repo_id."""
    return os.path.join(
        os.environ.get(
            "HF_LEROBOT_HOME",
            os.path.expanduser("~/.cache/huggingface/lerobot/lerobot"),
        ),
        repo_id,
    )


def _snapshot_dataset(dst_repo_id: str) -> None:
    """
    Create a snapshot of the finalized dataset as {dst_repo_id}_tmp.
    Uses rsync for streaming copy with constant memory usage.
    """
    src_dir = _get_dataset_dir(dst_repo_id)
    tmp_dir = _get_dataset_dir(f"{dst_repo_id}_tmp")

    if not os.path.exists(src_dir):
        return

    os.makedirs(tmp_dir, exist_ok=True)

    subprocess.run(
        ["rsync", "-a", "--delete", f"{src_dir}/", f"{tmp_dir}/"],
        check=True,
    )
    logger.info(f"Snapshot saved: {tmp_dir}")


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
    end_episode: Optional[int] = None,
    overlay_alpha: float = 0.0,
    resume: bool = False,
) -> None:
    """
    Dataset augmentation (Memory-efficient version).

    Args:
        start_episode: Episode index to start from (0-based, inclusive)
        end_episode: Episode index to end at (0-based, inclusive). None = last episode.
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

    # Load original dataset to extract metadata, then release
    original_ds = LeRobotDataset(src_repo_id)
    original_fps = original_ds.meta.info["fps"]
    original_features = original_ds.meta.info["features"]
    original_robot_type = original_ds.meta.info["robot_type"]
    meta_episodes = {
        "dataset_from_index": list(original_ds.meta.episodes["dataset_from_index"]),
        "dataset_to_index": list(original_ds.meta.episodes["dataset_to_index"]),
    }
    total_available = len(meta_episodes["dataset_from_index"])
    del original_ds
    force_memory_release()

    # =========================================================================
    # Resume logic (Fix #4B: modulo-based snapshot restore)
    # =========================================================================
    dst_ds = None
    tmp_repo_id = f"{dst_repo_id}_tmp"

    if resume:
        loaded_from = None
        existing_episodes = 0
        resumed_episodes = 0

        for try_repo_id, label in [
            (dst_repo_id, "main dataset"),
            (tmp_repo_id, "snapshot (_tmp)"),
        ]:
            try:
                existing_ds = LeRobotDataset(try_repo_id)
                existing_episodes = existing_ds.num_episodes
                resumed_episodes = existing_episodes // n_augment
                logger.info(
                    f"Found {label} with {existing_episodes} episodes "
                    f"({resumed_episodes} source episodes completed)"
                )
                loaded_from = try_repo_id
                del existing_ds
                force_memory_release()
                break
            except Exception as e:
                logger.info(f"Could not load {label} ({try_repo_id}): {e}")

        if loaded_from is not None and resumed_episodes > 0:
            # If main dataset has incomplete source episode, restore from snapshot
            if loaded_from == dst_repo_id and existing_episodes % n_augment != 0:
                logger.warning(
                    f"Main dataset has {existing_episodes} episodes "
                    f"(not divisible by {n_augment}). "
                    f"Interrupted mid-episode. Restoring from snapshot..."
                )
                tmp_dir = _get_dataset_dir(tmp_repo_id)
                if os.path.exists(tmp_dir):
                    try:
                        tmp_ds = LeRobotDataset(tmp_repo_id)
                        tmp_episodes = tmp_ds.num_episodes
                        resumed_episodes = tmp_episodes // n_augment
                        del tmp_ds
                        force_memory_release()

                        dst_dir = _get_dataset_dir(dst_repo_id)
                        if os.path.exists(dst_dir):
                            shutil.rmtree(dst_dir)
                        shutil.copytree(tmp_dir, dst_dir)
                        logger.info(
                            f"Restored from snapshot: {tmp_episodes} episodes "
                            f"({resumed_episodes} source episodes)"
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Main dataset is incomplete and snapshot restore failed: {e}\n"
                            f"Please inspect manually."
                        )
                else:
                    raise RuntimeError(
                        f"Main dataset has {existing_episodes} episodes "
                        f"(incomplete, not divisible by {n_augment}) "
                        f"and no snapshot found. Cannot resume safely."
                    )

            # If loaded from snapshot directly (main dataset unreadable)
            elif loaded_from == tmp_repo_id:
                logger.info("Restoring snapshot to main dataset directory...")
                dst_dir = _get_dataset_dir(dst_repo_id)
                tmp_dir = _get_dataset_dir(tmp_repo_id)
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(tmp_dir, dst_dir)
                logger.info(f"Restored {tmp_dir} -> {dst_dir}")

            start_episode = resumed_episodes
            dst_ds = LeRobotDataset(dst_repo_id)
            dst_ds.start_image_writer(
                num_processes=32,
                num_threads=2,
            )
        else:
            logger.info("No usable dataset found. Starting from scratch...")
            resume = False

            dst_path = _get_dataset_dir(dst_repo_id)
            if os.path.exists(dst_path):
                logger.warning(f"Removing incomplete dataset directory: {dst_path}")
                shutil.rmtree(dst_path)

    if dst_ds is None:
        dst_ds = LeRobotDataset.create(
            repo_id=dst_repo_id,
            fps=original_fps,
            features=original_features,
            robot_type=original_robot_type,
            use_videos=True,
            image_writer_processes=32,
            image_writer_threads=2,
        )

    start = time.time()

    # Calculate episode range
    ep_start = min(start_episode, total_available)
    if end_episode is not None:
        ep_end = min(
            end_episode + 1, total_available
        )  # +1 because end_episode is inclusive
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

        # Fix #1: Reload original_ds each episode to prevent HF cache buildup
        original_ds = LeRobotDataset(src_repo_id)
        new_task = generate_similar_instructions(original_ds[start_idx]["task"])
        frames = [original_ds[idx] for idx in range(start_idx, end_idx)]
        del original_ds
        force_memory_release()

        logger.info(f"Episode {ep_idx}: {len(frames)} frames")

        # Fix #4A: Process one aug_idx at a time, save immediately
        process_episode_all_augs_sequential_save(
            frames=frames,
            task=new_task,
            n_augment=n_augment,
            seed_base=ep_idx * 1000000,
            max_workers=max_workers,
            dst_ds=dst_ds,
            overlay_alpha=overlay_alpha,
        )

        # Release frames
        del frames
        force_memory_release()

        # Log memory usage
        mem_mb = get_memory_usage_mb()
        logger.info(f"Episode {ep_idx} complete. Memory: {mem_mb:.1f} MB")

        # Fix #2: Finalize, then explicit del before recreate
        logger.info(f"Finalizing dataset after episode {ep_idx}...")
        dst_ds.finalize()
        del dst_ds
        dst_ds = None
        force_memory_release()

        # Fix #5: Snapshot with rsync (constant memory)
        logger.info(f"Creating snapshot after episode {ep_idx}...")
        _snapshot_dataset(dst_repo_id)

        # Recreate dataset in append mode if more episodes to process
        if ep_idx < ep_end - 1:
            dst_ds = LeRobotDataset(dst_repo_id)
            dst_ds.start_image_writer(
                num_processes=32,
                num_threads=2,
            )

            mem_mb = get_memory_usage_mb()
            logger.info(f"Reopened dataset for next episode. Memory: {mem_mb:.1f} MB")

    diff_time = time.time() - start
    logger.info(f"Total time: {diff_time:.2f}s")

    # Merge (reload datasets fresh)
    force_memory_release()
    original_ds = LeRobotDataset(src_repo_id)
    aug_ds = LeRobotDataset(dst_repo_id)
    merged = merge_datasets(
        [original_ds, aug_ds], output_repo_id=f"{dst_repo_id}_merged"
    )
    merged.finalize()
    del original_ds, aug_ds, merged
    force_memory_release()

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
        "--end-episode",
        type=int,
        default=None,
        help="Episode index to end at (0-based, inclusive). None = last episode.",
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
        end_episode=args.end_episode,
        overlay_alpha=args.overlay_alpha,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
