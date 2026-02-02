"""
Qwen-Image-Edit API Server with SAM3 Segmentation + TorchVision Augmentation
(8 GPU Multiprocess)

Supports two types of image processing:
1. Qwen edit with SAM3 segmentation (background modification)
2. TorchVision augmentations (color, brightness, contrast, etc.)
"""

import io
import os
import math
import base64
import asyncio
import random
import multiprocessing as mp
from typing import Optional, Tuple, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torchvision.transforms import v2


# Configuration
MAX_QUEUE_SIZE = 1000
REQUEST_TIMEOUT = 3000
NUM_GPUS = 8
JPEG_QUALITY = 95

# Model paths
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_PATH = "/home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"

# Inference settings
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 1.0


def tensor_to_base64(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return torch.load(buffer, weights_only=True)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    tensor = tensor.clamp(0, 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    np_img = np.array(img.convert("RGB"))
    return torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0


def composite_images(
    original: Image.Image,
    edited: Image.Image,
    mask,
    overlay_alpha: float = 0.3,
) -> Image.Image:
    original_np = np.array(original).astype(np.float32)
    h, w = original_np.shape[:2]

    if edited.size != (w, h):
        edited = edited.resize((w, h), Image.LANCZOS)
    edited_np = np.array(edited).astype(np.float32)

    mask_np = mask.detach().cpu().numpy()
    if mask_np.shape != (h, w):
        mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_pil) > 127

    result = edited_np.copy()
    result[mask_np] = original_np[mask_np]

    red = np.zeros_like(result)
    red[..., 0] = 255
    result[mask_np] = (1.0 - overlay_alpha) * result[mask_np] + overlay_alpha * red[mask_np]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


@dataclass
class AugmentationConfig:
    """Configuration for torchvision augmentations."""
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    sharpness_range: Tuple[float, float] = (0.5, 2.0)
    blur_kernel_size: int = 5
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0)
    noise_std_range: Tuple[float, float] = (0.01, 0.05)


class TorchVisionAugmentor:
    """TorchVision-based image augmentation."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
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
                sigma=self.config.blur_sigma_range
            ),
            "sharpness": v2.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
        }
        
        self.augmentation_presets = [
            ["color_jitter_light"],
            ["color_jitter_medium"],
            ["color_jitter_strong"],
            ["brightness_up", "gaussian_blur"],
            ["brightness_down", "sharpness"],
            ["contrast_up"],
            ["contrast_down", "color_jitter_light"],
            ["gaussian_blur"],
            ["sharpness", "color_jitter_light"],
        ]
    
    def add_gaussian_noise(self, tensor: torch.Tensor, std: float) -> torch.Tensor:
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
    def apply_preset(self, tensor: torch.Tensor, preset_idx: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        random.seed(seed)
        
        if preset_idx >= len(self.augmentation_presets):
            preset_idx = preset_idx % len(self.augmentation_presets)
        
        preset = self.augmentation_presets[preset_idx]
        result = tensor.clone()
        
        for transform_name in preset:
            if transform_name in self.transforms:
                result = self.transforms[transform_name](result)
        
        if preset_idx in [1, 3, 5, 7]:
            noise_std = random.uniform(0.01, 0.03)
            result = self.add_gaussian_noise(result, noise_std)
        
        return result


# Global augmentor instance (CPU-based, thread-safe)
_augmentor = TorchVisionAugmentor()


def gpu_worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, ready_event: mp.Event):
    """Worker process for a single GPU."""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from diffusers.models import QwenImageTransformer2DModel
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda:0"

    print(f"[Worker {gpu_id}] Using physical GPU {gpu_id}, visible as cuda:0")
    print(f"[Worker {gpu_id}] Initializing models...")

    try:
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)

        torch_dtype = torch.bfloat16
        model = QwenImageTransformer2DModel.from_pretrained(
            MODEL_NAME, subfolder="transformer", torch_dtype=torch_dtype
        )

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            transformer=model,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        )
        pipe.load_lora_weights(LORA_PATH)
        pipe = pipe.to(device)

        print(f"[Worker {gpu_id}] Ready! Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        ready_event.set()

    except Exception as e:
        print(f"[Worker {gpu_id}] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        ready_event.set()
        return

    def get_mask_or_empty(output):
        masks = output["masks"]
        if masks.numel() == 0:
            return None
        return masks.any(dim=0).squeeze(0)

    def segment_task_objects(image, task):
        state = sam3_processor.set_image(image)

        robot_mask = None
        for prompt in ["robot", "robot arm", "robot base", "mobile robot", "gripper", "robot body"]:
            out_r = sam3_processor.set_text_prompt(state=state, prompt=prompt)
            m = get_mask_or_empty(out_r)
            if m is not None:
                robot_mask = m
                break

        object_mask = None
        for prompt in [s.strip() for s in task.lower().split(",")]:
            out_obj = sam3_processor.set_text_prompt(state=state, prompt=prompt)
            m = get_mask_or_empty(out_obj)
            if m is not None:
                object_mask = m if object_mask is None else object_mask | m

        masks = [m for m in [robot_mask, object_mask] if m is not None]
        if not masks:
            return None
        merged = masks[0]
        for m in masks[1:]:
            merged = merged | m
        return merged

    while True:
        try:
            job = task_queue.get()
            if job is None:
                print(f"[Worker {gpu_id}] Shutting down...")
                break

            job_id, image_bytes, original_size, prompt, negative_prompt, task, seed = job

            input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask = segment_task_objects(input_pil, task)

            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[input_pil],
                generator=generator,
                num_inference_steps=NUM_INFERENCE_STEPS,
                true_cfg_scale=TRUE_CFG_SCALE,
            )
            edited_pil = result.images[0]

            if mask is not None:
                output_pil = composite_images(input_pil, edited_pil, mask)
            else:
                h, w = original_size
                if edited_pil.size != (w, h):
                    output_pil = edited_pil.resize((w, h), Image.LANCZOS)
                else:
                    output_pil = edited_pil

            output_buffer = io.BytesIO()
            output_pil.save(output_buffer, format="JPEG", quality=95)
            output_bytes = output_buffer.getvalue()

            result_queue.put((job_id, output_bytes, None))
            print(f"[Worker {gpu_id}] Job {job_id} completed (Qwen edit)")

        except Exception as e:
            import traceback
            print(f"[Worker {gpu_id}] Error: {e}")
            traceback.print_exc()
            result_queue.put((job_id, None, str(e)))


class MultiGPUManager:
    """Manager for GPU worker processes."""

    def __init__(self, num_gpus: int = NUM_GPUS):
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.ready_events = []
        self.current_gpu = 0
        self.job_futures = {}
        self.job_counter = 0
        self._lock = asyncio.Lock()
        self._result_worker_task = None

    def initialize(self):
        print(f"Detected {torch.cuda.device_count()} GPUs, using {self.num_gpus}")
        print(f"Starting {self.num_gpus} GPU worker processes...")

        ctx = mp.get_context('spawn')
        self.result_queue = ctx.Queue()

        for gpu_id in range(self.num_gpus):
            task_queue = ctx.Queue()
            ready_event = ctx.Event()
            
            self.task_queues.append(task_queue)
            self.ready_events.append(ready_event)

            p = ctx.Process(
                target=gpu_worker_process,
                args=(gpu_id, task_queue, self.result_queue, ready_event),
            )
            p.start()
            self.processes.append(p)
            print(f"[Main] Started worker process for GPU {gpu_id} (PID: {p.pid})")

        print("[Main] Waiting for all workers to initialize...")
        for i, event in enumerate(self.ready_events):
            event.wait(timeout=300)
            print(f"[Main] Worker {i} is ready")

        print(f"All {self.num_gpus} worker processes are ready!")

    async def start_result_worker(self):
        self._result_worker_task = asyncio.create_task(self._collect_results())

    async def stop_result_worker(self):
        if self._result_worker_task:
            self._result_worker_task.cancel()
            try:
                await self._result_worker_task
            except asyncio.CancelledError:
                pass

    async def _collect_results(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                result = await loop.run_in_executor(None, self._get_result_with_timeout)
                if result is None:
                    await asyncio.sleep(0.001)
                    continue
                
                job_id, output_bytes, error = result
                
                if job_id in self.job_futures:
                    future = self.job_futures.pop(job_id)
                    if error:
                        future.set_exception(RuntimeError(error))
                    else:
                        future.set_result(output_bytes)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Main] Result collector error: {e}")
                await asyncio.sleep(0.01)

    def _get_result_with_timeout(self):
        try:
            return self.result_queue.get(timeout=0.01)
        except:
            return None

    def shutdown(self):
        print("[Main] Shutting down workers...")
        for q in self.task_queues:
            q.put(None)
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        print("[Main] All workers stopped")

    async def submit_qwen_job(
        self,
        input_tensor: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        task: str,
        seed: int,
    ) -> torch.Tensor:
        """Submit a Qwen edit job to GPU workers."""
        async with self._lock:
            job_id = self.job_counter
            self.job_counter += 1
            gpu_id = self.current_gpu
            self.current_gpu = (self.current_gpu + 1) % self.num_gpus

        original_h, original_w = input_tensor.shape[1], input_tensor.shape[2]

        input_pil = tensor_to_pil(input_tensor)
        buffer = io.BytesIO()
        input_pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        image_bytes = buffer.getvalue()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.job_futures[job_id] = future

        job = (job_id, image_bytes, (original_h, original_w), prompt, negative_prompt, task, seed)
        self.task_queues[gpu_id].put(job)
        print(f"[Main] Qwen job {job_id} submitted to Worker {gpu_id}")

        try:
            output_bytes = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            self.job_futures.pop(job_id, None)
            raise HTTPException(status_code=504, detail="Request timeout")

        output_pil = Image.open(io.BytesIO(output_bytes)).convert("RGB")
        return pil_to_tensor(output_pil)

    @property
    def is_ready(self) -> bool:
        return len(self.processes) > 0 and all(p.is_alive() for p in self.processes)


manager = MultiGPUManager(num_gpus=NUM_GPUS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.initialize()
    await manager.start_result_worker()
    yield
    await manager.stop_result_worker()
    manager.shutdown()


app = FastAPI(
    title="Qwen-Image-Edit + TorchVision Augmentation API (8 GPU)",
    version="7.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request/Response Models
# ============================================================

class ImageEditRequest(BaseModel):
    """Request for Qwen image editing."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor")
    prompt: str
    task: str = Field(..., description="Task description for SAM3 segmentation")
    negative_prompt: Optional[str] = Field(default="")
    seed: Optional[int] = Field(default=42)


class ImageEditResponse(BaseModel):
    """Response for image editing."""
    image_tensor_b64: str
    success: bool = True
    message: str = "OK"


class AugmentRequest(BaseModel):
    """Request for torchvision augmentation."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor")
    preset_idx: int = Field(..., ge=0, le=8, description="Augmentation preset index (0-8)")
    seed: Optional[int] = Field(default=42)


class BatchAugmentRequest(BaseModel):
    """Request for batch torchvision augmentation (1 image -> N augmentations)."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor")
    n_augment: int = Field(default=9, ge=1, le=20, description="Number of augmentations")
    seed: Optional[int] = Field(default=42)


class BatchAugmentResponse(BaseModel):
    """Response for batch augmentation."""
    image_tensors_b64: List[str] = Field(..., description="List of augmented tensors")
    success: bool = True
    message: str = "OK"


class CombinedEditRequest(BaseModel):
    """Request for combined Qwen edit + torchvision augmentations."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor")
    prompt: str
    task: str = Field(..., description="Task description for SAM3 segmentation")
    negative_prompt: Optional[str] = Field(default="")
    n_augment: int = Field(default=10, ge=1, le=20, description="Total output frames (1 Qwen + N-1 augment)")
    seed: Optional[int] = Field(default=42)


class CombinedEditResponse(BaseModel):
    """Response for combined edit."""
    image_tensors_b64: List[str] = Field(..., description="List of processed tensors [Qwen, Aug1, Aug2, ...]")
    success: bool = True
    message: str = "OK"


# ============================================================
# API Endpoints
# ============================================================

@app.post("/edit", response_model=ImageEditResponse)
async def edit_image(request: ImageEditRequest):
    """Qwen image editing with SAM3 segmentation."""
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)

        output_tensor = await manager.submit_qwen_job(
            input_tensor=input_tensor,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            task=request.task,
            seed=request.seed or 42,
        )

        return ImageEditResponse(
            image_tensor_b64=tensor_to_base64(output_tensor),
            success=True,
            message="OK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/augment", response_model=ImageEditResponse)
async def augment_image(request: AugmentRequest):
    """Apply single torchvision augmentation preset."""
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)
        
        # Run augmentation (CPU-based, fast)
        output_tensor = _augmentor.apply_preset(
            input_tensor, 
            request.preset_idx, 
            request.seed or 42
        )

        return ImageEditResponse(
            image_tensor_b64=tensor_to_base64(output_tensor),
            success=True,
            message="OK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/augment_batch", response_model=BatchAugmentResponse)
async def augment_batch(request: BatchAugmentRequest):
    """Generate multiple torchvision augmentations from single image."""
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)
        seed = request.seed or 42
        
        output_tensors = []
        for i in range(request.n_augment):
            aug_tensor = _augmentor.apply_preset(
                input_tensor,
                i % 9,  # 9 presets available
                seed + i * 1000
            )
            output_tensors.append(tensor_to_base64(aug_tensor))

        return BatchAugmentResponse(
            image_tensors_b64=output_tensors,
            success=True,
            message="OK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit_combined", response_model=CombinedEditResponse)
async def edit_combined(request: CombinedEditRequest):
    """
    Combined processing: 1 Qwen edit + (N-1) torchvision augmentations.
    
    Returns N images:
    - Index 0: Qwen-edited image
    - Index 1 to N-1: TorchVision augmented images
    """
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)
        seed = request.seed or 42
        
        output_tensors = []
        
        # 1. Qwen edit (GPU-based)
        qwen_output = await manager.submit_qwen_job(
            input_tensor=input_tensor,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            task=request.task,
            seed=seed,
        )
        output_tensors.append(tensor_to_base64(qwen_output))
        
        # 2. TorchVision augmentations (CPU-based)
        for i in range(1, request.n_augment):
            aug_tensor = _augmentor.apply_preset(
                input_tensor,
                (i - 1) % 9,  # preset 0-8
                seed + i * 1000
            )
            output_tensors.append(tensor_to_base64(aug_tensor))

        return CombinedEditResponse(
            image_tensors_b64=output_tensors,
            success=True,
            message="OK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ready": manager.is_ready,
        "num_gpus": manager.num_gpus,
        "processes_alive": sum(1 for p in manager.processes if p.is_alive()),
        "augmentation_presets": len(_augmentor.augmentation_presets),
    }


@app.get("/presets")
async def list_presets():
    """List available augmentation presets."""
    return {
        "presets": [
            {"index": i, "transforms": preset}
            for i, preset in enumerate(_augmentor.augmentation_presets)
        ],
        "total": len(_augmentor.augmentation_presets),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11303, workers=1)