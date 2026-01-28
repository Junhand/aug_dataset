"""
Qwen-Image-Edit API Server with SAM3 Segmentation
Preserves task-relevant objects (robot, shelf) and edits only the background.
"""

import io
import math
import base64
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Configuration
MAX_QUEUE_SIZE = 1000
REQUEST_TIMEOUT = 300  # seconds

# Model paths
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_PATH = "/home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"

# Inference settings
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 1.0


def tensor_to_base64(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return torch.load(buffer, weights_only=True)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert CHW tensor [0,1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    tensor = tensor.clamp(0, 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to CHW tensor [0,1]."""
    np_img = np.array(img.convert("RGB"))
    return torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0


class SAM3Segmenter:
    """SAM3-based segmentation for task-relevant objects."""

    def __init__(self):
        self.model = None
        self.processor = None

    def initialize(self):
        """Initialize SAM3 model."""
        print("Initializing SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        print("SAM3 initialized!")

    def _get_mask_or_empty(self, output) -> Optional[torch.Tensor]:
        """Extract mask from SAM3 output."""
        masks = output["masks"]  # (N,1,H,W) bool
        if masks.numel() == 0:
            return None
        return masks.any(dim=0).squeeze(0)  # (H,W)

    def segment_task_objects(
        self,
        image: Image.Image,
        task: str,
    ) -> Optional[torch.Tensor]:
        """
        Segment task-relevant objects (robot, shelf, etc.)

        Returns:
            mask: (H, W) bool tensor where True = keep (don't edit)
        """
        state = self.processor.set_image(image)

        # Segment shelf
        out_shelf = self.processor.set_text_prompt(state=state, prompt="shelf")
        shelf_mask = self._get_mask_or_empty(out_shelf)

        # Segment robot (try multiple prompts)
        robot_mask = None
        for prompt in [
            "robot",
            "robot arm",
            "robot base",
            "mobile robot",
            "gripper",
            "robot body",
        ]:
            out_r = self.processor.set_text_prompt(state=state, prompt=prompt)
            m = self._get_mask_or_empty(out_r)
            if m is not None:
                robot_mask = m
                break

        # Segment manipulated objects based on task keywords
        object_mask = None
        task_lower = task.lower()
        object_prompts = []

        # Extract potential objects from task
        if "toaster" in task_lower:
            object_prompts.append("toaster")
        if "oven" in task_lower:
            object_prompts.append("oven")
        if "drawer" in task_lower:
            object_prompts.append("drawer")
        if "door" in task_lower:
            object_prompts.append("door")
        if "cup" in task_lower:
            object_prompts.append("cup")
        if "bottle" in task_lower:
            object_prompts.append("bottle")

        for prompt in object_prompts:
            out_obj = self.processor.set_text_prompt(state=state, prompt=prompt)
            m = self._get_mask_or_empty(out_obj)
            if m is not None:
                if object_mask is None:
                    object_mask = m
                else:
                    object_mask = object_mask | m

        # Merge all masks
        masks = [m for m in [shelf_mask, robot_mask, object_mask] if m is not None]

        if not masks:
            return None

        merged = masks[0]
        for m in masks[1:]:
            merged = merged | m

        return merged  # (H, W) bool tensor


def composite_images(
    original: Image.Image,
    edited: Image.Image,
    mask: torch.Tensor,
) -> Image.Image:
    """
    Composite original and edited images using mask.

    Args:
        original: Original PIL image
        edited: Edited PIL image (may be different size)
        mask: (H, W) bool tensor where True = keep original

    Returns:
        Composited PIL image (same size as original)
    """
    original_np = np.array(original)
    h, w = original_np.shape[:2]

    # Resize edited to match original size
    if edited.size != (w, h):
        edited = edited.resize((w, h), Image.LANCZOS)
    edited_np = np.array(edited)

    # Resize mask if needed
    mask_np = mask.cpu().numpy()
    if mask_np.shape != (h, w):
        mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_pil) > 127

    # Composite: keep original where mask is True, use edited elsewhere
    result = edited_np.copy()
    result[mask_np] = original_np[mask_np]

    return Image.fromarray(result)


@dataclass
class EditJob:
    """Job for processing queue."""

    input_pil: Image.Image
    original_size: Tuple[int, int]  # (H, W)
    prompt: str
    negative_prompt: str
    task: str  # For SAM3 segmentation
    seed: int
    future: asyncio.Future


class PipelineManager:
    """Pipeline manager with SAM3 + Qwen."""

    def __init__(self):
        self.pipe = None
        self.segmenter = None
        self.device = None
        self.queue: asyncio.Queue[EditJob] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._lock = asyncio.Lock()
        self._worker_task = None

    def initialize(self):
        """Initialize both SAM3 and Qwen pipelines."""
        # Initialize SAM3
        self.segmenter = SAM3Segmenter()
        self.segmenter.initialize()

        # Initialize Qwen
        print("Initializing Qwen-Image-Edit pipeline (diffusers)...")

        if torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16
            self.device = "cuda"
        else:
            self.torch_dtype = torch.float32
            self.device = "cpu"

        model = QwenImageTransformer2DModel.from_pretrained(
            MODEL_NAME, subfolder="transformer", torch_dtype=self.torch_dtype
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

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            transformer=model,
            scheduler=scheduler,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.load_lora_weights(LORA_PATH)
        self.pipe = self.pipe.to(self.device)

        print(f"Pipeline initialized on {self.device}!")

    async def start_worker(self):
        self._worker_task = asyncio.create_task(self._process_queue())

    async def stop_worker(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _process_queue(self):
        while True:
            job = await self.queue.get()
            try:
                async with self._lock:
                    output_tensor = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_inference, job
                    )
                    job.future.set_result(output_tensor)
            except Exception as e:
                job.future.set_exception(e)
            finally:
                self.queue.task_done()

    def _run_inference(self, job: EditJob) -> torch.Tensor:
        """Run SAM3 segmentation + Qwen edit + composite."""
        try:
            # Step 1: Segment task-relevant objects with SAM3
            mask = self.segmenter.segment_task_objects(job.input_pil, job.task)

            # Step 2: Run Qwen image editing
            generator = torch.Generator(device=self.device).manual_seed(job.seed)

            result = self.pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                image=[job.input_pil],
                generator=generator,
                num_inference_steps=NUM_INFERENCE_STEPS,
                true_cfg_scale=TRUE_CFG_SCALE,
            )
            edited_pil = result.images[0]

            # Step 3: Composite - keep original where mask is True
            if mask is not None:
                output_pil = composite_images(job.input_pil, edited_pil, mask)
            else:
                # No mask found, resize edited to original size
                original_h, original_w = job.original_size
                if edited_pil.size != (original_w, original_h):
                    output_pil = edited_pil.resize(
                        (original_w, original_h), Image.LANCZOS
                    )
                else:
                    output_pil = edited_pil

            return pil_to_tensor(output_pil)

        except Exception as e:
            import traceback

            print(f"Inference error: {e}")
            traceback.print_exc()
            raise

    async def submit_job(
        self,
        input_tensor: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        task: str,
        seed: int,
    ) -> torch.Tensor:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        original_h, original_w = input_tensor.shape[1], input_tensor.shape[2]
        input_pil = tensor_to_pil(input_tensor)

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        job = EditJob(
            input_pil=input_pil,
            original_size=(original_h, original_w),
            prompt=prompt,
            negative_prompt=negative_prompt,
            task=task,
            seed=seed,
            future=future,
        )

        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Server overloaded")

        try:
            return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None and self.segmenter is not None


manager = PipelineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.initialize()
    await manager.start_worker()
    yield
    await manager.stop_worker()


app = FastAPI(
    title="Qwen-Image-Edit API with SAM3",
    version="4.0.0",
    lifespan=lifespan,
)


class ImageEditRequest(BaseModel):
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor")
    prompt: str
    task: str = Field(..., description="Task description for SAM3 segmentation")
    negative_prompt: Optional[str] = Field(default="")
    seed: Optional[int] = Field(default=42)


class ImageEditResponse(BaseModel):
    image_tensor_b64: str
    success: bool = True
    message: str = "OK"


@app.post("/edit", response_model=ImageEditResponse)
async def edit_image(request: ImageEditRequest):
    """Edit background while preserving task-relevant objects."""
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)

        output_tensor = await manager.submit_job(
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ready": manager.is_ready,
        "queue_size": manager.queue_size,
        "max_queue": MAX_QUEUE_SIZE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11303, workers=1)
