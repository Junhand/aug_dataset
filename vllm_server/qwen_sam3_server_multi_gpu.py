"""
Qwen-Image-Edit API Server with SAM3 Segmentation
(8 GPU Multiprocess)

v6: Quantized model (4-bit) + Lightning LoRA from HuggingFace
- Uses PipelineQuantizationConfig for 4-bit quantization
- Loads LoRA from 'lightx2v/Qwen-Image-Edit-2511-Lightning'
- /edit endpoint: 1 image in -> 1 image out
"""

import io
import os
import math
import base64
import asyncio
import multiprocessing as mp
from typing import Optional, Tuple
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Configuration
MAX_QUEUE_SIZE = 1000
REQUEST_TIMEOUT = 3000
NUM_GPUS = 8
JPEG_QUALITY = 95

# Model paths
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHT_NAME = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# Inference settings
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0


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
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    return torch.clamp(tensor, 0.0, 1.0)


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

    if overlay_alpha > 0:
        red = np.zeros_like(result)
        red[..., 0] = 255
        result[mask_np] = (1.0 - overlay_alpha) * result[mask_np] + overlay_alpha * red[mask_np]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def gpu_worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, ready_event: mp.Event, pending_jobs: mp.Array):
    """Worker process for a single GPU with quantized model."""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.quantizers import PipelineQuantizationConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda:0"

    print(f"[Worker {gpu_id}] Using physical GPU {gpu_id}, visible as cuda:0")
    print(f"[Worker {gpu_id}] Initializing models with 4-bit quantization...")

    try:
        # Initialize SAM3
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)

        # Quantization config for 4-bit
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                ),
                "text_encoder": TransformersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            }
        )

        # Scheduler config
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

        # Load pipeline with quantization
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            quantization_config=pipeline_quant_config,
            scheduler=scheduler,
        )

        # Load LoRA weights from HuggingFace
        pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
            adapter_name="lightning",
        )

        # Move to GPU (quantized models handle this differently)
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

            job_id, image_bytes, original_size, prompt, negative_prompt, task, seed, overlay_alpha = job

            input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask = segment_task_objects(input_pil, task)

            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Inference with quantized model
            with torch.inference_mode():
                result = pipe(
                    image=[input_pil],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    true_cfg_scale=TRUE_CFG_SCALE,
                    guidance_scale=GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                )
            edited_pil = result.images[0]

            if mask is not None:
                output_pil = composite_images(input_pil, edited_pil, mask, overlay_alpha=overlay_alpha)
            else:
                h, w = original_size
                if edited_pil.size != (w, h):
                    output_pil = edited_pil.resize((w, h), Image.LANCZOS)
                else:
                    output_pil = edited_pil

            output_buffer = io.BytesIO()
            output_pil.save(output_buffer, format="JPEG", quality=JPEG_QUALITY)
            output_bytes = output_buffer.getvalue()

            # Decrement pending jobs counter
            pending_jobs[gpu_id] -= 1
            
            result_queue.put((job_id, output_bytes, None))
            print(f"[Worker {gpu_id}] Job {job_id} completed (pending: {pending_jobs[gpu_id]})")

        except Exception as e:
            import traceback
            print(f"[Worker {gpu_id}] Error: {e}")
            traceback.print_exc()
            # Decrement pending jobs counter on error too
            pending_jobs[gpu_id] -= 1
            result_queue.put((job_id, None, str(e)))


class MultiGPUManager:
    """Manager for GPU worker processes with dynamic load balancing."""

    def __init__(self, num_gpus: int = NUM_GPUS):
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.ready_events = []
        self.job_futures = {}
        self.job_counter = 0
        self._lock = asyncio.Lock()
        self._result_worker_task = None
        
        # Track pending jobs per GPU for dynamic load balancing
        self.pending_jobs = None  # Will be initialized in initialize()
        self._ctx = None  # Multiprocessing context

    def initialize(self):
        print(f"Detected {torch.cuda.device_count()} GPUs, using {self.num_gpus}")
        print(f"Starting {self.num_gpus} GPU worker processes with 4-bit quantization...")

        # Use spawn context for everything
        self._ctx = mp.get_context('spawn')
        self.result_queue = self._ctx.Queue()
        
        # Shared array to track pending jobs per GPU (created in spawn context)
        self.pending_jobs = self._ctx.Array('i', [0] * self.num_gpus)

        for gpu_id in range(self.num_gpus):
            task_queue = self._ctx.Queue()
            ready_event = self._ctx.Event()
            
            self.task_queues.append(task_queue)
            self.ready_events.append(ready_event)

            p = self._ctx.Process(
                target=gpu_worker_process,
                args=(gpu_id, task_queue, self.result_queue, ready_event, self.pending_jobs),
            )
            p.start()
            self.processes.append(p)
            print(f"[Main] Started worker process for GPU {gpu_id} (PID: {p.pid})")

        print("[Main] Waiting for all workers to initialize...")
        for i, event in enumerate(self.ready_events):
            event.wait(timeout=600)  # Longer timeout for quantized model loading
            print(f"[Main] Worker {i} is ready")

        print(f"All {self.num_gpus} worker processes are ready!")

    def _get_least_loaded_gpu(self) -> int:
        """Get GPU with fewest pending jobs (dynamic load balancing)."""
        pending = [self.pending_jobs[i] for i in range(self.num_gpus)]
        return pending.index(min(pending))

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

    async def submit_job(
        self,
        input_tensor: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        task: str,
        seed: int,
        overlay_alpha: float = 0.0,
    ) -> torch.Tensor:
        """Submit an edit job to the least loaded GPU worker."""
        async with self._lock:
            job_id = self.job_counter
            self.job_counter += 1
            # Dynamic load balancing: pick GPU with fewest pending jobs
            gpu_id = self._get_least_loaded_gpu()
            self.pending_jobs[gpu_id] += 1

        original_h, original_w = input_tensor.shape[1], input_tensor.shape[2]

        input_pil = tensor_to_pil(input_tensor)
        buffer = io.BytesIO()
        input_pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        image_bytes = buffer.getvalue()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.job_futures[job_id] = future

        job = (job_id, image_bytes, (original_h, original_w), prompt, negative_prompt, task, seed, overlay_alpha)
        self.task_queues[gpu_id].put(job)
        
        # Log with pending job counts
        pending_str = ",".join(str(self.pending_jobs[i]) for i in range(self.num_gpus))
        print(f"[Main] Job {job_id} -> GPU {gpu_id} (pending: [{pending_str}])")

        try:
            output_bytes = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            self.job_futures.pop(job_id, None)
            self.pending_jobs[gpu_id] -= 1
            raise HTTPException(status_code=504, detail="Request timeout")

        return pil_to_tensor(Image.open(io.BytesIO(output_bytes)).convert("RGB"))

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
    title="Qwen-Image-Edit API with SAM3 (8 GPU, 4-bit Quantized)",
    description="SAM3 segmentation + Qwen-Image-Edit with 4-bit quantization and Lightning LoRA.",
    version="6.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request/Response Models
# ============================================================

class ImageEditRequest(BaseModel):
    """Request for Qwen image editing with SAM3 segmentation."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor [0,1]")
    prompt: str = Field(..., description="Edit prompt")
    task: str = Field(..., description="Task description for SAM3 segmentation (comma-separated objects)")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    seed: Optional[int] = Field(default=42, description="Random seed")
    overlay_alpha: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Overlay alpha for mask visualization (0.0 = no overlay, 1.0 = full red)")


class ImageEditResponse(BaseModel):
    """Response for image editing."""
    image_tensor_b64: str = Field(..., description="Base64 encoded CHW tensor [0,1]")
    success: bool = True
    message: str = "OK"


# ============================================================
# API Endpoints
# ============================================================

@app.post("/edit", response_model=ImageEditResponse)
async def edit_image(request: ImageEditRequest):
    """
    Edit image using SAM3 segmentation + Qwen-Image-Edit (4-bit quantized).
    
    - SAM3 segments robot and task objects to create a mask
    - Qwen edits the background while preserving masked regions
    - Returns 1 edited image
    """
    try:
        input_tensor = base64_to_tensor(request.image_tensor_b64)

        output_tensor = await manager.submit_job(
            input_tensor=input_tensor,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            task=request.task,
            seed=request.seed or 42,
            overlay_alpha=request.overlay_alpha or 0.0,
        )

        return ImageEditResponse(
            image_tensor_b64=tensor_to_base64(output_tensor),
            success=True,
            message="OK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    pending = [manager.pending_jobs[i] for i in range(manager.num_gpus)] if manager.pending_jobs else []
    return {
        "status": "healthy",
        "ready": manager.is_ready,
        "num_gpus": manager.num_gpus,
        "processes_alive": sum(1 for p in manager.processes if p.is_alive()),
        "pending_jobs_per_gpu": pending,
        "total_pending_jobs": sum(pending),
        "quantization": "4-bit (nf4)",
        "lora": LORA_WEIGHT_NAME,
        "load_balancing": "dynamic (least loaded)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11303, workers=1)