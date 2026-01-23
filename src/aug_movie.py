import os
import uuid
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from tqdm import tqdm


class QwenImageEditor:
    """
    QwenImageEditPlusPipeline wrapper.
    Input : numpy image (H, W, C, uint8)
    Output: numpy image (H, W, C, uint8)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str = ".cache",
        seed: int = 0,
        disable_progress_bar: bool | None = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)

        self.device = device
        self.cache_dir = cache_dir

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=disable_progress_bar)

        # per-instance RNG (avoid global torch.manual_seed)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def edit_image(
        self,
        frame: np.ndarray,
        prompt: str = "右から強い光が当たっている様子にして。",
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        save_path: str | None = "output_image_edit_2511.png",
    ) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("frame must be HWC with 3 (RGB) or 4 (RGBA) channels")

        tmp_image_path = os.path.join(self.cache_dir, f"input_{uuid.uuid4()}.png")
        Image.fromarray(frame).save(tmp_image_path)

        image = Image.open(tmp_image_path).convert("RGB")

        inputs = {
            "image": [image],
            "prompt": prompt,
            "generator": self.generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]

        if save_path is not None:
            output_image.save(save_path)
            print("image saved at", os.path.abspath(save_path))

        # PIL -> numpy (uint8)
        out = np.asarray(output_image)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def __call__(self, frame: np.ndarray, prompt: str) -> np.ndarray:
        return self.edit_image(frame=frame, prompt=prompt)


def edit_image_noise(frame, noise_level=30):
    """
    画像にガウスノイズを加える（Image2Imageの強度調整に近い処理）
    """
    noise = np.random.randn(*frame.shape) * noise_level
    noisy_frame = frame + noise
    # 値を0-255の範囲にクリップしてuint8に変換
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)


def process_video_robust(input_path, output_path, noise_intensity=25):
    # 入力ファイルを開く
    input_container = av.open(input_path)
    input_video = input_container.streams.video[0]

    # 出力ファイルの設定 (H.264 / yuv420p は互換性が非常に高い)
    output_container = av.open(output_path, mode="w")
    output_video = output_container.add_stream("libx264", rate=10)  # fps=10
    output_video.width = input_video.width
    output_video.height = input_video.height
    output_video.pix_fmt = "yuv420p"

    print(f"Processing: {input_path} -> {output_path}")

    # フレーム処理
    import pdb

    pdb.set_trace()
    for frame in tqdm(input_container.decode(video=0)):
        # 1. RGBのnumpy配列に変換
        img = frame.to_ndarray(format="rgb24")

        # 2. 加工処理（ノイズ付加）
        # editor = QwenImageEditor(
        #     model_id="Qwen/Qwen-Image-Edit-2511",
        #     torch_dtype=torch.bfloat16,
        # )

        # processed_img = editor.edit_image(
        #     frame,
        #     prompt="右から強い光が当たっている様子にして。",
        # )
        processed_img = edit_image_noise(img, noise_level=noise_intensity)

        # 3. 再びVideoFrameオブジェクトに戻して出力
        new_frame = av.VideoFrame.from_ndarray(processed_img, format="rgb24")
        for packet in output_video.encode(new_frame):
            output_container.mux(packet)

    # 残りのパケットを書き出し
    for packet in output_video.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    print("完了しました。")


# 実行
if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data"
    output_path = Path(__file__).parent.parent / "output"

    input_mp4 = input_path / "episode_000000.mp4"
    output_mp4 = output_path / "episode_noisy.mp4"

    process_video_robust(input_mp4, output_mp4, noise_intensity=40)
