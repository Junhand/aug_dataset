import os
import av
import uuid
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
dtype = torch.bfloat16
device = "cuda"

pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to(device)


def edit_image(frame, noise_level=30):
    tmp_image_name = f".cache/output_{uuid.uuid4()}.png"
    Image.fromarray(frame).save(tmp_image_name)
    #image1 = Image.open(tmp_image_name)
    image1 = Image.open(tmp_image_name).convert("RGB")

    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    prompt = "缶の色を赤色にしてください。"

    negative_prompt = " "
    generator = torch.Generator(device=device).manual_seed(0)
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=81,
        guidance_scale=3.5,
        num_inference_steps=40,
        generator=generator,
    ).frames[0]
    export_to_video(output, "i2v_output.mp4", fps=16)

    return np.clip(output, 0, 255).astype(np.uint8)


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
    output_container = av.open(output_path, mode='w')
    output_video = output_container.add_stream('libx264', rate=10) # fps=10
    output_video.width = input_video.width
    output_video.height = input_video.height
    output_video.pix_fmt = 'yuv420p'

    print(f"Processing: {input_path} -> {output_path}")

    # フレーム処理
    for frame in tqdm(input_container.decode(video=0)):
        # 1. RGBのnumpy配列に変換
        img = frame.to_ndarray(format='rgb24')
        
        # 2. 加工処理（ノイズ付加）
        processed_img = edit_image(img, noise_level=noise_intensity)
        
        # 3. 再びVideoFrameオブジェクトに戻して出力
        new_frame = av.VideoFrame.from_ndarray(processed_img, format='rgb24')
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
    input_mp4 = "episode_000000.mp4"
    output_mp4 = "episode_noisy.mp4"
    process_video_robust(input_mp4, output_mp4, noise_intensity=40)