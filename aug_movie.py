import os
import av
import uuid
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)


def edit_image(frame, noise_level=30):
    tmp_image_name = f".cache/output_{uuid.uuid4()}.png"
    Image.fromarray(frame).save(tmp_image_name)
    #image1 = Image.open(tmp_image_name)
    image1 = Image.open(tmp_image_name).convert("RGB")

    prompt = "右から強い光が当たっている様子にして。"

    inputs = {
        "image": [image1],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("output_image_edit_2511.png")
        print("image saved at", os.path.abspath("output_image_edit_2511.png"))

    return np.clip(output_image, 0, 255).astype(np.uint8)


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
    i = 0
    for frame in tqdm(input_container.decode(video=0)):
        i = i+1
        print(f"{i}がスタート", flush=True)
        # 1. RGBのnumpy配列に変換
        img = frame.to_ndarray(format='rgb24')
        
        # 2. 加工処理（ノイズ付加）
        processed_img = edit_image(img, noise_level=noise_intensity)
        print(processed_img, flush=True)
        
        # 3. 再びVideoFrameオブジェクトに戻して出力
        new_frame = av.VideoFrame.from_ndarray(processed_img, format='rgb24')
        for packet in output_video.encode(new_frame):
            output_container.mux(packet)
        
        if i>10:
            break

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