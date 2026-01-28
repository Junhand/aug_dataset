#!/bin/bash
#SBATCH --job-name=api_image_edit
#SBATCH -p part-group_25b505 
#SBATCH --nodes=1          
#SBATCH --nodelist=aic-gh2b-310034
#SBATCH --gpus-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/image-edit_%j.out
#SBATCH --error=logs/image-edit_%j.err

set -euo pipefail

# ---- modules / conda ----
module load cuda/12.8
source /home/user_00050_25b505/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.edit_env
set -u

# ---- model config ----
MODEL="Qwen-Image-Edit-2511"
HOST="0.0.0.0"
PORT="11303"

echo "Starting Image Edit server..."
echo "Model=${MODEL}"
echo "Port=${PORT}"

# ---- ★ここが本質：foreground 実行 ----
exec python vllm_server/qwen_sam3_server_multi_gpu.py