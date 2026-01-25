#!/bin/bash
#SBATCH --job-name=vllm-gpt-oss
#SBATCH -p part-group_25b505 
#SBATCH --nodes=1          
#SBATCH --nodelist=aic-gh2b-[310033-310034,310037-310038,310041-310042]
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

set -euo pipefail

# ---- modules / conda ----
module load cuda/12.8
source /home/user_00050_25b505/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.aug_env
set -u

# ---- model config ----
MODEL="openai/gpt-oss-120b"
HOST="0.0.0.0"
PORT="8000"
TP_SIZE=1   # GPU枚数に合わせる（例：8GPUなら8）

echo "Starting vLLM server..."
echo "Model=${MODEL}"
echo "Port=${PORT}"

# ---- ★ここが本質：foreground 実行 ----
exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}"
