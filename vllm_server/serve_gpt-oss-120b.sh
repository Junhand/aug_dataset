#!/bin/bash
#SBATCH --job-name=vllm-gpt-oss
#SBATCH -p part-group_25b505 
#SBATCH --nodes=1          
#SBATCH --nodelist=aic-gh2b-310033
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

set -euo pipefail

# ---- modules / conda ----
module load cuda/12.8
source /home/user_00050_25b505/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.vllm_env
set -u

# ---- model config ----
MODEL="${MODEL:-openai/gpt-oss-120b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting vLLM server..."
echo "Model=${MODEL}"
echo "Port=${PORT}"

# H200×4・高スループット初期値
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# Hopper向け公式レシピ（vLLM docs 準拠）
CONFIG_YAML="GPT-OSS_Hopper.yaml"
cat > "${CONFIG_YAML}" <<YAML
async-scheduling: true
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: ${MAX_NUM_BATCHED_TOKENS}
stream-interval: 20
YAML

# Hopper + vLLM v0.12 系の小concurrency退行回避（必要なら有効）
# export VLLM_MXFP4_USE_MARLIN=1

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --config "${CONFIG_YAML}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}"