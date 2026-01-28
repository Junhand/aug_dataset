#!/bin/bash
#SBATCH --job-name=aug_data
#SBATCH -p part-group_25b505 
#SBATCH --nodes=1          
#SBATCH --nodelist=aic-gh2b-310033
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/aug_%j.out
#SBATCH --error=logs/aug_%j.err

module load cuda/12.8
source /home/user_00050_25b505/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.aug_env
set -u

python src/augment_lerobot_dataset.py \
  --src-repo-id hsr/2025-09_task05_absolute \
  --dst-repo-id hsr/2025-09_task05_absolute_aug4 \
  --max-workers 56 \
  --offline \
  --use-batch 