#!/bin/bash
#SBATCH --job-name=aug_data
#SBATCH -p part-group_25b505 
#SBATCH --nodes=1          
#SBATCH --nodelist=aic-gh2b-310033
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/aug_%j.out
#SBATCH --error=logs/aug_%j.err

module load cuda/12.8
source /home/user_00050_25b505/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.aug_env
set -u

export TMPDIR=/home/group_25b505/group_5/kawagoshi/synthetic_dataset/aug_dataset/.tmp

python src/augment_lerobot_dataset.py \
  --src-repo-id hsr/2025-09_task05_bin_500_v10 \
  --dst-repo-id hsr/2025-09_task05_bin_500_v10_aug_task_and_image_strong \
  --api-url http://aic-gh2b-310033:11303 \
  --n-augment 20 \
  --max-workers 32 \
  --start-episode 0 \
  --end-episode 99 \
  --finalize-interval 5\
  --offline \
  --resume
