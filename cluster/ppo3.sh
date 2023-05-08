#!/bin/bash
#SBATCH --job-name=PPO-Breakout
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=compsci-gpu
#SBATCH --exclude=linux[1-40]
#SBATCH --mem=50G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RL

srun python train.py --save_path "trained/gpu/ppo_breakout.pt" --env "ALE/Breakout-v5"