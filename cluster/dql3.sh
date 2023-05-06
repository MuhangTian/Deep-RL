#!/bin/bash
#SBATCH --job-name=DQN-Pacman
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=compsci-gpu
#SBATCH --exclude=linux[1-40]
#SBATCH --mem=200G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RL

srun python train.py --save_path "trained/gpu/dqn_pacman.pt" --env "ALE/MsPacman-v5"