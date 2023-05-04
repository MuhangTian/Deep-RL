#!/bin/bash
#SBATCH --job-name=A2C
#SBATCH --time=90-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=compsci
#SBATCH --exclude=linux[1-40]
#SBATCH --mem=20G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RL

srun python train.py --model "an_cnn" --algo "a2c" --save_path "trained/a2c_skip_frame.pt"