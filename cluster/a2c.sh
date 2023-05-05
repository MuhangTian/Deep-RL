#!/bin/bash
#SBATCH --job-name=A2C-SKIP4
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

srun python train.py --model "an_cnn" --algo "a2c" --batch_size 8 --unroll_length 200 --learning_rate 0.0001 --save_path "trained/a2c_stochastic_validation_skip4.pt" --total_frames 50_000_000