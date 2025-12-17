#!/bin/bash
#SBATCH --job-name=gesture_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --output=/home/daryumin/HandGestures_experiments/experiments/logs/train_%j.log
#SBATCH --time=2-0
#SBATCH --constraint="type_e"

module load Python/PyTorch_GPU_v2.4

cd /home/daryumin/HandGestures_experiments/experiments
python train.py --config config.toml


