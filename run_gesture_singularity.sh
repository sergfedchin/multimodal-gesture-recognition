#!/bin/bash
#SBATCH --job-name=gesture
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=/home/daryumin/HandGestures_experiments/singularity/logs/gesture_%j.log

module load singularity

singularity run \
  --nv \
  --env GESTURE_NAME=grabbing \
  --bind /home/daryumin/Corpora/HaGRIDv2.0:/app/data/gestures:ro \
  --bind /home/daryumin/HandGestures_experiments/singularity/processed:/app/data/processed:rw \
  /home/daryumin/HandGestures_experiments/singularity/gesture.sif
