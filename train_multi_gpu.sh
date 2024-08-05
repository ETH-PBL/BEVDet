#!/bin/bash

#SBATCH  --gres=gpu:2
#SBATCH  --constraint='geforce_rtx_2080_ti|titan_xp'
#SBATCH  --output=sbatch_logs/bevdet_train_%j.out
#SBATCH  --time=48:00:00
#SBATCH  --mem=45G
#SBATCH  --cpus-per-task=4


SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

./tools/dist_train.sh configs/ca3dt/frankennet-r50-4d-depth-cbgs.py 2
