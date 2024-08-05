#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu.debug
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=sbatch_logs/jupyter.out

SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

cat /etc/hosts
jupyter notebook --ip=0.0.0.0 --port=8888