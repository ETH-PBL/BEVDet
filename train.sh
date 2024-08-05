#!/bin/bash

#SBATCH  --gres=gpu:1
#SBATCH  --constraint='geforce_rtx_2080_ti|titan_xp|geforce_gtx_1080_ti'
#SBATCH  --output=sbatch_logs/bevdet_train_%j.out
#SBATCH  --time=48:00:00
#SBATCH  --mem=40G


SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

# python tools/train.py configs/ca3dt/frankennet-r50-qd-track.py --validate --wandb "run" --wandbnotes "Smaller LR and eval plots"
python tools/train.py configs/ca3dt/frankennet-r50.py --validate --wandb "run" --wandbnotes "Ablation 1: BEV Compression True, Late Fusion False" --resume-from work_dirs/frankennet-r50/latest.pth
# python tools/train.py configs/ca3dt/frankennet-r50-norad-qd-track.py --validate --wandb "online" --wandbnotes "Testing No radar emb training" --load-from work_dirs/frankennet-r50/epoch_48.pth
