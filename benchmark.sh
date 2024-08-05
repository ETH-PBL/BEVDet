#!/bin/bash

#SBATCH  --gres=gpu:1
#SBATCH  --constraint='geforce_rtx_2080_ti|geforce_gtx_1080_ti'
#SBATCH  --output=sbatch_logs/bevdet_eval_%j.out


SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

# Change the checkpoints path to the path of the checkpoint you want to evaluate probably: work_dirs/bevdet-r50/latest.pth or checkpoints/bevdet-r50.pth

#CR3DT
echo "CR3DT BENCHMARKING:"
python tools/analysis_tools/benchmark.py configs/ca3dt/frankennet-r50.py work_dirs/frankennet-r50/artifacts/run_mniavefs_model:v6/epoch_40.pth --fuse-conv-bn

#BEVDET
echo "BEVDET BENCHMARKING:"
python tools/analysis_tools/benchmark.py configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/artifacts/run_n14i4gso_model:v11/epoch_57.pth --fuse-conv-bn