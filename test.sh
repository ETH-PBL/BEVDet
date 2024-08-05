#!/bin/bash

#SBATCH  --gres=gpu:1
#SBATCH  --constraint='geforce_rtx_2080_ti|geforce_gtx_1080_ti'
#SBATCH  --output=sbatch_logs/bevdet_eval_%j.out


SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

# Change the checkpoints path to the path of the checkpoint you want to evaluate probably: work_dirs/bevdet-r50/latest.pth or checkpoints/bevdet-r50.pth

# python tools/test.py configs/ca3dt/frankennet-r50_test_set.py work_dirs/frankennet-r50/latest.pth --format-only --eval-options jsonfile_prefix=nusc_eval_dir

python tools/test.py configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50/artifacts/run_n14i4gso_model:v11/epoch_57.pth --format-only --eval-options jsonfile_prefix=nusc_eval_dir

# python tools/analysis_tools/vis.py nusc_eval_dir/pts_bbox/results_nusc.json --root_path /scratch_net/friday/rl_course_29/bevdet/data/nuscenes_pkl/
