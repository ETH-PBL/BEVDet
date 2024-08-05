#!/bin/bash

#SBATCH  --gres=gpu:2
#SBATCH  --constraint='geforce_rtx_2080_ti|geforce_gtx_1080_ti'
#SBATCH  --output=sbatch_logs/bevdet_train_%j.out
#SBATCH  --time=48:00:00


SCRATCH_PATH=${CONDA_EXE%/conda/bin/conda}

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_docker

set -x

CONFIG="configs/ca3dt/frankennet-r50.py --validate"
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}