#!/bin/bash

#SBATCH  --gres=gpu:1

#SBATCH  --constraint='geforce_gtx_1080_ti'

source /scratch_net/biwidl216/rl_course_13/conda/etc/profile.d/conda.sh
conda activate bevdet_docker
#python setup.py develop
#pip install pytorch
pip install -v -e .
