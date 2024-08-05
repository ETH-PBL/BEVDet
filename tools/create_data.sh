#!/usr/bin/env bash

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate bevdet_test

# For TrainVal
python tools/create_data_bevdet.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes_out --extra-tag bevdetv2-nuscenes

# For Test
python tools/create_data_bevdet.py nuscenes --root-path ./data/nuscenes_test --out-dir ./data/nuscenes_out_test --extra-tag bevdetv2-nuscenes --version v1.0-test
