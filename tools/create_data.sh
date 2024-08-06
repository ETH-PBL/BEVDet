#!/usr/bin/env bash

# choose trainval, mini or test
option=$1

[[ -f ${CONDA_EXE} ]] && eval "$(${CONDA_EXE} shell.bash hook)"
conda activate cr3dt_docker

if [ "$option" == "trainval" ]; then
    python tools/create_data_bevdet.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag bevdetv2-nuscenes
elif [ "$option" == "mini" ]; then
    python tools/create_data_bevdet.py nuscenes --root-path ./data/nuscenes_mini --out-dir ./data/nuscenes_mini --extra-tag bevdetv2-nuscenes --version v1.0-mini
elif [ "$option" == "test" ]; then
    python tools/create_data_bevdet.py nuscenes --root-path ./data/nuscenes_test --out-dir ./data/nuscenes_test --extra-tag bevdetv2-nuscenes --version v1.0-test
else
    echo "Invalid option"
fi