#! /bin/bash

#dataset path as arg
data_dir=$1

docker run --tty \
    -d \
    --gpus all \
    --interactive \
    --volume $data_dir/v1.0-trainval:/root/cr3dt/data/nuscenes \
    --volume $data_dir/v1.0-mini:/root/cr3dt/data/nuscenes_mini \
    --volume $data_dir/v1.0-test:/root/cr3dt/data/nuscenes_test \
    --volume $data_dir/checkpoints:/root/cr3dt/checkpoints \
    --ipc=host\
    --network=host \
    --privileged \
    --name main_cr3dt \
    cr3dt_detector \
    /bin/bash
