#! /bin/bash

docker run --tty \
    -d \
    --gpus all \
    --interactive \
    --volume /home/forzapbldesktop/Downloads/v1.0-mini:/root/bevdet/data/nuscenes \
    --volume /home/forzapbldesktop/Downloads/nuscenes_out:/root/bevdet/data/nuscenes_out \
    --ipc=host\
    --network=host \
    --privileged \
    --name main_bevdet \
    pbl/bevdet \
    /bin/bash
