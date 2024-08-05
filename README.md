# BEVDet Fork for CR3DT Detector

This is the official code implementation to the paper [CR3DT: Camera-Radar 3D Tracking with Multi-Modal Fusion](https://arxiv.org/abs/2403.15313). 

## Introduction
The tracking is based on [Quasi-Dense Similarity Learning for Appearance-Only Multiple Object Tracking](https://arxiv.org/pdf/2210.06984.pdf) [github](https://github.com/SysCV/qdtrack)

## Get Started

#### Installation and Data Preparation

Step 1. Use the provided docker file to create the needed container [Docker](docker/Dockerfile).

Step 2. Download the dataset and save it in the folder ..., download the checkpoints from ... and the pkl files from (only works if the volume mount point within the container is unchanged)

Step 3. Start the docker container with the necessary flags using the provided utility script
```shell script
docker/main_docker.sh
```

Step 4. Finalize the setup of the conda environment
```shell script
conda activate bevdet_docker
pip install -v -e .
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Step 5. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for CR3DT by running (if not mounted externally):
```shell script
python tools/create_data_bevdet.py
```


**Note**: this package requires the BEV boxes to be in the format `[x_center, y_center, w, h, yaw]`. The `LiDARInstance3DBoxes` from `mmdet3d` are with "bottom center" origin.

#### Train model
```shell
# single gpu
python tools/train.py $config
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
```

#### Estimate the inference speed of BEVDet

```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py $config $checkpoint --fuse-conv-bn
```

#### Estimate the flops of BEVDet

```shell
python tools/analysis_tools/get_flops.py configs/bevdet/bevdet-r50.py --shape 256 704
```

#### Visualize the predicted result.

- Private implementation. (Visualization remotely/locally)

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Beside, there are some other attractive works extend the boundary of BEVDet.

- [BEVerse](https://github.com/zhangyp15/BEVerse)  for multi-task learning.
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)  for stero depth estimation.

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{baumann2024cr3dt,
  title={CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking},
  author={Baumann, Nicolas and Baumgartner, Michael and Ghignone, Edoardo and K{\"u}hne, Jonas and Fischer, Tobias and Yang, Yung-Hsu and Pollefeys, Marc and Magno, Michele},
  journal={arXiv preprint arXiv:2403.15313},
  year={2024}
}
```