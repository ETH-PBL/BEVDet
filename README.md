# BEVDet Fork for CR3DT Detector
This is the official code implementation to the paper [CR3DT: Camera-Radar 3D Tracking with Multi-Modal Fusion](https://arxiv.org/abs/2403.15313). 


## Get Started

#### Installation and Data Preparation

Step 1. Clone this repository.
```shell script
git clone git@github.com:ETH-PBL/CR3DT.git
```

Step 2. Use the provided docker file to create the needed container [Docker](docker/Dockerfile).
```shell script
docker build -t cr3dt_detector -f docker/Dockerfile .
```

Step 3. Create the folder structure below anywhere on your file system. You can chose to populate the folders with the nuScenes dataset and our provided pkl-files and checkpoints ([Google Drive with checkpoint and pkls](https://drive.google.com/drive/folders/1gHPZMUCDObDTHqbU_7Drw0CILx4pu_7i)), or just with the dataset and to create any pkl-files and checkpoints yourself. At least one of the three dataset folders (`v1.0-mini`, `v1.0-trainval`, or `v1.0-test`) needs to be populated.
```shell script
...
├── <your data directory>
│   ├── v1.0-mini
│   ├── v1.0-trainval
│   ├── v1.0-test
│   └── checkpoints
└ ...
```

Step 4. Start the docker container with the necessary flags using the provided utility script. After that you can open a second interactive shell to the docker using `sec_docker.sh`.
```shell script
./docker/main_docker.sh <path to your data directory>
./docker/sec_docker.sh
```

Step 5. Only needed once: Finalize the setup of the conda environment
```shell script
conda activate cr3dt_docker
pip install -v -e .
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Step 6. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl (if not mounted externally) for CR3DT by running the following script with the appropriate argument (`trainval`, `mini`, or `test`):
```shell script
./tools/create_data.sh trainval # mini, test
```
**Note**: this package requires the BEV boxes to be in the format `[x_center, y_center, w, h, yaw]`. The `LiDARInstance3DBoxes` from `mmdet3d` are with "bottom center" origin.

**Note**: To train or evaluate on the mini dataset, [the `cr3dt-r50.py` config](configs/cr3dt/cr3dt-r50.py#L258) needs to be adapted accordingly (see comment in file).

Step 7: To run the tracking evaluations follow the instructions on [CC-3DT++](https://github.com/ETH-PBL/cc-3dt-pp).

#### Train model
```shell
python tools/train.py configs/cr3dt/cr3dt-r50.py --validate
# or from the provided checkpoint 
python tools/train.py configs/cr3dt/cr3dt-r50.py --validate --resume-from checkpoints/cr3dt.pth
```

**Note**: See `python tools/train.py --help` for more options.

#### Test model
```shell
# single gpu
python tools/test.py configs/cr3dt/cr3dt-r50.py checkpoints/cr3dt.pth --eval mAP
```

#### Estimate the inference speed of CR3DT

```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py configs/cr3dt/cr3dt-r50.py checkpoints/cr3dt.pth --fuse-conv-bn
```

#### Estimate the flops of CR3DT

```shell
python tools/analysis_tools/get_flops.py configs/cr3dt/cr3dt-r50.py --shape 256 704
```

#### Visualize the predicted result.

- Implementation of original [BEVDet](https://github.com/HuangJunJie2017/BEVDet).

```shell
python tools/test.py configs/cr3dt/cr3dt-r50.py checkpoints/cr3dt.pth --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

**Note**: For mini add the flag: `--root_path ./data/nuscenes_mini`.

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@article{baumann2024cr3dt,
  title={CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking},
  author={Baumann, Nicolas and Baumgartner, Michael and Ghignone, Edoardo and K{\"u}hne, Jonas and Fischer, Tobias and Yang, Yung-Hsu and Pollefeys, Marc and Magno, Michele},
  journal={arXiv preprint arXiv:2403.15313},
  year={2024}
}
```
