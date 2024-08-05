# %%
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

# %%
cfg = Config.fromfile('configs/ca3dt/frankennet-r50.py')
# set multi-process settings
setup_multi_processes(cfg)

model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()
model.to('cpu')

datasets = [build_dataset(cfg.data.train)]
model.CLASSES = datasets[0].CLASSES

# %%
data_loaders = [
    build_mmdet_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=69,
        runner_type='EpochBasedRunner',
        persistent_workers=cfg.data.get('persistent_workers', False))
    for ds in datasets
]

# %%
single_data = next(data_loaders[0].__iter__())

# %%
single_data.keys()

# %%
def batch_pointcloud_to_bev(batch_points, bev_shape=(32, 128, 128), point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    batch_size = len(batch_points)
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    bev = torch.zeros((batch_size,) + bev_shape, dtype=torch.float32)

    scale_x = bev_shape[1] / (x_max - x_min)
    scale_y = bev_shape[2] / (y_max - y_min)
    scale_z = bev_shape[0] / (z_max - z_min)

    for batch_idx in range(batch_size):
        single_points = batch_points[batch_idx]
        # filter points outside range
        mask = ((single_points[:, 0] >= x_min) & (single_points[:, 0] <= x_max) & 
                (single_points[:, 1] >= y_min) & (single_points[:, 1] <= y_max) & 
                (single_points[:, 2] >= z_min) & (single_points[:, 2] <= z_max))
        single_points = single_points[mask]
        
        # if no points are left after filtering, continue
        if len(single_points) == 0:
            continue
        
        # translate points to start from (0,0,0)
        single_points[:, 0] -= x_min
        single_points[:, 1] -= y_min
        single_points[:, 2] -= z_min
        
        # rescale points to fit the bev grid
        single_points[:, 0] *= scale_x
        single_points[:, 1] *= scale_y
        single_points[:, 2] *= scale_z

        # get indices for assignment
        indices = single_points[:, :3].type(torch.long)
        illuminations = single_points[:, 3]
        
        # clamp indices to bev_shape, as some points may fall outside the last voxel due to rounding errors
        indices[:, 0] = indices[:, 0].clamp(0, bev_shape[2] - 1)
        indices[:, 1] = indices[:, 1].clamp(0, bev_shape[1] - 1)
        indices[:, 2] = indices[:, 2].clamp(0, bev_shape[0] - 1)

        # fill up the bev grid
        bev[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0]] = illuminations
    return bev


# %%
pts_feats = single_data['points'].data[0][0]
#pts_feats = torch.tensor([[0, 0, 0, 0, 0], [50, 50, 3, 0 , 0], [-50, -50, -5, 0, 0]], dtype=torch.float32)
pts_feats = torch.stack([pts_feats], dim=0)
pts_feats.shape

# %%
bev_boi = batch_pointcloud_to_bev(pts_feats)
bev_boi.shape
bev_boi.max()

# %%
imgs_feat = torch.rand((1, 256, 128, 128))
imgs_feat.shape

# %%
fused_feats = torch.cat([imgs_feat, bev_boi], dim=1)
fused_feats.shape

# %%
model.train_cfg.pts.point_cloud_range
# Generate BEV tensor (for example)
bev_tensor = batch_pointcloud_to_bev(pts_feats, bev_shape=(32, 128, 128), point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_occupancy_grid(occupancy_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=90, elev=90)

    # Normalizing intensity values to be between 0 and 1
    intensity_values = occupancy_grid[occupancy_grid.nonzero()]
    intensity_values = (intensity_values - intensity_values.min()) / (intensity_values.max() - intensity_values.min())

    z,x,y = occupancy_grid.nonzero()
    ax.scatter(x, y, z, zdir='z', c='red', alpha=intensity_values)

    plt.show()
    
# visualize the first sample in the batch
visualize_occupancy_grid(bev_tensor[0].numpy())

# %%
pts_feats = single_data['points'].data[0][0]
pts_feats[:, :3] #xyz
print('X min: {}, X max: {}'.format(pts_feats[:, 0].min(), pts_feats[:, 0].max()))
print('Y min: {}, Y max: {}'.format(pts_feats[:, 1].min(), pts_feats[:, 1].max()))
print('Z min: {}, Z max: {}'.format(pts_feats[:, 2].min(), pts_feats[:, 2].max()))

# %%
pts = single_data['points'].data[0]

voxels, num_points, coors = model.voxelize(pts)
voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)

# %%
voxel_features = torch.stack([voxel_features]*3, dim=0)
voxel_features.shape

# %%
img_data = single_data['img_inputs']

# %%
img = model.prepare_inputs(img_data)
x, _ = model.image_encoder(img[0])

# %%
print("The good shit.")
with torch.no_grad():
    x, depth = model.img_view_transformer([x] + img[1:7])


