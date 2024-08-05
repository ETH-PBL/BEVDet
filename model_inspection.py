# %%

import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import numpy as np
from PIL import Image

# %%
config = './configs/bevdet/bevdet-r50-inspection.py'
checkpoint = './checkpoints/bevdet-r50.pth'

cfg = Config.fromfile(config)

# Some backward compatibility modifications - not quite sure if they actually do something
cfg = compat_cfg(cfg)

cfg.model.pretrained = None
test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}

# build the dataloader
# TODO: need to change the config to load the data due to relative path problems
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, **test_loader_cfg)
 
# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

# %%
# Inspect the different model layers
print(model.pts_bbox_head)


# %%
# Inspect the NuScenes data
data_iterloader = iter(data_loader)

# %%
data = next(data_iterloader)


# %%
# Get a datapoint for inspection and to feed it to the model
data.keys()

# %%
# Check out img_metas
print(data['img_metas'])

# %%
print(data['points'])

# %%
# Run the model on the input
model.eval()
# model.cuda()
with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data)

print(result)

# TODO: Have to fix some CUDA issues