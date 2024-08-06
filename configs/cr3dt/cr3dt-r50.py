# mAP: 0.3515                                                                                                                                                                                          
# mATE: 0.6431
# mASE: 0.2907
# mAOE: 0.5936
# mAVE: 0.4708
# mAAE: 0.1982
# NDS: 0.4561
# Eval time: 44.0s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.656   0.370   0.161   0.100   0.434   0.196
# truck   0.308   0.589   0.223   0.159   0.348   0.219
# bus     0.385   0.718   0.220   0.122   0.768   0.201
# trailer 0.141   0.982   0.244   0.506   0.256   0.064
# construction_vehicle    0.100   0.873   0.519   1.253   0.113   0.369
# pedestrian      0.402   0.678   0.309   0.983   0.696   0.286
# motorcycle      0.300   0.598   0.283   0.878   0.655   0.232
# bicycle 0.240   0.566   0.293   1.190   0.496   0.020
# traffic_cone    0.525   0.500   0.353   nan     nan     nan
# barrier 0.458   0.556   0.302   0.151   nan     nan


#ON MINI:
# mAP: 0.3494                                                                                                                                                                                                                                        
# mATE: 0.6726
# mASE: 0.4735
# mAOE: 0.6878
# mAVE: 0.5607
# mAAE: 0.2955
# NDS: 0.4057
# Eval time: 1.5s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.660   0.409   0.173   0.159   0.142   0.073
# truck   0.496   0.352   0.256   0.097   0.072   0.000
# bus     0.589   0.448   0.087   0.104   0.970   0.018
# trailer 0.000   1.000   1.000   1.000   1.000   1.000
# construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000
# pedestrian      0.519   0.626   0.272   0.702   0.500   0.227
# motorcycle      0.403   0.733   0.292   1.323   0.073   0.000
# bicycle 0.180   0.739   0.249   0.805   0.729   0.047
# traffic_cone    0.647   0.421   0.404   nan     nan     nan
# barrier 0.000   1.000   1.000   1.000   nan     nan

import math


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True, 
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
image_grid_config = { # lower, upper, interval
    'x': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
    'y': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
    'z': [-5, 3, 8],          # 1 bin -> This was the original BEVDet approach to pooling
    'depth': [1.0, 60.0, 1.0],
}
img_z_num_cells = int((image_grid_config['z'][1] - image_grid_config['z'][0]) / image_grid_config['z'][2])

pts_grid_config = { # lower, upper, interval
    'x': image_grid_config['x'],
    'y': image_grid_config['y'],
    # 'z': [-5, 3, 1],         # 8 bin -> This is SimpleBEV style
    'z': [-5, 3, 8],          # 1 bin -> This was the original BEVDet approach to pooling
}
pts_z_num_cells = int((pts_grid_config['z'][1] - pts_grid_config['z'][0]) / pts_grid_config['z'][2])


# Because the lift-splat-shoot of the image features is based on the voxel sizes given above
voxel_size = [0.1, 0.1, 0.2]

######################
# ABLATION PARMATERS #
######################
numC_Trans = 64 # Original SimpleBEV uses 128
radar_feat_dim = 18

# ABLATION 1
# BEV compression -> Change the z_grid size above from 1 to >1 (either 8, 10)
bev_compression = False

if bev_compression:
    numC_Trans_Fused = numC_Trans
    radar_grid_config = { # lower, upper, interval
        'x': image_grid_config['x'],
        'y': image_grid_config['y'],
        'z': [-5, 3, 0.8],         # 10 bin 
    }
    radar_z_num_cells = int((radar_grid_config['z'][1] - radar_grid_config['z'][0]) / radar_grid_config['z'][2])
    image_grid_config = { # lower, upper, interval
        'x': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
        'y': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
        'z': [-5, 3, 0.8],       # 10 bins
        'depth': [1.0, 60.0, 1.0],
    }
    img_z_num_cells = int((image_grid_config['z'][1] - image_grid_config['z'][0]) / image_grid_config['z'][2])
else:
    numC_Trans_Fused = numC_Trans + radar_feat_dim
    radar_grid_config = { # lower, upper, interval
        'x': image_grid_config['x'],
        'y': image_grid_config['y'],
        'z': [-5, 3, 8],         # 1 bin -> This is SimpleBEV style
    }
    radar_z_num_cells = int((radar_grid_config['z'][1] - radar_grid_config['z'][0]) / radar_grid_config['z'][2])
    image_grid_config = { # lower, upper, interval
        'x': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
        'y': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
        'z': [-5, 3, 8],       # 1 bin
        'depth': [1.0, 60.0, 1.0],
    }
    img_z_num_cells = int((image_grid_config['z'][1] - image_grid_config['z'][0]) / image_grid_config['z'][2])

# ABLATION 3 
# radar encoding
radar_encoding = 'RadarPillarFE' #  'RadarPillarFE' or 'OccupancyVFE'

if radar_encoding == 'RadarPillarFE':
    # Our custom PillarFE which also rasterizes the points
    radar_feat_dim = 18
    radar_voxel_encoder = dict(type='RadarPillarFE', grid_config=radar_grid_config, transpose=False, rot=False, radar_feat_dim=radar_feat_dim)
elif radar_encoding == 'OccupancyVFE':
    # Our custom VFE which also voxelizes the points
    radar_feat_dim = 1
    radar_voxel_encoder = dict(type='OccupancyVFE', grid_config=radar_grid_config, transpose=False, rot=False, rot_angle=math.pi/2)
else:
    raise ValueError(f'Unknown radar encoding: {radar_encoding}')


# ABLATION 2
# late fusion
late_fusion = True
if late_fusion:
    late_fusion_dim = 256 + radar_feat_dim*radar_z_num_cells
else:
    late_fusion_dim = 256

#########
# MODEL #
#########
model = dict(
    type='CR3DTNet',
    late_fusion=late_fusion,
    # Image feature extractor
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=image_grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    # INTERMEDIATE FUSION
    # is decided above in ablation parameters
    radar_voxel_encoder=radar_voxel_encoder,
    # is decided above in ablation parameters
    bev_compressor=dict(img_feat_dim=numC_Trans, pts_feat_dim=pts_z_num_cells, radar_feat_dim=radar_feat_dim*radar_z_num_cells, img_grid_height=img_z_num_cells) if bev_compression else None, 
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans_Fused,
        num_channels=[numC_Trans_Fused * 2, numC_Trans_Fused * 4, numC_Trans_Fused * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans_Fused * 8 + numC_Trans_Fused * 2,
        out_channels=256),
    # LATE FUSION -> Directly in the network, that's why we have a jump in the channel size here
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=late_fusion_dim,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    ),
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
data_root_pkl = 'data/nuscenes/'
file_client_args = dict(backend='disk')

#For mini uncomment this
# data_root = 'data/nuscenes_mini/'
# data_root_pkl = 'data/nuscenes_mini/'

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadRadarPointsFromFile',
        sweeps_num=4),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[1., 1.],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.0), 
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'radar'])
]

# Same thing, needs to be adapted
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadRadarPointsFromFile',
        sweeps_num=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(704, 256),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'radar'])
        ])
]

# Lidar gets loaded anyways, but the other modalities only get loaded if set to True (currently only Lidar and Camera)
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet', 
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,
    ann_file=data_root_pkl + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root_pkl + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2), 
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=8)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])
runner = dict(type='EpochBasedRunner', max_epochs=60)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# Sets eval interval, THIS HAS TO BE THE SAME as in the Checkpoint interval!
evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)
