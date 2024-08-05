# Copyright (c) Ferrari PBL Team 2023

# mAP: -
# mATE: -
# mASE: -
# mAOE: -
# mAVE: -
# mAAE: -
# NDS: -
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	
# truck	
# bus	
# trailer	
# construction_vehicle	
# pedestrian	
# motorcycle	
# bicycle	
# traffic_cone	
# barrier	

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# TODO: Maybe change back to 10? In the base/datasets/nus-3d.py file it is 10
#       don't know why that gets overwritten here
# For nuScenes we usually do 7-class detection
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
    'flip': False,  #TURNED OFF FOR NOW
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# TODO: Radar config? Probably needs to be loaded similar to the _base_/datasets/nus-3d.py file for LIDAR

# Model
grid_config = { # lower, upper, interval
    'x': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
    'y': [-51.2, 51.2, 0.8], # 128 bins -> Original SimpleBEV uses 200
    'z': [-5, 3, 1],         # 8 bin -> This is SimpleBEV style
    # 'z': [-5, 3, 8],          # 1 bin
    'depth': [1.0, 60.0, 1.0],
}

# TODO: Where exactly is this used and is this reasonable?
# Because the lift-splat-shoot of the image features is based on the voxel sizes given above
voxel_size = [0.1, 0.1, 0.2]
voxel_size_lidarpts = [0.2, 0.2, 0.5]

numC_Trans = 64 # Original SimpleBEV uses 128

# TODO: Adapt all the variables for FrankenNet
model = dict(
    type='FrankenNet',
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
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    # Point feature extractor - OccupancyVFE or RadarPillarFE
    pts_voxel_encoder=dict(type='OccupancyVFE', grid_config=grid_config), # Our custom VFE which also voxelizes the points
    # TODO: Need to change the pts_feat_dim to 16 for radar or 8 for lidar occupancy grid
    bev_compressor=dict(img_feat_dim=numC_Trans, pts_feat_dim=0, grid_height_config=grid_config['z']), 
    # pts_voxel_encoder=dict(type='RadarPillarFE', grid_config=grid_config), # Our custom PillarFE which also rasterizes the points
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256, 
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
    # TODO: Have to add the correct lidar/radar loading here - _base_ is overwritten!
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
data_root_pkl = 'data/nuscenes_pkl/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# TODO: I think this is an intermediate variable and overwrites the _base_ config file completely!
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
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

# Lidar gets loaded anyways, but the other modalities only get loaded if set to True (currently only Lidar and Camera)
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet', # TODO: Change?
)

# TODO: I guess the .pkl should still be good?
test_data_config = dict(
    pipeline=test_pipeline,
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
# TODO: Change?
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# TODO: No fucking clue what this does
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# Sets eval interval, THIS HAS TO BE THE SAME as in the Checkpoint interval!
evaluation = dict(interval=5)
checkpoint_config = dict(interval=5)
# fp16 = dict(loss_scale='dynamic')
