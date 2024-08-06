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

import math

_base_ = ['./cr3dt-r50.py']

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


# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes_test/'
data_root_pkl = 'data/nuscenes_test/'
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
    ann_file=data_root_pkl + 'bevdetv2-nuscenes_infos_test.pkl')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
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
runner = dict(type='EpochBasedRunner', max_epochs=50)

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
