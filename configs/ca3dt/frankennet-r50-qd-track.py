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

_base_ = ["./frankennet-r50.py"]

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

data_config = {
    "cams": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,  # TURNED OFF FOR NOW
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

# Data
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
data_root_pkl = "data/nuscenes_out/"


file_client_args = dict(backend="disk")

bda_aug_conf = dict(rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5)
# rot_lim=(0, 0),
# scale_lim=(1, 1),
# flip_dx_ratio=0.0,
# flip_dy_ratio=0.0)
# TODO: I think this is an intermediate variable and overwrites the _base_ config file completely!
train_pipeline = [
    dict(type="PrepareImageInputs", is_train=True, data_config=data_config, qd_tracking=True),
    dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type="LoadRadarPointsFromFile", sweeps_num=4, qd_tracking=True),
    dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
    # rot_range=[-0.3925, 0.3925],
    # scale_ratio_range=[0.95, 1.05],
    # translation_std=[0, 0, 0]),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.0),  # TODO flip was 0.5
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range, qd_tracking=True),
    dict(type="ObjectNameFilter", classes=class_names, qd_tracking=True),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "img_inputs",
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes_3d_ref",
            "gt_labels_3d_ref",
            "radar",
            "radar_ref",
            "gt_match_indices",
            "instance_ids_key",
            "instance_ids_ref",
        ],
    ),
]

# Same thing, needs to be adapted
test_pipeline = [
    dict(type="PrepareImageInputs", data_config=data_config, qd_tracking=True),
    dict(type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type="LoadRadarPointsFromFile", sweeps_num=4, qd_tracking=True),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(704, 256),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(
                type="Collect3D",
                keys=[
                    "points",
                    "img_inputs",
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "gt_bboxes_3d_ref",
                    "gt_labels_3d_ref",
                    "radar",
                    "radar_ref",
                    "gt_match_indices",
                    "instance_ids_key",
                    "instance_ids_ref",
                ],
            ),
        ],
    ),
]

## Model
model = dict(
    type="FrankenNet",
    qd_tracking=True,
    img_backbone=dict(
        frozen_stages=-1,  # -1 for none # 4 is probably the maximum
    ),
    track_head=dict(
        type="QuasiDenseRoIHead",
        track_roi_extractor=dict(type="Rotated3DRoIExtractor", roi_shape=(7, 7)),
        track_head=dict(
            type="QuasiDenseEmbedHead",
            num_convs=4,
            num_fcs=1,
            in_channels=338,
            embed_channels=256,
            norm_cfg=dict(type="GN", num_groups=32),
            loss_track=dict(type="MultiPosCrossEntropyLoss", loss_weight=0.25),
            loss_track_aux=dict(
                type="L2Loss", neg_pos_ub=3, pos_margin=0, neg_margin=0.1, hard_mining=True, loss_weight=1.0
            ),
        ),
        track_train_cfg=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type="BboxOverlapsBEV"),  # here the modified Rotated BEV Calculator
            ),
            sampler=dict(
                type="CombinedSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type="InstanceBalancedPosSampler"),
                neg_sampler=dict(type="RandomSampler"),
            ),
        ),
    ),
)

# Lidar gets loaded anyways, but the other modalities only get loaded if set to True (currently only Lidar and Camera)
input_modality = dict(use_lidar=False, use_camera=True, use_radar=True, use_map=False, use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype="bevdetQD",  # TODO: Change?
)

test_data_config = dict(
    samples_per_gpu=2,
    data_root=data_root,
    pipeline=test_pipeline,
    ann_file=data_root_pkl + "bevdetv2-nuscenes_infos_val.pkl",
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root_pkl + "bevdetv2-nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=test_data_config,
    test=test_data_config,
)

for key in ["train", "val", "test"]:
    data[key].update(share_data_config)

# Optimizer
# TODO: Change?
optimizer = dict(type="AdamW", lr=2e-5, weight_decay=1e-07)
optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2), type="GradientCumulativeOptimizerHook", cumulative_iters=8
)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[
        24,
    ],
)
runner = dict(type="EpochBasedRunner", max_epochs=50)

# TODO: No fucking clue what this does
custom_hooks = [
    dict(
        type="MEGVIIEMAHook",
        init_updates=10560,
        priority="NORMAL",
    ),
]

# Sets eval interval, THIS HAS TO BE THE SAME as in the Checkpoint interval!
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
# fp16 = dict(loss_scale='dynamic')
