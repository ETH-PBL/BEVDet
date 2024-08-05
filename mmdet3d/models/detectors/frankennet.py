# Copyright (c) Phigent Robotics. All rights reserved.
import contextlib
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from mmcv.runner import force_fp32
import sys

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


@DETECTORS.register_module()
class FrankenNet(CenterPoint):
    r"""
    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """
    # TODO: add docs for other arguments.

    def __init__(self, late_fusion, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, bev_compressor=None, qd_tracking=False, 
                 bev_roi_extractor=None, track_head=None, use_radar=True, **kwargs):
        super(FrankenNet, self).__init__(**kwargs)
        self.late_fusion = late_fusion
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.qd_tracking = qd_tracking
        if self.qd_tracking:
            self.track_head = builder.build_head(track_head)
        self.use_radar = use_radar

        self.bev_compressor = bev_compressor
        if bev_compressor:
            img_feat_dim = bev_compressor['img_feat_dim']
            img_grid_height = bev_compressor['img_grid_height']
            # Make sure to only account for lidar & radar features if they are used
            point_feat_dim = bev_compressor['pts_feat_dim'] if self.pts_voxel_encoder else 0
            radar_feat_dim = bev_compressor['radar_feat_dim'] if self.radar_voxel_encoder else 0

            # TODO: How to load this better
            self.bev_compressor = nn.Sequential(
                # TODO: Change output dimension if wanted
                nn.Conv2d(img_feat_dim*img_grid_height + point_feat_dim + radar_feat_dim, img_feat_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(img_feat_dim),
                nn.GELU(),
            )

        self.counter_boi = 0

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    # TODO: Do we even need points in here?
    def extract_img_feat(self, img):
        """Extract features of images."""

        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        if self.qd_tracking:
            # separate reference data and original data becasuse they are intertangled
            # we should have [imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda]
            x_orig = x[:, :6, ...].contiguous()
            x_ref = x[:, 6:, ...].contiguous() 
            
            # just writing it explicitly for now, maybe TODO prettify this
            sensor2keyegos_orig = img[1][:, :6]
            sensor2keyegos_ref = img[1][:, 6:]

            ego2globals_orig = img[2][:, :6]
            ego2globals_ref = img[2][:, 6:]

            intrins_orig = img[3][:, :6]
            intrins_ref = img[3][:, 6:]

            post_rots_orig = img[4][:, :6]
            post_rots_ref = img[4][:, 6:]

            post_trans_orig = img[5][:, :6]
            post_trans_ref = img[5][:, 6:]

            bda_orig = img[6]
            bda_ref = img[6]
                            
            # transform features to BEV view
            x_orig, depth_orig = self.img_view_transformer([x_orig, sensor2keyegos_orig, ego2globals_orig, intrins_orig, post_rots_orig, post_trans_orig, bda_orig])
            x_ref, depth_ref = self.img_view_transformer([x_ref, sensor2keyegos_ref, ego2globals_ref, intrins_ref, post_rots_ref, post_trans_ref, bda_ref])

            # reconcatenate the shit
            x = torch.cat([x_orig, x_ref], dim=0)
            depth = torch.cat([depth_orig, depth_ref], dim=0)


        else:
            x, depth = self.img_view_transformer([x] + img[1:7])
        
        return x, depth
    

    def extract_feat(self, points, radar, img, radar_ref=None):

        """Extract features from images and points. Returns features in BEV view."""
        # TODO: Intermediate fusion currently happens withing extract_img_feat
        img_feats, depth = self.extract_img_feat(img)
        
        # Utilize our pointcloud voxel encoder - according to config either PillarFE or OccupancyVFE
        pts_feats = self.pts_voxel_encoder(points) if self.pts_voxel_encoder else None
        # TODO: Need to add config stuff for this
        radar_feats = self.radar_voxel_encoder(radar) if self.radar_voxel_encoder else None
        if radar_ref is not None and self.qd_tracking:
            radar_feats_ref = self.radar_voxel_encoder(radar_ref)
            radar_feats = torch.cat([radar_feats, radar_feats_ref], dim=0)
            
        return (img_feats, pts_feats, radar_feats, depth)


    def forward_train(self,
                      radar=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels_3d_ref=None,
                      gt_labels=None,
                      gt_labels_ref=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      radar_ref=None,
                      gt_bboxes_3d_ref=None,
                      gt_match_indices=None,
                      instance_ids_key=None,
                      instance_ids_ref=None,
                      **kwargs):
        """Forward training function.

        Args:
            radar (list[torch.Tensor], optional): Radar points of each sample.
                Defaults to None.
            points (list[torch.Tensor], optional): Lidar points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
            radar_ref (list[torch.Tensor], optional): Radar points of each
                reference sample, they are needed and used only for QD style 
                tracking. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # TODO: Depth could probably be used since we have radar now?
        #       Currently gets thrown away as the third element
        # TODO: Img_feats here already includes intermediate sensor fusion 

        # To freeze the entire backbone until after feature extraction
        context = torch.no_grad() if False else contextlib.nullcontext() 
        with context:
            if self.qd_tracking:
                if radar_ref is None and self.use_radar:
                        raise ValueError('radar_ref is needed for QD style tracking. Are you sure you activated qd_tracking in the radar loading pipeline?')
                img_feats, pts_feats, radar_feats, _ = self.extract_feat(
                    points, radar=radar, img=img_inputs, radar_ref=radar_ref)
                batch_size = img_feats.shape[0] // 2
            else:
                img_feats, pts_feats, radar_feats, _ = self.extract_feat(
                    points, radar=radar, img=img_inputs)
                batch_size = img_feats.shape[0]

            # img_feats.shape = torch.Size([8, 512, 128, 128]) # OUTDATED, should be 64 dims not 512 since we use BEVDet pooling style (just 1 pillar)
            # pts_feats.shape = torch.Size([8, 8, 128, 128])
            
            # print(f"Radar Features shape: {radar_feats.shape}", file=open("debug.txt", "a"))

            fused_feats = img_feats # Image only

            # Intermediate sensor fusion according to SimpleBEV -> Simple stacking
            if self.pts_voxel_encoder:
                fused_feats = torch.cat([fused_feats, pts_feats], dim=1) # + LiDAR
            if self.radar_voxel_encoder:
                fused_feats = torch.cat([fused_feats, radar_feats], dim=1) # + Radar 

            # print(f"Fused Features shape: {fused_feats.shape}", file=open("debug.txt", "a"))
            

            # For lidar occupancy: fused_feats.shape = torch.Size([8, 520, 128, 128]) # OUTDATED, should be 64 + 8 = 72 not 520
            # For radar pillar: fused_feats.shape = torch.Size([8, 530, 128, 128]) # OUTDATED, should be 64 + 18 = 82 not 530
            if self.bev_compressor:
                if self.qd_tracking:
                    raise ValueError('BEV compressor not compatible with QD style tracking yet!')
                fused_feats = self.bev_compressor(fused_feats)

            # print(f"fused_feats.shape after compression = {fused_feats.shape}", file=open("debug.txt", "a"))

            # fused_feats.shape after compression = torch.Size([8, 64, 128, 128])

            fused_bev_feats = [self.bev_encoder(fused_feats)]

            # fused_bev_feats.shape = torch.Size([8, 256, 128, 128])
            # unless qd_tracking, in that case it is torch.Size([16, 256, 128, 128])

        losses = dict()


        # Fuse image and points features AFTER feature extraction (LATE FUSION)
        # fused_bev_feats = fused_bev_feats # No residual connection
        if self.late_fusion:
            fused_bev_feats = [torch.cat([fused_bev_feats[0], radar_feats], dim=1)] # Residual connection 

        # print(f"fused_bev_feats.shape after late fusion = {fused_bev_feats[0].shape}", file=open("debug.txt", "a"))

        if self.qd_tracking:
            # This is a bit of a hack, but we need to split the fused features into key and ref
            fused_bev_feats_key = fused_bev_feats[0][:fused_bev_feats[0].shape[0]//2]
            fused_bev_feats_ref = fused_bev_feats[0][fused_bev_feats[0].shape[0]//2:]

            fused_bev_feats = [fused_bev_feats_key] # needed for detection head
            fused_bev_feats_ref = [fused_bev_feats_ref] # needed for tracking head

            # Update the detection head
            losses_pts, det_bboxes = self.forward_pts_train(fused_bev_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore,
                                                    True) # Return bboxes
            
            with torch.no_grad():
                _, det_bboxes_ref = self.forward_pts_train(fused_bev_feats_ref, gt_bboxes_3d_ref,
                                                        gt_labels_3d_ref, img_metas,
                                                        None,
                                                        True)
        else:
            # Update the detection head
            losses_pts = self.forward_pts_train(fused_bev_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore,
                                                False) # Dont return bboxes 
        losses.update(losses_pts)

        # Run the tracking head thingie
        if self.qd_tracking:
            # metadata for generating correct bbox coordinates in BEV 
            image_grid_config = self.img_view_transformer.grid_config
            voxel_sizes = [image_grid_config['x'][-1], image_grid_config['y'][-1], image_grid_config['z'][-1]]
            bev_grid_size = [
                (image_grid_config['x'][1]-image_grid_config['x'][0])/voxel_sizes[0],
                (image_grid_config['y'][1]-image_grid_config['y'][0])/voxel_sizes[1]
            ]


            #Attach the img_feats to the fused_bev_feats_key
            img_feats_key = img_feats[:img_feats.shape[0]//2]
            img_feats_ref = img_feats[img_feats.shape[0]//2:]
            fused_bev_feats_key = torch.cat([fused_bev_feats_key, img_feats_key], dim=1)
            fused_bev_feats_ref = [torch.cat([fused_bev_feats_ref[0], img_feats_ref], dim=1)]

            # TODO call the quasi dense tracking head from DETECTION
            loss_track, match_feats, asso_targets, bev_bboxes_key, bev_bboxes_ref, _ = self.track_head.forward_train(
                                                                                    fused_bev_feats_key,
                                                                                    fused_bev_feats_ref[0],
                                                                                    gt_bboxes_3d,
                                                                                    gt_bboxes_3d_ref,
                                                                                    voxel_sizes,
                                                                                    bev_grid_size,
                                                                                    img_metas,
                                                                                    gt_match_indices,
                                                                                    instance_ids_key,
                                                                                    instance_ids_ref,
                                                                                    batch_size,
                                                                                    gt_labels=gt_labels_3d,
                                                                                    gt_labels_ref=gt_labels_3d_ref,
                                                                                    detect_bboxes=det_bboxes,
                                                                                    detect_bboxes_ref=det_bboxes_ref,
                                                                                )
           
            losses.update(loss_track)
            

        #Vizualize occupancy grid
        self.counter_boi += 1
        if self.counter_boi % 100 == 0:
            if self.qd_tracking:
                # Assuming dists and track_targets are lists containing similarity matrices and targets for several samples
                sample_idx = 0  # For the first sample
                similarity_sample = match_feats[0][sample_idx].cpu().detach().numpy()
                target_sample = asso_targets[0][sample_idx].cpu().detach().numpy()
                print('LOSS TRACK: ', loss_track)
                if similarity_sample.size != 0:
                    self.visualize_similarity(similarity_sample, target_sample, gt_bboxes=gt_bboxes_3d[0])
                else:
                    print('Similarity Matrix is empty! No Viz!')

                # Only visualize if there is something to visualize
                if pts_feats is not None or radar_feats is not None:
                    pts_boi = pts_feats[0].detach().cpu().numpy() if pts_feats is not None else None
                    # img_boi = img_feats[0][0].detach().cpu().numpy() # Not used
                    radar_boi = radar_feats[0].detach().cpu().numpy() if radar_feats is not None else None
                    bboxes_key = bev_bboxes_key.clone().detach().cpu().numpy()
                    bboxes_ref = bev_bboxes_ref.clone().detach().cpu().numpy()
                    gt_match_ids = gt_match_indices[0].clone().detach().cpu().numpy()
                    self.visualize_tracker_grid_2D(occupancy_grid=pts_boi, radar_occ_grid=radar_boi, boxes_tensor=bboxes_key, boxes_refs=bboxes_ref, track_targets=target_sample, match_idx=gt_match_ids, sample_idx=sample_idx)
                     
            self.counter_boi = 0 

        return losses

    def forward_test(self,
                     radar=None,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_bboxes_3d=None,
                     gt_bboxes_3d_ref=None,
                     gt_match_indices=None,
                     radar_ref=None,
                     gt_labels_3d=None,
                     gt_labels_3d_ref=None,
                     **kwargs):
        """
        Args:
            radar (list[torch.Tensor], optional): Radar points of each sample.
                Defaults to None.
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            radar = [radar] if radar is None else radar # TODO: Needed?

            if self.qd_tracking:
                return self.simple_test(points=points[0],
                                        img_metas=img_metas[0],
                                        radar=radar[0],
                                        img=img_inputs[0], 
                                        gt_bboxes_3d=gt_bboxes_3d[0], 
                                        gt_bboxes_3d_ref=gt_bboxes_3d_ref[0], 
                                        gt_match_indices=gt_match_indices[0], 
                                        radar_ref=radar_ref[0],
                                        gt_labels_3d=gt_labels_3d,
                                        gt_labels_3d_ref=gt_labels_3d_ref, 
                                        **kwargs)
            else:
                return self.simple_test(points=points[0],
                                        img_metas=img_metas[0],
                                        radar=radar[0],
                                        img=img_inputs[0],
                                        **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    radar=None,
                    img=None,
                    rescale=False,
                    gt_bboxes_3d=None,
                    gt_bboxes_3d_ref=None,
                    gt_match_indices=None,
                    radar_ref=None,
                    instance_ids_key=None,
                    instance_ids_ref=None,
                    gt_labels_3d=None,
                    gt_labels_3d_ref=None,
                    **kwargs):
        """Test function without augmentaiton."""
        if self.qd_tracking:
            if radar_ref is None:
                    raise ValueError('radar_ref is needed for QD style tracking. Are you sure you activated qd_tracking in the radar loading pipeline?')
            img_feats, pts_feats, radar_feats, _ = self.extract_feat(
                points, radar=radar, img=img, radar_ref=radar_ref)
            batch_size = img_feats.shape[0] // 2
        else:
            img_feats, pts_feats, radar_feats, _ = self.extract_feat(
                points, radar=radar, img=img)
            batch_size = img_feats.shape[0]
        
        fused_feats = img_feats # Image only

        # Intermediate sensor fusion according to SimpleBEV -> Simple stacking
        if self.pts_voxel_encoder:
            fused_feats = torch.cat([fused_feats, pts_feats], dim=1) # + LiDAR
        if self.radar_voxel_encoder:
            fused_feats = torch.cat([fused_feats, radar_feats], dim=1) # + Radar

        if self.bev_compressor:
            fused_feats = self.bev_compressor(fused_feats)

        fused_bev_feats = [self.bev_encoder(fused_feats)]
        
        bbox_list = [dict() for _ in range(len(img_metas))]

        # Fuse image and points features AFTER feature extraction (late fusion)
        # fused_bev_feats = fused_bev_feats # Without residual connection
        if self.late_fusion:
            fused_bev_feats = [torch.cat([fused_bev_feats[0], radar_feats], dim=1)]

        if not self.qd_tracking:
            bbox_pts, _ = self.simple_test_pts(fused_bev_feats, img_metas, rescale=rescale)
        else:
            # This is a bit of a hack, but we need to split the fused features into key and ref
            fused_bev_feats_key = fused_bev_feats[0][:fused_bev_feats[0].shape[0]//2]
            fused_bev_feats_ref = fused_bev_feats[0][fused_bev_feats[0].shape[0]//2:]
            fused_bev_feats = [fused_bev_feats_key] # needed for detection head
            fused_bev_feats_ref = [fused_bev_feats_ref] # needed for tracking head

            # Update the detection head
            # print(f"Gt labels 3d {gt_labels_3d}", file=open("debug.txt", "a"))
            bbox_pts, det_bboxes = self.simple_test_pts(fused_bev_feats, img_metas, rescale=rescale)
            
            # ref detection results
            with torch.no_grad():
                _, det_bboxes_ref = self.simple_test_pts(fused_bev_feats_ref, img_metas, rescale=rescale)

            # metadata for generating correct bbox coordinates in BEV 
            image_grid_config = self.img_view_transformer.grid_config
            voxel_sizes = [image_grid_config['x'][-1], image_grid_config['y'][-1], image_grid_config['z'][-1]]
            bev_grid_size = [
                (image_grid_config['x'][1]-image_grid_config['x'][0])/voxel_sizes[0],
                (image_grid_config['y'][1]-image_grid_config['y'][0])/voxel_sizes[1]
            ]

            #Attach the img_feats to the fused_bev_feats_key
            img_feats_key = img_feats[:img_feats.shape[0]//2]
            fused_bev_feats_key = torch.cat([fused_bev_feats_key, img_feats_key], dim=1)

            # call the quasi dense tracking head
            with torch.no_grad():
                key_embeds = self.track_head.forward_test(fused_bev_feats_key,
                                                          det_bboxes,
                                                          voxel_sizes,
                                                          bev_grid_size,
                                                          img_metas,
                                                          batch_size)
            
            #Vizualize occupancy grid
            self.counter_boi += 1
            if self.counter_boi % 100 == 0:
                if self.qd_tracking:
                    # Assuming dists and track_targets are lists containing similarity matrices and targets for several samples
                    sample_idx = 0  # For the first sample
                    # similarity_sample = match_feats[0][sample_idx].cpu().detach().numpy()
                    # target_sample = asso_targets[0][sample_idx].cpu().detach().numpy()
                    # print('EVAL LOSS TRACK: ', loss_track)
                    # if similarity_sample.size != 0:
                    #     self.visualize_similarity(similarity_sample, target_sample, eval_bool=True)
                    # else:
                    #     print('EVAL Similarity Matrix is empty! No Viz!')

                    # Only visualize if there is something to visualize
                    if pts_feats is not None or radar_feats is not None:
                        pts_boi = pts_feats[0].detach().cpu().numpy() if pts_feats is not None else None
                        radar_boi = radar_feats[0].detach().cpu().numpy() if radar_feats is not None else None
                        gt_boi = gt_bboxes_3d[0].tensor.clone().detach().cpu().numpy()
                        gt_boi_ref = gt_bboxes_3d_ref[0].tensor.clone().detach().cpu().numpy()
                        gt_match_ids = gt_match_indices[0].clone().detach().cpu().numpy()
                        self.visualize_tracker_grid_2D(occupancy_grid=pts_boi, radar_occ_grid=radar_boi, boxes_tensor=gt_boi, boxes_refs=gt_boi_ref, match_idx=gt_match_ids, eval_bool=True)

        if self.qd_tracking:
            for result_dict, pts_bbox, key_embed in zip(bbox_list, bbox_pts,key_embeds):
                result_dict['pts_bbox'] = pts_bbox
                result_dict['pts_bbox']['embeddings'] = key_embed.cpu()
                # print(f"So we are saving: {result_dict['pts_bbox']['boxes_3d'], result_dict['pts_bbox']['embeddings'].shape}", file=open("debug.txt", "a"))
        else:
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      radar=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        
        img_feats, pts_feats, radar_feats, _ = self.extract_feat(points, radar=radar, img=img_inputs)        
        fused_feats = img_feats # Image only
        #put radar features on same device as fused_feats
        radar_feats = radar_feats.to(fused_feats.device)

        if self.radar_voxel_encoder:
            fused_feats = torch.cat([fused_feats, radar_feats], dim=1) # + Radar

        if self.bev_compressor:
            fused_feats = self.bev_compressor(fused_feats)

        fused_bev_feats = [self.bev_encoder(fused_feats)]
        
        # Fuse image and points features AFTER feature extraction (late fusion)
        # fused_bev_feats = fused_bev_feats # Without residual connection
        if self.late_fusion:
            fused_bev_feats = [torch.cat([fused_bev_feats[0], radar_feats], dim=1)]

        assert self.with_pts_bbox
        outs = self.pts_bbox_head(fused_bev_feats)
        return outs


####################################################################################################Visualization

    def visualize_similarity(self, similarity_matrix, track_target, eval_bool=False, detections_bool=False, gt_bboxes=[]):
        # Normalize the similarity matrix for better visualization
        normalized_similarity = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))

        matching_str = 'Detection' if detections_bool else 'GND Truth'
        
        plt.clf()
        plt.imshow(normalized_similarity, cmap='viridis', interpolation='none')
        plt.colorbar(label='Similarity Score')
        
        # Overlay the targets with distinct markers
        rows, cols = np.where(track_target == 1)
        plt.scatter(cols, rows, c='red', marker='x', label='{} Matches'.format(matching_str))
        
        plt.xlabel('Reference ROIs')
        plt.ylabel('Key ROIs')
        plt.legend(loc='best')

        # change ratio of x and y axis to match the image
        plt.gca().set_aspect('equal', adjustable='box')
        
        if not eval_bool:
            plt.title('Similarity Matrix with {} Matchesand {} gt bboxes'.format(matching_str, len(gt_bboxes)))
            plt.savefig('embedding_similarity.png')
        else:
            plt.title('EVAL Similarity Matrix with {} Matches and {} gt bboxes'.format(matching_str, len(gt_bboxes)))
            plt.savefig('eval_embedding_similarity.png')

    def visualize_tracker_grid_2D(self, occupancy_grid, radar_occ_grid, boxes_tensor, boxes_refs, match_idx, track_targets=None, eval_bool=False, sample_idx=0):
        #clear old plt
        plt.clf()
        
        # draw 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # fig, ax = plt.subplots()
        ax = axs[0]
        ax2 = axs[1]

        # Occupancy grid
        if occupancy_grid is not None:
            z, x, y = occupancy_grid.nonzero()
            ax.scatter(x, y, s=0.05, c='red')
            # Set the limits for x and y axes
            ax.set_xlim([0, occupancy_grid.shape[1]])
            ax.set_ylim([0, occupancy_grid.shape[2]])

        # Radar features
        z_img, x_img, y_img = radar_occ_grid.nonzero()
        #normalize radar velocity to plot it as alpha
        radar_vel = radar_occ_grid[z_img, x_img, y_img]
        radar_vel = (radar_vel - radar_vel.min()) / (radar_vel.max() - radar_vel.min())

        ax.scatter(y_img, x_img, s=5.5, c='green')

        # Bounding boxes centers
        for i in range(boxes_tensor.shape[0]):
            box_data = self.bev_box_to_plot_box(boxes_tensor[i], point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bev_shape=(64, 128, 128), rot=False)
            if box_data:
                if sample_idx==box_data[-1]:  # If the box is within the point cloud range
                    x_center, y_center, x_size_scaled, y_size_scaled, yaw, ind = box_data
                    ax.scatter(x_center, y_center, c='blue', marker='o')
                    
                    # Draw the patches for the KEYS
                    rectangle = patches.Rectangle((x_center - 0.5 * x_size_scaled, y_center - 0.5 * y_size_scaled), x_size_scaled, y_size_scaled, angle=0, fill=False, edgecolor='blue', linewidth=1.2)
                    t = transforms.Affine2D().rotate_around(x_center, y_center, yaw) + ax.transData
                    rectangle.set_transform(t)
                    ax.add_patch(rectangle)

                    # Annotate the rectangle with the box ID
                    match_id = np.argmax(track_targets[i])
                    annotation_position = (x_center + 1.25 * x_size_scaled, y_center)
                    ax.annotate(str(match_id), annotation_position, color='black', weight='bold', fontsize=10, ha='center', va='center')
            
        for i in range(boxes_refs.shape[0]):
            box_ref_data = self.bev_box_to_plot_box(boxes_refs[i], point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bev_shape=(64, 128, 128), rot=False)
            if box_ref_data:
                if sample_idx==box_ref_data[-1]:
                    x_center_ref, y_center_ref, x_size_scaled_ref, y_size_scaled_ref, yaw_ref, ind = box_ref_data
                    # Draw the patches for the REFS
                    ref_rectangle = patches.Rectangle((x_center_ref - 0.5 * x_size_scaled_ref, y_center_ref - 0.5 * y_size_scaled_ref), x_size_scaled_ref, y_size_scaled_ref, angle=0, fill=False, edgecolor='red', linewidth=1.2)
                    t_ref = transforms.Affine2D().rotate_around(x_center_ref, y_center_ref, yaw_ref) + ax2.transData
                    ref_rectangle.set_transform(t_ref)
                    ax2.add_patch(ref_rectangle)

                    # Annotate the rectangle with the box ID
                    annotation_position = (x_center_ref + 1.25 * x_size_scaled_ref, y_center_ref)
                    ax2.annotate(str(i), annotation_position, color='black', weight='bold', fontsize=10, ha='center', va='center')

        # Set aspect ratio to be equal, to keep the cells square
        ax.set_aspect('equal')
        ax.grid(True, which='both', alpha=0.3)
        ax2.set_aspect('equal')
        ax2.grid(True, which='both', alpha=0.3)

        # set ax2 lims to same of ax 
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        

        if not eval_bool:
            plt.savefig('track_bbox_2D.png')
        else:
            plt.savefig('eval_track_bbox_2D.png')

    # TODO: Probably broken
    def visualize_occupancy_grid(self, occupancy_grid, img_occ_grid, gt_boxes_tensor):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=270, elev=-90)

        z,x,y = occupancy_grid.nonzero()
        ax.scatter(x, y, z, s=0.05, zdir='z', c='red', alpha=1)

        # Plotting img features
        z_img, x_img, y_img = img_occ_grid.nonzero()
        # Normalizing intensity values to be between 0 and 1
        intensity_values = img_occ_grid[img_occ_grid.nonzero()]
        intensity_values = (intensity_values - intensity_values.min()) / (intensity_values.max() - intensity_values.min())
        ax.scatter(x_img, y_img, z_img, s=0.025, zdir='z', c='green', alpha=intensity_values)

        for i in range(gt_boxes_tensor.shape[0]):
            box = gt_boxes_tensor[i]
            lines = self.bbox_3d(box, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bev_shape=(64, 128, 128))
            if lines is not None:
                lc = Line3DCollection(lines, colors='blue', linewidths=3.5, alpha=0.7, linestyles='--')
                ax.add_collection(lc)

        plt.savefig('guggus.png') 

    def visualize_occupancy_grid_2d(self, lidar_occ_grid, radar_occ_grid, gt_boxes_tensor):
        fig, ax = plt.subplots()

        # Lidar occupancy grid
        if lidar_occ_grid is not None:
            z, x, y = lidar_occ_grid.nonzero()
            ax.scatter(x, y, s=0.05, c='red')

        # Radar features
        if radar_occ_grid is not None:
            z_img, x_img, y_img = radar_occ_grid.nonzero()
            #normalize radar velocity to plot it as alpha
            radar_vel = radar_occ_grid[z_img, x_img, y_img]
            radar_vel = (radar_vel - radar_vel.min()) / (radar_vel.max() - radar_vel.min())

            ax.scatter(x_img, y_img, s=5.5, c='green')

        # Bounding boxes centers
        for i in range(gt_boxes_tensor.shape[0]):
            box = self.bbox_2d_centroids(gt_boxes_tensor[i], point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bev_shape=(64, 128, 128), rot=False)
            # The box parameters are in the format [x, y, z, x_size, y_size, z_size, yaw]
            if box is not None:
                x_center, y_center = box[0], box[1]
                ax.scatter(x_center, y_center, c='blue', marker='o')

        # Set aspect ratio to be equal, to keep the cells square
        ax.set_aspect('equal')

        # Set the limits for x and y axes
        ax.set_xlim([0, lidar_occ_grid.shape[1] if lidar_occ_grid is not None else radar_occ_grid.shape[1]])
        ax.set_ylim([0, lidar_occ_grid.shape[2]] if lidar_occ_grid is not None else radar_occ_grid.shape[2])
        plt.savefig('guggus_2D.png')

    def bbox_3d(self, box_params, point_cloud_range, bev_shape, rot=False, rot_angle=0):
        """
        This function creates a 3D bounding box using the box parameters.
        The box parameters are expected to be in the format [x, y, z, x_size, y_size, z_size, yaw].
        """
        x, y, z, x_size, y_size, z_size, yaw = box_params[:7]  # Ignore additional parameters

        theta = rot_angle
        yaw += theta
        # Rotation matrix around roll axis
        R_theta = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        if rot:
            xyz = R_theta @ np.array([x, y, z])
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
        
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        scale_x = bev_shape[1] / (x_max - x_min)
        scale_y = bev_shape[2] / (y_max - y_min)
        scale_z = bev_shape[0] / (z_max - z_min)
        
        # Check if box is in point cloud range
        if (x < x_min or x > x_max or y < y_min or y > y_max or z < z_min or z > z_max):
            return None
        
        # Translate box to start from (0,0,0)
        x -= x_min
        y -= y_min
        z -= z_min
        
        # Rescale box to fit the bev grid
        x *= scale_x
        y *= scale_y
        z *= scale_z
        x_size *= scale_x
        y_size *= scale_y
        z_size *= scale_z
        
        # Rotation matrix
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,           0, 1]
        ])
        
        x_corners = x_size / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = y_size / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = z_size / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        
        # Rotate and translate corners
        corners = np.dot(R, corners)
        corners[0, :] = np.add(corners[0, :], x)
        corners[1, :] = np.add(corners[1, :], y)
        corners[2, :] = np.add(corners[2, :], z)

        # Define the lines of the bounding box0
        lines = [[corners[:,i], corners[:,j]] for i in range(4) for j in range(4) if np.linalg.norm(corners[:,i] - corners[:,j]) == x_size]
        lines += [[corners[:,i+4], corners[:,j+4]] for i in range(4) for j in range(4) if np.linalg.norm(corners[:,i+4] - corners[:,j+4]) == x_size]
        lines += [[corners[:,i], corners[:,i+4]] for i in range(4)]
        
        return lines

    def bbox_2d_centroids(self, box_params, point_cloud_range, bev_shape=(64, 128, 128), rot=False, rot_angle=0):
        """
        This function creates a 3D bounding box using the box parameters.
        The box parameters are expected to be in the format [x, y, z, x_size, y_size, z_size, yaw].
        """
        x, y, z, x_size, y_size, z_size, _ = box_params[:7]  # Ignore additional parameters

        theta = rot_angle
        # Rotation matrix around roll axis
        R_theta = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        if rot:
            xyz = R_theta @ np.array([x, y, z])
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
        
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # Check if box is in point cloud range
        if (x < x_min or x > x_max or y < y_min or y > y_max or z < z_min or z > z_max):
            return None
        
        # Translate box to start from (0,0,0)
        x -= x_min
        y -= y_min
        z -= z_min

        # rescale box to fit the bev grid
        scale_x = bev_shape[1] / (x_max - x_min)
        scale_y = bev_shape[2] / (y_max - y_min)
        scale_z = bev_shape[0] / (z_max - z_min)
        x *= scale_x
        y *= scale_y
        z *= scale_z
        
        return x, y
    
    def bbox_2d_track_centroids(self, box_params, point_cloud_range, bev_shape=(64, 128, 128), rot=False, rot_angle=0):
        """
        This function creates a 3D bounding box using the box parameters.
        The box parameters are expected to be in the format [x, y, z, x_size, y_size, z_size, yaw].
        """
        x, y, z, x_size, y_size, z_size, yaw = box_params[:7]  # Ignore additional parameters

        theta = rot_angle
        # Rotation matrix around roll axis
        R_theta = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        if rot:
            xyz = R_theta @ np.array([x, y, z])
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
        
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # Check if box is in point cloud range
        if (x < x_min or x > x_max or y < y_min or y > y_max or z < z_min or z > z_max):
            return None
        
        # Translate box to start from (0,0,0)
        x -= x_min
        y -= y_min
        z -= z_min

        # Rescale box to fit the bev grid
        scale_x = bev_shape[1] / (x_max - x_min)
        scale_y = bev_shape[2] / (y_max - y_min)
        scale_z = bev_shape[0] / (z_max - z_min)
        x *= scale_x
        y *= scale_y
        z *= scale_z
        x_size *= scale_x
        y_size *= scale_y
        
        return x, y, x_size, y_size, yaw
    
    def bev_box_to_plot_box(self, box_params, point_cloud_range, bev_shape=(64, 128, 128), rot=False, rot_angle=0):
        """
        outputs the proper bounding box for plotting
        """
        ind, x, y, x_size, y_size, yaw = box_params[:6]  # Ignore additional parameters

        
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # Check if box is in point cloud range
        # print(x,y)
        # if (x < x_min or x > x_max or y < y_min or y > y_max):
        #     return None
        
        # # Translate box to start from (0,0,0)
        # x -= x_min
        # y -= y_min

        # # Rescale box to fit the bev grid
        # scale_x = bev_shape[1] / (x_max - x_min)
        # scale_y = bev_shape[2] / (y_max - y_min)
        # x *= scale_x
        # y *= scale_y
        # x_size *= scale_x
        # y_size *= scale_y
        
        return x, y, x_size, y_size, yaw, ind

    def batch_pointcloud_to_bev(self, batch_points, bev_shape=(64, 128, 128), point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], transpose=False, rot=False, rot_angle=0):
        batch_size = len(batch_points)
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        bev = torch.zeros((batch_size,) + bev_shape, dtype=torch.float32, device=batch_points[0].device)

        scale_x = bev_shape[1] / (x_max - x_min)
        scale_y = bev_shape[2] / (y_max - y_min)
        scale_z = bev_shape[0] / (z_max - z_min)

        # rotation matrix for rotation around z-axis
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], 
                    [np.sin(rot_angle), np.cos(rot_angle), 0],
                    [0, 0, 1]])

        for batch_idx in range(batch_size):
            single_points = batch_points[batch_idx]

            if transpose:
                # transpose points
                single_points[:, [0, 1, 2]] = single_points[:, [1, 0, 2]]

            if rot:
                # rotate points
                rotated_points = np.dot(R, single_points[:, :3].cpu().numpy().T).T
                single_points[:, :3] = torch.from_numpy(rotated_points).to(single_points.device)
                
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
            #illuminations = single_points[:, 3]
            
            # clamp indices to bev_shape, as some points may fall outside the last voxel due to rounding errors
            indices[:, 0] = indices[:, 0].clamp(0, bev_shape[2] - 1)
            indices[:, 1] = indices[:, 1].clamp(0, bev_shape[1] - 1)
            indices[:, 2] = indices[:, 2].clamp(0, bev_shape[0] - 1)

            # fill up the bev grid
            bev[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0]] = 1 #illuminations
        return bev
        


@DETECTORS.register_module()
class FrankenNet4D(FrankenNet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(FrankenNet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        # TODO: Figure out how the temporal gathering shall be implemented
        # Use different strategies for radar and image data?
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


@DETECTORS.register_module()
class FrankenNetDepth4D(FrankenNet4D):

    def forward_train(self,
                      radar=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # TODO: Depth could probably be used since we have radar now?
        #       Currently gets thrown away as the third element
        # TODO: Img_feats here already includes intermediate sensor fusion
        img_feats, pts_feats, radar_feats, _ = self.extract_feat(
            points, radar=radar, img=img_inputs)
        
        # img_feats.shape = torch.Size([8, 512, 128, 128])
        # pts_feats.shape = torch.Size([8, 8, 128, 128])
        
        fused_feats = img_feats # Image only

        # Intermediate sensor fusion according to SimpleBEV -> Simple stacking
        if self.pts_voxel_encoder:
            fused_feats = torch.cat([fused_feats, pts_feats], dim=1) # + LiDAR
        if self.radar_voxel_encoder:
            fused_feats = torch.cat([fused_feats, radar_feats], dim=1) # + Radar
        

        # For lidar occupancy: fused_feats.shape = torch.Size([8, 520, 128, 128])
        # For radar pillar: fused_feats.shape = torch.Size([8, 528, 128, 128])?
        if self.bev_compressor:
            fused_feats = self.bev_compressor(fused_feats)

        # fused_feats.shape after compression = torch.Size([8, 64, 128, 128])

        fused_bev_feats = [self.bev_encoder(fused_feats)]

        # fused_bev_feats.shape = torch.Size([8, 256, 128, 128])

        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)

        # Fuse image and points features AFTER feature extraction (late fusion)
        # fused_bev_feats = fused_bev_feats # No residual connection
        fused_bev_feats = [torch.cat([fused_bev_feats[0], radar_feats], dim=1)] # Residual connection

        losses_pts = self.forward_pts_train(fused_bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
