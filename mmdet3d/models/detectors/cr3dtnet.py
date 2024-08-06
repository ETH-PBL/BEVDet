import contextlib
import torch
from torch import nn
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint

@DETECTORS.register_module()
class CR3DTNet(CenterPoint):
    r"""
    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
        use_radar (bool): Whether to use radar data. Default: True.
    """
    def __init__(self, late_fusion, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, bev_compressor=None, use_radar=True, **kwargs):
        super(CR3DTNet, self).__init__(**kwargs)
        self.late_fusion = late_fusion
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.use_radar = use_radar

        self.bev_compressor = bev_compressor
        if bev_compressor:
            img_feat_dim = bev_compressor['img_feat_dim']
            img_grid_height = bev_compressor['img_grid_height']
            # Make sure to only account for lidar & radar features if they are used
            point_feat_dim = bev_compressor['pts_feat_dim'] if self.pts_voxel_encoder else 0
            radar_feat_dim = bev_compressor['radar_feat_dim'] if self.radar_voxel_encoder else 0

            self.bev_compressor = nn.Sequential(
                # Change output dimension if wanted
                nn.Conv2d(img_feat_dim*img_grid_height + point_feat_dim + radar_feat_dim, img_feat_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(img_feat_dim),
                nn.GELU(),
            )

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

    def extract_img_feat(self, img):
        """Extract features of images."""

        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        
        return x, depth
    

    def extract_feat(self, points, radar, img, radar_ref=None):

        """Extract features from images and points. Returns features in BEV view."""
        # Intermediate fusion happens within extract_img_feat
        img_feats, depth = self.extract_img_feat(img)
        
        # Utilize our pointcloud voxel encoder - according to config either PillarFE or OccupancyVFE
        pts_feats = self.pts_voxel_encoder(points) if self.pts_voxel_encoder else None
        radar_feats = self.radar_voxel_encoder(radar) if self.radar_voxel_encoder else None
            
        return (img_feats, pts_feats, radar_feats, depth)


    def forward_train(self,
                      radar=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img_inputs=None,
                      gt_bboxes_ignore=None,
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
            img_inputs (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # To freeze the entire backbone until after feature extraction
        context = torch.no_grad() if False else contextlib.nullcontext() 
        with context:
            img_feats, pts_feats, radar_feats, _ = self.extract_feat(
                points, radar=radar, img=img_inputs)
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

        losses = dict()


        # Fuse image and points features AFTER feature extraction (LATE FUSION)
        # fused_bev_feats = fused_bev_feats # No residual connection
        if self.late_fusion:
            fused_bev_feats = [torch.cat([fused_bev_feats[0], radar_feats], dim=1)] # Residual connection 

        # Update the detection head
        losses_pts = self.forward_pts_train(fused_bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore,
                                            False) # Dont return bboxes 
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     radar=None,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
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
            img_inputs (list[torch.Tensor], optional): the outer
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
            radar = [radar] if radar is None else radar 

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
                    **kwargs):
        """Test function without augmentaiton."""
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

        bbox_pts, _ = self.simple_test_pts(fused_bev_feats, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      radar=None,
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