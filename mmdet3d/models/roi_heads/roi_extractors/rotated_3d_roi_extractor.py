# PBL 2023

import torch
from mmcv import ops
from mmcv.runner import BaseModule

from mmdet3d.models.builder import ROI_EXTRACTORS

from detectron2.layers.roi_align_rotated import ROIAlignRotated

@ROI_EXTRACTORS.register_module()
class Rotated3DRoIExtractor(BaseModule):
    """Rotated 3D RoI extractor to extract RoI from BEV feature maps.
    """

    def __init__(self, roi_shape: tuple=(7,7)):
        super(Rotated3DRoIExtractor, self).__init__()

        self.roi_shape=roi_shape

        self.roi_extractor = ROIAlignRotated(self.roi_shape, spatial_scale=(1.), sampling_ratio=0)

    def forward(self, feats, bboxes):
        """Extract RoI features from BEV feature maps.

        Args: 
            feats (torch.FloatTensor): BEV feature maps with shape (batch, channels, height, width).
            bboxes (torch.FloatTensor): interested bounding boxes with shape (n_boxes,  6).
                the 6 columns are [batch_index, x_center, y_center, width, length, angle]
                Requires angles in degrees.
        
        Returns:
            torch.FloatTensor: RoI features with shape (n_boxes, channels, roi_height, roi_width).
        """
        
        # necessary casting for cuda underling operations
        if not isinstance(feats, torch.Tensor):
            if isinstance(feats, list):
                feats = torch.stack(feats).to(bboxes.device)
            else:
                feats = torch.from_numpy(feats).to(bboxes.device)
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.from_numpy(bboxes.astype(float)).to(feats.device)
        
        feats = feats.to(torch.float32)
        bboxes = bboxes.to(torch.float32)

        rois = self.roi_extractor(feats, bboxes)

        return rois
    
