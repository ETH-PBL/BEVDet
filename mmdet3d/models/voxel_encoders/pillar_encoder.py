# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from mmcv.runner import force_fp32
from torch import nn

from ..builder import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator


# Copyright (c) PBL Lab ETHZ
@VOXEL_ENCODERS.register_module()
class RadarPillarFE(nn.Module):
    """Simple radar feature encoder used in SimpleBEV

    It simply takes all the pre-processed features of the radar point cloud and rasterizes them into a grid.

    Args:
        grid_config (dict[List]): lower, upper & interval of the x, y, z dimension of the grid.
    """
    def __init__(self, grid_config, transpose=False, rot=False, rot_angle=0, radar_feat_dim=18) -> None:
        super(RadarPillarFE, self).__init__()
        self.grid_config = grid_config
        self.transpose = transpose
        self.rot = rot
        self.rot_angle = rot_angle
        self.radar_feat_dim = radar_feat_dim
    
    @force_fp32(out_fp16=True)
    def forward(self, points):
        """Voxelize the points and build a feature grid per batch with the pre-processed radar data.

        Args:
            points (list[torch.Tensor]): List of radar point clouds with shape (B, N, radar_features).
                                         Radar features are typically x,y,z,+15 dimensions.
        
        Returns:
            The feature grid with shape (B, radar_feats_dim * Z, Y, X) with B being the batch size.
        """
        batch_size = len(points)
        x_min, x_max = self.grid_config['x'][:2]
        num_x_bins = int((x_max - x_min) / self.grid_config['x'][2])
        y_min, y_max = self.grid_config['y'][:2]
        num_y_bins = int((y_max - y_min) / self.grid_config['y'][2])
        z_min, z_max = self.grid_config['z'][:2]
        num_z_bins = int((z_max - z_min) / self.grid_config['z'][2])
        bev_shape = (num_z_bins, num_y_bins, num_x_bins) # Typically (1, 128, 128)
        bev_feat_shape = (num_z_bins, num_y_bins, num_x_bins, self.radar_feat_dim) # (1, 128, 128, radar_feats_dim) -> typically 18
        feature_grid = torch.zeros((batch_size,) + bev_feat_shape, dtype=torch.float32, device=points[0].device)
        # Counts the number of points in each voxel for the running average
        counter_grid = torch.zeros((batch_size,) + bev_shape, dtype=torch.float32, device=points[0].device)

        scale_x = bev_shape[2] / (x_max - x_min)
        scale_y = bev_shape[1] / (y_max - y_min)
        scale_z = bev_shape[0] / (z_max - z_min)

        # rotation matrix for rotation around z-axis
        R = np.array([[np.cos(self.rot_angle), -np.sin(self.rot_angle), 0], 
                    [np.sin(self.rot_angle), np.cos(self.rot_angle), 0],
                    [0, 0, 1]])

        # Point shape is (B, N, x, y, z, ...)
        for batch_idx in range(batch_size):
            single_batch_points = points[batch_idx]

            if self.transpose:
                # transpose points
                single_batch_points[:, [0, 1, 2]] = single_batch_points[:, [1, 0, 2]]

            if self.rot:
                # rotate points
                rotated_points = np.dot(R, single_batch_points[:, :3].cpu().numpy().T).T
                single_batch_points[:, :3] = torch.from_numpy(rotated_points).to(single_batch_points.device)

            # filter points outside range
            mask = ((single_batch_points[:, 0] >= x_min) & (single_batch_points[:, 0] <= x_max) &
                    (single_batch_points[:, 1] >= y_min) & (single_batch_points[:, 1] <= y_max) &
                    (single_batch_points[:, 2] >= z_min) & (single_batch_points[:, 2] <= z_max))
            single_batch_points = single_batch_points[mask]

            # if no points are left after filtering, continue
            if len(single_batch_points) == 0:
                continue

            # translate points to start in bin (0,0,0)
            single_batch_points[:, 0] -= x_min
            single_batch_points[:, 1] -= y_min
            single_batch_points[:, 2] -= z_min

            # Rescale points to fit the bev grid
            single_batch_points[:, 0] *= scale_x
            single_batch_points[:, 1] *= scale_y
            single_batch_points[:, 2] *= scale_z

            # Get indices for assignment -> round down
            indices = single_batch_points[:, :3].type(torch.long)

            # Clamp indices to bev_shape, as some points may fall outside the last voxel due to rounding errors
            indices[:, 0] = indices[:, 0].clamp(0, bev_shape[2] - 1)
            indices[:, 1] = indices[:, 1].clamp(0, bev_shape[1] - 1)
            indices[:, 2] = indices[:, 2].clamp(0, bev_shape[0] - 1)

            # Save the pre-processed features in the corresponding voxel with a running average
            counter_grid[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0]] += 1.0
            # Change the number of features extracted from the radar data here (in single_batch_points[:, 18-#features:]) -> automated now
            feature_grid[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0], :] = \
                feature_grid[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0], :] + \
                (single_batch_points[:, 18 - self.radar_feat_dim:].type(torch.float32) - \
                feature_grid[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0], :]) / \
                counter_grid[batch_idx, indices[:, 2], indices[:, 1], indices[:, 0]].unsqueeze(1)
            

        # Need to stack the features into the expected shape of (B, radar_feat_dim * Z, Y, X) from (B, Z, Y, X, radar_feat_dim)
        feature_grid = feature_grid.permute(0, 4, 1, 2, 3).contiguous()
        feature_grid = feature_grid.view(batch_size, self.radar_feat_dim * num_z_bins, num_y_bins, num_x_bins)


        return feature_grid

    

@VOXEL_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)


@VOXEL_ENCODERS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(DynamicPillarFeatureNet, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy)
        self.fp16_enabled = False
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggregated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.

        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the number of points.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors
