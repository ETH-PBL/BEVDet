# From https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/roi_heads/quasi_dense_roi_head.py

import torch
from mmcv.runner import BaseModule
from mmdet.core import build_assigner, build_sampler
from .. import builder
from mmdet.models import HEADS, build_head, build_roi_extractor
import numpy as np
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


@HEADS.register_module()
class QuasiDenseRoIHead(BaseModule):

    def __init__(self,
                 track_roi_extractor=None,
                 track_head=None,
                 track_train_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if track_head is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_head(track_roi_extractor, track_head)
            if self.track_train_cfg:
                self.init_track_assigner_sampler()

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get('assigner', None):
            self.track_roi_assigner = build_assigner(
                self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get('sampler', None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self)
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = builder.build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()

    def forward_train(self,
                      fused_bev_feats_key,
                      fused_bev_feats_ref,
                      gt_bboxes_3d_key,
                      gt_bboxes_3d_ref,
                      voxel_sizes,
                      bev_grid_size,
                      img_metas,
                      gt_match_indices,
                      instance_ids_key,
                      instance_ids_ref,
                      batch_size,
                    #   proposal_list,
                      gt_labels=None,
                      gt_labels_ref=None,
                    #   ref_img_metas,
                    #   ref_proposals,
                    #   ref_gt_labels,
                      detect_bboxes=None,
                      detect_bboxes_ref=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      *args,
                      **kwargs):
        if self.with_track:
            num_imgs = len(img_metas) # TODO we do not have images, but bev grids!
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]

            if detect_bboxes is not None and gt_labels is not None:
                with torch.no_grad():
                    proposal_bboxes_key = [el[0] for el in detect_bboxes]
                    proposal_bboxes_ref = [el[0] for el in detect_bboxes_ref]
                    
                    # obtain BEV bboxes
                    bev_gt_bboxes_key = bbox3d2bevbox(gt_bboxes_3d_key, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)
                    bev_gt_bboxes_ref = bbox3d2bevbox(gt_bboxes_3d_ref, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)
                    key_sample_ids = bev_gt_bboxes_key[:, 0].detach().cpu().numpy()
                    ref_sample_ids = bev_gt_bboxes_ref[:, 0].detach().cpu().numpy()
                    num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                    num_ref_rois = tuple(np.bincount(ref_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                    bev_gt_bboxes_key = torch.split(bev_gt_bboxes_key.to(proposal_bboxes_key[0].device), num_key_rois)
                    bev_gt_bboxes_ref = torch.split(bev_gt_bboxes_ref.to(proposal_bboxes_key[0].device), num_ref_rois)


                    # regroup the batches
                    bev_proposal_bboxes_key = bbox3d2bevbox(proposal_bboxes_key, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)
                    bev_proposal_bboxes_ref = bbox3d2bevbox(proposal_bboxes_ref, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)
                    key_sample_ids = bev_proposal_bboxes_key[:, 0].detach().cpu().numpy()
                    ref_sample_ids = bev_proposal_bboxes_ref[:, 0].detach().cpu().numpy()
                    num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                    num_ref_rois = tuple(np.bincount(ref_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                    # assert len(num_key_rois) == len(num_ref_rois), "Number of key and ref rois should be the same, maybe one sample does not have gt boxes in the last frame?"
                    bev_proposal_bboxes_key = torch.split(bev_proposal_bboxes_key, num_key_rois)
                    bev_proposal_bboxes_ref = torch.split(bev_proposal_bboxes_ref, num_ref_rois)

                                
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        bev_proposal_bboxes_key[i][:, 1:], bev_gt_bboxes_key[i][:, 1:], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        bev_proposal_bboxes_key[i][:, 1:],
                        bev_gt_bboxes_key[i][:, 1:],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in fused_bev_feats_key])
                    key_sampling_results.append(sampling_result)

                    ref_assign_result = self.track_roi_assigner.assign(
                        bev_proposal_bboxes_ref[i][:, 1:], bev_gt_bboxes_ref[i][:, 1:],
                        ref_gt_bboxes_ignore[i], gt_labels_ref[i])
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        bev_proposal_bboxes_ref[i][:, 1:],
                        bev_gt_bboxes_ref[i][:, 1:],
                        gt_labels_ref[i],
                        feats=[lvl_feat[i][None] for lvl_feat in fused_bev_feats_ref])
                    ref_sampling_results.append(ref_sampling_result)


                key_bboxes = samplres2bboxes(key_sampling_results, only_pos=True)
                ref_bboxes = samplres2bboxes(ref_sampling_results, only_pos=False)
                rois_key = self.track_roi_extractor(fused_bev_feats_key, key_bboxes)
                rois_ref = self.track_roi_extractor(fused_bev_feats_ref, ref_bboxes)
                key_feats = self.track_head(rois_key)
                ref_feats = self.track_head(rois_ref)

                match_feats = self.track_head.match(key_feats, ref_feats,
                                                key_sampling_results,
                                                ref_sampling_results)
                match_indices_bin_matrix = match_feats[0]
                
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results)
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                bev_bboxes_key = key_bboxes
                bev_bboxes_ref = ref_bboxes
                # Pass embeddings to store them in the results
                key_sample_ids = bev_bboxes_key[:, 0].detach().cpu().numpy()
                # ref_sample_ids = bev_bboxes_ref[:, 0]
                num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                # assert len(num_key_rois) == len(num_ref_rois), "Number of key and ref rois should be the same, maybe one sample does not have gt boxes in the last frame?"
                key_embeds = torch.split(key_feats, num_key_rois)

            else:
                # generate bboxes in correct format (meters -> grid coords)
                # final array should be 6-dimensional: [batch_index, x, y, w, l, angle]
                bev_bboxes_key = bbox3d2bevbox(gt_bboxes_3d_key, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)
                bev_bboxes_ref = bbox3d2bevbox(gt_bboxes_3d_ref, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)

                # generate RoI features
                rois_key = self.track_roi_extractor(fused_bev_feats_key, bev_bboxes_key)
                rois_ref = self.track_roi_extractor(fused_bev_feats_ref, bev_bboxes_ref)
                # and embed them

                #Handle case where there is nothing to track                
                if rois_key.numel() == 0 or rois_ref.numel() == 0:
                    print('EVAL NO ROIS')
                    loss_track = 0
                    match_feats = None
                    asso_targets = None
                
                else:
                    key_feats = self.track_head(rois_key)
                    ref_feats = self.track_head(rois_ref)

                    loss_track = 0
                    # TODO use full match when detection bboxes are used
                    # match_feats = self.track_head.match(key_feats, ref_feats,
                    #                                     key_sampling_results,
                    #                                     ref_sampling_results)

                    # temporary values generated for ground truth only
                    key_sample_ids = bev_bboxes_key[:, 0]
                    ref_sample_ids = bev_bboxes_ref[:, 0]
                    gt_match_indices = self.match_ids(instance_ids_key, instance_ids_ref, device=key_feats[0].device)
                    # self.check(gt_match_indices=gt_match_indices, bev_bboxes_key=bev_bboxes_key, bev_bboxes_ref=bev_bboxes_ref, batch_size=batch_size)
                    match_feats = self.track_head.match_gt_only(key_feats, ref_feats, key_sample_ids, ref_sample_ids, batch_size)

                    asso_targets = self.track_head.get_track_targets_gt_only(
                        gt_match_indices, key_sample_ids, ref_sample_ids, batch_size)
                    # for el, el2 in zip(asso_targets, match_feats):
                    #     print(f"{[e.device for e in el]} {[e.device for e in el2]}", file=open("debug.txt", "a"))
                    loss_track = self.track_head.loss(*match_feats, *asso_targets)

                    # TODO Currently this operation is done twice
                    # Pass embeddings to store them in the results
                    num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
                    # assert len(num_key_rois) == len(num_ref_rois), "Number of key and ref rois should be the same, maybe one sample does not have gt boxes in the last frame?"
                    key_embeds = torch.split(key_feats, num_key_rois)
        
        # print(f"Loss track {loss_track}", file=open("debug.txt", "a"))

        return loss_track, match_feats, asso_targets, bev_bboxes_key, bev_bboxes_ref, key_embeds

    def forward_test(self,
                      fused_bev_feats_key,
                      detect_bboxes,
                      voxel_sizes,
                      bev_grid_size,
                      img_metas,
                      batch_size,
                    #   proposal_list,
                      gt_labels=None,
                      gt_labels_ref=None,
                    #   ref_img_metas,
                    #   ref_proposals,
                    #   ref_gt_labels,
                      *args,
                      **kwargs):
        if self.with_track:
            num_imgs = len(img_metas) # TODO we do not have images, but bev grids!

            # generate bboxes in correct format (meters -> grid coords)
            # final array should be 6-dimensional: [batch_index, x, y, w, l, angle]
            detect_bboxes = [el[0] for el in detect_bboxes]
            bev_bboxes_key = bbox3d2bevbox(detect_bboxes, voxel_size=voxel_sizes, bev_grid_size=bev_grid_size)

            # generate RoI features
            rois_key = self.track_roi_extractor(fused_bev_feats_key, bev_bboxes_key)
            # and embed them

            #Handle case where there is nothing to track                
            if rois_key.numel() == 0:
                print('EVAL NO ROIS IN BATCH')
                # generate empty tensor with given size
                key_feats = torch.empty((0, 256), device=rois_key.device)
            
            else:
                key_feats = self.track_head(rois_key)
                # TODO use full match when detection bboxes are used

            key_sample_ids = bev_bboxes_key[:, 0].detach().cpu().numpy()
            num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
            # assert len(num_key_rois) == len(num_ref_rois), "Number of key and ref rois should be the same, maybe one sample does not have gt boxes in the last frame?"
            key_embeds = torch.split(key_feats, num_key_rois)
        
        # print(f"Loss track {loss_track}", file=open("debug.txt", "a"))

        return key_embeds
    
    
    def match_ids(self, instance_ids_key, instance_ids_ref, device):
        """Match instance ids between key and ref frame.
        
        """
        # num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
        # num_ref_rois = tuple(np.bincount(ref_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
        # assert len(np.concatenate(instance_ids_key, axis=None)) == len(key_sample_ids), f"Number of instance ids and number of key rois should be the same, they are instead {len(np.concatenate(instance_ids_key, axis=None))} and {len(key_sample_ids)}"
        # assert len(np.concatenate(instance_ids_ref, axis=None)) == len(ref_sample_ids), f"Number of instance ids and number of ref rois should be the same, they are instead {len(np.concatenate(instance_ids_ref, axis=None))} and {len(ref_sample_ids)}"

        # instance_ids_per_sample_key = torch.split(instance_ids_key, num_key_rois)
        # instance_ids_per_sample_ref = torch.split(instance_ids_ref, num_ref_rois)

        gt_match_indices = []
        for _inst_id_key, _inst_id_ref in zip(instance_ids_key, instance_ids_ref):
            # print(f"key {_inst_id_key} ref {_inst_id_ref}", file=open("debug.txt", "a"))
            _match_indices = torch.tensor([
                torch.where(_inst_id_ref==i)[0] if i in _inst_id_ref else -1
                for i in _inst_id_key
            ], device=device)
            gt_match_indices.append(_match_indices)

        return gt_match_indices
    
    def check(self, gt_match_indices, bev_bboxes_key, bev_bboxes_ref, batch_size):
        key_sample_ids = bev_bboxes_key[:, 0]
        ref_sample_ids = bev_bboxes_ref[:, 0]
        num_key_rois = tuple(np.bincount(key_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))
        num_ref_rois = tuple(np.bincount(ref_sample_ids.astype(np.uint8), minlength=batch_size).astype(int))

        print("The following equations should hold:", file=open("debug.txt", "a"))
        for _gt_match_indices, num_key_roi, num_ref_roi in zip(gt_match_indices,
                                                               num_key_rois,
                                                                num_ref_rois):
            print(f"KEY {len(_gt_match_indices) == num_key_roi}", file=open("debug.txt", "a"))
            print(f"REF {max(_gt_match_indices) + 1 <= num_ref_roi}", file=open("debug.txt", "a"))
        
    # def _track_forward(self, x, bboxes):
    #     """Track head forward function used in both training and testing."""
    #     rois = bbox2roi(bboxes)
    #     track_feats = self.track_roi_extractor(
    #         x[:self.track_roi_extractor.num_inputs], rois)
    #     track_feats = self.track_head(track_feats)
    #     return track_feats

    def extract_bbox_feats(self, x, det_bboxes, img_metas):

        if det_bboxes.size(0) == 0:
            return None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])

        return track_feats

def samplres2bboxes(sampling_results, only_pos=False):
    """Convert sampling results to BEV boxes.
    """
    bboxes = []
    # process every sample in the batch
    for idx, single_sample in enumerate(sampling_results):
        # every loop iteration here processes multiple boxes from one sample in the batch
        if only_pos:
            single_sample = single_sample.pos_bboxes
        else:
            single_sample = single_sample.bboxes
        # add batch index to the front
        index_arr = torch.full((single_sample.shape[0], 1), idx, dtype=torch.float32, device=single_sample.device)

        # bev = np.concatenate((index_arr, bev), axis=1)
        bev = torch.cat((index_arr, single_sample), dim=1)

        bboxes.append(bev)
    bboxes = torch.cat(bboxes, dim=0)

    return bboxes

def bbox3d2bevbox(boxes3d: LiDARInstance3DBoxes, voxel_size: tuple=(0.1, 0.1, 0.2), bev_grid_size: tuple=(128, 128)):
    """Convert LiDAR 3D boxes to BEV boxes.
    """
    bboxes = []
    # process every sample in the batch
    for idx, single_sample in enumerate(boxes3d):
        # every loop iteration here processes multiple boxes from one sample in the batch

        # obtains bev XYWHR from 3D box in lidar frame
        bev = single_sample.bev

        if type(bev) == np.ndarray:
            bev = torch.from_numpy(bev).to(boxes3d.device)
        bev = bev.detach()

        # add batch index to the front
        # index_arr = np.full((bev.shape[0], 1), idx)
        index_arr = torch.full((bev.shape[0], 1), idx, dtype=torch.float32, device=bev.device)

        # bev = np.concatenate((index_arr, bev), axis=1)
        bev = torch.cat((index_arr, bev), dim=1)

        bboxes.append(bev)
    
    # bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.cat(bboxes, dim=0)

    # pixel scaling
    bboxes[:, 1] = bboxes[:, 1] / voxel_size[0]
    bboxes[:, 2] = bboxes[:, 2] / voxel_size[1]
    bboxes[:, 3] = bboxes[:, 3] / voxel_size[0]
    bboxes[:, 4] = bboxes[:, 4] / voxel_size[1]
    # pixel translation (origin is in the middle of the image)
    bboxes[:, 1] = bboxes[:, 1] + bev_grid_size[0]/2
    bboxes[:, 2] = bboxes[:, 2] + bev_grid_size[1]/2
    # angles to degrees
    # bboxes[:, 5] = np.rad2deg(bboxes[:, 5])
    bboxes[:, 5] = torch.rad2deg(bboxes[:, 5])

    # print(f"Some sanity check: minx {min(bboxes[:, 1])}  maxx {max(bboxes[:, 1])} miny {min(bboxes[:, 2])} maxy {max(bboxes[:, 2])}", file=open("debug.txt", "a"))
    return bboxes
