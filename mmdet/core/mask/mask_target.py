import torch
import numpy as np
import mmcv
from ..utils import multi_apply


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def masktrack_target(pos_proposals_list, pos_assigned_gt_inds_list, pos_gt_pids_list,
                     gt_masks_list, ref_gt_masks_list, ref_bboxes_list,
                    cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets, cls_targets, cur_mask_targets, ref_mask_targets = multi_apply(masktrack_target_single,
                        pos_proposals_list, pos_assigned_gt_inds_list,
                       pos_gt_pids_list, gt_masks_list, ref_gt_masks_list, ref_bboxes_list,
                       cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    cls_targets = torch.cat(list(cls_targets))
    cur_mask_targets = torch.cat(list(cur_mask_targets))
    ref_mask_targets = torch.cat(list(ref_mask_targets))
    return mask_targets, cls_targets, cur_mask_targets, ref_mask_targets


def masktrack_target_single(pos_proposals, pos_assigned_gt_inds, pos_gt_pids,
                            gt_masks, ref_gt_masks, ref_bboxes, cfg):
    mask_size = cfg.mask_size
    track_size = cfg.track_size
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        '''mask'''
        mask_targets = []
        cur_mask_targets = []
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            if cfg.mask_det > 0:
                x1 = max(x1-cfg.mask_det, 0)
                y1 = max(y1-cfg.mask_det, 0)
                x2 = min(x2+cfg.mask_det, gt_masks.shape[1]-1)
                y2 = min(y2+cfg.mask_det, gt_masks.shape[0]-1)
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            cur_target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (track_size, track_size))
            mask_targets.append(target)
            cur_mask_targets.append(cur_target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
        cur_mask_targets = torch.from_numpy(np.stack(cur_mask_targets)).float().to(
            pos_proposals.device)
        '''track'''
        cls_targets = []
        ref_mask_targets = []
        ref_bboxes_np = ref_bboxes.cpu().numpy()
        pos_gt_pids = pos_gt_pids.cpu().numpy()
        for i in range(num_pos):
            if pos_gt_pids[i] == 0:
                cls_targets.append(np.ones((1,)))
                ref_mask_targets.append(np.zeros((track_size, track_size)))
            else:
                cls_targets.append(np.zeros((1,)))
                '''ref'''
                ref_gt_mask = ref_gt_masks[pos_gt_pids[i]-1]
                ref_bbox = ref_bboxes_np[pos_gt_pids[i]-1, :].astype(np.int32)
                x1, y1, x2, y2 = ref_bbox
                if cfg.mask_det > 0:
                    x1 = max(x1-cfg.mask_det, 0)
                    y1 = max(y1-cfg.mask_det, 0)
                    x2 = min(x2+cfg.mask_det, ref_gt_masks.shape[1]-1)
                    y2 = min(y2+cfg.mask_det, ref_gt_masks.shape[0]-1)
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                # mask is uint8 both before and after resizing
                target = mmcv.imresize(ref_gt_mask[y1:y1 + h, x1:x1 + w],
                                        (track_size, track_size))
                ref_mask_targets.append(target)

        cls_targets = torch.from_numpy(np.stack(cls_targets)).float().to(
            pos_proposals.device)
        ref_mask_targets = torch.from_numpy(np.stack(ref_mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
        cur_mask_targets = pos_proposals.new_zeros((0, track_size, track_size))
        ref_mask_targets = pos_proposals.new_zeros((0, track_size, track_size))
        cls_targets = pos_proposals.new_zeros((0))

    return mask_targets, cls_targets, cur_mask_targets, ref_mask_targets

