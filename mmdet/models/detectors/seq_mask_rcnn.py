from .two_stage import TwoStageDetector
from .. import builder
from ..registry import DETECTORS

from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.core import bbox_overlaps, bbox2result_with_id
from mmdet.core import delta2bbox

import numpy as np
import torch
import torch.nn.functional as F
import timeit


@DETECTORS.register_module
class SeqMaskRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 prop_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(SeqMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if prop_head is not None:
            self.prop_head = builder.build_head(prop_head)

        self.init_weights(pretrained=pretrained)

        self.target_means=[0., 0., 0., 0.]
        self.target_stds=[0.1, 0.1, 0.2, 0.2]

    def init_weights(self, pretrained=None):
        super(SeqMaskRCNN, self).init_weights(pretrained)
        # assert self.with_prop, "Prop head must be implemented"
        self.prop_head.init_weights()

    def extract_feat(self, img, ret_ori=False):
        x = self.backbone(img)
        if not ret_ori:
            if self.with_neck:
                x = self.neck(x)
            return x
        else:
            if self.with_neck:
                y = self.neck(x)
            return y, x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      ref_labels,
                      prop_ref_labels,
                      ref_img, # images of reference frame
                      ref_bboxes, # gt bbox of reference frame
                      prop_ref_bboxes,
                      gt_pids, # gt ids of current frame bbox mapped to reference frame
                      ref_gt_pids, # ref --> gt
                      prop_gt_pids,
                      gt_masks=None,
                      ref_masks=None,
                      prop_ref_masks=None,
                      proposals=None):
        ''''''

        x, x_ori = self.extract_feat(img, ret_ori=True)
        ref_x = self.extract_feat(ref_img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i], gt_pids[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # fetch bbox and object_id targets
            bbox_targets, (ids, id_weights) = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)
            ori_shape = (ref_x[0].shape[-2]*4, ref_x[0].shape[-1]*4)
            for k in range(len(ref_masks)):
                if ref_masks[k].shape[-2:] != ori_shape:
                    new_mask = np.zeros((ref_masks[k].shape[0],) + ori_shape, dtype=np.uint8)
                    new_mask[:,:ref_masks[k].shape[1],:ref_masks[k].shape[2]] = ref_masks[k]
                    ref_masks[k] = new_mask
                if gt_masks[k].shape[-2:] != ori_shape:
                    new_mask = np.zeros((gt_masks[k].shape[0],) + ori_shape, dtype=np.uint8)
                    new_mask[:,:gt_masks[k].shape[1],:gt_masks[k].shape[2]] = gt_masks[k]
                    gt_masks[k] = new_mask 
            ## stm ##
            ref_rois = bbox2roi(prop_ref_bboxes)
            ref_bbox_img_n = [cur_box.size(0) for cur_box in prop_ref_bboxes]
            ref_mask_feats = self.mask_roi_extractor(
                ref_x[:self.mask_roi_extractor.num_inputs], ref_rois)
            ref_mask_pred = self.mask_head(ref_mask_feats).detach()
            ref_mask_pred = torch.split(ref_mask_pred, ref_bbox_img_n, dim=0)
            ref_mask_pred2ori = []

            for k in range(len(ref_mask_pred)):
                assert prop_ref_labels[k].min() > 0, (prop_ref_labels[k])
                assert prop_ref_bboxes[k].shape[0] == prop_ref_labels[k].shape[0], (prop_ref_bboxes[k].shape, prop_ref_labels[k].shape)
                try:
                    cur_mask = self.mask_head.get_seg_masks_tensor(
                        ref_mask_pred[k], prop_ref_bboxes[k], prop_ref_labels[k]-1, self.test_cfg.rcnn, img_meta[k]['ori_shape'],
                        img_meta[k]['scale_factor'], rescale=False)
                except Exception as e:
                    print('e', e)
                    assert 1<0, (k, prop_ref_bboxes[k], prop_ref_labels[k], img_meta[k], ref_mask_pred[k].shape)
                ref_mask_pred2ori.append(cur_mask)

            prop_preds = self.prop_head(ref_x[0], x[0], x_ori[:2], ref_mask_pred2ori, img_meta)
            prop_targets = self.prop_head.get_target(gt_masks, prop_gt_pids)
            assert len(prop_preds) == len(prop_targets)
            for bs in range(len(prop_preds)):
                assert prop_preds[bs].shape[1] == ref_mask_pred2ori[bs].shape[0]+1, (bs, prop_preds[bs].shape, 
                                                                             ref_mask_pred2ori[bs].shape)
            cur_pred = torch.softmax(prop_preds[0].detach(), dim=1)
            loss_prop = self.prop_head.loss(prop_preds, prop_targets)
            losses.update(loss_prop)

        return losses


    def simple_test(self, img, img_meta, proposals=None, rescale=False, ret_cls_only=False,
                    ret_det_cls_atn=False, prop=None, op='feat', info=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        assert op in ['feat', 'det', 'prop-memorize', 'prop-segment', 'cls']

        if op == 'feat':
            x, x_ori = self.extract_feat(img, ret_ori=True) 
            return (x[0].cpu(), x[1].cpu(), x[2].cpu(), x[3].cpu(), x[4].cpu()), (x_ori[0].cpu(), x_ori[1].cpu()) # len: #5, #2
        elif op == 'det':
            x = info
            x = (x[0].cuda(), x[1].cuda(), x[2].cuda(), x[3].cuda(), x[4].cuda())
            #
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
            det_bboxes, det_labels = self.simple_test_bboxes( 
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            det_obj_ids = np.arange(det_bboxes.size(0))
            bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                               self.bbox_head.num_classes)

            if len(bbox_results.keys()) == 0:
                return None, None
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels,
                rescale=rescale, det_obj_ids=det_obj_ids)
            det_scores = []
            det_masks = []
            for k in sorted(bbox_results.keys()):
                det_scores.append(bbox_results[k]['bbox'][-1])
                det_masks.append(segm_results[k])
            det_masks = np.stack(det_masks, 0)
            return det_scores, det_masks
        elif op == 'prop-memorize':
            x, masks = info
            keys, values = self.prop_head.memorize(x[0].cuda(), masks)
            return keys, values
        elif op == 'prop-segment':
            if len(info) == 4:
                x, x_ori, mem_keys, mem_values = info
                flag_softagg = True
            else:
                x, x_ori, mem_keys, mem_values, flag_softagg = info
            assert isinstance(flag_softagg, bool), flag_softagg
            qry_key, qry_value = self.prop_head.query(x[0].cuda())
            preds = self.prop_head.segment(mem_keys, mem_values, qry_key, qry_value, x_ori[0], x_ori[1])
            if flag_softagg:
                preds = self.prop_head.soft_agg(preds)
            return preds
        elif op == 'cls':
            x, proposal_list = info
            x = (x[0].cuda(), x[1].cuda(), x[2].cuda(), x[3].cuda(), x[4].cuda())
            scale_factor = img_meta[0]['scale_factor']
            rois = bbox2roi([proposal_list])
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, _ = self.bbox_head(roi_feats)
            cls_score = F.softmax(cls_score, 1)
            return cls_score 
        else:
            raise NotImplemented(op)
