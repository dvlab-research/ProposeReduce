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
                      proposals=None):
        pass

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
