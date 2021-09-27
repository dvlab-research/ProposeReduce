from .two_stage import TwoStageDetector
from ..registry import DETECTORS

from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.core import bbox_overlaps, bbox2result_with_id
from mmdet.core import delta2bbox

import numpy as np
import torch
import torch.nn.functional as F
import timeit


@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):
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
        super(MaskRCNN, self).__init__(
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
