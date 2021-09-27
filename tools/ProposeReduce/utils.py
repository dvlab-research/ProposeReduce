import cv2
import numpy as np
import torch
import pycocotools.mask as maskUtils

def MaskNMS(mask_list, iou_thr=0.2):
    new_masks = []
    for cand_mask in mask_list:
        flag = True
        for ref_mask in new_masks:
            cur_iou = img_match_iou(cand_mask[None], ref_mask[None])
            if cur_iou >= iou_thr:
                flag = False
                break
        if flag:
            new_masks.append(cand_mask)
    return new_masks

def seq_match_iou_rle(seq0, seq1):
    assert len(seq0) == len(seq1), (len(seq0), len(seq1))
    i = .0
    u = .0
    for d, g in zip(seq0, seq1):
        if d and g:
            i += maskUtils.area(maskUtils.merge([d, g], True))
            u += maskUtils.area(maskUtils.merge([d, g], False))
        elif not d and g:
            u += maskUtils.area(g)
        elif d and not g:
            u += maskUtils.area(d)
    iou = i / u if u > .0 else .0
    return iou

def seq_match_iou(seq0, seq1): 
    assert len(seq0.shape) == 4 and seq0.shape[0] > 0, (seq0.shape)
    assert len(seq1.shape) == 4 and seq1.shape[0] > 0, (seq1.shape)
    seq0 = seq0.view(seq0.shape[0], -1) # [K1,THW]
    seq1 = seq1.view(seq1.shape[0], -1) # [K2,THW]
    intersec = torch.mm(seq0, seq1.t()) # [K1,K2]
    union = (seq0[:,None] + seq1[None] - seq0[:,None]*seq1[None]).sum(2) # [K1,K2]
    iou = intersec / union
    return iou

def img_match_iou(seq0, seq1): 
    assert len(seq0.shape) == 3 and seq0.shape[0] > 0, (seq0.shape)
    assert len(seq1.shape) == 3 and seq1.shape[0] > 0, (seq1.shape)
    seq0 = seq0.view(seq0.shape[0], -1) # [K1,HW]
    seq1 = seq1.view(seq1.shape[0], -1) # [K2,HW]
    intersec = torch.mm(seq0, seq1.t()) # [K1,K2]
    union = (seq0[:,None] + seq1[None] - seq0[:,None]*seq1[None]).sum(2) # [K1,K2]
    iou = intersec / union
    return iou

def mask2box(mask_v, expand_pixel=0):
    mask = np.asarray(mask_v)
    if np.sum(mask == 1) == 0:
        x0 = 0
        y0 = 0
        x1 = mask.shape[1] - 1
        y1 = mask.shape[0] - 1
    else:
        coord = np.array(np.nonzero(mask == 1)).transpose()
        x0 = coord[:, 1].min()
        x1 = coord[:, 1].max()
        y0 = coord[:, 0].min()
        y1 = coord[:, 0].max()

        x0 = max(x0 - 1 - expand_pixel, 0)
        y0 = max(y0 - 1 - expand_pixel, 0)
        x1 = min(x1 + 1 + expand_pixel, mask.shape[1] - 1)
        y1 = min(y1 + 1 + expand_pixel, mask.shape[0] - 1)

    box = np.array([x0, y0, x1, y1], dtype=np.int32)
    return box