import mmcv
import numpy as np
import torch

from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device, proposals=None, spec_ind=None):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    if proposals is not None:
        ori_shape = img.shape[:2][::-1]
        img_shape = cfg.data.test.img_scale
        rat_w = img_shape[0]*1./ori_shape[0]
        rat_h = img_shape[1]*1./ori_shape[1]
        proposals[0,0] *= rat_w
        proposals[0,2] *= rat_w
        proposals[0,1] *= rat_h
        proposals[0,3] *= rat_h
        data['proposals'] = [torch.FloatTensor(proposals).cuda()]
    data['spec_ind'] = spec_ind
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0', proposals=None, spec_ind=None):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device, proposals, spec_ind)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3):
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)
