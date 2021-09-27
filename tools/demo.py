import argparse

import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../libs/mmcv'))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../libs/cocoapi/PythonAPI'))

import pycocotools.mask as mask_util 
import numpy as np

import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor
from mmdet.core import get_classes
from mmdet.models import build_detector
import pickle as pkl
import cv2
import math
from tqdm import tqdm
from PIL import Image

from ProposeReduce import ProposeReduce

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_divisor = 32
img_transform = ImageTransform(
    size_divisor=size_divisor, **img_norm_cfg)

def prepare_img(input_dir, video_name, frame_name, scale=(640,360), flip=False):
    img = mmcv.imread(osp.join(input_dir, video_name, frame_name))
    _img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale, flip, keep_ratio=True)
    _img = to_tensor(_img)[None]
    _img_meta = dict(
        ori_shape=img.shape,
        img_shape=img_shape,
        pad_shape=pad_shape,
        frame_id =frame_name.split('.')[0],
        scale_factor=scale_factor,
        flip=flip,
        video_name = video_name,
        frame_name = frame_name.split('.')[0],
    )
    return _img, _img_meta

def prepare_video(img, scale=(640,360), flip=False):
    _img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale, flip, keep_ratio=True)
    _img = to_tensor(_img)[None]
    _img_meta = dict(
        ori_shape=img.shape,
        img_shape=img_shape,
        pad_shape=pad_shape,
        scale_factor=scale_factor,
        flip=flip,
    )
    return _img, _img_meta

def read_input(root, vid):
    buffer_data = []
    buffer_meta = []
    if osp.isdir(osp.join(root, vid)):
        for frame in sorted(os.listdir(osp.join(root, vid))):
            img, img_meta = prepare_img(root, vid, frame)
            data = dict(img=[img], img_meta=[[img_meta]])
            buffer_meta.append(img_meta)
            buffer_data.append(data)
    elif vid[-4:] in ['.mp4']:
        cap = cv2.VideoCapture(osp.join(root, vid))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img, img_meta = prepare_video(frame)
            data = dict(img=[img], img_meta=[[img_meta]])
            buffer_meta.append(img_meta)
            buffer_data.append(data)
        cap.release()
    else:
        raise NotImplemented
    return buffer_data, buffer_meta

def write_output(res_list, root_src, root_dst, vid, buffer_meta):
    if osp.isdir(osp.join(root_src, vid)):
        save_dir = osp.join(root_dst, vid)
        os.makedirs(save_dir, exist_ok=True)
        for t,res in enumerate(res_list):
            cur_save_path = osp.join(save_dir, '%s.jpg' % buffer_meta[t]['frame_name'])
            cv2.imwrite(cur_save_path, res)
    elif vid[-4:] in ['.mp4']:
        cap = cv2.VideoCapture(osp.join(root_src, vid))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = res_list[0].shape[:2]
        writer = cv2.VideoWriter(osp.join(root_dst, vid), fourcc, fps, (width,height))
        for res in res_list:
            writer.write(res)
        cap.release()
        writer.release()
    else:
        raise NotImplemented

def single_test(args, infer_paradigm):
    for vid in tqdm(os.listdir(args.input_dir)):
        buffer_data, buffer_meta = read_input(args.input_dir, vid)
        ## infer
        seq_masks, seq_scores = infer_paradigm(buffer_data, buffer_meta)
        if seq_masks is None: 
            print('%s has no segmented sequences, continue...' % vid)
            continue
        res_list = infer_paradigm.show_result(buffer_data, buffer_meta, seq_masks, seq_scores, img_norm_cfg,
                                              ret_file=True, use_img=args.use_img)
        write_output(res_list, args.input_dir, args.save_dir, vid, buffer_meta)
            

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save_path', 
        type=str,
        help='path to save visual result')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--gpu', default='0', type=str, help='GPU Index')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--load_result', 
        action='store_true', 
        help='whether to load existing result')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    parser.add_argument('--max-agg', action='store_true')
    parser.add_argument('--score-thr', type=int, default=0.1)
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--key-num', type=int, default=5)
    parser.add_argument('--mem-step', type=int, default=5)
    parser.add_argument('--network', type=str, default='x101')
    parser.add_argument('--use-img', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('args', args)

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])
    
    ## paradigm 
    cfg.test_cfg.paradigm.propose.key_num = args.key_num
    cfg.test_cfg.paradigm.propose.mem_step = args.mem_step
    if cfg.test_cfg.paradigm.type == 'Propose_Reduce':
        infer_paradigm = ProposeReduce(model.eval(), cfg.test_cfg.paradigm, get_classes('ytvos19'))
    else:
        raise NotImplemented

    single_test(args, infer_paradigm)

if __name__ == '__main__':
    main()
