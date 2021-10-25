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

from mmdet import datasets
from mmdet.core import get_classes
from mmdet.datasets import build_dataloader 
from mmdet.models import build_detector, detectors
from tqdm import tqdm
import timeit
import json
from PIL import Image

from ProposeReduce import ProposeReduce

def save_PIL(seq_masks, seq_scores, buffer_img_meta, save_dir, score_thr=0.001):
    seq_scores = seq_scores.cpu()
    palette = Image.open('.palette/00000.png').getpalette()
    for tp in range(len(seq_masks[0])):
        img_meta = buffer_img_meta[tp]
        ori_shape = img_meta['ori_shape'][:2]
        img_shape = img_meta['img_shape'][:2]
        res_mask = []
        for k in range(seq_scores.shape[0]):
            cur_mask = mask_util.decode(seq_masks[k][tp])
            assert len(cur_mask.shape) == 2, (img_meta,  cur_mask.shape)
            if cur_mask.shape != img_shape:
                cur_mask = cur_mask[:img_shape[0], :img_shape[1]].copy()
            if cur_mask.shape != ori_shape:
                cur_mask = torch.FloatTensor(cur_mask).cuda()
                cur_mask = F.interpolate(cur_mask[None,None], ori_shape, mode='bilinear', align_corners=False)[0,0]
                cur_mask = (cur_mask>0.5).cpu().numpy().astype(np.uint8)
            res_mask.append(cur_mask.astype(np.uint8))
        res_mask = np.stack(res_mask, axis=0)
        res_mask = np.concatenate((np.zeros(res_mask.shape[1:], dtype=np.uint8)[None], res_mask), axis=0)
        res_mask = np.argmax(res_mask, axis=0).astype(np.uint8)
        # save
        res_mask = Image.fromarray(res_mask)
        res_mask.putpalette(palette)
        res_mask.save(osp.join(save_dir, '%s.png' % img_meta['frame_name']))

def single_test(args, infer_paradigm, data_loader, show=False, save_path=''):
    dataset = data_loader.dataset
    buffer_data = []
    buffer_meta = []
    proj_name = 'vis_%s-k%s' % (args.network, args.key_num)
    tot_times = []
    tot_frames = []
    print('proj_name', proj_name)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        img_meta = data['img_meta'][0].data[0][0]
        buffer_meta.append(img_meta)
        buffer_data.append(data)

        if img_meta['is_last']:
            torch.cuda.synchronize()
            ts = timeit.default_timer()

            vid_name = img_meta['video_name']
            save_dir_PIL = osp.join(args.save_dir, '%s-seqs_PIL' % proj_name, vid_name)
            os.makedirs(save_dir_PIL, exist_ok=True)

            seq_masks, seq_scores = infer_paradigm(buffer_data, buffer_meta)
            if seq_masks is None:
                buffer_data, buffer_meta = [], []
                batch_size = data['img'][0].size(0)
                for _ in range(batch_size):
                    prog_bar.update()
                continue

            torch.cuda.synchronize()
            te = timeit.default_timer()
            tot_times.append(te-ts)
            tot_frames.append(len(buffer_data))

            ## show
            if show:
                cur_save_dir = osp.join(args.save_dir, '%s-seqs' % proj_name, vid_name)
                infer_paradigm.show_result(buffer_data, buffer_meta, seq_masks, seq_scores, dataset.img_norm_cfg,
                                           save_dir=cur_save_dir)

            save_PIL(seq_masks, seq_scores, buffer_meta, save_dir_PIL, args.score_thr)

            buffer_data, buffer_meta = [], []

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    print()
    print('frames', sum(tot_frames), len(tot_frames))
    print('times', sum(tot_times), len(tot_times))
    print('avg-time', sum(tot_times) / sum(tot_frames))


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
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load_result', 
        action='store_true', 
        help='whether to load existing result')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--max-agg', action='store_true')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--key-num', type=int)
    parser.add_argument('--network', type=str, default='x101')
    parser.add_argument('--use-cate-reduce', action='store_true')
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
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    ## paradigm
    if cfg.test_cfg.paradigm.type == 'Propose_Reduce':
        if args.key_num is not None:
            cfg.test_cfg.paradigm.propose.key_num = args.key_num
            print('Set Key Num', cfg.test_cfg.paradigm.propose.key_num)
        else:
            print('Config Key Num', cfg.test_cfg.paradigm.propose.key_num)
        infer_paradigm = ProposeReduce(model.eval(), cfg.test_cfg.paradigm, get_classes(cfg.test_cfg.paradigm.classes),
                                       use_cate_reduce=args.use_cate_reduce)
    else:
        raise NotImplemented

    single_test(args, infer_paradigm, data_loader, args.show, save_path=args.save_path)

if __name__ == '__main__':
    main()
