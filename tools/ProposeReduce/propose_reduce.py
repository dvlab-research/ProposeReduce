import argparse

import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../libs/cocoapi/PythonAPI'))

import pycocotools.mask as mask_util
import numpy as np

import torch
import torch.nn as nn
import mmcv

from mmdet.core import tensor2imgs
from . import utils
import cv2
from PIL import Image
import math

class ProposeReduce(nn.Module):
    def __init__(self, model, cfg, class_names, use_cate_reduce=False):
        super(ProposeReduce, self).__init__() 
        self.model = model
        self.buffer_x = None
        self.buffer_x_ori = None
        self.propose = cfg.propose
        self.reduce = cfg.reduce
        self.class_names = class_names
        self.use_cate_reduce = use_cate_reduce
        if use_cate_reduce:
            self.cate_reduce = cfg.cate_reduce
        assert ('mem_step' in self.propose) ^ ('mem_num' in self.propose)

    def forward(self, buffer_data, buffer_meta):
        # initilize buffer for saving memory
        self.init_buffer(buffer_data)

        # sequence propose
        key_step = (len(buffer_data)-1) * 1. / (self.propose.key_num-1)

        seq_masks, seq_scores = [], []
        for idx_k in range(self.propose.key_num):
            t = int(key_step * idx_k)
            cur_seq_mask = self.seq_generate(t)
            if cur_seq_mask is None:
                continue
            ## seq scoring
            cur_seq_scores = self.seq_scoring(cur_seq_mask)
            seq_scores += cur_seq_scores
            cur_seq_mask = self.seq_mask2rle(cur_seq_mask)
            seq_masks += cur_seq_mask
        assert len(seq_scores) == len(seq_masks), (len(seq_scores), len(seq_masks))
        if len(seq_masks) == 0:
            return None, None

        # sequence reduce
        seq_scores = torch.stack(seq_scores, dim=0)
        seq_max_scores = seq_scores[:,1:].max(dim=1)[0]
        ## for davis, run seq_soft_nms here
        if self.reduce.type == 'seq_nms':
            seq_id = self.seq_nms(seq_masks, seq_max_scores)
            seq_scores = seq_scores[seq_id]
        elif self.reduce.type == 'seq_soft_nms':
            # assert 0, (len(seq_masks), seq_max_scores.shape)
            seq_id, seq_scores = self.seq_soft_nms(seq_masks, seq_max_scores.clone(), score_thr=self.reduce.score_thr, 
                                                   max_seq_num=self.reduce.max_seq_num, iou_thr=self.reduce.iou_thr)
        else:
            raise NotImplemented(self.reduce.type)
        ret_seq_masks = []
        for k,cur_id in enumerate(seq_id):
            ret_seq_masks.append(seq_masks[cur_id])
        assert len(ret_seq_masks)  == seq_scores.shape[0], (len(ret_seq_masks), seq_scores.shape[0])

        if self.use_cate_reduce:
            ret_seq_masks, seq_scores = self.cate_aware_reduce(ret_seq_masks, seq_scores)

        return ret_seq_masks, seq_scores

    def init_buffer(self, buffer_data):
        self.buffer_data = buffer_data
        self.buffer_x = []
        self.buffer_x_ori = []
        for data in self.buffer_data:
            with torch.no_grad():
                cur_x, cur_x_ori = self.model(return_loss=False, rescale=False, **data)
            self.buffer_x.append(cur_x)
            self.buffer_x_ori.append(cur_x_ori)

    def seq_generate(self, t):
        ## det
        with torch.no_grad():
            data_t = self.buffer_data[t]
            data_t['op'] = 'det'
            data_t['info'] = self.buffer_x[t]
            cur_scores, cur_masks = self.model(return_loss=False, rescale=False, **data_t)
        ## prop
        if cur_masks is None or len(cur_masks) == 0:
            return None
        cur_masks = [torch.FloatTensor(m).cuda() for m in cur_masks]
        cur_masks = utils.MaskNMS(cur_masks)
        if len(cur_masks) == 0:
            return None
        cur_masks = torch.stack(cur_masks, dim=0) # [K,H,W]
        # memorize self
        with torch.no_grad():
            data_t = self.buffer_data[t]
            data_t['op'] = 'prop-memorize'
            data_t['info'] = [self.buffer_x[t], cur_masks]
            mem_key, mem_value  = self.model(return_loss=False, rescale=False, **data_t)
        mem_key, mem_value = mem_key[:,:,None], mem_value[:,:,None]
        with torch.no_grad():
            data_t = self.buffer_data[t]
            data_t['op'] = 'prop-segment'
            data_t['info'] = [self.buffer_x[t], self.buffer_x_ori[t], mem_key, mem_value]
            cur_pred = self.model(return_loss=False, rescale=False, **data_t)
            cur_pred = torch.softmax(cur_pred, dim=1)
        cur_pred = cur_pred[0,1:] # [K,H,W]
        det_pred = [cur_pred]
        if 'mem_num' in self.propose:
            mem_step = max(len(self.buffer_data) // self.propose.mem_num, 1)
        else:
            mem_step = self.propose.mem_step
        # |-->
        cur_mem_key = mem_key.clone()
        cur_mem_value = mem_value.clone()
        for tp in range(t+1,len(self.buffer_data)):
            with torch.no_grad():
                data_tp = self.buffer_data[tp]
                data_tp['op'] = 'prop-segment'
                data_tp['info'] = [self.buffer_x[tp], self.buffer_x_ori[tp], cur_mem_key, cur_mem_value]
                cur_pred = self.model(return_loss=False, rescale=False, **data_tp)
                cur_pred = torch.softmax(cur_pred, dim=1)
            cur_pred = cur_pred[0,1:] # [K,H,W]
            det_pred.append(cur_pred)
            if (tp-t) % mem_step != 0:
                continue
            with torch.no_grad(): 
                data_tp['op'] = 'prop-memorize'
                data_tp['info'] = [self.buffer_x[tp], cur_pred]
                cur_key, cur_value = self.model(return_loss=False, rescale=False, **data_tp)
            cur_key, cur_value = cur_key[:,:,None], cur_value[:,:,None]
            cur_mem_key = torch.cat((cur_mem_key, cur_key), dim=2)
            cur_mem_value = torch.cat((cur_mem_value, cur_value), dim=2)
        # <--|
        cur_mem_key = mem_key.clone()
        cur_mem_value = mem_value.clone()
        for tp in range(t-1, -1, -1):
            data_tp = self.buffer_data[tp]
            with torch.no_grad():
                data_tp = self.buffer_data[tp]
                data_tp['op'] = 'prop-segment'
                data_tp['info'] = [self.buffer_x[tp], self.buffer_x_ori[tp], cur_mem_key, cur_mem_value]
                cur_pred = self.model(return_loss=False, rescale=False, **data_tp)
                cur_pred = torch.softmax(cur_pred, dim=1)
            cur_pred = cur_pred[0,1:] # [K,H,W]
            det_pred = [cur_pred] + det_pred
            if (t-tp) % mem_step != 0:
                continue
            with torch.no_grad():
                data_tp['op'] = 'prop-memorize'
                data_tp['info'] = [self.buffer_x[tp], cur_pred]
                cur_key, cur_value = self.model(return_loss=False, rescale=False, **data_tp)
            cur_key, cur_value = cur_key[:,:,None], cur_value[:,:,None]
            cur_mem_key = torch.cat((cur_mem_key, cur_key), dim=2)
            cur_mem_value = torch.cat((cur_mem_value, cur_value), dim=2)
        assert len(det_pred) == len(self.buffer_x)

        return det_pred

    def seq_scoring(self, det_pred):
        ## get seq scores
        cur_seq_scores = []
        seq_det_scores_dict = dict()
        for tp in range(len(det_pred)):
            det_id = []
            det_bboxes = []
            for k in range(det_pred[tp].shape[0]):
                if (det_pred[tp][k]>=0.5).sum() > 0:
                    cur_box = utils.mask2box((det_pred[tp][k]>=0.5).long().cpu().numpy())
                    det_bboxes.append(cur_box)
                    det_id.append(k)
            if len(det_bboxes) > 0:
                det_bboxes = np.stack(det_bboxes, 0).astype(np.float) # [K,4]
                with torch.no_grad():
                    data_tp = self.buffer_data[tp]
                    data_tp['op'] = 'cls'
                    data_tp['info'] = (self.buffer_x[tp], torch.FloatTensor(det_bboxes).cuda())
                    cls_score = self.model(return_loss=False, rescale=False, **data_tp)
                assert cls_score.shape[0] == det_bboxes.shape[0], (cls_score.shape, det_bboxes.shape)
                for k,obj_id in enumerate(det_id):
                    if obj_id not in seq_det_scores_dict:
                        seq_det_scores_dict[obj_id] = []
                    seq_det_scores_dict[obj_id].append(cls_score[k])
        ## score summary
        for k in range(det_pred[0].shape[0]):
            if k not in seq_det_scores_dict:
                cur_det_score = torch.zeros(len(self.class_names)+1).cuda().float()
                cur_det_score[0] = 1 # bg
            else:
                cur_det_score = torch.stack(seq_det_scores_dict[k], 0).mean(0)
            cur_seq_scores.append(cur_det_score)
        return cur_seq_scores

    def seq_mask2rle(self, det_pred):
        ## save --> rle
        cur_seq_rles = []
        for k in range(det_pred[0].shape[0]):
            cur_seq = []
            for tp in range(len(det_pred)):
                cur_mask = (det_pred[tp][k]>=0.5).cpu().numpy().astype(np.uint8)
                rle = mask_util.encode(np.array(cur_mask, order='F'))
                # rle['counts'] = str(rle['counts'])
                cur_seq.append(rle)
            cur_seq_rles.append(cur_seq)
        return cur_seq_rles

    def seq_nms(self, seq_masks, seq_scores):
        assert len(seq_scores.shape) == 1, seq_scores.shape # [K]
        assert len(seq_masks) == seq_scores.shape[0], (len(seq_masks), seq_scores.shape)
        flag = [False for _ in range(seq_scores.shape[0])]
        seq_id = []
        for _ in range(seq_scores.shape[0]):
            max_idx = -1
            for k in range(seq_scores.shape[0]):
                if not flag[k] and (max_idx == -1 or seq_scores[k] > seq_scores[max_idx]):
                    max_idx = k
            if max_idx == -1:
                break
            if seq_scores[max_idx] < self.reduce.score_thr and len(seq_id) > 0:
                break
            flag[max_idx] = True
            seq_id.append(max_idx)
            for k in range(seq_scores.shape[0]):
                if not flag[k]:
                    ## We found using rle to calculate IoU is faster
                    cur_iou = utils.seq_match_iou_rle(seq_masks[max_idx], seq_masks[k])
                    if cur_iou >= self.reduce.iou_thr:
                        flag[k] = True
        assert len(seq_id) > 0
        seq_id = torch.LongTensor(seq_id).cuda()
        return seq_id

    def seq_soft_nms(self, seq_masks, seq_scores, max_seq_num, score_thr, iou_thr):
        assert len(seq_scores.shape) == 1, seq_scores.shape # [K]
        assert len(seq_masks) == seq_scores.shape[0], (len(seq_masks), seq_scores.shape)
        flag = [False for _ in range(seq_scores.shape[0])]
        seq_id = []
        ret_seq_scores = []
        for _ in range(seq_scores.shape[0]):
            max_idx = -1
            for k in range(seq_scores.shape[0]):
                if not flag[k] and (max_idx == -1 or seq_scores[k] > seq_scores[max_idx]):
                    max_idx = k
            if max_idx == -1:
                break
            if seq_scores[max_idx] < score_thr:
                break
            flag[max_idx] = True
            seq_id.append(max_idx)
            ret_seq_scores.append(seq_scores[max_idx][None])
            if len(seq_id) >= max_seq_num:
                break
            for k in range(seq_scores.shape[0]):
                if not flag[k] and seq_scores[k] >= score_thr:
                    cur_iou = utils.seq_match_iou_rle(seq_masks[max_idx], seq_masks[k])
                    seq_scores[k] *= math.exp(-(cur_iou**2)/iou_thr)
        if len(seq_id) > 0:
            ret_seq_scores = torch.cat(ret_seq_scores, dim=0)
        else:
            ret_seq_scores = None
        return seq_id, ret_seq_scores

    def cate_aware_reduce(self, seq_masks, seq_scores):
        res_seq_scores = seq_scores.clone()
        for c in range(1, seq_scores.shape[1]):
            cur_seq_id, cur_seq_scores = self.seq_soft_nms(seq_masks, seq_scores[:, c].clone(), score_thr=self.cate_reduce.score_thr, 
                                                           max_seq_num=self.cate_reduce.max_seq_num, iou_thr=self.cate_reduce.iou_thr)
            for k in range(res_seq_scores.shape[0]):
                if k not in cur_seq_id:
                    res_seq_scores[k,c] = -1 # abandon
            for idx,k in enumerate(cur_seq_id):
                res_seq_scores[k,c] = cur_seq_scores[idx]
        return seq_masks, res_seq_scores

    def show_result(self, 
                    buffer_data,
                    buffer_meta,
                    seq_masks,
                    seq_scores,
                    img_norm_cfg,
                    save_dir='save_dir',
                    score_thr=0.2,
                    use_img=False,
                    ret_file=False):
        palette = Image.open('.palette/00000.png').getpalette()

        res_list = []
        for t in range(len(buffer_data)):
            img_tensor = buffer_data[t]['img'][0]
            img_meta = buffer_meta[t]

            img = tensor2imgs(img_tensor, **img_norm_cfg)[0]
            obj_scores, obj_label = seq_scores.max(1)

            res_masks = []
            bboxes = []
            labels = []
            for k in range(seq_scores.shape[0]): # seq_score is sorted descending
                if obj_scores[k] < score_thr:
                    continue
                mask_t = mask_util.decode(seq_masks[k][t])
                res_masks.append(mask_t)
                x,y,w,h = cv2.boundingRect(mask_t)
                bboxes.append(np.array([x, y, x+w, y+h, obj_scores[k]]))
                labels.append(obj_label[k]-1)
            # mask
            res_masks = [np.zeros(img_meta['pad_shape'][:2])] + res_masks
            res_masks = np.stack(res_masks).argmax(0).astype(np.uint8)
            res_mask = Image.fromarray(res_masks)
            res_mask.putpalette(palette)
            color_mask = res_mask.convert('RGB')
            color_mask = np.array(color_mask)[...,::-1]
            color_mask = color_mask.astype(np.uint8)
            # bbox
            bboxes = np.vstack(bboxes)
            labels = np.array(labels)

            h, w, _ = img_meta['img_shape']
            if use_img:
                ratio = 0.7
                blend = color_mask * ratio + img * (1-ratio)
                bg = (np.array(res_mask) == 0)
                blend[bg] = img[bg]
                img_show = blend[:h, :w, :].copy()
            else:
                img_show = color_mask[:h, :w, :].copy()

            if ret_file:
                img_show = mmcv.imshow_det_bboxes(
                        img_show,
                        bboxes,
                        labels,
                        class_names=self.class_names,
                        show=False,
                        text_color ='white',
                        ret_file=True)
                res_list.append(img_show)
            else:
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                cur_save_path = osp.join(save_dir, '%s.jpg' % img_meta['frame_name'])
                mmcv.imshow_det_bboxes(
                    img_show,
                    bboxes,
                    labels,
                    class_names=self.class_names,
                    show=False,
                    text_color ='white',
                    out_file=cur_save_path)
        if ret_file:
            return res_list
