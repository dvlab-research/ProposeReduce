import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..registry import HEADS
from ..utils import ConvModule

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x)) 
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + r 

class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

@HEADS.register_module
class SeqPropHead(nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super(SeqPropHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels
                           if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        ## encoderM
        mdim = 256
        self.enc_pos = nn.Conv2d(1, mdim, kernel_size=7, padding=3)
        self.enc_fuse = nn.Sequential(ResBlock(mdim, mdim, 2),
                                      ResBlock(mdim, mdim*2, 2),
                                     )
        self.atn_key = nn.Conv2d(mdim*2, mdim//4, kernel_size=1, padding=0)
        self.atn_value = nn.Conv2d(mdim*2, mdim, kernel_size=1, padding=0)
        ## encoderQ
        self.qry_fuse = nn.Sequential(ResBlock(mdim, mdim, 2),
                                      ResBlock(mdim, mdim*2, 2),
                                     )
        self.qry_atn_key = nn.Conv2d(mdim*2, mdim//4, kernel_size=1, padding=0)
        self.qry_atn_value = nn.Conv2d(mdim*2, mdim, kernel_size=1, padding=0)
        ## decoder
        self.convFM = nn.Conv2d(mdim*2, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(mdim*2, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(mdim, mdim) # 1/4 -> 1
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    def memorize(self, P2, masks):
        if masks.shape[-2] != P2.shape[-2] * 4 or masks.shape[-1] != P2.shape[-1] * 4:
            pad_right = max(0, P2.shape[-1] * 4 - masks.shape[-1])
            pad_down = max(0, P2.shape[-2] * 4 - masks.shape[-2])
            masks = F.pad(masks, (0,pad_right,0,pad_down))
        # assert masks.shape[-2] == P2.shape[-2] * 4 and masks.shape[-1] == P2.shape[-1] * 4, (masks.shape, P2.shape)
        assert abs(masks.shape[-2] - P2.shape[-2] * 4) <= 1 and abs(masks.shape[-1] - P2.shape[-1] * 4) <= 1, (masks.shape, P2.shape)
        masks = F.interpolate(masks[:,None], P2.shape[-2:], mode='bilinear', align_corners=False) # [K,1,H,W]
        B = masks.shape[0]
        x = P2.expand(B,-1,-1,-1) + self.enc_pos(masks)
        x = self.enc_fuse(x)
        keys = self.atn_key(x)
        values = self.atn_value(x)
        return keys, values
    
    def query(self, P2):
        x = self.qry_fuse(P2)
        keys = self.qry_atn_key(x)
        values = self.qry_atn_value(x)
        return keys, values

    def segment(self, m_in, m_out, q_in, q_out, C2, C3):
        # mem [B,C,T,H,W], qry [1,C,H,W]
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()
        q_in = q_in.expand(B,-1,-1,-1)
        q_out = q_out.expand(B,-1,-1,-1)

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        # seg
        m4 = self.ResMM(self.convFM(mem_out))
        m3 = self.RF3(C3, m4) # out: 1/8, 256
        m2 = self.RF2(C2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        p = F.softmax(p, dim=1)
        p = p[:,1:]
        return p

    def soft_agg(self, ps):
        K, _, H, W = ps.shape
        ps = ps[:,0] # [K,H,W]
        em = torch.zeros(1, K+1, H, W).cuda().float()
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:K+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em))) # [1,K+1,H,W]
        return logit

    def forward(self, ref_x, x, x_ori, ref_masks, img_meta):
        ## ref_x/x -- tensor(P2), x_ori -- list(C2,C3,C4)
        keys, values = [], []
        for bs in range(len(ref_masks)):
            cur_key, cur_value = self.memorize(ref_x[bs:bs+1], ref_masks[bs])
            keys.append(cur_key)
            values.append(cur_value)
        preds = []
        for bs in range(len(ref_masks)):
            cur_key, cur_value = self.query(x[bs:bs+1])
            cur_pred = self.segment(keys[bs][:,:,None], values[bs][:,:,None], cur_key, cur_value, 
                                    x_ori[0][bs:bs+1], x_ori[1][bs:bs+1])
            preds.append(cur_pred) # [K,1,H,W]
        for bs in range(len(preds)):
            preds[bs] = self.soft_agg(preds[bs]) # [1,1+K,H,W]
        return preds

    def get_target(self, gt_masks, gt_pids):
        prop_targets = [] 
        for bs in range(len(gt_masks)):
            assert gt_masks[bs].shape[0] == gt_pids[bs].shape[0], (gt_masks[bs].shape[0], gt_pids[bs])
            cur_gt_masks = torch.LongTensor(gt_masks[bs]).cuda()
            # cur_masks = torch.zeros(cur_gt_masks.shape[-2:]).cuda().long()
            cur_masks = dict()
            for k in range(cur_gt_masks.shape[0]):
                if gt_pids[bs][k] != 0: 
                    # if bs == 1:
                        # print('bs', bs, k, gt_pids[bs][k])
                    # print('shape', (cur_masks.shape), (cur_gt_masks[k]==1).shape)
                    # cur_masks[cur_gt_masks[k]==1] = gt_pids[bs][k]
                    # print('gt_pids', gt_pids[bs])
                    # print('k', k, gt_pids[bs][k])
                    # print('k.s', k, gt_pids[bs][k].shape)
                    # print('k-', k, gt_pids[bs][k].item())
                    # assert 1<0, (gt_pids[bs][k])
                    assert (cur_gt_masks[k]==1).sum() > 0, (cur_gt_masks[k]==1).sum()
                    cur_masks[gt_pids[bs][k].item()] = (cur_gt_masks[k] == 1)
                    # if bs == 1:
                    #     cv2.imwrite('../sample/%d_%d_gt.png' % (bs,k), (cur_gt_masks[k]==1).cpu().numpy()*255)
                    #     cv2.imwrite('../sample/%d_%d_prop.png' % (bs,k), (cur_masks==gt_pids[bs][k]).cpu().numpy()*255)
                    #     print('area', (cur_masks==gt_pids[bs][k]).sum())
                    #     print('area2', (cur_masks==k+1).sum())
            prop_targets.append(cur_masks)

        # prop_targets = torch.stack(prop_targets, 0)
        return prop_targets

    def calc_iouloss(self, pred, label, eps=1e-6):
        assert pred.shape == label.shape, (pred.shape, label.shape)
        assert len(pred.shape) == 2, (pred.shape) # [H,W]
        assert pred.min() >= 0 and pred.max() <= 1, (pred.min(), pred.max())
        assert label.min() >= 0 and label.max() <= 1, (label.min(), label.max())
        # print('pred', pred.min(), pred.max())
        # print('label', label.min(), label.max())
        cur_loss = 1 - torch.min(pred, label).sum() / (torch.max(pred,label).sum() + eps)
        return cur_loss

    def loss(self, preds, targets):
        assert len(preds) == len(targets), (len(preds), len(targets))
        loss = dict()
        loss_prop = []
        for bs in range(len(preds)):
            for k in range(1, preds[bs].shape[1]):
                if k in targets[bs] and targets[bs][k].sum() > 0:
                    cur_loss = self.calc_iouloss(preds[bs].softmax(1)[0,k], targets[bs][k].float())
                    loss_prop.append(cur_loss)
                else:
                    assert 1<0, (k, bs, preds[bs].shape, targets[bs].keys(), (targets[bs][k].sum()), (targets[bs][k]==1).sum())
        loss['loss_prop'] = sum(loss_prop) / len(loss_prop)
        return loss
