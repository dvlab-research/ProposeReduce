import numpy as np
import numpy.random as npr
import os.path as osp
import random
import mmcv
from .custom import CustomDataset
from .extra_aug import ExtraAugmentation
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale
 
class YTVOSDataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 neigh_sample=False,
                 neigh_max_dis=1,
                 allow_len1_vid=False,
                 spec_idx=None,
                 dataset_flip=False,
                 iter_mul=None,
                 ):
        # prefix of images path
        self.img_prefix = img_prefix
        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)
        img_ids = []
        if spec_idx is not None:
            print('set spec_idx', spec_idx)
        for idx, vid_info in enumerate(self.vid_infos):
            if test_mode and spec_idx is not None and idx != spec_idx:
                continue
            video_name = vid_info['filenames'][0].split('/')[0]
            if spec_idx is not None and video_name != spec_idx:
                continue
            cur_img_ids = []
            for frame_id in range(len(vid_info['filenames'])):
                cur_img_ids.append((idx, frame_id))
            if not dataset_flip:
                img_ids = img_ids + cur_img_ids
            else:
                img_ids = cur_img_ids + img_ids
        self.img_ids = img_ids
        print('pre-img_ids', len(img_ids))

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]

        if iter_mul != None:
            print('len-before', len(self.img_ids))
            print('mul', iter_mul)
            self.img_ids = self.img_ids * iter_mul # * 200
            print('len-after', len(self.img_ids))

        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        # always sample the neigh frame during training
        self.neigh_sample = neigh_sample
        self.neigh_max_dis = neigh_max_dis
        self.allow_len1_vid = allow_len1_vid

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])
        cnt = 0
        ## some unknown errer might happen, we fix it by iterating multiple times
        ## need to be fixed later
        while True:
            if cnt == 0:
                try:
                    data = self.prepare_train_img(self.img_ids[idx])
                except Exception as e:
                    pass
                else:
                    break
                try:
                    data = self.prepare_train_img(self.img_ids[idx], self2ref=True)
                except Exception as e:
                    pass
                else:
                    break
            try:
                data = self.prepare_train_img(self.img_ids[random.randint(0, len(self.img_ids)-1)], self2ref=True)
            except Exception as e:
                pass
            else:
                break
            cnt += 1
            assert cnt < 10, (cnt, 'error too many times...')
        return data
    
    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]
        
        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1,new_x2y2)).astype(np.float32)
        return bbox

    ## for same class, if intersec>0, set(tgt_obj_id) should be covered by set(ref_obj_id)
    def ref_cover_tgt(self, vid, frame_id, ref_frame_id):
        ann = self.get_ann_info(vid, frame_id)
        ref_ann = self.get_ann_info(vid, ref_frame_id)
        # first rule
        ann_objid = set(ann['obj_ids'])
        ref_ann_objid = set(ref_ann['obj_ids'])
        if len(ann_objid.intersection(ref_ann_objid)) == 0:
            return False
        # second rule
        tgt_obj_id = dict()
        ref_obj_id = dict()
        # init
        for i in range(len(ann['obj_ids'])):
            if ann['labels'][i] not in tgt_obj_id:
                tgt_obj_id[ann['labels'][i]] = set()
            tgt_obj_id[ann['labels'][i]].add(ann['obj_ids'][i])
        for i in range(len(ref_ann['obj_ids'])):
            if ref_ann['labels'][i] not in ref_obj_id:
                ref_obj_id[ref_ann['labels'][i]] = set()
            ref_obj_id[ref_ann['labels'][i]].add(ref_ann['obj_ids'][i])
        # compare
        flag = True
        flag_atleastone = False
        for c in tgt_obj_id.keys():
            if (c not in ref_obj_id):
                continue
            if tgt_obj_id[c].intersection(ref_obj_id[c]) != tgt_obj_id[c]:
                flag = False
                break
            flag_atleastone = True
        if flag and flag_atleastone:
            # print(tgt_obj_id, ref_obj_id)
            return True
        else:
            return False

    def sample_ref(self, idx, self2ref=False):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        if self.allow_len1_vid and len(vid_info['filenames']) == 1:
            return idx
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
          ref_idx = (vid, i)
          if i != frame_id and ref_idx in self.img_ids and self.ref_cover_tgt(vid, frame_id, i):
              valid_samples.append(ref_idx)
        if len(valid_samples) == 0 or self2ref:
            # print(idx, 'len(valid_sample)==0, add itself as ref..')
            valid_samples.append(idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_img(self, idx, self2ref=False):
        # prepare a pair of image in a sequence
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
        basename = osp.basename(vid_info['filenames'][frame_id])
        _, ref_frame_id = self.sample_ref(idx, self2ref=self2ref)
        ref_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][ref_frame_id]))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(vid, frame_id)
        ref_ann = self.get_ann_info(vid, ref_frame_id)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        ref_bboxes = ref_ann['bboxes']
        ref_labels = ref_ann['labels']
        # obj ids attribute does not exist in current annotation
        # need to add it
        ref_ids = ref_ann['obj_ids']
        gt_ids = ann['obj_ids']
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
        ref_gt_pids = [gt_ids.index(i)+1 if i in gt_ids else 0 for i in ref_ids]
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        
        img = img.copy() 
        ref_img, ref_img_shape, _, ref_scale_factor = self.img_transform(
            ref_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        ref_img = ref_img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        ref_bboxes = self.bbox_transform(ref_bboxes, ref_img_shape, ref_scale_factor,
                                          flip)
        if self.aug_ref_bbox_param is not None:
            ref_bboxes = self.bbox_aug(ref_bboxes, ref_img_shape)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)
            ref_masks = self.mask_transform(ref_ann['masks'], pad_shape,
                                           scale_factor, flip)
        ## filter gt_objs w/o fg mask
        valid_gt_ids = []
        for k,obj_id in enumerate(gt_ids):
            if gt_masks[k].sum() > 0:
                valid_gt_ids.append(obj_id)
        assert len(valid_gt_ids) > 0, (idx, gt_ids, valid_gt_ids)
        ## select ref_ids in gt_ids for prop
        prop_ref_ids = []
        prop_ref_bboxes = []
        prop_ref_labels = []
        prop_ref_masks = []
        for k,obj_id in enumerate(ref_ids):
            # if obj_id in gt_ids:
            if obj_id in valid_gt_ids:
                prop_ref_ids.append(obj_id)
                prop_ref_bboxes.append(ref_bboxes[k])
                prop_ref_labels.append(ref_labels[k])
                prop_ref_masks.append(ref_masks[k])
        assert len(prop_ref_ids) > 0, (idx, gt_ids, ref_ids, valid_gt_ids)
        prop_ref_bboxes = np.stack(prop_ref_bboxes, 0)
        prop_ref_labels = np.stack(prop_ref_labels, 0)
        prop_ref_masks = np.stack(prop_ref_masks, 0)
        ## too much objs make out of memory in the prop head
        obj_num_thr = 5 
        if len(prop_ref_ids) > obj_num_thr:
            select_idx = npr.choice(len(prop_ref_ids), obj_num_thr, replace=False)
            prop_ref_ids = np.array(prop_ref_ids)[select_idx].tolist()
            prop_ref_bboxes = prop_ref_bboxes[select_idx]
            ## constraint boundary box
            for k in range(prop_ref_bboxes.shape[0]):
                if prop_ref_bboxes[k][2] >= img_shape[1] or prop_ref_bboxes[k][3] >= img_shape[0]:
                    # print('prop_ref_bboxes - before', k, prop_ref_bboxes[k], img_shape)
                    prop_ref_bboxes[k][2] = min(prop_ref_bboxes[k][2], img_shape[1])
                    prop_ref_bboxes[k][3] = min(prop_ref_bboxes[k][3], img_shape[0])
                    # print('prop_ref_bboxes - after', k, prop_ref_bboxes[k], img_shape)
            prop_ref_labels = prop_ref_labels[select_idx]
            prop_ref_masks = prop_ref_masks[select_idx]
        prop_gt_pids = [prop_ref_ids.index(i)+1 if i in prop_ref_ids else 0 for i in gt_ids]

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        ts = float(frame_id) / len(vid_info['filenames'])
        ref_ts = float(ref_frame_id) / len(vid_info['filenames'])
        vid_name = vid_info['filenames'][0]
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            frame_id=frame_id,
            ref_frame_id=ref_frame_id,
            ts = ts,
            ref_ts = ref_ts,
            frame_num=len(vid_info['filenames']),
            vid_name=vid_name
        )

        data = dict(
            img=DC(to_tensor(img), stack=True),
            ref_img=DC(to_tensor(ref_img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            ref_bboxes = DC(to_tensor(ref_bboxes)),
            prop_ref_bboxes = DC(to_tensor(prop_ref_bboxes))
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['ref_labels'] = DC(to_tensor(ref_labels))
            data['prop_ref_labels'] = DC(to_tensor(prop_ref_labels))
        if self.with_track:
            data['gt_pids'] = DC(to_tensor(gt_pids))
            data['ref_gt_pids'] = DC(to_tensor(ref_gt_pids))
            data['prop_gt_pids'] = DC(to_tensor(prop_gt_pids))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
            data['ref_masks'] = DC(ref_masks, cpu_only=True)
            data['prop_ref_masks'] = DC(prop_ref_masks, cpu_only=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
        proposal = None

        def prepare_single(img, frame_id, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            video_name, frame_name = vid_info['filenames'][frame_id].split('.')[0].split('/')
            ts = float(frame_id) / len(vid_info['filenames'])
            _img_meta = dict(
                ori_shape=(vid_info['height'], vid_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                is_first=True,
                is_last=(frame_id==len(vid_info['filenames'])-1),
                video_id=vid,
                frame_id =frame_id,
                ts = ts,
                scale_factor=scale_factor,
                flip=flip,
                frame_num=len(vid_info['filenames']),
                video_name = video_name,
                frame_name = frame_name,
            )
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, frame_id, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
