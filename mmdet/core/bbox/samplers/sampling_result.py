import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags, bboxes_idx=None):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        '''
        len(bboxes) == 2 -> rpn stage
                       3 -> seq_obx stage
        '''
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.gt_flags = gt_flags

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        '''Set neg_assigned_gt_inds as -1'''
        self.neg_assigned_gt_inds = assign_result.gt_inds.new_zeros(neg_inds.shape[0])-1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        self.gt_bboxes = gt_bboxes

        # self.pos_max_overlaps = assign_result.max_overlaps[:,pos_inds]
        # self.neg_max_overlaps = assign_result.max_overlaps[:,neg_inds]
        self.max_overlaps = assign_result.max_overlaps

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        if assign_result.pids is not None:
            self.pos_gt_pids = assign_result.pids[pos_inds]
            self.neg_gt_pids = torch.zeros(neg_inds.shape[0]).cuda().long() - 1
        else:
            self.pos_gt_pids = None

        if bboxes_idx is not None:
            self.pos_bboxes_idx = bboxes_idx[pos_inds]
            self.neg_bboxes_idx = bboxes_idx[neg_inds]

        if assign_result.sep_flag is not None:
            self.pos_bboxes_sep_flag = assign_result.sep_flag[pos_inds]
            self.neg_bboxes_sep_flag = assign_result.sep_flag[neg_inds]

        if assign_result.overlaps is not None:
            self.pos_gt_overlaps = assign_result.overlaps[:, self.pos_inds]
            # self.neg_gt_overlaps = assign_result.overlaps[:, self.neg_inds] * 0.
            self.neg_gt_overlaps = assign_result.overlaps[:, self.neg_inds]

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    @property
    def inds(self):
        return torch.cat([self.pos_inds, self.neg_inds])

    @property
    def assigned_gt_inds(self):
        return torch.cat([self.pos_assigned_gt_inds, self.neg_assigned_gt_inds])

    @property
    def gt_pids(self):
        return torch.cat([self.pos_gt_pids, self.neg_gt_pids])

    @property
    def bboxes_idx(self):
        return torch.cat([self.pos_bboxes_idx, self.neg_bboxes_idx])

    @property
    def bboxes_sep_flag(self):
        return torch.cat([self.pos_bboxes_sep_flag, self.neg_bboxes_sep_flag])

    @property
    def bboxes_gt_overlaps(self):
        return torch.cat([self.pos_gt_overlaps, self.neg_gt_overlaps], 1)

    @property
    def bboxes_max_overlaps(self):
        return torch.cat([self.pos_max_overlaps, self.neg_neg_overlaps], 1)

