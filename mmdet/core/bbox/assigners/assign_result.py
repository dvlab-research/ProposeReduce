import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None, pids=None, sep_flag=None,
                 overlaps=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.pids = pids
        self.sep_flag = sep_flag
        self.overlaps = overlaps

    def add_gt_(self, gt_labels, gt_pids=None, gt_sep_flag=None):
        if gt_labels is not None:
            self_inds = torch.arange(
                1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        else:
            self_inds = torch.arange(
                1, len(gt_pids) + 1, dtype=torch.long, device=gt_pids.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        self.overlaps = torch.cat(
            [torch.eye(self.num_gts).cuda().float(), self.overlaps], 1
        )
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
        if self.pids is not None:
            self.pids = torch.cat([gt_pids, self.pids])
        if self.sep_flag is not None:
            self.sep_flag = torch.cat([gt_sep_flag, self.sep_flag], 0)
