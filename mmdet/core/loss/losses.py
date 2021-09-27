# TODO merge naive and weighted loss.
import torch
import torch.nn.functional as F


def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None,
                           reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

def weighted_focal_cross_entropy(pred, label, weight, avg_factor=None,
                           reduce=True, gamma=2.0, alpha=0.25):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    pred_simoid = pred.sigmoid()
    self_shape = torch.arange(pred.shape[0]).cuda().long()
    pt = pred_simoid[self_shape, label]
    target = (label>0).float()
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None, logit=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    if logit:
        return F.binary_cross_entropy_with_logits(
            pred, label.float(), weight.float(),
            reduction='sum')[None] / avg_factor
    else:
        return F.binary_cross_entropy(
            pred, label.float(), weight.float(),
            reduction='sum')[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='elementwise_mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.cuda().float().detach()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    weight = weight.detach()
    return F.binary_cross_entropy_with_logits(
        pred, target, weight, reduction=reduction)


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='elementwise_mean')[None]

def mask_iou_loss(pred, label, eps=1e-6):
    pred = pred.view(pred.shape[0], -1)
    label = label.view(label.shape[0], -1)
    intsec = torch.min(pred, label).sum(1)
    union = torch.max(pred, label).sum(1)
    ones = torch.ones(pred.shape[0]).cuda().float()
    iou = ones - intsec.float() / (union.float()+eps)
    return iou.mean()

def seq_mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, :, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='elementwise_mean')[None]

def weighted_mask_cross_entropy(pred, target, label, weights):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, :, label].squeeze(1)
    loss =F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='none')
    '''
    loss/pred_slice/target -- [B',T,M,M]
    weight -- [B',T,1,1] -- float
    '''
    weights = weights.expand(-1,-1,loss.shape[-2],loss.shape[-1])
    loss = (loss * weights).sum() / (weights.sum()+1e-6)
    return loss[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='elementwise_mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 0:
        return loss
    elif reduction == 1:
        return loss.sum() / pred.numel()
    elif reduction == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

def accuracy_thr(pred, target, thr=0):
    pred_label = (pred > thr)
    target_label = (target > thr)
    correct = pred_label.eq(target_label)
    assert len(correct.shape) == 1
    res = correct.float().sum(0, keepdim=True).mul_(100. / correct.size(0))
    assert (len(res.shape) == 1 and res.shape[0] == 1)
    return res


def iou_loss(pred, target, ignore_index=-1, eps=1e-6):
    '''

    :param pred: [N,N] flaot tensor
    :param target: [N,N] float/long tensor
    :return:
    '''
    weight = (target!=ignore_index).float()
    intsec = torch.min(pred, target.float())
    union = torch.max(pred, target.float())
    loss = 1 - (intsec*weight).sum() / ((union*weight).sum()+eps)
    return loss