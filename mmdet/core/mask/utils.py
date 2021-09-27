import mmcv
import torch


def split_combined_polys(polys, poly_lens, polys_per_mask):
    """Split the combined 1-D polys into masks.

    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.

    Args:
        polys (list): a list (length = image num) of 1-D tensors
        poly_lens (list): a list (length = image num) of poly length
        polys_per_mask (list): a list (length = image num) of poly number
            of each mask

    Returns:
        list: a list (length = image num) of list (length = mask num) of
            list (length = poly num) of numpy array
    """
    mask_polys_list = []
    for img_id in range(len(polys)):
        polys_single = polys[img_id]
        polys_lens_single = poly_lens[img_id].tolist()
        polys_per_mask_single = polys_per_mask[img_id].tolist()

        split_polys = mmcv.slice_list(polys_single, polys_lens_single)
        mask_polys = mmcv.slice_list(split_polys, polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list

def mask_overlaps(a, b, eps=1e-6):
    """

    :param a: tensor, [N,H,W]
    :param b: tensor, [M,H,W]
    :return:  iou [N,M]
    """
    assert len(a.shape) == 3 and len(b.shape) == 3, (a.shape, b.shape)
    a = a.contiguous().view(a.shape[0], -1) # [N,HW]
    b = b.contiguous().view(b.shape[0], -1) # [N,HW]
    a = (a>0.5).float()
    b = (b>0.5).float()
    a = a[:,None]
    b = b[None]
    intsec = torch.min(a,b) # [N,M,HW]
    union = torch.max(a,b) # [N,M,HW]
    iou = intsec.sum(2) / (union.sum(2) + eps)
    return iou


