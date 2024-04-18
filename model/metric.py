import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F
import math
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology
from skimage.metrics import structural_similarity as compare_ssim


# segmeantaion metric
# save
def dice_coeff(input: Tensor, target: Tensor):
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    return dice

# save
def iou_coeff(input: Tensor, target: Tensor):
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    union = (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) - intersection.sum(1) + smooth)
    union = union.sum() / num
    return union

# save
def multiclass_dice_coeff(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert y_pred.size() == y_true.size()
    dice = 0
    # remove backgroud region
    for channel in range(1, y_true.shape[1]):
        dice += dice_coeff(y_pred[:, channel, ...], y_true[:, channel, ...])
    return dice / (input.shape[1] - 1)

# save
def multiclass_iou_coeff(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert input.size() == target.size()
    union = 0
    # remove backgroud region
    for channel in range(1, input.shape[1]):
        union += iou_coeff(y_pred[:, channel, ...], y_true[:, channel, ...])
    return union / (input.shape[1] - 1)


