import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz import _lovasz_hinge, _lovasz_softmax
from typing import Optional, Union

# mutil loss
class MutilCrossEntropyLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
        y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
        mask = y_true_onehot.sum((0, 2)) > 0
        loss = F.cross_entropy(y_pred_logits.float(), y_true.long(), weight=mask.to(y_pred_logits.dtype))
        return loss

class MutilFocalLoss(nn.Module):
    """
    """

    def __init__(self, alpha, gamma=2, torch=True):
        super(MutilFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.torch = torch

    def forward(self, y_pred_logits, y_true):
        if torch:
            Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
            y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
            y_true = y_true.long().contiguous().view(Batchsize, -1)
            y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
            y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
            mask = y_true_onehot.sum((0, 2)) > 0
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=mask.to(y_pred_logits.dtype))
            logpt = CE_loss(y_pred_logits.float(), y_true.long())
            pt = torch.exp(-logpt)
            loss = (((1 - pt) ** self.gamma) * logpt).mean()
            return loss

class MutilDiceLoss(nn.Module):
    """
        multi label dice loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        # print('y_true.max()',y_true.max())
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        smooth = 1.e-5
        eps = 1e-7
        assert y_pred.size() == y_true.size()
        intersection = torch.sum(y_true * y_pred, dim=(0, 2))
        denominator = torch.sum(y_true + y_pred, dim=(0, 2))
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
        loss = - gen_dice_coef
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return (loss * self.alpha).sum() / torch.count_nonzero(mask)

        