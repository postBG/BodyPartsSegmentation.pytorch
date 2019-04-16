import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets.pascal_parts import IGNORE_LABEL, CLASS_WEIGHT


def get_weights_as_tensor(class_weight, device):
    return torch.Tensor(CLASS_WEIGHT[class_weight]).to(device)


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.


    FROM https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(256)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot[:, :25, :, :], dims)
    cardinality = torch.sum(probas + true_1_hot[:, :25, :, :], dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def create_criterion(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,
                                   weight=get_weights_as_tensor(args.class_weight, args.device))

    elif args.criterion == 'dice':
        return dice_loss
    else:
        raise ValueError('Loss {} is not supported.')
