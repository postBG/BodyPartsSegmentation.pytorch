import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets.pascal_parts import IGNORE_LABEL, CLASS_WEIGHT


def get_weights_as_tensor(class_weight, device):
    return torch.Tensor(CLASS_WEIGHT[class_weight]).to(device)


def dice_loss(logits, targets, eps=1e-7, ignore_label=IGNORE_LABEL):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (C should be bigger than 1)
        targets: a tensor of shape [B, 1, H, W].
        eps: added to the denominator for numerical stability.
        ignore_label: Default = 255
    Returns:
        dice_loss: the Sørensen–Dice loss.


    FROM https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]

    one_hot_targets = torch.eye(ignore_label + 1)[targets.squeeze(1)]
    one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).float()
    predictions = F.softmax(logits, dim=1)

    one_hot_targets = one_hot_targets.type(logits.type())
    dims = (0,) + tuple(range(2, targets.ndimension()))
    intersection = torch.sum(predictions * one_hot_targets[:, :num_classes, :, :], dims)
    cardinality = torch.sum(predictions + one_hot_targets[:, :num_classes, :, :], dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


def create_criterion(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,
                                   weight=get_weights_as_tensor(args.class_weight, args.device))

    elif args.criterion == 'dice':
        return dice_loss
    else:
        raise ValueError('Loss {} is not supported.')
