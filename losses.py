import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets.pascal_parts import IGNORE_LABEL, CLASS_WEIGHT

ONE_HOT_EMBEDDING = torch.eye(IGNORE_LABEL + 1)


def get_weights_as_tensor(class_weight, device):
    return torch.Tensor(CLASS_WEIGHT[class_weight]).to(device)


def dice_loss(logits, targets, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss. 255 will be ignored
    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (C should be bigger than 1)
        targets: a tensor of shape [B, 1, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.


    FROM https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]

    one_hot_targets = ONE_HOT_EMBEDDING[targets.squeeze(1)]
    one_hot_targets = one_hot_targets.permute(0, 3, 1, 2)[:, :num_classes, :, :]
    one_hot_targets = one_hot_targets.type(logits.type())
    predictions = F.softmax(logits, dim=1)

    dims = (0,) + tuple(range(2, targets.ndimension()))
    intersection = torch.sum(predictions * one_hot_targets, dims)
    cardinality = torch.sum(predictions + one_hot_targets, dims)
    loss = (2. * intersection / (cardinality + eps)).mean()
    return -loss


def create_criterion(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,
                                   weight=get_weights_as_tensor(args.class_weight, args.device))

    elif args.criterion == 'dice':
        return dice_loss
    else:
        raise ValueError('Loss {} is not supported.')
