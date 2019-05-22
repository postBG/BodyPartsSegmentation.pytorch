import torch
from torch import nn as nn

from datasets.pascal_parts import IGNORE_LABEL, CLASS_WEIGHT
from losses.dice import DiceLoss
from losses.lovasz import LovaszSoftmaxLoss


def get_weights_as_tensor(class_weight, device):
    return torch.Tensor(CLASS_WEIGHT[class_weight]).to(device)


def create_criterion(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,
                                   weight=get_weights_as_tensor(args.class_weight, args.device))
    elif args.criterion == 'dice':
        return DiceLoss(args.device, ignore_index=IGNORE_LABEL)
    elif args.criterion == 'lovasz':
        return LovaszSoftmaxLoss(ignore_index=IGNORE_LABEL)
    else:
        raise ValueError('Loss {} is not supported.')
