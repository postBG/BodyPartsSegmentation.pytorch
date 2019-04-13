import torch
import torch.nn as nn

from datasets.pascal_parts import IGNORE_LABEL, CLASS_WEIGHT


def get_weights_as_tensor(class_weight, device):
    return torch.Tensor(CLASS_WEIGHT[class_weight]).to(device)


def create_criterion(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,
                                   weight=get_weights_as_tensor(args.class_weight, args.device))
    else:
        raise ValueError('Loss {} is not supported.')
