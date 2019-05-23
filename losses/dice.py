import torch
from torch.nn import functional as F


class DiceLoss(object):
    def __init__(self, device, eps=1e-7, ignore_index=None, use_weights=False):
        self.device = device
        self.eps = eps
        self.ONE_HOT_EMBEDDING = torch.eye(ignore_index + 1)
        self.use_weights = use_weights

    def __call__(self, logits, targets):
        """
            Args:
                logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model. (C should be bigger than 1)
                targets: a tensor of shape [B, H, W].
            Returns:
                dice_loss: the Sørensen–Dice loss.
            """
        num_classes = logits.shape[1]
        weights = self._calculate_weights(targets, num_classes, self.use_weights)

        one_hot_targets = self.ONE_HOT_EMBEDDING[targets.squeeze(1)]
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2)[:, :num_classes, :, :]
        one_hot_targets = one_hot_targets.type(logits.type())
        predictions = F.softmax(logits, dim=1)

        dims = (0,) + tuple(range(2, logits.ndimension()))
        intersection = torch.sum(weights * torch.sum(predictions * one_hot_targets, dims))
        cardinality = torch.sum(weights * torch.sum(predictions + one_hot_targets, dims))

        loss = 1 - 2 * (intersection + self.eps) / (cardinality + self.eps)
        return loss

    def _calculate_weights(self, targets, num_classes, use_weights):
        if not use_weights:
            return torch.Tensor([1. for _ in range(num_classes)]).to(self.device)

        num_pixels = torch.Tensor([torch.sum(targets == label) for label in range(num_classes)]).to(self.device)
        return 1 / (num_pixels * num_pixels + self.eps)
