from collections import OrderedDict

import torch
import torch.nn as nn
from abc import ABC


class AbstractModel(nn.Module, ABC):
    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        super().load_state_dict(new_state_dict, strict=strict)

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def stash_grad(self, grad_dict):
        for k, v in self.named_parameters():
            if k in grad_dict:
                grad_dict[k] += v.grad.clone()
            else:
                grad_dict[k] = v.grad.clone()
        self.zero_grad()
        return grad_dict

    def restore_grad(self, grad_dict):
        for k, v in self.named_parameters():
            grad = grad_dict[k] if k in grad_dict else torch.zeros_like(v.grad)

            if v.grad is None:
                v.grad = grad
            else:
                v.grad += grad


class AbstractFeatureExtractor(AbstractModel, ABC):
    def get_add_layer(self):
        raise NotImplementedError
