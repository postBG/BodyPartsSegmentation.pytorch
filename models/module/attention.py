import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, backbone, num_scales=3):
        super(Attention, self).__init__()

        if backbone == 'drn':
            inplanes = 512 * num_scales
        elif backbone == 'mobilenet':
            inplanes = 320 * num_scales
        else:
            inplanes = 256 * num_scales

        self.block_conv = nn.Conv2d(inplanes, 512, 3, stride=1, padding=1, bias=False)
        self.decode_scale = nn.Conv2d(512, num_scales, 1, bias=False)

    def forward(self, features):
        features = self.block_conv(features)
        features = self.decode_scale(features)

        return features


def build_attention(backbone, num_scales):
    return Attention(backbone=backbone, num_scales=num_scales)
