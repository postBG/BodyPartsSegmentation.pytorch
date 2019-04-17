import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm=None):
        super(Decoder, self).__init__()
        inplanes = 128
        mid_filters = 48
        filters = 256
        self.low_conv = nn.Conv2d(inplanes, mid_filters, 1, 1, 0, 1, 1, bias=False)
        self.low_bn = BatchNorm(mid_filters)
        self.relu = nn.ReLU()

        self.refine_conv = nn.Sequential(
            nn.Conv2d(mid_filters + filters, filters, kernel_size=3, bias=False),
            BatchNorm(filters),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, kernel_size=3, bias=False),
            BatchNorm(filters),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(filters, num_classes, kernel_size=1, stride=1)
        )

        self._init_weight()

    def forward(self, x, low_feat):
        low_input = self.low_conv(low_feat)
        low_input = self.low_bn(low_input)
        low_input = self.relu(low_input)  # output_stride: 8
        x = F.interpolate(x, size=low_input.size()[2:], mode='bilinear', align_corners=True)

        output = torch.cat((low_input, x), dim=1)
        output = self.refine_conv(output)

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

