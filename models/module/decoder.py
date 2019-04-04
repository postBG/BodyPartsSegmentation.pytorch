import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm=None):
        super(Decoder, self).__init__()
        filters = 256

        self.low_conv = nn.Conv2d(128, filters, 1, 1, 0, 1, 1, bias=False)
        self.low_bn = BatchNorm(filters)
        self.relu = nn.ReLU()

        self.refine_conv = nn.Conv2d(2 * filters, num_classes, kernel_size=3, bias=False)

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
            if isinstance(m, nn.Conv2d):  # xavier initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

