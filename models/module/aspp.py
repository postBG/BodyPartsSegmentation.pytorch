import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # pad last dim(w) and 2nd to last dim(h) by (pad_beg, pad_end)
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class AtrousLayer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(AtrousLayer, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = fixed_padding(x, self.conv.kernel_size[0], self.conv.dilation[0])
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # xavier initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SpecialPooling(nn.Module):
    def __init__(self, inplanes, planes, bias=False, BatchNorm=None):
        super(SpecialPooling, self).__init__()

        self.gab = nn.AvgPool2d(kernel_size=1)
        self.conv = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, bias=bias)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, input):
        x = self.gab(input)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # xavier initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm=None):
        super(ASPP, self).__init__()

        filters = 256
        if output_stride == 8:
            dilation = [6, 12, 18]
        else:
            dilation = [12, 24, 36]

        self.conv1 = nn.Conv2d(inplanes, filters, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = AtrousLayer(inplanes, filters, kernel_size=3, stride=1, dilation=dilation[0], bias=False, BatchNorm=BatchNorm)
        self.conv3 = AtrousLayer(inplanes, filters, kernel_size=3, stride=1, dilation=dilation[1], bias=False, BatchNorm=BatchNorm)
        self.conv4 = AtrousLayer(inplanes, filters, kernel_size=3, stride=1, dilation=dilation[2], bias=False, BatchNorm=BatchNorm)

        self.image_pooling = SpecialPooling(inplanes, filters, BatchNorm=BatchNorm)

        self.last_conv = nn.Conv2d(5 * filters, filters, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn = BatchNorm(filters)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        low_feature = self.image_pooling(x)

        output = torch.cat((x1, x2, x3, x4, low_feature), dim=1)
        output = self.last_conv(output)
        output = self.bn(output)
        output = self.relu(output)

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # xavier initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



