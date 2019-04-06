import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from models.module import get_decoder, get_aspp


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=25, freeze_bn=False):
        super(DeepLab, self).__init__()

        self.backbone = get_backbone(backbone, output_stride, nn.BatchNorm2d, pretrained=True)
        self.aspp = get_aspp(inplanes=2048, output_stride=output_stride, BatchNorm=nn.BatchNorm2d)
        self.decoder = get_decoder(num_classes=num_classes, BatchNorm=nn.BatchNorm2d)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
