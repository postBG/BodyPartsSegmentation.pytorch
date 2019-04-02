import torch.nn as nn
import torch.nn.functional as F

import models.backbone as bb
import models.module as module


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=6, freeze_bn=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = bb.get_backbone(backbone, output_stride, BatchNorm, pretrained=True)
        self.aspp = module.get_aspp(inplanes=2048, output_stride=output_stride, BatchNorm=BatchNorm)
        self.decoder = module.get_decoder(num_classes=num_classes, BatchNorm=BatchNorm)

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
