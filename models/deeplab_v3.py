import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bases import AbstractModel
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.module.aspp import build_aspp
from models.module.decoder import build_decoder
from models.backbone import build_backbone


class DeepLab(AbstractModel):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        batch_norm_cls = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batch_norm_cls)
        self.aspp = build_aspp(backbone, output_stride, batch_norm_cls)
        self.decoder = build_decoder(num_classes, backbone, batch_norm_cls)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, is_test=False):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        output = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if False and is_test:
            h, w = input.size()[2:]
            y = F.interpolate(input, size=[h // 2, w // 2], mode='bilinear', align_corners=True)
            y, low_level_feat = self.backbone(y)
            y = self.aspp(y)
            y = self.decoder(y, low_level_feat)
            output2 = F.interpolate(y, size=input.size()[2:], mode='bilinear', align_corners=True)

            z = F.interpolate(input, size=[2 * h, 2 * w], mode='bilinear', align_corners=True)
            z, low_level_feat = self.backbone(z)
            z = self.aspp(z)
            z = self.decoder(z, low_level_feat)
            output3 = F.interpolate(z, size=input.size()[2:], mode='bilinear', align_corners=True)

            output = (output + output2 + output3) / 3

        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
