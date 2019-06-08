import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bases import AbstractModel
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.module.aspp import build_aspp
from models.module.decoder import build_decoder
from models.module.attention import build_attention
from models.backbone import build_backbone


class DeepLab(AbstractModel):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        self.attention_scales = [1.0, 0.75, 0.5]
        batch_norm_cls = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batch_norm_cls)
        self.attention = build_attention(backbone, num_scales=len(self.attention_scales))
        self.aspp = build_aspp(backbone, output_stride, batch_norm_cls)
        self.decoder = build_decoder(num_classes, backbone, batch_norm_cls)

        self.softmax2d = nn.Softmax2d()

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        h, w = input.size()[2:]
        input2 = F.interpolate(input, size=[int(self.attention_scales[1] * h), int(self.attention_scales[1] * w)],
                               mode='bilinear', align_corners=True)
        y, low_level_feat2 = self.backbone(input2)
        y = self.aspp(y)
        y = self.decoder(y, low_level_feat2)

        input3 = F.interpolate(input, size=[int(self.attention_scales[2] * h), int(self.attention_scales[2] * w)],
                               mode='bilinear', align_corners=True)
        z, low_level_feat3 = self.backbone(input3)
        z = self.aspp(z)
        z = self.decoder(z, low_level_feat3)

        #resize_feat1 = F.interpolate(low_level_feat, size=input.size()[2:], mode='bilinear', align_corners=True)
        resize_feat2 = F.interpolate(low_level_feat2, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        resize_feat3 = F.interpolate(low_level_feat3, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        merged_feat = torch.cat([low_level_feat, resize_feat2, resize_feat3], dim=1)

        attention_map = self.attention(merged_feat)
        scale_weight_map = self.softmax2d(attention_map)
        scale_weight_map = F.interpolate(scale_weight_map, size=input.size()[2:], mode='bilinear', align_corners=True)

        output1 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        output2 = F.interpolate(y, size=input.size()[2:], mode='bilinear', align_corners=True)
        output3 = F.interpolate(z, size=input.size()[2:], mode='bilinear', align_corners=True)

        output = scale_weight_map[:, 0:1, ...] * output1 + scale_weight_map[:, 1:2, ...] * output2 + scale_weight_map[:, 2:3, ...] * output3

        return output, x, y, z

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
