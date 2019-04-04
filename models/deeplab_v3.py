import torch.nn as nn
import torch.nn.functional as F

# from models.backbone import get_backbone
# from models.module import get_decoder, get_aspp


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=6, freeze_bn=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = get_backbone(backbone, output_stride, BatchNorm, pretrained=True)
        self.aspp = get_aspp(inplanes=2048, output_stride=output_stride, BatchNorm=BatchNorm)
        self.decoder = get_decoder(num_classes=num_classes, BatchNorm=BatchNorm)

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


if __name__ == "__main__":
    import os.path as osp
    import sys

    this_dir = osp.dirname(__file__)
    root_path = osp.join(this_dir, '../')
    sys.path.append(root_path)

    # comment same import on top of this file
    from models.backbone import get_backbone
    from models.module import get_decoder, get_aspp

    import torch

    input = torch.randn(1, 3, 513, 513)
    model = DeepLab(backbone='xception', output_stride=16, num_classes=6, freeze_bn=False)
    with torch.no_grad():
        output = model(input)
        print(output.shape)

