from models.backbone.aligned_xception import AlignedXception
from models.backbone.Xception import xception
from models.backbone import resnet


def build_backbone(name, output_stride, batch_norm=None, pretrained=True):
    backbone = None
    if name == 'xception':
        backbone = xception(pretrained=pretrained)  # AlignedXception(output_stride, BatchNorm, pretrained=pretrained)

    elif name == 'resnet':
        backbone = resnet.ResNet101(output_stride, batch_norm)

    return backbone
