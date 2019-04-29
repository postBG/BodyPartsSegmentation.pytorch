from models.backbone.Xception import AlignedXception
from models.backbone.Xception2 import get_xception


def get_backbone(name, output_stride, BatchNorm=None, pretrained=True):
    backbone = None
    if name == 'xception':
        backbone = get_xception(pretrained=True) #AlignedXception(output_stride, BatchNorm, pretrained=pretrained)

    return backbone
