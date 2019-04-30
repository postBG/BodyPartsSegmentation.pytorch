from models.backbone.Xception import AlignedXception
from models.backbone.Xception2 import get_xception
from models.backbone import resnet

def build_backbone(name, output_stride, BatchNorm=None, pretrained=True):
    backbone = None
    if name == 'xception':
        backbone = get_xception(pretrained=True) #AlignedXception(output_stride, BatchNorm, pretrained=pretrained)

    elif name == 'resnet':
        backbone = resnet.ResNet101(output_stride, BatchNorm)

    return backbone
