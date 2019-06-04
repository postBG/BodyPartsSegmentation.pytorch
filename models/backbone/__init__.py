from models.backbone.aligned_xception import AlignedXception
from models.backbone.Xception import xception
from models.backbone import resnet


def build_backbone(name, output_stride, batch_norm_cls=None, pretrained=True):
    backbone = None
    if name == 'xception':
        backbone = AlignedXception(output_stride, batch_norm_cls, pretrained=pretrained)

    elif name == 'resnet':
        backbone = resnet.ResNet101(output_stride, batch_norm_cls)

    return backbone
