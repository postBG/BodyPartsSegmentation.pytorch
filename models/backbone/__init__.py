from models.backbone.Xception import AlignedXception


def get_backbone(name, output_stride, BatchNorm=None, pretrained=True):
    backbone = None
    if name == 'xception':
        backbone = AlignedXception(output_stride, BatchNorm, pretrained=pretrained)

    return backbone
