from models.backbone.aligned_xception import AlignedXception
from models.backbone.Xception import xception
from models.backbone import resnet
from models.backbone.drn import drn_d_54


def build_backbone(name, output_stride, batch_norm_cls=None, pretrained=True):
    if name == 'xception':
        return AlignedXception(output_stride, batch_norm_cls)

    elif name == 'resnet':
        return resnet.ResNet101(output_stride, batch_norm_cls)

    elif name == 'drn':
        return drn_d_54(batch_norm_cls)
    else:
        raise ValueError
