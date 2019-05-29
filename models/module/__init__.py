from models.module.aspp import ASPP
from models.module.decoder import Decoder
from models.module.attention import Attention

def get_aspp(inplanes, output_stride, BatchNorm=None):
    header = ASPP(inplanes=inplanes, output_stride=output_stride, batch_norm_cls=BatchNorm)
    return header


def get_decoder(num_classes, BatchNorm=None):
    decoder = Decoder(num_classes=num_classes, batch_norm_cls=BatchNorm)
    return decoder

def get_attention(backbone, num_scales):
    attention = Attention(backbone=backbone, num_scales=num_scales)
    return attention