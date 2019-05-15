from models.module.aspp import ASPP
from models.module.decoder import Decoder


def get_aspp(inplanes, output_stride, BatchNorm=None):
    header = ASPP(inplanes=inplanes, output_stride=output_stride, batch_norm_cls=BatchNorm)
    return header


def get_decoder(num_classes, BatchNorm=None):
    decoder = Decoder(num_classes=num_classes, batch_norm_cls=BatchNorm)
    return decoder

