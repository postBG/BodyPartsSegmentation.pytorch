from models.deeplab_v3 import DeepLab
from models.fcn import ResFCN


def model_factory(args):
    if args.seg_model == 'deeplab_v3':
        return DeepLab(backbone=args.backbone, output_stride=16, num_classes=25, freeze_bn=False)
    elif args.seg_model == 'fcn':
        return ResFCN(num_classes=25)
    else:
        raise ValueError("{} is not supported.".format(args.seg_model))
