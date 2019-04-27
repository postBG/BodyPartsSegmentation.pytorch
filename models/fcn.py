import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from models import extended_resnet
from models.bases import AbstractModel, AbstractFeatureExtractor


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)  # default align_corners=False
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class Fusion2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fusion2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.dropout(self.relu(out))

        return out


class ResFCN(nn.Module):
    """
    img_size: torch.Size([512, 1024])
    conv_x: torch.Size([1, 64, 256, 512])
    pool_x: torch.Size([1, 64, 128, 256])
    fm2: torch.Size([1, 512, 64, 128])
    fm3: torch.Size([1, 1024, 32, 64])
    fm4: torch.Size([1, 2048, 16, 32])
    """

    def __init__(self, num_classes, layer='50', input_ch=3):
        super(ResFCN, self).__init__()

        self.num_classes = num_classes
        print('resnet' + layer)

        if layer == '18':
            resnet = extended_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '34':
            resnet = extended_resnet.resnet34(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = extended_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = extended_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = extended_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            raise ValueError("{} is not supported".format(layer))

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def forward(self, x):
        input_size = x.size()
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, input_size[2:])

        out = self.out5(fsfm5)

        return out

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )


class ResBase(AbstractFeatureExtractor):
    def get_add_layer(self):
        return 'fm4'

    def __init__(self, base_model='resnet50', input_ch=3, use_dropout_at_layer4=True):
        super(ResBase, self).__init__()

        print(base_model)
        if base_model == 'resnet18':
            resnet = extended_resnet.resnet18(pretrained=True, input_ch=input_ch,
                                              use_dropout_at_layer4=use_dropout_at_layer4)
        elif base_model == 'resnet50':
            resnet = extended_resnet.resnet50(pretrained=True, input_ch=input_ch,
                                              use_dropout_at_layer4=use_dropout_at_layer4)
        elif base_model == 'resnet101':
            resnet = extended_resnet.resnet101(pretrained=True, input_ch=input_ch,
                                               use_dropout_at_layer4=use_dropout_at_layer4)
        elif base_model == 'resnet152':
            resnet = extended_resnet.resnet152(pretrained=True, input_ch=input_ch,
                                               use_dropout_at_layer4=use_dropout_at_layer4)
        else:
            raise ValueError("{} is not supported".format(base_model))

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4_1 = self.layer4(fm3)
        fm4_2 = self.layer4(fm3)

        out_dic1 = {
            "conv_x": conv_x,
            "pool_x": pool_x,
            "fm2": fm2,
            "fm3": fm3,
            "fm4": fm4_1
        }

        out_dic2 = {
            "conv_x": conv_x,
            "pool_x": pool_x,
            "fm2": fm2,
            "fm3": fm3,
            "fm4": fm4_2
        }

        return out_dic1, out_dic2


class ResClassifier(AbstractModel):
    def __init__(self, num_classes):
        super(ResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)
        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic, img_size=(512, 1024)):
        gen_out_dic = edict(gen_out_dic)
        fsfm1 = self.fs1(gen_out_dic.fm3, self.upsample1(gen_out_dic.fm4, gen_out_dic.fm3.size()[2:]))
        fsfm2 = self.fs2(gen_out_dic.fm2, self.upsample2(fsfm1, gen_out_dic.fm2.size()[2:]))
        fsfm3 = self.fs4(gen_out_dic.pool_x, self.upsample3(fsfm2, gen_out_dic.pool_x.size()[2:]))
        fsfm4 = self.fs5(gen_out_dic.conv_x, self.upsample4(fsfm3, gen_out_dic.conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, img_size)
        out = self.out5(fsfm5)
        return out
