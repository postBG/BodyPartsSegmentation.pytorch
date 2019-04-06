import unittest

import torch

from models.deeplab_v3 import DeepLab


class DeepLabTest(unittest.TestCase):
    def test_input_output_shapes(self):
        inputs = torch.randn(1, 3, 513, 513)
        model = DeepLab(backbone='xception', output_stride=16, num_classes=6, freeze_bn=False)
        with torch.no_grad():
            output = model(inputs)
        self.assertTupleEqual((1, 6, 513, 513), output.size())