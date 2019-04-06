import unittest
import random

import numpy as np

from datasets import PascalPartsDataSet
from datasets.utils import JointResize


class DataSetTest(unittest.TestCase):
    def test_train_shape_of_dataset(self):
        train_dataset = PascalPartsDataSet(is_train=True)
        val_dataset = PascalPartsDataSet(is_train=False)
        self.assertEqual(2867, len(train_dataset))
        self.assertEqual(717, len(val_dataset))

    def test_train_data_shape(self):
        train_dataset = PascalPartsDataSet(is_train=True, joint_transform=JointResize(513, 513))
        idx = random.randint(a=0, b=len(train_dataset))
        img, label = train_dataset[idx]
        self.assertTupleEqual((513, 513, 3), np.shape(img))
        self.assertTupleEqual((513, 513), np.shape(label))
