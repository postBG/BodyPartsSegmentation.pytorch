import os
from os.path import expanduser

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

DEFAULT_ROOT = "%s/VOCdevkit/" % expanduser("~")

from datasets.utils import JointToTensor

# TODO: Calculate This
CLASS_WEIGHT = []

# TODO: Find out this
IGNORE_LABEL = None

STATISTICS_SET = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


class PascalPartsDataSet(Dataset):
    def __init__(self, root=DEFAULT_ROOT,
                 joint_transform=JointToTensor(),
                 img_transform=None,
                 mask_transform=None,
                 is_train=False,
                 sliding_crop=None, ):
        self.images_dir = os.path.join(root, "VOC2010/JPEGImages/")
        self.labels_dir = os.path.join(root, "Annotations_Part")

        self.image_list = []
        if is_train:
            with open("%s/train.txt" % DEFAULT_ROOT, "r") as f:
                for image in f:
                    self.image_list.append(image.replace("\n", ""))
        else:
            with open("%s/val.txt" % DEFAULT_ROOT, "r") as f:
                for image in f:
                    self.image_list.append(image.replace("\n", ""))

        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, mask_path = os.path.join(self.images_dir, self.image_list[idx] + ".jpg"), \
                              os.path.join(self.labels_dir, self.image_list[idx] + ".png")
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        if self.joint_transform:
            img, mask = self.joint_transform(img, mask)
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask


if __name__ == "__main__":

    for img, label in DataLoader(PascalPartsDataSet()):
        print(np.shape(img), np.shape(label), np.unique(label))
        break
