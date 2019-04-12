import os
from os.path import expanduser

from PIL import Image
from torch.utils.data import Dataset

# TODO: Calculate This
CLASS_WEIGHT = []

IGNORE_LABEL = 255

STATISTICS_SET = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DEFAULT_ROOT = "%s/VOCdevkit/" % expanduser("~")


class PascalPartsDataSet(Dataset):
    def __init__(self, root=DEFAULT_ROOT,
                 joint_transform=None,
                 img_transform=None,
                 mask_transform=None,
                 is_train=False):
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
