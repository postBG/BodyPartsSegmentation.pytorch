import os

# TODO: Set this value
from PIL import Image
from torch.utils import data

DEFAULT_ROOT = ''

# TODO: Calculate This
CLASS_WEIGHT = []

# TODO: Find out this
IGNORE_LABEL = None

STATISTICS_SET = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


class PascalPartsDataSet(data.Dataset):
    def __init__(self, root=DEFAULT_ROOT, joint_transform=None, img_transform=None, mask_transform=None, is_train=False,
                 sliding_crop=None):
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'images')
        self.image_list = os.path.join(root, 'list')

        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        img_path, mask_path = os.path.join(self.images_dir, self.image_list[idx]), \
                              os.path.join(self.labels_dir, self.image_list[idx])
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        if self.joint_transform:
            img, mask = self.joint_transform(img, mask)
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask
