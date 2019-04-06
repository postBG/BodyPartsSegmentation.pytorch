import random
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize


class Compose(object):
    """
    Compose multiple transforms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontalFlip(object):
    """
    Random horizontal flip (50%)
    """

    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomCrop(object):
    def __init__(self, height=512):
        self.crop_height = height
        self.crop_width = height * 2

    def __call__(self, img, mask):
        img_h, img_w = img.size[1], img.size[0]
        rand_x = np.random.randint(0, img_w - self.crop_width - 1)
        rand_y = np.random.randint(0, img_h - self.crop_height - 1)

        return img.crop((rand_x, rand_y, rand_x + self.crop_width, rand_y + self.crop_height)), mask.crop(
            (rand_x, rand_y, rand_x + self.crop_width, rand_y + self.crop_height))


class CenterCrop(object):
    """
    Tuple of (width, height)
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        tw, th = self.width, self.height
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class JointResize(object):
    """pil to pil"""

    def __init__(self, height, width):
        size = (height, width)
        self.img_resize_transform = Resize(size, Image.BILINEAR)
        self.mask_resize_transform = Resize(size, Image.NEAREST)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return self.img_resize_transform(img), self.mask_resize_transform(mask)


class JointToTensor(object):
    def __init__(self):
        self.totensor = ToTensor()

    def __call__(self, img, mask):
        return self.totensor(img), torch.from_numpy(np.array(mask).astype(np.int))


class CombinedDataSet(Dataset):
    """
    source_dataset and augmented_source_dataset must be aligned
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        target_index = (index + random.randint(0, len(self.target_dataset) - 1)) % len(self.target_dataset)

        return self.source_dataset[source_index], self.target_dataset[target_index]

    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))
