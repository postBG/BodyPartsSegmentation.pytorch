import random
import torch
from PIL import Image, ImageOps, ImageFilter

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize


class JointCompose(object):
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


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, mask


class RandomRotate(object):
    def __init__(self, degree=180):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class RandomScaleCrop(object):
    def __init__(self, base_size=513, crop_size=513, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, mask):
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class FixedResize(object):
    def __init__(self, size):
        self.size = size  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size=513):
        self.crop_size = crop_size

    def __call__(self, img, mask):

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


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
