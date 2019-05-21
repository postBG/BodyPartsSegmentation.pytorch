import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from datasets.utils import JointToTensor, RandomHorizontalFlip, RandomCrop, JointResize, JointCompose, RandomScaleCrop, \
    RandomGaussianBlur, RandomRotate
from datasets.pascal_parts import PascalPartsDataSet, STATISTICS_SET


def get_joint_transform(transform_type):
    transformations = {
        'none': JointCompose([JointResize(513, 513), JointToTensor()]),
        'val': JointCompose([JointResize(513, 513), JointToTensor()]),
        'random': JointCompose([RandomScaleCrop(513, 513),
                                RandomRotate(),
                                # RandomGaussianBlur(),
                                # RandomHorizontalFlip(),
                                JointToTensor()])
    }

    return transformations[transform_type]


def dataset_factory(transform_type, is_train=True):
    joint_transform = get_joint_transform(transform_type)
    normalization_transform = transforms.Normalize(**STATISTICS_SET)
    dataset = PascalPartsDataSet(joint_transform=joint_transform, img_transform=normalization_transform,
                                 is_train=is_train)
    return dataset


def dataloaders_factory(train_dataset, val_dataset, batch_size, test_mode=False):
    if test_mode:
        train_dataset = Subset(train_dataset, np.random.randint(0, len(train_dataset), batch_size * 5))
        val_dataset = Subset(val_dataset, np.random.randint(0, len(val_dataset), batch_size * 5))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

    return {
        'train': train_dataloader,
        'val': val_dataloader,
    }
