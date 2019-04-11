import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torchvision.transforms import Normalize, ToPILImage

palette = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255, 192, 192, 192, 128, 128,
           128,
           128, 0, 0, 128, 128, 0, 0, 128, 0, 128, 0, 128, 0, 128, 128, 0, 0, 128, 255, 69, 0, 255, 215, 0, 0, 100, 0,
           175, 238, 238, 30, 144, 255, 255, 20, 147, 210, 105, 30, 245, 255, 250, 188, 143, 143, 0, 0, 0]


def colorize_mask(mask):
    """
    Colorize GT mask
    :param mask: input img as PIL or np array
    :return: PIL image of colorized mask
    """
    if not type(mask) == np.ndarray:
        mask = np.array(mask)
    # mask: numpy array of the mask
    mask_copy = mask.copy()

    new_mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def create_dataset_for_visualization(dataset, indicies=None):
    indicies = indicies if indicies else np.random.randint(0, len(dataset), 5)
    dataset_for_visualization = Subset(dataset, indicies)
    return dataset_for_visualization


def tensor_to_PIL(img, mean, std, clamp=True):
    """
    Converts tensor to PIL image for visualization
    :param: Original mean used to normalize image, list
    :param: Original std used to normalize image, list
    :return: PIL Image
    """
    new_std = 1. / np.array(std)
    new_mean = -1 * np.array(mean) / np.array(std)
    unnormalize = Normalize(mean=new_mean, std=new_std)
    img = img.clone().cpu().detach()
    unnormalized_img = unnormalize(img)

    if clamp:
        unnormalized_img = unnormalized_img.clamp(0, 1)
    return ToPILImage()(unnormalized_img)
