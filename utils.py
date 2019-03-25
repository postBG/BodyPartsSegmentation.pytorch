import numpy as np
from PIL import Image


# TODO: 팔레트 값 넣기
palette = []


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
