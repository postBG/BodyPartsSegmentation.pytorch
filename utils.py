import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

palette = [0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,255,0, 0,255,255, 255,0,255, 192,192,192, 128,128,128,
           128,0,0, 128,128,0, 0,128,0, 128,0,128, 0,128,128, 0,0,128, 255,69,0, 255,215,0, 0,100,0,
           175,238,238, 30,144,255, 255,20,147, 210,105,30, 245,255,250, 188,143,143, 0,0,0]


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
