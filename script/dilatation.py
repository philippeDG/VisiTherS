"""
Usefule to dilate and erode the mask for ablation studies.

author:  Philippe Duplessis-Guindon
date: 2022-07-26
"""

import PIL
from PIL import Image
import numpy as np
import scipy.ndimage
import sys

def dilate_mask(image_array, iterations):

    img = np.asarray(image_array)

    if img.shape[-1] == 4:
        image_last_dim = img[:,:,-1] // 255

        dil_mask = scipy.ndimage.binary_dilation(image_last_dim, iterations=iterations)
        img[:,:,-1] = np.uint8(dil_mask)*255

    return img

def erode_mask(image_array, iterations):

    img = np.asarray(image_array)

    if img.shape[-1] == 4:
        image_last_dim = img[:,:,-1] // 255

        dil_mask = scipy.ndimage.binary_erosion(image_last_dim, iterations=iterations)
        img[:,:,-1] = np.uint8(dil_mask)*255

    return img