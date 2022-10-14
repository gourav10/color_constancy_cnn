# -*- coding: utf-8 -*-
"""
@author: Gourav Beura
"""

import cv2
import numpy as np

def correct_color_single_image(img, illuminant):
    """
    Apply color correction via Von-Kries Model
    Args:
        img (np.array): input (linear) image
        illuminant (np.array): RGB  color of light source
    Returns:
        img_corr (np.array): corrected image s.t. to be
        taken under a canonical perfect white light source
    """
    highest_gain = np.max(illuminant)
    gain = highest_gain / illuminant
    img_corr = img.copy()
    for idx_channel in range(img.shape[2]):
        img_corr[:, :, idx_channel] = img_corr[:, :, idx_channel] * gain[idx_channel]

    return img_corr



def srgb_gamma(img):
    """
    Apply gamma correction (forward transformation)
    (https://en.wikipedia.org/wiki/SRGB)

    Args:
        img (np.array): input (linear) image
    Returns:
        img (np.array): non-linear/gamma-compressed image
    """

    for idx_channel in range(img.shape[2]):
        this_channel = img[:, :, idx_channel]
        img[:, :, idx_channel] = 12.92 * this_channel * (this_channel <= 0.0031308) + (
                    1.055 * this_channel ** (1 / 2.4) - 0.055) * (this_channel > 0.0031308)

    return img