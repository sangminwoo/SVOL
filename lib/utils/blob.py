# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import cv2
import numpy as np


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    # changed to use pytorch models
    im /= 255. # Convert range to [0,1]
    # normalization for pytroch pretrained models.
    # https://pytorch.org/docs/stable/torchvision/models.html
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdens = [0.229, 0.224, 0.225]
    
    # normalize manual
    im -= pixel_means # Minus mean    
    im /= pixel_stdens # divide by stddev

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def crop(image, purpose, size):

    h, w, c = image.shape
    cut_image = image[int(purpose[1]):int(purpose[3]),int(purpose[0]):int(purpose[2]),:]

    height, width = cut_image.shape[0:2]
    max_hw   = max(height, width)
    cty, ctx = [height // 2, width // 2]

    cropped_image  = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)

    x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
    y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]

    return cv2.resize(cropped_image, (size,size), interpolation=cv2.INTER_LINEAR)
