"""
This module contains simple helper functions.
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image as square to the disk.
    
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
        aspect_ratio (float)      -- the ratio between height and width
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), resample=Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), resample=Image.BICUBIC)
    image_pil.save(image_path)


def tensor2im(input_image, imtype=np.uint8):
    """Convert a Tensor array into a numpy image array.
    
    Parameters:
        input_image (tensor) -- the input image tensor array
        imtype (type)        -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # detach the tensor from current graph
            image_tensor = input_image.detach()
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))  # a -> [a,a,a]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # [c, h, w] -> [h,w,c] & [-1,1] -> [0,255]
    else:  # if it is a numoy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std and size of a numpy array
    
    Parameters:
        x (numpy array) -- the target numpy array
        val (bool)      -- if print the value of the numpy array
        shp (bool)      -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape:', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std = %3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdir(path):
    """Create a single empty directory recursively if it didn't exist.

    Parameters:
        path (str) -- a single directory path

    P.S. os.mkdir() is used to create a directory named path and NOT recursive.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """Create empty directories for a list of directory paths if they don't exist

    Parameter:
        paths (str list) -- a list of directory paths

    P.S. os.makedirs() is used to create A directory recursively.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
