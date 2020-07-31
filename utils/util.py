"""
This module contains simple helper functions.
"""
import os
import cv2
import torch
import torchvision
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
    """Convert a batch Tensor array into a numpy image array.
    
    Parameters:
        input_image (tensor) -- the input image tensor array
        imtype (type)        -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # detach the tensor from current graph
            image_tensor = input_image.detach()
        else:
            raise TypeError("Type of the input is neither `np.ndarray` nor `torch.Tensor`")
        image_numpy = image_tensor[0].cpu().float().numpy()  # .numpy() will cause deviation on pixels  e.g. tensor(-0.5059) -> array(0.5058824)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))  # a -> [a,a,a]
        image_numpy = np.round((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0)  # [c, h, w] -> [h,w,c] & [-1,1] -> [0,255]
    else:  # if it is a numoy array, do nothing or transform to RGB
        image_numpy = input_image
        if image_numpy.ndim == 2:
            image_numpy = np.expand_dims(image_numpy, 2)
        if image_numpy.shape[2] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (1, 1, 3))  # a -> [a,a,a]
    return image_numpy.astype(imtype)


def im2tensor(input_image):
    """Convert a numpy/PIL image array into a batch Tensor"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    return transform(input_image).unsqueeze_(dim=0)


def bits2im(input_bits, imtype=np.uint8):
    """Convert a bits Tensor (8x3 channels) array into a numpy image array.
    
    In the begining, images are read by Image; which means color images are read in the order of `R->G->B` (different from cv2)
    """
    if not isinstance(input_bits, torch.Tensor):
        raise TypeError("Type of input bit layers should be torch.Tensor")
    bits_tensor = input_bits.detach()
    bits_numpy = bits_tensor[0].cpu().float().numpy()                       # get the first batch
    if bits_numpy.shape[0] == 8:
        bits_numpy = np.tile(bits_numpy, (3, 1, 1))
    bits_numpy = np.round((np.transpose(bits_numpy, (1, 2, 0)) + 1) / 2.0)  # (256, 256, 24)
    RGB = []
    pow2 = np.array([128, 64, 32, 16, 8, 4, 2, 1])
    for i in range(3):
        RGB.append(np.sum(bits_numpy[:, :, i*8:(i+1)*8] * pow2, axis=2))
    image_numpy = np.array(RGB).transpose((1, 2, 0))                        # (256, 256, 3)
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
