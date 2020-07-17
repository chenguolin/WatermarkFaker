"""
This module contains some helper functions related to watermarks.
"""
import os
import cv2
import argparse
import numpy as np
from . import util


def embed_dataset(alg, source_dir, watermark_path, output_dir, RGB=True, combine=True):
    """Embed original dataset by a watermark algorithm.

    Parameters:
        alg (class instance) -- a watermark algorithm
        source_dir (str)     -- the path of dir that contains original images
        watermark_path (str) -- the path of the watermark image
        output_dir (str)     -- the path to dir to save watermarked images
        RGB (bool)           -- read RGB (True) or grayscale (False)
        combine (bool)       -- if True: combine original and watermarked images as paired images
    """
    util.mkdir(output_dir)
    for file_name in os.listdir(source_dir):
        image_path = os.path.join(source_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        image = cv2.imread(image_path, flags=RGB)
        watermark = cv2.imread(watermark_path, flags=RGB)
        image_wm = alg.embed(image, watermark)

        if combine:
            output = np.concatenate((image, image_wm), axis=1)  # concatenate by column
        else:
            output = image_wm

        cv2.imwrite(output_path, output)
        break


def test_watermark(algorithm):
    """Apply the `lena.png` as watermark algorithm to `test.png`.
    
    Parameter:
        algorithm (class) -- a watermark algorithm
    """
    alg = algorithm()

    if isinstance(alg, lsb.LSB):
        image = cv2.imread("./images/test.png", flags=1)  # RGB
        watermark = cv2.imread("./images/lena.png", flags=1)

        image_wm = alg.embed(image, watermark)
        cv2.imwrite("./images/test_.png", image_wm)

        watermark_ = alg.extract(image_wm)
        cv2.imwrite("./images/lena_.png", watermark_)

    elif isinstance(alg, dft.DFT):
        image = cv2.imread("./images/test.png", flags=0)  # grayscale
        h1, w1 = image.shape
        watermark = cv2.resize(cv2.imread("./images/lena.png", flags=0), (int(h1/2), int(w1/2)), interpolation=cv2.INTER_CUBIC)

        image_wm = alg.embed(image, watermark)
        cv2.imwrite("./images/test_.png", image_wm)

        watermark_ = alg.extract(image_wm, image)
        cv2.imwrite("./images/lena_.png", watermark_)

    elif isinstance(alg, dct.DCT):
        image = cv2.imread("./images/test.png", flags=0)
        h1, w1 = image.shape
        watermark = cv2.resize(cv2.imread("./images/lena.png", flags=0), (256, 256), interpolation=cv2.INTER_CUBIC)

        image_wm = alg.embed(image, watermark)
        cv2.imwrite("./images/test_.png", image_wm)

        watermark_ = alg.extract(image_wm, image)
        cv2.imwrite("./images/lena_.png", watermark_)

    else:
        raise NotImplementedError("Please use watermark classes [LSB | DFT | DWT]")


def combine(left, right, output, RGB=True):
    """Combine paired images.
    
    Parameters:
        left (str)   -- the path of original images
        right (str)  -- the path of watermarked images
        output (str) -- the path to save aligned images
        RGB (bool)   -- read RGB (True) or grayscale (False)
    """
    util.mkdir(output)
    for filename in os.listdir(left):
        img1 = cv2.imread(os.path.join(left, filename), flags=RGB)
        img2 = cv2.imread(os.path.join(right, filename), flags=RGB)
        img = np.concatenate((img1, img2), axis=1)  # concatenate by column
        cv2.imwrite(os.path.join(output, filename), img)
