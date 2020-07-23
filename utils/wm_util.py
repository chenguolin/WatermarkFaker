"""
This module contains some helper functions related to watermarks.
"""
import os
import cv2
import skimage
import argparse
import numpy as np
from . import util


def embed_dataset(alg, source_dir, watermark_path, output_dir, RGB_im=True, 
                    RGB_wm=True, combine=True):
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
        if 'A' in source_dir:  # get original images from unaligned-files
            output_path = os.path.join(output_dir, os.path.splitext(file_name)[0][:-2]+os.path.splitext(file_name)[1])
        else:
            output_path = os.path.join(output_dir, file_name)

        image = cv2.imread(image_path, flags=int(RGB_im))
        watermark = cv2.imread(watermark_path, flags=int(RGB_wm))
        image_wm = alg.embed(image, watermark)

        if combine:
            output = np.concatenate((image, image_wm), axis=1)  # concatenate by column
        else:
            output = image_wm

        cv2.imwrite(output_path, output)


def test_watermark(alg, image_path="./images/test.png", watermark_path="./images/lena.png", 
                    suffix='', RGB_im=True, RGB_wm=True):
    """Apply the `lena.png` as watermark algorithm to `test.png`.
    
    Parameter:
        alg (class instance) -- a watermark algorithm
        RGB (bool)           -- read RGB (True) or grayscale (False)
    """
    image = cv2.imread(image_path, flags=int(RGB_im))
    watermark = cv2.imread(watermark_path, flags=int(RGB_wm))
    
    image_wm = alg.embed(image, watermark)
    cv2.imwrite("./images/test_" + suffix + ".png", util.tensor2im(image_wm))

    watermark_ = alg.extract(image_wm, image)
    cv2.imwrite("./images/cross_" + suffix + ".png", util.tensor2im(watermark_))


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


def read_image_from_txt(path="./images/lena_b.txt", shape=(256, 256)):
    """Read a binary image from a .txt file which only contains 0 & 1.
    
    Parameter:
        path (str)    -- path of the .txt file
        shape (tuple) -- shape of the output image

    Return:
        image (numpy.array) -- a binary image whose pixels are either 0 or 255
    """
    with open(path, 'r') as f:
        string = f.read()
        print(len(string))
        image = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                image[i, j] = int(string[i*shape[0] + j]) * 255
        cv2.imwrite("./images/binary_image.png", image.astype('uint8'))
