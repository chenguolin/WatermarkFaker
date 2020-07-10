"""
This module contains some helper functions related to watermarks.
"""
import os
import cv2
import argparse
from watermarks.lsb import LSB
from utils import util


def embed_dataset(algorithm, source_path, watermark_path, output_path):
    """Embed original dataset by a watermark algorithm

    Parameters:
        algorithm (class)    -- a watermark algorithm
        source_path (str)    -- the path of original images file
        watermark_path (str) -- the path of the watermark image
        output_path (str)    -- the path to save watermarked images
    """
    alg = algorithm()
    util.mkdir(output_path)
    for file_name in os.listdir(source_path):
        image_path = os.path.join(source_path, file_name)
        image_wm_path = os.path.join(output_path, file_name)

        image = cv2.imread(image_path, flags=1)
        watermark = cv2.imread(watermark_path, flags=1)

        cv2.imwrite(image_wm_path, alg.embed(image, watermark))
        break


def test_watermark(algorithm):
    """Apply the `lena.png` as watermark algorithm to `test.png`.
    
    Parameter:
        algorithm (class) -- a watermark algorithm
    """
    alg = algorithm()

    image = cv2.imread("./images/test.png")
    watermark = cv2.imread("./images/lena.png")

    image_wm = alg.embed(image, watermark)
    cv2.imwrite("./images/test_.png", image_wm)

    watermark_ = alg.extract(image_wm)
    cv2.imwrite("./images/lena_.png", watermark_)


def combine(left, right, output):
    """Combine paired images.
    
    Parameters:
        left (str)   -- the path of original images
        right (str)  -- the path of watermarked images
        output (str) -- the path to save aligned images
    """
    util.mkdir(output)
    for filename in os.listdir(left):
        img1 = cv2.imread(os.path.join(left, filename), flags=1)
        img2 = cv2.imread(os.path.join(right, filename), flags=1)
        img = np.concatenate((img1, img2), axis=1)  # concatenate by column
        cv2.imwrite(os.path.join(output, filename), img)


if __name__ == '__main__':
    pass