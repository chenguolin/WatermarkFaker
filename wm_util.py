"""
This module contains some helper functions related to watermarks.
"""
import os
import cv2
import argparse
from watermarks import lsb, dft
from utils import util


def embed_dataset(algorithm, source_path, watermark_path, output_path, RGB=True):
    """Embed original dataset by a watermark algorithm

    Parameters:
        algorithm (class)    -- a watermark algorithm
        source_path (str)    -- the path of original images file
        watermark_path (str) -- the path of the watermark image
        output_path (str)    -- the path to save watermarked images
        RGB (bool)           -- read RGB (True) or grayscale (False)
    """
    alg = algorithm()
    util.mkdir(output_path)
    for file_name in os.listdir(source_path):
        image_path = os.path.join(source_path, file_name)
        image_wm_path = os.path.join(output_path, file_name)

        image = cv2.imread(image_path, flags=RGB)
        watermark = cv2.imread(watermark_path, flags=RGB)

        cv2.imwrite(image_wm_path, alg.embed(image, watermark))


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
        watermark = cv2.resize(cv2.imread("./images/lena.png", flags=0), (128, 128), interpolation=cv2.INTER_CUBIC)

        image_wm = alg.embed(image, watermark)
        cv2.imwrite("./images/test_.png", image_wm)

        watermark_ = alg.extract(image_wm, image)
        cv2.imwrite("./images/lena_.png", watermark_)

    else:
        raise NotImplementedError("Please use watermark classes [LSB | DFT]")


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


if __name__ == '__main__':
    test_watermark(dft.DFT)