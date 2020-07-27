import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
from watermarks.base_watermark import BaseWatermark


class DCT(BaseWatermark):
    def __init__(self, alpha=10, block_size=8, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.alpha = alpha
        self.block_size = block_size

    def embed(self, image, watermark):
        h1, w1 = image.shape
        image = image.astype('float32')
        h2, w2 = watermark.shape
        watermark = np.round(watermark / 255)  # [0, 255] -> {0, 1}; binary image

        if len(image.shape) != 2:
            raise TypeError("Image to embed DFT should be grayscale")
        if h1 != h2 * self.block_size and w1 != w2 * self.block_size:
            raise ValueError("Watermark's shape should be the `1/block_size` of image's shape")
        
        if self.save_watermark:
            cv2.imwrite('./images/dct_watermark.png', (watermark * 255).astype('uint8'))

        B = self.block_size  # temporary value
        image_wm = np.zeros(image.shape)
        for i in range(h2):
            for j in range(w2):
                sub_image = image[i*B : (i+1)*B, j*B : (j+1)*B]
                # sub_image_dct = dctn(sub_image, norm='ortho')
                sub_image_dct = cv2.dct(sub_image)
                sub_image_dct[0, 0] += (watermark[i, j] * 2 - 1) * self.alpha
                # image_wm[i*B : (i+1)*B, j*B : (j+1)*B] = idctn(sub_image_dct, norm='ortho')
                image_wm[i*B : (i+1)*B, j*B : (j+1)*B] = cv2.idct(sub_image_dct)
        return image_wm.astype('uint8')

    def extract(self, image_wm, image):
        h1, w1 = image.shape
        B = self.block_size
        h2, w2 = int(h1/B), int(w1/B)
        watermark_ = np.zeros((h2, w2))

        for i in range(h2):
            for j in range(w2):
                sub_image = image[i*B : (i+1)*B, j*B : (j+1)*B].astype('float32')
                sub_image_wm = image_wm[i*B : (i+1)*B, j*B : (j+1)*B].astype('float32')
                watermark_[i, j] = int(
                    # np.sum(dctn(sub_image_wm, norm='ortho') > dctn(sub_image, norm='ortho')) / self.block_size**2 > 0.5
                    cv2.dct(sub_image_wm)[0, 0] > cv2.dct(sub_image)[0, 0]
                )
        return (watermark_ * 255).astype('uint8')
