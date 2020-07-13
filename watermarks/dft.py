import numpy as np
from watermarks.base_watermark import BaseWatermark


class DFT(BaseWatermark):
    def __init__(self, alpha=5, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.alpha = alpha

    def embed(self, image, watermark):
        if len(image.shape) != 2:
            raise TypeError("Image to embed DFT should be grayscale")
        if watermark.shape[0] != image.shape[0] / 2 and watermark.shape[1] != image.shape[1] / 2:
            raise ValueError("Watermark's shape should be the half of image's")

        # fourier's transformation
        image_f = np.fft.fft2(image)
        # broad watermark
        broad_watermark = np.zeros(image.shape)
        for i in range(watermark.shape[0]):
            for j in range(watermark.shape[1]):
                broad_watermark[i][j] = watermark[i][j]
                broad_watermark[image.shape[0] - i - 1][image.shape[1] - j - 1] = broad_watermark[i][j]
        if self.save_watermark:
            import cv2
            cv2.imwrite('./images/dft_watermark.png', broad_watermark)
        
        image_wm_f = image_f + self.alpha * broad_watermark
        image_wm = np.real(np.fft.ifft2(image_wm_f))
        return image_wm

    def extract(self, image_wm, image):
        image_f = image
        image_wm_f = np.fft.fft2(image_wm)
        
        watermark_ = np.real((image_wm_f - image_f) / self.alpha)
        h, w = watermark_.shape
        return watermark_[:int(h/2), :int(w/2)]
