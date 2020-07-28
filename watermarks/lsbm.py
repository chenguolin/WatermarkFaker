import numpy as np
from watermarks.base_watermark import BaseWatermark


class LSBMatching(BaseWatermark):
    def __init__(self, channel=0, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.channel = channel

    def __pixel_embed(self, origin, info):
        if info != 0 and info != 1:
            raise TypeError("Watermark for LSB Matching alg should be a binary image")
        
        if origin % 2 != info:
            if origin == 255:
                output = 254
            elif origin == 0:
                output = 1
            else:
                output = origin + np.random.randint(0, 2) * 2 -1
        else:
            output = origin
        return output

    def embed(self, image, watermark):
        watermark //= 255  # watermark should be a binary image
        image = image.astype('uint8')
        watermark = watermark.astype('uint8')

        if self.save_watermark:
            import cv2
            cv2.imwrite("./images/lsbm_watermark.png", watermark * 255)

        image_wm = np.copy(image)
        image_wm[:, :, self.channel] = np.vectorize(self.__pixel_embed)(image[:, :, self.channel], watermark)
        return image_wm

    def extract(self, image_wm, image=None):
        watermark_ = np.zeros(image_wm.shape)
        for i in range(image_wm.shape[0]):
            for j in range(image_wm.shape[1]):
                watermark_[i, j] = image_wm[i, j, self.channel] % 2 * 255
        return watermark_.astype('uint8')
