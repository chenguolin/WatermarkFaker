import numpy as np
from watermarks.base_watermark import BaseWatermark


class LSB(BaseWatermark):
    def __init__(self, bits=2, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.bits = bits

    def __pixel_embed(self, origin, info):
        origin = format(origin, '08b')
        info = format(info, '08b')
        info_msb = info[:self.bits]
        origin_nolsb = origin[:-self.bits]
        output = int(origin_nolsb + info_msb, 2)
        return output

    def embed(self, image, watermark):
        if not isinstance(image[0][0], np.uint8):
            image.astype('uint8')
        if not isinstance(watermark[0][0], np.uint8):
            watermark.astype('uint8')
        
        if self.save_watermark:
            import cv2
            cv2.imwrite("./images/lsb_watermark.png", watermark)
        
        image_wm = np.vectorize(self.__pixel_embed)(image, watermark)
        return image_wm

    def __pixel_extract(self, output):
        output = format(output, '08b')
        lsb = output[-self.bits:]
        info_ = int(lsb + '0'*(8-self.bits), 2)
        return info_

    def extract(self, image_wm, image=None):
        watermark_ = np.vectorize(self.__pixel_extract)(image_wm)
        return watermark_
