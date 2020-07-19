import numpy as np
from watermarks.base_watermark import BaseWatermark


class LSB(BaseWatermark):
    def __init__(self, bits=2, binary=False, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.binary = binary
        if self.binary:
            self.bits = 1
        else:
            self.bits = bits

    def __pixel_embed(self, origin, info):
        origin = format(origin, '08b')
        info = format(info, '08b')
        info_msb = info[:self.bits]
        origin_nolsb = origin[:-self.bits]
        output = int(origin_nolsb + info_msb, 2)
        return output

    def embed(self, image, watermark):
        image.astype('uint8')
        watermark.astype('uint8')
        
        if self.save_watermark:
            import cv2
            cv2.imwrite("./images/lsb_watermark.png", watermark)
        if self.binary:
            image_wm = np.copy(image)
            image_wm[:, :, 0] = np.vectorize(self.__pixel_embed)(image[:, :, 0], watermark)
        else:
            image_wm = np.vectorize(self.__pixel_embed)(image, watermark)
        return image_wm

    def __pixel_extract(self, output):
        output = format(output, '08b')
        lsb = output[-self.bits:]
        if self.binary:
            info_ = int(lsb) * 255
        else:
            info_ = int(lsb + '0'*(8-self.bits), 2)
        return info_

    def extract(self, image_wm, image=None):
        if self.binary:
            watermark_ = np.vectorize(self.__pixel_extract)(image_wm[:, :, 0])
        else:
            watermark_ = np.vectorize(self.__pixel_extract)(image_wm)
        return watermark_
