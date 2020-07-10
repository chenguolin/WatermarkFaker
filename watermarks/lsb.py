import numpy as np


class LSB:
    def __init__(self):
        super().__init__()

    def __pixel_embed(self, origin, info):
        origin = format(origin, '08b')
        info = format(info, '08b')
        info_msb = info[:2]
        origin_nolsb = origin[:-2]
        output = int(origin_nolsb + info_msb, 2)
        return output

    def embed(self, image, watermark):
        image_wm = np.vectorize(self.__pixel_embed)(image, watermark)
        return image_wm

    def __pixel_extract(self, output):
        output = format(output, '08b')
        lsb = output[-2:]
        info_ = int(lsb + '0'*6, 2)
        return info_

    def extract(self, image_wm):
        watermark_ = np.vectorize(self.__pixel_extract)(image_wm)
        return watermark_
