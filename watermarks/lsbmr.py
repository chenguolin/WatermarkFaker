import numpy as np
from watermarks.base_watermark import BaseWatermark


class LSBMR(BaseWatermark):
    def __init__(self, channel=0, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.channel = channel

    def __func(self, x1, x2):
        f = (x1 // 2 + x2) % 2
        return f

    def embed(self, image, watermark):
        h, w = image.shape[:2]
        if image.ndim == 3:
            image_c = image[:, :, self.channel].flatten()
        elif image.ndim == 2:
            image_c = image.flatten()
        watermark = (watermark // 255).flatten()
        
        if len(image_c) != len(watermark):
            raise TypeError("Watermark size should be the same as cover image; different size version maybe implemented in the future")
        
        if self.save_watermark:
            import cv2
            cv2.imwrite("./images/lsbmr.png", watermark.reshape((h, w)))

        image_wm = np.copy(image)
        if image.ndim ==3:
            image_wm_c = image_wm[:, :, self.channel].flatten()
        elif image.ndim == 2:
            image_wm_c = image_wm.flatten()
        # after embedding, w_i = LSB(y_i); w_i+1 = __func(y_i, y_i+1)
        for i in range(0, h*w-1, 2):
            if watermark[i] == image_c[i] % 2:
                if watermark[i+1] != self.__func(image_c[i], image_c[i+1]):
                    image_wm_c[i+1] = image_c[i+1] + 1
                else:
                    image_wm_c[i+1] = image_c[i+1]
                image_wm_c[i] = image_c[i]
            else:
                if watermark[i+1] == self.__func(image_c[i]-1, image_c[i+1]):
                    image_wm_c[i] = image_c[i] - 1
                else:
                    image_wm_c[i] = image_c[i] + 1
                image_wm_c[i+1] = image_c[i+1]
        if image.ndim == 3:
            image_wm[:, :, self.channel] = image_wm_c.reshape((h, w))
        elif image.ndim == 2:
            image_wm = image_wm_c.reshape((h, w))
        return image_wm

    def extract(self, image_wm, image=None):
        watermark_ = []
        h, w = image_wm.shape[:2]
        if image_wm.ndim == 3:
            image_wm_c = image_wm[:, :, self.channel].flatten()
        elif image_wm.ndim == 2:
            image_wm_c = image_wm.flatten()
        for i in range(0, h*w-1, 2):
            w1 = image_wm_c[i] % 2
            w2 = self.__func(image_wm_c[i], image_wm_c[i+1])
            watermark_.append(w1)
            watermark_.append(w2)
        return (np.array(watermark_).reshape((h, w)) * 255).astype('uint8')
