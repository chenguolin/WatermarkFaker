import numpy as np
from watermarks.base_watermark import BaseWatermark


class LSBMR(BaseWatermark):
    def __init__(self, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)

    def __func(self, x1, x2):
        f = (x1 // 2 + x2) % 2
        return f

    def embed(self, image, watermark):

        h, w = image.shape[:2]
        if image.ndim == 3:
            image_c = image[:, :, 0].flatten()
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
            image_wm_c = image_wm[:, :, 0].flatten()
        elif image.ndim == 2:
            image_wm_c = image_wm.flatten()

        # after embedding, w_i = LSB(y_i); w_i+1 = __func(y_i, y_i+1)
        for i in range(0, h*w-1, 2):
            y1, y2 = 0, 0
            if watermark[i] == image_c[i] % 2:
                if watermark[i+1] != self.__func(image_c[i], image_c[i+1]):
                    y2 = image_c[i+1] + 1
                else:
                    y2 = image_c[i+1]
                y1 = image_c[i]
            else:
                if watermark[i+1] == self.__func(image_c[i]-1, image_c[i+1]):
                    y1 = image_c[i] - 1
                else:
                    y1 = image_c[i] + 1
                y2 = image_c[i+1]
            image_wm_c[i] = y1
            image_wm_c[i+1] = y2

        if image.ndim == 3:
            image_wm[:, :, 0] = image_wm_c.reshape((h, w))
        elif image.ndim == 2:
            image_wm = image_wm_c.reshape((h, w))
        return image_wm

    def extract(self, image_wm, image):
        watermark_ = []
        h, w = image_wm.shape[:2]
        
        if image_wm.ndim == 3:
            image_wm_c = image_wm[:, :, 0].flatten()
            image_c = image[:, :, 0].flatten()
        elif image_wm.ndim == 2:
            image_wm_c = image_wm.flatten()
            image_c = image.flatten()

        for i in range(0, h*w-1, 2):
            y1 = image_wm_c[i]
            y2 = image_wm_c[i+1]

            x1 = image_c[i]
            x2 = image_c[i+1]

            w1 = x1 % 2
            if x1 == y1:
                watermark_.append(w1)
                w2 = self.__func(x1, x2)
                if y2-1 == x2:
                    wt = 1 - w2
                    watermark_.append(wt)
                else:
                    watermark_.append(w2)
            else:
                watermark_.append(1 - w1)
                w2 = self.__func(x1-1, x2)
                if x1 == y1+1:
                    watermark_.append(w2)
                else:
                    watermark_.append(1 - w2)
        return (np.array(watermark_).reshape((h, w)) * 255).astype('uint8')
