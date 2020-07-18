import numpy as np
from skimage.measure import compare_ssim
from watermarks.base_watermark import BaseWatermark


class RobustLSB(BaseWatermark):
    def __init__(self, k=32, threshold=0.97, verbose=False, save_watermark=False):
        BaseWatermark.__init__(self, save_watermark)
        self.k = k  # divide the image and watermark into blocks, whose shape is k * k
        self.threshold = threshold
        self.verbose = verbose
        self.save_watermark = save_watermark
        self.bit = 0

    def __pixel_replace(self, origin, info):
        origin = format(origin, '08b')
        info = format(info, '08b')
        info_bit = info[self.bit]
        output_binary = origin[:(7-self.bit)] + info_bit + origin[(7-self.bit+1):]
        output = int(output_binary, 2)
        return output

    def embed(self, image, watermark):
        if not isinstance(image[0][0], np.uint8):
            image.astype('uint8')
        if not isinstance(watermark[0][0], np.uint8):
            watermark.astype('uint8')
        
        if self.save_watermark:
            import cv2
            cv2.imwrite("./images/rlsb_watermark.png", watermark)

        k = self.k
        h, w = image.shape[:2]
        RGB = (image.ndim == 3 and image.shape[2] == 3)
        image_wm = np.zeros(image.shape)
        
        for i in range(int(h/k)):
            for j in range(int(w/k)):
                sub_image = image[i*k:(i+1)*k, j*k:(j+1)*k]
                sub_watermark = watermark[i*k:(i+1)*k, j*k:(j+1)*k]    
                sub_image_wm = np.vectorize(self.__pixel_replace)(sub_image, sub_watermark)
                self.bit += 1
                while compare_ssim(sub_image, sub_image_wm, multichannel=RGB, data_range=255) >= self.threshold:
                    sub_image_wm = np.vectorize(self.__pixel_replace)(sub_image_wm, sub_watermark)
                    self.bit += 1
                image_wm[i*k:(i+1)*k, j*k:(j+1)*k] = sub_image_wm
                if self.verbose:
                    print(i, j, self.bit)
                    if RGB:
                        print(format(sub_image[0, 0, 0], '08b'), format(sub_image_wm[0, 0, 0], '08b'), 
                                format(sub_watermark[0, 0, 0], '08b'))
                    else:
                        print(format(int(sub_image[0, 0]), '08b'), format(int(sub_image_wm[0, 0]), '08b'), 
                                format(int(sub_watermark[0, 0]), '08b'))
                self.bit = 0
        return image_wm

    def __find(self, string, target):
        index = string.find(target)
        if index == -1:
            index = 8  # didn't found
        return index

    def __find_embed_bits(self, sub_image_wm, sub_image):
        xor = np.bitwise_xor(sub_image_wm, sub_image)
        binary_form = np.vectorize(format)(xor, '08b')
        bits = 8 - np.min(np.vectorize(self.__find)(binary_form, '1'))
        return bits

    def __reverse(self, string):
        return string[::-1]

    def __byte2int(self, byte):
        return int(byte, 2)

    def __sub_extract(self, sub_image_wm, bits):
        mask = (1 << bits) - 1  # e.g. bits:3 -> mask: 00000111b
        masked = np.bitwise_and(sub_image_wm, mask)
        wm_binary_form = np.vectorize(self.__reverse)(np.vectorize(format)(masked, '08b'))
        sub_watermark_ = np.vectorize(self.__byte2int)(wm_binary_form)
        return sub_watermark_

    def extract(self, image_wm, image):
        k = self.k
        h, w = image.shape[:2]
        RGB = (image.ndim == 3 and image.shape[2] == 3)
        watermark_ = np.zeros(image.shape)

        for i in range(int(h/k)):
            for j in range(int(w/k)):
                sub_image_wm = image_wm[i*k:(i+1)*k, j*k:(j+1)*k].astype('uint8')
                sub_image = image[i*k:(i+1)*k, j*k:(j+1)*k].astype('uint8')

                bits = self.__find_embed_bits(sub_image_wm, sub_image)
                sub_watermark_ = self.__sub_extract(sub_image_wm, bits)
                watermark_[i*k:(i+1)*k, j*k:(j+1)*k] = sub_watermark_
        return watermark_
