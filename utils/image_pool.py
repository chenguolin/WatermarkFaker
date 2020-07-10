import torch
import random


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
     
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    
    i.e. it helps the discriminator doesn't forget what it has done wrong before
    reference: https://arxiv.org/pdf/1612.07828.pdf Sec2.3
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameter:
            pool_size (int) -- the size of image buffer; if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
        
    def query(self, images):
        """Return an image from the pool.

        Parameter:
            images (tensor) - the lateset generated images from the generator

        Returns images from the buffer.

        By 50% chance, the buffer will return input images.
        By 50% chance, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do noting
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full: keep inserting current images to the buffer
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image and replace it by the current image
                    # helps the discriminator doesn't forget what it has done wrong before
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by anthoer 50% chance, the buffer will return the current image
                    return_images.append(image)
            return_images = torch.cat(return_images, 0)  # collect all the images and return
            return return_images
