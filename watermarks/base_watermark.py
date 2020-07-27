from abc import ABC, abstractmethod


class BaseWatermark(ABC):
    """This class an abstract class (ABC) for watermarks.

    To create a subclass, you need to implement the following three functions:
        -- <__init__>: initialize the class; first call BaceWatermark.__init__(self)
        -- <embed>:    embed `image` (numpy.ndarray) by `watermark` (numpy.ndarray)
        -- <extract>:  extract `watermark_` from `image_wm`; some algorithms require the original image
    """

    def __init__(self, save_watermark):
        """Initialize some parameters used in this watermark algorithm."""
        self.save_watermark = save_watermark
        pass

    @abstractmethod
    def embed(self, image, watermark):
        """Embed the image by the watermark.
        
        Parameters:
            image (numpy.ndarray)     -- the targe image i.e. cover
            watermark (numpy.ndarray) -- the information expected to hide

        Return:
            image_wm (numpy.ndarray)  -- the image with invisible watermark i.e. stego
        """
        pass

    @abstractmethod
    def extract(self, image_wm, image):
        """Extract the watermark (i.e. information) from embeded image (i.e. stego); some algorithms require the original image (i.e. cover).
        
        Parameters:
            image_wm (numpy.ndarray)     -- embeded image (i.e. stego)
            image (numpy.ndarray / None) -- original image; if don't need, it will be `None`

        Return:
            watermarl_ (numpy.ndarray) -- the watermark extracted from stego; it has a little different between the original watermark
        """
        pass
