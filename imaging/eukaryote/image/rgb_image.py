class RGBImage(object):
    """A wrapper that discards the alpha channel
    
    This is meant to be used if the image is 3-d + alpha but the alpha
    channel is discarded
    """

    def __init__(self, image):
        self.__image = image

    def __getattr__(self, name):
        return getattr(self.__image, name)

    def get_pixel_data(self):
        '''Return the pixel data without the alpha channel'''
        return self.__image.pixel_data[:, :, :3]

    pixel_data = property(get_pixel_data)
