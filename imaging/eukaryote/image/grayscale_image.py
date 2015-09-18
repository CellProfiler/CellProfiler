class GrayscaleImage(object):
    """A wrapper around a non-grayscale image
    
    This is meant to be used if the image is 3-d but all channels
       are the same or if the image is binary.
    """

    def __init__(self, image):
        self.__image = image

    def __getattr__(self, name):
        return getattr(self.__image, name)

    def get_pixel_data(self):
        """One 2-d channel of the color image as a numpy array"""
        if self.__image.pixel_data.dtype.kind == 'b':
            return self.__image.pixel_data.astype(np.float64)
        return self.__image.pixel_data[:, :, 0]

    pixel_data = property(get_pixel_data)
