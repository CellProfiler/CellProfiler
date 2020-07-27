from ._abstract_image import AbstractImage


class VanillaImage(AbstractImage):
    """This image provider returns the image given to it in the constructor

    """

    def __init__(self, name, image):
        """Constructor takes the name of the image and the CellProfiler.Image.Image instance to be returned
        """
        self.__name = name
        self.__image = image

    def get_name(self):
        return self.__name

    def provide_image(self, image_set):
        return self.__image

    def release_memory(self):
        self.__image = None
