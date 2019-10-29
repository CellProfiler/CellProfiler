import logging

logger = logging.getLogger(__name__)


class AbstractImageProvider:
    """Represents an image provider that returns images
    """

    def provide_image(self, image_set):
        """Return the image that is associated with the image set
        """
        raise NotImplementedError("Please implement ProvideImage for your class")

    def __get_name(self):
        """Call the abstract function, "get_name"
        """
        return self.get_name()

    def get_name(self):
        """The user-visible name for the image
        """
        raise NotImplementedError("Please implement get_name for your class")

    def release_memory(self):
        """Release whatever memory is associated with the image"""
        logger.warning(
            "Warning: no memory release function implemented for %s image",
            self.get_name(),
        )

    name = property(__get_name)


class CallbackImageProvider(AbstractImageProvider):
    """An image provider proxy that calls the indicated callback functions (presumably in your module) to implement the methods
    """

    def __init__(self, name, image_provider_fn):
        """Constructor
        name              - name returned by the Name method
        image_provider_fn - function called during ProvideImage with the arguments, image_set and the CallbackImageProvider instance
        """

        self.__name = name
        self.__image_provider_fn = image_provider_fn

    def provide_image(self, image_set):
        return self.__image_provider_fn(image_set, self)

    def get_name(self):
        return self.__name


class VanillaImageProvider(AbstractImageProvider):
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
