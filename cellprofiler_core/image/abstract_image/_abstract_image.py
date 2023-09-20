import logging


LOGGER = logging.getLogger(__name__)

class AbstractImage:
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
        LOGGER.warning(
            "Warning: no memory release function implemented for %s image",
            self.get_name(),
        )

    name = property(__get_name)
