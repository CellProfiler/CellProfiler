import logging

import numpy

from ._grayscale_image import GrayscaleImage
from ._rgb_image import RGBImage
from .abstract_image import VanillaImage


LOGGER = logging.getLogger(__name__)

class ImageSet:
    """Represents the images for a particular iteration of a pipeline

    An image set is composed of one image provider per image in the set.
    The image provider loads or creates an image, given a dictionary of keys
    (which might represent things like the plate/well for the image set or the
    frame number in a movie, etc.)
    """

    def __init__(self, number, keys, legacy_fields):
        """Constructor:
        number = image set index
        keys = dictionary of key/value pairs that uniquely identify the image set
        """
        self.__image_providers = []
        self.__images = {}
        self.keys = keys
        self.number = number
        self.legacy_fields = legacy_fields
        self.image_number = number + 1

    def get_image(
        self,
        name,
        must_be_binary=False,
        must_be_color=False,
        must_be_grayscale=False,
        must_be_rgb=False,
    ):
        """Return the image associated with the given name

        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        must_be_rgb - raise an exception if 2-d or if # channels not 3 or 4,
                      discard alpha channel.
        """
        name = str(name)
        if name not in self.__images:
            image = self.get_image_provider(name).provide_image(self)

        else:
            image = self.__images[name]

        if image.multichannel:
            if must_be_binary:
                raise ValueError("Image must be binary, but it was color")

            if must_be_grayscale:
                pd = image.pixel_data

                pd = pd.transpose(-1, *list(range(pd.ndim - 1)))

                if (
                    pd.shape[-1] >= 3
                    and numpy.all(pd[0] == pd[1])
                    and numpy.all(pd[0] == pd[2])
                ):
                    return GrayscaleImage(image)

                raise ValueError("Image must be grayscale, but it was color")

            if must_be_rgb:
                if image.pixel_data.shape[-1] not in (3, 4):
                    raise ValueError(
                        "Image must be RGB, but it had %d channels"
                        % image.pixel_data.shape[-1]
                    )

                if image.pixel_data.shape[-1] == 4:
                    LOGGER.warning("Discarding alpha channel.")

                    return RGBImage(image)

            return image

        if must_be_binary and image.pixel_data.dtype != bool:
            raise ValueError("Image was not binary")

        if must_be_grayscale and image.pixel_data.dtype.kind == "b":
            return GrayscaleImage(image)

        if must_be_rgb:
            raise ValueError("Image must be RGB, but it was grayscale")

        if must_be_color:
            raise ValueError("Image must be color, but it was grayscale")

        return image

    @property
    def providers(self):
        """The list of providers (populated during the image discovery phase)"""
        return tuple(self.__image_providers)
    
    def add_provider(self, provider):
        self.__image_providers.append(provider)

    def get_image_provider(self, name):
        """Get a named image provider

        name - return the image provider with this name
        """
        providers = [x for x in self.__image_providers if x.name == name]
        assert len(providers) > 0, "No provider of the %s image" % name
        assert len(providers) == 1, "More than one provider of the %s image" % name
        return providers[0]

    def remove_image_provider(self, name):
        """Remove a named image provider

        name - the name of the provider to remove
        """
        self.__image_providers = [x for x in self.__image_providers if x.name != name]

    def clear_image(self, name):
        """Remove the image memory associated with a provider

        name - the name of the provider
        """
        self.get_image_provider(name).release_memory()
        if name in self.__images:
            del self.__images[name]

    @property
    def names(self):
        """Get the image provider names
        """
        return [provider.name for provider in self.providers]

    def add(self, name, image):
        old_providers = [
            provider for provider in self.providers if provider.name == name
        ]
        if len(old_providers) > 0:
            self.clear_image(name)
        for provider in old_providers:
            self.remove_image_provider(provider.name)
        provider = VanillaImage(name, image)
        self.add_provider(provider)
