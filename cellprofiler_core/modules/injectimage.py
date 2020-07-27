import hashlib

import numpy

from ..image import Image
from ..module import Module
from ..setting.text import Name


class InjectImage(Module):
    """This module is intended for testing. It injects an image into the
    image set.
    """

    module_name = "InjectImage"
    variable_revision_number = 1

    def __init__(self, image_name, image, mask=None, release_image=False):
        """Initializer

        image_name - the name of the image to put into the image set

        image - either the pixel data for the image if adding the same
                image to every set or a list or tuple of pixel data,
                one per image set

        mask - None for no mask (default), a binary 2-d matrix if same mask
               for all image sets or a list or tuple of masks if one
               different mask per image set.
        """
        super(InjectImage, self).__init__()
        self.__image_name = image_name
        self.__image = image
        self.__mask = mask
        self.image_name = Name("Hardwired image name", "imagegroup", image_name)
        self.__release_image = release_image

    def settings(self):
        return [self.image_name]

    def visible_settings(self):
        return [self.image_name]

    def get_help(self):
        """Return help text for the module

        """
        raise NotImplementedError(
            "Please implement GetHelp in your derived module class"
        )

    def write_to_handles(self, handles):
        """Write out the module's state to the handles

        """

    def write_to_text(self, file):
        """Write the module's state, informally, to a text file
        """

    def prepare_run(self, workspace):
        digest = hashlib.md5()
        digest.update(numpy.ascontiguousarray(self.__image).data)

        workspace.measurements.add_measurement(
            "Image", "MD5Digest_%s" % self.__image_name, 1, image_set_number=1,
        )

        return True

    def run(self, workspace):
        """Run the module (abstract method)

        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        if isinstance(self.__image, (tuple, list)):
            image = self.__image[workspace.image_set.image_number - 1]
        else:
            image = self.__image
        if isinstance(self.__mask, (tuple, list)):
            mask = self.__mask[workspace.image_set.image_number - 1]
        else:
            mask = self.__mask
        image = Image(image, mask)
        workspace.image_set.add(self.__image_name, image)

    def post_run(self, workspace):
        if self.__release_image:
            del self.__image
            del self.__mask

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
