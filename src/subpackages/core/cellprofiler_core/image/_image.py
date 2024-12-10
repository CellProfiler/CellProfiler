import math
import numpy

from ..utilities.image import crop_image


class Image:
    """
    image - a Numpy array

    mask - a binary image indicating the points of interest in the image. The
    mask is the same size as the child image.

    crop_mask - the binary image used to crop the parent image to the
    dimensions of the child (this) image. The crop_mask is the same size as
    the parent image.

    parent_image - for derived images, the parent that was used to create this
    image. This image may inherit attributes from the parent image, such as
    the masks used to create the parent

    masking_objects - the labels matrix from these objects is used to mask and
    crop the parent image to make this image. The labels are available as
    mask_labels and crop_labels.

    convert - true to try to coerce whatever dtype passed (other than bool or
    float) to a scaled image.

    path_name - the path name to the file holding the image or None for a
    derived image

    file_name - the file name of the file holding the image or None for a
    derived image

    scale - the scaling suggested by the initial image format (e.g., 4095 for
    a 12-bit a/d converter).

    Resolution of mask and cropping_mask properties:

    The Image class looks for the mask and cropping_mask in the following
    places:

    * self: if set using the properties or specified in the initializer

    * masking_objects: if set using the masking_object property or specified
    in the initializer. The crop_mask and mask are composed of all of the
    labeled points.

    * parent_image: if set using the initializer. The child image inherits the
    mask and cropping mask of the parent.

    Otherwise, the image has no mask or cropping mask and all pixels are
    significant.
    """

    def __init__(
        self,
        image=None,
        mask=None,
        crop_mask=None,
        parent_image=None,
        masking_objects=None,
        convert=True,
        path_name=None,
        file_name=None,
        scale=None,
        dimensions=2,
        spacing=None,
        channelstack=False
    ):
        self.__image = None

        self.__mask = None

        self.__has_mask = False

        self.parent_image = parent_image

        self.__crop_mask = None

        self.__has_crop_mask = False

        if crop_mask is not None:
            self.crop_mask = crop_mask

        self.masking_objects = masking_objects

        self.__scale = scale

        if image is not None:
            self.set_image(image, convert)

        if mask is not None:
            self.mask = mask

        self.__file_name = file_name

        self.__path_name = path_name

        self.channel_names = None

        self.dimensions = dimensions

        self.__spacing = spacing

        self.__channelstack = channelstack

    @property
    def multichannel(self):
        return True if self.pixel_data.ndim == self.dimensions + 1 else False

    @property
    def volumetric(self):
        if self.dimensions == 3:
            return True

        return False

    @property
    def spacing(self):
        if self.__spacing is not None:
            return tuple(numpy.divide(self.__spacing, self.__spacing[1]))

        if self.parent_image is None:
            return (1.0,) * self.dimensions

        return self.parent_image.spacing

    @spacing.setter
    def spacing(self, spacing):
        self.__spacing = spacing

    def get_image(self):
        """Return the primary image"""
        return self.__image

    def set_image(self, image, convert=True):
        """Set the primary image

        Convert the image to a numpy array of dtype = np.float64.
        Rescale according to Matlab's rules for im2double:
        * single/double values: keep the same
        * uint8/16/32/64: scale 0 to max to 0 to 1
        * int8/16/32/64: scale min to max to 0 to 1
        * logical: save as is (and get if must_be_binary)
        """
        img = numpy.asanyarray(image)
        if img.dtype.name == "bool" or not convert:
            self.__image = img
            return
        mval = 0.0
        scale = 1.0
        fix_range = False
        # unsigend and signed integer types
        if numpy.issubdtype(img.dtype, numpy.integer):
            scale = float(numpy.iinfo(img.dtype).max)
        # signed integer types need shifting
        if numpy.issubdtype(img.dtype, numpy.signedinteger):
            mval = float(numpy.iinfo(img.dtype).min)
            fix_range = True
        # Avoid temporaries by doing the shift/scale in place.
        img = img.astype(numpy.float32)
        img -= mval
        img /= scale
        if fix_range:
            # These types will always have ranges between 0 and 1. Make it so.
            numpy.clip(img, 0, 1, out=img)
        self.__image = img

    image = property(get_image, set_image)
    pixel_data = property(get_image, set_image)

    @property
    def has_parent_image(self):
        """True if this image has a defined parent"""
        return self.parent_image is not None

    @property
    def has_masking_objects(self):
        """True if the image was cropped with objects

        If this is true, there will also be a valid labels matrix
        available through the labels property
        """
        return self.masking_objects is not None

    @property
    def labels(self):
        """Get the segmentation labels from the masking objects

        returns the "segmented" labels: others are available through
        the masking_object.
        """
        if not self.has_masking_objects:
            return None

        return self.crop_image_similarly(self.masking_objects.segmented)

    @property
    def mask(self):
        """Return the mask (pixels to be considered) for the primary image
        """
        if self.__mask is not None:
            return self.__mask

        if self.has_masking_objects:
            return self.crop_image_similarly(self.crop_mask)

        if self.has_parent_image:
            mask = self.parent_image.mask

            return self.crop_image_similarly(mask)

        image = self.image

        #
        # Exclude channel, if present, from shape
        #
        shape = image.shape

        if self.multichannel:
            shape = shape[:-1]

        return numpy.ones(shape, dtype=bool)

    @mask.setter
    def mask(self, mask):
        """Set the mask (pixels to be considered) for the primary image

        Convert the input into a numpy array. If the input is numeric,
        we convert it to boolean by testing each element for non-zero.
        """
        m = numpy.array(mask)

        if not (m.dtype.type is bool):
            m = m != 0

        self.__mask = m
        self.__has_mask = True

    @property
    def has_mask(self):
        """True if the image has a mask"""
        if self.__has_mask:
            return True

        if self.has_crop_mask:
            return True

        if self.parent_image is not None:
            return self.parent_image.has_mask

        return False

    @property
    def crop_mask(self):
        """Return the mask used to crop this image"""
        if self.__crop_mask is not None:
            return self.__crop_mask

        if self.has_masking_objects:
            return self.masking_objects.segmented != 0

        if self.has_parent_image:
            return self.parent_image.crop_mask

        #
        # If no crop mask, return the mask which should be all ones
        #
        return self.mask

    @crop_mask.setter
    def crop_mask(self, crop_mask):
        self.__has_crop_mask = True
        self.__crop_mask = crop_mask

    @property
    def has_crop_mask(self):
        """True if the image or its ancestors has a crop mask"""
        return (
            self.__has_crop_mask
            or self.has_masking_objects
            or (self.has_parent_image and self.parent_image.has_crop_mask)
        )

    def crop_image_similarly(self, image):
        """Crop a 2-d or 3-d image using this image's crop mask

        image - a np.ndarray to be cropped (of any type)
        """
        if image.shape[:2] == self.pixel_data.shape[:2]:
            # Same size - no cropping needed
            return image
        if any(
            [
                my_size > other_size
                for my_size, other_size in zip(self.pixel_data.shape, image.shape)
            ]
        ):
            raise ValueError(
                "Image to be cropped is smaller: %s vs %s"
                % (repr(image.shape), repr(self.pixel_data.shape))
            )
        if not self.has_crop_mask:
            raise RuntimeError(
                "Images are of different size and no crop mask available.\n"
                "Use the Crop and Align modules to match images of different sizes."
            )
        cropped_image = crop_image(image, self.crop_mask)
        if cropped_image.shape[0:2] != self.pixel_data.shape[0:2]:
            raise ValueError(
                "Cropped image is not the same size as the reference image: %s vs %s"
                % (repr(cropped_image.shape), repr(self.pixel_data.shape))
            )
        return cropped_image

    @property
    def file_name(self):
        """The name of the file holding this image

        If the image is derived, then return the file name of the first
        ancestor that has a file name. Return None if the image does not have
        an ancestor or if no ancestor has a file name.
        """
        if self.__file_name is not None:
            return self.__file_name
        elif self.has_parent_image:
            return self.parent_image.file_name
        else:
            return None

    @property
    def path_name(self):
        """The path to the file holding this image

        If the image is derived, then return the path name of the first
        ancestor that has a path name. Return None if the image does not have
        an ancestor or if no ancestor has a file name.
        """
        if not self.__path_name is None:
            return self.__path_name
        elif self.has_parent_image:
            return self.parent_image.path_name
        else:
            return None

    @property
    def has_channel_names(self):
        """True if there are channel names on this image"""
        return self.channel_names is not None

    @property
    def scale(self):
        """The scale at acquisition

        This is the intensity scale used by the acquisition device. For
        instance, a microscope might use a 12-bit a/d converter to acquire
        an image and store that information using the TIF MaxSampleValue
        tag = 4095.
        """
        if self.__scale is None and self.has_parent_image:
            return self.parent_image.scale

        return self.__scale

    @property
    def channelstack(self):
        if self.__channelstack is not None:
            return self.__channelstack

    @channelstack.setter
    def channelstack(self, channelstack):
        self.__channelstack = channelstack
