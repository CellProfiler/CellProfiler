import numpy


class Image(object):
    """An image composed of a Numpy array plus secondary attributes such as mask and label matrices

    The secondary attributes:
    mask - a binary image indicating the points of interest in the image.
           The mask is the same size as the child image.
    crop_mask - the binary image used to crop the parent image to the
                dimensions of the child (this) image. The crop_mask is
                the same size as the parent image.
    parent_image - for derived images, the parent that was used to create
                   this image. This image may inherit attributes from
                   the parent image, such as the masks used to create the
                   parent
    masking_objects - the labels matrix from these objects is used to
                      mask and crop the parent image to make this image.
                      The labels are available as mask_labels and crop_labels.
    convert - true to try to coerce whatever dtype passed (other than bool
               or float) to a scaled image.
    path_name - the path name to the file holding the image or None
                for a derived image
    file_name - the file name of the file holding the image or None for a
                derived image
    scale - the scaling suggested by the initial image format (e.g. 4095 for
            a 12-bit a/d converter).

    Resolution of mask and cropping_mask properties:
    The Image class looks for the mask and cropping_mask in the following
    places:
    * self: if set using the properties or specified in the initializer
    * masking_objects: if set using the masking_object property or
                       specified in the initializer. The crop_mask and
                       mask are composed of all of the labeled points.
    * parent_image: if set using the initializer. The child image inherits
                    the mask and cropping mask of the parent.
    Otherwise, the image has no mask or cropping mask and all pixels are
    significant.
    """

    def __init__(self, data=None, mask=None, crop_mask=None, parent=None, masking_objects=None, convert=True, pathname=None, filename=None, scale=None):
        self.__image = data
        self.image = data
        self.pixel_data = data
        self.__mask = None
        self.__has_mask = False
        self.parent = parent
        self.__crop_mask = None
        if crop_mask is not None:
            self.set_crop_mask(crop_mask)
            self.__has_crop_mask = True
        else:
            self.__has_crop_mask = False
        self.__masking_objects = masking_objects
        self.masking_objects = masking_objects
        self.__scale = scale
        if mask is not None:
            self.set_mask(mask)
        # self.filename = filename
        self.pathname = pathname
        self.channel_names = None
        self.has_parent_image = self.parent is not None
        self.has_masking_objects = self.__masking_objects is not None
        self.labels = self.crop_image_similarly(self.masking_objects.segmented) if self.has_masking_objects else None
        self.has_channel_names = self.channel_names is not None
        self.scale = self.parent.scale if self.__scale is None and self.has_parent_image else self.__scale

    def grayscale(self):
        return self.pixel_data[:, :, 0]

    def rgb(self):
        return self.pixel_data[:, :, :3]

    def get_mask(self):
        """Return the mask (pixels to be considered) for the primary image
        """
        if not self.__mask is None:
            return self.__mask

        if self.has_masking_objects:
            return self.crop_image_similarly(self.crop_mask)

        if self.has_parent_image:
            mask = self.parent.mask

            return self.crop_image_similarly(mask)

        image = self.image

        #
        # Exclude channel, if present, from shape
        #
        if image.ndim == 2:
            shape = image.shape
        elif image.ndim == 3:
            shape = image.shape[:2]
        else:
            shape = image.shape[1:]

        return numpy.ones(shape, dtype=numpy.bool)

    def set_mask(self, mask):
        """Set the mask (pixels to be considered) for the primary image

        Convert the input into a numpy array. If the input is numeric,
        we convert it to boolean by testing each element for non-zero.
        """
        m = numpy.array(mask)

        if not (m.dtype.type is numpy.bool):
            m = (m != 0)

        check_consistency(self.image, m)

        self.__mask = m

        self.__has_mask = True

    mask = property(get_mask, set_mask)

    def get_has_mask(self):
        """True if the image has a mask"""
        if self.__has_mask:
            return True

        if self.has_crop_mask:
            return True

        if self.parent is not None:
            return self.parent.has_mask

        return False

    has_mask = property(get_has_mask)

    def get_crop_mask(self):
        """Return the mask used to crop this image"""
        if not self.__crop_mask is None:
            return self.__crop_mask

        if self.has_masking_objects:
            return self.masking_objects.segmented != 0

        if self.has_parent_image:
            return self.parent.crop_mask
        #
        # If no crop mask, return the mask which should be all ones
        #
        return self.mask

    def set_crop_mask(self, crop_mask):
        self.__crop_mask = crop_mask

    crop_mask = property(get_crop_mask, set_crop_mask)

    @property
    def has_crop_mask(self):
        '''True if the image or its ancestors has a crop mask'''
        return (self.__crop_mask is not None or self.has_masking_objects or (self.has_parent_image and self.parent.has_crop_mask))

    def crop_image_similarly(self, image):
        """Crop a 2-d or 3-d image using this image's crop mask

        image - a np.ndarray to be cropped (of any type)
        """
        if image.shape[:2] == self.pixel_data.shape[:2]:
            # Same size - no cropping needed
            return image

        if any([my_size > other_size for my_size, other_size in zip(self.pixel_data.shape, image.shape)]):
            raise ValueError("Image to be cropped is smaller: %s vs %s" % (repr(image.shape), repr(self.pixel_data.shape)))

        if not self.has_crop_mask:
            raise RuntimeError("Images are of different size and no crop mask available.\nUse the Crop and Align modules to match images of different sizes.")

        cropped_image = crop_image(image, self.crop_mask)

        if cropped_image.shape[0:2] != self.pixel_data.shape[0:2]:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s" % (repr(cropped_image.shape), repr(self.pixel_data.shape)))

        return cropped_image


# TODO: crop_image should be a method on Image
# TODO: implement crop by mask in skimage and use skimage version
def crop_image(image, crop_mask, crop_internal=False):
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum = numpy.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum = numpy.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return numpy.zeros((0, 0), dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = numpy.argwhere(i_histogram > 0)
        j_keep = numpy.argwhere(j_histogram > 0)
        #
        # Then slice the array by I, then by J to get what's not blank
        #
        return image[i_keep.flatten(), :][:, j_keep.flatten()].copy()
    else:
        #
        # The first non-blank row and column are where the cumsum is 1
        # The last are at the first where the cumsum is it's max (meaning
        # what came after was all zeros and added nothing)
        #
        i_first = numpy.argwhere(i_cumsum == 1)[0]
        i_last = numpy.argwhere(i_cumsum == i_cumsum.max())[0]
        i_end = i_last + 1
        j_first = numpy.argwhere(j_cumsum == 1)[0]
        j_last = numpy.argwhere(j_cumsum == j_cumsum.max())[0]
        j_end = j_last + 1
        if image.ndim == 3:
            return image[i_first:i_end, j_first:j_end, :].copy()
        return image[i_first:i_end, j_first:j_end].copy()


def check_consistency(image, mask):
    """Check that the image, mask and labels arrays have the same shape and that the arrays are of the right dtype"""
    assert (image is None) or (len(image.shape) in (2, 3)), "Image must have 2 or 3 dimensions"
    assert (mask is None) or (len(mask.shape) == 2), "Mask must have 2 dimensions"
    assert (image is None) or (mask is None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"
    assert (mask is None) or (mask.dtype.type is numpy.bool_), "Mask must be boolean, was %s" % (repr(mask.dtype.type))


class AbstractImageProvider(object):
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
        pass

    name = property(__get_name)


class VanillaImageProvider(AbstractImageProvider):
    """This image provider returns the image given to it in the constructor

    """

    def __init__(self, name, image):
        """Constructor takes the name of the image and the CellProfiler.Image.Image instance to be returned
        """
        self.__name = name
        self.__image = image

    def provide_image(self, image_set):
        return self.__image

    def get_name(self):
        return self.__name

    def release_memory(self):
        pass


class ImageSet(object):
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
        self.__keys = keys
        self.keys = self.__keys
        self.__number = number
        self.number = self.__number
        self.image_number = self.__number + 1
        self.__legacy_fields = legacy_fields

    def get_image(self, name, must_be_binary=False, must_be_color=False, must_be_grayscale=False, must_be_rgb=False, cache=True):
        """Return the image associated with the given name

        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        must_be_rgb - raise an exception if 2-d or if # channels not 3 or 4,
                      discard alpha channel.
        """
        name = str(name)

        if not self.__images.has_key(name):
            image = self.get_image_provider(name).provide_image(self)

            if cache:
                self.__images[name] = image
        else:
            image = self.__images[name]

        if must_be_binary and image.pixel_data.ndim == 3:
            raise ValueError("Image must be binary, but it was color")

        if must_be_binary and image.pixel_data.dtype != numpy.bool:
            raise ValueError("Image was not binary")

        if must_be_color and image.pixel_data.ndim != 3:
            raise ValueError("Image must be color, but it was grayscale")

        if (must_be_grayscale and (image.pixel_data.ndim != 2)):
            pd = image.pixel_data

            if pd.shape[2] >= 3 and numpy.all(pd[:, :, 0] == pd[:, :, 1]) and numpy.all(pd[:, :, 0] == pd[:, :, 2]):
                return image.grayscale()

            raise ValueError("Image must be grayscale, but it was color")

        if must_be_grayscale and image.pixel_data.dtype.kind == 'b':
            return image.grayscale()

        if must_be_rgb:
            if image.pixel_data.ndim != 3:
                raise ValueError("Image must be RGB, but it was grayscale")
            elif image.pixel_data.shape[2] not in (3, 4):
                raise ValueError("Image must be RGB, but it had %d channels" % image.pixel_data.shape[2])
            elif image.pixel_data.shape[2] == 4:
                return image.rgb()

        return image

    @property
    def providers(self):
        """The list of providers (populated during the image discovery phase)"""
        return self.__image_providers

    def get_image_provider(self, name):
        """Get a named image provider

        name - return the image provider with this name
        """
        providers = filter(lambda x: x.name == name, self.__image_providers)

        assert len(providers) > 0, "No provider of the %s image" % name

        assert len(providers) == 1, "More than one provider of the %s image" % name

        return providers[0]

    def get_names(self):
        """Get the image provider names
        """
        return [provider.name for provider in self.providers]

    names = property(get_names)

    def get_legacy_fields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.

        """
        return self.__legacy_fields

    legacy_fields = property(get_legacy_fields)

    def add(self, name, image):
        old_providers = [provider for provider in self.providers if provider.name == name]

        for provider in old_providers:
            self.providers.remove(provider)

        provider = VanillaImageProvider(name, image)

        self.providers.append(provider)


class ImageSetList(object):
    """Represents the list of image sets in a pipeline run

    """

    def __init__(self, test_mode=False):
        self.__image_sets = []
        self.__image_sets_by_key = {}
        self.__legacy_fields = {}
        self.__associating_by_key = None
        self.test_mode = test_mode
        self.combine_path_and_file = False

    def get_image_set(self, keys_or_number):
        """Return either the indexed image set (keys_or_number = index) or the image set with matching keys

        """
        if not isinstance(keys_or_number, dict):
            keys = {'number': keys_or_number}
            number = keys_or_number
            if self.__associating_by_key is None:
                self.__associating_by_key = False
            k = make_dictionary_key(keys)
        else:
            keys = keys_or_number

            k = make_dictionary_key(keys)

            if self.__image_sets_by_key.has_key(k):
                number = self.__image_sets_by_key[k].get_number()
            else:
                number = len(self.__image_sets)

            self.__associating_by_key = True

        if number >= len(self.__image_sets):
            self.__image_sets += [None] * (number - len(self.__image_sets) + 1)

        if self.__image_sets[number] is None:
            image_set = ImageSet(number, keys, self.__legacy_fields)
            self.__image_sets[number] = image_set
            self.__image_sets_by_key[k] = image_set
            if self.associating_by_key:
                k = make_dictionary_key(dict(number=number))
                self.__image_sets_by_key[k] = image_set
        else:
            image_set = self.__image_sets[number]
        return image_set

    @property
    def associating_by_key(self):
        '''True if some image set has been added with a key instead of a number

        This will return "None" if no association has been done.
        '''
        return self.__associating_by_key

    def count(self):
        return len(self.__image_sets)

    def get_legacy_fields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.

        """
        return self.__legacy_fields

    legacy_fields = property(get_legacy_fields)

    def get_groupings(self, keys):
        """Return the groupings of an image set list over a set of keys

        keys - a sequence of keys that match some of the image set keys

        returns an object suitable for use by CPModule.get_groupings:
        tuple of keys, groupings
        keys - the keys as passed into the function
        groupings - a sequence of groupings of image sets where
                    each element of the sequence is a two-tuple.
                    The first element of the two-tuple is a dictionary
                    that gives the group's values for each key.
                    The second element is a list of image numbers of
                    the images in the group
        """
        #
        # Sort order for dictionary keys
        #
        sort_order = []

        #
        # Dictionary of key_values to list of image numbers
        #
        d = {}

        for i in range(self.count()):
            image_set = self.get_image_set(i)

            assert isinstance(image_set, ImageSet)

            key_values = tuple([str(image_set.keys[key]) for key in keys])

            if not d.has_key(key_values):
                d[key_values] = []
                sort_order.append(key_values)

            d[key_values].append(i + 1)

        return keys, [(dict(zip(keys, k)), d[k]) for k in sort_order]


def make_dictionary_key(key):
    '''Make a dictionary into a stable key for another dictionary'''
    return u", ".join([u":".join([unicode(y) for y in x]) for x in sorted(key.iteritems())])

