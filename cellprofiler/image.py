import math
import sys
import StringIO
import cPickle
import struct
import zlib
import numpy


class Abstract(object):
    """Represents an image provider that returns images
    """

    def provide_image(self, image_set):
        """Return the image that is associated with the image set
        """
        raise NotImplementedError("Please implement ProvideImage for your class")

    @property
    def name(self):
        """Call the abstract function, "get_name"
        """
        return self.get_name()

    def get_name(self):
        """The user-visible name for the image
        """
        raise NotImplementedError("Please implement get_name for your class")

    def release_memory(self):
        '''Release whatever memory is associated with the image'''
        pass


class Callback(Abstract):
    def __init__(self, name, image_provider_fn):
        self.__name = name
        self.__image_provider_fn = image_provider_fn

    def provide_image(self, image_set):
        return self.__image_provider_fn(image_set, self)

    def get_name(self):
        return self.__name


class Vanilla(Abstract):
    def __init__(self, name, image):
        self.__name = name
        self.__image = image

    def provide_image(self, image_set):
        return self.__image

    def get_name(self):
        return self.__name

    def release_memory(self):
        self.__image = None


class Cache(object):
    IC_MONOCHROME = "Monochrome"
    IC_COLOR = "Color"
    IC_5D = "5D"

    def __init__(self, image):
        '''Initialize with the image to control'''
        self.__backing_store = None
        self.__name = None
        if image.ndim == 2:
            self.__type = Cache.IC_MONOCHROME
            self.__image = image.reshape(1, 1, 1, image.shape[0], image.shape[1])
        elif image.ndim == 3:
            self.__type = Cache.IC_COLOR
            self.__image = image.transpose(2, 0, 1).reshape(
                    image.shape[2], 1, 1, image.shape[0], image.shape[1])
        else:
            self.__type = Cache.IC_5D
            self.__image = image

    def is_cached(self):
        '''Return True if image is already cached by a backing store

        '''
        return self.__backing_store is not None

    def cache(self, name, backing_store):
        '''Cache an image into a backing store

        name - unique channel name of the image
        backing_store - an HDF5ImageSet
        '''
        self.__backing_store = backing_store
        self.__name = name
        self.__backing_store.set_image(self.__name, self.__image)
        del self.__image

    def get(self):
        '''Get the image in its original format'''
        if self.is_cached():
            image = self.__backing_store.get_image(self.__name)
        else:
            image = self.__image

        if self.__type == Cache.IC_MONOCHROME:
            return image.reshape(image.shape[3], image.shape[4])
        elif self.__type == Cache.IC_COLOR:
            return image.reshape(image.shape[0], image.shape[3], image.shape[4]).transpose(1, 2, 0)


class Grayscale(object):
    def __init__(self, image):
        self.image = image

    def __getattr__(self, name):
        return getattr(self.image, name)

    @property
    def data(self):
        return self.image.data[:, :, 0]


class Image(object):
    def __init__(self, image=None, mask=None, crop_mask=None, parent=None, masking_objects=None, convert=True, path=None, filename=None, scale=None):
        self.__image__ = None
        self.__mask__ = None
        self.__masked__ = False
        self.__parent__ = parent
        self.__crop_mask__ = None

        if crop_mask is not None:
            self.crop_mask = crop_mask

            self.__has_crop_mask = True
        else:
            self.__has_crop_mask = False

        self.__masking_objects__ = masking_objects
        self.__scale__ = scale

        if image is not None:
            self.set_image(image, convert)

        if mask is not None:
            self.mask = mask

        self.__filename__ = filename
        self.__pathname__ = path
        self.__channel_names = None

    def get_image(self):
        if self.__image__ is None:
            return

        return self.__image__.get()

    def set_image(self, image, convert=True):
        img = numpy.asanyarray(image)

        if img.dtype.name == "bool" or not convert:
            self.__image__ = Cache(img)
            return

        mval = 0.

        scale = 1.

        fix_range = False

        if img.dtype.type is numpy.uint8:
            scale = math.pow(2.0, 8.0) - 1
        elif img.dtype.type is numpy.uint16:
            scale = math.pow(2.0, 16.0) - 1
        elif img.dtype.type is numpy.uint32:
            scale = math.pow(2.0, 32.0) - 1
        elif img.dtype.type is numpy.uint64:
            scale = math.pow(2.0, 64.0) - 1
        elif img.dtype.type is numpy.int8:
            scale = math.pow(2.0, 8.0)

            mval = -scale / 2.0

            scale -= 1

            fix_range = True
        elif img.dtype.type is numpy.int16:
            scale = math.pow(2.0, 16.0)

            mval = -scale / 2.0

            scale -= 1

            fix_range = True
        elif img.dtype.type is numpy.int32:
            scale = math.pow(2.0, 32.0)

            mval = -scale / 2.0

            scale -= 1

            fix_range = True
        elif img.dtype.type is numpy.int64:
            scale = math.pow(2.0, 64.0)

            mval = -scale / 2.0

            scale -= 1

            fix_range = True

        img = img.astype(numpy.float32)

        img -= mval

        img /= scale

        if fix_range:
            numpy.clip(img, 0, 1, out=img)

        check_consistency(img, self.__mask__)

        self.__image__ = Cache(img)

    image = property(get_image, set_image)

    data = property(get_image, set_image)

    @property
    def parent(self):
        return self.__parent__

    @parent.setter
    def parent(self, image):
        self.__parent__ = image

    @property
    def has_parent_image(self):
        return self.__parent__ is not None

    @property
    def masking_objects(self):
        return self.__masking_objects__

    @masking_objects.setter
    def masking_objects(self, value):
        self.__masking_objects__ = value

    @property
    def has_masking_objects(self):
        return self.__masking_objects__ is not None

    @property
    def labels(self):
        if not self.has_masking_objects:
            return None

        return self.crop_image_similarly(self.masking_objects.segmented)

    @property
    def mask(self):
        if self.__mask__ is not None:
            return self.__mask__.get()

        if self.has_masking_objects:
            return self.crop_image_similarly(self.crop_mask)

        if self.has_parent_image:
            mask = self.parent.mask

            return self.crop_image_similarly(mask)

        image = self.image

        if image.ndim == 2:
            shape = image.shape
        elif image.ndim == 3:
            shape = image.shape[:2]
        else:
            shape = image.shape[1:]

        return numpy.ones(shape, dtype=numpy.bool)

    @mask.setter
    def mask(self, mask):
        m = numpy.array(mask)

        if not (m.dtype.type is numpy.bool):
            m = (m != 0)

        check_consistency(self.image, m)

        self.__mask__ = Cache(m)

        self.__masked__ = True

    @property
    def masked(self):
        if self.__masked__:
            return True

        if self.has_crop_mask:
            return True

        if self.parent is not None:
            return self.parent.masked

        return False

    @property
    def crop_mask(self):
        if not self.__crop_mask__ is None:
            return self.__crop_mask__.get()

        if self.has_masking_objects:
            return self.masking_objects.segmented != 0

        if self.has_parent_image:
            return self.parent.crop_mask

        return self.mask

    @crop_mask.setter
    def crop_mask(self, crop_mask):
        self.__crop_mask__ = Cache(crop_mask)

    @property
    def has_crop_mask(self):
        return self.__crop_mask__ is not None or self.has_masking_objects or (self.has_parent_image and self.parent.has_crop_mask)

    @property
    def filename(self):
        if self.__filename__ is not None:
            return self.__filename__
        elif self.has_parent_image:
            return self.parent.filename
        else:
            return None

    @property
    def path(self):
        if self.__pathname__ is not None:
            return self.__pathname__
        elif self.has_parent_image:
            return self.parent.path
        else:
            return None

    @property
    def channel_names(self):
        return self.__channel_names

    @channel_names.setter
    def channel_names(self, names):
        self.__channel_names = tuple(names)

    @property
    def has_channel_names(self):
        return self.__channel_names is not None

    @property
    def scale(self):
        if self.__scale__ is None and self.has_parent_image:
            return self.parent.scale

        return self.__scale__

    def cache(self, name, hdf5_file):
        from cellprofiler.utilities.hdf5_dict import HDF5ImageSet

        if isinstance(self.__image__, Cache) and not self.__image__.is_cached():
            self.__image__.cache(name, HDF5ImageSet(hdf5_file))

        if isinstance(self.__mask__, Cache) and not self.__mask__.is_cached():
            self.__mask__.cache(name, HDF5ImageSet(hdf5_file, "Masks"))

        if isinstance(self.__crop_mask__, Cache) and not self.__crop_mask__.is_cached():
            self.__crop_mask__.cache(name, HDF5ImageSet(hdf5_file, "CropMasks"))

    def crop_image_similarly(self, image):
        if image.shape[:2] == self.data.shape[:2]:
            return image

        if any([my_size > other_size for my_size, other_size in zip(self.data.shape, image.shape)]):
            raise ValueError("Image to be cropped is smaller: %s vs %s" % (repr(image.shape), repr(self.data.shape)))

        if not self.has_crop_mask:
            raise RuntimeError("Images are of different size and no crop mask available.\nUse the Crop and Align modules to match images of different sizes.")

        cropped_image = crop_image(image, self.crop_mask)

        if cropped_image.shape[0:2] != self.data.shape[0:2]:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s" % (repr(cropped_image.shape), repr(self.data.shape)))

        return cropped_image


class List(object):
    def __init__(self, test_mode=False):
        self.__image_sets = []
        self.__image_sets_by_key = {}
        self.__legacy_fields = {}
        self.__associating_by_key = None
        self.__test_mode = test_mode
        self.combine_path_and_file = False

    @property
    def test_mode(self):
        '''True if we are in test mode'''
        return self.__test_mode

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
                number = self.__image_sets_by_key[k].number
            else:
                number = len(self.__image_sets)
            self.__associating_by_key = True
        if number >= len(self.__image_sets):
            self.__image_sets += [None] * (number - len(self.__image_sets) + 1)
        if self.__image_sets[number] is None:
            image_set = Set(number, keys, self.__legacy_fields)
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

    def purge_image_set(self, number):
        """Remove the memory associated with an image set"""
        keys = self.__image_sets[number].keys
        image_set = self.__image_sets[number]
        image_set.clear_cache()
        for provider in image_set.providers:
            provider.release_memory()
        self.__image_sets[number] = None
        self.__image_sets_by_key[repr(keys)] = None

    def add_provider_to_all_image_sets(self, provider):
        """Provide an image to every image set

        provider - an instance of AbstractImageProvider
        """
        for image_set in self.__image_sets:
            image_set.providers.append(provider)

    def count(self):
        return len(self.__image_sets)

    def get_legacy_fields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.

        """
        return self.__legacy_fields

    legacy_fields = property(get_legacy_fields)

    def get_groupings(self, keys):
        '''Return the groupings of an image set list over a set of keys

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
        '''
        #
        # Sort order for dictionary keys
        #
        sort_order = []
        dictionaries = []
        #
        # Dictionary of key_values to list of image numbers
        #
        d = {}
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, Set)
            key_values = tuple([str(image_set.keys[key]) for key in keys])
            if not d.has_key(key_values):
                d[key_values] = []
                sort_order.append(key_values)
            d[key_values].append(i + 1)
        return keys, [(dict(zip(keys, k)), d[k]) for k in sort_order]

    def save_state(self):
        '''Return a string that can be used to load the image_set_list's state

        load_state will restore the image set list's state. No image_set can
        have image providers before this call.
        '''
        f = StringIO.StringIO()
        cPickle.dump(self.count(), f)
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, Set)
            assert len(image_set.providers) == 0, "An image set cannot have providers while saving its state"
            cPickle.dump(image_set.keys, f)
        cPickle.dump(self.legacy_fields, f)
        return f.getvalue()

    def load_state(self, state):
        '''Load an image_set_list's state from the string returned from save_state'''

        self.__image_sets = []
        self.__image_sets_by_key = {}

        # Make a safe unpickler
        p = cPickle.Unpickler(StringIO.StringIO(state))

        def find_global(module_name, class_name):
            if module_name not in ("numpy", "numpy.core.multiarray"):
                raise ValueError("Illegal attempt to unpickle class %s.%s",
                                 (module_name, class_name))
            __import__(module_name)
            mod = sys.modules[module_name]
            return getattr(mod, class_name)

        p.find_global = find_global

        count = p.load()
        all_keys = [p.load() for i in range(count)]
        self.__legacy_fields = p.load()
        #
        # Have to do in this order in order for the image set's
        # legacy_fields property to hook to the right legacy_fields
        #
        for i in range(count):
            self.get_image_set(all_keys[i])


class RGB(object):
    def __init__(self, image):
        self.image = image

    def __getattr__(self, name):
        return getattr(self.image, name)

    @property
    def data(self):
        return self.image.data[:, :, :3]


class Set(object):
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
        self.__number = number
        self.__legacy_fields = legacy_fields

    @property
    def number(self):
        """The (zero-based) image set index
        """
        return self.__number

    @property
    def image_number(self):
        '''The image number as used in measurements and the database'''
        return self.__number + 1

    @property
    def keys(self):
        """The keys that uniquely identify the image set
        """
        return self.__keys

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

        if must_be_binary and image.data.ndim == 3:
            raise ValueError("Image must be binary, but it was color")

        if must_be_binary and image.data.dtype != numpy.bool:
            raise ValueError("Image was not binary")

        if must_be_color and image.data.ndim != 3:
            raise ValueError("Image must be color, but it was grayscale")

        if must_be_grayscale and (image.data.ndim != 2):
            data = image.data

            if data.shape[2] >= 3 and numpy.all(data[:, :, 0] == data[:, :, 1]) and numpy.all(data[:, :, 0] == data[:, :, 2]):
                return Grayscale(image)

            raise ValueError("Image must be grayscale, but it was color")

        if must_be_grayscale and image.data.dtype.kind == "b":
            return Grayscale(image)

        if must_be_rgb:
            if image.data.ndim != 3:
                raise ValueError("Image must be RGB, but it was grayscale")
            elif image.data.shape[2] not in (3, 4):
                raise ValueError("Image must be RGB, but it had %d channels" % image.data.shape[2])
            elif image.data.shape[2] == 4:
                return RGB(image)

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

    def remove_image_provider(self, name):
        """Remove a named image provider

        name - the name of the provider to remove
        """
        self.__image_providers = filter(lambda x: x.name != name,
                                        self.__image_providers)

    def clear_image(self, name):
        '''Remove the image memory associated with a provider

        name - the name of the provider
        '''
        self.get_image_provider(name).release_memory()
        if self.__images.has_key(name):
            del self.__images[name]

    def clear_cache(self):
        '''Remove all of the cached images'''
        self.__images.clear()

    @property
    def names(self):
        return [provider.name for provider in self.providers]

    @property
    def legacy_fields(self):
        return self.__legacy_fields

    def add(self, name, image):
        old_providers = [provider for provider in self.providers if provider.name == name]

        if len(old_providers) > 0:
            self.clear_image(name)

        for provider in old_providers:
            self.providers.remove(provider)

        provider = Vanilla(name, image)

        self.providers.append(provider)


def check_consistency(image, mask):
    """Check that the image, mask and labels arrays have the same shape and that the arrays are of the right dtype"""
    assert (image is None) or (len(image.shape) in (2, 3)), "Image must have 2 or 3 dimensions"
    assert (mask is None) or (len(mask.shape) == 2), "Mask must have 2 dimensions"
    assert (image is None) or (mask is None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"
    assert (mask is None) or (mask.dtype.type is numpy.bool_), "Mask must be boolean, was %s" % (repr(mask.dtype.type))


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


def make_dictionary_key(key):
    '''Make a dictionary into a stable key for another dictionary'''
    return u", ".join([u":".join([unicode(y) for y in x])
                       for x in sorted(key.iteritems())])


def readc01(fname):
    '''Read a Cellomics file into an array

    fname - the name of the file
    '''

    def readint(f):
        return struct.unpack("<l", f.read(4))[0]

    def readshort(f):
        return struct.unpack("<h", f.read(2))[0]

    f = open(fname, "rb")

    # verify it's a c01 format, and skip the first four bytes
    assert readint(f) == 16 << 24

    # decompress
    g = StringIO.StringIO(zlib.decompress(f.read()))

    # skip four bytes
    g.seek(4, 1)

    x = readint(g)
    y = readint(g)

    nplanes = readshort(g)
    nbits = readshort(g)

    compression = readint(g)
    assert compression == 0, "can't read compressed pixel data"

    # skip 4 bytes
    g.seek(4, 1)

    pixelwidth = readint(g)
    pixelheight = readint(g)
    colors = readint(g)
    colors_important = readint(g)

    # skip 12 bytes
    g.seek(12, 1)

    data = numpy.fromstring(g.read(), numpy.uint16 if nbits == 16 else numpy.uint8, x * y)
    return data.reshape(x, y).T
