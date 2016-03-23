import math
import numpy
import pickle
import StringIO
import sys


def crop_image(image, crop_mask, crop_internal=False):
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


class Abstract:
    def __init__(self):
        pass

    def free(self):
        pass

    def source(self, images):
        pass


class Cache:
    def __init__(self, image):
        self.__store__ = None

        self.__name = None

        if image.ndim == 2:
            reshaped = image.reshape(1, 1, 1, image.shape[0], image.shape[1])

            self.__image__ = reshaped

            self.__type__ = "Monochrome"
        elif image.ndim == 3:
            reshaped = image.transpose(2, 0, 1).reshape(image.shape[2], 1, 1, image.shape[0], image.shape[1])

            self.__image__ = reshaped

            self.__type__ = "Color"
        else:
            self.__image__ = image

            self.__type__ = "5D"

    def cache(self, name, store):
        self.__store__ = store

        self.__name = name

        self.__store__.image(self.__name, self.__image__)

        del self.__image__

    def cached(self):
        return self.__store__ is not None

    def get(self):
        if self.cached():
            image = self.__store__.image(self.__name)
        else:
            image = self.__image__

        if self.__type__ == "Monochrome":
            return image.reshape(image.shape[3], image.shape[4])
        elif self.__type__ == "Color":
            return image.reshape(image.shape[0], image.shape[3], image.shape[4]).transpose(1, 2, 0)


class Callback(Abstract):
    def __init__(self, name, function):
        Abstract.__init__(self)

        self.__function__ = function

        self.name = name

    def free(self):
        pass

    def source(self, images):
        return self.__function__(images, self)


class Grayscale:
    def __getattr__(self, name):
        return getattr(self.__image__, name)

    def __init__(self, image):
        self.__image__ = image

    @property
    def data(self):
        if self.__image__.data.dtype.kind == "b":
            return self.__image__.data.astype(numpy.float64)

        return self.__image__.data[:, :, 0]


class Image:
    def __init__(self, image=None, mask=None, crop_mask=None, parent_image=None, masking_objects=None, convert=True, pathname=None, filename=None, scale=None):
        self.__crop_mask__ = None
        self.__filename__ = filename
        self.__has_mask = False
        self.__image__ = None
        self.__mask__ = None
        self.__pathname__ = pathname
        self.__scale__ = scale

        self.parent_image = parent_image

        if crop_mask is not None:
            self.crop_mask = crop_mask

            self.__has_crop_mask__ = True
        else:
            self.__has_crop_mask__ = False

        self.masking_objects = masking_objects

        if image is not None:
            self.image(image, convert)

        if mask is not None:
            self.mask = mask

        self.__channel_names__ = None

    @property
    def image(self):
        if self.__image__ is None:
            return

        return self.__image__.get()

    @image.setter
    def image(self, image, convert=True):
        img = numpy.asanyarray(image)

        if img.dtype.name == "bool" or not convert:
            self.__image__ = Cache(img)

            return

        mval = 0.

        scale = 1.

        fix_range = False

        if issubclass(img.dtype.type, numpy.float):
            pass
        elif img.dtype.type is numpy.uint8:
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

        self.check_consistency(img, self.__mask__)

        self.__image__ = Cache(img)

    data = property(image.getter, image.setter)

    @property
    def has_parent_image(self):
        return self.parent_image is not None

    @property
    def has_masking_objects(self):
        return self.masking_objects is not None

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
            mask = self.parent_image.mask

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

        self.check_consistency(self.image, m)

        self.__mask__ = Cache(m)

        self.__has_mask = True

    @property
    def has_mask(self):
        if self.__has_mask:
            return True

        if self.has_crop_mask:
            return True

        if self.parent_image is not None:
            return self.parent_image.has_mask

        return False

    @property
    def crop_mask(self):
        if self.__crop_mask__ is not None:
            return self.__crop_mask__.get()

        if self.has_masking_objects:
            return self.masking_objects.segmented != 0

        if self.has_parent_image:
            return self.parent_image.crop_mask

        return self.mask

    @crop_mask.setter
    def crop_mask(self, crop_mask):
        self.__crop_mask__ = Cache(crop_mask)

    @property
    def has_crop_mask(self):
        return self.__crop_mask__ is not None or self.has_masking_objects or (self.has_parent_image and self.parent_image.has_crop_mask)

    @property
    def filename(self):
        if self.__filename__ is not None:
            return self.__filename__
        elif self.has_parent_image:
            return self.parent_image.file_name
        else:
            return None

    @property
    def pathname(self):
        if self.__pathname__ is not None:
            return self.__pathname__
        elif self.has_parent_image:
            return self.parent_image.path_name
        else:
            return None

    @property
    def channel_names(self):
        return self.__channel_names__

    @channel_names.setter
    def channel_names(self, names):
        self.__channel_names__ = tuple(names)

    @property
    def has_channel_names(self):
        return self.__channel_names__ is not None

    @property
    def scale(self):
        if self.__scale__ is None and self.has_parent_image:
            return self.parent_image.scale

        return self.__scale__

    def cache(self, name, hdf5_file):
        from cellprofiler.utilities.hdf5_dict import HDF5ImageSet

        if isinstance(self.__image__, Cache) and not self.__image__.cached():
            self.__image__.cache(name, HDF5ImageSet(hdf5_file))

        if isinstance(self.__mask__, Cache) and not self.__mask__.cached():
            self.__mask__.cache(name, HDF5ImageSet(hdf5_file, "Masks"))

        if isinstance(self.__crop_mask__, Cache) and not self.__crop_mask__.cached():
            self.__crop_mask__.cache(name, HDF5ImageSet(hdf5_file, "CropMasks"))

    def crop_image_similarly(self, image):
        if image.shape[:2] == self.data.shape[:2]:
            return image

        if any([my_size > other_size for my_size, other_size in zip(self.data.shape, image.shape)]):
            raise ValueError("Image to be cropped is smaller: {0:s} vs {1:s}".format(repr(image.shape), repr(self.data.shape)))

        if not self.has_crop_mask:
            raise RuntimeError("Images are of different size and no crop mask available.\n" "Use the Crop and Align modules to match images of different sizes.")

        cropped_image = crop_image(image, self.crop_mask)

        if cropped_image.shape[0:2] != self.data.shape[0:2]:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s" % (repr(cropped_image.shape), repr(self.data.shape)))

        return cropped_image

    @staticmethod
    def check_consistency(image, mask):
        assert (image is None) or (len(image.shape) in (2, 3)), "Image must have 2 or 3 dimensions"

        assert (mask is None) or (len(mask.shape) == 2), "Mask must have 2 dimensions"

        assert (image is None) or (mask is None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"

        assert (mask is None) or (mask.dtype.type is numpy.bool_), "Mask must be boolean, was {0:s}".format(repr(mask.dtype.type))


class List:
    def __init__(self, test_mode=False):
        self.__image_sets__ = []

        self.__image_sets_by_key__ = {}

        self.legacy_fields = {}

        self.associating_by_key = None

        self.test_mode = test_mode

        self.combine_path_and_file = False

    def get_image_set(self, keys_or_number):
        def make_dictionary_key(key):
            return u", ".join([u":".join([unicode(y) for y in x]) for x in sorted(key.iteritems())])

        if not isinstance(keys_or_number, dict):
            keys = {'number': keys_or_number}

            number = keys_or_number

            if self.__associating_by_key is None:
                self.__associating_by_key = False

            k = make_dictionary_key(keys)
        else:
            keys = keys_or_number

            k = make_dictionary_key(keys)

            if self.__image_sets_by_key__.has_key(k):
                number = self.__image_sets_by_key__[k].get_number()
            else:
                number = len(self.__image_sets__)

            self.__associating_by_key = True

        if number >= len(self.__image_sets__):
            self.__image_sets__ += [None] * (number - len(self.__image_sets__) + 1)

        if self.__image_sets__[number] is None:
            image_set = Set(number, keys, self.__legacy_fields__)

            self.__image_sets__[number] = image_set

            self.__image_sets_by_key__[k] = image_set

            if self.associating_by_key:
                k = make_dictionary_key(dict(number=number))

                self.__image_sets_by_key__[k] = image_set
        else:
            image_set = self.__image_sets__[number]

        return image_set

    def purge_image_set(self, number):
        keys = self.__image_sets__[number].keys
        image_set = self.__image_sets__[number]
        image_set.clear_cache()
        for provider in image_set.providers:
            provider.free()
        self.__image_sets__[number] = None
        self.__image_sets_by_key__[repr(keys)] = None

    def add_provider_to_all_image_sets(self, provider):
        for image_set in self.__image_sets__:
            image_set.providers.append(provider)

    def count(self):
        return len(self.__image_sets__)

    def get_groupings(self, keys):
        sort_order = []
        dictionaries = []
        dictionary = {}
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, Set)
            key_values = tuple([str(image_set.keys[key]) for key in keys])
            if key_values not in dictionary:
                dictionary[key_values] = []
                sort_order.append(key_values)
            dictionary[key_values].append(i + 1)
        return keys, [(dict(zip(keys, k)), dictionary[k]) for k in sort_order]

    def write(self):
        f = StringIO.StringIO()
        pickle.dump(self.count(), f)
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, Set)
            assert len(image_set.providers) == 0, "An image set cannot have providers while saving its state"
            pickle.dump(image_set.keys, f)
        pickle.dump(self.legacy_fields, f)
        return f.getvalue()

    def read(self, state):
        self.__image_sets__ = []
        self.__image_sets_by_key__ = {}
        p = pickle.Unpickler(StringIO.StringIO(state))

        def find_global(module_name, class_name):
            if module_name not in ("numpy", "numpy.core.multiarray"):
                message = "Illegal attempt to unpickle class %s.%s"
                raise ValueError(message, (module_name, class_name))
            __import__(module_name)
            mod = sys.modules[module_name]
            return getattr(mod, class_name)

        p.find_global = find_global
        count = p.load()
        all_keys = [p.load() for i in range(count)]
        self.__legacy_fields__ = p.load()
        for i in range(count):
            self.get_image_set(all_keys[i])


class RGB:
    def __getattr__(self, name):
        return getattr(self.__image__, name)

    def __init__(self, image):
        self.__image__ = image

    def data(self):
        return self.__image__.data[:, :, :3]


class Set:
    def __init__(self, number, keys, legacy_fields):
        self.__image_providers__ = []

        self.__images__ = {}

        self.keys = keys

        self.legacy_fields = legacy_fields

        self.number = number

        self.providers = self.__image_providers__

    @property
    def image_number(self):
        return self.number + 1

    @property
    def names(self):
        return [provider.name for provider in self.providers]

    def find_image_by(self, name, must_be_binary=False, must_be_color=False, must_be_grayscale=False, must_be_rgb=False, cache=True):
        name = str(name)

        if name not in self.__images__:
            image = self.find_source_by(name).source(self)

            if cache:
                self.__images__[name] = image
        else:
            image = self.__images__[name]

        if must_be_binary and image.data.ndim == 3:
            message = "Image must be binary, but it was color"

            raise ValueError(message)

        if must_be_binary and image.data.dtype != numpy.bool:
            message = "Image was not binary"

            raise ValueError(message)

        if must_be_color and image.data.ndim != 3:
            message = "Image must be color, but it was grayscale"

            raise ValueError(message)

        if must_be_grayscale and (image.data.ndim != 2):
            pd = image.data

            if pd.shape[2] >= 3 and numpy.all(pd[:, :, 0] == pd[:, :, 1]) and numpy.all(pd[:, :, 0] == pd[:, :, 2]):
                return Grayscale(image)

            message = "Image must be grayscale, but it was color"

            raise ValueError(message)

        if must_be_grayscale and image.data.dtype.kind == 'b':
            return Grayscale(image)

        if must_be_rgb:
            if image.data.ndim != 3:
                message = "Image must be RGB, but it was grayscale"

                raise ValueError(message)
            elif image.data.shape[2] not in (3, 4):
                message = u"Image must be RGB, but it had {0:d} channels".format(image.data.shape[2])

                raise ValueError(message)
            elif image.data.shape[2] == 4:
                return RGB(image)

        return image

    def find_source_by(self, name):
        sources = filter(lambda x: x.name == name, self.__image_providers__)

        assert len(sources) > 0, u"No provider of the {0:s} image".format(name)

        assert len(sources) == 1, u"More than one provider of the {0:s} image".format(name)

        return sources[0]

    def add(self, name, image):
        old_providers = [provider for provider in self.providers if provider.name == name]

        if len(old_providers) > 0:
            self.clear_image(name)

        for provider in old_providers:
            self.providers.remove(provider)

        provider = Vanilla(name, image)

        self.providers.append(provider)

    def clear_cache(self):
        self.__images__.clear()

    def clear_image(self, name):
        self.find_source_by(name).free()

        if name in self.__images__:
            del self.__images__[name]

    def remove_source(self, name):
        self.__image_providers__ = filter(lambda x: x.name != name, self.__image_providers__)


class Vanilla(Abstract):
    def __init__(self, name, image):
        Abstract.__init__(self)

        self.__image__ = image

        self.name = name

    def free(self):
        self.__image__ = None

    def source(self, images):
        return self.__image__
