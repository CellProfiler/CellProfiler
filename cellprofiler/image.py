import hashlib
import os
import tempfile
import urllib
import urlparse

import cellprofiler.measurement
import cellprofiler.preferences
import cellprofiler.utilities
import numpy
import scipy.io

'''Tag for loading images as images'''
IO_IMAGES = "Images"
'''Tag for loading images as segmentation results'''
IO_OBJECTS = "Objects"
IO_ALL = (IO_IMAGES, IO_OBJECTS)

SUPPORTED_IMAGE_EXTENSIONS = {'.ppm', '.grib', '.im', '.rgba', '.rgb', '.pcd', '.h5', '.jpe', '.jfif', '.jpg', '.fli',
                              '.sgi', '.gbr', '.pcx', '.mpeg', '.jpeg', '.ps', '.flc', '.tif', '.hdf', '.icns', '.gif',
                              '.palm', '.mpg', '.fits', '.pgm', '.mic', '.fit', '.xbm', '.eps', '.emf', '.dcx', '.bmp',
                              '.bw', '.pbm', '.dib', '.ras', '.cur', '.fpx', '.png', '.msp', '.iim', '.wmf', '.tga',
                              '.bufr', '.ico', '.psd', '.xpm', '.arg', '.pdf', '.tiff'}

SUPPORTED_IMAGE_EXTENSIONS.add(".mat")
# The following is a list of the extensions as gathered from Bio-formats
# Missing are .cfg, .csv, .html, .htm, .log, .txt, .xml and .zip which are likely
# not to be images but you are welcome to add if needed
#
SUPPORTED_IMAGE_EXTENSIONS.update(
        [".1sc", ".2fl", ".acff", ".afi", ".afm", ".aiix", ".aim", ".aisf",
         ".al3d", ".ali", ".am", ".amiramesh", ".ano", ".apl", ".arf", ".atsf",
         ".avi", ".bip", ".bmp", ".btf", ".c01", ".cr2", ".crw",
         ".cxd", ".czi", ".dat", ".dcm", ".df3", ".dib", ".dic", ".dicom", ".dm2",
         ".dm3", ".dm4", ".dti", ".dv", ".dv.log", ".eps", ".epsi", ".ets",
         ".exp", ".fake", ".fdf", ".fff", ".ffr", ".fits", ".flex", ".fli",
         ".frm", ".fts", ".gel", ".gif", ".grey", ".hdr", ".hed", ".his", ".htd",
         ".hx", ".ics", ".ids", ".ima", ".img", ".ims", ".inf", ".inr", ".ipl",
         ".ipm", ".ipw", ".j2k", ".j2ki", ".j2kr", ".jp2", ".jpe", ".jpeg",
         ".jpf", ".jpg", ".jpk", ".jpx", ".l2d", ".labels", ".lei", ".lif",
         ".liff", ".lim", ".lsm", ".lut", ".map", ".mdb", ".mea",
         ".mnc", ".mng", ".mod", ".mrc", ".mrw", ".msr", ".mtb",
         ".mvd2", ".naf", ".nd", ".nd2", ".ndpi", ".ndpis", ".nef", ".nhdr",
         ".nii", ".nrrd", ".obf", ".oib", ".oif", ".ome", ".ome.tif",
         ".ome.tiff", ".par", ".pcoraw", ".pct", ".pcx", ".pgm", ".pic",
         ".pict", ".png", ".pnl", ".pr3", ".ps", ".psd", ".pst", ".pty",
         ".r3d", ".r3d.log", ".r3d_d3d", ".raw", ".rec", ".res", ".scn",
         ".sdt", ".seq", ".sif", ".sld", ".sm2", ".sm3", ".spi", ".spl",
         ".st", ".stk", ".stp", ".svs", ".sxm", ".tf2", ".tf8", ".tfr",
         ".tga", ".thm", ".tif", ".tiff", ".tim", ".tnb", ".top",
         ".v", ".vms", ".vsi", ".vws", ".wat", ".wav", ".wlz", ".xdce",
         ".xlog", ".xqd", ".xqf", ".xv", ".xys", ".zfp", ".zfr",
         ".zpo", ".zvi"])

SUPPORTED_MOVIE_EXTENSIONS = {'.avi', '.mpeg', '.stk', '.flex', '.mov', '.tif', '.tiff', '.zvi'}

SUPPORTED_IMAGE_EXTENSIONS.update([
    ".1sc", ".2fl", ".afm", ".aim", ".avi", ".co1", ".flex", ".fli", ".gel",
    ".ics", ".ids", ".im", ".img", ".j2k", ".lif", ".lsm", ".mpeg", ".pic",
    ".pict", ".ps", ".raw", ".svs", ".stk", ".tga", ".zvi", ".c01", ".xdce"])

SUPPORTED_MOVIE_EXTENSIONS.update(['mng'])

'''STK TIFF Tag UIC1 - for MetaMorph internal use'''
UIC1_TAG = 33628

'''STK TIFF Tag UIC2 - stack z distance, creation time...'''
UIC2_TAG = 33629

'''STK TIFF TAG UIC3 - wavelength'''
UIC3_TAG = 33630

'''STK TIFF TAG UIC4 - internal'''
UIC4_TAG = 33631

# strings for choice variables
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGEXP = 'Text-Regular expressions'
MS_ORDER = 'Order'
FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi,mov movies'
FF_AVI_MOVIES_OLD = ['avi movies']
FF_OTHER_MOVIES = 'tif,tiff,flex,zvi movies'
FF_OTHER_MOVIES_OLD = ['tif,tiff,flex movies', 'tif,tiff,flex movies, zvi movies']
IMAGE_FOR_OBJECTS_F = "IMAGE_FOR_%s"
FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
M_NONE = "None"
M_FILE_NAME = "File name"
M_PATH = "Path"
M_BOTH = "Both"
M_Z = "Z"
M_T = "T"

'''The provider name for the image file image provider'''
P_IMAGES = "LoadImagesImageProvider"
'''The version number for the __init__ method of the image file image provider'''
V_IMAGES = 1

'''The provider name for the movie file image provider'''
P_MOVIES = "LoadImagesMovieProvider"
'''The version number for the __init__ method of the movie file image provider'''
V_MOVIES = 2

'''The provider name for the flex file image provider'''
P_FLEX = 'LoadImagesFlexFrameProvider'
'''The version number for the __init__ method of the flex file image provider'''
V_FLEX = 1


'''Interleaved movies'''
I_INTERLEAVED = "Interleaved"

'''Separated movies'''
I_SEPARATED = "Separated"

'''Subfolder choosing options'''
SUB_NONE = "None"
SUB_ALL = "All"
SUB_SOME = "Some"


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

    def __init__(self, data=None, mask=None, crop_mask=None, parent=None, masking_objects=None, convert=True,
                 pathname=None, filename=None, scale=None):
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
        self.filename = filename
        self.pathname = pathname
        self.channel_names = None
        self.has_parent_image = self.parent is not None
        self.has_masking_objects = self.__masking_objects is not None
        self.labels = self.crop_image_similarly(self.masking_objects.segmented) if self.has_masking_objects else None
        self.has_channel_names = self.channel_names is not None
        self.scale = self.parent.scale if self.__scale is None and self.has_parent_image else self.__scale

    def grayscale(self):
        data = self.pixel_data if self.pixel_data.dtype.kind == "b" else self.pixel_data[:, :, 0]

        return Image(data)

    def rgb(self):
        data = self.pixel_data[:, :, :3]

        return Image(data)

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
        """True if the image or its ancestors has a crop mask"""
        return self.__crop_mask is not None or self.has_masking_objects or (
        self.has_parent_image and self.parent.has_crop_mask)

    def crop_image_similarly(self, image):
        """Crop a 2-d or 3-d image using this image's crop mask

        image - a np.ndarray to be cropped (of any type)
        """
        if image.shape[:2] == self.pixel_data.shape[:2]:
            # Same size - no cropping needed
            return image

        if any([my_size > other_size for my_size, other_size in zip(self.pixel_data.shape, image.shape)]):
            raise ValueError(
                "Image to be cropped is smaller: %s vs %s" % (repr(image.shape), repr(self.pixel_data.shape)))

        if not self.has_crop_mask:
            raise RuntimeError(
                "Images are of different size and no crop mask available.\nUse the Crop and Align modules to match images of different sizes.")

        cropped_image = crop_image(image, self.crop_mask)

        if cropped_image.shape[0:2] != self.pixel_data.shape[0:2]:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s" % (
            repr(cropped_image.shape), repr(self.pixel_data.shape)))

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

    def get_image(self, name, must_be_binary=False, must_be_color=False, must_be_grayscale=False, must_be_rgb=False,
                  cache=True):
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

        if must_be_grayscale and (image.pixel_data.ndim != 2):
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
                number = self.__image_sets_by_key[k].number
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
        """True if some image set has been added with a key instead of a number

        This will return "None" if no association has been done.
        """
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
    """Make a dictionary into a stable key for another dictionary"""
    return u", ".join([u":".join([unicode(y) for y in x]) for x in sorted(key.iteritems())])


def default_cpimage_name(index):
    # the usual suspects
    names = ['DNA', 'Actin', 'Protein']
    if index < len(names):
        return names[index]
    return 'Channel%d' % (index + 1)


def well_metadata_tokens(tokens):
    """Return the well row and well column tokens out of a set of metadata tokens"""

    well_row_token = None
    well_column_token = None
    for token in tokens:
        if cellprofiler.measurement.is_well_row_token(token):
            well_row_token = token
        if cellprofiler.measurement.is_well_column_token(token):
            well_column_token = token
    return well_row_token, well_column_token


def needs_well_metadata(tokens):
    """Return true if, based on a set of metadata tokens, we need a well token

    Check for a row and column token and the absence of the well token.
    """
    if cellprofiler.measurement.FTR_WELL.lower() in [x.lower() for x in tokens]:
        return False
    well_row_token, well_column_token = well_metadata_tokens(tokens)
    return (well_row_token is not None) and (well_column_token is not None)


def is_image(filename):
    """Determine if a filename is a potential image file based on extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_movie(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_MOVIE_EXTENSIONS


class LoadImagesImageProviderBase(cellprofiler.image.AbstractImageProvider):
    """Base for image providers: handle pathname and filename & URLs"""

    def __init__(self, name, pathname, filename):
        """Initializer

        name - name of image to be provided
        pathname - path to file or base of URL
        filename - filename of file or last chunk of URL
        """
        if pathname.startswith(cellprofiler.utilities.url.FILE_SCHEME):
            pathname = cellprofiler.utilities.url.url2pathname(pathname)
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__cached_file = None
        self.__is_cached = False
        self.__cacheing_tried = False
        if pathname is None:
            self.__url = filename
        elif any([pathname.startswith(s + ":") for s in cellprofiler.utilities.url.PASSTHROUGH_SCHEMES]):
            if filename is not None:
                self.__url = pathname + "/" + filename
            else:
                self.__url = pathname
        elif filename is None:
            self.__url = cellprofiler.utilities.url.pathname2url(pathname)
        else:
            self.__url = cellprofiler.utilities.url.pathname2url(os.path.join(pathname, filename))

    def get_name(self):
        return self.__name

    def get_pathname(self):
        return self.__pathname

    def get_filename(self):
        return self.__filename

    def cache_file(self):
        """Cache a file that needs to be HTTP downloaded

        Return True if the file has been cached
        """
        if self.__cacheing_tried:
            return self.__is_cached
        self.__cacheing_tried = True
        #
        # Check to see if the pathname can be accessed as a directory
        # If so, handle normally
        #
        path = self.get_pathname()
        if len(path) == 0:
            filename = self.get_filename()
            if os.path.exists(filename):
                return False
            parsed_path = urlparse.urlparse(filename)
            url = filename
            if len(parsed_path.scheme) < 2:
                raise IOError("Test for access to file failed. File: %s" % filename)
        elif os.path.exists(path):
            return False
        else:
            parsed_path = urlparse.urlparse(path)
            url = '/'.join((path, self.get_filename()))
            #
            # Scheme length == 0 means no scheme
            # Scheme length == 1 - probably DOS drive letter
            #
            if len(parsed_path.scheme) < 2:
                raise IOError("Test for access to directory failed. Directory: %s" % path)
        if parsed_path.scheme == 'file':
            self.__cached_file = cellprofiler.utilities.url.url2pathname(path)
        elif self.is_matlab_file():
            #
            # urlretrieve uses the suffix of the path component of the URL
            # to name the temporary file, so we replicate that behavior
            #
            temp_dir = cellprofiler.preferences.get_temporary_directory()
            tempfd, temppath = tempfile.mkstemp(suffix=".mat", dir=temp_dir)
            self.__cached_file = temppath
            try:
                self.__cached_file, headers = urllib.urlretrieve(
                        url, filename=temppath)
            finally:
                os.close(tempfd)
        else:
            from bioformats.formatreader import get_image_reader
            rdr = get_image_reader(id(self), url=url)
            self.__cached_file = rdr.path
        self.__is_cached = True
        return True

    def get_full_name(self):
        self.cache_file()
        if self.__is_cached:
            return self.__cached_file
        return os.path.join(self.get_pathname(), self.get_filename())

    def get_url(self):
        """Get the URL representation of the file location"""
        return self.__url

    def is_matlab_file(self):
        """Return True if the file name ends with .mat (no Bio-formats)"""
        path = urlparse.urlparse(self.get_url())[2]
        return path.lower().endswith(".mat")

    def get_md5_hash(self, measurements):
        """Compute the MD5 hash of the underlying file or use cached value

        measurements - backup for case where MD5 is calculated on image data
                       directly retrieved from URL
        """
        #
        # Cache the MD5 hash on the image reader
        #
        if self.is_matlab_file():
            rdr = None
        else:
            from bioformats.formatreader import get_image_reader
            rdr = get_image_reader(None, url=self.get_url())
        if rdr is None or not hasattr(rdr, "md5_hash"):
            hasher = hashlib.md5()
            path = self.get_full_name()
            if not os.path.isfile(path):
                # No file here - hash the image
                image = self.provide_image(measurements)
                hasher.update(image.pixel_data.tostring())
            else:
                with open(self.get_full_name(), "rb") as fd:
                    while True:
                        buf = fd.read(65536)
                        if len(buf) == 0:
                            break
                        hasher.update(buf)
            if rdr is None:
                return hasher.hexdigest()
            rdr.md5_hash = hasher.hexdigest()
        return rdr.md5_hash

    def release_memory(self):
        """Release any image memory

        Possibly delete the temporary file"""
        if self.__is_cached:
            if self.is_matlab_file():
                try:
                    os.remove(self.__cached_file)
                except:
                    pass
            else:
                from bioformats.formatreader import release_image_reader
                release_image_reader(id(self))
            self.__is_cached = False
            self.__cacheing_tried = False
            self.__cached_file = None

    def __del__(self):
        # using __del__ is all kinds of bad, but we need to remove the
        # files to keep the system from filling up.
        self.release_memory()


class LoadImagesImageProvider(LoadImagesImageProviderBase):
    """Provide an image by filename, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, rescale=True,
                 series=None, index=None, channel=None):
        super(LoadImagesImageProvider, self).__init__(name, pathname, filename)
        self.rescale = rescale
        self.series = series
        self.index = index
        self.channel = channel

    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        from bioformats.formatreader import get_image_reader
        self.cache_file()
        filename = self.get_filename()
        channel_names = []
        if isinstance(self.rescale, float):
            rescale = False
        else:
            rescale = self.rescale
        if self.is_matlab_file():
            with open(self.get_full_name(), "rb") as fd:
                imgdata = scipy.io.matlab.mio.loadmat(
                        fd, struct_as_record=True)
            img = imgdata["Image"]
            # floating point - scale = 1:1
            self.scale = 1.0
            pixel_type_scale = 1.0
        else:
            url = self.get_url()
            if url.lower().startswith("omero:"):
                rdr = get_image_reader(self.get_name(), url=url)
            else:
                rdr = get_image_reader(
                        self.get_name(), url=self.get_url())
            if numpy.isscalar(self.index) or self.index is None:
                img, self.scale = rdr.read(
                        c=self.channel,
                        series=self.series,
                        index=self.index,
                        rescale=self.rescale,
                        wants_max_intensity=True,
                        channel_names=channel_names)
            else:
                # It's a stack
                stack = []
                if numpy.isscalar(self.series):
                    series_list = [self.series] * len(self.index)
                else:
                    series_list = self.series
                if not numpy.isscalar(self.channel):
                    channel_list = [self.channel] * len(self.index)
                else:
                    channel_list = self.channel
                for series, index, channel in zip(
                        series_list, self.index, channel_list):
                    img, self.scale = rdr.read(
                            c=channel,
                            series=series,
                            index=index,
                            rescale=self.rescale,
                            wants_max_intensity=True,
                            channel_names=channel_names)
                    stack.append(img)
                img = numpy.dstack(stack)
        if isinstance(self.rescale, float):
            # Apply a manual rescale
            img = img.astype(numpy.float32) / self.rescale
        image = cellprofiler.image.Image(img,
                                         pathname=self.get_pathname(),
                                         filename=self.get_filename(),
                                         scale=self.scale)
        if img.ndim == 3 and len(channel_names) == img.shape[2]:
            image.channel_names = list(channel_names)
        return image


class LoadImagesImageProviderURL(LoadImagesImageProvider):
    """Reference an image via a URL"""

    def __init__(self, name, url, rescale=True,
                 series=None, index=None, channel=None):
        if url.lower().startswith("file:"):
            path = cellprofiler.utilities.url.url2pathname(url)
            pathname, filename = os.path.split(path)
        else:
            pathname = ""
            filename = url
        super(LoadImagesImageProviderURL, self).__init__(
                name, pathname, filename, rescale, series, index, channel)
        self.url = url

    def get_url(self):
        if self.cache_file():
            return super(LoadImagesImageProviderURL, self).get_url()
        return self.url


class LoadImagesMovieFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale):
        super(LoadImagesMovieFrameProvider, self).__init__(
                name, pathname, filename, rescale, index=frame)


class LoadImagesFlexFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, series, index, rescale):
        super(LoadImagesFlexFrameProvider, self).__init__(
                name, pathname, filename,
                rescale=rescale,
                series=series,
                index=index)


class LoadImagesSTKFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame from an STK file"""

    def __init__(self, name, pathname, filename, frame, rescale):
        """Initialize the provider

        name - name of the provider for access from image set
        pathname - path to the file
        filename - name of the file
        frame - # of the frame to provide
        """
        super(LoadImagesSTKFrameProvider, self).__init__(
                name, pathname, filename, rescale=rescale, index=frame)


def convert_image_to_objects(image):
    """Interpret an image as object indices

    image - a greyscale or color image, assumes zero == background

    returns - a similarly shaped integer array with zero representing background
              and other values representing the indices of the associated object.
    """
    assert isinstance(image, numpy.ndarray)
    if image.ndim == 2:
        unique_indices = numpy.unique(image.ravel())
        if (len(unique_indices) * 2 > max(numpy.max(unique_indices), 254) and
                numpy.all(numpy.abs(numpy.round(unique_indices, 1) - unique_indices) <=
                           numpy.finfo(float).eps)):
            # Heuristic: reinterpret only if sparse and roughly integer
            return numpy.round(image).astype(int)
        sorting = lambda x: [x]
        comparison = lambda i0, i1: image.ravel()[i0] != image.ravel()[i1]
    else:
        i, j = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
        sorting = lambda x: [x[:, :, 2], x[:, :, 1], x[:, :, 0]]
        comparison = lambda i0, i1: \
            numpy.any(image[i.ravel()[i0], j.ravel()[i0], :] !=
                   image[i.ravel()[i1], j.ravel()[i1], :], 1)
    order = numpy.lexsort([x.ravel() for x in sorting(image)])
    different = numpy.hstack([[False], comparison(order[:-1], order[1:])])
    index = numpy.cumsum(different)
    image = numpy.zeros(image.shape[:2], index.dtype)
    image.ravel()[order] = index
    return image


def bad_sizes_warning(first_size, first_filename,
                      second_size, second_filename):
    """Return a warning message about sizes being wrong

    first_size: tuple of height / width of first image
    first_filename: file name of first image
    second_size: tuple of height / width of second image
    second_filename: file name of second image
    """
    warning = ("Warning: loading image files of different dimensions.\n\n"
               "%s: width = %d, height = %d\n"
               "%s: width = %d, height = %d") % (
                  first_filename, first_size[1], first_size[0],
                  second_filename, second_size[1], second_size[0])
    return warning
