"""Image.py

Image        - Represents an image with secondary attributes such as a mask and labels
ImageSetList - Represents the list of image filenames that make up a pipeline run

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 1$"

import numpy
import math
import sys

from struct import unpack
from zlib import decompress
from StringIO import StringIO
from numpy import fromstring, uint8, uint16

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
    
    Resolution of mask and cropping_mask properties:
    The Image class looks for the mask and cropping_mask in the following 
    places:
    * self: if set using the properties or specified in the initializer
    * masking_objects: if set using the masking_object property or
                       specified in the initializer. The crop_mask and
                       mask are composed of all of the labeled points.
    * parent_image: if set using the initializer. The child image inherits
                    the mask and cropping mask of the parent.
    * convert: true to try to coerce whatever dtype passed (other than bool
               or float) to a scaled image.
    Otherwise, the image has no mask or cropping mask and all pixels are
    significant.
    """
    def __init__(self,
                 image=None,
                 mask=None,
                 crop_mask = None, 
                 parent_image=None,
                 masking_objects = None,
                 convert = True):
        self.__image = None
        self.__mask = None
        self.__has_mask = False
        self.__parent_image = parent_image
        self.__crop_mask = crop_mask
        self.__masking_objects = masking_objects
        if image!=None:
            self.set_image(image, convert)
        if mask!=None:
            self.set_mask(mask)
        
    def get_image(self):
        """Return the primary image"""
        return self.__image
    
    def set_image(self,image,convert=True):
        """Set the primary image
        
        Convert the image to a numpy array of dtype = numpy.float64.
        Rescale according to Matlab's rules for im2double:
        * single/double values: keep the same
        * uint8/16/32/64: scale 0 to max to 0 to 1
        * int8/16/32/64: scale min to max to 0 to 1
        * logical: save as is (and get if must_be_binary)
        """
        img = numpy.array(image)
        if img.dtype.name == "bool" or not convert:
            self.__image = img
            return
        mval  = 0.
        scale = 1.
        fix_range = False
        if issubclass(img.dtype.type,numpy.floating):
            pass
        elif img.dtype.type is numpy.uint8:
            scale = math.pow(2.0,8.0)-1
        elif img.dtype.type is numpy.uint16:
            scale = math.pow(2.0,16.0)-1
        elif img.dtype.type is numpy.uint32:
            scale = math.pow(2.0,32.0)-1
        elif img.dtype.type is numpy.uint64:
            scale = math.pow(2.0,64.0)-1
        elif img.dtype.type is numpy.int8:
            scale = math.pow(2.0,8.0)
            mval  = -scale / 2.0
            scale -=1
            fix_range = True
        elif img.dtype.type is numpy.int16:
            scale = math.pow(2.0,16.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is numpy.int32:
            scale = math.pow(2.0,32.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is numpy.int64:
            scale = math.pow(2.0,64.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        
        img = (img.astype(numpy.float64) - mval)/scale
        if fix_range:
            # These types will always have ranges between 0 and 1. Make it so.
            img[img<0]=0
            img[img>1]=1
        if len(img.shape)==3:
            if img.shape[2] == 4:
                sys.stderr.write("Warning: discarding alpha channel in color image.\n")
                img = img[:,:,:-1]
        check_consistency(img,self.__mask)
        self.__image = img
    
    image=property(get_image,set_image)
    pixel_data=property(get_image,set_image)

    def get_parent_image(self):
        """The image from which this one was derived"""
        return self.__parent_image
    
    def set_parent_image(self, parent_image):
        self.__parent_image = parent_image
        
    parent_image = property(get_parent_image, set_parent_image)
    
    def get_has_parent_image(self):
        """True if this image has a defined parent"""
        return self.__parent_image != None
    has_parent_image = property(get_has_parent_image)

    def get_masking_objects(self):
        """The objects used to crop and mask this image"""
        return self.__masking_objects
    
    def set_masking_objects(self, value):
        self.__masking_objects = value
    
    masking_objects = property(get_masking_objects, set_masking_objects)
    
    def get_has_masking_objects(self):
        """True if the image was cropped with objects
        
        If this is true, there will also be a valid labels matrix
        available through the labels property
        """
        return self.__masking_objects != None
    has_masking_objects = property(get_has_masking_objects)
    
    def get_labels(self):
        """Get the segmentation labels from the masking objects
        
        returns the "segmented" labels: others are available through
        the masking_object.
        """
        if not self.has_masking_objects:
            return None
        return self.crop_image_similarly(self.masking_objects.segmented)
    labels = property(get_labels)
    
    def get_mask(self):
        """Return the mask (pixels to be considered) for the primary image
        """
        if not self.__mask == None:
            return self.__mask
        
        if self.has_masking_objects:
            return self.crop_image_similarly(self.crop_mask)
        
        if self.has_parent_image:
            return self.parent_image.mask
        
        return numpy.ones(self.__image.shape[0:2],dtype=numpy.bool)
    
    def set_mask(self, mask):
        """Set the mask (pixels to be considered) for the primary image
        
        Convert the input into a numpy array. If the input is numeric,
        we convert it to boolean by testing each element for non-zero.
        """
        m = numpy.array(mask)
        if not(m.dtype.type is numpy.bool):
            m = (m != 0)
        check_consistency(self.__image,m)
        self.__mask = m
        self.__has_mask = True

    mask=property(get_mask,set_mask)
    
    def get_has_mask(self):
        """True if the image has a mask"""
        if (not self.__has_mask) and self.parent_image != None:
            return self.parent_image.has_mask
        return self.__has_mask
    
    has_mask = property(get_has_mask)
    
    def get_crop_mask(self):
        """Return the mask used to crop this image"""
        if not self.__crop_mask == None:
            return self.__crop_mask
        
        if self.has_masking_objects:
            return self.masking_objects.segmented != 0
        
        if self.has_parent_image:
            return self.parent_image.crop_mask
        #
        # If no crop mask, return the mask which should be all ones
        #
        return self.mask
    
    def set_crop_mask(self,crop_mask):
        self.__crop_mask = crop_mask
        
    crop_mask = property(get_crop_mask, set_crop_mask)
    
    def crop_image_similarly(self, image):
        """Crop a 2-d or 3-d image using this image's crop mask
        
        image - a numpy.ndarray to be cropped (of any type)
        """
        if image.shape == self.pixel_data.shape:
            # Same size - no cropping needed
            return image
        if any([my_size > other_size
                for my_size,other_size
                in zip(self.pixel_data.shape,image.shape)]):
            raise ValueError("Image to be cropped is smaller: %s vs %s"%
                             (repr(image.shape),
                              repr(self.pixel_data.shape)))
        if self.crop_mask == None:
            raise RuntimeError("Images are of different size and no crop mask available")
        cropped_image = crop_image(image,self.crop_mask)
        if cropped_image.shape != self.pixel_data.shape:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s"%
                             (repr(cropped_image.shape),
                              repr(self.pixel_data.shape)))
        return cropped_image

def crop_image(image, crop_mask,crop_internal = False):
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum    = numpy.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum    = numpy.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return numpy.zeros((0,0),dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = numpy.argwhere(i_histogram>0)
        j_keep = numpy.argwhere(j_histogram>0)
        #
        # Then slice the array by I, then by J to get what's not blank
        #
        return image[i_keep.flatten(),:][:,j_keep.flatten()].copy()
    else:
        #
        # The first non-blank row and column are where the cumsum is 1
        # The last are at the first where the cumsum is it's max (meaning
        # what came after was all zeros and added nothing)
        #
        i_first     = numpy.argwhere(i_cumsum==1)[0]
        i_last      = numpy.argwhere(i_cumsum==i_cumsum.max())[0]
        i_end       = i_last+1
        j_first     = numpy.argwhere(j_cumsum==1)[0]
        j_last      = numpy.argwhere(j_cumsum==j_cumsum.max())[0]
        j_end       = j_last+1
        if image.ndim == 3:
            return image[i_first:i_end,j_first:j_end,:].copy()
        return image[i_first:i_end,j_first:j_end].copy()

class GrayscaleImage(object):
    """A wrapper around the image object if the image is 3-d but all channels
       are the same
    """
    def __init__(self, image):
        self.__image = image
    
    def get_pixel_data(self):
        """One 2-d channel of the color image as a numpy array"""
        return self.__image.pixel_data[:,:,0]
    
    pixel_data = property(get_pixel_data)
    
    def get_mask(self):
        return self.__image.get_mask()
    mask=property(get_mask)
    
    def get_has_mask(self):
        return self.__image.get_has_mask()
    has_mask=property(get_has_mask)
    
    def get_crop_mask(self):
        return self.__image.crop_mask
    
    crop_mask = property(get_crop_mask)
    
    def get_parent_image(self):
        return self.__image.parent_mask
    
    parent_image = property(get_parent_image)
    
def check_consistency(image, mask):
    """Check that the image, mask and labels arrays have the same shape and that the arrays are of the right dtype"""
    assert (image==None) or (len(image.shape) in (2,3)),"Image must have 2 or 3 dimensions"
    assert (image==None) or (len(image.shape)==2) or (image.shape[2] in (1,3)),"3-dimensional images must have either one or three colors"
    assert (mask==None) or (len(mask.shape)==2),"Mask must have 2 dimensions"
    assert (image==None) or (mask==None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"
    assert (image==None) or (image.dtype.type is numpy.float64), "Image must be float64, was %s"%(repr(image.dtype.type))
    assert (mask==None) or (mask.dtype.type is numpy.bool_), "Mask must be boolean, was %s"%(repr(mask.dtype.type))

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

    name = property(__get_name)

class VanillaImageProvider(AbstractImageProvider):
    """This image provider returns the image given to it in the constructor
    
    """
    def __init__(self,name,image):
        """Constructor takes the name of the image and the CellProfiler.Image.Image instance to be returned
        """
        self.__name = name
        self.__image = image
    def provide_image(self, image_set):
        return self.__image
    
    def get_name(self):
        return self.__name
    
    

class CallbackImageProvider(AbstractImageProvider):
    """An image provider proxy that calls the indicated callback functions (presumably in your module) to implement the methods
    """
    def __init__(self,name,image_provider_fn):
        """Constructor
        name              - name returned by the Name method
        image_provider_fn - function called during ProvideImage with the arguments, image_set and the CallbackImageProvider instance 
        """
        
        self.__name = name
        self.__image_provider_fn = image_provider_fn
        
    def provide_image(self, image_set):
        return self.__image_provider_fn(image_set,self)
    
    def get_name(self):
        return self.__name

class ImageSet(object):
    """Represents the images for a particular iteration of a pipeline
    
    An image set is composed of one image provider per image in the set.
    The image provider loads or creates an image, given a dictionary of keys
    (which might represent things like the plate/well for the image set or the
    frame number in a movie, etc.) 
    """
    def __init__(self, number, keys,legacy_fields):
        """Constructor: 
        number = image set index 
        keys = dictionary of key/value pairs that uniquely identify the image set
        """
        self.__image_providers = []
        self.__images = {}
        self.__keys = keys
        self.__number = number
        self.__legacy_fields = legacy_fields
    
    def get_number(self):
        """The (zero-based) image set index
        """ 
        return self.__number
    
    number = property(get_number)
    
    def get_keys(self):
        """The keys that uniquely identify the image set
        """
        return self.__keys
    
    keys = property(get_keys)
    
    def get_image(self, name,
                 must_be_binary=False,
                 must_be_color=False,
                 must_be_grayscale=False,
                 cache = True):
        """Return the image associated with the given name
        
        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        """
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
        if must_be_grayscale and image.pixel_data.ndim != 2:
            pd = image.pixel_data
            if pd.shape[2] >= 3 and\
               numpy.all(pd[:,:,0]==pd[:,:,1]) and\
               numpy.all(pd[:,:,0]==pd[:,:,2]):
                return GrayscaleImage(image)
            raise ValueError("Image must be grayscale, but it was color") 
        return image
    
    def get_providers(self):
        """The list of providers (populated during the image discovery phase)"""
        return self.__image_providers
    
    providers = property(get_providers)
    
    def get_image_provider(self, name):
        """Get a named image provider
        
        name - return the image provider with this name
        """
        providers = filter(lambda x: x.name == name, self.__image_providers)
        assert len(providers)>0, "No provider of the %s image"%(name)
        assert len(providers)==1, "More than one provider of the %s image"%(name)
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
        old_providers = [provider for provider in self.providers
                         if provider.name == name]
        for provider in old_providers:
            self.providers.remove(provider)
        provider = VanillaImageProvider(name,image)
        self.providers.append(provider)

class ImageSetList(object):
    """Represents the list of image sets in a pipeline run
    
    """
    def __init__(self):
        self.__image_sets = []
        self.__image_sets_by_key = {}
        self.__legacy_fields = {}
    
    def get_image_set(self,keys_or_number):
        """Return either the indexed image set (keys_or_number = index) or the image set with matching keys
        
        """
        if not isinstance(keys_or_number, dict):
            keys = {'number':keys_or_number }
            number = keys_or_number
            assert number <= len(self.__image_sets)
        else:
            keys = keys_or_number
            if self.__image_sets_by_key.has_key(repr(keys)):
                number = self.__image_sets_by_key[repr(keys)].get_number()
            else:
                number = len(self.__image_sets)
        if number == len(self.__image_sets):
            image_set = ImageSet(number,keys,self.__legacy_fields)
            self.__image_sets.append(image_set)
            self.__image_sets_by_key[repr(keys)] = image_set
        else:
            image_set = self.__image_sets[number]
        return image_set
    
    def purge_image_set(self, number):
        """Remove the memory associated with an image set"""
        keys = self.__image_sets[number].keys
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
    

def readc01(fname):
    '''Read a Cellomics file into an array
    
    fname - the name of the file
    '''
    def readint(f):
        return unpack("<l", f.read(4))[0]
    
    def readshort(f):
        return unpack("<h", f.read(2))[0]
    
    f = open(fname, "rb")
    
    # verify it's a c01 format, and skip the first four bytes
    assert readint(f) == 16 << 24

    # decompress
    g = StringIO(decompress(f.read()))
    
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


    data = fromstring(g.read(), uint16 if nbits == 16 else uint8, x * y)
    return data.reshape(x, y).T

