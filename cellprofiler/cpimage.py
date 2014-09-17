"""Image.py

Image        - Represents an image with secondary attributes such as a mask and labels
ImageSetList - Represents the list of image filenames that make up a pipeline run

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import logging
import numpy as np
import math
import sys

from struct import unpack
from zlib import decompress
from StringIO import StringIO
from numpy import fromstring, uint8, uint16
from cPickle import dump, Unpickler

logger = logging.getLogger(__name__)

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
    def __init__(self,
                 image=None,
                 mask=None,
                 crop_mask = None, 
                 parent_image=None,
                 masking_objects = None,
                 convert = True,
                 path_name = None,
                 file_name = None,
                 scale = None):
        self.__image = None
        self.__mask = None
        self.__has_mask = False
        self.__parent_image = parent_image
        self.__crop_mask = None
        if crop_mask is not None:
            self.set_crop_mask(crop_mask)
            self.__has_crop_mask = True
        else:
            self.__has_crop_mask = False
        self.__masking_objects = masking_objects
        self.__scale = scale
        if image!=None:
            self.set_image(image, convert)
        if mask!=None:
            self.set_mask(mask)
        self.__file_name = file_name
        self.__path_name = path_name
        self.__channel_names = None
        
    def get_image(self):
        """Return the primary image"""
        if self.__image is None:
            return
        return self.__image.get()
    
    def set_image(self,image,convert=True):
        """Set the primary image
        
        Convert the image to a numpy array of dtype = np.float64.
        Rescale according to Matlab's rules for im2double:
        * single/double values: keep the same
        * uint8/16/32/64: scale 0 to max to 0 to 1
        * int8/16/32/64: scale min to max to 0 to 1
        * logical: save as is (and get if must_be_binary)
        """
        img = np.asanyarray(image)
        if img.dtype.name == "bool" or not convert:
            self.__image = ImageCache(img)
            return
        mval  = 0.
        scale = 1.
        fix_range = False
        if issubclass(img.dtype.type,np.floating):
            pass
        elif img.dtype.type is np.uint8:
            scale = math.pow(2.0,8.0)-1
        elif img.dtype.type is np.uint16:
            scale = math.pow(2.0,16.0)-1
        elif img.dtype.type is np.uint32:
            scale = math.pow(2.0,32.0)-1
        elif img.dtype.type is np.uint64:
            scale = math.pow(2.0,64.0)-1
        elif img.dtype.type is np.int8:
            scale = math.pow(2.0,8.0)
            mval  = -scale / 2.0
            scale -=1
            fix_range = True
        elif img.dtype.type is np.int16:
            scale = math.pow(2.0,16.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is np.int32:
            scale = math.pow(2.0,32.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is np.int64:
            scale = math.pow(2.0,64.0)
            mval  = -scale / 2.0
            scale -= 1
            fix_range = True
        # Avoid temporaries by doing the shift/scale in place.
        img = img.astype(np.float32)
        img -= mval
        img /= scale
        if fix_range:
            # These types will always have ranges between 0 and 1. Make it so.
            np.clip(img, 0, 1, out=img)
        check_consistency(img,self.__mask)
        self.__image = ImageCache(img)
    
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
            return self.__mask.get()
        
        if self.has_masking_objects:
            return self.crop_image_similarly(self.crop_mask)
        
        if self.has_parent_image:
            mask = self.parent_image.mask
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
        return np.ones(shape, dtype=np.bool)
    
    def set_mask(self, mask):
        """Set the mask (pixels to be considered) for the primary image
        
        Convert the input into a numpy array. If the input is numeric,
        we convert it to boolean by testing each element for non-zero.
        """
        m = np.array(mask)
        if not(m.dtype.type is np.bool):
            m = (m != 0)
        check_consistency(self.image, m)
        self.__mask = ImageCache(m)
        self.__has_mask = True

    mask=property(get_mask,set_mask)
    
    def get_has_mask(self):
        """True if the image has a mask"""
        if self.__has_mask:
            return True
        if self.has_crop_mask:
            return True
        if self.parent_image != None:
            return self.parent_image.has_mask
        return False
    
    has_mask = property(get_has_mask)
    
    def get_crop_mask(self):
        """Return the mask used to crop this image"""
        if not self.__crop_mask == None:
            return self.__crop_mask.get()
        
        if self.has_masking_objects:
            return self.masking_objects.segmented != 0
        
        if self.has_parent_image:
            return self.parent_image.crop_mask
        #
        # If no crop mask, return the mask which should be all ones
        #
        return self.mask
    
    def set_crop_mask(self,crop_mask):
        self.__crop_mask = ImageCache(crop_mask)
        
    crop_mask = property(get_crop_mask, set_crop_mask)
    
    @property
    def has_crop_mask(self):
        '''True if the image or its ancestors has a crop mask'''
        return (self.__crop_mask is not None or
                self.has_masking_objects or
                (self.has_parent_image and self.parent_image.has_crop_mask))
    
    def crop_image_similarly(self, image):
        """Crop a 2-d or 3-d image using this image's crop mask
        
        image - a np.ndarray to be cropped (of any type)
        """
        if image.shape[:2] == self.pixel_data.shape[:2]:
            # Same size - no cropping needed
            return image
        if any([my_size > other_size
                for my_size,other_size
                in zip(self.pixel_data.shape,image.shape)]):
            raise ValueError("Image to be cropped is smaller: %s vs %s"%
                             (repr(image.shape),
                              repr(self.pixel_data.shape)))
        if not self.has_crop_mask:
            raise RuntimeError(
                "Images are of different size and no crop mask available.\n"
                "Use the Crop and Align modules to match images of different sizes.")
        cropped_image = crop_image(image,self.crop_mask)
        if cropped_image.shape[0:2] != self.pixel_data.shape[0:2]:
            raise ValueError("Cropped image is not the same size as the reference image: %s vs %s"%
                             (repr(cropped_image.shape),
                              repr(self.pixel_data.shape)))
        return cropped_image
    
    def get_file_name(self):
        '''The name of the file holding this image
        
        If the image is derived, then return the file name of the first
        ancestor that has a file name. Return None if the image does not have 
        an ancestor or if no ancestor has a file name.
        '''
        if not self.__file_name is None:
            return self.__file_name
        elif self.has_parent_image:
            return self.parent_image.file_name
        else:
            return None
    
    file_name = property(get_file_name)
    
    def get_path_name(self):
        '''The path to the file holding this image

        If the image is derived, then return the path name of the first
        ancestor that has a path name. Return None if the image does not have 
        an ancestor or if no ancestor has a file name.
        '''
        if not self.__path_name is None:
            return self.__path_name
        elif self.has_parent_image:
            return self.parent_image.path_name
        else:
            return None
    
    path_name = property(get_path_name)
    
    def get_channel_names(self):
        '''The user-defined names of the channels in a channel stack'''
        return self.__channel_names
    
    def set_channel_names(self, names):
        self.__channel_names = tuple(names)

    channel_names = property(get_channel_names, set_channel_names)
    
    @property
    def has_channel_names(self):
        '''True if there are channel names on this image'''
        return self.__channel_names is not None
    
    def get_scale(self):
        '''The scale at acquisition
        
        This is the intensity scale used by the acquisition device. For
        instance, a microscope might use a 12-bit a/d converter to acquire
        an image and store that information using the TIF MaxSampleValue
        tag = 4095.
        '''
        if self.__scale is None and self.has_parent_image:
            return self.parent_image.scale
        return self.__scale
    scale = property(get_scale)
    
    def cache(self, name, hdf5_file):
        '''Move all images into backing stores
        
        name - the channel name of the image
        hdf5_file - an HDF5 file or group
        
        We utilize the sub-groups, "Images", "Masks" and "CropMasks".
        The best practice is to use a temporary file dedicated to images and
        maybe objects.
        '''
        from cellprofiler.utilities.hdf5_dict import HDF5ImageSet
        if isinstance(self.__image, ImageCache) and\
           not self.__image.is_cached():
            self.__image.cache(name, HDF5ImageSet(hdf5_file))
        if isinstance(self.__mask, ImageCache) and\
           not self.__mask.is_cached():
            self.__mask.cache(name, HDF5ImageSet(hdf5_file, "Masks"))
        if isinstance(self.__crop_mask, ImageCache) and\
           not self.__crop_mask.is_cached():
            self.__crop_mask.cache(name, HDF5ImageSet(hdf5_file, "CropMasks"))
    
class ImageCache(object):
    '''An HDF5 cache that can store an image, mask or crop mask
    
    '''
    IC_MONOCHROME = "Monochrome"
    IC_COLOR = "Color"
    IC_5D = "5D"
    
    def __init__(self, image):
        '''Initialize with the image to control'''
        self.__backing_store = None
        self.__name = None
        if image.ndim == 2:
            self.__type = ImageCache.IC_MONOCHROME
            self.__image = image.reshape(1, 1, 1, image.shape[0], image.shape[1])
        elif image.ndim == 3:
            self.__type = ImageCache.IC_COLOR
            self.__image = image.transpose(2, 0, 1).reshape(
                image.shape[2], 1, 1, image.shape[0], image.shape[1])
        else:
            self.__type = ImageCache.IC_5D
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
        if self.__type == ImageCache.IC_MONOCHROME:
            return image.reshape(image.shape[3], image.shape[4])
        elif self.__type == ImageCache.IC_COLOR:
            return image.reshape(
                image.shape[0], image.shape[3], image.shape[4]).transpose(1, 2, 0)
        
    
def crop_image(image, crop_mask,crop_internal = False):
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum    = np.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum    = np.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return np.zeros((0,0),dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = np.argwhere(i_histogram>0)
        j_keep = np.argwhere(j_histogram>0)
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
        i_first     = np.argwhere(i_cumsum==1)[0]
        i_last      = np.argwhere(i_cumsum==i_cumsum.max())[0]
        i_end       = i_last+1
        j_first     = np.argwhere(j_cumsum==1)[0]
        j_last      = np.argwhere(j_cumsum==j_cumsum.max())[0]
        j_end       = j_last+1
        if image.ndim == 3:
            return image[i_first:i_end,j_first:j_end,:].copy()
        return image[i_first:i_end,j_first:j_end].copy()

class GrayscaleImage(object):
    """A wrapper around a non-grayscale image
    
    This is meant to be used if the image is 3-d but all channels
       are the same or if the image is binary.
    """
    def __init__(self, image):
        self.__image = image

    def __getattr__(self, name):
        return getattr(self.__image, name)
    
    def get_pixel_data(self):
        """One 2-d channel of the color image as a numpy array"""
        if self.__image.pixel_data.dtype.kind == 'b':
            return self.__image.pixel_data.astype(np.float64)
        return self.__image.pixel_data[:,:,0]
    
    pixel_data = property(get_pixel_data)
    
class RGBImage(object):
    """A wrapper that discards the alpha channel
    
    This is meant to be used if the image is 3-d + alpha but the alpha
    channel is discarded
    """
    def __init__(self, image):
        self.__image = image

    def __getattr__(self, name):
        return getattr(self.__image, name)
    
    def get_pixel_data(self):
        '''Return the pixel data without the alpha channel'''
        return self.__image.pixel_data[:,:,:3]
    
    pixel_data = property(get_pixel_data)
    
def check_consistency(image, mask):
    """Check that the image, mask and labels arrays have the same shape and that the arrays are of the right dtype"""
    assert (image==None) or (len(image.shape) in (2,3)),"Image must have 2 or 3 dimensions"
    assert (mask==None) or (len(mask.shape)==2),"Mask must have 2 dimensions"
    assert (image==None) or (mask==None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"
    assert (mask==None) or (mask.dtype.type is np.bool_), "Mask must be boolean, was %s"%(repr(mask.dtype.type))

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
        '''Release whatever memory is associated with the image'''
        logger.warning("Warning: no memory release function implemented for %s image",
                       self.get_name())

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
    
    def release_memory(self):
        self.__image = None
    

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
    
    @property
    def image_number(self):
        '''The image number as used in measurements and the database'''
        return self.__number + 1
    
    def get_keys(self):
        """The keys that uniquely identify the image set
        """
        return self.__keys
    
    keys = property(get_keys)
    
    def get_image(self, name,
                 must_be_binary=False,
                 must_be_color=False,
                 must_be_grayscale=False,
                 must_be_rgb = False,
                 cache = True):
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
        if must_be_binary and image.pixel_data.dtype != np.bool:
            raise ValueError("Image was not binary")
        if must_be_color and image.pixel_data.ndim != 3:
            raise ValueError("Image must be color, but it was grayscale")
        if (must_be_grayscale and 
            (image.pixel_data.ndim != 2)):
            pd = image.pixel_data
            if pd.shape[2] >= 3 and\
               np.all(pd[:,:,0]==pd[:,:,1]) and\
               np.all(pd[:,:,0]==pd[:,:,2]):
                return GrayscaleImage(image)
            raise ValueError("Image must be grayscale, but it was color")
        if must_be_grayscale and image.pixel_data.dtype.kind == 'b':
            return GrayscaleImage(image)
        if must_be_rgb:
            if image.pixel_data.ndim != 3:
                raise ValueError("Image must be RGB, but it was grayscale")
            elif image.pixel_data.shape[2] not in (3,4):
                raise ValueError("Image must be RGB, but it had %d channels" %
                                 image.pixel_data.shape[2])
            elif image.pixel_data.shape[2] == 4:
                logger.warning("Discarding alpha channel.")
                return RGBImage(image)
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
        if len(old_providers) > 0:
            self.clear_image(name)
        for provider in old_providers:
            self.providers.remove(provider)
        provider = VanillaImageProvider(name,image)
        self.providers.append(provider)
    
class ImageSetList(object):
    """Represents the list of image sets in a pipeline run
    
    """
    def __init__(self, test_mode = False):
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
    
    def get_image_set(self,keys_or_number):
        """Return either the indexed image set (keys_or_number = index) or the image set with matching keys
        
        """
        if not isinstance(keys_or_number, dict):
            keys = {'number':keys_or_number }
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
            self.__image_sets += [ None ]*(number - len(self.__image_sets)+1)
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
            assert isinstance(image_set, ImageSet)
            key_values = tuple([str(image_set.keys[key]) for key in keys])
            if not d.has_key(key_values):
                d[key_values] = []
                sort_order.append(key_values)
            d[key_values].append(i+1)
        return (keys, [(dict(zip(keys,k)),d[k]) for k in sort_order])
    
    def save_state(self):
        '''Return a string that can be used to load the image_set_list's state
        
        load_state will restore the image set list's state. No image_set can
        have image providers before this call.
        '''
        f = StringIO()
        dump(self.count(),f)
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, ImageSet)
            assert len(image_set.providers)==0, "An image set cannot have providers while saving its state"
            dump(image_set.keys, f)
        dump(self.legacy_fields, f)
        return f.getvalue()
    
    def load_state(self, state):
        '''Load an image_set_list's state from the string returned from save_state'''
        
        self.__image_sets = []
        self.__image_sets_by_key = {}

        # Make a safe unpickler
        p = Unpickler(StringIO(state))
        def find_global(module_name, class_name):
            logger.debug("Pickler wants %s:%s",module_name, class_name)
            if (module_name not in ("numpy", "numpy.core.multiarray")):
                logger.critical(
                    "WARNING WARNING WARNING - your batch file has asked to load %s.%s."
                    " If this looks in any way suspicious please contact us at www.cellprofiler.org",
                    module_name, class_name)
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

def make_dictionary_key(key):
    '''Make a dictionary into a stable key for another dictionary'''
    return u", ".join([u":".join([unicode(y) for y in x])
                       for x in sorted(key.iteritems())])

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

