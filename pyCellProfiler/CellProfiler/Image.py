"""Image.py

Image        - Represents an image with secondary attributes such as a mask and labels
ImageSetList - Represents the list of image filenames that make up a pipeline run
"""
import numpy
import math
import sys

class Image(object):
    """An image composed of a Numpy array plus secondary attributes such as mask and label matrices
    """
    def __init__(self,image=None,mask=None):
        self.__image = None
        self.__mask = None
        self.__has_mask = False
        if image!=None:
            self.SetImage(image)
        if mask!=None:
            self.SetMask(mask)
        
    def GetImage(self):
        """Return the primary image"""
        return self.__image
    
    def SetImage(self,image):
        """Set the primary image
        
        Convert the image to a numpy array of dtype = numpy.float64.
        Rescale according to Matlab's rules for im2double:
        * single/double values: keep the same
        * uint8/16/32/64: scale 0 to max to 0 to 1
        * int8/16/32/64: scale min to max to 0 to 1
        * logical: False = 0, True = 1
        """
        img = numpy.array(image)
        if img.dtype.type is numpy.bool:
            img2 = numpy.zeros(shape=self.__image.shape,dtype=numpy.float64)
            img2[img]=1.0
            self.__image = img2
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
    
    Image=property(GetImage,SetImage)
    """The primary image - a Numpy array representing an image"""

    def GetMask(self):
        """Return the mask (pixels to be considered) for the primary image
        """
        if self.__mask == None and self.__image != None:
            self.__mask = numpy.ones(self.__image.shape,dtype=numpy.bool)
        return self.__mask
    
    def SetMask(self, mask):
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

    Mask=property(GetMask,SetMask)
    
    def GetHasMask(self):
        return self.__has_mask
    
    HasMask = property(GetHasMask)
    
    
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
    def ProvideImage(self, image_set):
        """Return the image that is associated with the image set
        """
        raise NotImplementedError("Please implement ProvideImage for your class")
    
    def Name(self):
        """The user-visible name for the image
        """
        raise NotImplementedError("Please implement Name for your class")

class VanillaImageProvider(AbstractImageProvider):
    """This image provider returns the image given to it in the constructor
    
    """
    def __init__(self,name,image):
        """Constructor takes the name of the image and the CellProfiler.Image.Image instance to be returned
        """
        self.__name = name
        self.__image = image
    def ProvideImage(self, image_set):
        return self.__image
    
    def Name(self):
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
        
    def ProvideImage(self, image_set):
        return self.__image_provider_fn(image_set,self)
    
    def Name(self):
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
    
    def GetNumber(self):
        """The (zero-based) image set index
        """ 
        return self.__number
    
    Number = property(GetNumber)
    
    def GetKeys(self):
        """The keys that uniquely identify the image set
        """
        return self.__keys
    
    Keys = property(GetKeys)
    
    def GetImage(self, name):
        """Return the image associated with the given name
        """
        if not self.__images.has_key(name):
            providers = filter(lambda x: x.Name() == name, self.__image_providers)
            assert len(providers)>0, "No provider of the %s image"%(name)
            assert len(providers)==1, "More than one provider of the %s image"%(name)
            image = providers[0].ProvideImage(self)
            self.__images[name] = image
        return self.__images[name]
    
    def GetProviders(self):
        """The list of providers (populated during the image discovery phase)"""
        return self.__image_providers
    
    Providers = property(GetProviders)
    
    def GetNames(self):
        """Get the image provider names
        """
        return [provider.Name() for provider in Providers]
    
    Names = property(GetNames)
    
    def GetLegacyFields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.
        
        """
        return self.__legacy_fields
    
    LegacyFields = property(GetLegacyFields)

class ImageSetList(object):
    """Represents the list of image sets in a pipeline run
    
    """
    def __init__(self):
        self.__image_sets = []
        self.__image_sets_by_key = {}
        self.__legacy_fields = {}
    
    def GetImageSet(self,keys_or_number):
        """Return either the indexed image set (keys_or_number = index) or the image set with matching keys
        
        """
        if not isinstance(keys_or_number, dict):
            keys = {'number':keys_or_number }
            number = keys_or_number
            assert number <= len(self.__image_sets)
        else:
            keys = keys_or_number
            if self.__image_sets_by_key.has_key(repr(keys)):
                number = self.__image_sets_by_key[repr(keys)].GetNumber()
            else:
                number = len(self.__image_sets)
        if number == len(self.__image_sets):
            image_set = ImageSet(number,keys,self.__legacy_fields)
            self.__image_sets.append(image_set)
            self.__image_sets_by_key[repr(keys)] = image_set
        else:
            image_set = self.__image_sets[number]
        return image_set
    
    def Count(self):
        return len(self.__image_sets)

    def GetLegacyFields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.
        
        """
        return self.__legacy_fields
    
    LegacyFields = property(GetLegacyFields)
    
