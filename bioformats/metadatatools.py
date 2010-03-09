''' metadatatools.py - mechanism to wrap some bioformats metadata classes

'''

__version__ = "$Revision: 1$"

import cellprofiler.utilities.jutil as jutil
import bioformats
import cellprofiler.utilities.javabridge as javabridge


def createOMEXMLMetadata():
    '''Creates an OME-XML metadata object using reflection, to avoid direct 
    dependencies on the optional loci.formats.ome package.
    '''
    return jutil.static_call('loci/formats/MetadataTools', 'createOMEXMLMetadata', '()Lloci/formats/meta/IMetadata;')


class MetadataStore(object):
    '''  '''
    def __init__(self, o):
        self.o = o
        
    createRoot = jutil.make_method('createRoot', '()V', '')
    setPixelsBigEndian = jutil.make_method('setPixelsBigEndian', '(Ljava/lang/Boolean;II)V',
                                            'for a particular Pixels, sets endianness of the pixels set.')
    setPixelsDimensionOrder = jutil.make_method('setPixelsDimensionOrder', '(Ljava/lang/String;II)V', 
                                                'For a particular Pixels, sets the dimension order of the pixels set.')
    setPixelsPixelType = jutil.make_method('setPixelsPixelType', '(Ljava/lang/String;II)V', 
                                           'For a particular Pixels, sets the pixel type.')
    setPixelsSizeX = jutil.make_method('setPixelsSizeX', '(Ljava/lang/Integer;II)V', 
                                       'For a particular Pixels, sets The size of an individual plane or section\'s X axis (width).')
    setPixelsSizeY = jutil.make_method('setPixelsSizeY', '(Ljava/lang/Integer;II)V', 
                                       'For a particular Pixels, sets The size of an individual plane or section\'s Y axis (height).')
    setPixelsSizeZ = jutil.make_method('setPixelsSizeZ', '(Ljava/lang/Integer;II)V', 
                                       'For a particular Pixels, sets number of optical sections per stack.')
    setPixelsSizeC = jutil.make_method('setPixelsSizeC', '(Ljava/lang/Integer;II)V', 
                                       'For a particular Pixels, sets number of channels per timepoint.')
    setPixelsSizeT = jutil.make_method('setPixelsSizeT', '(Ljava/lang/Integer;II)V', 
                                       'For a particular Pixels, sets number of timepoints.')
    setLogicalChannelSamplesPerPixel = jutil.make_method('setLogicalChannelSamplesPerPixel',
                                                         '(Ljava/lang/Integer;II)V',
                                                         'For a particular LogicalChannel, sets number of channel components in the logical channel.')


class MetadataRetrieve(object):
    '''  '''
    def __init__(self, o):
        self.o = o
        
    getPixelsBigEndian = jutil.make_method('getPixelsBigEndian', '(II)Ljava/lang/Boolean;',
                                            'For a particular Pixels, gets endianness of the pixels set.')
    getPixelsDimensionOrder = jutil.make_method('getPixelsDimensionOrder', '(II)Ljava/lang/String;', 
                                                'For a particular Pixels, gets the dimension order of the pixels set.')
    getPixelsPixelType = jutil.make_method('getPixelsPixelType', '(II)Ljava/lang/String;', 
                                           'For a particular Pixels, gets the pixel type.')
    getPixelsSizeX = jutil.make_method('getPixelsSizeX', '(II)Ljava/lang/Integer;', 
                                       'For a particular Pixels, gets The size of an individual plane or section\'s X axis (width).')
    getPixelsSizeY = jutil.make_method('getPixelsSizeY', '(II)Ljava/lang/Integer;', 
                                       'For a particular Pixels, gets The size of an individual plane or section\'s Y axis (height).')
    getPixelsSizeZ = jutil.make_method('getPixelsSizeZ', '(II)Ljava/lang/Integer;', 
                                       'For a particular Pixels, gets number of optical sections per stack.')
    getPixelsSizeC = jutil.make_method('getPixelsSizeC', '(II)Ljava/lang/Integer;', 
                                       'For a particular Pixels, gets number of channels per timepoint.')
    getPixelsSizeT = jutil.make_method('getPixelsSizeT', '(II)Ljava/lang/Integer;', 
                                       'For a particular Pixels, gets number of timepoints.')
    getLogicalChannelSamplesPerPixel = jutil.make_method('getLogicalChannelSamplesPerPixel', '(II)Ljava/lang/Integer;',
                                                         'For a particular LogicalChannel, gets number of channel components in the logical channel.')


def wrap_imetadata_object(o):
    ''' Returns a python object wrapping the functionality of the given
    IMetaData object (as returned by createOMEXMLMetadata) '''
    class IMetadata(MetadataStore, MetadataRetrieve):
        '''  '''
        def __init__(self, o):
            MetadataStore.__init__(self, o)
            MetadataRetrieve.__init__(self, o)
            self.o = o
    
    return IMetadata(o)

