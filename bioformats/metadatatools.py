''' metadatatools.py - mechanism to wrap some bioformats metadata classes

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__ = "$Revision$"

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
    def setPixelsBigEndian(self, bigEndian, imageIndex, binDataIndex):
        '''Set the endianness for a particular image
        
        bigEndian - True for big-endian, False for little-endian
        imageIndex - index of the image in question from IFormatReader.get_index?
        binDataIndex - ???
        '''
        # Post loci_tools 4.2
        try:
            jutil.call(self.o, 'setPixelsBinDataBigEndian',
                       '(Ljava/lang/Boolean;II)V',
                       bigEndian, imageIndex, binDataIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsBigEndian', '(Ljava/lang/Boolean;II)V',
                       bigEndian, imageIndex, binDataIndex)

    def setPixelsDimensionOrder(self, dimension_order, imageIndex, binDataIndex):
        '''Set the dimension order for a series'''
        # Post loci_tools 4.2 - use ome.xml.model.DimensionOrder
        try:
            jdimension_order = jutil.static_call(
                'ome/xml/model/enums/DimensionOrder', 'fromString',
                '(Ljava/lang/String;)Lome/xml/model/enums/DimensionOrder;',
                dimension_order)
            jutil.call(self.o, 'setPixelsDimensionOrder',
                       '(Lome/xml/model/enums/DimensionOrder;I)V',
                       jdimension_order, imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsDimensionOrder',
                       '(Ljava/lang/String;II)V',
                       dimension_order, imageIndex, binDataIndex)
            
    setPixelsPixelType = jutil.make_method(
        'setPixelsPixelType', '(Ljava/lang/String;II)V', 
        '''Sets the pixel storage type
        pixel_type - text representation of the type, e.g. "uint8"
        imageIndex - ?
        binDataIndex - ?
        
        WARNING: only available in BioFormats < 4.2
        ''')
    setPixelsType = jutil.make_method(
        'setPixelsType', '(Lome/xml/model/enums/PixelType;I)V',
        '''Set the pixel storage type
        
        pixel_type - one of the enumerated values from PixelType.
        imageIndex - ?
        
        See the ome.xml.model.enums.PixelType and make_pixel_type_class's
        PixelType for possible values.
        ''')
    
    def setPixelsSizeX(self, x, imageIndex, binDataIndex):
        try:
            jutil.call(self.o, 'setPixelsSizeX',
                       '(Lome/xml/model/primitives/PositiveInteger;I)V',
                       PositiveInteger(x), imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsSizeX', 
                       '(Ljava/lang/Integer;II)V', x, imageIndex, binDataIndex)

    def setPixelsSizeY(self, y, imageIndex, binDataIndex):
        try:
            jutil.call(self.o, 'setPixelsSizeY',
                       '(Lome/xml/model/primitives/PositiveInteger;I)V',
                       PositiveInteger(y), imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsSizeY', 
                       '(Ljava/lang/Integer;II)V', y, imageIndex, binDataIndex)
            
    def setPixelsSizeZ(self, z, imageIndex, binDataIndex):
        try:
            jutil.call(self.o, 'setPixelsSizeZ',
                       '(Lome/xml/model/primitives/PositiveInteger;I)V',
                       PositiveInteger(z), imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsSizeZ', 
                       '(Ljava/lang/Integer;II)V', z, imageIndex, binDataIndex)

    def setPixelsSizeC(self, c, imageIndex, binDataIndex):
        try:
            jutil.call(self.o, 'setPixelsSizeC',
                       '(Lome/xml/model/primitives/PositiveInteger;I)V',
                       PositiveInteger(c), imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsSizeC', 
                       '(Ljava/lang/Integer;II)V', c, imageIndex, binDataIndex)

    def setPixelsSizeT(self, t, imageIndex, binDataIndex):
        try:
            jutil.call(self.o, 'setPixelsSizeT',
                       '(Lome/xml/model/primitives/PositiveInteger;I)V',
                       PositiveInteger(t), imageIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setPixelsSizeT', 
                       '(Ljava/lang/Integer;II)V', t, imageIndex, binDataIndex)
            
    def setLogicalChannelSamplesPerPixel(self, samplesPerPixel, imageIndex, channelIndex):
        'For a particular LogicalChannel, sets number of channel components in the logical channel.'
        try:
            jutil.call(self.o, 'setChannelSamplesPerPixel',
                       '(Lome/xml/model/primitives/PositiveInteger;II)V',
                       PositiveInteger(samplesPerPixel),
                       imageIndex, channelIndex)
        except jutil.JavaException:
            jutil.call(self.o, 'setLogicalChannelSamplesPerPixel',
                       '(Ljava/lang/Integer;II)V', samplesPerPixel,
                       imageIndex, channelIndex)
    setImageID = jutil.make_method(
        'setImageID', '(Ljava/lang/String;I)V',
        '''Tag the indexed image with a name
        
        id - the name, for instance Image:0
        imageIndex - the index of the image (series???)
        ''')
    setPixelsID = jutil.make_method(
        'setPixelsID', '(Ljava/lang/String;I)V',
        '''Tag the pixels with a name (???)
        
        id - the name, for instance Pixels:0
        imageIndex - the index of the image (???)
        ''')
    setChannelID = jutil.make_method(
        'setChannelID', '(Ljava/lang/String;II)V',
        '''Give an ID name to the given channel
        
        id - the name of the channel
        imageIndex - (???)
        channelIndex - index of the channel to be ID'ed''')

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
    getChannelName = jutil.make_method('getChannelName',
                                       '(II)Ljava/lang/String;',
                                       '''Get the name for a particular channel.
                                       
                                       imageIndex - image # to query (use C = 0)
                                       channelIndex - channel # to query''')
    getChannelID = jutil.make_method('getChannelID',
                                     '(II)Ljava/lang/String;',
                                     '''Get the OME channel ID for a particular channel.
                                     
                                     imageIndex - image # to query (use C = 0)
                                     channelIndex - channel # to query''')


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

__pixel_type_class = None
def make_pixel_type_class():
    '''The class, ome.xml.model.enums.PixelType
    
    The Java class has enumerations for the various image data types
    such as UINT8 or DOUBLE
    '''
    global __pixel_type_class
    if __pixel_type_class is None:
        class PixelType(object):
            '''Provide enums from ome.xml.model.enums.PixelType'''
            klass = jutil.get_env().find_class('ome/xml/model/enums/PixelType')
            INT8 = jutil.get_static_field(klass, 'INT8', 'Lome/xml/model/enums/PixelType;')
            INT16 = jutil.get_static_field(klass, 'INT16', 'Lome/xml/model/enums/PixelType;')
            INT32 = jutil.get_static_field(klass, 'INT32', 'Lome/xml/model/enums/PixelType;')
            UINT8 = jutil.get_static_field(klass, 'UINT8', 'Lome/xml/model/enums/PixelType;')
            UINT16 = jutil.get_static_field(klass, 'UINT16', 'Lome/xml/model/enums/PixelType;')
            UINT32 = jutil.get_static_field(klass, 'UINT32', 'Lome/xml/model/enums/PixelType;')
            FLOAT = jutil.get_static_field(klass, 'FLOAT', 'Lome/xml/model/enums/PixelType;')
            BIT = jutil.get_static_field(klass, 'BIT', 'Lome/xml/model/enums/PixelType;')
            DOUBLE = jutil.get_static_field(klass, 'DOUBLE', 'Lome/xml/model/enums/PixelType;')
            COMPLEX = jutil.get_static_field(klass, 'COMPLEX', 'Lome/xml/model/enums/PixelType;')
            DOUBLECOMPLEX = jutil.get_static_field(klass, 'DOUBLECOMPLEX', 'Lome/xml/model/enums/PixelType;')
        __pixel_type_class = PixelType()
    return __pixel_type_class

MINIMUM = 'MINIMUM'
NO_OVERLAYS = 'NO_OVERLAYS'
ALL = 'ALL'

def get_metadata_options(level):
    '''Get an instance of the MetadataOptions interface
    
    level - MINIMUM, NO_OVERLAYS or ALL to set the metadata retrieval level
    
    The object returned can be used in setMetadataOptions in a format reader.
    '''
    jlevel = jutil.get_static_field('loci/formats/in/MetadataLevel', level,
                                    'Lloci/formats/in/MetadataLevel;')
    return jutil.make_instance('loci/formats/in/DefaultMetadataOptions',
                               '(Lloci/formats/in/MetadataLevel;)V',
                               jlevel)
                               

def PositiveInteger(some_number):
    '''Return an instance of ome.xml.model.primitives.PositiveInteger
    
    some_number - the number to be wrapped up in the class
    '''
    return jutil.make_instance('ome/xml/model/primitives/PositiveInteger',
                               '(Ljava/lang/Integer;)V', some_number)

