''' metadatatools.py - mechanism to wrap some bioformats metadata classes

'''

__version__ = "$Revision: 1$"

import cellprofiler.utilities.jutil as jutil
import bioformats
import cellprofiler.utilities.javabridge as javabridge


def make_OMEXMLMetadata_class():
    class OMEXMLMetadata():
        pass

def make_metadata_tools_class():
    class MetadataTools(object):
        '''A wrapper for loci.formats.MetadataTools

        A utility class for working with metadata objects, including 
        MetadataStore, MetadataRetrieve, and OME-XML strings. Most of the 
        methods require the optional loci.formats.ome package, and optional 
        ome-xml.jar library, to be present at runtime.

        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/MetadataTools.html
        '''
        env = jutil.get_env()
        klass = env.find_class('loci/formats/MetadataTools')
        createOMEXMLMetadata = jutil.make_method('createOMEXMLMetadata',
                                                 '()Lloci/formats/meta/IMetadata;',
                                                 ' Creates an OME-XML metadata object using reflection, to avoid direct dependencies on the optional loci.formats.ome package.') 
    return MetadataTools

def make_metadata_store_class():
    class MetadataStore(object):
        '''
        A proxy whose responsibility it is to marshal biological image data 
        into a particular storage medium.

        The MetadataStore interface encompasses the metadata that any specific 
        storage medium (file, relational database, etc.) should be expected to 
        store into its backing data model.

        The MetadataStore interface goes hand in hand with the MetadataRetrieve 
        interface. Essentially, MetadataRetrieve provides the "getter" methods 
        for a storage medium, and MetadataStore provides the "setter" methods.

        Since it often makes sense for a storage medium to implement both 
        interfaces, there is also an IMetadata interface encompassing both 
        MetadataStore and MetadataRetrieve, which reduces the need to cast 
        between object types.

        See OMEXMLMetadata for an example implementation.

        Important note: It is strongly recommended that applications (e.g., 
        file format readers) using MetadataStore populate information in a 
        linear order. Specifically, iterating over entities from "leftmost" 
        index to "rightmost" index is required for certain MetadataStore 
        implementations such as OMERO's OMEROMetadataStore. For example, when 
        populating Image, Pixels and Plane information, an outer loop should 
        iterate across imageIndex, an inner loop should iterate across 
        pixelsIndex, and an innermost loop should handle planeIndex. For an 
        illustration of the ideal traversal order, see 
        MetadataConverter.convertMetadata(loci.formats.meta.MetadataRetrieve, 
                                          loci.formats.meta.MetadataStore).
        '''
        createRoot = jutil.make_method('createRoot', '()Ljava/lang/Object;', '')
        setPixelsBigEndian = jutil.make_method('setPixelsBigEndian', '(ZIII)V',
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
    return MetadataStore


def make_metadata_retrieve_class():
    class MetadataRetrieve(object):
        '''
        A proxy whose responsibility it is to extract biological image data 
        from a particular storage medium.

        The MetadataRetrieve interface encompasses the metadata that any 
        specific storage medium (file, relational database, etc.) should be 
        expected to access from its backing data model.

        The MetadataRetrieve interface goes hand in hand with the MetadataStore 
        interface. Essentially, MetadataRetrieve provides the "getter" methods 
        for a storage medium, and MetadataStore provides the "setter" methods.

        Since it often makes sense for a storage medium to implement both 
        interfaces, there is also an IMetadata interface encompassing both 
        MetadataStore and MetadataRetrieve, which reduces the need to cast 
        between object types.

        See OMEXMLMetadata for an example implementation.
        '''
        getPixelsBigEndian = jutil.make_method('getPixelsBigEndian', '(II)Z',
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
    return MetadataRetrieve
