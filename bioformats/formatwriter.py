'''formatwriter.py - mechanism to wrap a bioformats WriterWrapper and ImageWriter

The following file formats can be written using Bio-Formats:

- TIFF (uncompressed or LZW)
- OME-TIFF (uncompressed or LZW)
- JPEG
- PNG
- AVI (uncompressed)
- QuickTime (uncompressed is supported natively; additional codecs use QTJava)
- Encapsulated PostScript (EPS)

Support for OME-XML in the near future.

The writer API (see loci.formats.IFormatWriter) is very similar to the reader
API, in that files are written one plane at time (rather than all at once).

All writers allow the output file to be changed before the last plane has
been written.  This allows you to write to any number of output files using
the same writer and output settings (compression, frames per second, etc.),
and is especially useful for formats that do not support multiple images per
file.

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

import numpy as np
import os
import sys

import cellprofiler.utilities.jutil as jutil
import bioformats
import cellprofiler.utilities.javabridge as javabridge
import bioformats.omexml as ome


def write_image(pathname, pixels, pixel_type, 
                c = 0, z = 0, t = 0,
                size_c = 1, size_z = 1, size_t = 1,
                channel_names = None):
    """Write the image using bioformats
    
        filename - save to this filename
        
        pixels - the image to save
        
        pixel_type - save using this pixel type
        
        c - the image's channel index
        
        z - the image's z index
        
        t - the image's t index
        
        sizeC - # of channels in the stack
        
        sizeZ - # of z stacks
        
        sizeT - # of timepoints in the stack
        
        channel_names - names of the channels (make up names if not present
        """
    omexml = ome.OMEXML()
    omexml.image(0).Name = os.path.split(pathname)[1]
    p = omexml.image(0).Pixels
    assert isinstance(p, ome.OMEXML.Pixels)
    p.SizeX = pixels.shape[1]
    p.SizeY = pixels.shape[0]
    p.SizeC = size_c
    p.SizeT = size_t
    p.SizeZ = size_z
    p.DimensionOrder = ome.DO_XYCZT
    p.PixelType = pixel_type
    index = c + size_c * z + size_c * size_z * t
    if pixels.ndim == 3:
        p.SizeC = pixels.shape[2]
        p.Channel(0).SamplesPerPixel = pixels.shape[2]
        omexml.structured_annotations.add_original_metadata(
            ome.OM_SAMPLES_PER_PIXEL, str(pixels.shape[2]))
    elif size_c > 1:
        p.channel_count = size_c

    pixel_buffer = convert_pixels_to_buffer(pixels, pixel_type)
    xml = omexml.to_xml()
    script = """
    importClass(Packages.loci.formats.services.OMEXMLService,
                Packages.loci.common.services.ServiceFactory,
                Packages.loci.formats.ImageWriter);
    var service = new ServiceFactory().getInstance(OMEXMLService);
    var metadata = service.createOMEXMLMetadata(xml);
    var writer = new ImageWriter();
    writer.setMetadataRetrieve(metadata);
    writer.setId(path);
    writer.setInterleaved(true);
    writer.saveBytes(index, buffer);
    writer.close();
    """
    jutil.run_script(script,
                     dict(path=pathname,
                          xml=xml,
                          index=index,
                          buffer=pixel_buffer))
    
def convert_pixels_to_buffer(pixels, pixel_type):
    '''Convert the pixels in the image into a buffer of the right pixel type
    
    pixels - a 2d monochrome or color image
    
    pixel_type - one of the OME pixel types
    
    returns a 1-d byte array
    '''
    if pixel_type in (ome.PT_UINT8, ome.PT_INT8, ome.PT_BIT):
        as_dtype = np.uint8
    elif pixel_type in (ome.PT_UINT16, ome.PT_INT16):
        as_dtype = "<u2"
    elif pixel_type in (ome.PT_UINT32, ome.PT_INT32):
        as_dtype = "<u4"
    elif pixel_type == ome.PT_FLOAT:
        as_dtype = "<f4"
    elif pixel_type == ome.PT_DOUBLE:
        as_dtype = "<f8"
    else:
        raise NotImplementedError("Unsupported pixel type: %d" % pixel_type)
    buf = np.frombuffer(np.ascontiguousarray(pixels, as_dtype).data, np.uint8)
    env = jutil.get_env()
    return env.make_byte_array(buf)
        
def make_iformat_writer_class(class_name):
    '''Bind a Java class that implements IFormatWriter to a Python class
    
    Returns a class that implements IFormatWriter through calls to the
    implemented class passed in. The returned class can be subclassed to
    provide additional bindings.
    '''
    class IFormatWriter(object):
        '''A wrapper for loci.formats.IFormatWriter
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageWriter.html
        '''
        canDoStacks = jutil.make_method('canDoStacks', '()Z',
                                        'Reports whether the writer can save multiple images to a single file.')
        getColorModel = jutil.make_method('getColorModel', '()Ljava/awt/image/ColorModel;',
                                          'Gets the color model.')
        getCompression = jutil.make_method('getCompression', '()Ljava/lang/String;',
                                           'Gets the current compression type.')
        getCompressionTypes = jutil.make_method('getCompressionTypes', '()[Ljava/lang/String;',
                                                'Gets the available compression types.')
        getFramesPerSecond = jutil.make_method('getFramesPerSecond', '()I', 
                                               'Gets the frames per second to use when writing.')
        getMetadataRetrieve = jutil.make_method('getMetadataRetrieve', '()Lloci/formats/meta/MetadataRetrieve;',
                                                'Retrieves the current metadata retrieval object for this writer.')
        getPixelTypes = jutil.make_method('getPixelTypes', '()[I', 
                                          'Gets the supported pixel types.')
#        getPixelTypes = jutil.make_method('getPixelTypes', '(Ljava/lang/String;)[I', 
#                                          'Gets the supported pixel types for the given codec.')
        isInterleaved = jutil.make_method('isInterleaved', '()Z', 
                                          'Gets whether or not the channels in an image are interleaved.')
        isSupportedType = jutil.make_method('isSupportedType', '(I)Z', 
                                            'Checks if the given pixel type is supported.')
        saveBytes = jutil.make_method('saveBytes', '([BZ)V', 
                                      'Saves the given byte array to the current file.')
        saveBytesIB = jutil.make_method('saveBytes', '(I[B)V',
                                        'Saves bytes, first arg is image #')
#        saveBytes = jutil.make_method('saveBytes', '([BIZZ)V', 
#                                      'Saves the given byte array to the given series in the current file.')
        savePlane = jutil.make_method('savePlane', '(Ljava/lang/Object;Z)V', 
                                      'Saves the given image plane to the current file.')
#        savePlane = jutil.make_method('savePlane', '(Ljava/lang/Object;IZZ)V', 
#                                      'Saves the given image plane to the given series in the current file.')
        setColorModel = jutil.make_method('setColorModel', '(Ljava/awt/image/ColorModel;)V', 
                                          'Sets the color model.')
        setCompression = jutil.make_method('setCompression', '(Ljava/lang/String;)V', 
                                           'Sets the current compression type.')
        setFramesPerSecond = jutil.make_method('setFramesPerSecond', '(I)V', 
                                               'Sets the frames per second to use when writing.')
        setInterleaved = jutil.make_method('setInterleaved', '(Z)V', 
                                           'Sets whether or not the channels in an image are interleaved.')
        setMetadataRetrieve = jutil.make_method('setMetadataRetrieve', '(Lloci/formats/meta/MetadataRetrieve;)V', 
                                                'Sets the metadata retrieval object from which to retrieve standardized metadata.')
        setValidBitsPerPixel = jutil.make_method(
            'setValidBitsPerPixel', '(I)V',
            'Sets the number of valid bits per pixel')
        setSeries = jutil.make_method(
            'setSeries', '(I)V',
            '''Set the series for the image file
            
            series - the zero-based index of the image stack in the file,
            for instance in a multi-image tif.''')
        
    return IFormatWriter
    
def make_image_writer_class():
    '''Return an image writer class for the given Java environment'''
    env = jutil.get_env()
    class_name = 'loci/formats/ImageWriter'
    klass = env.find_class(class_name)
    base_klass = env.find_class('loci/formats/IFormatWriter')
    IFormatWriter = make_iformat_writer_class(class_name)
    #
    # This uses the writers.txt file from inside the loci_tools.jar
    #
    class_list = jutil.make_instance("loci/formats/ClassList", 
                                     "(Ljava/lang/String;"
                                     "Ljava/lang/Class;" # base
                                     "Ljava/lang/Class;)V", # location in jar
                                     "writers.txt", base_klass, klass)
    class ImageWriter(IFormatWriter):
        new_fn = jutil.make_new(class_name, '(Lloci/formats/ClassList;)V')
        def __init__(self):
            self.new_fn(class_list)
            
        setId = jutil.make_method('setId', '(Ljava/lang/String;)V', 
                                  'Sets the current file name.')
        addStatusListener = jutil.make_method('addStatusListener', '()Lloci/formats/StatusListener;',
                                              'Adds a listener for status update events.')
        close = jutil.make_method('close','()V',
                                  'Closes currently open file(s) and frees allocated memory.')
        getFormat = jutil.make_method('getFormat', '()Ljava/lang/String;', 
                                      'Gets the name of this file format.')
        getNativeDataType = jutil.make_method('getNativeDataType', '()Ljava/lang/Class;',
                                              'Returns the native data type of image planes for this reader, as returned by IFormatReader.openPlane(int, int, int, int, int) or IFormatWriter#saveData.')
        getStatusListeners = jutil.make_method('getStatusListeners', '()[Lloci/formats/StatusListener;',
                                               'Gets a list of all registered status update listeners.')
        getSuffixes = jutil.make_method('getSuffixes', '()Ljava/lang/String;', 
                                        'Gets the default file suffixes for this file format.')
        getWriter = jutil.make_method('getWriter', '()Lloci/formats/IFormatWriter;', 
                                      'Gets the writer used to save the current file.')
#        getWriter = jutil.make_method('getWriter', '(Ljava/lang/Class)Lloci/formats/IFormatWriter;', 
#                                      'Gets the file format writer instance matching the given class.')
#        getWriter = jutil.make_method('getWriter', '(Ljava/lang/String;)Lloci/formats/IFormatWriter;', 
#                                      'Gets the writer used to save the given file.')
        getWriters = jutil.make_method('getWriters', '()[Lloci/formats/IFormatWriter;', 
                                       'Gets all constituent file format writers.')
        isThisType = jutil.make_method('isThisType', '(Ljava/lang/String;)Z', 
                                       'Checks if the given string is a valid filename for this file format.')
        removeStatusListener = jutil.make_method('removeStatusListener', '(Lloci/formats/StatusListener;)V',
                                                 'Saves the given byte array to the current file.')
    return ImageWriter

def make_ome_tiff_writer_class():
    '''Return a class that wraps loci.formats.out.OMETiffWriter'''
    env = jutil.get_env()
    class_name = 'loci/formats/out/OMETiffWriter'
    klass = env.find_class(class_name)
    base_klass = env.find_class('loci/formats/IFormatWriter')
    IFormatWriter = make_iformat_writer_class(class_name)
    class OMETiffWriter(IFormatWriter):
        new_fn = jutil.make_new(class_name, '()V')
        def __init__(self):
            self.new_fn()
        setId = jutil.make_method('setId', '(Ljava/lang/String;)V', 
                                  'Sets the current file name.')
        close = jutil.make_method(
            'close','()V',
            'Closes currently open file(s) and frees allocated memory.')
        saveBytesIFD = jutil.make_method(
            'saveBytes', '(I[BLloci/formats/tiff/IFD;)V',
            '''save a byte array to an image channel
            
            index - image index
            bytes - byte array to save
            ifd - a loci.formats.tiff.IFD instance that gives all of the
                  IFD values associated with the channel''')
    return OMETiffWriter
    
def make_writer_wrapper_class(class_name):
    '''Make an ImageWriter wrapper class
    
    class_name - the name of the wrapper class
    
    You can instantiate an instance of the wrapper class like this:
    writer = XXX(ImageWriter())
    '''
    IFormatWriter = make_iformat_writer_class(class_name)
    class WriterWrapper(IFormatWriter):
        __doc__ = '''A wrapper for %s
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageWriter.html
        '''%class_name
        new_fn = jutil.make_new(class_name, '(Lloci/formats/IFormatWriter;)V')
        def __init__(self, writer):
            self.new_fn(writer)
            
        setId = jutil.make_method('setId', '(Ljava/lang/String;)V',
                                  'Sets the current file name.')
    return WriterWrapper


def make_format_writer_class(class_name):
    '''Make a FormatWriter wrapper class
    
    class_name - the name of a class that implements loci.formats.FormatWriter
                 Known names in the loci.formats.out package:
                     APNGWriter, AVIWriter, EPSWriter, ICSWriter, ImageIOWriter,
                     JPEG2000Writer, JPEGWriter, LegacyQTWriter, OMETiffWriter,
                     OMEXMLWriter, QTWriter, TiffWriter
    '''
    new_fn = jutil.make_new(class_name, 
                            '(Ljava/lang/String;Ljava/lang/String;)V')
    class FormatWriter(object):
        __doc__ = '''A wrapper for %s implementing loci.formats.FormatWriter
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/FormatWriter'''%class_name
        def __init__(self):
            self.new_fn()
            
        canDoStacks = jutil.make_method('canDoStacks','()Z',
                                        'Reports whether the writer can save multiple images to a single file')
        getColorModel = jutil.make_method('getColorModel',
                                          '()Ljava/awt/image/ColorModel;',
                                          'Gets the color model')
        getCompression = jutil.make_method('getCompression',
                                           '()Ljava/lang/String;',
                                           'Gets the current compression type')
        getCompressionTypes = jutil.make_method('getCompressionTypes',
                                                '()[Ljava/lang/String;',
                                                'Gets the available compression types')
        getFramesPerSecond = jutil.make_method('getFramesPerSecond',
                                               '()I', "Gets the frames per second to use when writing")
        getMetadataRetrieve = jutil.make_method('getMetadataRetrieve',
                                                '()Lloci/formats/meta/MetadataRetrieve;',
                                                'Retrieves the current metadata retrieval object for this writer.')
        
        getPixelTypes = jutil.make_method('getPixelTypes',
                                          '()[I')
        isInterleaved = jutil.make_method('isInterleaved','()Z',
                                          'Gets whether or not the channels in an image are interleaved')
        isSupportedType = jutil.make_method('isSupportedType','(I)Z',
                                            'Checks if the given pixel type is supported')
        saveBytes = jutil.make_method('saveBytes', '([BZ)V',
                                      'Saves the given byte array to the current file')
        setColorModel = jutil.make_method('setColorModel',
                                          '(Ljava/awt/image/ColorModel;)V',
                                          'Sets the color model')
        setCompression = jutil.make_method('setCompression',
                                           '(Ljava/lang/String;)V',
                                           'Sets the current compression type')
        setFramesPerSecond = jutil.make_method('setFramesPerSecond',
                                               '(I)V',
                                               'Sets the frames per second to use when writing')
        setId = jutil.make_method('setId','(Ljava/lang/String;)V',
                                  'Sets the current file name')
        setInterleaved = jutil.make_method('setInterleaved', '(Z)V',
                                           'Sets whether or not the channels in an image are interleaved')
        setMetadataRetrieve = jutil.make_method('setMetadataRetrieve',
                                                '(Lloci/formats/meta/MetadataRetrieve;)V',
                                                'Sets the metadata retrieval object from which to retrieve standardized metadata')
    return FormatWriter

def getRGBColorSpace():
    '''Get a Java object that represents an RGB color space
    
    See java.awt.color.ColorSpace: this returns the linear RGB color space
    '''
    cs_linear_rgb = jutil.get_static_field('java/awt/color/ColorSpace',
                                           'CS_LINEAR_RGB', 'I')
    return jutil.static_call('java/awt/color/ColorSpace', 'getInstance',
                             '(I)Ljava/awt/color/ColorSpace;',
                             cs_linear_rgb)

def getGrayColorSpace():
    '''Get a Java object that represents an RGB color space
    
    See java.awt.color.ColorSpace: this returns the linear RGB color space
    '''
    cs_gray = jutil.get_static_field('java/awt/color/ColorSpace',
                                           'CS_GRAY', 'I')
    return jutil.static_call('java/awt/color/ColorSpace', 'getInstance',
                             '(I)Ljava/awt/color/ColorSpace;',
                             cs_gray)

'''Constant for color model transparency indicating bitmask transparency'''
BITMASK = 'BITMASK'
'''Constant for color model transparency indicting an opaque color model'''
OPAQUE = 'OPAQUE'
'''Constant for color model transparency indicating a transparent color model'''
TRANSPARENT = 'TRANSPARENT'
'''Constant for color model transfer type indicating byte per pixel'''
TYPE_BYTE = 'TYPE_BYTE'
'''Constant for color model transfer type indicating unsigned short per pixel'''
TYPE_USHORT = 'TYPE_USHORT'
'''Constant for color model transfer type indicating integer per pixel'''
TYPE_INT = 'TYPE_INT'

def getColorModel(color_space, 
                  has_alpha=False, 
                  is_alpha_premultiplied = False,
                  transparency = OPAQUE,
                  transfer_type = TYPE_BYTE):
    '''Return a java.awt.image.ColorModel color model
    
    color_space - a java.awt.color.ColorSpace such as returned by
    getGrayColorSpace or getRGBColorSpace
    
    has_alpha - True if alpha channel is specified
    
    is_alpha_premultiplied - True if other channel values have already
    been reduced by the alpha multiplier, False if the channel values are
    independent of the multiplier.
    
    transparency - one of BITMASK, OPAQUE or TRANSPARENT.
    
    transfer_type - one of TYPE_BYTE, TYPE_USHORT, TYPE_INT
    '''
    jtransparency = jutil.get_static_field('java/awt/Transparency',
                                           transparency,
                                           'I')
    jtransfer_type = jutil.get_static_field('java/awt/image/DataBuffer',
                                            transfer_type, 'I')
    return jutil.make_instance('java/awt/image/ComponentColorModel',
                               '(Ljava/awt/color/ColorSpace;ZZII)V',
                               color_space, has_alpha, is_alpha_premultiplied,
                               jtransparency, jtransfer_type)
if __name__ == "__main__":
    import wx
    import matplotlib.backends.backend_wxagg as mmmm
    import bioformats
    from formatreader import *
    from metadatatools import *
    
    app = wx.PySimpleApp()

#    dlg = wx.FileDialog(None)
#    if dlg.ShowModal()==wx.ID_OK:
#        filename = dlg.Path
#    else:
#        app.Exit()
#        sys.exit()

    filename = '/Users/afraser/Desktop/cpa_example/images/AS_09125_050116000001_A01f00d0.png'
    filename = '/Users/afraser/Desktop/wedding/header.jpg'

    out_file = '/Users/afraser/Desktop/test_output.avi'
    try:
        os.remove(out_file)
        print 'previous output file deleted'
    except:
        print 'no output file to delete'
    
    env = jutil.attach()
    ImageReader = make_image_reader_class()
    ChannelSeparator = make_reader_wrapper_class("loci/formats/ChannelSeparator")
    FormatTools = make_format_tools_class()

    # writer testing
    ImageWriter = make_image_writer_class()
    writer = ImageWriter()
    
    w = 400
    h = 400
    c = 3
    z = 1
    t = 4
    images = []
    for tt in range(t):
        images += [(np.random.rand(w, h, c) * 255).astype('uint8')]
                
    imeta = createOMEXMLMetadata()
    meta = wrap_imetadata_object(imeta)
    meta.createRoot()
    meta.setPixelsBigEndian(True, 0, 0)
    meta.setPixelsDimensionOrder('XYCZT', 0, 0)
    meta.setPixelsPixelType(FormatTools.getPixelTypeString(FormatTools.UINT8), 0, 0)
    meta.setPixelsSizeX(w, 0, 0)
    meta.setPixelsSizeY(h, 0, 0)
    meta.setPixelsSizeC(c, 0, 0)
    meta.setPixelsSizeZ(z, 0, 0)
    meta.setPixelsSizeT(t, 0, 0)
    meta.setLogicalChannelSamplesPerPixel(c, 0, 0)
    
    print 'big endian:', meta.getPixelsBigEndian(0, 0)
    print 'dim order:', meta.getPixelsDimensionOrder(0, 0)
    print 'pixel type:', meta.getPixelsPixelType(0, 0)
    print 'size x:', meta.getPixelsSizeX(0, 0)
    print 'size y:', meta.getPixelsSizeY(0, 0)
    print 'size c:', meta.getPixelsSizeC(0, 0)
    print 'size z:', meta.getPixelsSizeZ(0, 0)
    print 'size t:', meta.getPixelsSizeT(0, 0)
    print 'samples per pixel:', meta.getLogicalChannelSamplesPerPixel(0, 0)

    writer.setMetadataRetrieve(meta)
    writer.setId(out_file)
    for image in images:
        if len(image.shape)==3 and image.shape[2] == 3:  
            save_im = np.array([image[:,:,0], image[:,:,1], image[:,:,2]]).astype(np.uint8).flatten()
        else:
            save_im = image.astype(np.uint8).flatten()
        writer.saveBytes(env.make_byte_array(save_im), (image is images[-1]))
    writer.close()
    
    print 'Done writing image :)'
#    import PIL.Image as Image
#    im = Image.open(out_file, 'r')
#    im.show()
    
    jutil.detach()
    app.MainLoop()
    
    
