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
'''

__version__ = "$Revision: 1$"

import numpy as np
import os
import sys

import cellprofiler.utilities.jutil as jutil
import bioformats
import cellprofiler.utilities.javabridge as javabridge


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
#        saveBytes = jutil.make_method('saveBytes', '([BZ)V', 
#                                      'Saves the given byte array to the current file.')
        saveBytes = jutil.make_method('saveBytes', '([BIZZ)V', 
                                      'Saves the given byte array to the given series in the current file.')
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
        
if __name__ == "__main__":
    import wx
    import matplotlib.backends.backend_wxagg as mmmm
    import bioformats
    from formatreader import *
    from metadatatools import *
    
    env = jutil.attach()
    ImageReader = make_image_reader_class()
    ChannelSeparator = make_reader_wrapper_class("loci/formats/ChannelSeparator")
    FormatTools = make_format_tools_class()
    class MyApp(wx.App):
        def OnInit(self):
            self.PrintMode = 0
            dlg = wx.FileDialog(None)
            if dlg.ShowModal()==wx.ID_OK:
                rdr = ImageReader()
                filename = dlg.Path
                print filename
                rdr.setId(filename)
                print "Format = %s"%rdr.getFormat()
                w = rdr.getSizeX()
                h = rdr.getSizeY()
                pixel_type = rdr.getPixelType()
                little_endian = rdr.isLittleEndian()
                metadata = rdr.getMetadata()
                d = jutil.jdictionary_to_string_dictionary(metadata)
                for key in d.keys():
                    print key+"="+d[key]
                if pixel_type == FormatTools.INT8:
                    dtype = np.char
                elif pixel_type == FormatTools.UINT8:
                    dtype = np.uint8
                elif pixel_type == FormatTools.UINT16:
                    dtype = '<u2' if little_endian else '>u2'
                elif pixel_type == FormatTools.INT16:
                    dtype = '<i2' if little_endian else '>i2'
                elif pixel_type == FormatTools.UINT32:
                    dtype = '<u4' if little_endian else '>u4'
                elif pixel_type == FormatTools.INT32:
                    dtype = '<i4' if little_endian else '>i4'
                elif pixel_type == FormatTools.FLOAT:
                    dtype = '<f4' if little_endian else '>f4'
                elif pixel_type == FormatTools.DOUBLE:
                    dtype = '<f8' if little_endian else '>f8'
                    
                if rdr.getRGBChannelCount() > 1:
                    rdr.close()
                    rdr = ChannelSeparator(ImageReader())
                    rdr.setId(dlg.Path)
                    red_image, green_image, blue_image = [
                        np.frombuffer(rdr.openBytes(rdr.getIndex(0,i,0)),dtype)
                        for i in range(3)]
                    image = np.dstack((red_image, green_image, blue_image))
                    image.shape=(h,w,3)
                else:
                    image = np.frombuffer(rdr.openBytes(0),dtype)
                    image.shape = (h,w)
                rdr.close()
                
                
                # writer testing
                ImageWriter = make_image_writer_class()
                writer = ImageWriter()
                
                MetadataTools = make_metadata_tools_class()
                meta = jutil.static_call('loci/formats/MetadataTools', 'createOMEXMLMetadata', '()Lloci/formats/meta/IMetadata;')
                print meta
                jutil.call(meta, 'createRoot', '()V')
                t = jutil.make_instance('java/lang/Boolean', '(Z)V', True)
                jutil.call(meta, 'setPixelsBigEndian', '(Ljava/lang/Boolean;II)V', t, 0, 0)
#                meta = MetadataTools.createOMEXMLMetadata()
                
#                meta.createRoot()
#                meta.setPixelsBigEndian(True, 0, 0)
#                meta.setPixelsDimensionOrder("XYZCT", 0, 0)
#                meta.setPixelsPixelType(FormatTools.getPixelTypeString(pixelType), 0, 0)
#                meta.setPixelsSizeX(w, 0, 0)
#                meta.setPixelsSizeY(h, 0, 0)
#                meta.setPixelsSizeZ(1, 0, 0)
#                meta.setPixelsSizeC(3, 0, 0)
#                meta.setPixelsSizeT(1, 0, 0)
                
                writer.setId('/Users/afraser/Desktop/test_output.tiff')
                for i in range(image.shape[0]):
                    writer.saveBytes(env.make_byte_array(image[i].astype(np.uint8)), 0, True, True)
                writer.close()
                
                
                fig = mmmm.Figure()
                axes = fig.add_subplot(1,1,1)
                axes.imshow(image)
                frame = mmmm.FigureFrameWxAgg(1,fig)
                frame.Show()
                jutil.detach()
                return True
            jutil.detach()
            return False
    app = MyApp(0)
    app.MainLoop()
    
    