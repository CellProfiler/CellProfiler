'''formatreader.py - mechanism to wrap a bioformats ReaderWrapper and ImageReader

Example:
    import bioformats.formatreader as biordr
    
    env = biordr.get_env()

    ChannelSeparator = biordr.make_reader_wrapper_class(env, 'loci/formats/ChannelSeparator')
    ImageReader = biordr.make_image_reader_class(env)

    cs = ChannelSeparator(ImageReader('/path/to/file.tif'))

    my_red_image, my_green_image, my_blue_image = \
        [cs.open_bytes(cs.getIndex(0,i,0)) for i in range(3)]
'''
    
__version__ = "$Revision: 1$"

import numpy as np
import os

import cellprofiler.utilities.jutil as jutil
import cellprofiler.utilities.javabridge as javabridge

__env = None
def get_env():
    '''Get a Java environment with the loci_tools jar on the classpath'''
    global __env
    
    if __env is not None:
        return __env
    path = os.path.abspath(os.path.split(__file__)[0])
    loci_jar = os.path.join(path, "loci_tools.jar")
    __env = javabridge.JB_Env()
    __env.create(["-Djava.class.path="+loci_jar,
                  "-Djava.awt.headless=true"])
    return __env

def make_format_tools_class(env):
    '''Get a wrapper for the loci/formats/FormatTools class
    
    The FormatTools class has many of the constants needed by
    other classes as statics.
    '''
    class FormatTools(object):
        '''A wrapper for loci.formats.FormatTools
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/FormatTools.html
        '''
        klass = env.find_class('loci/formats/FormatTools')
        CAN_GROUP = jutil.get_static_field(env, klass, 'CAN_GROUP','I')
        CANNOT_GROUP = jutil.get_static_field(env, klass, 'CANNOT_GROUP','I')
        DOUBLE = jutil.get_static_field(env, klass, 'DOUBLE','I')
        FLOAT = jutil.get_static_field(env, klass, 'FLOAT', 'I')
        INT16 = jutil.get_static_field(env, klass, 'INT16', 'I')
        INT32 = jutil.get_static_field(env, klass, 'INT32', 'I')
        INT8 = jutil.get_static_field(env, klass, 'INT8', 'I')
        MUST_GROUP = jutil.get_static_field(env, klass, 'MUST_GROUP', 'I')
        UINT16 = jutil.get_static_field(env, klass, 'UINT16', 'I')
        UINT32 = jutil.get_static_field(env, klass, 'UINT32', 'I')
        UINT8 = jutil.get_static_field(env, klass, 'UINT8', 'I')
    return FormatTools

def make_iformat_reader_class(env, klass_arg):
    '''Bind a Java class that implements IFormatReader to a Python class
    
    Returns a class that implements IFormatReader through calls to the
    implemented class passed in. The returned class can be subclassed to
    provide additional bindings.
    '''
    class IFormatReader(object):
        '''A wrapper for loci.formats.IFormatReader
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
        '''
        klass = klass_arg
        close = jutil.make_method(env, klass, 'close','()V',
                                  'Close the currently open file and free memory')
        getDimensionOrder = jutil.make_method(env, klass, 'getDimensionOrder',
                                              '()Ljava/lang/String;',
                                              'Return the dimension order as a five-character string, e.g. "XYCZT"')
        getMetadata = jutil.make_method(env, klass, 'getMetadata',
                                              '()Ljava/util/Hashtable;',
                                              'Obtains the hashtable containing the metadata field/value pairs')
        getMetadataValue = jutil.make_method(env, klass, 'getMetadataValue',
                                             '(Ljava/lang/String;)'
                                             'Ljava/lang/Object;',
                                             'Look up a specific metadata value from the store')
        getImageCount = jutil.make_method(env, klass, 'getImageCount',
                                          '()I','Determines the number of images in the current file')
        getIndex = jutil.make_method(env, klass, 'getIndex', '(III)I',
                                     'Get the plane index given z, c, t')
        getRGBChannelCount = jutil.make_method(env, klass, 'getRGBChannelCount',
                                               '()I','Gets the number of channels per RGB image (if not RGB, this returns 1')
        getSizeC = jutil.make_method(env, klass, 'getSizeC', '()I',
                                     'Get the number of color planes')
        getSizeT = jutil.make_method(env, klass, 'getSizeT', '()I',
                                     'Get the number of frames in the image')
        getSizeX = jutil.make_method(env, klass, 'getSizeX', '()I',
                                     'Get the image width')
        getSizeY = jutil.make_method(env, klass, 'getSizeY', '()I',
                                     'Get the image height')
        getSizeZ = jutil.make_method(env, klass, 'getSizeZ', '()I',
                                     'Get the image depth')
        getPixelType = jutil.make_method(env, klass, 'getPixelType', '()I',
                                         'Get the pixel type: see FormatTools for types')
        isLittleEndian = jutil.make_method(env, klass, 'isLittleEndian',
                                           '()Z','Return True if the data is in little endian order')
        isRGB = jutil.make_method(env, klass, 'isRGB', '()Z',
                                  'Return True if images in the file are RGB')
        isInterleaved = jutil.make_method(env, klass, 'isInterleaved', '()Z',
                                          'Return True if image colors are interleaved within a plane')
        openBytes = jutil.make_method(env, klass, 'openBytes','(I)[B',
                                      'Get the specified image plane as a byte array')
        openBytesXYWH = jutil.make_method(env, klass, 'openBytes','(IIIII)[B',
                                          '''Get the specified image plane as a byte array
                                          
                                          (corresponds to openBytes(int no, int x, int y, int w, int h))
                                          no - image plane number
                                          x,y - offset into image
                                          w,h - dimensions of image to return''')
    return IFormatReader
    
def make_image_reader_class(env):
    '''Return an image reader class for the given Java environment'''
    klass = env.find_class('loci/formats/ImageReader')
    base_klass = env.find_class('loci/formats/IFormatReader')
    assert klass is not None
    assert base_klass is not None
    IFormatReader = make_iformat_reader_class(env, klass)
    #
    # This uses the reader.txt file from inside the loci_tools.jar
    #
    class_list = jutil.make_instance(env, "loci/formats/ClassList", 
                                     "(Ljava/lang/String;"
                                     "Ljava/lang/Class;" # base
                                     "Ljava/lang/Class;)V", # location in jar
                                     "readers.txt", base_klass, klass)
    class ImageReader(IFormatReader):
        new_fn = jutil.make_new(env, klass, '(Lloci/formats/ClassList;)V')
        def __init__(self):
            self.new_fn(class_list)
        setId = jutil.make_method(env, klass, 'setId', '(Ljava/lang/String;)V',
                                  'Set the name of the data file')
        getFormat = jutil.make_method(env, klass, 'getFormat',
                                      '()Ljava/lang/String;',
                                      'Get a string describing the format of this file')
        getReader = jutil.make_method(env, klass, 'getReader',
                                      '()Lloci/formats/IFormatReader;')
    return ImageReader

        
def make_reader_wrapper_class(env, class_name):
    '''Make an ImageReader wrapper class
    
    class_name - the name of the wrapper class, for instance, 
                 "loci/formats/ChannelSeparator"
    
    You can instantiate an instance of the wrapper class like this:
    rdr = ChannelSeparator(ImageReader())
    '''
    klass = env.find_class(class_name)
    assert klass is not None
    IFormatReader = make_iformat_reader_class(env, klass)
    class ReaderWrapper(IFormatReader):
        __doc__ = '''A wrapper for %s
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
        '''%class_name
        new_fn = jutil.make_new(env, klass, '(Lloci/formats/IFormatReader;)V')
        def __init__(self, rdr):
            self.new_fn(rdr)
            
        setId = jutil.make_method(env, klass, 'setId', '(Ljava/lang/String;)V',
                                  'Set the name of the data file')
    return ReaderWrapper

def make_format_writer_class(env, class_name):
    '''Make a FormatWriter wrapper class
    
    class_name - the name of a class that implements loci.formats.FormatWriter
                 Known names in the loci.formats.out package:
                     APNGWriter, AVIWriter, EPSWriter, ICSWriter, ImageIOWriter,
                     JPEG2000Writer, JPEGWriter, LegacyQTWriter, OMETiffWriter,
                     OMEXMLWriter, QTWriter, TiffWriter
    '''
    klass = env.find_class(class_name)
    assert klass is not None
    new_fn = jutil.make_new(env, klass, 
                            '(Ljava/lang/String;Ljava/lang/String;)V')
    class FormatWriter(object):
        __doc__ = '''A wrapper for %s implementing loci.formats.FormatWriter
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/FormatWriter'''%class_name
        def __init__(self):
            self.new_fn()
            
        canDoStacks = jutil.make_method(env, klass, 'canDoStacks','()Z',
                                        'Reports whether the writer can save multiple images to a single file')
        getColorModel = jutil.make_method(env, klass, 'getColorModel',
                                          '()Ljava/awt/image/ColorModel;',
                                          'Gets the color model')
        getCompression = jutil.make_method(env, klass, 'getCompression',
                                           '()Ljava/lang/String;',
                                           'Gets the current compression type')
        getCompressionTypes = jutil.make_method(env, klass, 'getCompressionTypes',
                                                '()[Ljava/lang/String;',
                                                'Gets the available compression types')
        getFramesPerSecond = jutil.make_method(env, klass, 'getFramesPerSecond',
                                               '()I', "Gets the frames per second to use when writing")
        getMetadataRetrieve = jutil.make_method(env, klass, 'getMetadataRetrieve',
                                                '()Lloci/formats/meta/MetadataRetrieve;',
                                                'Retrieves the current metadata retrieval object for this writer.')
        
        getPixelTypes = jutil.make_method(env, klass, 'getPixelTypes',
                                          '()[I')
        isInterleaved = jutil.make_method(env, klass, 'isInterleaved','()Z',
                                          'Gets whether or not the channels in an image are interleaved')
        isSupportedType = jutil.make_method(env, klass, 'isSupportedType','(I)Z',
                                            'Checks if the given pixel type is supported')
        saveBytes = jutil.make_method(env, klass, 'saveBytes', '([BZ)V',
                                      'Saves the given byte array to the current file')
        setColorModel = jutil.make_method(env, klass, 'setColorModel',
                                          '(Ljava/awt/image/ColorModel;)V',
                                          'Sets the color model')
        setCompression = jutil.make_method(env, klass, 'setCompression',
                                           '(Ljava/lang/String;)V',
                                           'Sets the current compression type')
        setFramesPerSecond = jutil.make_method(env, klass, 'setFramesPerSecond',
                                               '(I)V',
                                               'Sets the frames per second to use when writing')
        setId = jutil.make_method(env, klass, 'setId','(Ljava/lang/String;)V',
                                  'Sets the current file name')
        setInterleaved = jutil.make_method(env, klass, 'setInterleaved', '(Z)V',
                                           'Sets whether or not the channels in an image are interleaved')
        setMetadataRetrieve = jutil.make_method(env, klass, 'setMetadataRetrieve',
                                                '(Lloci/formats/meta/MetadataRetrieve;)V',
                                                'Sets the metadata retrieval object from which to retrieve standardized metadata')
    return FormatWriter
        
if __name__ == "__main__":
    import wx
    import matplotlib.backends.backend_wxagg as mmmm
    
    my_env = get_env()
    ImageReader = make_image_reader_class(my_env)
    ChannelSeparator = make_reader_wrapper_class(my_env, "loci/formats/ChannelSeparator")
    FormatTools = make_format_tools_class(my_env)
    class MyApp(wx.App):
        def OnInit(self):
            self.PrintMode = 0
            dlg = wx.FileDialog(None)
            if dlg.ShowModal()==wx.ID_OK:
                rdr = ImageReader()
                rdr.setId(dlg.Path)
                print "Format = %s"%rdr.getFormat()
                w = rdr.getSizeX()
                h = rdr.getSizeY()
                pixel_type = rdr.getPixelType()
                little_endian = rdr.isLittleEndian()
                metadata = rdr.getMetadata()
                d = jutil.jdictionary_to_string_dictionary(my_env, metadata)
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
                fig = mmmm.Figure()
                axes = fig.add_subplot(1,1,1)
                axes.imshow(image)
                frame = mmmm.FigureFrameWxAgg(1,fig)
                frame.Show()
                return True
            return False
    app = MyApp(0)
    app.MainLoop()
    
    