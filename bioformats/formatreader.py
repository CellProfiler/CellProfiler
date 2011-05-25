'''formatreader.py - mechanism to wrap a bioformats ReaderWrapper and ImageReader

Example:
    import bioformats.formatreader as biordr
    
    env = biordr.get_env()

    ChannelSeparator = biordr.make_reader_wrapper_class(env, 'loci/formats/ChannelSeparator')
    ImageReader = biordr.make_image_reader_class(env)

    cs = ChannelSeparator(ImageReader('/path/to/file.tif'))

    my_red_image, my_green_image, my_blue_image = \
        [cs.open_bytes(cs.getIndex(0,i,0)) for i in range(3)]

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org'
'''
    
__version__ = "$Revision$"

import numpy as np
import os
import sys

import cellprofiler.utilities.jutil as jutil
import bioformats
import cellprofiler.utilities.javabridge as javabridge

def make_format_tools_class():
    '''Get a wrapper for the loci/formats/FormatTools class
    
    The FormatTools class has many of the constants needed by
    other classes as statics.
    '''
    class FormatTools(object):
        '''A wrapper for loci.formats.FormatTools
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/FormatTools.html
        '''
        env = jutil.get_env()
        klass = env.find_class('loci/formats/FormatTools')
        CAN_GROUP = jutil.get_static_field(klass, 'CAN_GROUP','I')
        CANNOT_GROUP = jutil.get_static_field(klass, 'CANNOT_GROUP','I')
        DOUBLE = jutil.get_static_field(klass, 'DOUBLE','I')
        FLOAT = jutil.get_static_field(klass, 'FLOAT', 'I')
        INT16 = jutil.get_static_field(klass, 'INT16', 'I')
        INT32 = jutil.get_static_field(klass, 'INT32', 'I')
        INT8 = jutil.get_static_field(klass, 'INT8', 'I')
        MUST_GROUP = jutil.get_static_field(klass, 'MUST_GROUP', 'I')
        UINT16 = jutil.get_static_field(klass, 'UINT16', 'I')
        UINT32 = jutil.get_static_field(klass, 'UINT32', 'I')
        UINT8 = jutil.get_static_field(klass, 'UINT8', 'I')
        
        @classmethod
        def getPixelTypeString(cls, pixel_type):
            return jutil.static_call('loci/formats/FormatTools', 'getPixelTypeString', '(I)Ljava/lang/String;', pixel_type)
        
    return FormatTools

def make_iformat_reader_class():
    '''Bind a Java class that implements IFormatReader to a Python class
    
    Returns a class that implements IFormatReader through calls to the
    implemented class passed in. The returned class can be subclassed to
    provide additional bindings.
    '''
    class IFormatReader(object):
        '''A wrapper for loci.formats.IFormatReader
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
        '''
        close = jutil.make_method('close','()V',
                                  'Close the currently open file and free memory')
        getDimensionOrder = jutil.make_method('getDimensionOrder',
                                              '()Ljava/lang/String;',
                                              'Return the dimension order as a five-character string, e.g. "XYCZT"')
        getMetadata = jutil.make_method('getMetadata',
                                        '()Ljava/util/Hashtable;',
                                        'Obtains the hashtable containing the metadata field/value pairs')
        getMetadataValue = jutil.make_method('getMetadataValue',
                                             '(Ljava/lang/String;)'
                                             'Ljava/lang/Object;',
                                             'Look up a specific metadata value from the store')
        getSeriesMetadata = jutil.make_method('getSeriesMetadata',
                                              '()Ljava/util/Hashtable;',
                                              'Obtains the hashtable contaning the series metadata field/value pairs')
        getSeriesCount = jutil.make_method('getSeriesCount',
                                           '()I',
                                           'Return the # of image series in the file')
        getSeries = jutil.make_method('getSeries', '()I',
                                      'Return the currently selected image series')
        getImageCount = jutil.make_method('getImageCount',
                                          '()I','Determines the number of images in the current file')
        getIndex = jutil.make_method('getIndex', '(III)I',
                                     'Get the plane index given z, c, t')
        getRGBChannelCount = jutil.make_method('getRGBChannelCount',
                                               '()I','Gets the number of channels per RGB image (if not RGB, this returns 1')
        getSizeC = jutil.make_method('getSizeC', '()I',
                                     'Get the number of color planes')
        getSizeT = jutil.make_method('getSizeT', '()I',
                                     'Get the number of frames in the image')
        getSizeX = jutil.make_method('getSizeX', '()I',
                                     'Get the image width')
        getSizeY = jutil.make_method('getSizeY', '()I',
                                     'Get the image height')
        getSizeZ = jutil.make_method('getSizeZ', '()I',
                                     'Get the image depth')
        getPixelType = jutil.make_method('getPixelType', '()I',
                                         'Get the pixel type: see FormatTools for types')
        isLittleEndian = jutil.make_method('isLittleEndian',
                                           '()Z','Return True if the data is in little endian order')
        isRGB = jutil.make_method('isRGB', '()Z',
                                  'Return True if images in the file are RGB')
        isInterleaved = jutil.make_method('isInterleaved', '()Z',
                                          'Return True if image colors are interleaved within a plane')
        openBytes = jutil.make_method('openBytes','(I)[B',
                                      'Get the specified image plane as a byte array')
        openBytesXYWH = jutil.make_method('openBytes','(IIIII)[B',
                                          '''Get the specified image plane as a byte array
                                          
                                          (corresponds to openBytes(int no, int x, int y, int w, int h))
                                          no - image plane number
                                          x,y - offset into image
                                          w,h - dimensions of image to return''')
        setSeries = jutil.make_method('setSeries','(I)V','Set the currently selected image series')
        setGroupFiles = jutil.make_method('setGroupFiles', '(Z)V', 
                                          'Force reader to group or not to group files in a multi-file set')
        setMetadataStore = jutil.make_method('setMetadataStore',
                                             '(Lloci/formats/meta/MetadataStore;)V',
                                             'Sets the default metadata store for this reader.')
        setMetadataOptions = jutil.make_method('setMetadataOptions',
                                               '(Lloci/formats/in/MetadataOptions;)V',
                                               'Sets the metadata options used when reading metadata')
        isThisTypeS = jutil.make_method(
            'isThisType',
            '(Ljava/lang/String;)Z',
            'Return true if the filename might be handled by this reader')
        isThisTypeSZ = jutil.make_method(
            'isThisType',
            '(Ljava/lang/String;Z)Z',
            '''Return true if the named file is handled by this reader.
            
            filename - name of file
            
            allowOpen - True if the reader is allowed to open files
                        when making its determination
            ''')
        isThisTypeStream = jutil.make_method(
            'isThisType',
            '(Lloci/common/RandomAccessInputStream;)Z',
            '''Return true if the stream might be parseable by this reader.
            
            stream - the RandomAccessInputStream to be used to read the file contents
            
            Note that both isThisTypeS and isThisTypeStream must return true
            for the type to truly be handled.''')
        setId = jutil.make_method('setId', '(Ljava/lang/String;)V',
                                  'Set the name of the data file')
        getMetadataStore = jutil.make_method('getMetadataStore', '()Lloci/formats/meta/MetadataStore;',
                                             'Retrieves the current metadata store for this reader.')
        def get_class_name(self):
            return jutil.call(jutil.call(self.o, 'getClass', '()Ljava/lang/Class;'),
                              'getName', '()Ljava/lang/String;')
        
        @property
        def suffixNecessary(self):
            if self.get_class_name() == 'loci.formats.in.JPKReader':
                return True;
            env = jutil.get_env()
            klass = env.get_object_class(self.o)
            field_id = env.get_field_id(klass, "suffixNecessary", "Z")
            if field_id is None:
                return None
            return env.get_boolean_field(self.o, field_id)
            
        @property
        def suffixSufficient(self):
            if self.get_class_name() == 'loci.formats.in.JPKReader':
                return True;
            env = jutil.get_env()
            klass = env.get_object_class(self.o)
            field_id = env.get_field_id(klass, "suffixSufficient", "Z")
            if field_id is None:
                return None
            return env.get_boolean_field(self.o, field_id)
            
            
    return IFormatReader

def get_class_list():
    '''Return a wrapped instance of loci.formats.ClassList'''
    #
    # This uses the reader.txt file from inside the loci_tools.jar
    #
    class ClassList(object):
        remove_class = jutil.make_method(
            'removeClass', '(Ljava/lang/Class;)V',
            'Remove the given class from the class list')
        add_class = jutil.make_method(
            'addClass', '(Ljava/lang/Class;)V',
            'Add the given class to the back of the class list')
        get_classes = jutil.make_method(
            'getClasses', '()[Ljava/lang/Class;',
            'Get the classes in the list as an array')
            
        def __init__(self):
            env = jutil.get_env()
            class_name = 'loci/formats/ImageReader'
            klass = env.find_class(class_name)
            base_klass = env.find_class('loci/formats/IFormatReader')
            self.o = jutil.make_instance("loci/formats/ClassList", 
                                         "(Ljava/lang/String;"
                                         "Ljava/lang/Class;" # base
                                         "Ljava/lang/Class;)V", # location in jar
                                         "readers.txt", base_klass, klass)
            problem_classes = [
                # BDReader will read all .tif files in an experiment if it's
                # called to load a .tif.
                #
                'loci.formats.in.BDReader'
                ]
            for problem_class in problem_classes:
                # Move to back
                klass = jutil.class_for_name(problem_class)
                self.remove_class(klass)
                self.add_class(klass)
    return ClassList()
    
    
def make_image_reader_class():
    '''Return an image reader class for the given Java environment'''
    env = jutil.get_env()
    class_name = 'loci/formats/ImageReader'
    klass = env.find_class(class_name)
    base_klass = env.find_class('loci/formats/IFormatReader')
    IFormatReader = make_iformat_reader_class()
    class_list = get_class_list()
        
    class ImageReader(IFormatReader):
        new_fn = jutil.make_new(class_name, '(Lloci/formats/ClassList;)V')
        def __init__(self):
            self.new_fn(class_list.o)
        getFormat = jutil.make_method('getFormat',
                                      '()Ljava/lang/String;',
                                      'Get a string describing the format of this file')
        getReader = jutil.make_method('getReader',
                                      '()Lloci/formats/IFormatReader;')
        def allowOpenToCheckType(self, allow):
            '''Allow the "isThisType" function to open files
            
            For the cluster, you want to tell potential file formats
            not to open the image file to test if it's their format.
            '''
            if not hasattr(self, "allowOpenToCheckType_method"):
                self.allowOpenToCheckType_method = None
                class_wrapper = jutil.get_class_wrapper(self.o)
                methods = class_wrapper.getMethods()
                for method in jutil.get_env().get_object_array_elements(methods):
                    m = jutil.get_method_wrapper(method)
                    if m.getName() in ('allowOpenToCheckType', 'setAllowOpenFiles'):
                        self.allowOpenToCheckType_method = m
            if self.allowOpenToCheckType_method is not None:
                object_class = env.find_class('java/lang/Object')
                jexception = jutil.get_env().exception_occurred()
                if jexception is not None:
                    raise jutil.JavaException(jexception)
                
                boolean_value = jutil.make_instance('java/lang/Boolean', 
                                                    '(Z)V', allow)
                args = jutil.get_env().make_object_array(1, object_class)
                jexception = jutil.get_env().exception_occurred()
                if jexception is not None:
                    raise jutil.JavaException(jexception)
                jutil.get_env().set_object_array_element(args, 0, boolean_value)
                jexception = jutil.get_env().exception_occurred()
                if jexception is not None:
                    raise jutil.JavaException(jexception)
                self.allowOpenToCheckType_method.invoke(self.o, args)
    return ImageReader

        
def make_reader_wrapper_class(class_name):
    '''Make an ImageReader wrapper class
    
    class_name - the name of the wrapper class, for instance, 
                 "loci/formats/ChannelSeparator"
    
    You can instantiate an instance of the wrapper class like this:
    rdr = ChannelSeparator(ImageReader())
    '''
    IFormatReader = make_iformat_reader_class()
    class ReaderWrapper(IFormatReader):
        __doc__ = '''A wrapper for %s
        
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
        '''%class_name
        new_fn = jutil.make_new(class_name, '(Lloci/formats/IFormatReader;)V')
        def __init__(self, rdr):
            self.new_fn(rdr)
            
        setId = jutil.make_method('setId', '(Ljava/lang/String;)V',
                                  'Set the name of the data file')
    return ReaderWrapper

        
if __name__ == "__main__":
    import wx
    import matplotlib.backends.backend_wxagg as mmmm
    import bioformats
    
    jutil.attach()
    ImageReader = make_image_reader_class()
    ChannelSeparator = make_reader_wrapper_class("loci/formats/ChannelSeparator")
    FormatTools = make_format_tools_class()
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
                fig = mmmm.Figure()
                axes = fig.add_subplot(1,1,1)
                axes.imshow(image)
                frame = mmmm.FigureFrameWxAgg(1,fig)
                frame.Show()
                return True
            return False
    app = MyApp(0)
    app.MainLoop()
    jutil.detach()
    
    
