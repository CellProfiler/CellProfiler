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
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org'
'''
    
__version__ = "$Revision$"

import logging
logger = logging.getLogger(__name__)
import numpy as np
import os
import sys
import urllib
import urllib2
import shutil
import tempfile

import cellprofiler.utilities.jutil as jutil
import bioformats
import bioformats.metadatatools as metadatatools
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
        isIndexed = jutil.make_method('isIndexed', '()Z',
                                      'Return True if the raw data is indexes in a lookup table')
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
        def setId(self, path):
            '''Set the name of the file'''
            jutil.call(self.o, 'setId', 
                       '(Ljava/lang/String;)V',
                       path)
                                            
        getMetadataStore = jutil.make_method('getMetadataStore', '()Lloci/formats/meta/MetadataStore;',
                                             'Retrieves the current metadata store for this reader.')
        get8BitLookupTable = jutil.make_method(
            'get8BitLookupTable',
            '()[[B', 'Get a lookup table for 8-bit indexed images')
        get16BitLookupTable = jutil.make_method(
            'get16BitLookupTable',
            '()[[S', 'Get a lookup table for 16-bit indexed images')
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
                'loci.formats.in.BDReader',
                #
                # MRCReader will read .stk files which should be read
                # by MetamorphReader
                #
                'loci.formats.in.MRCReader'
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

__omero_server = None
__omero_username = None
__omero_session_id = None
__omero_port = None

def set_omero_credentials(omero_server, omero_port, omero_username, omero_password):
    '''Set the credentials to be used to connect to the Omero server
    
    omero_server - DNS name of the server

    omero_port - use this port to connect to the server

    omero_username - log on as this user

    omero_password - log on using this password
    
    The session ID is valid after this is called. An exception is thrown
    if the login fails. omero_logout() can be called to log out.
    '''
    global __omero_server
    global __omero_username
    global __omero_session_id
    global __omero_port
    __omero_server = omero_server
    __omero_port = omero_port
    __omero_username = omero_username
    script = """
    var client = Packages.omero.client(server, port);
    var serverFactory = client.createSession(user, password);
    client.getSessionId();
    """
    __omero_session_id = jutil.run_script(script, dict(
        server = __omero_server,
        port = __omero_port,
        user = __omero_username,
        password = omero_password))
    return __omero_session_id
    
def get_omero_credentials():
    '''Return a pickleable dictionary representing the Omero credentials
    
    Call use_omero_credentials in some other process to use this.
    '''
    if __omero_session_id is None:
        omero_login()
        
    return dict(omero_server = __omero_server,
                omero_port = __omero_port,
                omero_user = __omero_username,
                omero_session_id = __omero_session_id)

def omero_login():
    __omero_login_fn()
    return __omero_session_id
    
def omero_logout():
    '''Abandon any current Omero session'''
    global __omero_session_id
    __omero_session_id = None

def use_omero_credentials(credentials):
    '''Use the session ID from an extant login as credentials
    
    credentials - credentials from get_omero_credentials
    '''
    global __omero_server
    global __omero_username
    global __omero_session_id
    global __omero_port
    __omero_server = credentials["omero_server"]
    __omero_port = credentials["omero_port"]
    __omero_username = credentials["omero_user"]
    __omero_session_id = credentials["omero_session_id"]
    
__omero_login_fn = None
def set_omero_login_hook(fn):
    '''Set the function to be called when a login to Omero is needed'''
    global __omero_login_fn
    __omero_login_fn = fn
    
def get_omero_reader():
    '''Return an loci.ome.io.OMEROReader instance, wrapped as a FormatReader'''
    script = """
    var rdr = new Packages.loci.ome.io.OmeroReader();
    rdr.setServer(server);
    rdr.setPort(port);
    rdr.setUsername(username);
    rdr.setSessionID(sessionID);
    rdr;
    """
    if __omero_session_id is None:
        omero_login()
        
    jrdr = jutil.run_script(script, dict(
        server = __omero_server,
        port = __omero_port,
        username = __omero_username,
        sessionID = __omero_session_id))
        
    rdr = make_iformat_reader_class()()
    rdr.o = jrdr
    return rdr


def load_using_bioformats_url(url, c=None, z=0, t=0, series=None, index=None,
                          rescale = True,
                          wants_max_intensity = False,
                          channel_names = None):
    '''Load a file from Bio-formats via a URL
    
    '''
    file_scheme = "file:"
    if url.lower().startswith("omero:"):
        return load_using_bioformats(
            url, c=c, z=z, t=t, series=series, index=index,
            rescale=rescale,
            wants_max_intensity=wants_max_intensity, channel_names=channel_names)
    if not url.lower().startswith(file_scheme):
        ext = url[url.rfind("."):]
        src = urllib2.urlopen(url)
        dest_fd, dest_path = tempfile.mkstemp(suffix=ext)
        try:
            dest = os.fdopen(dest_fd, 'wb')
            shutil.copyfileobj(src, dest)
            src.close()
            dest.close()
            return load_using_bioformats(
                dest_path, c=c, z=z, t=t, series=series, index=index, 
                rescale=rescale,
                wants_max_intensity=wants_max_intensity,
                channel_names=channel_names)
        finally:
            os.remove(dest_path)
    
    utf8_url = urllib.url2pathname(url[len(file_scheme):])
    pathname = unicode(utf8_url, 'utf-8')
    return load_using_bioformats(
        pathname,
        c=c, z=z, t=t, series=series, index=index, rescale=rescale,
        wants_max_intensity=wants_max_intensity, channel_names=channel_names)
    

class GetImageReader(object):
    '''Find the appropriate reader for a file
    
    This class is meant to be harnessed to a scope like this:
    
    with GetImageReader(path) as rdr:
        ....
        
    It uses __enter__ and __exit__ to manage the random access stream
    that can be used to cache the file contents in memory.
    '''
    def __init__(self, path):
        self.stream = None
        if path.lower().startswith("omero:"):
            self.rdr = get_omero_reader()
            return
        self.rdr = None
        class_list = get_class_list()
        find_rdr_script = """
        var classes = class_list.getClasses();
        var rdr = null;
        var lc_filename = java.lang.String(filename.toLowerCase());
        for (pass=0; pass < 3; pass++) {
            for (class_idx in classes) {
                var maybe_rdr = classes[class_idx].newInstance();
                if (pass == 0) {
                    if (maybe_rdr.isThisType(filename, false)) {
                        rdr = maybe_rdr;
                        break;
                    }
                    continue;
                } else if (pass == 1) {
                    var suffixes = maybe_rdr.getSuffixes();
                    var suffix_found = false;
                    for (suffix_idx in suffixes) {
                        var suffix = java.lang.String(suffixes[suffix_idx]);
                        suffix = suffix.toLowerCase();
                        if (lc_filename.endsWith(suffix)) {
                            suffix_found = true;
                            break;
                        }
                    }
                    if (! suffix_found) continue;
                }
                if (maybe_rdr.isThisType(stream)) {
                    rdr = maybe_rdr;
                    break;
                }
            }
            if (rdr) break;
        }
        rdr;
        """
        self.stream = jutil.make_instance('loci/common/RandomAccessInputStream',
                                          '(Ljava/lang/String;)V', path)
        filename = os.path.split(path)[1]
        IFormatReader = make_iformat_reader_class()
        jrdr = jutil.run_script(find_rdr_script, dict(class_list = class_list,
                                                      filename = filename,
                                                      stream = self.stream))
        if jrdr is None:
            raise ValueError("Could not find a Bio-Formats reader for %s", path)
        self.rdr = IFormatReader()
        self.rdr.o = jrdr
        
    def __enter__(self):
        return self.rdr
        
    def __exit__(self, type_class, value, traceback):
        if self.rdr is not None:
            self.rdr.close()
            del self.rdr.o
            del self.rdr
        if  self.stream is not None:
            jutil.call(self.stream, 'close', '()V')
        del self.stream
        #
        # Run the Java garbage collector here.
        #
        jutil.static_call("java/lang/System", "gc","()V")
    
    
def load_using_bioformats(path, c=None, z=0, t=0, series=None, index=None,
                          rescale = True,
                          wants_max_intensity = False,
                          channel_names = None):
    '''Load the given image file using the Bioformats library
    
    path: path to the file
    z: the frame index in the z (depth) dimension.
    t: the frame index in the time dimension.
    channel_names: None if you don't want them, a list which will be filled if you do
    
    Returns either a 2-d (grayscale) or 3-d (2-d + 3 RGB planes) image
    '''
    #
    # We loop as long as the user is willing to try logging
    # in after timeout.
    #
    while True:
        try:
            return __load_using_bioformats(
                path, c, z, t, series, index, rescale, wants_max_intensity,
                channel_names)
        except jutil.JavaException, e:
            je = e.throwable
            if jutil.is_instance_of(
                je, "loci/formats/FormatException"):
                je = jutil.call(je, "getCause", 
                                "()Ljava/lang/Throwable;")
            if jutil.is_instance_of(
                je, "Glacier2/PermissionDeniedException"):
                omero_logout()
                omero_login()
            else:
                import errno
                import exceptions
                import traceback
                logger.warn(e.message)
                for line in traceback.format_exc().split("\n"):
                    logger.warn(line)
                e2 = exceptions.IOError(
                    errno.EINVAL, "Could not load the file as an image (see log for details)", path.encode('utf-8'))
                raise e2
    
def __load_using_bioformats(path, c, z, t, series, index, rescale,
                            wants_max_intensity, channel_names):
    FormatTools = make_format_tools_class()
    ImageReader = make_image_reader_class()
    ChannelSeparator = make_reader_wrapper_class(
        "loci/formats/ChannelSeparator")
    
    #
    # Bioformats is more picky about slashes than Python
    #
    if sys.platform.startswith("win"):
        path = path.replace("/",os.path.sep)
    #
    # Bypass the ImageReader and scroll through the class list. The
    # goal here is to ask the FormatHandler if it thinks it could
    # possibly parse the file, then only give the FormatReader access
    # to the open file stream so it can't damage the file server.
    #
    # Pass 0 - allow access to stream if reader believes it is the correct
    #          reader no matter what the file contents.
    # Pass 1 - allow access to stream if reader indicates that it supports
    #          files with this file's suffix.
    # Pass 2 - allow all readers access to the stream
    #
    
    env = jutil.get_env()
    with GetImageReader(path) as rdr:
        mdoptions = metadatatools.get_metadata_options(metadatatools.ALL)
        rdr.setMetadataOptions(mdoptions)
        rdr.setGroupFiles(False)
        metadata = metadatatools.createOMEXMLMetadata()
        rdr.setMetadataStore(metadata)
        rdr.setId(path)
        width = rdr.getSizeX()
        height = rdr.getSizeY()
        pixel_type = rdr.getPixelType()
        little_endian = rdr.isLittleEndian()
        if pixel_type == FormatTools.INT8:
            dtype = np.char
            scale = 255
        elif pixel_type == FormatTools.UINT8:
            dtype = np.uint8
            scale = 255
        elif pixel_type == FormatTools.UINT16:
            dtype = '<u2' if little_endian else '>u2'
            scale = 65535
        elif pixel_type == FormatTools.INT16:
            dtype = '<i2' if little_endian else '>i2'
            scale = 65535
        elif pixel_type == FormatTools.UINT32:
            dtype = '<u4' if little_endian else '>u4'
            scale = 2**32
        elif pixel_type == FormatTools.INT32:
            dtype = '<i4' if little_endian else '>i4'
            scale = 2**32-1
        elif pixel_type == FormatTools.FLOAT:
            dtype = '<f4' if little_endian else '>f4'
            scale = 1
        elif pixel_type == FormatTools.DOUBLE:
            dtype = '<f8' if little_endian else '>f8'
            scale = 1
        max_sample_value = rdr.getMetadataValue('MaxSampleValue')
        if max_sample_value is not None:
            try:
                scale = jutil.call(max_sample_value, 'intValue', '()I')
            except:
                bioformats.logger.warning("WARNING: failed to get MaxSampleValue for image. Intensities may be improperly scaled.")
        if series is not None:
            rdr.setSeries(series)
        if index is not None:
            image = np.frombuffer(rdr.openBytes(index), dtype)
            if len(image) / height / width in (3,4):
                n_channels = int(len(image) / height / width)
                if rdr.isInterleaved():
                    image.shape = (height, width, n_channels)
                else:
                    image.shape = (n_channels, height, width)
                    image = image.transpose(1, 2, 0)
            else:
                image.shape = (height, width)
        elif rdr.isRGB() and rdr.isInterleaved():
            index = rdr.getIndex(z,0,t)
            image = np.frombuffer(rdr.openBytes(index), dtype)
            image.shape = (height, width, 3)
        elif c is not None and rdr.getRGBChannelCount() == 1:
            index = rdr.getIndex(z,c,t)
            image = np.frombuffer(rdr.openBytes(index), dtype)
            image.shape = (height, width)
        elif rdr.getRGBChannelCount() > 1:
            rdr.close()
            rdr = ImageReader()
            rdr.allowOpenToCheckType(False)
            rdr = ChannelSeparator(rdr)
            rdr.setGroupFiles(False)
            rdr.setId(path)
            red_image, green_image, blue_image = [
                np.frombuffer(rdr.openBytes(rdr.getIndex(z,i,t)),dtype)
                for i in range(3)]
            image = np.dstack((red_image, green_image, blue_image))
            image.shape=(height,width,3)
        elif rdr.getSizeC() > 1:
            images = [np.frombuffer(rdr.openBytes(rdr.getIndex(z,i,t)), dtype)
                      for i in range(rdr.getSizeC())]
            image = np.dstack(images)
            image.shape = (height, width, rdr.getSizeC())
            if not channel_names is None:
                metadata = metadatatools.MetadataRetrieve(metadata)
                for i in range(rdr.getSizeC()):
                    index = rdr.getIndex(z, 0, t)
                    channel_name = metadata.getChannelName(index, i)
                    if channel_name is None:
                        channel_name = metadata.getChannelID(index, i)
                    channel_names.append(channel_name)
        elif rdr.isIndexed():
            #
            # The image data is indexes into a color lookup-table
            # But sometimes the table is the identity table and just generates
            # a monochrome RGB image
            #
            index = rdr.getIndex(z,0,t)
            image = np.frombuffer(rdr.openBytes(index),dtype)
            if pixel_type in (FormatTools.INT16, FormatTools.UINT16):
                lut = rdr.get16BitLookupTable()
                lut = np.array(
                    [env.get_short_array_elements(d)
                     for d in env.get_object_array_elements(lut)]).transpose()
            else:
                lut = rdr.get8BitLookupTable()
                lut = np.array(
                    [env.get_byte_array_elements(d)
                     for d in env.get_object_array_elements(lut)]).transpose()
            image.shape = (height, width)
            if not np.all(lut == np.arange(lut.shape[0])[:, np.newaxis]):
                image = lut[image, :]
        else:
            index = rdr.getIndex(z,0,t)
            image = np.frombuffer(rdr.openBytes(index),dtype)
            image.shape = (height,width)
            
        rdr.close()
    if rescale:
        image = image.astype(np.float32) / float(scale)
    if wants_max_intensity:
        return image, scale
    return image

def get_omexml_metadata(path):
    '''Read the OME metadata from a file using Bio-formats
    
    path - path to the file
    
    allowopenfiles - allow the image reader to open files while looking for
                     the proper reader class.
                     
    groupfiles - utilize the groupfiles option to take the directory structure
                 into account.
    '''
    with GetImageReader(path) as rdr:
        #
        # Below, "in" is a keyword and Rhino's parser is just a little wonky I fear.
        #
        # It is critical that setGroupFiles be set to false, goodness knows
        # why, but if you don't the series count is wrong for flex files.
        #
        script = """
        importClass(Packages.loci.common.services.ServiceFactory,
                    Packages.loci.formats.services.OMEXMLService,
                    Packages.loci.formats['in'].DefaultMetadataOptions,
                    Packages.loci.formats['in'].MetadataLevel);
        reader.setGroupFiles(false);
        reader.setOriginalMetadataPopulated(true);
        var service = new ServiceFactory().getInstance(OMEXMLService);
        var metadata = service.createOMEXMLMetadata();
        reader.setMetadataStore(metadata);
        reader.setMetadataOptions(new DefaultMetadataOptions(MetadataLevel.ALL));
        reader.setId(path);
        var xml = service.getOMEXML(metadata);
        xml;
        """
        xml = jutil.run_script(script, 
                               dict(path=path,
                                    reader = rdr))
        rdr.close()
        return xml
