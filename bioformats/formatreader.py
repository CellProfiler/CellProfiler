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
Copyright (c) 2009-2014 Broad Institute
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

K_OMERO_SERVER = "omero_server"
K_OMERO_PORT = "omero_port"
K_OMERO_USER = "omero_user"
K_OMERO_SESSION_ID = "omero_session_id"
K_OMERO_CONFIG_FILE = "omero_config_file"
'''The cleartext password - only used if password is provided on command-line'''
K_OMERO_PASSWORD = "omero_password"

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
__omero_config_file = None
#
# Only set if user enters password in plaintext on command-line
#
__omero_password = None

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
    global __omero_config_file
    global __omero_session_id
    global __omero_server
    global __omero_username
    global __omero_port
    global __omero_password
    if __omero_config_file is not None and os.path.isfile(__omero_config_file):
        env = jutil.get_env()
        config = env.make_object_array(1, env.find_class("java/lang/String"))
        env.set_object_array_element(
            config, 0, env.new_string(u"--Ice.Config=%s" % __omero_config_file))
        script = """
        var client = Packages.omero.client(config);
        client.createSession();
        client.getSessionId();
        """
        __omero_session_id = jutil.run_script(script, dict(config=config))
    elif all([x is not None for x in 
              __omero_server, __omero_port, __omero_username, __omero_password]):
        set_omero_credentials(__omero_server, __omero_port, __omero_username, 
                              __omero_password)
    else:
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
    global __omero_config_file
    global __omero_password
    __omero_server = credentials.get(K_OMERO_SERVER, None)
    __omero_port = credentials.get(K_OMERO_PORT, None)
    __omero_username = credentials.get(K_OMERO_USER, None)
    __omero_session_id = credentials.get(K_OMERO_SESSION_ID, None)
    __omero_config_file = credentials.get(K_OMERO_CONFIG_FILE, None)
    __omero_password = credentials.get(K_OMERO_PASSWORD, None)
    
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
    with ImageReader(url=url) as rdr:
        return rdr.read(c, z, t, series, index, rescale, wants_max_intensity,
                        channel_names)
    

class ImageReader(object):
    '''Find the appropriate reader for a file
    
    This class is meant to be harnessed to a scope like this:
    
    with GetImageReader(path) as rdr:
        ....
        
    It uses __enter__ and __exit__ to manage the random access stream
    that can be used to cache the file contents in memory.
    '''
    
    def __init__(self, path = None, url= None, perform_init=True):
        self.stream = None
        file_scheme = "file:"
        self.url = url
        self.using_temp_file = False
        if url is not None and url.lower().startswith(file_scheme):
            utf8_url = urllib.url2pathname(url[len(file_scheme):])
            path = unicode(utf8_url, 'utf-8')
        self.path = path
        if path is None:
            if url.lower().startswith("omero:"):
                while True:
                    #
                    # We keep trying to contact the OMERO server via the
                    # login dialog until the user gives up or we connect.
                    #
                    try:
                        self.rdr = get_omero_reader()
                        self.path = url
                        if perform_init:
                            self.init_reader()
                        return
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
                            if jutil.is_instance_of(
                                je, "java/io/FileNotFoundException"):
                                raise exceptions.IOError(
                                    errno.ENOENT, 
                                    "The file, \"%s\", does not exist." % path,
                                    path)
                            e2 = exceptions.IOError(
                                errno.EINVAL, "Could not load the file as an image (see log for details)", path.encode('utf-8'))
                            raise e2
            else:
                #
                # Other URLS, copy them to a tempfile location
                #
                ext = url[url.rfind("."):]
                src = urllib2.urlopen(url)
                dest_fd, self.path = tempfile.mkstemp(suffix=ext)
                try:
                    dest = os.fdopen(dest_fd, 'wb')
                    shutil.copyfileobj(src, dest)
                except:
                    src.close()
                    dest.close()
                    os.remove(self.path)
                self.using_temp_file = True
                src.close()
                dest.close()
                urlpath = urllib2.urlparse.urlparse(url)[2]
                filename = urllib2.unquote(urlpath.split("/")[-1])
        else:
            if sys.platform.startswith("win"):
                self.path = self.path.replace("/", os.path.sep)
            filename = os.path.split(path)[1]

        self.stream = jutil.make_instance('loci/common/RandomAccessInputStream',
                                          '(Ljava/lang/String;)V', 
                                          self.path)
            
        self.rdr = None
        class_list = get_class_list()
        find_rdr_script = """
        var classes = class_list.getClasses();
        var rdr = null;
        var lc_filename = java.lang.String(filename.toLowerCase());
        if (lc_filename.endsWith(".tif")) {
            var omerdr = new Packages.loci.formats.in.OMETiffReader()
            if (omerdr.isThisType(stream)) {
                rdr = omerdr;
            }
        }
        for (pass=0; (pass < 3) && (rdr == null); pass++) {
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
        IFormatReader = make_iformat_reader_class()
        jrdr = jutil.run_script(find_rdr_script, dict(class_list = class_list,
                                                      filename = filename,
                                                      stream = self.stream))
        if jrdr is None:
            raise ValueError("Could not find a Bio-Formats reader for %s", self.path)
        self.rdr = IFormatReader()
        self.rdr.o = jrdr
        if perform_init:
            self.init_reader()
        
    def __enter__(self):
        return self
        
    def __exit__(self, type_class, value, traceback):
        self.close()
        
    def close(self):
        if hasattr(self, "rdr"):
            self.rdr.close()
            del self.rdr.o
            del self.rdr
        if hasattr(self, "stream") and self.stream is not None:
            jutil.call(self.stream, 'close', '()V')
            del self.stream
        if self.using_temp_file:
            os.remove(self.path)
            self.using_temp_file = False
        #
        # Run the Java garbage collector here.
        #
        jutil.static_call("java/lang/System", "gc","()V")
        
    def init_reader(self):
        mdoptions = metadatatools.get_metadata_options(metadatatools.ALL)
        self.rdr.setMetadataOptions(mdoptions)
        self.rdr.setGroupFiles(False)
        self.metadata = metadatatools.createOMEXMLMetadata()
        self.rdr.setMetadataStore(self.metadata)
        try:
            self.rdr.setId(self.path)
        except jutil.JavaException, e:
            import errno
            import exceptions
            import traceback
            logger.warn(e.message)
            for line in traceback.format_exc().split("\n"):
                logger.warn(line)
            je = e.throwable
            if jutil.is_instance_of(
                je, "Glacier2/PermissionDeniedException"):
                # Handle at a higher level
                raise
            if jutil.is_instance_of(
                je, "loci/formats/FormatException"):
                je = jutil.call(je, "getCause", 
                                "()Ljava/lang/Throwable;")
            if jutil.is_instance_of(
                je, "java/io/FileNotFoundException"):
                raise exceptions.IOError(
                    errno.ENOENT, 
                    "The file, \"%s\", does not exist." % path,
                    path)
            e2 = exceptions.IOError(
                errno.EINVAL, "Could not load the file as an image (see log for details)",
                self.path.encode('utf-8'))
            raise e2
            
        
    def read(self, c = None, z = 0, t = 0, series = None, index = None,
             rescale = True, wants_max_intensity = False, channel_names = None):
        '''Read a single plane from the image reader file.
        
        c - read from this channel. None = read color image if multichannel
            or interleaved RGB
        z - z-stack index
        t - time index
        series - series for .flex and similar multi-stack formats
        index - if None, fall back to zct, otherwise load the indexed frame
        rescale - True to rescale the intensity scale to 0 and 1, False to
                  return the raw values native to the file
        wants_max_intensity - if False,  only return the image, if True
                  return a tuple of image and max intensity
        channel_names - provide the channel names for the OME metadata
        '''
        FormatTools = make_format_tools_class()
        ChannelSeparator = make_reader_wrapper_class(
            "loci/formats/ChannelSeparator")
        env = jutil.get_env()
        if series is not None:
            self.rdr.setSeries(series)
        width = self.rdr.getSizeX()
        height = self.rdr.getSizeY()
        pixel_type = self.rdr.getPixelType()
        little_endian = self.rdr.isLittleEndian()
        if pixel_type == FormatTools.INT8:
            dtype = np.int8
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
        max_sample_value = self.rdr.getMetadataValue('MaxSampleValue')
        if max_sample_value is not None:
            try:
                scale = jutil.call(max_sample_value, 'intValue', '()I')
            except:
                bioformats.logger.warning("WARNING: failed to get MaxSampleValue for image. Intensities may be improperly scaled.")
        if index is not None:
            image = np.frombuffer(self.rdr.openBytes(index), dtype)
            if len(image) / height / width in (3,4):
                n_channels = int(len(image) / height / width)
                if self.rdr.isInterleaved():
                    image.shape = (height, width, n_channels)
                else:
                    image.shape = (n_channels, height, width)
                    image = image.transpose(1, 2, 0)
            else:
                image.shape = (height, width)
        elif self.rdr.isRGB() and self.rdr.isInterleaved():
            index = self.rdr.getIndex(z,0,t)
            image = np.frombuffer(self.rdr.openBytes(index), dtype)
            image.shape = (height, width, 3)
        elif c is not None and self.rdr.getRGBChannelCount() == 1:
            index = self.rdr.getIndex(z,c,t)
            image = np.frombuffer(self.rdr.openBytes(index), dtype)
            image.shape = (height, width)
        elif self.rdr.getRGBChannelCount() > 1:
            n_planes = self.rdr.getRGBChannelCount()
            rdr = ChannelSeparator(self.rdr)
            planes = [
                np.frombuffer(rdr.openBytes(rdr.getIndex(z,i,t)),dtype)
                for i in range(n_planes)]
            if len(planes) > 3:
                planes = planes[:3]
            elif len(planes) < 3:
                # > 1 and < 3 means must be 2
                # see issue #775
                planes.append(np.zeros(planes[0].shape, planes[0].dtype))
            image = np.dstack(planes)
            image.shape=(height, width, 3)
            del rdr
        elif self.rdr.getSizeC() > 1:
            images = [
                np.frombuffer(self.rdr.openBytes(self.rdr.getIndex(z,i,t)), dtype)
                      for i in range(self.rdr.getSizeC())]   
            image = np.dstack(images)
            image.shape = (height, width, self.rdr.getSizeC())
            if not channel_names is None:
                metadata = metadatatools.MetadataRetrieve(self.metadata)
                for i in range(self.rdr.getSizeC()):
                    index = self.rdr.getIndex(z, 0, t)
                    channel_name = metadata.getChannelName(index, i)
                    if channel_name is None:
                        channel_name = metadata.getChannelID(index, i)
                    channel_names.append(channel_name)
        elif self.rdr.isIndexed():
            #
            # The image data is indexes into a color lookup-table
            # But sometimes the table is the identity table and just generates
            # a monochrome RGB image
            #
            index = self.rdr.getIndex(z,0,t)
            image = np.frombuffer(self.rdr.openBytes(index),dtype)
            if pixel_type in (FormatTools.INT16, FormatTools.UINT16):
                lut = self.rdr.get16BitLookupTable()
                lut = np.array(
                    [env.get_short_array_elements(d)
                     for d in env.get_object_array_elements(lut)]).transpose()
            else:
                lut = self.rdr.get8BitLookupTable()
                lut = np.array(
                    [env.get_byte_array_elements(d)
                     for d in env.get_object_array_elements(lut)]).transpose()
            image.shape = (height, width)
            if not np.all(lut == np.arange(lut.shape[0])[:, np.newaxis]):
                image = lut[image, :]
        else:
            index = self.rdr.getIndex(z,0,t)
            image = np.frombuffer(self.rdr.openBytes(index),dtype)
            image.shape = (height,width)
            
        if rescale:
            image = image.astype(np.float32) / float(scale)
        if wants_max_intensity:
            return image, scale
        return image

###################
#
# A cache mechanism for image readers
#
# CellProfiler's analysis worker will read image planes from the same
# file across different jobs, so only a global cache of image readers
# will work. Here, we try and keep around one reader per key - the key
# typically being its image name in CellProfiler. We also need to clear
# the cache globally.
#
####################

# The key cache associates key with path/url
# This allows us to have two keys point to the same reader, e.g. read
# multiple channels from a stack.
__image_reader_key_cache = {}
# The image reader cache associates path/url with a reader
__image_reader_cache = {}

def get_image_reader(key, path=None, url=None):
    '''Make or find an image reader appropriate for the given path
    
    path - pathname to the reader on disk.
    
    key - use this key to keep only a single cache member associated with
          that key open at a time.
    '''
    if key in __image_reader_key_cache:
        old_path, old_url = __image_reader_key_cache[key]
        old_count, rdr = __image_reader_cache[old_path, old_url]
        if old_path == path and old_url == url:
            return rdr
        release_image_reader(key)
    if (path, url) in __image_reader_cache:
        old_count, rdr = __image_reader_cache[path, url]
    else:
        rdr = ImageReader(path=path, url=url)
        old_count = 0
    __image_reader_cache[path, url] = (old_count+1, rdr)
    __image_reader_key_cache[key] = (path, url)
    return rdr

def release_image_reader(key):
    '''Tell the cache that it should flush the reference for the given key
    
    '''
    if key in __image_reader_key_cache:
        path, url = __image_reader_key_cache[key]
        del __image_reader_key_cache[key]
        old_count, rdr = __image_reader_cache[path, url]
        if old_count == 1:
            rdr.close()
            del __image_reader_cache[path, url]
        else:
            __image_reader_cache[path, url] = (old_count-1, rdr)

def clear_image_reader_cache():
    '''Get rid of any open image readers'''
    for use_count, rdr in __image_reader_cache.values():
        rdr.close()
    __image_reader_cache.clear()
    __image_reader_key_cache.clear()
    
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
    
    with ImageReader(path=path) as rdr:
        return rdr.read(c, z, t, series, index, rescale, wants_max_intensity,
                        channel_names)
    
def get_omexml_metadata(path=None, url=None):
    '''Read the OME metadata from a file using Bio-formats
    
    path - path to the file
    
    allowopenfiles - allow the image reader to open files while looking for
                     the proper reader class.
                     
    groupfiles - utilize the groupfiles option to take the directory structure
                 into account.
    '''
    with ImageReader(path=path, url=url, perform_init=False) as rdr:
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
        xml = jutil.run_script(script, dict(path=rdr.path, reader = rdr.rdr))
        return xml
