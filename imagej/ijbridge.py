'''
ijbridge provides a high-level interface to ImageJ so in-process and inter-
process execution of ImageJ can be abstracted and branched cleanly.
'''
import sys
import logging
import numpy as np
import os, tempfile
from subprocess import Popen, PIPE, STDOUT
import shlex
import socket
import struct
import cellprofiler.utilities.jutil as J
import cellprofiler.preferences as cpprefs
from cellprofiler.utilities.singleton import Singleton
from bioformats import USE_IJ2
import imagej.macros as ijmacros
import imagej.windowmanager as ijwm
import imagej.imageprocessor as ijiproc
import imagej.imageplus as ijip
if USE_IJ2:
    import imagej.imagej2 as IJ2

logger = logging.getLogger(__name__)
if hasattr(sys, 'frozen'):
    __root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
else:
    __root_path = os.path.abspath(os.path.split(__file__)[0])
    __root_path = os.path.split(__root_path)[0]
__path = os.path.join(__root_path, 'bioformats')
__imagej_path = os.path.join(__root_path, 'imagej')
__loci_jar = os.path.join(__path, "loci_tools.jar")
__ij_jar = os.path.join(__imagej_path, "ij.jar")
__imglib_jar = os.path.join(__imagej_path, "imglib.jar")
__javacl_jar = os.path.join(__imagej_path, "javacl-1.0-beta-4-shaded.jar")
__precompiled_headless_jar = os.path.join(__imagej_path, "precompiled_headless.jar")
__class_path = os.pathsep.join((__loci_jar, __ij_jar, __imglib_jar, 
                                __javacl_jar, __imagej_path))

if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
    # Have to find tools.jar
    from cellprofiler.utilities.setup import find_jdk
    jdk_path = find_jdk()
    if jdk_path is not None:
        __tools_jar = os.path.join(jdk_path, "lib","tools.jar")
        __class_path += os.pathsep + __tools_jar
    else:
        logger.warning("Failed to find tools.jar\n")
if os.environ.has_key('CLASSPATH'):
    __class_path += os.pathsep + os.environ['CLASSPATH']
os.environ['CLASSPATH'] = __class_path


def get_ij_bridge():
    '''Returns an an ijbridge that will work given the platform and preferences
    '''
    if USE_IJ2:
        return ij2_bridge.getInstance()
    if True or sys.platform != 'darwin':
        return in_proc_ij_bridge.getInstance()
    else: # sys.platform == 'darwin':
        return inter_proc_ij_bridge.getInstance()


class ij_bridge(object):
    '''This class provides a high-level interface for running ImageJ from
    CellProfiler. It is intended to abstract away whether IJ is being run within
    the same process or in a separate process.
    '''   
    def inject_image(self, pixels, name=None):
        '''inject an image into ImageJ for processing'''
        raise NotImplementedError

    def get_current_image(self):
        '''returns the WindowManager's current image as a numpy float array'''
        raise NotImplementedError

    def get_commands(self):
        '''returns a list of the available command strings'''
        raise NotImplementedError

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        raise NotImplementedError

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ

        macro_text - the macro program to be run
        '''
        raise NotImplementedError

    def show_imagej(self):
        '''show the ImageJ user interface'''
        raise NotImplementedError


class in_proc_ij_bridge(ij_bridge, Singleton):
    '''Interface for running ImageJ in a the same process as CellProfiler.
    '''
    def __init__(self):
        J.attach()

    def __del__(self):
        '''call del on this object to detach from javabridge. If the object is
        declared locally the javabridge will be detached once the program leaves 
        it's scope'''
        J.detach()

    def inject_image(self, pixels, name=''):
        '''inject an image into ImageJ for processing'''
        ij_processor = ijiproc.make_image_processor(
            (pixels * 255.0).astype('float32'))
        script = """
        new java.lang.Runnable() {
            run: function() {
                var imp = Packages.ij.ImagePlus(name, ij_processor);
                imp.show();
                Packages.ij.WindowManager.setCurrentWindow(imp.getWindow());
            }};"""
        r = J.run_script(script, bindings_in = {
            "name":name,
            "ij_processor": ij_processor})
        J.execute_runnable_in_main_thread(r, True)

    def get_current_image(self):
        '''returns the WindowManager's current image as a numpy float array'''
        image_plus = ijwm.get_current_image()
        ij_processor = image_plus.getProcessor()
        pixels = ijiproc.get_image(ij_processor).astype('float32') / 255.0
        return pixels

    def get_commands(self):
        '''returns a list of the available command strings'''
        return ijmacros.get_commands()

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        ijmacros.execute_command(command, options)

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ

        macro_text - the macro program to be run
        '''
        ijmacros.execute_macro(macro_text)

    def show_imagej(self):
        '''show the ImageJ user interface'''
        ijmacros.show_imagej()

class ij2_bridge(ij_bridge, Singleton):
    def __init__(self):
        services = [
            "imagej.event.EventService",
            "imagej.object.ObjectService",
            "imagej.display.OverlayService",
            "imagej.display.DisplayService",
            "imagej.platform.PlatformService",
            "imagej.ext.plugin.PluginService",
            "imagej.ext.module.ModuleService", 
            "imagej.ui.UIService",
            "imagej.tool.ToolService"
        ]

        self.context = IJ2.create_context(services)

    def inject_image(self, pixels, name=None):
        '''inject an image into ImageJ for processing'''
        dataset = IJ2.create_dataset(pixels, name)
        display_service = IJ2.get_display_service(self.context)
        display = display_service.createDisplay(dataset)
        display_service.setActiveDisplay(display.o)

    def get_current_image(self):
        '''returns the WindowManager's current image as a numpy float array'''
        display_service = IJ2.get_display_service(self.context)
        current_display = display_service.getActiveImageDisplay()
        dataset = display_service.getActiveDataset(current_display)
        return dataset.get_pixel_data()

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        module_service = IJ2.get_module_service(self.context)
        matches = [x for x in module_service.getModules()
                   if x.getName() == command]
        if len(matches) != 1:
            raise ValueError("Could not find %s module" % command)
        module_service.run(matches[0].createModule())

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ

        macro_text - the macro program to be run
        '''
        raise NotImplementedError

    def show_imagej(self):
        '''show the ImageJ user interface'''
        pass


def read_nbytes(socket, nbytes):
    '''use in place of socket.recv(nbytes)'''
    assert nbytes >= 0
    if nbytes == 0:
        return ''
    data = socket.recv(nbytes)
    while len(data) < nbytes:
        data += socket.recv(nbytes - len(data))
    return data

def communicate(socket, cmd, data=None):
    '''Sends a command and optional data (string) to the client socket.
    Message format: [SIZE (4 bytes) | COMMAND (8 bytes) | DATA (N bytes)]
    Returns a tuple containing the response message and data.
    '''
    assert len(cmd) == 8, ('Commands must contain exactly 8 characters.'
                           'Command was: "%s"'%(cmd))
    # Send the message size
    if data is None: 
        data = ''
    msg_size = len(data)
    print '<SERVER> sending msg: [ %s | %s | <DATA: %s bytes> ]'%(msg_size, cmd, len(data))
    nbytes = np.array([msg_size], ">i4").tostring()
    socket.send(nbytes + cmd + data)

    print '<SERVER> waiting for response...'
    # Get the response size
    size = read_nbytes(socket, 4)   
    size = struct.unpack('>i4', size)[0]
    print '<SERVER> response size:', repr(size)
    # Get the response message
    msg = read_nbytes(socket, 8)
    print '<SERVER> response to %s:%s'%(cmd, msg)
    # Get any attached data
    data = ''
    if size > 0:
        data = read_nbytes(socket, size)
    return msg, data


class inter_proc_ij_bridge(ij_bridge, Singleton):
    '''Interface for running ImageJ in a separate process from CellProfiler.
    '''
    # We use a limited vocabulary of command names to talk to ImageJ.
    # All command names must contain exactly 8 characters and must be changed in
    # TCPClient.py if they are changed here.
    INJECT       = 'inject  '
    GET_IMAGE    = 'getimg  '
    GET_COMMANDS = 'getcmds '
    COMMAND      = 'command '
    MACRO        = 'macro   '
    SHOW_IMAGEJ  = 'showij  '
    QUIT         = 'quit    '

    def __init__(self):
        self.start_ij()

    def __del__(self):
        '''call del on this object to close ImageJ and the TCPClient.'''
        self.quit()

    def start_ij(self):
        try:
            self.client_socket.close()
            self.server_socket.close()
        except: pass
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("", 0))
        _, self.port = self.server_socket.getsockname()
        self.server_socket.listen(5)
        self.server_socket.listen(5)
        print "ImageJ bridge TCPServer waiting for client on port", self.port
        from cellprofiler.preferences import get_ij_plugin_directory
        plugin_dir =  get_ij_plugin_directory()
        if plugin_dir is not None:
            command = 'java -Xmx512m -Dplugins.dir=%s TCPClient %s' % (
                plugin_dir, self.port)
        else:
            command = 'java -Xmx512m TCPClient %s' % self.port
        self.ijproc = Popen(shlex.split(command),
                            stdin=None, stdout=None, stderr=None)
        self.client_socket, address = self.server_socket.accept()
        print "ImageJ bridge got a connection from", address      

    def inject_image(self, pixels, name=None):
        '''inject an image into ImageJ for processing'''
        if self.ijproc.poll() is not None:
            self.start_ij()
        assert pixels.ndim == 2, 'Inject image currently only supports single channel images.'
        data = (np.array([pixels.shape[1]], ">i4").tostring() + 
                np.array([pixels.shape[0]], ">i4").tostring() + 
                (pixels).astype('>f4').tostring())
        msg, data = communicate(self.client_socket, self.INJECT, data)
        assert msg.startswith('success')

    def get_current_image(self):
        '''returns the WindowManager's current image as a numpy float array'''
        if self.ijproc.poll() is not None:
            raise Exception("Can't retrieve current image from ImageJ because the subprocess was closed.")
        msg, data = communicate(self.client_socket, self.GET_IMAGE)
        w = struct.unpack('>i4',data[:4])[0]
        h = struct.unpack('>i4',data[4:8])[0]
        pixels = data[8:]
        if msg.startswith('success'):
            pixels = np.fromstring(pixels, dtype='>f4').reshape(h,w)
            pixels = pixels.astype('float32')
            return pixels
        else:
            raise Exception("Get current image failed to return an image")

    def get_commands(self):
        '''returns a list of the available command strings'''
        if self.ijproc.poll() is not None:
            self.start_ij()
        msg, data = communicate(self.client_socket, self.GET_COMMANDS)
        return data.split('\n')

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        if self.ijproc.poll() is not None:
            raise Exception("Can't execute \"%s\" in ImageJ because the subprocess was closed."%(command))
        msg, data = communicate(self.client_socket, self.COMMAND, command+';'+(options or ''))
        assert msg.startswith('success')

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ
        macro_text - the macro program to be run
        '''
        if self.ijproc.poll() is not None:
            raise Exception("Can't execute \"%s\" in ImageJ because the subprocess was closed."%(macro_text))
        msg, data = communicate(self.client_socket, self.MACRO, macro_text)
        assert msg.startswith('success')

    def show_imagej(self):
        '''show the ImageJ user interface'''
        if self.ijproc.poll() is not None:
            self.start_ij()
        msg, data = communicate(self.client_socket, self.SHOW_IMAGEJ)
        assert msg.startswith('success')

    def quit(self):
        '''close the java process'''
        if self.ijproc.poll() is not None:
            raise Exception("Can't quit ImageJ because the subprocess was closed.")
        print '<SERVER> quit'
        msg, data = communicate(self.client_socket, self.QUIT)
        self.client_socket.close()
        self.server_socket.close()
        if not msg.startswith('success'):
            os.kill(self.ijproc.pid, 9)


if __name__ == '__main__':
    import wx
    from time import time

    app = wx.PySimpleApp()
    PIXELS = np.tile(np.linspace(0,200,200), 300).reshape((300,200)).T
    PIXELS[50:150,100:200] = 0.0
    PIXELS[100, 150] = 1.0
    ipb = inter_proc_ij_bridge.getInstance()

    f = wx.Frame(None)
    b1 = wx.Button(f, -1, 'inject')
    b2 = wx.Button(f, -1, 'get')
    b3 = wx.Button(f, -1, 'get cmds')
    b4 = wx.Button(f, -1, 'cmd:add noise')
    b5 = wx.Button(f, -1, 'macro:invert')
    b6 = wx.Button(f, -1, 'quit')
    f.SetSizer(wx.BoxSizer(wx.VERTICAL))
    f.Sizer.Add(b1)
    f.Sizer.Add(b2)
    f.Sizer.Add(b3)
    f.Sizer.Add(b4)
    f.Sizer.Add(b5)
    f.Sizer.Add(b6)

    def on_getcmds(evt):
        print ipb.get_commands()

    b1.Bind(wx.EVT_BUTTON, lambda(x): ipb.inject_image(PIXELS, 'name ignored'))
    b2.Bind(wx.EVT_BUTTON, lambda(x): ipb.get_current_image())
    b3.Bind(wx.EVT_BUTTON, on_getcmds)
    b4.Bind(wx.EVT_BUTTON, lambda(x): ipb.execute_command('Add Noise'))
    b5.Bind(wx.EVT_BUTTON, lambda(x): ipb.execute_macro('run("Invert");'))
    b6.Bind(wx.EVT_BUTTON, lambda(x): ipb.quit())
    f.Show()

    app.MainLoop()

    ipb.quit()

    try:
        import cellprofiler.utilities.jutil as jutil
        jutil.kill_vm()
    except:
        logger.warning("Caught exception while killing VM", exc_info=True)
