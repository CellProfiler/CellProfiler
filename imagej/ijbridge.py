'''
ijbridge provides a high-level interface to ImageJ so in-process and inter-
process execution of ImageJ can be abstracted and branched cleanly.
'''
import sys
import numpy as np
import os, tempfile
import Image as PILImage
from subprocess import Popen, PIPE, STDOUT
import shlex
import socket
import cellprofiler.utilities.jutil as J
import imagej.macros as ijmacros
import imagej.windowmanager as ijwm
import imagej.imageprocessor as ijiproc
import imagej.imageplus as ijip
import struct


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
        sys.stderr.write("Warning: Failed to find tools.jar\n")
if os.environ.has_key('CLASSPATH'):
   __class_path += os.pathsep + os.environ['CLASSPATH']
os.environ['CLASSPATH'] = __class_path


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


class in_proc_ij_bridge(ij_bridge):
   '''Interface for running ImageJ in a the same process as CellProfiler.
   '''
##   def __init__(self):
##      J.attach()
##   def __del__(self):
##      ''' call del on this object to detach from javabridge. If the object is
##      declared locally the javabridge will be detached once the program leaves 
##      it's scope'''
##      J.detach()
      
   def inject_image(self, pixels, name=''):
      '''inject an image into ImageJ for processing'''
      ij_processor = ijiproc.make_image_processor(pixels * 255.0)
      image_plus = ijip.make_imageplus_from_processor(name, ij_processor)
      if sys.platform == "darwin":
         ijwm.set_temp_current_image(image_plus)
      else:
         ijwm.set_current_image(image_plus)
   
   def get_current_image(self):
      '''returns the WindowManager's current image as a numpy float array'''
      image_plus = ijwm.get_current_image()
      ij_processor = image_plus.getProcessor()
      pixels = ijiproc.get_image(ij_processor) / 255.0
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
   # Send the number of bytes in the data
   socket.send(nbytes)
   # Send the 8 byte command 
   socket.send(cmd)
   if len(data) > 0:
      socket.send(data)
   
   print '<SERVER> waiting for response...'
   # Get the response size
   size = read_nbytes(socket, 4)   
   size = struct.unpack('>i4', size)[0]
   print '<SERVER> response size:', repr(size)
   # Get the response message
   msg = read_nbytes(socket, 8)
   print '<SERVER> response to %s:%s'%(cmd, msg)
   # Get any attached data
   data = read_nbytes(socket, size)
   return msg, data
   

class inter_proc_ij_bridge(ij_bridge):
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
      self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.server_socket.bind(("", 0))
      hostaddr, port = self.server_socket.getsockname()
      self.server_socket.listen(5)
      print "ImageJ bridge TCPServer waiting for client on port", port
      self.ijproc = Popen(shlex.split('java -Xmx512m TCPClient %s'%(port)),
                          stdin=None, stdout=None, stderr=None)
      self.client_socket, address = self.server_socket.accept()
      print "ImageJ bridge got a connection from", address
   
   def inject_image(self, pixels, name=None):
      '''inject an image into ImageJ for processing'''
      data = (np.array([pixels.shape[1]], ">i4").tostring() + 
              np.array([pixels.shape[0]], ">i4").tostring() + 
              (pixels*255).astype('uint8').tostring())
      msg, data = communicate(self.client_socket, self.INJECT, data)
      assert msg.startswith('success')

   def get_current_image(self):
      '''returns the WindowManager's current image as a numpy float array'''
      msg, data = communicate(self.client_socket, self.GET_IMAGE)
      w = struct.unpack('>i4',data[:4])[0]
      h = struct.unpack('>i4',data[4:8])[0]
      pixels = data[8:]
      if msg.startswith('success'):
         im = PILImage.fromstring('L', (w,h), pixels)
##         im.show()
         return pil_to_np(im)
      else:
         raise Exception("Get current image failed to return an image")

   def get_commands(self):
      '''returns a list of the available command strings'''
      msg, data = communicate(self.client_socket, self.GET_COMMANDS)
      return data.split('\n')

   def execute_command(self, command, options=None):
      '''execute the named command within ImageJ'''
      msg, data = communicate(self.client_socket, self.COMMAND, command)
      assert msg.startswith('success')

   def execute_macro(self, macro_text):
      '''execute a macro in ImageJ
      macro_text - the macro program to be run
      '''
      msg, data = communicate(self.client_socket, self.MACRO, macro_text)
      assert msg.startswith('success')
   
   def show_imagej(self):
      '''show the ImageJ user interface'''
      msg, data = communicate(self.client_socket, self.SHOW_IMAGEJ)
      assert msg.startswith('success')
      
   def quit(self):
      print '<SERVER> quit'
      msg, data = communicate(self.client_socket, self.QUIT)
      self.client_socket.close()
      self.server_socket.close()
      assert msg.startswith('success')
##      os.kill(self.ijproc.pid, 9)
      

def np2pil(imdata):
   '''Convert np image data to PIL Image'''
   if len(imdata.shape) == 2:
      buf = np.dstack([imdata, imdata, imdata])
   elif len(imdata.shape) == 3:
      buf = imdata
      assert imdata.shape[2] >= 3, 'Cannot convert the given numpy array to PIL'
   if buf.dtype != 'uint8':
      buf = (buf * 255.0).astype('uint8')
   im = PILImage.fromstring(mode='RGB', size=(buf.shape[1],buf.shape[0]),
                            data=buf.tostring())
   return im
 
def pil_to_np( pilImage ):
   """
   load a PIL image and return it as a numpy array of uint8.  For
   grayscale images, the return array is MxN.  For RGB images, the
   return value is MxNx3.  For RGBA images the return value is MxNx4
   """
   def toarray(im):
      'return a 1D array of floats'
      x_str = im.tostring('raw', im.mode)
      x = np.fromstring(x_str,np.uint8)
      return x
   
   if pilImage.mode[0] == 'P':
      im = pilImage.convert('RGBA')
      x = toarray(im)
      x = x.reshape(-1, 4)
      if np.all(x[:,0] == x):
         im = pilImage.convert('L')
      pilImage = im

      if pilImage.mode[0] in ('1', 'L', 'I', 'F'):
         x = toarray(pilImage)
         x.shape = pilImage.size[1], -1
         return x
      else:
         x = toarray(pilImage.convert('RGBA'))
         x.shape = pilImage.size[1], pilImage.size[0], 4
         # discard alpha if all 1s
         if (x[:,:,3] == 255).all():
            return x[:,:,:3]
         return x

     
if __name__ == '__main__':
   from time import time
   ipb = inter_proc_ij_bridge()

   pixels = np.random.standard_normal((400,600))
   pixels[100:300,200:400] = 0
   pixels[pixels<0] = 0
   pixels /= pixels.max()
   t0 = time()
   ipb.inject_image(pixels, 'name ignored')
   print 'time to inject:',time()-t0
   ipb.get_current_image()
   ipb.execute_macro('run("Invert");')
   ipb.execute_command('Add Noise')
   t0 = time()
   ipb.get_current_image()
   print 'time to get image:',time()-t0
   cmds = ipb.get_commands()
   print 'ImageJ commands:', cmds
   ipb.quit()
   J.kill_vm()
