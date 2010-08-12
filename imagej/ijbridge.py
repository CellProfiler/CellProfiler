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


class ij_bridge(object):
   '''This class provides a high-level interface for running ImageJ from
   CellProfiler. It is intended to abstract away whether IJ is being run within
   the same process or in a separate process.
   '''   
   def inject_image(self, pixel_data, name=None):
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




import cellprofiler.utilities.jutil as J
import imagej.macros as ijmacros
import imagej.windowmanager as ijwm
import imagej.imageprocessor as ijiproc
import imagej.imageplus as ijip

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
      
   def inject_image(self, pixel_data, name=''):
      '''inject an image into ImageJ for processing'''
      ij_processor = ijiproc.make_image_processor(pixel_data * 255.0)
      image_plus = ijip.make_imageplus_from_processor(name, ij_processor)
      if sys.platform == "darwin":
         ijwm.set_temp_current_image(image_plus)
      else:
         ijwm.set_current_image(image_plus)
   
   def get_current_image(self):
      '''returns the WindowManager's current image as a numpy float array'''
      image_plus = ijwm.get_current_image()
      ij_processor = image_plus.getProcessor()
      pixel_data = ijiproc.get_image(ij_processor) / 255.0
      return pixel_data
   
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
      


class inter_proc_ij_bridge(ij_bridge):
   '''Interface for running ImageJ in a separate process from CellProfiler.
   '''
   def __init__(self):
      os.environ['CLASSPATH'] = '/Users/afraser/CellProfiler/:'+os.environ['CLASSPATH']
      self.ijproc = Popen(shlex.split('java Adam'), stdin=PIPE, stdout=PIPE,
                          stderr=None)
   
##   def __del__(self):
##      os.kill(self.ijproc.pid)
   
   def inject_image(self, pixel_data, name=None):
      '''inject an image into ImageJ for processing'''
      tmpdir = tempfile.mkdtemp()
      filename = os.path.join(tmpdir, 'cp_image')

##      print 'INJECT_IMAGE: writing image to',filename
      
      if pixel_data.ndim == 3 and pixel_data.shape[2] == 4:
         mode = 'RGBA'
      elif pixel_data.ndim == 3:
         mode = 'RGB'
      else:
         mode = 'L'
      pil = PILImage.fromarray(pixel_data, mode)
      pil.save(filename, 'png')
      
      filename = '/Users/afraser/cpa_example/images/AS_09125_050116000001_A01f00d0.png'
      
      print communicate(self.ijproc, 'inject %s'%(filename))
      

   def get_current_image(self):
      '''returns the WindowManager's current image as a numpy float array'''
      tmpdir = tempfile.mkdtemp()
      filename = os.path.join(tmpdir, 'ij_image.png')
      
      print communicate('get %s'%(filename))
##      output = 'wrote file /Users/afraser/cpa_example/images/AS_09125_050116000001_A01f00d0.png'
      if output.startswith('wrote file '):
         filename = output.strip('wrote file ')
         pil = PILImage.open(filename, 'r')
         pil.show()
##         pixels = np.frombuffer(pil.tostring())
##         pixels.reshape(

   def get_commands(self):
      '''returns a list of the available command strings'''
      self.ijproc.stdin.write('get_commands\n')
      self.ijproc.stdin.flush()
      print self.ijproc.stdout.readlines()

   def execute_command(self, command, options=None):
      '''execute the named command within ImageJ'''
      self.ijproc.stdin.write('command %s\n'%(command))
      self.ijproc.stdin.flush()

   def execute_macro(self, macro_text):
      '''execute a macro in ImageJ
    
      macro_text - the macro program to be run
      '''
      self.ijproc.stdin.write('macro %s\n'%(macro_text))
      self.ijproc.stdin.flush()
   
   def show_imagej(self):
      '''show the ImageJ user interface'''
      self.ijproc.stdin.write('show_imagej\n')
      self.ijproc.stdin.flush()
      
   def quit(self):
      self.ijproc.stdin.write('quit\n')
      self.ijproc.stdin.flush()
      
      
def communicate(proc, inp):
   '''
   '''
   proc.stdin.write(inp+'\n')
   proc.stdin.flush()
   return proc.stdout.readline()
      

if __name__ == '__main__':
   ipb = inter_proc_ij_bridge()

   pixels = (np.ones((100,100)) * 255.).astype(np.uint8)
   ipb.inject_image(pixels, 'name ignored')

##   ipb.execute_macro('run("Invert");')
##
##   ipb.get_current_image()
##   
##   ipb.get_commands()
##
##   ipb.quit()
##   sys.exit()
   