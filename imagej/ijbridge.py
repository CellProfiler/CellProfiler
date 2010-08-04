import sys

class ij_bridge(object):
   '''This class provides a high-level interface for running ImageJ from
   CellProfiler. It is intended to abstract away whether IJ is being run within
   the same process or in a separate process.
   '''   
   def inject_image(self, image, name=None):
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
   def inject_image(self, image, name=None):
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

      
      
def get_imageplus_wrapper(imageplus_obj):
   '''Wrap the imageplus object as a Java class'''
   class ImagePlus(object):
      def __init__(self):
         self.o = imageplus_obj
      def lockSilently(self): pass
      def unlock(self): pass
      def getBitDepth(self): pass
      def getBytesPerPixel(self): pass
      def getChannel(self): pass
      def getChannelProcessor(self): pass
      def getCurrentSlice(self): pass
      def getDimensions(self): pass
      def getFrame(self): pass
      def getHeight(self): pass
      def getID(self): pass
      def getImageStackSize(self): pass
      def getNChannels(self): pass
      def getNDimensions(self): pass
      def getNFrames(self): pass
      def getNSlices(self): pass
      def getProcessor(self): pass
      def getTitle(self): pass
      def getWidth(self): pass
      def setPosition(self, channel, slice, frame): pass
      def setSlice(self, slice): pass
      def setTitle(self, title): pass
      def show(self): pass
      def show_with_message(self, message): pass
      def hide(self): pass
      def getWindow(self): pass
   return ImagePlus()
      