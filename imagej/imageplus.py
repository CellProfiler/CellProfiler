# imagej.imageplus.py - wrapper for imageplus class
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
import bioformats
import cellprofiler.utilities.jutil as J

def get_imageplus_wrapper(imageplus_obj):
    '''Wrap the imageplus object as a Java class'''
    class ImagePlus(object):
        def __init__(self):
            self.o = imageplus_obj
        
        lockSilently = J.make_method(
            'lockSilently', '()Z',
            '''Locks the image so other threads can test to see if it
            is in use. Returns true if the image was successfully locked.
            Beeps, displays a message in the status bar, and returns
            false if the image is already locked.''')
        unlock = J.make_method('unlock', '()V', 'Unlocks the image')
        getBitDepth = J.make_method('getBitDepth', '()I', 
                                    'Returns the bit depth: 8, 16, 24 or 32')
        getBytesPerPixel = J.make_method('getBytesPerPixel', '()I',
                                       'Returns the number of bytes per pixel')
        getChannel = J.make_method('getChannel', '()I')
        getChannelProcessor = J.make_method(
            'getChannelProcessor', '()Lij/process/ImageProcessor;',
            'Returns a reference to the current ImageProcessor.')
        getCurrentSlice = J.make_method(
            'getCurrentSlice', '()I', 'Returns the current stack slice number')
        def getDimensions(self):
            'Returns the dimensions of this image: (width, height, nChannels, nSlices, nFrames)'
            jresult = J.call(self.o, 'getDimensions', '()[I')
            return jresult
        
        getFrame = J.make_method('getFrame', '()I')
        getHeight = J.make_method('getHeight', '()I')
        getID = J.make_method('getID', '()I')
        getImageStackSize = J.make_method('getImageStackSize', '()I')
        getNChannels = J.make_method('getNChannels', '()I', 
                                   'Returns the number of channels')
        getNDimensions = J.make_method('getNDimensions', '()I',
                                       'Returns the number of dimensions')
        getNFrames = J.make_method('getNFrames', '()I',
                                   'Returns the number of frames (time-points)')
        getNSlices = J.make_method('getNSlices', '()I',
                                 'Returns the image depth (# of z-slices)')
        getProcessor = J.make_method('getProcessor', '()Lij/process/ImageProcessor;',
                                     'Returns a reference to the current image processor')
        getTitle = J.make_method('getTitle', '()Ljava/lang/String;')
        getWidth = J.make_method('getWidth', '()I')
        
        setPosition = J.make_method('setPosition', '(III)V',
                                    'setPosition(channel, slice, frame)')
        setSlice = J.make_method('setSlice', '(I)V')
        setTitle = J.make_method('setTitle', '(Ljava/lang/String;)V')
        
        def show(self):
            J.execute_runnable_in_main_thread(J.run_script("""
            new java.lang.Runnable() {
            run: function() { o.show(); }}""", dict(o=self.o)), synchronous=True)
        def hide(self):
            J.execute_runnable_in_main_thread(J.run_script("""
            new java.lang.Runnable() {
            run: function() { o.hide(); }}""", dict(o=self.o)), synchronous=True)
        getWindow = J.make_method('getWindow', '()Lij/gui/ImageWindow;',
                                  'Get the ImageWindow associated with this image. getWindow() will return null unless you have previously called show()')
    return ImagePlus()

def load_imageplus(file_name):
    '''Create an ImagePlus instance from a file'''
    imageplus_obj = J.make_instance('ij/ImagePlus',
                                    '(Ljava/lang/String;)V',
                                    file_name)
    return get_imageplus_wrapper(imageplus_obj)

def make_imageplus_from_processor(title, processor):
    '''Make an ImagePlus object from an ImageProcessor object
    
    title - the window title for the ImagePlus object
    processor - the ImageProcessor instance
    '''
    imageplus_obj = J.make_instance(
        'ij/ImagePlus',
        '(Ljava/lang/String;Lij/process/ImageProcessor;)V',
        title, processor)
    return get_imageplus_wrapper(imageplus_obj)
