'''windowmanager.py - functions to interact with the ImageJ window manager

'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import cellprofiler.utilities.jutil as J
from imagej.imageplus import get_imageplus_wrapper

def get_current_image():
    '''Get the WindowManager's current image
    
    returns a wrapped ImagePlus object
    '''
    imageplus_obj = J.static_call('ij/WindowManager','getCurrentImage',
                                  '()Lij/ImagePlus;')
    return get_imageplus_wrapper(imageplus_obj)

def get_id_list():
    '''Get the list of IDs of open images'''
    jid_list = J.static_call('ij/WindowManager', 'getIDList', '()[I')
    return jid_list

def get_image_by_id(imagej_id):
    '''Get an ImagePlus object by its ID'''
    return get_imageplus_wrapper(J.static_call(
        'ij/WindowManager', 'getImage', '(I)Lij/ImagePlus;', imagej_id))

def get_image_by_name(title):
    '''Get the ImagePlus object whose title (in the window) matches "title"'''
    return get_imageplus_wrapper(J.static_call(
        'ij/WindowManager', 'getImage', '(Ljava/lang/String;)Lij/ImagePlus;',
        title))

def get_temp_current_image():
    '''Get the temporary ImagePlus object for the current thread'''
    return get_imageplus_wrapper(J.static_call(
        'ij/WindowManager', 'getTempCurrentImage','()Lij/ImagePlus;'))

def make_unique_name(proposed_name):
    '''Create a unique title name for an imageplus object'''
    return J.static_call('ij/WindowManager', 'makeUniqueName',
                         '(Ljava/lang/String;)Ljava/lang/String;',
                         proposed_name)

def set_temp_current_image(imagej_obj):
    '''Set the temporary current image for this thread'''
    J.static_call('ij/WindowManager', 'setTempCurrentImage',
                  '(Lij/ImagePlus;)V', imagej_obj.o)

def set_current_image(imagej_obj):
    '''Set the currently active window'''
    imagej_obj.show()
    image_window = imagej_obj.getWindow()
    J.execute_runnable_in_main_thread(J.run_script(
        """new java.lang.Runnable() {
        run:function() { Packages.ij.WindowManager.setCurrentWindow(w); }}
        """, dict(w=image_window)), synchronous=True)

def close_all_windows():
    '''Close all ImageJ windows
    
    Hide the ImageJ windows so that they don't go through the Save dialog,
    then call the Window Manager's closeAllWindows to get the rest.
    '''
    jimage_list = J.static_call('ij/WindowManager', 'getIDList', '()[I')
    if jimage_list is None:
        return
    image_list = J.get_env().get_int_array_elements(jimage_list)
    for image_id in image_list:
        ip = J.static_call('ij/WindowManager', 'getImage', 
                           '(I)Lij/ImagePlus;', image_id)
        ip = get_imageplus_wrapper(ip)
        ip.hide()
    J.static_call('ij/WindowManager', 'closeAllWindows', '()Z')
    