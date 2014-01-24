"""DirectoryView.py - a directory viewer geared to image directories

    TODO - long-term, this should Matlab imformats or CPimread to get the list of images
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import logging
import os
import sys
import traceback
import wx

logger = logging.getLogger(__name__)
import scipy.io.matlab
import matplotlib
import matplotlib.image
import matplotlib.figure
import matplotlib.backends.backend_wx

import cellprofiler.preferences
from cellprofiler.modules.loadimages import LoadImagesImageProvider, is_image
import cellprofiler.gui.cpfigure as FIG

class DirectoryView(object):
    """A directory viewer that displays file names and has smartish clicks
    
    """
    
    def __init__(self,panel):
        """Build a listbox into the panel to display the directory
        
        """
        self.__image_extensions = ['bmp','cur','fts','fits','gif','hdf','ico','jpg','jpeg','pbm','pcx','pgm','png','pnm','ppm','ras','tif','tiff','xwd','dib','c01','mat','fig','zvi']
        self.__list_box = wx.ListBox(panel,-1)
        sizer = wx.BoxSizer()
        sizer.Add(self.__list_box,1,wx.EXPAND)
        panel.SetSizer(sizer)
        self.__best_height = 0
        self.refresh()
        cellprofiler.preferences.add_image_directory_listener(self.__on_image_directory_changed)
        panel.Bind(wx.EVT_LISTBOX_DCLICK,self.__on_list_box_d_click,self.__list_box)
        self.__pipeline_listeners = []
    
    def close(self):
        '''Disconnect from the preferences when the window closes'''
        cellprofiler.preferences.remove_image_directory_listener(self.__on_image_directory_changed)
        
    def add_pipeline_listener(self,listener):
        """Add a listener that will be informed when the user wants to open a pipeline
        
        The listener should be a function to be called back with the parameters:
        * caller - the directory view
        * event - a LoadPipelineRequestEvent whose Path is the pipeline to open
        """
        self.__pipeline_listeners.append(listener)
        
    def remove_pipeline_listener(self,listener):
        self.__pipeline_listeners.remove(listener)
        
    def notify_pipeline_listeners(self,event):
        """Notify all pipeline listeners of an event that indicates that the user
        wants to open a pipeline
        
        """
        for listener in self.__pipeline_listeners:
            listener(self,event)
            
    def set_height(self,height):
        self.__best_height = height
    
    def refresh(self):
        try:
            self.__list_box.Clear()
        except wx.PyDeadObjectError:
            # Refresh can get called when the image directory changes, even
            # after this window has closed down.
            logger.warning("Warning: GUI not available during directoryview refresh\n")
            return
        try:
            files = [x 
                     for x in os.listdir(cellprofiler.preferences.get_default_image_directory()) 
                     if is_image(x) or x.endswith(".cp")]
        except Exception, e:
            logger.warning(
                "Warning: Could not refresh default image directory %s.\n" %
                (cellprofiler.preferences.get_default_image_directory()),
                exc_info = True)
            files = ['Could not refresh files (%s)'%(e.__class__.__name__)]
        files.sort()
        self.__list_box.AppendItems(files)
    
    def __on_image_directory_changed(self,event):
        self.refresh()
    
    def __on_list_box_d_click(self,event):
        selections = self.__list_box.GetSelections()
        if len(selections) > 0:
            selection = self.__list_box.GetItems()[selections[0]]
        else:
            selection = self.__list_box.GetItems()[self.__list_box.GetSelection()]
        filename = os.path.join(cellprofiler.preferences.get_default_image_directory(),selection)

        try:
            if os.path.splitext(selection)[1].lower() == '.cp':
                self.notify_pipeline_listeners(LoadPipelineRequestEvent(filename))
            elif os.path.splitext(selection)[1].lower() == '.mat':
                # A matlab file might be an image or a pipeline
                handles=scipy.io.matlab.mio.loadmat(filename, struct_as_record=True)
                if handles.has_key('Image'):
                    self.__display_matlab_image(handles, filename)
                else:
                    self.notify_pipeline_listeners(LoadPipelineRequestEvent(filename))
            else:
                self.__display_image(filename)
        except Exception, x:
            logger.error("Failed to display image", exc_info=True)
            wx.MessageBox("Unable to display %s.\n%s"%
                          (selection, str(x)),"Failed to display image")
    
    def __display_matlab_image(self,handles, filename):
        image=handles["Image"]
        frame = FIG.CPFigureFrame(self.__list_box.GetTopLevelParent(),
                                  title = filename,
                                  subplots = (1,1))
        if image.ndim == 3:
            frame.subplot_imshow(0,0,image,filename)
        else:
            frame.subplot_imshow_grayscale(0,0,image,filename)
        frame.Refresh()
    
    def __display_image(self,filename):
        lip = LoadImagesImageProvider("dummy", "", filename, True)
        image = lip.provide_image(None).pixel_data
        frame = FIG.CPFigureFrame(self.__list_box.GetTopLevelParent(),
                                  title = filename,
                                  subplots=(1,1))
        if image.ndim == 3:
            frame.subplot_imshow(0,0,image,filename)
        else:
            frame.subplot_imshow_grayscale(0,0,image,filename)
        frame.Refresh()

class LoadPipelineRequestEvent:
    """The user wants to load a pipeline
    
    This event represents some user action that might indicate that
    they want to load a pipeline. The Path attribute is the path+filename
    of the file to open.
    """
    def __init__(self,path):
        self.Path = path
