"""DirectoryView.py - a directory viewer geared to image directories

    TODO - long-term, this should Matlab imformats or CPimread to get the list of images
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import os
import sys
import wx

import scipy.io.matlab
import PIL.Image
import matplotlib
import matplotlib.image
import matplotlib.figure
import matplotlib.backends.backend_wx

import cellprofiler.preferences


class DirectoryView(object):
    """A directory viewer that displays file names and has smartish clicks
    
    """
    
    def __init__(self,panel):
        """Build a listbox into the panel to display the directory
        
        """
        self.__image_extensions = ['bmp','cur','fts','fits','gif','hdf','ico','jpg','jpeg','pbm','pcx','pgm','png','pnm','ppm','ras','tif','tiff','xwd','dib','mat','fig','zvi']
        self.__list_box = wx.ListBox(panel,-1)
        sizer = wx.BoxSizer()
        sizer.Add(self.__list_box,1,wx.EXPAND)
        panel.SetSizer(sizer)
        self.__best_height = 0
        self.refresh()
        cellprofiler.preferences.add_image_directory_listener(self.__on_image_directory_changed)
        panel.Bind(wx.EVT_LISTBOX_DCLICK,self.__on_list_box_d_click,self.__list_box)
        self.__pipeline_listeners = []
    
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
            sys.stderr.write("Warning: GUI not available during directoryview refresh\n")
            return
        files = [x 
                 for x in os.listdir(cellprofiler.preferences.get_default_image_directory()) 
                     if os.path.splitext(x)[1][1:].lower() in self.__image_extensions]
        files.sort()
        self.__list_box.AppendItems(files)
    
    def __on_image_directory_changed(self,event):
        self.refresh()
    
    def __on_list_box_d_click(self,event):
        selections = self.__list_box.GetSelections()
        if len(selections) > 0:
            selection = self.__list_box.GetItems()[selections[0]]
        filename = os.path.join(cellprofiler.preferences.get_default_image_directory(),selection)
        if os.path.splitext(selection)[1].lower() =='.mat':
            # A matlab file might be an image or a pipeline
            handles=scipy.io.matlab.mio.loadmat(filename, struct_as_record=True)
            if handles.has_key('Image'):
                self.__display_matlab_image(handles, filename)
            else:
                self.notify_pipeline_listeners(LoadPipelineRequestEvent(filename))
        else:
            self.__display_image(filename)
    
    def __display_matlab_image(self,handles, filename):
            frame = ImageFrame(self.__list_box.GetTopLevelParent(),
                               filename,
                               image=handles["Image"])
            frame.Show()
    
    def __display_image(self,filename):
        frame = ImageFrame(self.__list_box.GetTopLevelParent(),filename)
        frame.Show()

class ImageFrame(wx.Frame):
    def __init__(self,parent,filename, image=None):
        wx.Frame.__init__(self,parent,-1,filename)
        if image != None:
            self.__image = image
        else:
            pil_image = PIL.Image.open(filename)
            self.__image = matplotlib.image.pil_to_array(pil_image)
        sizer = wx.BoxSizer()
        self.__figure= matplotlib.figure.Figure()
        self.__axes = self.__figure.add_subplot(111)
        self.__axes.imshow(self.__image)
        self.__panel = matplotlib.backends.backend_wx.FigureCanvasWx(self,-1,self.__figure)
        sizer.Add(self.__panel,1,wx.EXPAND)
        self.SetSizerAndFit(sizer)
        self.Bind(wx.EVT_PAINT,self.on_paint)
        
    def on_paint(self,event):
        dc = wx.PaintDC(self)
        self.__panel.draw(dc)
        
class LoadPipelineRequestEvent:
    """The user wants to load a pipeline
    
    This event represents some user action that might indicate that
    they want to load a pipeline. The Path attribute is the path+filename
    of the file to open.
    """
    def __init__(self,path):
        self.Path = path
