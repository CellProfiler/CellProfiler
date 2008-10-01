"""DirectoryView.py - a directory viewer geared to image directories

    $Revision$
    TODO - long-term, this should Matlab imformats or CPimread to get the list of images
"""
import os
import wx
import CellProfiler.Preferences

class DirectoryView:
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
        self.Refresh()
        CellProfiler.Preferences.AddImageDirectoryListener(self.__OnImageDirectoryChanged)
    
    def SetHeight(self,height):
        self.__best_height = height
    
    def Refresh(self):
        self.__list_box.Clear()
        files = [x 
                 for x in os.listdir(CellProfiler.Preferences.GetDefaultImageDirectory()) 
                     if os.path.splitext(x)[1][1:] in self.__image_extensions]
        files.sort()
        self.__list_box.AppendItems(files)
    
    def __OnImageDirectoryChanged(self,event):
        self.Refresh()
    
    