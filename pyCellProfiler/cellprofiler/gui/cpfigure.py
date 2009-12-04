""" cpfigure.py - provides a frame with a figure inside

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import numpy as np
import wx
import matplotlib
import matplotlib.cm
import numpy.ma
import matplotlib.patches
import matplotlib.colorbar
import matplotlib.backends.backend_wxagg
import scipy.misc
from cStringIO import StringIO
import sys

from cellprofiler.gui import get_icon
import cellprofiler.preferences as cpprefs

def create_or_find(parent=None, id=-1, title="", 
                   pos=wx.DefaultPosition, size=wx.DefaultSize,
                   style=wx.DEFAULT_FRAME_STYLE, name=wx.FrameNameStr,
                   subplots=None ):
    """Create or find a figure frame window"""
    if parent:
        window = parent.FindWindowByName(name)
        if window:
            if len(title) and title != window.Title:
                window.Title = title
            window.clf()
            if subplots!=None:
                window.subplots = np.zeros(subplots,dtype=object)
                window.zoom_rects = np.zeros(subplots,dtype=object)
                
            return window
    return CPFigureFrame(parent, id, title, pos, size, style, name, subplots)

def close_all(parent):
    windows = [x for x in parent.GetChildren()
               if isinstance(x, CPFigureFrame)]
    for window in windows:
        window.Close()
        
MENU_FILE_SAVE = wx.NewId()
MENU_ZOOM_IN = wx.NewId()
MENU_ZOOM_OUT = wx.NewId()
MENU_TOOLS_SHOW_PIXEL_DATA = wx.NewId()

'''mouse tool mode - do nothing'''
MODE_NONE = 0

'''mouse tool mode - zoom in'''   
MODE_ZOOM = 1

'''mouse tool mode - show pixel data'''
MODE_SHOW_PIXEL_DATA = 2

class CPFigureFrame(wx.Frame):
    """A wx.Frame with a figure inside"""
    
    def __init__(self, parent=None, id=-1, title="", 
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE, name=wx.FrameNameStr, 
                 subplots=None):
        """Initialize the frame:
        
        parent   - parent window to this one, typically CPFrame
        id       - window ID
        title    - title in title bar
        pos      - 2-tuple position on screen in pixels
        size     - 2-tuple size of frame in pixels
        style    - window style
        name     - searchable window name
        subplots - 2-tuple indicating the layout of subplots inside the window
        """
        super(CPFigureFrame,self).__init__(parent, id, title, pos, size, style, name)
        self.BackgroundColour = cpprefs.get_background_color()
        self.mouse_mode = MODE_NONE
        self.zoom_stack = []
        self.length_arrow = None
        self.colorbar = {}
        self.mouse_down = None
        sizer = wx.BoxSizer()
        self.SetSizer(sizer)
        self.figure = figure= matplotlib.figure.Figure()
        self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure)
        sizer.Add(self.panel,1,wx.EXPAND) 
        self.status_bar = self.CreateStatusBar()
        wx.EVT_PAINT(self, self.on_paint)
        if subplots:
            self.subplots = np.zeros(subplots,dtype=object)
            self.zoom_rects = np.zeros(subplots,dtype=object)
        self.add_menu()
        self.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.SetIcon(get_icon())
        self.Fit()
        self.Show()
    
    def add_menu(self):
        self.MenuBar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(MENU_FILE_SAVE,"&Save")
        self.MenuBar.Append(self.__menu_file,"&File")
        
        self.__menu_zoom = wx.Menu()
        self.__menu_item_zoom_in = \
            self.__menu_zoom.AppendCheckItem(MENU_ZOOM_IN,"&Zoom in")
        wx.EVT_MENU(self,MENU_ZOOM_IN,self.on_zoom_in)
        self.__menu_item_zoom_out = \
            self.__menu_zoom.Append(MENU_ZOOM_OUT,"&Zoom out")
        wx.EVT_MENU(self,MENU_ZOOM_OUT,self.on_zoom_out)
        self.__menu_item_zoom_out.Enable(len(self.zoom_stack) > 0)
        self.MenuBar.Append(self.__menu_zoom, "&Zoom")
        
        self.__menu_tools = wx.Menu()
        self.__menu_item_show_pixel_data = \
            self.__menu_tools.AppendCheckItem(MENU_TOOLS_SHOW_PIXEL_DATA,
                                              "Show &pixel data")
        self.MenuBar.Append(self.__menu_tools, "&Tools")
        wx.EVT_MENU(self, MENU_TOOLS_SHOW_PIXEL_DATA, self.on_show_pixel_data)
    
    def clf(self):
        '''Clear the figure window, resetting the display'''
        self.figure.clf()
        self.subplots[:,:] = None
        self.zoom_rects[:,:] = None
        
    def on_paint(self, event):
        dc = wx.PaintDC(self)
        self.panel.draw(dc)
        event.Skip()
        del dc
    
    def on_zoom_in(self,event):
        if self.__menu_item_zoom_in.IsChecked():
            self.mouse_mode = MODE_ZOOM
            self.__menu_item_show_pixel_data.Check(False)
        elif self.mouse_mode == MODE_ZOOM:
            self.mouse_mode = MODE_NONE

    def on_zoom_out(self, event):
        if self.subplots != None and len(self.zoom_stack) > 0:
            old_extents = self.zoom_stack.pop()
            for subplot in self.subplots.flatten():
                if subplot:
                    subplot.set_xlim(old_extents[0][0],old_extents[0][1])
                    subplot.set_ylim(old_extents[1][0],old_extents[1][1])
        self.__menu_item_zoom_out.Enable(len(self.zoom_stack) > 0)
        self.Refresh()
    
    def on_show_pixel_data(self, event):
        if self.__menu_item_show_pixel_data.IsChecked():
            self.mouse_mode = MODE_SHOW_PIXEL_DATA
            self.__menu_item_zoom_in.Check(False)
            self.Layout()
        elif self.mouse_mode == MODE_SHOW_PIXEL_DATA:
            self.mouse_mode = MODE_NONE
            
    def on_button_press(self, event):
        if event.inaxes in self.subplots.flatten():
            self.mouse_down = (event.xdata,event.ydata)
            if self.mouse_mode == MODE_ZOOM:
                self.on_zoom_mouse_down(event)
            elif self.mouse_mode == MODE_SHOW_PIXEL_DATA:
                self.on_show_pixel_data_mouse_down(event)
    
    def on_zoom_mouse_down(self, event):
        for x in range(self.subplots.shape[0]):
            for y in range(self.subplots.shape[1]):
                plot = self.subplots[x,y]
                if plot:
                    self.zoom_rects[x,y] = \
                        matplotlib.patches.Rectangle(self.mouse_down, 
                                                     1, 1,
                                                     fill=False,
                                                     edgecolor='red',
                                                     linewidth=1,
                                                     linestyle='solid')
                    plot.add_patch(self.zoom_rects[x,y])
        self.figure.canvas.draw()
        self.Refresh()
    
    def on_show_pixel_data_mouse_down(self, event):
        pass
    
    def on_mouse_move(self, event):
        if self.mouse_down is None:
            x0 = event.xdata
            x1 = event.xdata
            y0 = event.ydata
            y1 = event.ydata
        else:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)
        if self.mouse_mode == MODE_ZOOM:
            self.on_mouse_move_zoom(event, x0, y0, x1, y1)
        elif self.mouse_mode == MODE_SHOW_PIXEL_DATA:
            self.on_mouse_move_show_pixel_data(event, x0, y0, x1, y1)
    
    def on_mouse_move_zoom(self, event, x0, y0, x1, y1):
        if event.inaxes in self.subplots.flatten() and self.mouse_down:
            for zoom_rect in self.zoom_rects.flatten():
                if zoom_rect:
                    zoom_rect.set_x(x0)
                    zoom_rect.set_y(y0)
                    zoom_rect.set_width(x1-x0)
                    zoom_rect.set_height(y1-y0)
            self.figure.canvas.draw()
            self.Refresh()
    
    def on_mouse_move_show_pixel_data(self, event, x0, y0, x1, y1):
        if event.xdata is None or event.ydata is None:
            return
        xi = int(event.xdata+.5)
        yi = int(event.ydata+.5)
        fields = ["X: %d"%xi, "Y: %d"%yi]
        if event.inaxes:
            images = event.inaxes.get_images()
            if len(images) == 1:
                image = images[0]
                array = image.get_array()
                if array.dtype.type == np.uint8:
                    def fn(x):
                        return float(x) / 255.0
                else:
                    def fn(x):
                        return x
                if array.ndim == 2:
                    fields += ["Intensity: %.4f"%fn(array[yi,xi])]
                elif array.ndim == 3:
                    fields += ["Red: %.4f"%fn(array[yi,xi,0]),
                               "Green: %.4f"%fn(array[yi,xi,1]),
                               "Blue: %.4f"%fn(array[yi,xi,2])]
        if self.mouse_down is not None:
            length = np.sqrt((x0-x1)**2 +(y0-y1)**2)
            fields.append("Length: %.1f"%length)
            if self.length_arrow is not None:
                self.length_arrow.remove()
            xinterval = event.inaxes.xaxis.get_view_interval()
            yinterval = event.inaxes.yaxis.get_view_interval()
            diagonal = np.sqrt((xinterval[1]-xinterval[0])**2 +
                                  (yinterval[1]-yinterval[0])**2)
            mutation_scale = min(int(length*100/diagonal), 20) 
            self.length_arrow =\
                matplotlib.patches.FancyArrowPatch((self.mouse_down[0],
                                                    self.mouse_down[1]),
                                                   (event.xdata,
                                                    event.ydata),
                                                   edgecolor='red',
                                                   arrowstyle='<->',
                                                   mutation_scale=mutation_scale)
            try:
                event.inaxes.add_patch(self.length_arrow)
            except:
                print "Failed to add arrow from %f,%f to %f,%f"%(x0,y0,x1,y1)
                self.length_arrow = None
            self.figure.canvas.draw()
            self.Refresh()
        self.status_bar.SetFields(fields)
    
    def on_button_release(self,event):
        if event.inaxes in self.subplots.flatten() and self.mouse_down:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)
            if self.mouse_mode == MODE_ZOOM:
                self.on_zoom_done( event, x0, y0, x1, y1)
            elif self.mouse_mode == MODE_SHOW_PIXEL_DATA:
                self.on_show_pixel_data_done(event, x0, y0, x1, y1)
        elif self.mouse_down:
            if self.mouse_mode == MODE_ZOOM:
                self.on_zoom_canceled(event)
            elif self.mouse_mode == MODE_SHOW_PIXEL_DATA:
                self.on_show_pixel_data_canceled(event)
        self.mouse_down = None
    
    def on_zoom_done(self, event, x0, y0, x1, y1):
            old_limits = None
            for x in range(self.subplots.shape[0]):
                for y in range(self.subplots.shape[1]):
                    if self.zoom_rects[x,y]:
                        self.zoom_rects[x,y].remove()
                        self.zoom_rects[x,y] = 0
                    if self.subplots[x,y]:
                        if abs(x1 - x0) >= 5 and abs(y1-y0) >= 5:
                            if not old_limits:
                                old_x0,old_x1 = self.subplots[x,y].get_xlim()
                                old_y0,old_y1 = self.subplots[x,y].get_ylim()  
                                old_limits = ((old_x0, old_x1),
                                              (old_y0, old_y1))
                            self.subplots[x,y].set_xlim(x0,x1)
                            self.subplots[x,y].set_ylim(y1,y0)
                            self.zoom_stack.append(old_limits)
                            self.__menu_item_zoom_out.Enable(True)
            self.figure.canvas.draw()
            self.Refresh()
    
    def on_zoom_canceled(self, event):
        # cancel if released outside of axes
        for x in range(self.subplots.shape[0]):
            for y in range(self.subplots.shape[1]):
                if self.zoom_rects[x,y]:
                    self.zoom_rects[x,y].remove()
                    self.zoom_rects[x,y] = 0
        self.figure.canvas.draw()
        self.Refresh()
    
    def on_show_pixel_data_done(self, event, x0, y0, x1, y1):
        self.on_show_pixel_data_canceled(event)
    
    def on_show_pixel_data_canceled(self, event):
        if self.length_arrow is not None:
            self.length_arrow.remove()
            self.length_arrow = None
        self.figure.canvas.draw()
        self.Refresh()
    
    def subplot(self,x,y):
        """Return the indexed subplot
        
        x - column
        y - row
        """
        if not self.subplots[x,y]:
            rows, cols = self.subplots.shape
            plot = self.figure.add_subplot(cols,rows,x+y*rows+1)
            self.subplots[x,y] = plot
        return self.subplots[x,y]
    
    def set_subplot_title(self,title,x,y):
        """Set a subplot's title in the standard format
        
        title - title for subplot
        x - subplot's column
        y - subplot's row
        """
        self.subplot(x,y).set_title(title,
                                   fontname=cpprefs.get_title_font_name(),
                                   fontsize=cpprefs.get_title_font_size())
    
    def clear_subplot(self, x, y):
        """Clear a subplot of its gui junk

        x - subplot's column
        y - subplot's row
        """
        axes = self.subplot(x,y)
        axes.clear()
    
    def subplot_imshow(self, x,y,image, title=None, clear=True,
                       colormap=None, colorbar=False, vmin=None, vmax=None):
        '''Show an image in a subplot
        
        x,y   - show image in this subplot
        image - image to show
        title - add this title to the subplot
        clear - clear the subplot axes before display if true
        colormap - for a grayscale or labels image, use this colormap
                   to assign colors to the image
        colorbar - display a colorbar if true
        ''' 
        if clear:
            self.clear_subplot(x, y)
        subplot = self.subplot(x,y)
        if colormap == None:
            result = subplot.imshow(image)
        else:
            result = subplot.imshow(image, colormap, vmin=vmin, vmax=vmax)
        if title != None:
            self.set_subplot_title(title, x, y)
        if colorbar:
            if self.colorbar.has_key(subplot):
                axc =self.colorbar[subplot]
            else:
                axc, kw = matplotlib.colorbar.make_axes(subplot)
                self.colorbar[subplot] = axc
            cb = matplotlib.colorbar.Colorbar(axc, result)
            result.colorbar = cb
        return result
    
    def subplot_imshow_color(self, x, y, image, title=None, clear=True, 
                             normalize=True, vmin=None,vmax=None):
        if clear:
            self.clear_subplot(x, y)
        if normalize:
            image = image.astype(np.float32)
            for i in range(3):
                im_min = np.min(image[:,:,i])
                im_max = np.max(image[:,:,i])
                if im_min != im_max:
                    image[:,:,i] -= im_min
                    image[:,:,i] /= (im_max - im_min)
        elif image.dtype.type == np.float64:
            image = image.astype(np.float32)
        subplot = self.subplot(x,y)
        result = subplot.imshow(image,vmin=vmin, vmax=vmax)
        if title != None:
            self.set_subplot_title(title, x, y)
        return result
    
    def subplot_imshow_labels(self, x,y,labels, title=None, clear=True, 
                              renumber=True):
        if renumber:
            labels = renumber_labels_for_display(labels)
        if np.all(labels == 0):
            cm=matplotlib.cm.gray
        else:
            cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
            cm.set_bad((0,0,0))
            labels = numpy.ma.array(labels, mask=labels==0)
        return self.subplot_imshow(x,y,labels,title,clear,cm)
    
    def subplot_imshow_grayscale(self, x,y,image, title=None, clear=True,
                                 vmin=None, vmax=None):
        if image.dtype.type == np.float64:
            image = image.astype(np.float32)
        return self.subplot_imshow(x, y, image, title, clear, 
                                   matplotlib.cm.Greys_r,
                                   vmin=vmin, vmax=vmax)
    
    def subplot_imshow_bw(self, x,y,image, title=None, clear=True):
        return self.subplot_imshow(x, y, image, title, clear, 
                                   matplotlib.cm.binary_r)
    
    def subplot_table(self, x, y, statistics, 
                      ratio = (.6, .4),
                      loc = 'center',
                      cellLoc = 'left',
                      clear = True):
        """Put a table into a subplot
        
        x,y - subplot's column and row
        statistics - a sequence of sequences that form the values to
                     go into the table
        ratio - the ratio of column widths
        loc   - placement of the table within the axes
        cellLoc - alignment of text within cells
        """
        if clear:
            self.clear_subplot(x, y)
            
        table_axes = self.subplot(x, y)
        table = table_axes.table(cellText=statistics,
                                 colWidths=ratio,
                                 loc=loc,
                                 cellLoc=cellLoc)
        table_axes.set_frame_on(False)
        table_axes.set_axis_off()
        table.auto_set_font_size(False)
        table.set_fontsize(cpprefs.get_table_font_size())
        # table.set_fontfamily(cpprefs.get_table_font_name())
    
def renumber_labels_for_display(labels):
    """Scramble the label numbers randomly to make the display more discernable
    
    The colors of adjacent indices in a color map are less discernable than
    those of far-apart indices. Nearby labels tend to be adjacent or close,
    so a random numbering has more color-distance between labels than a
    straightforward one
    """
    np.random.seed(0)
    nlabels = np.max(labels)
    if nlabels <= 255:
        label_copy = labels.astype(np.uint8)
    elif nlabels < 2**16:
        label_copy = labels.astype(np.uint16)
    else:
        label_copy = labels.copy()
    renumber = np.random.permutation(np.max(label_copy))
    label_copy[label_copy != 0] = renumber[label_copy[label_copy!=0]-1]+1
    return label_copy

def figure_to_image(figure):
    '''Convert a figure to a numpy array'''
    #
    # Save the figure as a .PNG and then load it using scipy.misc.imread
    #
    fd = StringIO()
    figure.savefig(fd, format='png')
    fd.seek(0)
    image = scipy.misc.imread(fd)
    return image[:,:,:3]

if __name__ == "__main__":
    import numpy as np
    import gc
    
    ID_TEST_ADD_IMAGE = wx.NewId()
    class MyApp(wx.App):
        def OnInit(self):
            wx.InitAllImageHandlers()
            self.frame = CPFigureFrame(subplots=(1,1))
            menu = wx.Menu()
            menu.Append(ID_TEST_ADD_IMAGE, "Add image")
            def add_image(event):
                self.frame.clf()
                img = np.random.uniform(size=(1000,1000,3))
                self.frame.subplot_imshow_color(0,0,img,"Random image")
                self.frame.figure.canvas.draw()
                gc.collect()
            wx.EVT_MENU(self.frame, ID_TEST_ADD_IMAGE, add_image)
            self.frame.MenuBar.Append(menu, "Test")
            self.SetTopWindow(self.frame)
            self.frame.Show()
            return True
    app = MyApp()
    app.MainLoop()
        
