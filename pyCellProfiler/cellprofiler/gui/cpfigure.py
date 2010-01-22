""" cpfigure.py - provides a frame with a figure inside

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import numpy as np
import os
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

window_ids = []

def window_name(module):
    '''Return a module's figure window name'''
    return "CellProfiler:%s:%s" % (module.module_name, module.module_num)

def find_fig(parent=None, title="", name=wx.FrameNameStr, subplots=None):
    """Find a figure frame window. Returns the window or None"""
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

def create_or_find(parent=None, id=-1, title="", 
                   pos=wx.DefaultPosition, size=wx.DefaultSize,
                   style=wx.DEFAULT_FRAME_STYLE, name=wx.FrameNameStr,
                   subplots=None,
                   on_close=None):
    """Create or find a figure frame window"""
    win = find_fig(parent, title, name, subplots)
    return win or CPFigureFrame(parent, id, title, pos, size, style, name, 
                                subplots, on_close)

def close_all(parent):
    windows = [x for x in parent.GetChildren()
               if isinstance(x, CPFigureFrame)]
    for window in windows:
        window.Close()
        
MENU_FILE_SAVE = wx.NewId()
MENU_CLOSE_WINDOW = wx.NewId()
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
                 subplots=None, on_close = None):
        """Initialize the frame:
        
        parent   - parent window to this one, typically CPFrame
        id       - window ID
        title    - title in title bar
        pos      - 2-tuple position on screen in pixels
        size     - 2-tuple size of frame in pixels
        style    - window style
        name     - searchable window name
        subplots - 2-tuple indicating the layout of subplots inside the window
        on_close - a function to run when the window closes
        """
        global window_ids
        super(CPFigureFrame,self).__init__(parent, id, title, pos, size, style, name)
        self.close_fn = on_close
        self.BackgroundColour = cpprefs.get_background_color()
        self.mouse_mode = MODE_NONE
        self.zoom_stack = []
        self.length_arrow = None
        self.colorbar = {}
        self.mouse_down = None
        self.remove_menu = []
        sizer = wx.BoxSizer()
        self.SetSizer(sizer)
        self.figure = figure= matplotlib.figure.Figure()
        self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure)
        sizer.Add(self.panel,1,wx.EXPAND) 
        self.status_bar = self.CreateStatusBar()
        wx.EVT_PAINT(self, self.on_paint)
        wx.EVT_CLOSE(self, self.on_close)
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
        if sys.platform.lower().startswith("win"):
            try:
                parent_menu_bar = parent.MenuBar
            except:
                # when testing, there may be no parent
                parent_menu_bar = None
            if (parent_menu_bar is not None and 
                isinstance(parent_menu_bar, wx.MenuBar)):
                for menu, label in parent_menu_bar.GetMenus():
                    if label == "Window":
                        menu_ids = [menu_item.Id 
                                    for menu_item in menu.MenuItems]
                        for window_id in window_ids+[None]:
                            if window_id not in menu_ids:
                                break
                        if window_id is None:
                            window_id = wx.NewId()
                            window_ids.append(window_id)
                        assert isinstance(menu,wx.Menu)
                        menu.Append(window_id, title)
                        def on_menu_command(event):
                            self.Raise()
                        wx.EVT_MENU(parent, window_id, on_menu_command)
                        self.remove_menu.append([menu, window_id])
    
    def add_menu(self):
        self.MenuBar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(MENU_FILE_SAVE,"&Save")
        wx.EVT_MENU(self, MENU_FILE_SAVE, self.on_file_save)
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
        accelerators = wx.AcceleratorTable(
            [(wx.ACCEL_CMD, ord('W'), MENU_CLOSE_WINDOW)])
        self.SetAcceleratorTable(accelerators)
        wx.EVT_MENU(self, MENU_CLOSE_WINDOW, self.on_close)
    
    def clf(self):
        '''Clear the figure window, resetting the display'''
        self.figure.clf()
        if hasattr(self,"subplots"):
            self.subplots[:,:] = None
        if hasattr(self,"zoom_rects"):
            self.zoom_rects[:,:] = None
        
    def on_paint(self, event):
        dc = wx.PaintDC(self)
        self.panel.draw(dc)
        event.Skip()
        del dc
    
    def on_close(self, event):
        if self.close_fn is not None:
            self.close_fn(event)
        for menu, menu_id in self.remove_menu:
            print "Removing menu ID %d"%menu_id
            self.Parent.Unbind(wx.EVT_MENU, id=menu_id)
            menu.Delete(menu_id)
        self.Destroy()

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
                if subplot and len(subplot.images) > 0:
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
        if not hasattr(self, "subplots"):
            return
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
        if not hasattr(self, "subplots"):
            return
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
                        axes = self.subplots[x,y]
                        if len(axes.images) == 0:
                            continue
                        if abs(x1 - x0) >= 5 and abs(y1-y0) >= 5:
                            if not old_limits:
                                old_x0,old_x1 = axes.get_xlim()
                                old_y0,old_y1 = axes.get_ylim()  
                                old_limits = ((old_x0, old_x1),
                                              (old_y0, old_y1))
                            axes.set_xlim(x0,x1)
                            axes.set_ylim(y1,y0)
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
    
    def on_file_save(self, event):
        dlg = wx.FileDialog(self, "Save figure", 
                            wildcard = ("PDF file (*.pdf)|*.pdf|"
                                        "Png image (*.png)|*.png|"
                                        "Postscript file (*.ps)|*.ps"),
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = os.path.join(dlg.GetPath())
            if dlg.FilterIndex == 1:
                format = "png"
            elif dlg.FilterIndex == 0:
                format = "pdf"
            elif dlg.FilterIndex == 2:
                format = "ps"
            else:
                format = "pdf"
            self.figure.savefig(path, format = format)
            
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
        try:
            del self.images[(x,y)]
            del self.popup_menus[(x,y)]
        except: pass
        axes.clear()
        
        
    def show_imshow_popup_menu(self, (x, y), image, subplot):
        '''
        shows a popup menu at pos x,y with items to:
        - Show image histogram
        - Change contrast stretching
        '''
        # Manage a dict of popup menus keyed by each subplot (x,y) location
        if 'popup_menus' not in self.__dict__:
            self.popup_menus = {}
        popup = self.popup_menus.get(subplot, None)
        if popup == None:
            # If no popup has been built for this subplot yet, then create one 
            MENU_CONTRAST_NORMAL = wx.NewId()
            MENU_CONTRAST_STRETCH = wx.NewId()
            MENU_CONTRAST_LOG = wx.NewId()
            self.popup_menus[subplot] = popup = wx.Menu()
            show_hist_item = wx.MenuItem(popup, -1, 'Show image histogram')
            popup.AppendItem(show_hist_item)
            submenu = wx.Menu()
            submenu.Append(MENU_CONTRAST_NORMAL, 'Normal', 'Do not transform pixel intensities', wx.ITEM_RADIO)
            item_stretch = submenu.Append(MENU_CONTRAST_STRETCH, 'Stretched', 'Stretch pixel intensities to fit the interval [0,1]', wx.ITEM_RADIO)
            submenu.Append(MENU_CONTRAST_LOG, 'Log stretched', 'Log transform pixel intensities, then stretch them to fit the interval [0,1]', wx.ITEM_RADIO)
            popup.AppendMenu(-1, 'Image contrast', submenu)
            
            def show_hist(evt):
                '''Callback for "Show image histogram" popup menu item'''
                new_title = '%s %s image histogram'%(self.Title, subplot)
                fig = create_or_find(self, -1, new_title, subplots=(1,1), name=new_title)
                fig.subplot_histogram(0, 0, image.flatten(), bins=200, xlabel='pixel intensity')
                fig.figure.canvas.draw()
                
            def change_contrast(evt):
                '''Callback for Image contrast menu items'''
                def log_transform(im):
                    '''returns log(image) scaled to the interval [0,1]'''
                    (min, max) = (im.min(), im.max())
                    if np.any((im>min)&(im<max)):
                        im = im.clip(im[im>0].min(), im.max())
                        im = np.log(im)
                        im -= im.min()
                        if im.max() > 0:
                            im /= im.max()
                    return im
                
                def auto_contrast(im):
                    '''returns image scaled to the interval [0,1]'''
                    im = im.copy()
                    (min, max) = (im.min(), im.max())
                    # Check that the image isn't binary 
                    if np.any((im>min)&(im<max)):
                        im -= im.min()
                        if im.max() > 0:
                            im /= im.max()
                    return im
                
                ax = self.subplot(subplot[0], subplot[1])
                if evt.Id == MENU_CONTRAST_NORMAL:
                    new_image = self.images[subplot]
                elif evt.Id == MENU_CONTRAST_STRETCH:
                    new_image = auto_contrast(self.images[subplot])
                elif evt.Id == MENU_CONTRAST_LOG:
                    new_image = log_transform(self.images[subplot])
                ax.get_images()[0].set_array(new_image)
                self.figure.canvas.draw()
            self.Bind(wx.EVT_MENU, show_hist, show_hist_item)
            self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_NORMAL)
            self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_STRETCH)
            self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_LOG)
        self.PopupMenu(popup, (x,y))
    
    
    def subplot_imshow(self, x,y, image, title=None, clear=True,
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
        if 'images' not in self.__dict__:
            self.images= {}
        if clear:
            self.clear_subplot(x, y)
        # Store the raw image keyed by it's subplot location
        self.images[(x,y)] = image
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
            
        def on_release(evt):
            if evt.inaxes == subplot:
                self.show_imshow_popup_menu((evt.x, self.figure.canvas.GetSize()[1]-evt.y), image, subplot=(x,y))
        # NOTE: We bind this event each time imshow is called to a new closure
        #    of on_release so that each function will be called when a
        #    button_release_event is fired.  It might be cleaner to bind the
        #    event outside of subplot_imshow, and define a handler that iterates
        #    through each subplot to determine what kind of action should be
        #    taken. In this case each subplot_xxx call would have to append
        #    an action response to a dictionary keyed by subplot.
        self.figure.canvas.mpl_connect('button_release_event', on_release)
        
        # Attempt to update histogram plot if one was created
        hist_fig = find_fig(self, name='%s %s image histogram'%(self.Title, (x,y)))
        if hist_fig:
            hist_fig.subplot_histogram(0, 0, image.flatten(), bins=200, xlabel='pixel intensity')
            hist_fig.figure.canvas.draw()
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
            
        def on_release(evt):
            if evt.inaxes== subplot:
                self.show_imshow_popup_menu((evt.x, self.figure.canvas.GetSize()[1]-evt.y), image, subplot=(x,y)) 
        self.figure.canvas.mpl_connect('button_release_event', on_release)
        
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
        
    def subplot_scatter(self, x, y, points, 
                        xlabel='',
                        ylabel='',
                        xscale='linear',
                        yscale='linear',
                        title='',
                        clear=True):
        """Put a scatterplot into a subplot
        
        x,y - subplot's column and row
        points - values to plot (a sequence of pairs)
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g. 'log' or 'linear')
        yscale - scaling of the y axis (e.g. 'log' or 'linear')
        title  - string title for the plot
        """
        self.figure.set_facecolor((1,1,1))
        self.figure.set_edgecolor((1,1,1))
        points = np.array(points)
        if clear:
            self.clear_subplot(x, y)

        axes = self.subplot(x, y)
        plot = axes.scatter(points[:,0], points[:,1],
                            facecolor=(0.0, 0.62, 1.0),
                            edgecolor='none',
                            alpha=0.75)
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        
        return plot
        
    def subplot_histogram(self, x, y, points,
                          bins=20, 
                          xlabel='',
                          xscale=None,
                          yscale='linear',
                          title='',
                          clear=True):
        """Put a histogram into a subplot
        
        x,y - subplot's column and row
        points - values to plot
        bins - number of bins to aggregate data in
        xlabel - string label for x axis
        xscale - 'log' to log-transform the data
        yscale - scaling of the y axis (e.g. 'log')
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1,1,1))
        self.figure.set_edgecolor((1,1,1))
        points = np.array(points)
        if xscale=='log':
            points = np.log(points[points>0])
            xlabel = 'Log(%s)'%(xlabel or '?')
        # hist apparently doesn't like nans, need to preen them out first
        self.points = points[~ np.isnan(points)]
        # nothing to plot?
        if len(points)==0 or points==[[]]: return
        
        axes = self.subplot(x, y)
        plot = axes.hist(points, bins, 
                          facecolor=(0.0, 0.62, 1.0), 
                          edgecolor='none',
                          log=(yscale=='log'),
                          alpha=0.75)
        axes.set_xlabel(xlabel)
        axes.set_title(title)
        
        return plot


    def subplot_density(self, x, y, points,
                        gridsize=100,
                        xlabel='',
                        ylabel='',
                        xscale='linear',
                        yscale='linear',
                        bins=None, 
                        cmap='jet',
                        title='',
                        clear=True):
        """Put a histogram into a subplot
        
        x,y - subplot's column and row
        points - values to plot
        gridsize - x & y bin size for data aggregation
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g. 'log' or 'linear')
        yscale - scaling of the y axis (e.g. 'log' or 'linear')
        bins - scaling of the color map (e.g. None or 'log', see mpl.hexbin)
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1,1,1))
        self.figure.set_edgecolor((1,1,1))
        
        points = np.array(points)
        
        # Clip to positives if in log space
        if xscale == 'log':
            points = points[(points[:,0]>0)]
        if yscale == 'log':
            points = points[(points[:,1]>0)]
        
        # nothing to plot?
        if len(points)==0 or points==[[]]: return
            
        axes = self.subplot(x, y)
        plot = axes.hexbin(points[:, 0], points[:, 1], 
                           gridsize=gridsize,
                           xscale=xscale,
                           yscale=yscale,
                           bins=bins,
                           cmap=matplotlib.cm.get_cmap(cmap))
        cb = self.figure.colorbar(plot)
        if bins=='log':
            cb.set_label('log10(N)')
            
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        
        xmin = np.nanmin(points[:,0])
        xmax = np.nanmax(points[:,0])
        ymin = np.nanmin(points[:,1])
        ymax = np.nanmax(points[:,1])

        # Pad all sides
        if xscale=='log':
            xmin = xmin/1.5
            xmax = xmax*1.5
        else:
            xmin = xmin-(xmax-xmin)/20.
            xmax = xmax+(xmax-xmin)/20.
            
        if yscale=='log':
            ymin = ymin/1.5
            ymax = ymax*1.5
        else:
            ymin = ymin-(ymax-ymin)/20.
            ymax = ymax+(ymax-ymin)/20.

        axes.axis([xmin, xmax, ymin, ymax])
        
        return plot
        
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
            self.frame = CPFigureFrame(subplots=(1, 1))
            menu = wx.Menu()
            menu.Append(ID_TEST_ADD_IMAGE, "Add image")

            def add_image(event):
                self.frame.clf()
                img = np.random.uniform(size=(1000, 1000, 3))
                self.frame.subplot_imshow_color(0, 0, img, "Random image")
                self.frame.figure.canvas.draw()
                gc.collect()

            wx.EVT_MENU(self.frame, ID_TEST_ADD_IMAGE, add_image)
            self.frame.MenuBar.Append(menu, "Test")
            self.SetTopWindow(self.frame)
            self.frame.Show()
            return True

    app = MyApp()
    app.MainLoop()
