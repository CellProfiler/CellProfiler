""" cpfigure.py - provides a frame with a figure inside

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import logging
import numpy as np
import os
import sys
import uuid
import wx
import matplotlib
import matplotlib.cm
import numpy.ma
import matplotlib.patches
import matplotlib.colorbar
import matplotlib.backends.backend_wxagg
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import cellprofiler.utilities.matplotlib_axes_monkey_patch
from cellprofiler.preferences import update_cpfigure_position, get_next_cpfigure_position, reset_cpfigure_position
import scipy.misc
from scipy.sparse import coo_matrix
from cStringIO import StringIO
import sys

from cellprofiler.gui import get_cp_icon
from cellprofiler.gui.help import make_help_menu, FIGURE_HELP
import cellprofiler.preferences as cpprefs
from cellprofiler.cpmath.cpmorphology import distance_color_labels

g_use_imshow = False

def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    orig = im
    try:
        im = im.copy()
        im[np.isnan(im)] = 0
        (min, max) = (im[im > 0].min(), im[np.isfinite(im)].max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return orig

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

def is_color_image(im):
    return im.ndim==3 and im.shape[2]>=2


COLOR_NAMES = ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'White']
COLOR_VALS = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 1, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]

def wraparound(list):
    while True:
        for l in list:
            yield l

def make_1_or_3_channels(im):
    if im.ndim == 2 or im.shape[2] == 1:
        return im.astype(np.float32)
    if im.shape[2] == 3:
        return (im * 255).clip(0, 255).astype(np.uint8)
    out = np.zeros((im.shape[0], im.shape[1], 3), np.float32)
    for chanidx, weights in zip(range(im.shape[2]), wraparound(COLOR_VALS)):
        for idx, v in enumerate(weights):
            out[:, :, idx] += v * im[:, :, chanidx]
    return (out * 255).clip(0, 255).astype(np.uint8)

def make_3_channels_float(im):
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[:,:,0]
    if im.ndim == 2:
        return np.dstack((im,im,im)).astype(np.double).clip(0,1)
    out = np.zeros((im.shape[0], im.shape[1], 3), np.double)
    for chanidx, weights in zip(range(im.shape[2]), wraparound(COLOR_VALS)):
        for idx, v in enumerate(weights):
            out[:, :, idx] += v * im[:, :, chanidx]
    return out.clip(0,1)

def getbitmap(im):
    if im.ndim == 2:
        im = (255 * np.dstack((im, im, im))).astype(np.uint8)
    h, w, _ = im.shape
    outim = wx.EmptyImage(w, h)
    b = buffer(im) # make sure buffer exists through the remainder of function
    outim.SetDataBuffer(b)
    return outim.ConvertToBitmap()

def match_rgbmask_to_image(rgb_mask, image):
    rgb_mask = list(rgb_mask) # copy
    nchannels = image.shape[2]
    del rgb_mask[nchannels:]
    if len(rgb_mask) < nchannels:
        rgb_mask = rgb_mask + [1] * (nchannels - len(rgb_mask))
    return rgb_mask

    

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
               if isinstance(x, wx.Frame)]
        
    for window in windows:
        if isinstance(window, CPFigureFrame):
            window.on_close(None)
        else:
            window.Destroy()
        
    reset_cpfigure_position()
    try:
        from imagej.windowmanager import close_all_windows
        from cellprofiler.utilities.jutil import attach, detach
        attach()
        try:
            close_all_windows()
        finally:
            detach()
    except:
        pass
        
MENU_FILE_SAVE = wx.NewId()
MENU_CLOSE_WINDOW = wx.NewId()
MENU_TOOLS_MEASURE_LENGTH = wx.NewId()
MENU_CLOSE_ALL = wx.NewId()

'''mouse tool mode - do nothing'''
MODE_NONE = 0

'''mouse tool mode - show pixel data'''
MODE_MEASURE_LENGTH = 2

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
        if pos == wx.DefaultPosition:
            pos = get_next_cpfigure_position()
        super(CPFigureFrame,self).__init__(parent, id, title, pos, size, style, name)
        self.close_fn = on_close
        self.BackgroundColour = cpprefs.get_background_color()
        self.mouse_mode = MODE_NONE
        self.length_arrow = None
        self.images = {}
        self.colorbar = {}
        self.subplot_params = {}
        self.subplot_user_params = {}
        self.event_bindings = {}
        self.popup_menus = {}
        self.subplot_menus = {}
        self.mouse_down = None
        self.remove_menu = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        if cpprefs.get_use_more_figure_space():
            matplotlib.rcParams.update(dict([('figure.subplot.left', 0.025),
                                             ('figure.subplot.right', 0.975),
                                             ('figure.subplot.top', 0.975),
                                             ('figure.subplot.bottom', 0.025),
                                             ('figure.subplot.wspace', 0.05),
                                             ('figure.subplot.hspace', 0.05),
                                             ('axes.labelsize', 'x-small'),
                                             ('xtick.labelsize', 'x-small'),
                                             ('ytick.labelsize', 'x-small')]))
        else:
            matplotlib.rcdefaults()
        self.figure = figure = matplotlib.figure.Figure()
        self.panel = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self, -1, self.figure)
        sizer.Add(self.panel, 1, wx.EXPAND) 
        self.status_bar = self.CreateStatusBar()
        wx.EVT_PAINT(self, self.on_paint)
        wx.EVT_CLOSE(self, self.on_close)
        if subplots:
            self.subplots = np.zeros(subplots,dtype=object)
        self.create_menu()
        self.create_toolbar()
        self.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_button_release)
        try:
            self.SetIcon(get_cp_icon())
        except:
            pass
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
    
    def create_menu(self):
        self.MenuBar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(MENU_FILE_SAVE,"&Save")
        wx.EVT_MENU(self, MENU_FILE_SAVE, self.on_file_save)
        self.MenuBar.Append(self.__menu_file,"&File")
                
        self.__menu_tools = wx.Menu()
        self.__menu_item_measure_length = \
            self.__menu_tools.AppendCheckItem(MENU_TOOLS_MEASURE_LENGTH,
                                              "Measure &length")
        self.MenuBar.Append(self.__menu_tools, "&Tools")
        
        self.menu_subplots = wx.Menu()
        self.MenuBar.Append(self.menu_subplots, 'Subplots')
            
        wx.EVT_MENU(self, MENU_TOOLS_MEASURE_LENGTH, self.on_measure_length)

        # work around mac window menu losing bindings
        if wx.Platform == '__WXMAC__':        
            hidden_menu = wx.Menu()
            hidden_menu.Append(MENU_CLOSE_ALL, "&L")
            self.Bind(wx.EVT_MENU, lambda evt: close_all(self.Parent), id=MENU_CLOSE_ALL)
            accelerators = wx.AcceleratorTable(
                [(wx.ACCEL_CMD, ord('W'), MENU_CLOSE_WINDOW),
                 (wx.ACCEL_CMD, ord('L'), MENU_CLOSE_ALL)])
        else:
            accelerators = wx.AcceleratorTable(
                [(wx.ACCEL_CMD, ord('W'), MENU_CLOSE_WINDOW)])

        self.SetAcceleratorTable(accelerators)
        wx.EVT_MENU(self, MENU_CLOSE_WINDOW, self.on_close)
        self.MenuBar.Append(make_help_menu(FIGURE_HELP, self), "&Help")
    
    def create_toolbar(self):
        self.navtoolbar = NavigationToolbar(self.figure.canvas)
        self.SetToolBar(self.navtoolbar)
        if wx.VERSION != (2, 9, 1, 1, ''):
            # avoid crash on latest wx 2.9
            self.navtoolbar.DeleteToolByPos(6)
#        ID_LASSO_TOOL = wx.NewId()
#        lasso = self.navtoolbar.InsertSimpleTool(5, ID_LASSO_TOOL, lasso_tool.ConvertToBitmap(), '', '', isToggle=True)
#        self.navtoolbar.Realize()
#        self.Bind(wx.EVT_TOOL, self.toggle_lasso_tool, id=ID_LASSO_TOOL)

    def clf(self):
        '''Clear the figure window, resetting the display'''
        self.figure.clf()
        if hasattr(self,"subplots"):
            self.subplots[:,:] = None
        # Remove the subplot menus
        for (x,y) in self.subplot_menus:
            self.menu_subplots.RemoveItem(self.subplot_menus[(x,y)])
        for (x,y) in self.event_bindings:
            [self.figure.canvas.mpl_disconnect(b) for b in self.event_bindings[(x,y)]]
        self.subplot_menus = {}
        self.subplot_params = {}
        self.subplot_user_params = {}
        self.colorbar = {}
        self.images = {}
        
    def on_paint(self, event):
        dc = wx.PaintDC(self)
        self.panel.draw(dc)
        event.Skip()
        del dc
    
    def on_close(self, event):
        if self.close_fn is not None:
            self.close_fn(event)
        self.clf() # Free memory allocated by imshow
        for menu, menu_id in self.remove_menu:
            self.Parent.Unbind(wx.EVT_MENU, id=menu_id)
            menu.Delete(menu_id)
        self.Destroy()

    def on_measure_length(self, event):
        '''Measure length menu item selected.'''
        if self.__menu_item_measure_length.IsChecked():
            self.mouse_mode = MODE_MEASURE_LENGTH
            self.Layout()
        elif self.mouse_mode == MODE_MEASURE_LENGTH:
            self.mouse_mode = MODE_NONE
            
    def on_button_press(self, event):
        if not hasattr(self, "subplots"):
            return
        if event.inaxes in self.subplots.flatten():
            self.mouse_down = (event.xdata,event.ydata)
            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_mouse_down(event)
    
    def on_measure_length_mouse_down(self, event):
        pass

    def on_mouse_move(self, evt):
        #
        # LAAAME SAUCE -- Crosshair cursor is all black on Windows making it
        #    virtually invisible on dark images. Use custom cursor instead.
        #
        if (sys.platform.lower().startswith('win') and 
            evt.inaxes and
            'zoom rect' in self.navtoolbar.mode.lower()):  # NOTE: There are no constants for the navbar modes
            #
            # Build the crosshair cursor image as a numpy array.
            #
            buf = np.ones((16,16,3), dtype='uint8') * 255
            buf[7,1:-1,:] = buf[1:-1,7,:] = 0
            abuf = np.ones((16,16), dtype='uint8') * 255
            abuf[:6,:6] = abuf[9:,:6] = abuf[9:,9:] = abuf[:6,9:] = 0
            im = wx.ImageFromBuffer(16, 16, buf.tostring(), abuf.tostring())
            im.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 7)
            im.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 7)
            cursor = wx.CursorFromImage(im)
            self.figure.canvas.SetCursor(cursor)
            
        if self.mouse_down is None:
            x0 = evt.xdata
            x1 = evt.xdata
            y0 = evt.ydata
            y1 = evt.ydata
        else:
            x0 = min(self.mouse_down[0], evt.xdata)
            x1 = max(self.mouse_down[0], evt.xdata)
            y0 = min(self.mouse_down[1], evt.ydata)
            y1 = max(self.mouse_down[1], evt.ydata)
        if self.mouse_mode == MODE_MEASURE_LENGTH:
            self.on_mouse_move_measure_length(evt, x0, y0, x1, y1)
        elif not self.mouse_mode == MODE_MEASURE_LENGTH:
            self.on_mouse_move_show_pixel_data(evt, x0, y0, x1, y1)
    
    def get_pixel_data_fields_for_status_bar(self, im, xi, yi):
        fields = []
        if not self.in_bounds(im, xi, yi):
            return fields
        if im.dtype.type == np.uint8:
            im = im.astype(np.float32) / 255.0
        if im.ndim == 2:
            fields += ["Intensity: %.4f"%(im[yi,xi])]
        elif im.ndim == 3 and im.shape[2] == 3:
            fields += ["Red: %.4f"%(im[yi,xi,0]),
                       "Green: %.4f"%(im[yi,xi,1]),
                       "Blue: %.4f"%(im[yi,xi,2])]
        elif im.ndim == 3: 
            fields += ["Channel %d: %.4f"%(idx + 1, im[yi, xi, idx]) for idx in range(im.shape[2])]
        return fields
    
    @staticmethod
    def in_bounds(im, xi, yi):
        '''Return false if xi or yi are outside of the bounds of the image'''
        return not (im is None or xi >= im.shape[1] or yi >= im.shape[0]
                    or xi < 0 or yi < 0)

    def on_mouse_move_measure_length(self, event, x0, y0, x1, y1):
        if event.xdata is None or event.ydata is None:
            return
        xi = int(event.xdata+.5)
        yi = int(event.ydata+.5)
        im = None
        if event.inaxes:
            fields = ["X: %d"%xi, "Y: %d"%yi]
            im = self.find_image_for_axes(event.inaxes)
            if im is not None:
                fields += self.get_pixel_data_fields_for_status_bar(im, x1, yi)
                
        if self.mouse_down is not None and im is not None:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)
            
            length = np.sqrt((x0-x1)**2 +(y0-y1)**2)
            fields.append("Length: %.1f"%length)
            xinterval = event.inaxes.xaxis.get_view_interval()
            yinterval = event.inaxes.yaxis.get_view_interval()
            diagonal = np.sqrt((xinterval[1]-xinterval[0])**2 +
                               (yinterval[1]-yinterval[0])**2)
            mutation_scale = min(int(length*100/diagonal), 20) 
            if self.length_arrow is not None:
                self.length_arrow.set_positions((self.mouse_down[0],
                                                        self.mouse_down[1]),
                                                       (event.xdata,
                                                        event.ydata))
            else:
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
                    self.length_arrow = None
            self.figure.canvas.draw()
            self.Refresh()
        self.status_bar.SetFields(fields)
    
    def on_mouse_move_show_pixel_data(self, event, x0, y0, x1, y1):
        if event.xdata is None or event.ydata is None:
            return
        xi = int(event.xdata+.5)
        yi = int(event.ydata+.5)
        if event.inaxes:
            im = self.find_image_for_axes(event.inaxes)
            if im is not None:
                fields = ["X: %d"%xi, "Y: %d"%yi]
                fields += self.get_pixel_data_fields_for_status_bar(im, xi, yi)
                self.status_bar.SetFields(fields)
                return
            else:
                self.status_bar.SetFields([event.inaxes.format_coord(event.xdata, event.ydata)])
        
    def find_image_for_axes(self, axes):
        for i, sl in enumerate(self.subplots):
            for j, slax in enumerate(sl):
                if axes == slax:
                    return self.images.get((i, j), None)
        return None
    
    def on_button_release(self,event):
        if not hasattr(self, "subplots"):
            return
        if event.inaxes in self.subplots.flatten() and self.mouse_down:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)
            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_done(event, x0, y0, x1, y1)
        elif self.mouse_down:
            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_canceled(event)
        self.mouse_down = None
    
    def on_measure_length_done(self, event, x0, y0, x1, y1):
        self.on_measure_length_canceled(event)
    
    def on_measure_length_canceled(self, event):
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
            
    def subplot(self, x, y, sharex=None, sharey=None):
        """Return the indexed subplot
        
        x - column
        y - row
        sharex - If creating a new subplot, you can specify a subplot instance 
                 here to share the X axis with. eg: for zooming, panning
        sharey - If creating a new subplot, you can specify a subplot instance 
                 here to share the Y axis with. eg: for zooming, panning
        """
        if not self.subplots[x,y]:
            rows, cols = self.subplots.shape
            plot = self.figure.add_subplot(cols, rows, x + y * rows + 1,
                                           sharex=sharex, sharey=sharey)
            self.subplots[x,y] = plot
        return self.subplots[x,y]
    
    def set_subplot_title(self,title,x,y):
        """Set a subplot's title in the standard format
        
        title - title for subplot
        x - subplot's column
        y - subplot's row
        """
        fontname = fontname=cpprefs.get_title_font_name()
            
        self.subplot(x,y).set_title(title,
                                   fontname=fontname,
                                   fontsize=cpprefs.get_title_font_size())
    
    def clear_subplot(self, x, y):
        """Clear a subplot of its gui junk. Noop if no subplot exists at x,y

        x - subplot's column
        y - subplot's row
        """
        if not self.subplots[x,y]:
            return
        axes = self.subplot(x,y)
        try:
            del self.images[(x,y)]
            del self.popup_menus[(x,y)]
        except: pass
        axes.clear()
        
    def show_imshow_popup_menu(self, pos, subplot_xy):
        popup = self.get_imshow_menu(subplot_xy)
        self.PopupMenu(popup, pos)
        
    def get_imshow_menu(self, (x,y)):
        '''returns a menu corresponding to the specified subplot with items to:
        - launch the image in a new cpfigure window
        - Show image histogram
        - Change contrast stretching
        - Toggle channels on/off
        Note: Each item is bound to a handler.
        '''
        params = self.subplot_params[(x,y)]
            
        # If no popup has been built for this subplot yet, then create one 
        MENU_CONTRAST_RAW = wx.NewId()
        MENU_CONTRAST_NORMALIZED = wx.NewId()
        MENU_CONTRAST_LOG = wx.NewId()
        popup = wx.Menu()
        self.popup_menus[(x,y)] = popup
        open_in_new_figure_item = wx.MenuItem(popup, -1, 
                                              'Open image in new window')
        popup.AppendItem(open_in_new_figure_item)
        show_hist_item = wx.MenuItem(popup, -1, 'Show image histogram')
        popup.AppendItem(show_hist_item)
        
        submenu = wx.Menu()
        item_raw = submenu.Append(MENU_CONTRAST_RAW, 'Raw', 
                                  'Do not transform pixel intensities', 
                                  wx.ITEM_RADIO)
        item_normalized = submenu.Append(MENU_CONTRAST_NORMALIZED, 
                                         'Normalized', 
                                         'Stretch pixel intensities to fit '
                                         'the interval [0,1]', 
                                         wx.ITEM_RADIO)
        item_log = submenu.Append(MENU_CONTRAST_LOG, 'Log normalized', 
                                  'Log transform pixel intensities, then '
                                  'stretch them to fit the interval [0,1]', 
                                  wx.ITEM_RADIO)

        if params['normalize'] == 'log':
            item_log.Check()
        elif params['normalize'] == True:
            item_normalized.Check()
        else:
            item_raw.Check()
        popup.AppendMenu(-1, 'Image contrast', submenu)
        
        def open_image_in_new_figure(evt):
            '''Callback for "Open image in new window" popup menu item '''
            # Store current zoom limits
            xlims = self.subplot(x,y).get_xlim()
            ylims = self.subplot(x,y).get_ylim()
            new_title = self.subplot(x,y).get_title()
            fig = create_or_find(self, -1, new_title, subplots=(1,1), 
                                 name=str(uuid.uuid4()))
            fig.subplot_imshow(0, 0, self.images[(x,y)], **params)
            
            # XXX: Cheat here so the home button works.
            # This needs to be fixed so it copies the view history for the 
            # launched subplot to the new figure.
            fig.navtoolbar.push_current()

            # Set current zoom
            fig.subplot(0,0).set_xlim(xlims[0], xlims[1])
            fig.subplot(0,0).set_ylim(ylims[0], ylims[1])      
            fig.figure.canvas.draw()
        
        def show_hist(evt):
            '''Callback for "Show image histogram" popup menu item'''
            new_title = '%s %s image histogram'%(self.Title, (x,y))
            fig = create_or_find(self, -1, new_title, subplots=(1,1), name=new_title)
            fig.subplot_histogram(0, 0, self.images[(x,y)].flatten(), bins=200, xlabel='pixel intensity')
            fig.figure.canvas.draw()
            
        def change_contrast(evt):
            '''Callback for Image contrast menu items'''
            # Store zoom limits
            xlims = self.subplot(x,y).get_xlim()
            ylims = self.subplot(x,y).get_ylim()
            if evt.Id == MENU_CONTRAST_RAW:
                params['normalize'] = False
            elif evt.Id == MENU_CONTRAST_NORMALIZED:
                params['normalize'] = True
            elif evt.Id == MENU_CONTRAST_LOG:
                params['normalize'] = 'log'
            self.subplot_imshow(x, y, self.images[(x,y)], **params)
            # Restore plot zoom
            self.subplot(x,y).set_xlim(xlims[0], xlims[1])
            self.subplot(x,y).set_ylim(ylims[0], ylims[1])                
            self.figure.canvas.draw()
            
        if is_color_image(self.images[x,y]):
            submenu = wx.Menu()
            rgb_mask = match_rgbmask_to_image(params['rgb_mask'], self.images[x,y])
            ids = [wx.NewId() for _ in rgb_mask]
            for name, value, id in zip(wraparound(COLOR_NAMES), rgb_mask, ids):
                item = submenu.Append(id, name, 'Show/Hide the %s channel'%(name), wx.ITEM_CHECK)
                if value != 0:
                    item.Check()
            popup.AppendMenu(-1, 'Channels', submenu)
            
            def toggle_channels(evt):
                '''Callback for channel menu items.'''
                # Store zoom limits
                xlims = self.subplot(x,y).get_xlim()
                ylims = self.subplot(x,y).get_ylim()
                if 'rgb_mask' not in params:
                    params['rgb_mask'] = list(rgb_mask)
                else:
                    # copy to prevent modifying shared values
                    params['rgb_mask'] = list(params['rgb_mask'])
                for idx, id in enumerate(ids):
                    if id == evt.Id:
                        params['rgb_mask'][idx] = not params['rgb_mask'][idx]
                self.subplot_imshow(x, y, self.images[(x,y)], **params)
                # Restore plot zoom
                self.subplot(x,y).set_xlim(xlims[0], xlims[1])
                self.subplot(x,y).set_ylim(ylims[0], ylims[1])   
                self.figure.canvas.draw()

            for id in ids:
                self.Bind(wx.EVT_MENU, toggle_channels, id=id)
        
        self.Bind(wx.EVT_MENU, open_image_in_new_figure, open_in_new_figure_item)
        self.Bind(wx.EVT_MENU, show_hist, show_hist_item)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_RAW)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_NORMALIZED)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_LOG)
        return popup
    
    
    def subplot_imshow(self, x, y, image, title=None, clear=True, colormap=None,
                       colorbar=False, normalize=True, vmin=0, vmax=1, 
                       rgb_mask=(1, 1, 1), sharex=None, sharey=None,
                       use_imshow = False):
        '''Show an image in a subplot
        
        x, y  - show image in this subplot
        image - image to show
        title - add this title to the subplot
        clear - clear the subplot axes before display if true
        colormap - for a grayscale or labels image, use this colormap
                   to assign colors to the image
        colorbar - display a colorbar if true
        normalize - whether or not to normalize the image. If True, vmin, vmax
                    are ignored.
        vmin, vmax - Used to scale a luminance image to 0-1. If either is None, 
                     the min and max of the luminance values will be used.
                     If normalize is True, vmin and vmax will be ignored.
        rgb_mask - 3-element list to be multiplied to all pixel values in the
                   image. Used to show/hide individual channels in color images.
        sharex, sharey - specify a subplot to link axes with (for zooming and
                         panning). Specify a subplot using CPFigure.subplot(x,y)
        use_imshow - True to use Axes.imshow to paint images, False to fill
                     the image into the axes after painting.
        '''
        orig_vmin = vmin
        orig_vmax = vmax
        # NOTE: self.subplot_user_params is used to store changes that are made 
        #    to the display through GUI interactions (eg: hiding a channel).
        #    Once a subplot that uses this mechanism has been drawn, it will
        #    continually load defaults from self.subplot_user_params instead of
        #    the default values specified in the function definition.
        kwargs = {'title' : title,
                  'clear' : clear,
                  'colormap' : colormap,
                  'colorbar' : colorbar,
                  'normalize' : normalize,
                  'vmin' : vmin,
                  'vmax' : vmax,
                  'rgb_mask' : rgb_mask,
                  'use_imshow' : use_imshow}
        if (x,y) not in self.subplot_user_params:
            self.subplot_user_params[(x,y)] = {}
        if (x,y) not in self.subplot_params:
            self.subplot_params[(x,y)] = {}
        # overwrite keyword arguments with user-set values
        kwargs.update(self.subplot_user_params[(x,y)])
        self.subplot_params[(x,y)].update(kwargs)
        if kwargs["colormap"] is None:
            kwargs["colormap"] = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())

        # and fetch back out
        title = kwargs['title']
        clear = kwargs['clear']
        colormap = kwargs['colormap']
        colorbar = kwargs['colorbar']
        normalize = kwargs['normalize']
        vmin = kwargs['vmin']
        vmax = kwargs['vmax']
        rgb_mask = kwargs['rgb_mask']
        
        # Note: if we do not do this, then passing in vmin,vmax without setting
        # normalize=False will cause the normalized image to be stretched 
        # further which makes no sense.
        # ??? - We may want to change the normalize vs vmin,vmax behavior so if 
        # vmin,vmax are passed in, then normalize is ignored.
        if normalize != False:
            vmin, vmax = 0, 1
        
        if clear:
            self.clear_subplot(x, y)
        # Store the raw image keyed by it's subplot location
        self.images[(x,y)] = image
        
        # Draw (actual image drawing in on_redraw() below)
        subplot = self.subplot(x, y, sharex=sharex, sharey=sharey)
        subplot._adjustable = 'box-forced'
        subplot.plot([0, 0], list(image.shape[:2]), 'k')
        subplot.set_xlim([-0.5, image.shape[1] - 0.5])
        subplot.set_ylim([image.shape[0] - 0.5, -0.5])
        subplot.set_aspect('equal')

        # Set title
        if title != None:
            self.set_subplot_title(title, x, y)
        
        # Update colorbar
        if colorbar and not is_color_image(image):
            if not subplot in self.colorbar:
                cax = matplotlib.colorbar.make_axes(subplot)[0]
                self.colorbar[subplot] = (cax, matplotlib.colorbar.ColorbarBase(cax, cmap=colormap, ticks=[]))
            cax, _ = self.colorbar[subplot]
            cax.set_yticks(np.linspace(0, 1, 10))
            if normalize == True:
                cax.set_yticklabels(['%0.1f'%(v) for v in np.linspace(image.min(), image.max(), 10)])
            elif normalize == 'log':
                if image.max() > 0 and image.max() > image[image > 0].min():
                    lo = image[image > 0].min()
                    hi = image.max()
                    cax.set_yticklabels(['%0.1f'%(v) for v in lo * np.logspace(image.min(), image.max(), 10, base=(hi / lo))])
                else:
                    cax.set_yticklabels([''] * 10)
            elif (orig_vmin is not None) and (orig_vmax is not None):
                cax.set_yticklabels(['%0.1f'%(v) for v in np.linspace(orig_vmin, orig_vmax, 10)])
            else:
                cax.set_yticklabels(['%0.1f'%(v) for v in np.linspace(0, 1, 10)])
                                      

        # NOTE: We bind this event each time imshow is called to a new closure
        #    of on_release so that each function will be called when a
        #    button_release_event is fired.  It might be cleaner to bind the
        #    event outside of subplot_imshow, and define a handler that iterates
        #    through each subplot to determine what kind of action should be
        #    taken. In this case each subplot_xxx call would have to append
        #    an action response to a dictionary keyed by subplot.
        if (x,y) in self.event_bindings:
            [self.figure.canvas.mpl_disconnect(b) for b in self.event_bindings[(x,y)]]
            
        def on_release(evt):
            if evt.inaxes == subplot:
                if evt.button != 1:
                    self.show_imshow_popup_menu((evt.x, self.figure.canvas.GetSize()[1] - evt.y), (x,y))
        self.event_bindings[(x, y)] = [
            self.figure.canvas.mpl_connect('button_release_event', on_release)]

        if use_imshow or g_use_imshow:
            image = self.images[(x, y)]
            subplot.imshow(self.normalize_image(image, **kwargs))
        else:
            class CPImageArtist(matplotlib.artist.Artist):
                def __init__(self, image, frame, kwargs):
                    super(CPImageArtist, self).__init__()
                    self.image = image
                    self.frame = frame
                    self.kwargs = kwargs
                    #
                    # The radius for the gaussian blur of 1 pixel sd
                    #
                    self.filterrad = 4.0
                    
                def draw(self, renderer):
                    image = self.frame.normalize_image(self.image, 
                                                       **self.kwargs)
                    magnification = renderer.get_image_magnification()
                    #
                    # Code partially borrowed from matplotlib/image.py
                    # AxesImage.make_image
                    #
                    dxintv = image.shape[1]
                    dyintv = image.shape[0]
            
                    # the viewport scale factor
                    sx = dxintv/self.axes.viewLim.width
                    sy = dyintv/self.axes.viewLim.height
                    flip_ud = sy < 0
                    if flip_ud:
                        image = np.flipud(image)
                    sy = abs(sy)
                    numrows, numcols = self.image.shape[:2]
                    if sx > 2:
                        x0 = self.axes.viewLim.x0/dxintv * numcols
                        ix0 = max(0, int(x0 - self.filterrad))
                        x1 = self.axes.viewLim.x1/dxintv * numcols
                        ix1 = min(numcols, int(x1 + self.filterrad))
                        xslice = slice(ix0, ix1)
                        xmin = ix0*dxintv/numcols
                        xmax = ix1*dxintv/numcols
                        dxintv = xmax - xmin
                        sx = dxintv/self.axes.viewLim.width
                    else:
                        xmin = 0
                        xmax = numcols
                        xslice = slice(0, numcols)
            
                    if sy > 1:
                        y0 = self.axes.viewLim.y1/dyintv * numrows
                        iy0 = max(0, int(y0 - self.filterrad))
                        y1 = self.axes.viewLim.y0/dyintv * numrows
                        iy1 = min(numrows, int(y1 + self.filterrad))
                        yslice = slice(numrows-iy1, numrows-iy0)
                        ymin = iy0*dyintv/numrows
                        ymax = iy1*dyintv/numrows
                        dyintv = ymin - ymax
                        sy = abs(dyintv/self.axes.viewLim.height)
                    else:
                        ymin = 0
                        ymax = numrows
                        yslice = slice(0, numrows)
            
                    im = matplotlib.image.fromarray(
                        image[yslice, xslice, :], 0)
                    im.is_grayscale = False
                    im.set_interpolation(matplotlib.image.NEAREST)
                    fc = self.axes.patch.get_facecolor()
                    bg = matplotlib.colors.colorConverter.to_rgba(fc, 0)
                    im.set_bg( *bg)
            
                    # image input dimensions
                    im.reset_matrix()
                    numrows, numcols = im.get_size()
                    if numrows < 1 or numcols < 1:  # out of range
                        return
            
                    # the viewport translation
                    tx = (xmin-self.axes.viewLim.x0)/dxintv * numcols
                    ty = (self.axes.viewLim.y1-ymin)/dyintv * numrows
            
                    l, b, r, t = self.axes.bbox.extents
                    widthDisplay = (round(r) + 0.5) - (round(l) - 0.5)
                    heightDisplay = (round(t) + 0.5) - (round(b) - 0.5)
                    widthDisplay *= magnification
                    heightDisplay *= magnification
                    im.apply_translation(tx, ty)
            
                    # resize viewport to display
                    rx = widthDisplay / numcols
                    ry = heightDisplay  / numrows
                    im.apply_scaling(rx*sx, ry*sy)
                    im.resize(int(widthDisplay+0.5), int(heightDisplay+0.5),
                              norm=1, radius=self.filterrad)
                    bbox = self.axes.bbox.frozen()
                    im._url = self.frame.Title
                    
                    # Two ways to do this, try by version
                    mplib_version = matplotlib.__version__.split(".")
                    if mplib_version[0] == '0':
                        renderer.draw_image(l, b, im, bbox)
                    else:
                        gc = renderer.new_gc()
                        renderer.draw_image(gc, l, b, im)
            subplot.add_artist(CPImageArtist(self.images[(x,y)], self, kwargs))
        
        # Also add this menu to the main menu
        if (x,y) in self.subplot_menus:
            # First trash the existing menu if there is one
            self.menu_subplots.RemoveItem(self.subplot_menus[(x,y)])
        menu_pos = 0
        for yy in range(y + 1):
            if yy == y:
                cols = x
            else:
                cols = self.subplots.shape[0] 
            for xx in range(cols):
                if (xx,yy) in self.images:
                    menu_pos += 1
        self.subplot_menus[(x,y)] = self.menu_subplots.InsertMenu(menu_pos, 
                                        -1, (title or 'Subplot (%s,%s)'%(x,y)), 
                                        self.get_imshow_menu((x,y)))
        
        # Attempt to update histogram plot if one was created
        hist_fig = find_fig(self, name='%s %s image histogram'%(self.Title, 
                                                                (x,y)))
        if hist_fig:
            hist_fig.subplot_histogram(0, 0, self.images[(x,y)].flatten(), 
                                       bins=200, xlabel='pixel intensity')
            hist_fig.figure.canvas.draw()
        return subplot
    
    def subplot_imshow_color(self, x, y, image, title=None, clear=True, 
                             normalize=True, rgb_mask=[1,1,1],
                             sharex=None, sharey=None,
                             use_imshow=False):
        return self.subplot_imshow(x, y, image, title=title, clear=clear, 
                                   normalize=normalize, rgb_mask=rgb_mask, 
                                   sharex=sharex, sharey=sharey,
                                   use_imshow = use_imshow)
    
    def subplot_imshow_labels(self, x, y, labels, title=None, clear=True, 
                              renumber=True, sharex=None, sharey=None,
                              use_imshow = False):
        '''Show a labels matrix using the default color map
        
        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        '''
        if renumber:
            labels = renumber_labels_for_display(labels)
        if np.all(labels == 0):
            image = np.zeros(labels.shape)
        else:
            cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
            cm.set_bad((0,0,0))
            labels = numpy.ma.array(labels, mask=labels==0)
            mappable = matplotlib.cm.ScalarMappable(cmap = cm)
            mappable.set_clim(1, labels.max())
            image = mappable.to_rgba(labels)[:,:,:3]
        return self.subplot_imshow(x, y, image, title, clear, 
                                   normalize=False, vmin=None, vmax=None,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow = use_imshow)
    
    def subplot_imshow_ijv(self, x, y, ijv, shape = None, title=None, 
                           clear=True, renumber=True, sharex=None, sharey=None,
                           use_imshow = False):
        '''Show an ijv-style labeling using the default color map
        
        x,y - the subplot's coordinates
        ijv - a pixel-by-pixel labeling where ijv[:,0] is the i coordinate,
              ijv[:,1] is the j coordinate and ijv[:,2] is the label
        shape - the shape of the final image. If "none", we try to infer
                from the maximum I and J
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        '''
        if shape is None:
            if len(ijv) == 0:
                shape = [1,1]
            else:
                shape = [np.max(ijv[:,0])+1, np.max(ijv[:,1])+1]
        image = np.zeros(list(shape) + [3], np.uint8)
        if len(ijv) > 0:
            cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
            max_label = np.max(ijv[:,2])
            if renumber:
                np.random.seed(0)
                order = np.random.permutation(max_label)
            else:
                order = np.arange(max_label)
            order = np.hstack(([0], order))
            colors = matplotlib.cm.ScalarMappable(cmap = cm).to_rgba(order)
            r,g,b,a = [coo_matrix((colors[ijv[:,2],i],(ijv[:,0],ijv[:,1])),
                                  shape = shape).toarray()
                       for i in range(4)]
            for i, plane in enumerate((r,g,b)):
                image[a != 0,i] = plane[a != 0] * 255. / a[a != 0]
        return self.subplot_imshow(x, y, image, title, clear, 
                                   normalize=False, vmin=None, vmax=None,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow = use_imshow)
        
    def subplot_imshow_grayscale(self, x, y, image, title=None, clear=True,
                                 colorbar=False, normalize=True, vmin=0, vmax=1,
                                 sharex=None, sharey=None, 
                                 use_imshow = False):
        '''Show an intensity image in shades of gray
        
        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        colorbar - show a colorbar relating intensity to color
        normalize - True to normalize to all shades of gray, False to
                    map grays between vmin and vmax
        vmin, vmax - the minimum and maximum intensities
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        '''
        if image.dtype.type == np.float64:
            image = image.astype(np.float32)
        return self.subplot_imshow(x, y, image, title, clear, 
                                   matplotlib.cm.Greys_r, normalize=normalize,
                                   colorbar=colorbar, vmin=vmin, vmax=vmax,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow = use_imshow)
    
    def subplot_imshow_bw(self, x, y, image, title=None, clear=True, 
                          sharex=None, sharey=None, use_imshow = False):
        '''Show a binary image in black and white
        
        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        '''
#        a = 0.3
#        b = 0.59
#        c = 0.11
#        if is_color_image(image):
#            # Convert to luminance
#            image = np.sum(image * (a,b,c), axis=2)
        return self.subplot_imshow(x, y, image, title, clear, 
                                   matplotlib.cm.binary_r,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow = use_imshow)
    
    def normalize_image(self, image, **kwargs):
        '''Produce a color image normalized according to user spec'''
        colormap = kwargs['colormap']
        normalize = kwargs['normalize']
        vmin = kwargs['vmin']
        vmax = kwargs['vmax']
        rgb_mask = kwargs['rgb_mask']
        image = image.astype(np.float32)
        # Perform normalization
        if normalize == True:
            if is_color_image(image):
                image = np.dstack([auto_contrast(image[:,:,ch]) 
                                   for ch in range(image.shape[2])])
            else:
                image = auto_contrast(image)
        elif normalize == 'log':
            if is_color_image(image):
                image = np.dstack([log_transform(image[:,:,ch]) 
                                   for ch in range(image.shape[2])])
            else:
                image = log_transform(image)

        # Apply rgb mask to hide/show channels
        if is_color_image(image):
            rgb_mask = match_rgbmask_to_image(rgb_mask, image)
            image *= rgb_mask
            if image.shape[2] == 2:
                image = np.dstack([image[:,:,0], 
                                   image[:,:,1], 
                                   np.zeros(image.shape[:2], image.dtype)])
        if not is_color_image(image):
            mappable = matplotlib.cm.ScalarMappable(cmap=colormap)
            mappable.set_clim(vmin, vmax)
            image = mappable.to_rgba(image)[:,:,:3]
        return image
    
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
        
    def subplot_scatter(self, x , y,
                        xvals, yvals, 
                        xlabel='', ylabel='',
                        xscale='linear', yscale='linear',
                        title='',
                        clear=True):
        """Put a scatterplot into a subplot
        
        x, y - subplot's column and row
        xvals, yvals - values to scatter
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g. 'log' or 'linear')
        yscale - scaling of the y axis (e.g. 'log' or 'linear')
        title  - string title for the plot
        """
        xvals = np.array(xvals).flatten()
        yvals = np.array(yvals).flatten()
        if clear:
            self.clear_subplot(x, y)

        self.figure.set_facecolor((1,1,1))
        self.figure.set_edgecolor((1,1,1))

        axes = self.subplot(x, y)
        plot = axes.scatter(xvals, yvals,
                            facecolor=(0.0, 0.62, 1.0),
                            edgecolor='none',
                            alpha=0.75)
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        
        return plot
        
    def subplot_histogram(self, x, y, values,
                          bins=20, 
                          xlabel='',
                          xscale=None,
                          yscale='linear',
                          title='',
                          clear=True):
        """Put a histogram into a subplot
        
        x,y - subplot's column and row
        values - values to plot
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
        values = np.array(values).flatten()
        if xscale=='log':
            values = np.log(values[values>0])
            xlabel = 'Log(%s)'%(xlabel or '?')
        # hist apparently doesn't like nans, need to preen them out first
        # (infinities are not much better)
        values = values[np.isfinite(values)]
        # nothing to plot?
        if values.shape[0] == 0:
            axes = self.subplot(x, y)
            plot = axes.text(0.1, 0.5, "No valid values to plot.")
            axes.set_xlabel(xlabel)
            axes.set_title(title)
            return plot
        
        axes = self.subplot(x, y)
        plot = axes.hist(values, bins, 
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
    
    def subplot_platemap(self, x, y, plates_dict, plate_type,
                         cmap=matplotlib.cm.jet, colorbar=True, title='',
                         clear=True):
        '''Draws a basic plate map (as an image).
        x, y       - subplot's column and row (should be 0,0)
        plates_dict - dict of the form: d[plate][well] --> numeric value
                     well must be in the form "A01"
        plate_type - '96' or '384'
        cmap       - a colormap from matplotlib.cm 
                     Warning: gray is currently used for NaN values)
        title      - name for this subplot
        clear      - clear the subplot axes before display if True
        '''
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        
        alphabet = 'ABCDEFGHIJKLMNOP'  #enough letters for a 384 well plate
        plate_names = sorted(plates_dict.keys())
        
        if 'plate_choice' not in self.__dict__:
            platemap_plate = plate_names[0]
            # Add plate selection choice
            sz = wx.BoxSizer(wx.HORIZONTAL)
            sz.AddStretchSpacer()
            plate_static_text = wx.StaticText(self, -1, 'Plate: ')
            self.plate_choice = wx.Choice(self, -1, choices=plate_names)
            self.plate_choice.SetSelection(0)
            sz.Add(plate_static_text, 0, wx.EXPAND)
            sz.Add(self.plate_choice, 0, wx.EXPAND)
            sz.AddStretchSpacer()
            self.Sizer.Insert(0, sz, 0, wx.EXPAND)
            self.Layout()
        else:
            selection = self.plate_choice.GetStringSelection()
            self.plate_choice.SetItems(plate_names)
            if selection in plate_names:
                self.plate_choice.SetStringSelection(selection)
            else:
                self.plate_choice.SetSelection(0)
        def on_plate_selected(evt):
            self.subplot_platemap(x,y, plates_dict, plate_type, cmap=cmap, 
                                  colorbar=colorbar, title=title, clear=True)
        self.plate_choice.Bind(wx.EVT_CHOICE, on_plate_selected)
        
        platemap_plate = self.plate_choice.GetStringSelection()
        data = format_plate_data_as_array(plates_dict[platemap_plate], plate_type)
        
        nrows, ncols = data.shape

        # Draw NaNs as gray
        # XXX: What if colormap with gray in it?
        cmap.set_bad('gray', 1.)
        clean_data = np.ma.array(data, mask=np.isnan(data))
        
        plot = axes.imshow(clean_data, cmap=cmap, interpolation='nearest',
                           shape=data.shape)
        axes.set_title(title)
        axes.set_xticks(range(ncols))
        axes.set_yticks(range(nrows))
        axes.set_xticklabels(range(1, ncols+1), minor=True)
        axes.set_yticklabels(alphabet[:nrows], minor=True)
        axes.axis('image')

        if colorbar:
            subplot = self.subplot(x,y)
            if self.colorbar.has_key(subplot):
                cb = self.colorbar[subplot]
                self.colorbar[subplot] = self.figure.colorbar(plot, cax=cb.ax)
            else:
                self.colorbar[subplot] = self.figure.colorbar(plot)
                
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if (0 <= col < ncols) and (0 <= row < nrows):
                val = data[row, col]
                res = '%s%02d - %1.4f'%(alphabet[row], int(col+1), val)
            else:
                res = '%s%02d'%(alphabet[row], int(col+1))
            # TODO:
##            hint = wx.TipWindow(self, res)
##            wx.FutureCall(500, hint.Close)
            return res
        
        axes.format_coord = format_coord
        
        return plot
        
def format_plate_data_as_array(plate_dict, plate_type):
    ''' Returns an array shaped like the given plate type with the values from
    plate_dict stored in it.  Wells without data will be set to np.NaN
    plate_dict  -  dict mapping well names to data. eg: d["A01"] --> data
                   data values must be of numerical or string types
    plate_type  - '96' (return 8x12 array) or '384' (return 16x24 array)
    '''
    if plate_type == '96':
        plate_shape = (8, 12)
    elif plate_type == '384':
        plate_shape = (16, 24)
    alphabet = 'ABCDEFGHIJKLMNOP'
    data = np.zeros(plate_shape)
    data[:] = np.nan
    display_error = True
    for well, val in plate_dict.items():
        r = alphabet.index(well[0].upper())
        c = int(well[1:]) - 1
        if r >= data.shape[0] or c >= data.shape[1]:
            if display_error:
                logging.getLogger("cellprofiler.gui.cpfigure").warning(
                    'A well value (%s) does not fit in the given plate type.\n'%(well))
                display_error = False
            continue
        data[r,c] = val
    return data
        
def renumber_labels_for_display(labels):
    """Scramble the label numbers randomly to make the display more discernable
    
    The colors of adjacent indices in a color map are less discernable than
    those of far-apart indices. Nearby labels tend to be adjacent or close,
    so a random numbering has more color-distance between labels than a
    straightforward one
    """
    return distance_color_labels(labels)

def only_display_image(figure, shape):
    '''Set up a figure so that the image occupies the entire figure
    
    figure - a matplotlib figure
    shape - i/j size of the image being displayed
    '''
    assert isinstance(figure, matplotlib.figure.Figure)
    figure.set_frameon(False)
    ax = figure.axes[0]
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, 0, 0)
    dpi = figure.dpi
    width = float(shape[1]) / dpi
    height = float(shape[0]) / dpi
    figure.set_figheight(height)
    figure.set_figwidth(width)
    bbox = matplotlib.transforms.Bbox(
        np.array([[0.0, 0.0], [width, height]]))
    transform = matplotlib.transforms.Affine2D(
        np.array([[dpi, 0, 0],
                  [0, dpi, 0],
                  [0,   0, 1]]))
    figure.bbox = matplotlib.transforms.TransformedBbox(bbox, transform)
    
def figure_to_image(figure, *args, **kwargs):
    '''Convert a figure to a numpy array'''
    #
    # Save the figure as a .PNG and then load it using scipy.misc.imread
    #
    fd = StringIO()
    kwargs = kwargs.copy()
    kwargs["format"] = 'png'
    figure.savefig(fd, *args, **kwargs)
    fd.seek(0)
    image = scipy.misc.imread(fd)
    return image[:,:,:3]

if __name__ == "__main__":
    import numpy as np

    app = wx.PySimpleApp()
    
##    f = CPFigureFrame(subplots=(4, 2))
    f = CPFigureFrame(subplots=(1, 1))
    f.Show()
    
    img = np.random.uniform(.4, .6, size=(100, 50, 3))
    img[range(30), range(30), 0] = 1
    
    pdict = {'plate 1': {'A01':1, 'A02':3, 'A03':2},
             'plate 2': {'C01':1, 'C02':3, 'C03':2},
             }
    
##    f.subplot_platemap(0, 0, pdict, '96', title='platemap test')
##    f.subplot_histogram(1, 0, np.random.randn(1000), 50, 'x', title="hist")
##    f.subplot_scatter(2, 0, np.random.randn(1000), np.random.randn(1000), title="scatter")
##    f.subplot_density(3, 0, np.random.randn(100).reshape((50,2)), title="density")
##    f.subplot_imshow(0, 0, img[:,:,0], "1-channel colormapped", sharex=f.subplot(0,0), sharey=f.subplot(0,0), colormap=matplotlib.cm.jet, colorbar=True)
    f.subplot_imshow_grayscale(0, 0, img[:,:,0], "1-channel grayscale", sharex=f.subplot(0,0), sharey=f.subplot(0,0))
##    f.subplot_imshow_bw(2, 0, img[:,:,0], "1-channel bw", sharex=f.subplot(0,0), sharey=f.subplot(0,0))
##    f.subplot_imshow_grayscale(2, 0, img[:,:,0], "1-channel raw", normalize=False, colorbar=True)
##    f.subplot_imshow_grayscale(3, 0, img[:,:,0], "1-channel minmax=(.5,.6)", vmin=.5, vmax=.6, normalize=False, colorbar=True)
##    f.subplot_imshow(0, 1, img, "rgb")
##    f.subplot_imshow(1, 1, img, "rgb raw", normalize=False, sharex=f.subplot(0,1), sharey=f.subplot(0,1))
##    f.subplot_imshow(2, 1, img, "rgb raw disconnected")
##    f.subplot_imshow(2, 1, img, "rgb, log normalized", normalize='log')
##    f.subplot_imshow_bw(3, 1, img[:,:,0], "B&W")

    f.figure.canvas.draw()
    
    app.MainLoop()
