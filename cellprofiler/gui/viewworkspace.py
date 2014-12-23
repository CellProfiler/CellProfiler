"""ViewWorkspace.py - view the image sets and object sets in a workspace

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import numpy as np
import matplotlib
import wx
from wx.lib.intctrl import IntCtrl, EVT_INT
from wx.lib.resizewidget import ResizeWidget
from wx.lib.colourselect import ColourSelect, EVT_COLOURSELECT
from wx.lib.scrolledpanel import ScrolledPanel

from cellprofiler.gui.cpfigure import \
     CPFigureFrame, get_matplotlib_interpolation_preference
from cellprofiler.gui.cpartists import \
     CPImageArtist, ImageData, ObjectsData, MaskData, ColorMixin,\
     MODE_COLORIZE, MODE_HIDE, MODE_LINES,\
     NORMALIZE_LINEAR, NORMALIZE_LOG, NORMALIZE_RAW,\
     INTERPOLATION_BICUBIC, INTERPOLATION_BILINEAR, INTERPOLATION_NEAREST
from cellprofiler.modules.identify import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs

__the_workspace_viewer = None

def show_workspace_viewer(parent, workspace):
    global __the_workspace_viewer
    if __the_workspace_viewer is None:
        __the_workspace_viewer = ViewWorkspace(parent, workspace)
    else:
        __the_workspace_viewer.set_workspace(workspace)
        __the_workspace_viewer.frame.Show()
        
def update_workspace_viewer(workspace):
    if __the_workspace_viewer is not None:
        __the_workspace_viewer.set_workspace(workspace)

def bind_data_class(data_class, color_select, fn_redraw):
    '''Bind ImageData etc to synchronize to color select button

    data_class - ImageData, ObjectData or MaskData
    color_select - a color select button whose color synchronizes
                   to that of the data
    fn_redraw - function to be called
    '''
    assert issubclass(data_class, ColorMixin)
    assert isinstance(color_select, ColourSelect)
    class bdc(data_class):
        def _on_color_changed(self):
            super(bdc, self)._on_color_changed()
            r, g, b = [int(x*255) for x in self.color]
            rold, gold, bold = self.color_select.GetColour()
            if r != rold or g != gold or b != bold:
                self.color_select.SetColour(wx.Colour(r, g, b))
    bdc.color_select = color_select
    return bdc
         
class VWRow(object):
    '''A row of controls and a data item'''
    def __init__(self, vw, color, can_delete):
        self.vw = vw
        panel = vw.panel
        self.chooser = wx.Choice(panel)
        self.color_ctrl = ColourSelect(panel, colour=color)
        self.show_check = wx.CheckBox(panel)
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        self.remove_button = wx.BitmapButton(
            panel, bitmap = bitmap)
        if not can_delete:
            self.remove_button.Hide()
        self.chooser.Bind(wx.EVT_CHOICE, self.on_choice)
        self.color_ctrl.Bind(EVT_COLOURSELECT, self.on_color_change)
        self.show_check.Bind(wx.EVT_CHECKBOX, self.on_check_change)
        self.update_chooser(first=True)
            
    @property
    def color(self):
        '''The color control's current color scaled for matplotlib'''
        return tuple([float(x)/255 for x in self.color_ctrl.GetColour()])

    def on_choice(self, event):
        self.data.name = self.chooser.GetStringSelection()
        self.vw.redraw()
        
    def on_color_change(self, event):
        self.data.color = tuple(
            [float(c) / 255. for c in self.color_ctrl.GetColour()])
        self.vw.redraw()
        
    def on_check_change(self, event):
        self.vw.redraw()
        
    def update(self):
        name = self.chooser.GetStringSelection()
        names = sorted(self.get_names())
        image_set = self.vw.workspace.image_set
        if self.show_check.IsChecked() and name in names:
            self.data.name = name
            self.update_data(name)
            if self.data.mode == MODE_HIDE:
                self.data.mode = self.last_mode
        elif self.data.mode != MODE_HIDE:
            self.last_mode = self.data.mode
            self.data.mode = MODE_HIDE
        self.update_chooser()

    def update_chooser(self, first = False):
        '''Update the chooser with the given list of names'''
        name = self.chooser.GetStringSelection()
        names = self.get_names()
        current_names = sorted(self.chooser.GetItems())
        if tuple(current_names) != tuple(names):
            if name not in names:
                names = sorted(list(names) + [name])
            self.chooser.SetItems(names)
            self.chooser.SetStringSelection(name)
        if first and len(names) > 0:
            name = names[0]
            self.chooser.SetStringSelection(name)
            
        
class VWImageRow(VWRow):
    def __init__(self, vw, color, can_delete):
        super(VWImageRow, self).__init__(vw, color, can_delete)
        image_set = vw.workspace.image_set
        name = self.chooser.GetStringSelection()
        
        im = cpprefs.get_intensity_mode()
        if im == cpprefs.INTENSITY_MODE_LOG:
            normalization = NORMALIZE_LOG
        elif im == cpprefs.INTENSITY_MODE_NORMAL:
            normalization = NORMALIZE_LINEAR
        else:
            normalization = NORMALIZE_RAW
        self.data = bind_data_class(ImageData, self.color_ctrl, vw.redraw)(
            name, None,
            mode = MODE_HIDE,
            color = self.color,
            colormap = cpprefs.get_default_colormap(),
            alpha = .5,
            normalization = normalization)
        vw.image.add(self.data)
        self.last_mode = MODE_COLORIZE
    
    def get_names(self):
        return self.vw.workspace.image_set.get_names()
    
    def update_data(self, name):
        '''Update the image data from the workspace'''
        image_set = self.vw.workspace.image_set
        image = image_set.get_image(name)
        self.data.pixel_data = image.pixel_data
                
class VWObjectsRow(VWRow):
    '''A row of controls for controlling objects'''
    def __init__(self, vw, color, can_delete):
        super(VWObjectsRow, self).__init__(vw, color, can_delete)
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = bind_data_class(ObjectsData, self.color_ctrl, vw.redraw)(
            name, None, 
            outline_color = self.color, 
            colormap = cpprefs.get_default_colormap(), 
            alpha = 1, 
            mode = MODE_HIDE)
        vw.image.add(self.data)
        self.last_mode = MODE_LINES
    
    def get_names(self):    
        object_set = self.vw.workspace.object_set
        return object_set.get_object_names()
    
    def update_data(self, name):
        object_set = self.vw.workspace.object_set
        objects = object_set.get_objects(name)
        self.data.labels = [l for l, i in objects.get_labels()]
        
class ViewWorkspace(object):
    C_CHOOSER = 0
    C_COLOR = 1
    C_SHOW = 2
    C_REMOVE = 3
    def __init__(self, parent, workspace):
        self.frame = CPFigureFrame(
            parent,
            title = "CellProfiler Workspace",
            secret_panel_class=ScrolledPanel)
        self.workspace = workspace
        self.ignore_redraw = False
        self.image_rows = []
        self.object_rows = []
        self.measurement_rows = []
        self.frame.set_subplots((1, 1))
        self.axes = self.frame.subplot(0, 0)
        interpolation = cpprefs.get_interpolation_mode()
        if interpolation == cpprefs.IM_NEAREST:
            interpolation = INTERPOLATION_NEAREST
        elif interpolation == cpprefs.IM_BILINEAR:
            interpolation = INTERPOLATION_BILINEAR
        else:
            interpolation = INTERPOLATION_BICUBIC
        self.image = CPImageArtist(interpolation = interpolation)
        assert isinstance(self.axes, matplotlib.axes.Axes)
        self.axes.add_artist(self.image)
        self.axes.set_aspect('equal')
        self.__axes_scale = None
        
        panel = self.frame.secret_panel
        panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = panel
        #
        # Make a grid of image controls
        #
        panel.Sizer.AddSpacer(4)
        self.image_grid = wx.GridBagSizer(vgap = 3, hgap = 3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(
            sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.image_grid, 0, wx.ALIGN_LEFT)
        self.image_grid.Add(
            wx.StaticText(panel, label="Images"), (0, self.C_CHOOSER), 
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.image_grid.Add(
            wx.StaticText(panel, label="Color"), (0, self.C_COLOR),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.image_grid.Add(
            wx.StaticText(panel, label="Show"), (0, self.C_SHOW),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.image_grid.Add(
            wx.StaticText(panel, label="Remove"), (0, self.C_REMOVE),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.add_image_row(can_delete = False)
        add_image_button = wx.Button(panel,
                                     label = "Add image")
        sub_sizer.Add(add_image_button, 0, wx.ALIGN_RIGHT)
        add_image_button.Bind(
            wx.EVT_BUTTON,
            lambda event:self.add_image_row())
        panel.Sizer.AddSpacer(4)
        panel.Sizer.Add(wx.StaticLine(panel, style = wx.LI_HORIZONTAL), 
                        0, wx.EXPAND)
        panel.Sizer.AddSpacer(4)
        #
        # Make a grid of object controls
        #
        self.object_grid = wx.GridBagSizer(vgap = 3, hgap = 3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.object_grid, 0, wx.ALIGN_LEFT)
        self.object_grid.Add(
            wx.StaticText(panel, label="Objects"), (0, self.C_CHOOSER),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.object_grid.Add(
            wx.StaticText(panel, label="Color"), (0, self.C_COLOR),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.object_grid.Add(
            wx.StaticText(panel, label="Show"), (0, self.C_SHOW),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.object_grid.Add(
            wx.StaticText(panel, label="Remove"), (0, self.C_REMOVE),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.add_objects_row(can_delete = False)
        add_object_button = wx.Button(panel,
                                      label = "Add objects")
        sub_sizer.Add(add_object_button, 0, wx.ALIGN_RIGHT)
        add_object_button.Bind(
            wx.EVT_BUTTON,
            lambda event:self.add_objects_row())
        panel.Sizer.AddSpacer(4)
        panel.Sizer.Add(wx.StaticLine(panel, style = wx.LI_HORIZONTAL),
                        0, wx.EXPAND)
        panel.Sizer.AddSpacer(4)
        #
        # Make a grid of measurements to display
        #
        self.m_grid = wx.GridBagSizer(vgap = 3, hgap = 3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.m_grid, 0, wx.ALIGN_LEFT)
        self.m_grid.Add(
            wx.StaticText(panel, label="Measurement"), (0, self.C_CHOOSER),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.m_grid.Add(
            wx.StaticText(panel, label="Font"), (0, self.C_COLOR),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.m_grid.Add(
            wx.StaticText(panel, label="Show"), (0, self.C_SHOW),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.m_grid.Add(
            wx.StaticText(panel, label="Remove"), (0, self.C_REMOVE),
            flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
        self.add_measurement_row(can_delete = False)
        add_measurement_button = wx.Button(panel,
                                      label = "Add Measurement")
        sub_sizer.Add(add_measurement_button, 0, wx.ALIGN_RIGHT)
        add_measurement_button.Bind(
            wx.EVT_BUTTON,
            self.on_add_measurement_row)
        
        self.image.add_to_menu(self.frame, self.frame.menu_subplots)
        self.frame.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.frame.Bind(wx.EVT_CLOSE, self.on_frame_close)
        self.set_workspace(workspace)
        self.frame.secret_panel.Show()
        w, h = self.frame.GetSize()
        w += self.frame.secret_panel.GetMinWidth()
        self.frame.SetSize(wx.Size(w, h))

    def scale_axes(self):
        '''Set the axes limits appropriate to the images we have'''
        ax = self.image.axes
        if self.__axes_scale == None or \
           (self.__axes_scale[0] == ax.get_ylim()[1] and
            self.__axes_scale[1] == ax.get_xlim()[1]):
            max_x = max_y = 0
            for image_row in self.image_rows:
                if image_row.data.mode != MODE_HIDE:
                    shape = image_row.data.pixel_data.shape
                    max_x = max(shape[1], max_x)
                    max_y = max(shape[0], max_y)
            if max_x > 0 and max_y > 0:
                max_x -= .5
                max_y -= .5
                ax.set_xlim(-.5, max_x)
                ax.set_ylim(-.5, max_y)
                self.__axes_scale = (max_y, max_x)
        
    def layout(self):
        self.panel.SetMinSize((self.panel.GetVirtualSize()[0], 
                               self.panel.GetMinHeight()))
        self.panel.Layout()
        self.frame.secret_panel.Layout()
        self.panel.SetupScrolling()
        
    def on_frame_close(self, event):
        assert isinstance(event, wx.CloseEvent)
        if event.CanVeto():
            self.frame.Hide()
            event.Veto()
            
    def add_image_row(self, can_delete = True):
        self.add_row(
            VWImageRow,
            self.image_rows, self.image_grid,
            can_delete)
        
    def add_row(self, row_class, rows, grid_sizer, can_delete):
        row = len(rows) + 1
        color = wx.RED if row == 1 else wx.GREEN if row == 2 \
            else wx.BLUE if row == 3 else wx.WHITE
        vw_row = row_class(self, color, can_delete)
        
        grid_sizer.Add(vw_row.chooser, (row, self.C_CHOOSER), flag = wx.EXPAND)
        grid_sizer.Add(vw_row.color_ctrl, (row, self.C_COLOR),
                       flag = wx.EXPAND)
        grid_sizer.Add(vw_row.show_check, (row, self.C_SHOW),
                       flag = wx.ALIGN_CENTER)
        grid_sizer.Add(vw_row.remove_button, (row, self.C_REMOVE),
                            flag = wx.ALIGN_CENTER)
        rows.append(vw_row)
        if can_delete:
            self.update_menu(self.frame.menu_subplots)
            self.layout()
        
    def remove_row(self, rows, grid_sizer, remove_button):
        for i, vw_row in enumerate(rows):
            if row.remove_button == remove_button:
                break
        else:
            return
        for control in vw_row.chooser, vw_row.color_ctrl, vw_row.show_check, \
            vw_row.remove_button:
            grid_sizer.Remove(control)
            control.Destroy()
        self.image.remove(vw_row)
        rows.remove(vw_row)
        for ii in range(i, len(rows)):
            vw_row = rows[ii]
            for j, control in enumerate(
                vw_row.chooser, vw_row.color_ctrl, vw_row.show_check,
                vw_row.remove_button):
                grid_sizer.SetItemPosition(control, (ii, j))
        self.update_menu(self.frame.menu_subplots)
        self.layout()
        self.redraw()
    
    def add_objects_row(self, can_delete = True):
        self.add_row(VWObjectsRow, self.object_rows, self.object_grid,
                     can_delete)
    
    def on_add_measurement_row(self, event):
        self.add_measurement_row()
        self.layout()
        self.redraw()
        
    def add_measurement_row(self, can_delete = True):
        row_idx = len(self.measurement_rows)+1
        mr = []
        panel = self.panel
        row = MeasurementRow(panel,
                             self.m_grid,
                             row_idx,
                             lambda : self.on_measurement_changed(mr[0]))
        mr.append(row)
        self.measurement_rows.append(row)
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        
        remove_button = wx.BitmapButton(panel, 
                                        bitmap = bitmap)
        self.m_grid.Add(remove_button, (row_idx, self.C_REMOVE),
                        flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP)
        remove_button.Bind(
            wx.EVT_BUTTON, 
            lambda event: self.remove_measurement_row(row, remove_button))
        if not can_delete:
            remove_button.Hide()
        row.update(self.workspace)
    
    def remove_measurement_row(self, measurement_row, remove_button):
        if measurement_row in self.measurement_rows:
            idx = self.measurement_rows.index(measurement_row)
            measurement_row.destroy(self.m_grid)
            self.m_grid.Remove(remove_button)
            remove_button.Destroy()
            self.measurement_rows.remove(measurement_row)
            for ii in range(idx, len(self.measurement_rows)):
                for j in (self.C_CHOOSER, self.C_COLOR, 
                          self.C_SHOW, self.C_REMOVE):
                    item = self.m_grid.FindItemAtPosition(
                        wx.GBPosition(ii+1, j))
                    self.m_grid.SetItemPosition(item, (ii, j))
            self.layout()
            self.redraw()
                        
                             
    def on_measurement_changed(self, measurement_row):
        assert isinstance(measurement_row, MeasurementRow)
        measurement_row.update(self.workspace)
        self.layout()
        self.redraw()
    
    def set_workspace(self, workspace):
        '''Rebuild the workspace control panel'''
        self.workspace = workspace
        self.ignore_redraw = True
        try:
            self.update_menu(self.frame.menu_subplots)
            self.update_choices(self.image_rows,
                                workspace.image_set.get_names())
            self.update_choices(self.object_rows,
                                workspace.object_set.get_object_names())
            for measurement_row in self.measurement_rows:
                measurement_row.update(workspace)
        finally:
            self.ignore_redraw = False
        self.redraw()
        
    def update_menu(self, menu):
        event = wx.CommandEvent(wx.EVT_MENU_OPEN.evtType[0],
                                self.frame.GetId())
        event.SetEventObject(self.frame)
        self.image.on_update_menu(event, menu)
        
    def update_choices(self, rows, names):
        for row in rows:
            row.update_chooser()
        
    def on_context_menu(self, event):
        menu = wx.Menu()
        try:
            self.image.add_to_menu(self.frame, menu)
            self.update_menu(menu)
            self.frame.PopupMenu(menu)
        finally:
            menu.Destroy()
            
    def redraw(self, event=None):
        if self.ignore_redraw:
            return
        for vw_row in self.image_rows + self.object_rows:
            vw_row.update()
        if not self.frame.figure.canvas.IsShown():
            self.frame.figure.canvas.Show()
            self.layout()
        #
        # Remove all the old text labels
        #
        to_remove = []
        for artist in list(self.axes.texts):
            artist.remove()
        
        m = self.workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        title_lines = []
        object_values = {}
        for measurement_row in self.measurement_rows:
            assert isinstance(measurement_row, MeasurementRow)
            if not measurement_row.should_show():
                continue
            object_name = measurement_row.get_object_name()
            if object_name is None or object_name not in m.get_object_names():
                continue
            feature = measurement_row.get_measurement_name()
            if feature is None or not m.has_feature(object_name, feature):
                continue
            
            value = m[object_name, feature]
            if object_name in (cpmeas.IMAGE, cpmeas.EXPERIMENT):
                if isinstance(value, int):
                    fmt = "%s: %d" 
                elif isinstance(value, float):
                    fmt = "%s: %.4f"
                else:
                    fmt = "%s: %s"
                title_lines.append(fmt % (feature, value))
            else:
                if object_name not in object_values:
                    if any([not m.has_feature(object_name, lf) for lf in
                            M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y]):
                        continue
                    object_values[object_name] = []
                object_values[object_name].append(
                    (value, measurement_row))
        if len(title_lines) > 0:
            self.axes.set_title("\n".join(title_lines))
        else:
            self.axes.set_title("Image set # %d" % m.image_number)
        for object_name, value_rows in object_values.items():
            values = [vr[0] for vr in value_rows]
            measurement_rows = [vr[1] for vr in value_rows]
            x, y = [m[object_name, ftr] for ftr in 
                    M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y]
            for i in range(len(x)):
                xi, yi = x[i], y[i]
                if np.isnan(xi) or np.isnan(yi):
                    continue
                height = 0
                for j, measurement_row in enumerate(measurement_rows):
                    if len(values[j]) <= i or np.isnan(values[j][i]):
                        continue
                    value = values[j][i]
                    font = measurement_row.font
                    if font.GetStyle() == wx.ITALIC:
                        fontstyle="italic"
                    else:
                        fontstyle="normal"
                    color = measurement_row.foreground_color
                    fontcolor, backgroundcolor = [
                        tuple([float(c)/255 for c in color][:3]) for color in
                        measurement_row.foreground_color,
                        measurement_row.background_color]
                    
                    fmt = "%%.%df" % measurement_row.precision
                    a = self.axes.annotate(
                        fmt % value,
                        (xi, yi), 
                        xytext = (0, -height),
                        textcoords = "offset points",
                        ha = "center",
                        va = "center",
                        bbox = {
                            "boxstyle": measurement_row.box_style,
                            "fc": backgroundcolor,
                            "alpha": measurement_row.background_alpha
                            },
                        color = fontcolor,
                        family = font.GetFaceName(),
                        fontsize = font.GetPointSize(),
                        fontstyle = fontstyle,
                        weight = font.GetWeight())
                    height += font.GetPointSize() + 1
        
        self.scale_axes()
        self.frame.figure.canvas.draw()
            
class MeasurementRow(object):
    '''Container for measurement controls'''
    def __init__(self, panel, grid_sizer, row_idx, on_change):
        '''MeasurementRow contstructor
        
        panel - the panel that's going to be the host for the controls
        grid_sizer - put the controls in this grid sizer
        row_idx - the row # in the grid sizer
        on_change - a function (with no args) that's called whenever any control
                    is changed. This handler should call MeasurementRow.update
        '''
        #
        # Create three-tiered measurement choice:
        #    object name
        #    category
        #    measurement name
        #
        assert isinstance(grid_sizer, wx.GridBagSizer)
        self.process_events = True
        self.change_fn = on_change
        self.choice_panel = wx.Panel(panel)
        grid_sizer.Add(self.choice_panel, (row_idx, ViewWorkspace.C_CHOOSER),
                       flag=wx.EXPAND)
        self.choice_sizer = wx.BoxSizer(wx.VERTICAL)
        self.choice_panel.SetSizer(self.choice_sizer)
        self.object_choice = wx.Choice(self.choice_panel)
        self.choice_sizer.Add(self.object_choice, 0, wx.EXPAND)
        self.category_choice = wx.Choice(self.choice_panel)
        self.category_choice.Hide()
        self.choice_sizer.Add(self.category_choice, 0, wx.EXPAND | wx.TOP, 2)
        self.measurement_choice = wx.Choice(self.choice_panel)
        self.measurement_choice.Hide()
        self.choice_sizer.Add(self.measurement_choice, 0, wx.EXPAND | wx.TOP, 2)
        #
        # Font button
        #
        self.font_button = wx.Button(panel, label = "Font")
        grid_sizer.Add(self.font_button, (row_idx, ViewWorkspace.C_COLOR),
                       flag = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP)
        self.font_button.Bind(wx.EVT_BUTTON, self.on_choose_font)
        self.show_ctrl = wx.CheckBox(panel)
        grid_sizer.Add(self.show_ctrl, (row_idx, ViewWorkspace.C_SHOW), 
                       flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP)
        #
        # The drawing characteristics
        #
        self.font = self.font_button.Font
        self.foreground_color = wx.SystemSettings.GetColour(
            wx.SYS_COLOUR_BTNTEXT)
        self.background_color = wx.SystemSettings.GetColour(
            wx.SYS_COLOUR_WINDOW)
        self.background_alpha = 0.5
        self.box_style = "round"
        self.precision = 4 # of decimal places
        
        for control, event in (
            (self.object_choice, wx.EVT_CHOICE),
            (self.category_choice, wx.EVT_CHOICE),
            (self.measurement_choice, wx.EVT_CHOICE),
            (self.show_ctrl, wx.EVT_CHECKBOX)):
            control.Bind(event, self.on_change)
        
    def on_choose_font(self, event):
        with wx.Dialog(self.choice_panel.Parent,
                       title = "Measurement appearance") as dlg:
            labels = []
            def add_label(sizer, label):
                ctrl = wx.StaticText(dlg, label=label)
                sizer.Add(ctrl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                labels.append(ctrl)
                sizer.AddSpacer(2)
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 5)
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Font")
            font_picker = wx.FontPickerCtrl(dlg)
            font_picker.SetSelectedFont(self.font)
            font_picker.SetPickerCtrlGrowable(True)
            font_picker.Bind(wx.EVT_FONTPICKER_CHANGED,
                             lambda event: dlg.Layout())
            subsizer.Add(font_picker, 0, wx.ALIGN_LEFT)
            #
            # Foreground color
            #
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Text color")
            foreground_color = ColourSelect(
                dlg, colour = self.foreground_color)
            subsizer.Add(foreground_color, 0, wx.ALIGN_LEFT)
            #
            # Background color and alpha
            #
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Background color")
            background_color = ColourSelect(
                dlg, colour = self.background_color)
            subsizer.Add(background_color, 0, wx.ALIGN_LEFT)
            
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Alpha")
            alpha = wx.Slider(
                dlg, value=self.background_alpha * 100,
                minValue = 0, maxValue = 100,
                style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
            alpha.SetMinSize(wx.Size(200, alpha.GetMinSize()[0]))
            subsizer.Add(alpha, 0, wx.EXPAND)

            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Box shape")
            box_style = wx.Choice(
                dlg, choices = ["circle", "round", "roundtooth", "sawtooth",
                                "square"])
            box_style.SetStringSelection(self.box_style)
            subsizer.Add(box_style, 0, wx.ALIGN_LEFT)
            
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Precision")
            precision = wx.SpinCtrl(dlg, value=str(self.precision), min = 0)
            subsizer.AddSpacer(2)
            subsizer.Add(precision, 0, wx.ALIGN_LEFT)
            
            width = 0
            for label in labels:
                width = max(width, label.GetBestSize()[0])
            for label in labels:
                label.SetMinSize(wx.Size(width, label.GetBestSize()[1]))
            
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL |wx.ALL, 5)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()
            dlg.Fit()
            if dlg.ShowModal() == wx.ID_OK:
                self.font = font_picker.GetSelectedFont()
                self.foreground_color = foreground_color.GetColour()
                self.background_color = background_color.GetColour()
                self.background_alpha = float(alpha.Value) / 100
                self.box_style = box_style.GetStringSelection()
                self.precision = precision.Value
                self.on_change(event)
        
    def on_change(self, event):
        if self.process_events:
            self.change_fn()
    
    @staticmethod
    def __get_selected_choice(control):
        assert isinstance(control, wx.Choice)
        if not control.IsShown():
            return None
        selection = control.GetSelection()
        if selection == wx.NOT_FOUND:
            return None
        return control.GetItems()[selection]
    
    def get_object_name(self):
        return self.__get_selected_choice(self.object_choice)
    
    def get_measurement_name(self):
        if self.get_object_name() is None:
            return None
        category = self.__get_selected_choice(self.category_choice)
        if category is None:
            return None
        feature = self.__get_selected_choice(self.measurement_choice)
        if feature is None:
            return None
        return "_".join((category, feature))
    
    def should_show(self):
        return self.show_ctrl.IsChecked()
    
    def update_choices(self, control, choices):
        assert isinstance(control, wx.Choice)
        old_names = control.GetItems()
        if tuple(sorted(old_names)) == tuple(sorted(choices)):
            return
        self.process_events = False
        try:
            old_choice = self.__get_selected_choice(control)
            if old_choice is not None and old_choice not in choices:
                choices = choices + [old_choice]
            control.SetItems(choices)
            if old_choice is not None:
                control.SetStringSelection(old_choice)
        finally:
            self.process_events = True
        
    def update(self, workspace):
        m = workspace.measurements
        self.update_choices(self.object_choice, m.get_object_names())
        object_name = self.get_object_name()
        if object_name is None or object_name not in m.get_object_names():
            self.category_choice.Hide()
            self.measurement_choice.Hide()
            return
        self.category_choice.Show()
        self.measurement_choice.Show()
        categories = set()
        measurements = set()
        current_category = self.__get_selected_choice(self.category_choice)
        for feature in m.get_feature_names(object_name):
            category, measurement = feature.split("_", 1)
            categories.add(category)
            if category == current_category:
                measurements.add(measurement)
        self.update_choices(self.category_choice, sorted(categories))
        self.update_choices(self.measurement_choice, sorted(measurements))
        
    def destroy(self, grid_sizer):
        grid_sizer.Remove(self.choice_panel)
        grid_sizer.Remove(self.font_button)
        grid_sizer.Remove(self.show_ctrl)
        self.object_choice.Destroy()
        self.category_choice.Destroy()
        self.measurement_choice.Destroy()
        self.choice_panel.Destroy()
        self.font_button.Destroy()
        self.show_ctrl.Destroy()
            
        
    