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
     CPFigureFrame, CPImageArtist, get_matplotlib_interpolation_preference, \
     CPLD_LABELS, CPLD_NAME, CPLD_OUTLINE_COLOR, CPLDM_OUTLINES, \
     CPLD_MODE, CPLD_LINE_WIDTH, CPLD_ALPHA_COLORMAP, CPLD_ALPHA_VALUE, CPLD_SHOW
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
        self.image = None
        
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
        
        self.frame.Bind(wx.EVT_CLOSE, self.on_frame_close)
        self.set_workspace(workspace)
        self.frame.secret_panel.Show()
        self.frame.on_size(None)
        
    def layout(self):
        self.frame.secret_panel.SetupScrolling()
        
    def on_frame_close(self, event):
        assert isinstance(event, wx.CloseEvent)
        if event.CanVeto():
            self.frame.Hide()
            event.Veto()
            
    def add_image_row(self, can_delete = True):
        self.add_row(
            self.image_rows, self.image_grid,
            self.workspace.image_set.get_names(),
            can_delete)
        
    def add_row(self, rows, grid_sizer, names, can_delete):
        row = len(rows) + 1
        controls = []
        panel = self.panel
        chooser = wx.Choice(panel, choices = names)
        grid_sizer.Add(chooser, (row, self.C_CHOOSER), flag = wx.EXPAND)
        controls.append(chooser)
        chooser.Bind(wx.EVT_CHOICE, self.redraw)
        
        color = ColourSelect(
            panel,
            colour = wx.RED if row == 1 else wx.GREEN if row == 2 \
            else wx.BLUE if row == 3 else wx.WHITE)
        color.Bind(EVT_COLOURSELECT, self.redraw)
        controls.append(color)
        grid_sizer.Add(color, (row, self.C_COLOR),
                       flag = wx.EXPAND)
        show_check = wx.CheckBox(panel)
        show_check.SetValue(False)
        grid_sizer.Add(show_check, (row, self.C_SHOW),
                       flag = wx.ALIGN_CENTER)
        show_check.Bind(wx.EVT_CHECKBOX, self.redraw)
        controls.append(show_check)
        
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        
        remove_button = wx.BitmapButton(panel, 
                                        bitmap = bitmap)
        grid_sizer.Add(remove_button, (row, self.C_REMOVE),
                            flag = wx.ALIGN_CENTER)
        remove_button.Bind(
            wx.EVT_BUTTON, 
            lambda event: self.remove_row(rows, grid_sizer, remove_button))
        if not can_delete:
            remove_button.Hide()
        controls.append(remove_button)
        rows.append(controls)
        self.layout()
        
    def remove_row(self, rows, grid_sizer, remove_button):
        for i, row in enumerate(rows):
            if row[self.C_REMOVE] == remove_button:
                break
        else:
            return
        for j, control in enumerate(rows[i]):
            grid_sizer.Remove(control)
            control.Destroy()
        rows.remove(row)
        for ii in range(i, len(rows)):
            for j, control in enumerate(rows[ii]):
                grid_sizer.SetItemPosition(control, (ii, j))
        self.layout()
        self.redraw()
    
    def add_objects_row(self, can_delete = True):
        self.add_row(self.object_rows, self.object_grid,
                     self.workspace.object_set.get_object_names(), 
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
            self.update_choices(self.image_rows,
                                workspace.image_set.get_names())
            self.update_choices(self.object_rows,
                                workspace.object_set.get_object_names())
            for measurement_row in self.measurement_rows:
                measurement_row.update(workspace)
        finally:
            self.ignore_redraw = False
        self.redraw()
        
    def update_choices(self, rows, names):
        for row in rows:
            choice = row[self.C_CHOOSER]
            assert isinstance(choice, wx.Choice)
            current_selection = choice.GetCurrentSelection()
            current_names = choice.GetItems()
            if current_selection != wx.NOT_FOUND:
                if current_names[current_selection] not in names:
                    names.append(current_names[current_selection])
            if tuple(sorted(names)) != tuple(sorted(current_names)):
                choice.SetItems(names)
                if current_selection >=0 and current_selection < len(current_names):
                    choice.SetStringSelection(current_names[current_selection])
        
    def redraw(self, event=None):
        if self.ignore_redraw:
            return
        min_height = min_width = np.iinfo(np.int32).max
        smallest = None
        size_mismatch = False
        images = []
        for chooser, color, check, _ in self.image_rows:
            if not check.IsChecked():
                continue
            assert isinstance(chooser, wx.Choice)
            selection = chooser.GetCurrentSelection()
            items = chooser.GetItems()
            if selection < 0 or selection >= len(items):
                continue
            image_name = items[selection]
            if not image_name in self.workspace.image_set.get_names():
                continue
            image = self.workspace.image_set.get_image(image_name)
            red, green, blue = color.GetValue()
            images.append((image, red, green, blue))
            height, width = image.pixel_data.shape[:2]
            if height < min_height or width < min_width:
                min_height, min_width = height, width
                smallest = image
                size_mismatch = True
            elif height > min_height or width > min_width:
                size_mismatch = True
            
        if len(images) == 0:
            self.frame.figure.canvas.Hide()
            return

        cplabels = []
        for chooser, color, check, _ in self.object_rows:
            if not check.IsChecked():
                continue
            idx = chooser.GetCurrentSelection()
            names = chooser.GetItems()
            if idx < 0 or idx >= len(names):
                continue
            objects_name = names[idx]
            if objects_name not in self.workspace.object_set.get_object_names():
                continue
            objects = self.workspace.object_set.get_objects(objects_name)
            red, green, blue = color.GetValue()
            color = (red, green, blue)
            alpha_colormap = cpprefs.get_default_colormap()
            height, width = objects.shape[:2]
            if height < min_height or width < min_width:
                min_height, min_width = height, width
                smallest = objects
                size_mismatch = True
            elif height > min_height or width > min_width:
                size_mismatch = True
            
            cplabels.append( {
                CPLD_NAME: objects_name,
                CPLD_LABELS: [x[0] for x in objects.get_labels()],
                CPLD_OUTLINE_COLOR: color,
                CPLD_MODE: CPLDM_OUTLINES,
                CPLD_ALPHA_VALUE: .25,
                CPLD_ALPHA_COLORMAP: alpha_colormap,
                CPLD_LINE_WIDTH: 1,
                CPLD_SHOW: True})
            
        if size_mismatch:
            for d in cplabels:
                d[CPLD_LABELS] = [
                    smallest.crop_image_similarly(l) for l in d[CPLD_LABELS]]
        
        if not self.frame.figure.canvas.IsShown():
            self.frame.figure.canvas.Show()
            self.layout()
        width, height = min_width, min_height
        image = np.zeros((height, width, 3))
        for src_image, red, green, blue in images:
            pixel_data = src_image.pixel_data.astype(np.float32)
            if size_mismatch:
                pixel_data = smallest.crop_image_similarly(pixel_data)
            if pixel_data.ndim == 3:
                src_depth = min(pixel_data.shape[2], 3)
                image[:, :, :src_depth] += \
                    pixel_data[:, :, :src_depth]
            else:
                image[:, :, 0] += pixel_data * red / 255
                image[:, :, 1] += pixel_data * green / 255
                image[:, :, 2] += pixel_data * blue / 255
                
        if self.image is not None and \
           tuple(self.image.image.shape) != tuple(image.shape):
            self.image = None
            self.axes.cla()
        if self.image == None:
            self.frame.subplot_imshow_color(
                0, 0, image,
                clear = False,
                cplabels = cplabels,
                normalize=None)
            self.axes.set_xbound(-.5, width -.5)
            self.axes.set_ybound(-.5, height -.5)
            for artist in self.axes.artists:
                if isinstance(artist, CPImageArtist):
                    self.image = artist
        else:
            self.image.image = image
            old_cplabels = self.image.kwargs["cplabels"]
            for cplabel in old_cplabels:
                cplabel[CPLD_SHOW] = False
            for cplabel in cplabels:
                name = cplabel[CPLD_NAME]
                matches = filter((lambda x: x[CPLD_NAME] == name), old_cplabels)
                if len(matches) == 0:
                    old_cplabels.append(cplabel)
                else:
                    matches[0][CPLD_LABELS] = cplabel[CPLD_LABELS]
                    matches[0][CPLD_OUTLINE_COLOR] = cplabel[CPLD_OUTLINE_COLOR]
                    matches[0][CPLD_SHOW] = True
            self.image.kwargs["cplabels"] = old_cplabels
            self.frame.subplot_params[(0, 0)]['cplabels'] = old_cplabels
            self.frame.update_line_labels(self.axes, self.image.kwargs)
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
            
        
    