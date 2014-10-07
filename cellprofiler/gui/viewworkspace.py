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
from wx.lib.colourselect import ColourSelect, EVT_COLOURSELECT

from cellprofiler.gui.cpfigure import \
     CPFigureFrame, CPImageArtist, get_matplotlib_interpolation_preference, \
     CPLD_LABELS, CPLD_NAME, CPLD_OUTLINE_COLOR, CPLDM_OUTLINES, \
     CPLD_MODE, CPLD_LINE_WIDTH, CPLD_ALPHA_COLORMAP, CPLD_ALPHA_VALUE
import cellprofiler.preferences as cpprefs

__the_workspace_viewer = None

def show_workspace_viewer(parent, workspace):
    global __the_workspace_viewer
    if __the_workspace_viewer is None:
        __the_workspace_viewer = ViewWorkspace(parent, workspace)
    else:
        __the_workspace_viewer.set_workspace(workspace)
        __the_workspace_viewer.frame.show()
        
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
            title = "CellProfiler Workspace")
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
        panel.Sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND)
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
        panel.Sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND)
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
            wx.StaticText(panel, label="Color"), (0, self.C_COLOR),
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
            lambda event:self.add_measurement_row())
        
        self.frame.Bind(wx.EVT_CLOSE, self.on_frame_close)
        self.set_workspace(workspace)
        panel.Show()
        self.frame.Layout()
        for child in panel.GetChildren():
            child.Refresh()
        panel.Refresh()

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
        panel = self.frame.secret_panel
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
        show_check.SetValue(True)
        grid_sizer.Add(show_check, (row, self.C_SHOW),
                       flag = wx.ALIGN_CENTER)
        show_check.Bind(wx.EVT_CHECKBOX, self.redraw)
        controls.append(show_check)
        
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        
        remove_button = wx.BitmapButton(panel, 
                                        bitmap = bitmap)
        self.image_grid.Add(remove_button, (row, self.C_REMOVE),
                            flag = wx.ALIGN_CENTER)
        remove_button.Bind(
            wx.EVT_BUTTON, 
            lambda event: self.remove_row(rows, grid_sizer, remove_button))
        if not can_delete:
            remove_button.Hide()
        controls.append(remove_button)
        rows.append(controls)
        self.frame.Layout()
        
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
        self.frame.Layout()
        self.redraw()
    
    def add_objects_row(self, can_delete = True):
        self.add_row(self.object_rows, self.object_grid,
                     self.workspace.object_set.get_object_names(), 
                     can_delete)
    
    def add_measurement_row(self, can_delete = True):
        pass
    
    def set_workspace(self, workspace):
        '''Rebuild the workspace control panel'''
        self.workspace = workspace
        self.ignore_redraw = True
        try:
            self.update_choices(self.image_rows,
                                workspace.image_set.get_names())
            self.update_choices(self.object_rows,
                                workspace.object_set.get_object_names())
        finally:
            self.ignore_redraw = False
        self.redraw()
        
    def update_choices(self, rows, names):
        for row in rows:
            choice = row[self.C_CHOOSER]
            assert isinstance(choice, wx.Choice)
            current_selection = choice.GetCurrentSelection()
            current_names = choice.GetItems()
            if tuple(sorted(names)) != tuple(sorted(current_names)):
                choice.SetItems(names)
                if current_selection >=0 and current_selection < len(current_names):
                    choice.SetStringSelection(current_names[current_selection])
        
    def redraw(self, event=None):
        if self.ignore_redraw:
            return
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
        if len(images) == 0:
            self.frame.figure.canvas.Hide()
            return
        
        if not self.frame.figure.canvas.IsShown():
            self.frame.figure.canvas.Show()
            self.frame.Layout()
        width = height = 0
        for image, _, _, _ in images:
            width = max(width, image.pixel_data.shape[1])
            height = max(height, image.pixel_data.shape[0])
        image = np.zeros((height, width, 3))
        for src_image, red, green, blue in images:
            pixel_data = src_image.pixel_data
            src_height, src_width = pixel_data.shape[:2]
            if pixel_data.ndim == 3:
                src_depth = min(pixel_data.shape[2], 3)
                image[:src_height, :src_width, :src_depth] += \
                    pixel_data[:, :, :src_depth]
            else:
                image[:src_height, :src_width, 0] += pixel_data * red / 255
                image[:src_height, :src_width, 1] += pixel_data * green / 255
                image[:src_height, :src_width, 2] += pixel_data * blue / 255
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
            
            cplabels.append( {
                CPLD_NAME: objects_name,
                CPLD_LABELS: [x[0] for x in objects.get_labels()],
                CPLD_OUTLINE_COLOR: color,
                CPLD_MODE: CPLDM_OUTLINES,
                CPLD_ALPHA_VALUE: .25,
                CPLD_ALPHA_COLORMAP: alpha_colormap,
                CPLD_LINE_WIDTH: 1})
                
        if self.image is not None and \
           tuple(self.image.image.shape) != tuple(image.shape):
            self.image = None
            self.axes.cla()
        if self.image == None:
            self.frame.subplot_imshow_color(
                0, 0, image,
                clear = False,
                cplabels = cplabels)
            self.axes.set_xbound(-.5, width -.5)
            self.axes.set_ybound(-.5, height -.5)
            for artist in self.axes.artists:
                if isinstance(artist, CPImageArtist):
                    self.image = artist
        else:
            self.image.image = image
            self.image.kwargs["cplabels"] = cplabels
        self.frame.figure.canvas.draw()
            
        
