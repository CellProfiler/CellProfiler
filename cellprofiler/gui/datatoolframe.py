'''datatoolframe.py - Holder for a data tool

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
from scipy.io.matlab.mio import loadmat
import wx
import wx.lib.scrolledpanel

import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
from cellprofiler.gui.moduleview import ModuleView
from cellprofiler.modules import instantiate_module
from cellprofiler.gui import get_cp_icon

ID_FILE_LOAD_MEASUREMENTS = wx.NewId()
ID_FILE_SAVE_MEASUREMENTS = wx.NewId()
ID_FILE_EXIT = wx.NewId()

ID_IMAGE_CHOOSE = wx.NewId()

class DataToolFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        '''Instantiate a data tool frame
        
        module_name: name of module to instantiate
        measurements_file_name: name of measurements file
        '''
        assert kwds.has_key("module_name"), "DataToolFrame() needs a module_name argument"
        assert kwds.has_key("measurements_file_name"), "DataToolFrame() needs a measurements_file_name argument"
        module_name = kwds["module_name"]
        measurements_file_name = kwds["measurements_file_name"]

        kwds_copy = kwds.copy()
        del kwds_copy["module_name"]
        del kwds_copy["measurements_file_name"]
        kwds_copy["title"]="%s data tool"%module_name
        wx.Frame.__init__(self, *args, **kwds_copy)
        self.module = instantiate_module(module_name)
        self.pipeline = cpp.Pipeline()
        self.pipeline.load(measurements_file_name)
        self.load_measurements(measurements_file_name)
        self.module.module_num = len(self.pipeline.modules())+1
        self.pipeline.add_module(self.module)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        module_panel = wx.lib.scrolledpanel.ScrolledPanel(self,-1,style=wx.SUNKEN_BORDER)
        module_panel.BackgroundColour = cpprefs.get_background_color()
        self.BackgroundColour = cpprefs.get_background_color()

        self.module_view = ModuleView(module_panel, self.pipeline, True)
        self.module_view.set_selection(self.module.module_num)
        def on_change(caller, event):
            setting = event.get_setting()
            proposed_value = event.get_proposed_value()
            setting.value = proposed_value
            self.pipeline.edit_module(event.get_module().module_num)
            self.module_view.reset_view()
        self.module_view.add_listener(on_change)

        #
        # Add a panel for the "run" button
        #
        panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel, label = "Run")

        self.sizer.Add(module_panel, 1, wx.EXPAND)
        self.sizer.Add(panel, 0, wx.EXPAND)

        panel_sizer.AddStretchSpacer()
        panel_sizer.Add(button, 0, wx.RIGHT, button.Size[1])
        panel.SetSizer(panel_sizer)

        wx.EVT_BUTTON(self, button.Id, self.on_run)
        #
        # Add a file menu
        #
        file_menu = wx.Menu()
        file_menu.Append(ID_FILE_LOAD_MEASUREMENTS, "&Load measurements")
        file_menu.Append(ID_FILE_SAVE_MEASUREMENTS, "&Save measurements")
        file_menu.Append(ID_FILE_EXIT, "E&xit")
        self.MenuBar = wx.MenuBar()
        self.MenuBar.Append(file_menu, "&File")
        self.Bind(wx.EVT_MENU, self.on_load_measurements, id=ID_FILE_LOAD_MEASUREMENTS)
        self.Bind(wx.EVT_MENU, self.on_save_measurements, id=ID_FILE_SAVE_MEASUREMENTS)
        self.Bind(wx.EVT_MENU, self.on_exit, id=ID_FILE_EXIT)
        accelerators = wx.AcceleratorTable([
            (wx.ACCEL_CMD, ord("W"), ID_FILE_EXIT),
            (wx.ACCEL_CMD, ord("O"), ID_FILE_LOAD_MEASUREMENTS),
            (wx.ACCEL_CMD, ord("S"), ID_FILE_SAVE_MEASUREMENTS)])
        self.SetAcceleratorTable(accelerators)
        #
        # Add an image menu
        #
        image_menu = wx.Menu()
        image_menu.Append(ID_IMAGE_CHOOSE, "&Choose")
        self.MenuBar.Append(image_menu, "&Image")
        self.Bind(wx.EVT_MENU, self.on_image_choose, id=ID_IMAGE_CHOOSE)
        
        self.SetSizer(self.sizer)
        self.Size = (self.module_view.get_max_width(), self.Size[1])
        module_panel.Layout()
        self.Show()
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_cp_icon(), "CellProfiler2.0")
        self.SetIcon(get_cp_icon())
    
    def on_load_measurements(self, event):
        dlg = wx.FileDialog(self, "Load a measurements file",
                            wildcard = "Measurements file (*.mat)|*.mat",
                            style = wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.load_measurements(dlg.GetPath())
    
    def on_save_measurements(self, event):
        dlg = wx.FileDialog(self, "Save measurements file",
                            wildcard = "Measurements file (*.mat)|*.mat",
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self.pipeline.save_measurements(dlg.GetPath(),
                                            self.measurements)
    
    def on_exit(self, event):
        self.Close()
        
    def on_image_choose(self, event):
        '''Choose an image from the image set'''
        dlg = wx.Dialog(self)
        dlg.Title = "Choose an image set"
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        choose_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(choose_sizer, 1, wx.EXPAND | wx.ALL, 5)
        choose_sizer.Add(wx.StaticText(dlg, -1, label="Image set:"),
                         0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        metadata_features = [ 
            x for x in self.measurements.get_feature_names(cpmeas.IMAGE)
            if x.startswith('Metadata')]
        metadata_values = [ 
            self.measurements.get_all_measurements(cpmeas.IMAGE, feature)
            for feature in metadata_features]
        metadata_kv = [ ','.join(['%s=%s' % (feature, value[i])
                                  for feature, value in zip(metadata_features,
                                                            metadata_values)])
                        for i in range(self.measurements.image_set_count)]
        choices = ["%d: %s" % (i+1, metadata_kv[i])
                   for i in range(self.measurements.image_set_count)]
        choice_ctl = wx.Choice(dlg, -1, choices=choices)
        # Select the current image set
        choice_ctl.SetSelection(self.measurements.image_set_index)
        choose_sizer.Add(choice_ctl, 1, wx.EXPAND | wx.LEFT, 5)
        button_sizer = wx.StdDialogButtonSizer()
        ok_button = wx.Button(dlg, wx.ID_OK)
        ok_button.SetDefault()
        ok_button.SetHelpText("Press the OK button to change the current image to the one selected above")
        button_sizer.AddButton(ok_button)
        
        cancel_button = wx.Button(dlg, wx.ID_CANCEL)
        cancel_button.SetHelpText("Press the cancel button if you do not want to change the current image")
        button_sizer.AddButton(cancel_button)
        button_sizer.Realize()
        sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT|wx.ALL, 5)
        dlg.Fit()
        if dlg.ShowModal() == wx.ID_OK:
            index = choice_ctl.GetSelection()
            self.measurements.image_set_number = index+1
        
    def load_measurements(self, measurements_file_name):
        self.measurements = cpmeas.Measurements(can_overwrite = True)
        self.measurements.load(measurements_file_name)
        # Start on the first image
        self.measurements.set_image_set_number(1)
        
    def on_run(self, event):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(self.pipeline,
                                  self.module,
                                  image_set,
                                  cpo.ObjectSet(),
                                  self.measurements,
                                  image_set_list,
                                  frame=self)
        self.module.run_as_data_tool(workspace)
        workspace.refresh()
