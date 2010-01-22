'''datatoolframe.py - Holder for a data tool

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

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
from cellprofiler.gui import get_icon

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

        self.module_view = ModuleView(module_panel, self.pipeline)
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
        self.SetSizer(self.sizer)
        self.Size = (self.module_view.get_max_width(), self.Size[1])
        module_panel.Layout()
        self.Show()
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_icon(), "CellProfiler2.0")
        self.SetIcon(get_icon())
    
    def load_measurements(self, measurements_file_name):
        handles = loadmat(measurements_file_name, struct_as_record=True)
        m = handles["handles"][0,0][cpp.MEASUREMENTS][0,0]
        self.measurements = cpmeas.Measurements()
        for object_name in m.dtype.fields.keys():
            omeas = m[object_name][0,0]
            for feature_name in omeas.dtype.fields.keys():
                if object_name == cpmeas.IMAGE:
                    values = [x[0] for x in omeas[feature_name][0]]
                else:
                    values = omeas[feature_name][0].tolist()
                self.measurements.add_all_measurements(object_name,
                                                       feature_name,
                                                       values)
        
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
