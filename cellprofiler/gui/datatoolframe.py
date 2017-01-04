# coding=utf-8
"""datatoolframe.py - Holder for a data tool
"""

import cellprofiler.image
import cellprofiler.gui
import cellprofiler.gui.figure
import cellprofiler.gui.moduleview
import cellprofiler.gui.pipeline
import cellprofiler.measurement
import cellprofiler.modules
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.gui.workspace
import h5py
import wx
import wx.lib.scrolledpanel

ID_FILE_LOAD_MEASUREMENTS = wx.NewId()
ID_FILE_SAVE_MEASUREMENTS = wx.NewId()
ID_FILE_EXIT = wx.NewId()

ID_IMAGE_CHOOSE = wx.NewId()


class DataToolFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Instantiate a data tool frame

        module_name: name of module to instantiate
        measurements_file_name: name of measurements file
        """
        assert "module_name" in kwds, "DataToolFrame() needs a module_name argument"
        assert "measurements_file_name" in kwds, "DataToolFrame() needs a measurements_file_name argument"
        module_name = kwds["module_name"]
        measurements_file_name = kwds["measurements_file_name"]

        kwds_copy = kwds.copy()
        del kwds_copy["module_name"]
        del kwds_copy["measurements_file_name"]
        kwds_copy["title"] = "%s data tool" % module_name
        wx.Frame.__init__(self, *args, **kwds_copy)
        self.module = cellprofiler.modules.instantiate_module(module_name)
        self.module.use_as_data_tool = True
        self.pipeline = cellprofiler.gui.pipeline.Pipeline()
        if h5py.is_hdf5(measurements_file_name):
            self.workspace = cellprofiler.gui.workspace.Workspace(self.pipeline, self.module, None, None, None,
                                                              None)
            self.workspace.load(measurements_file_name, True)
            self.measurements = self.workspace.measurements
        else:
            self.pipeline.load(measurements_file_name)
            self.load_measurements(measurements_file_name)
            self.workspace = cellprofiler.gui.workspace.Workspace(self.pipeline, self.module, None, None,
                                                              self.measurements, None)

        self.module.module_num = len(self.pipeline.modules()) + 1
        self.pipeline.add_module(self.module)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        module_panel = wx.lib.scrolledpanel.ScrolledPanel(self, style=wx.SUNKEN_BORDER)

        self.module_view = cellprofiler.gui.moduleview.ModuleView(module_panel, self.workspace, True)
        self.module_view.set_selection(self.module.module_num)

        def on_change(caller, event):
            setting = event.get_setting()
            proposed_value = event.get_proposed_value()
            setting.value = proposed_value
            self.pipeline.edit_module(event.get_module().module_num, False)
            self.module_view.reset_view()
            self.module_view.request_validation()

        self.module_view.add_listener(on_change)

        #
        # Add a panel for the "run" button
        #
        panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel, label="Run")

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
        self.tbicon.SetIcon(cellprofiler.gui.get_cp_icon(), "CellProfiler2.0")
        self.SetIcon(cellprofiler.gui.get_cp_icon())

    def on_load_measurements(self, event):
        dlg = wx.FileDialog(self, "Load a measurements file",
                            wildcard="Measurements file (*.mat,*.h5)|*.mat;*.h5",
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.load_measurements(dlg.GetPath())

    def on_save_measurements(self, event):
        with wx.FileDialog(self, "Save measurements file", wildcard=
        "CellProfiler measurements file (*.h5)|*.h5|"
        "Matlab measurements file (*.mat)|*.mat",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            assert isinstance(dlg, wx.FileDialog)
            if dlg.ShowModal() == wx.ID_OK:
                if dlg.GetFilterIndex() == 0:
                    new_measurements = cellprofiler.measurement.Measurements(
                            filename=dlg.Path,
                            copy=self.measurements)
                    new_measurements.flush()
                    new_measurements.close()
                else:
                    self.pipeline.save_measurements(dlg.GetPath(),
                                                    self.measurements)

    def on_exit(self, event):
        self.Close()

    def on_image_choose(self, event):
        """Choose an image from the image set"""
        dlg = wx.Dialog(self)
        dlg.Title = "Choose an image set"
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        choose_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(choose_sizer, 1, wx.EXPAND | wx.ALL, 5)
        choose_sizer.Add(wx.StaticText(dlg, -1, label="Image set:"),
                         0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        metadata_db = {}
        metadata_features = [
            x for x in self.measurements.get_feature_names(cellprofiler.measurement.IMAGE)
            if x.startswith('Metadata')]
        sel = None
        for i in self.measurements.get_image_numbers():
            metadata_key = ','.join(['%s=%s' % (
                feature,
                self.measurements.get_measurement(cellprofiler.measurement.IMAGE, feature, i))
                                     for feature in metadata_features])
            metadata_db[i] = metadata_key
            if i == self.measurements.image_set_number:
                sel = i
        choices = ["%d: %s" % (i, metadata_db[i])
                   for i in self.measurements.get_image_numbers()]
        choice_ctl = wx.Choice(dlg, -1, choices=choices)
        # Select the current image set
        if sel is not None:
            choice_ctl.SetSelection(sel)
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
        sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT | wx.ALL, 5)
        dlg.Fit()
        if dlg.ShowModal() == wx.ID_OK:
            index = choice_ctl.GetSelection()
            self.measurements.image_set_number = index + 1

    def load_measurements(self, measurements_file_name):
        self.measurements = cellprofiler.measurement.load_measurements(
                measurements_file_name, can_overwrite=True)
        # Start on the first image
        self.measurements.next_image_set(1)

    def on_run(self, event):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.gui.workspace.Workspace(self.pipeline,
                                                     self.module,
                                                     image_set,
                                                     cellprofiler.object.ObjectSet(),
                                                     self.measurements,
                                                     image_set_list,
                                                     frame=self)
        self.module.show_window = True  # to make sure it saves display data
        self.module.run_as_data_tool(workspace)
        self.measurements.flush()
        if self.module.show_window:
            fig = cellprofiler.gui.figure.create_or_find(
                    parent=self,
                    title="%s Output" % self.module.module_name,
                    name="CellProfiler:DataTool:%s" % self.module.module_name)
            self.module.display(workspace, fig)
            fig.figure.canvas.draw()
