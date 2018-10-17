# coding=utf-8
""" runmultiplepipelinesdialog.py - Dialog to collect info for RunMultiplePipelines
"""

import cellprofiler.gui.pipeline
import cellprofiler.pipeline
import cellprofiler.preferences
import datetime
import os
import sys
import wx

FC_FILENAME_COLUMN = 0
FC_DATE_COLUMN = 1
FC_MODULE_COUNT_COLUMN = 2

P_FILENAME_COLUMN = 0
P_INPUT_DIRECTORY_COLUMN = 1
P_OUTPUT_DIRECTORY_COLUMN = 2
P_OUTPUT_FILE_COLUMN = 3
P_REMOVE_BUTTON_COLUMN = 4


class RunMultplePipelinesDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super(RunMultplePipelinesDialog, self).__init__(*args, **kwargs)

        #
        # Main layout:
        #
        # # # # # # # # # # # # # # # # # # # #
        #                         #           #
        #    File chooser         #  select / #
        #                         #  add      #
        #                         #  buttons  #
        # # # # # # # # # # # # # # # # # # # #
        # Directory picker                   #
        # # # # # # # # # # # # # # # # # # # #
        #                                     #
        # Pipeline list box                   #
        #                                     #
        # # # # # # # # # # # # # # # # # # # #
        # OK / Cancel / Help buttons          #
        # # # # # # # # # # # # # # # # # # # #

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.top_sizer, 1, wx.EXPAND)
        self.add_file_chooser(self.top_sizer)
        self.add_directory_picker(self.sizer)
        self.add_pipeline_list_box(self.sizer)
        self.add_dialog_buttons(self.sizer)

        self.hookup_events()
        self.set_path(cellprofiler.preferences.get_default_output_directory())
        self.Layout()

    def add_file_chooser(self, sizer):
        """Add UI elements for displaying files in a directory."""
        #
        # File list box on left, buttons on right
        #
        self.file_chooser = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.file_chooser.InsertColumn(FC_FILENAME_COLUMN, "File name")
        self.file_chooser.InsertColumn(FC_DATE_COLUMN, "Last modified")
        self.file_chooser.InsertColumn(FC_MODULE_COUNT_COLUMN, "# modules")
        self.file_chooser.SetColumnWidth(FC_MODULE_COUNT_COLUMN, wx.LIST_AUTOSIZE_USEHEADER)
        sizer.Add(self.file_chooser, 1, wx.EXPAND | wx.ALL, 5)

        self.file_button_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.file_button_sizer, 0, wx.EXPAND)
        self.file_select_all_button = wx.Button(self, label="Select all")
        self.file_button_sizer.Add(self.file_select_all_button)
        self.file_deselect_all_button = wx.Button(self, label="Deselect all")
        self.file_button_sizer.Add(self.file_deselect_all_button)
        self.file_add_button = wx.Button(self, label="Add")
        self.file_button_sizer.Add(self.file_add_button)

    def add_directory_picker(self, sizer):
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 5)
        sub_sizer.Add(wx.StaticText(self, label="Folder:"), 0,
                      wx.ALIGN_RIGHT | wx.RIGHT, 3)
        self.directory_picker = wx.DirPickerCtrl(self)
        sub_sizer.Add(self.directory_picker, 1, wx.EXPAND)

    def add_pipeline_list_box(self, sizer):
        self.plv_image_list = wx.ImageList(16, 16)
        delete_bmp = wx.ArtProvider.GetBitmap(wx.ART_DELETE, size=(16, 16))
        self.delete_bmp_idx = self.plv_image_list.Add(delete_bmp)
        self.pipeline_list_view = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.pipeline_list_view.SetImageList(self.plv_image_list, wx.IMAGE_LIST_SMALL)
        sizer.Add(self.pipeline_list_view, 1, wx.EXPAND | wx.ALL, 5)
        self.pipeline_list_view.InsertColumn(P_FILENAME_COLUMN, "Pipeline")
        self.pipeline_list_view.InsertColumn(P_INPUT_DIRECTORY_COLUMN,
                                             "Default input folder")
        self.pipeline_list_view.InsertColumn(P_OUTPUT_DIRECTORY_COLUMN,
                                             "Default output folder")
        self.pipeline_list_view.InsertColumn(P_OUTPUT_FILE_COLUMN,
                                             "Measurements file")
        self.pipeline_list_view.InsertColumn(P_REMOVE_BUTTON_COLUMN, "Remove")

    def add_dialog_buttons(self, sizer):
        self.btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        self.btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        self.btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_HELP)
        self.btnsizer.AddButton(btn)
        self.btnsizer.Realize()
        sizer.Add(self.btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)

    def hookup_events(self):
        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_path_changed, self.directory_picker)
        self.Bind(wx.EVT_TEXT, self.on_path_changed, self.directory_picker.TextCtrl)
        self.Bind(wx.EVT_BUTTON, self.on_add, self.file_add_button)
        self.Bind(wx.EVT_BUTTON, self.on_select_all, self.file_select_all_button)
        self.Bind(wx.EVT_BUTTON, self.on_deselect_all, self.file_deselect_all_button)
        self.Bind(wx.EVT_BUTTON, self.on_help, id=wx.ID_HELP)
        self.pipeline_list_view.Bind(wx.EVT_LEFT_DOWN, self.on_pipeline_list_down)
        if sys.platform.startswith('linux'):
            for child in self.pipeline_list_view.GetChildren():
                child.Bind(wx.EVT_LEFT_DOWN, self.on_pipeline_list_down)

    def set_path(self, path):
        self.directory_picker.Path = path
        self.set_file_chooser_path(path)

    def on_path_changed(self, event):
        self.set_file_chooser_path(event.Path)

    def set_file_chooser_path(self, path):
        file_names = []
        for file_name in os.listdir(path):
            ext = os.path.splitext(file_name)[1].lower()
            if len(ext) > 0 and ext[1:] in cellprofiler.preferences.EXT_PIPELINE_CHOICES:
                file_names.append(file_name)
        self.file_chooser.DeleteAllItems()
        module_count = [None]

        def on_pipeline_event(caller, event):
            if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
                module_count[0] = None
            elif isinstance(event, cellprofiler.pipeline.PipelineLoadedEvent):
                module_count[0] = len(caller.modules())

        pipeline = cellprofiler.gui.pipeline.Pipeline()
        pipeline.add_listener(on_pipeline_event)
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            mtime = os.path.getmtime(file_path)
            mtime = datetime.datetime.fromtimestamp(mtime)
            mtime = mtime.ctime()
            pipeline.load(file_path)
            if module_count[0] is not None:
                index = self.file_chooser.InsertStringItem(
                        sys.maxsize, file_name)
                self.file_chooser.SetStringItem(index, FC_DATE_COLUMN, mtime)
                self.file_chooser.SetStringItem(index, FC_MODULE_COUNT_COLUMN,
                                                str(module_count[0]))
        self.file_chooser.SetColumnWidth(FC_FILENAME_COLUMN, wx.LIST_AUTOSIZE)
        self.file_chooser.SetColumnWidth(FC_DATE_COLUMN, wx.LIST_AUTOSIZE)

    def on_select_all(self, event):
        for i in range(self.file_chooser.ItemCount):
            self.file_chooser.Select(i, True)
            self.file_chooser.SetFocus()

    def on_deselect_all(self, event):
        for i in range(self.file_chooser.ItemCount):
            self.file_chooser.Select(i, False)
        self.file_chooser.SetFocus()

    def on_add(self, event):
        for i in range(self.file_chooser.ItemCount):
            if self.file_chooser.IsSelected(i):
                path = os.path.join(
                        self.directory_picker.Path,
                        self.file_chooser.GetItemText(i))
                index = self.pipeline_list_view.InsertStringItem(
                        sys.maxsize, path)
                self.pipeline_list_view.SetStringItem(
                        index, P_INPUT_DIRECTORY_COLUMN,
                        cellprofiler.preferences.get_default_image_directory())
                self.pipeline_list_view.SetStringItem(
                        index, P_OUTPUT_DIRECTORY_COLUMN,
                        cellprofiler.preferences.get_default_output_directory())
                self.pipeline_list_view.SetStringItem(
                        index, P_OUTPUT_FILE_COLUMN,
                        cellprofiler.preferences.get_output_file_name())
                self.pipeline_list_view.SetItemColumnImage(
                        index, P_REMOVE_BUTTON_COLUMN, self.delete_bmp_idx)
                self.file_chooser.Select(i, False)
        self.pipeline_list_view.SetColumnWidth(P_FILENAME_COLUMN, wx.LIST_AUTOSIZE)
        self.pipeline_list_view.SetColumnWidth(P_INPUT_DIRECTORY_COLUMN, wx.LIST_AUTOSIZE)
        self.pipeline_list_view.SetColumnWidth(P_OUTPUT_DIRECTORY_COLUMN, wx.LIST_AUTOSIZE)
        self.pipeline_list_view.SetColumnWidth(P_OUTPUT_FILE_COLUMN, wx.LIST_AUTOSIZE)

    def on_pipeline_list_down(self, event):
        if sys.platform.startswith("win"):
            item, hit_code, subitem = self.pipeline_list_view.HitTestSubItem(event.Position)
        else:
            # Mac's HitTestSubItem does not work. Sorry.
            #
            item, hit_code = self.pipeline_list_view.HitTest(event.Position)
            widths = [self.pipeline_list_view.GetColumnWidth(i) for i in range(4)]
            start = 0
            for subitem in range(4):
                if event.Position[0] < start + widths[subitem]:
                    break
                start += widths[subitem]
        if 0 <= item < self.pipeline_list_view.ItemCount and (hit_code & wx.LIST_HITTEST_ONITEM):
            if subitem == P_REMOVE_BUTTON_COLUMN:
                self.pipeline_list_view.DeleteItem(item)
            elif subitem == P_FILENAME_COLUMN:
                dlg = wx.FileDialog(self, "Choose a pipeline file", style=wx.FD_OPEN)
                dlg.Path = self.pipeline_list_view.GetItemText(item)
                dlg.Wildcard = "CellProfiler pipeline (*.cp)|*.cp|Measurements file (*.mat)|*.mat"
                result = dlg.ShowModal()
                if result == wx.ID_OK:
                    self.pipeline_list_view.SetItemText(item, dlg.Path)
            elif subitem in (P_INPUT_DIRECTORY_COLUMN, P_OUTPUT_DIRECTORY_COLUMN):
                dest = "input" if subitem == P_INPUT_DIRECTORY_COLUMN else "output"
                dlg = wx.DirDialog(self, "Choose the default %s folder to use for the pipeline")
                list_item = self.pipeline_list_view.GetItem(item, subitem)
                dlg.Path = list_item.GetText()
                if dlg.ShowModal() == wx.ID_OK:
                    self.pipeline_list_view.SetStringItem(item, subitem, dlg.Path)
            elif subitem == P_OUTPUT_FILE_COLUMN:
                dlg = wx.FileDialog(self, "Choose an output measurements file",
                                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
                list_item = self.pipeline_list_view.GetItem(item, subitem)
                dlg.Path = list_item.GetText()
                dlg.Wildcard = "Measurements file (*.mat)|*.mat"
                if dlg.ShowModal() == wx.ID_OK:
                    self.pipeline_list_view.SetStringItem(item, subitem, dlg.Filename)

    def on_help(self, event):
        import cellprofiler.gui.help.content
        import cellprofiler.gui.html.utils
        import cellprofiler.gui.htmldialog
        content = cellprofiler.gui.help.content.read_content("other_multiple_pipelines.rst")
        with cellprofiler.gui.htmldialog.HTMLDialog(self,
                                                    "Run multiple pipelines help",
                                                    cellprofiler.gui.html.utils.rst_to_html_fragment(content)
                                                    ) as dlg:
            dlg.ShowModal()

    def get_pipelines(self):
        """Return the user's chosen pipelines & other details

        The return value is a list of objects with the following attributes
        .path - path to the pipeline file
        .default_input_folder - default input folder when running the pipeline
        .default_output_folder - default output folder when running the pipeline
        .measurements_file - .mat file to use when storing measurements. Can
                             be None if none wanted.
        """
        result = []

        class PipelineDetails(object):
            def __init__(self, path, default_input_folder, default_output_folder,
                         measurements_file):
                self.path = path
                self.default_input_folder = default_input_folder
                self.default_output_folder = default_output_folder
                self.measurements_file = measurements_file

        for i in range(self.pipeline_list_view.ItemCount):
            path, default_input_folder, default_output_folder, measurements_file = \
                [self.pipeline_list_view.GetItem(i, j).GetText()
                 for j in (P_FILENAME_COLUMN, P_INPUT_DIRECTORY_COLUMN,
                           P_OUTPUT_DIRECTORY_COLUMN, P_OUTPUT_FILE_COLUMN)]
            result.append(PipelineDetails(
                    path, default_input_folder, default_output_folder,
                    measurements_file))
        return result
