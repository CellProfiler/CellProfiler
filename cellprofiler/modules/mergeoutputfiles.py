'''<b>MergeOutputFiles</b> merges several output .mat files into one.
<hr>

This data tool lets you collect the output .mat files from several runs, for instance,
as might be created by running CellProfiler in batch mode.

<b>MergeOutputFiles</b> is a pure data tool - <i>you cannot use it as a module</i>,
and it will generate an error if you try to do so. To use it as a data tool,
choose it from the <i>Data Tools</i> menu to bring up the <b>MergeOutputFiles</b> dialog.
The dialog has the following parts:
<ul><li><i>Destination file:</i> This is the name of the file that will be 
created. The file will contain all merged input data files in MATLAB format.</li>
<li><i>File list:</i> The file list is the box with the columns, "Folder" and
"File". It will be empty until you add files using the "Add..." button.
Measurement files are written out to the destination file in the order they
appear in this list.
You can select multiple files in this box to move them up or down or to
remove them.</li>
<li><i>Add button:</i> The Add button brings up a file chooser when you press
it. You can select multiple files from the file chooser and they will be
added in alphabetical order to the bottom of the current list of files.</li>
<li><i>Remove button:</i> The Remove button removes all currently selected
files from the list.</li>
<li><i>Up button</i> The Up button moves the currently selected files up in
the list.</li>
<li><i>Down button</i> The Down button moves the currently selected files
down in the list.</li>
<li><i>OK button</i> The OK button accepts the file list and writes it to
the output.</li>
<li><i>Cancel button</i> The cancel button closes the dialog without
performing any operation.</li></ul>

Once merged, this output file will be compatible with other data tools. 
Output files can be quite large, so prior to merging,
be sure that the total size of the merged output file is of a reasonable
size to be opened on your computer (based on the amount of memory
available on your computer). It may be preferable instead to import data
from individual output files directly into a database using <b>ExportDatabase</b> 
as a data tool.

See also <b>CreateBatchFiles</b>, <b>ExportToDatabase</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
#
__version__="$Revision$"

import numpy as np
import os

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
from cellprofiler.preferences import get_headless

class MergeOutputFiles(cpm.CPModule):
    module_name = "MergeOutputFiles"
    category = "Data Tools"
    do_not_check = True
    def run_as_data_tool(self):
        '''Run the module as a data tool'''
        import wx
        from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
        from cellprofiler.gui import get_cp_icon
        #
        # Portions of this were cribbed from the wx listctrl demo code
        # which is part of the wx source distribution
        #
        class AWListCtrl(wx.ListCtrl, ListCtrlAutoWidthMixin):
            '''A list control with autosizing of the last column'''
            def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
                wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
                ListCtrlAutoWidthMixin.__init__(self)

        dlg = wx.Dialog(None, title = "Merge output files",
                        size=(640,480),
                        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | 
                        wx.THICK_FRAME)
        dlg.SetIcon(get_cp_icon())
        #
        # Layout:
        # Dialog box
        #    sizer (vertical)
        #        list-control
        #        sizer (horizontal)
        #            Add...
        #            Remove
        #            Up
        #            Down
        #            button sizer
        #                OK
        #                Cancel
        #
        dlg.Sizer = sizer = wx.BoxSizer(wx.VERTICAL)
        subsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(subsizer, 0, wx.EXPAND | wx.ALL, 5)
        subsizer.Add(wx.StaticText(dlg, -1, "Destination file:"), 0, 
                     wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3)
        dest_file_ctrl = wx.TextCtrl(dlg)
        subsizer.Add(dest_file_ctrl, 1, wx.EXPAND | wx.LEFT, 3)
        browse_button = wx.Button(dlg, -1, "Browse...")
        subsizer.Add(browse_button, 0, wx.EXPAND)
        list_control = AWListCtrl(dlg, style=wx.LC_REPORT)
        list_control.InsertColumn(0, "Folder",wx.LC_ALIGN_LEFT, 200)
        list_control.InsertColumn(1, "File",wx.LC_ALIGN_LEFT, -1)
        sizer.Add(list_control, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticLine(dlg), 0, wx.EXPAND)
        subsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(subsizer, 0, wx.EXPAND | wx.ALL, 5)
        add_button = wx.Button(dlg, -1, "Add...")
        subsizer.Add(add_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        remove_button = wx.Button(dlg, wx.ID_REMOVE, "Remove selected")
        subsizer.Add(remove_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        up_button = wx.Button(dlg, wx.ID_UP, "Up")
        subsizer.Add(up_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        down_button = wx.Button(dlg, wx.ID_DOWN, "Down")
        subsizer.Add(down_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        
        button_sizer = wx.StdDialogButtonSizer()
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
        button_sizer.Realize()
        subsizer.Add(button_sizer, 0, wx.ALIGN_RIGHT )
        dlg.Layout()
        #
        # Order is a map of item ID to its position in the list control
        #
        order = {}
        
        add_button.Bind(wx.EVT_BUTTON, 
                        lambda event: self.on_add(event, list_control, order))
        remove_button.Bind(wx.EVT_BUTTON,
                           lambda event: self.on_remove(event, list_control, 
                                                        order))
        browse_button.Bind(wx.EVT_BUTTON,
                           lambda event: self.on_browse(event, dest_file_ctrl))
        up_button.Bind(wx.EVT_BUTTON,
                       lambda event: self.on_up(event, list_control, order))
        down_button.Bind(wx.EVT_BUTTON,
                         lambda event: self.on_down(event, list_control, order))
        if dlg.ShowModal() == wx.ID_OK:
            sources = []
            for i in range(list_control.ItemCount):
                item_id = list_control.GetItemData(i)
                path = list_control.GetItem(i, 0).GetText()
                filename = list_control.GetItem(i, 1).GetText()
                sources.append(os.path.join(path, filename))
            self.merge_files(dest_file_ctrl.Value, sources)
        dlg.Destroy()
        
    @staticmethod
    def on_add(event, list_control, order):
        '''Handle the add button being pressed'''
        import wx
        assert isinstance(list_control, wx.ListCtrl)
        dlg = wx.FileDialog(list_control.Parent, 
                            message = "Select data files to merge",
                            style = wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
        dlg.Wildcard = "CellProfiler data (*.mat)|*.mat|All files (*.*)|*.*"
        if dlg.ShowModal() == wx.ID_OK:
            for path in sorted(dlg.Paths):
                folder, filename = os.path.split(path)
                item_id = list_control.Append([folder, filename])
                list_control.SetItemData(item_id, item_id)
                order[item_id] = list_control.ItemCount - 1
        dlg.Destroy()
        
    @staticmethod
    def on_remove(event, list_control, order):
        '''Remove the selected items for the list control'''
        import wx
        assert isinstance(list_control, wx.ListCtrl)
        def selections():
            current_selection = list_control.GetFirstSelected()
            while(current_selection != -1):
                yield current_selection
                current_selection = list_control.GetNextSelected(current_selection)
                
        for selection in reversed(list(selections())):
            list_control.DeleteItem(selection)
            del order[selection]
        #
        # Renumber with consecutive values
        #
        anti_order = MergeOutputFiles.get_anti_order(order)
        for i, key in enumerate(sorted(anti_order.keys())):
            order[anti_order[key]] = i
    @staticmethod
    def get_anti_order(order):
        '''Return a dictionary whose values are the keys of the input and vice versa'''
        anti_order = {}
        for key in order.keys():
            anti_order[order[key]] = key
        return anti_order
        
    @staticmethod
    def on_browse(event, ctrl):
        import wx
        assert isinstance(ctrl, wx.TextCtrl)
        dlg = wx.FileDialog(ctrl.Parent,
                            message = "Merged output file name",
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        dlg.Wildcard = "CellProfiler data (*.mat)|*.mat"
        if len(ctrl.Value) > 0:
            dlg.Path = ctrl.Value
        if dlg.ShowModal() == wx.ID_OK:
            ctrl.Value = dlg.Path
            
    @staticmethod
    def on_up(event, list_ctrl, order):
        import wx
        assert isinstance(list_ctrl, wx.ListCtrl)
        anti_order = MergeOutputFiles.get_anti_order(order)
        last = None
        for i in sorted(anti_order.keys()):
            current_id = anti_order[i]
            # IsSelected takes the index of the item, not its ID
            if list_ctrl.IsSelected(i):
                if last is not None:
                    #
                    # Switch the positions of the selected and the last
                    # unselected
                    #
                    last_i = order[last]
                    order[last] = order[current_id]
                    order[current_id] = last_i
            else:
                last = current_id
        MergeOutputFiles.sort(list_ctrl, order)
    
    @staticmethod
    def on_down(event, list_ctrl, order):
        import wx
        assert isinstance(list_ctrl, wx.ListCtrl)
        anti_order = MergeOutputFiles.get_anti_order(order)
        last = None
        for i in reversed(anti_order.keys()):
            current_id = anti_order[i]
            if list_ctrl.IsSelected(i):
                if last is not None:
                    #
                    # Switch the positions of the unselected and the last
                    # selected
                    #
                    last_i = order[last]
                    order[last] = order[current_id]
                    order[current_id] = last_i
            else:
                last = current_id
        MergeOutputFiles.sort(list_ctrl, order)
        
    @staticmethod
    def sort(list_ctrl, order):
        '''Sort the items in the list control according to the order
        
        list_ctrl - list control with items indicated by order's keys
        
        order - a dictionary where the keys are the item ids
                and the values are the relative order of those ids
                with respect to each other
        '''
        def sortfn(item1, item2):
            return cmp(order[item1], order[item2])
        list_ctrl.SortItems(sortfn)
        
    @staticmethod
    def merge_files(destination, sources, force_headless=False):
        is_headless = force_headless or get_headless()
        if not is_headless:
            import wx
        if len(sources) == 0:
            return
        if not is_headless:
            progress = wx.ProgressDialog("Writing " + destination,
                                         "Loading " + sources[0],
                                         maximum = len(sources) * 4 + 1,
                                         style = wx.PD_CAN_ABORT | 
                                         wx.PD_APP_MODAL |
                                         wx.PD_ELAPSED_TIME |
                                         wx.PD_REMAINING_TIME)
        count = 0
        try:
            pipeline = cpp.Pipeline()
            has_error = [False]
            def callback(caller, event):
                if isinstance(event, cpp.LoadExceptionEvent):
                    has_error = True
                    wx.MessageBox(
                        message = "Could not load %s: %s" % (
                            sources[0], event.error),
                        caption = "Failed to load %s" % sources[0])
                    has_error[0] = True

            pipeline.add_listener(callback)
            #pipeline.load(sources[0])
            if has_error[0]:
                return

            # distributed processing passes a list of functions, not a
            # list of filenames.
            pipeline.load(sources[0]() if callable(sources[0]) else sources[0])
            mdest = cpmeas.Measurements()
            image_set_count = 0
            for source in sources:
                if not is_headless:
                    count += 1
                    keep_going, skip = progress.Update(count, "Loading " + source)
                    if not keep_going:
                        return
                # distributed processing passes a list of functions
                msource = cpmeas.load_measurements(source() if callable(source) else source)
                source_image_set_count = 0
                for object_name in msource.get_object_names():
                    if object_name in mdest.get_object_names():
                        destfeatures = mdest.get_feature_names(object_name)
                    else:
                        destfeatures = []
                    for feature in msource.get_feature_names(object_name):
                        src_values = msource.get_all_measurements(object_name,
                                                                  feature)
                        if np.isscalar(src_values):
                            # For something like "Experiment", there is a single
                            # value. Keep the first value seen among all sources.
                            #
                            if not feature in destfeatures:
                                mdest.add_all_measurements(object_name, feature,
                                                           [src_values])
                        else:
                            source_image_count = max(source_image_set_count,
                                                     len(src_values))
                            if feature in destfeatures:
                                dest_values = mdest.get_all_measurements(object_name,
                                                                         feature)
                                if not isinstance(dest_values, list):
                                    dest_values = dest_values.tolist()
                            else:
                                dest_values = [None] * image_set_count

                            if isinstance(src_values, list):
                                dest_values += src_values
                            else:
                                dest_values += src_values.tolist()
                            mdest.add_all_measurements(object_name, feature, dest_values)
                    destset = set(destfeatures)
                    #
                    # These are features that are in the destination, but not
                    # in the source. We have to add Nones to the destination.
                    #
                    for feature in destset.difference(
                        msource.get_feature_names(object_name)):
                        dest_values = mdest.get_all_measurements(
                            object_name, feature)
                        dest_values += [None] * source_image_count
                #
                # The source may not have all of the objects that are in
                # the destination. Add blanks here.
                #
                destobjects = set(mdest.get_object_names())
                for object_name in destobjects.difference(
                    msource.get_object_names()):
                    for feature in mdest.get_feature_names(object_name):
                        dest_values = mdest.get_all_measurements(
                            object_name, feature)
                        dest_values += [None] * source_image_count
                        mdest.add_all_measurements(object_name, feature, dest_values)
                image_set_count += source_image_count
            mdest.image_set_number = image_set_count
            if not is_headless:
                keep_going, skip = progress.Update(count+1, "Saving to "+destination)
                if not keep_going:
                    return
            pipeline.save_measurements(destination, mdest)
        finally:
            if not is_headless:
                progress.Destroy()
        
if __name__ == "__main__":
    import cellprofiler.modules
    import wx
    app = wx.PySimpleApp(False)
    mof = MergeOutputFiles()
    mof.run_as_data_tool()
    
