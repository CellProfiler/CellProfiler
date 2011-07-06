"""ModuleView.py - implements a view on a module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import codecs
import logging
import matplotlib.cm
import numpy as np
import os
import stat
import time
import traceback
import wx
import wx.grid
import sys

logger = logging.getLogger(__name__)
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.html import HtmlClickableWindow
from regexp_editor import edit_regexp
from htmldialog import HTMLDialog
from treecheckboxdialog import TreeCheckboxDialog
from metadatactrl import MetadataControl

ERROR_COLOR = wx.RED
WARNING_COLOR = wx.Colour(224,224,0,255)
RANGE_TEXT_WIDTH = 40 # number of pixels in a range text box TO_DO - calculate it
ABSOLUTE = "Absolute"
FROM_EDGE = "From edge"

CHECK_TIMEOUT_SEC = 2

class SettingEditedEvent:
    """Represents an attempt by the user to edit a setting
    
    """
    def __init__(self, setting, module, proposed_value, event):
        self.__module = module
        self.__setting = setting
        self.__proposed_value = proposed_value
        self.__event = event
        self.__accept_change = True
    
    def get_setting(self):
        """Return the setting being edited
        
        """
        return self.__setting
    
    def get_proposed_value(self):
        """Return the value proposed by the user
        
        """
        return self.__proposed_value
    
    def get_module(self):
        """Get the module holding the setting"""
        return self.__module
    
    def cancel(self):
        self.__accept_change = False
        
    def accept_change(self):
        return self.__accept_change
    def ui_event(self):
        """The event from the UI that triggered the edit
        
        """
        return self.__event

def text_control_name(v):
    """Return the name of a setting's text control
    v - the setting
    The text control name is built using the setting's key
    """
    return "%s_text"%(str(v.key()))

def button_control_name(v):
    """Return the name of a setting's button
    v - the setting
    """
    return "%s_button"%(str(v.key()))

def edit_control_name(v):
    """Return the name of a setting's edit control
    v - the setting
    The edit control name is built using the setting's key
    """
    return str(v.key())

def min_control_name(v):
    """For a range, return the control that sets the minimum value
    v - the setting
    """
    return "%s_min"%(str(v.key()))

def max_control_name(v):
    """For a range, return the control that sets the maximum value
    v - the setting
    """
    return "%s_max"%(str(v.key()))

def absrel_control_name(v):
    """For a range, return the control that chooses between absolute and relative
    
    v - the setting
    Absolute - far coordinate is an absolute value
    From edge - far coordinate is a distance from the far edge
    """
    return "%s_absrel"%(str(v.key()))

def x_control_name(v):
    """For coordinates, return the control that sets the x value
    v - the setting
    """
    return "%s_x"%(str(v.key()))

def y_control_name(v):
    """For coordinates, return the control that sets the y value
    v - the setting
    """
    return "%s_y"%(str(v.key()))

def category_control_name(v):
    '''For measurements, return the control that sets the measurement category
    
    v - the setting
    '''
    return "%s_category"%(str(v.key()))

def category_text_control_name(v):
    return "%s_category_text"%(str(v.key()))

def feature_control_name(v):
    '''For measurements, return the control that sets the feature name

    v - the setting
    '''
    return "%s_feature"%(str(v.key()))

def feature_text_control_name(v):
    return "%s_feature_text"%(str(v.key()))

def image_control_name(v):
    '''For measurements, return the control that sets the image name

    v - the setting
    '''
    return "%s_image"%(str(v.key()))

def image_text_control_name(v):
    return "%s_image_text"%(str(v.key()))

def object_control_name(v):
    '''For measurements, return the control that sets the object name

    v - the setting
    '''
    return "%s_object"%(str(v.key()))

def object_text_control_name(v):
    return "%s_object_text"%(str(v.key()))

def scale_control_name(v):
    '''For measurements, return the control that sets the measurement scale

    v - the setting
    '''
    return "%s_scale"%(str(v.key()))

def scale_text_ctrl_name(v):
    return "%s_scale_text"%(str(v.key()))

def combobox_ctrl_name(v):
    return "%s_combobox"%(str(v.key()))

def colorbar_ctrl_name(v):
    return "%s_colorbar"%(str(v.key()))

def help_ctrl_name(v):
    return "%s_help" % str(v.key())

def subedit_control_name(v):
    return "%s_subedit" % str(v.key())

def custom_label_name(v):
    return "%s_customlabel" % str(v.key())

def encode_label(text):
    """Encode text escapes for the static control and button labels
    
    The ampersand (&) needs to be encoded as && for wx.StaticText
    and wx.Button in order to keep it from signifying an accelerator.
    """
    return text.replace('&','&&')


class ModuleView:

    """The module view implements a view on CellProfiler.Module
    
    The module view implements a view on CellProfiler.Module. The view consists
    of a table composed of one row per setting. The first column of the table
    has the explanatory text and the second has a control which
    gives the ui for editing the setting.
    """
    
    def __init__(self, module_panel, pipeline, as_datatool=False):
        #############################################
        #
        # Build the top-level GUI windows
        #
        #############################################
        self.top_panel = module_panel
        self.notes_panel = wx.Panel(self.top_panel)
        self.__module_panel = wx.Panel(self.top_panel)
        self.__sizer = ModuleSizer(0, 3)
        self.module_panel.Bind(wx.EVT_CHILD_FOCUS, self.skip_event)
        self.module_panel.SetSizer(self.__sizer)
        self.top_level_sizer = wx.BoxSizer(wx.VERTICAL)
        self.top_panel.SetSizer(self.top_level_sizer)
        self.make_notes_gui()
        self.module_panel.Hide()
        self.notes_panel.Hide()
        if not as_datatool:
            self.top_level_sizer.Add(self.notes_panel, 0, wx.EXPAND | wx.ALL, 4)
        self.top_level_sizer.Add(self.module_panel, 1, wx.EXPAND | wx.ALL, 4)

        self.__pipeline = pipeline
        self.__as_datatool = as_datatool
        pipeline.add_listener(self.__on_pipeline_event)
        self.__listeners = []
        self.__value_listeners = []
        self.__module = None
        self.__inside_notify = False
        self.__handle_change = True
        self.__notes_text = None
        if cpprefs.get_startup_blurb():
            self.__startup_blurb = HtmlClickableWindow(self.top_panel, wx.ID_ANY, style=wx.NO_BORDER)
            self.__startup_blurb.load_startup_blurb()
            self.top_level_sizer.Add(self.__startup_blurb, 1, wx.EXPAND)
        else:
            self.__startup_blurb = None
        wx.EVT_SIZE(self.top_panel, self.on_size)
        wx.EVT_IDLE(self.top_panel, self.on_idle)

    def skip_event(self, event):
        event.Skip(False)

    def get_module_panel(self):
        """The panel that hosts the module controls
        
        This is exposed for testing purposes.
        """
        return self.__module_panel
    
    module_panel = property(get_module_panel)

    # ~*~
    def get_current_module(self):
        return self.__module
    # ~^~
    
    def clear_selection(self):
        if self.__module:
            for listener in self.__value_listeners:
                listener['notifier'].remove_listener(listener['listener'])
            self.__value_listeners = []
            self.__module = None
        self.__sizer.Reset(0,3)
        self.notes_panel.Hide()
    
    def hide_settings(self):
        for child in self.__module_panel.Children:
            child.Hide()
        
    def set_selection(self,module_num):
        """Initialize the controls in the view to the settings of the module"""
        self.top_panel.Freeze()
        if self.__startup_blurb:
            self.__startup_blurb.Destroy()
            self.__startup_blurb = None
        self.module_panel.Show()
        self.__module_panel.SetVirtualSizeWH(0, 0)
        self.top_panel.SetupScrolling(scrollToTop=False)
        self.__handle_change = False
        try:
            new_module          = self.__pipeline.module(module_num)
            reselecting         = (self.__module and
                                   self.__module.id == new_module.id)
            if not reselecting:
                self.clear_selection()
            if not self.__as_datatool:
                self.notes_panel.Show()
            self.__module       = new_module
            self.__controls     = []
            self.__static_texts = []
            data                = []
            settings            = self.__module.visible_settings()
            try:
                assert len(settings) > 0
            except:
                wx.MessageBox("Module %s.visible_settings() did not return a list!\n  value: %s"%(self.__module.module_name, settings),
                              "Pipeline Error", wx.ICON_ERROR, self.__module_panel)
                settings = []
            
            self.__sizer.Reset(len(settings), 3, False)
            sizer    = self.__sizer
            if reselecting:
                self.hide_settings()
                
            #################################
            #
            # Set the module's notes
            #
            #################################
            self.module_notes_control.Value = "\n".join(self.__module.notes)
            
            #################################
            #
            # Populate the GUI elements for each of the settings
            #
            #################################
            for i, v in enumerate(settings):
                flag = wx.EXPAND
                border = 0
                control_name = edit_control_name(v)
                text_name    = text_control_name(v)
                static_text  = self.__module_panel.FindWindowByName(text_name)
                control      = self.__module_panel.FindWindowByName(control_name)
                if static_text:
                    static_text.Show()
                    static_text.Label = encode_label(v.text)
                else:
                    static_text = wx.StaticText(self.__module_panel,
                                                -1,
                                                encode_label(v.text),
                                                style=wx.ALIGN_RIGHT,
                                                name=text_name)
                sizer.Add(static_text,3,wx.EXPAND|wx.ALL,2)
                if control:
                    control.Show()
                self.__static_texts.append(static_text)
                if isinstance(v,cps.Binary):
                    control = self.make_binary_control(v,control_name,control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cps.MeasurementMultiChoice):
                    control = self.make_measurement_multichoice_control(
                        v, control_name, control)
                elif isinstance(v, cps.SubdirectoryFilter):
                    control = self.make_subdirectory_filter_control(
                        v, control_name, control)
                elif isinstance(v, cps.MultiChoice):
                    control = self.make_multichoice_control(v, control_name, 
                                                            control)
                elif isinstance(v,cps.CustomChoice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name, 
                                                       wx.CB_DROPDOWN,
                                                       control)
                elif isinstance(v,cps.Colormap):
                    control = self.make_colormap_control(v, control_name, 
                                                         control)
                elif isinstance(v,cps.Choice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name, 
                                                       wx.CB_READONLY,
                                                       control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v,cps.NameSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_choice_control(v, choices,
                                                       control_name, 
                                                       wx.CB_READONLY,
                                                       control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v,cps.FigureSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_choice_control(v, choices,
                                                       control_name, 
                                                       wx.CB_DROPDOWN,
                                                       control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cps.DoSomething):
                    control = self.make_callback_control(v, control_name,
                                                         control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cps.IntegerRange) or\
                     isinstance(v, cps.FloatRange):
                    control = self.make_range_control(v, control)
                elif isinstance(v,
                                cps.IntegerOrUnboundedRange):
                    control = self.make_unbounded_range_control(v, control)
                elif isinstance(v, cps.Coordinates):
                    control = self.make_coordinates_control(v,control)
                elif isinstance(v, cps.RegexpText):
                    control = self.make_regexp_control(v, control)
                elif isinstance(v, cps.Measurement):
                    control = self.make_measurement_control(v, control)
                elif isinstance(v, cps.Divider):
                    if control is None:
                        if v.line:
                            control = wx.StaticLine(self.__module_panel, 
                                                    name = control_name)
                        else:
                            control = wx.StaticText(self.__module_panel, 
                                                    name = control_name)
                    flag = wx.EXPAND|wx.ALL
                    border = 2
                elif isinstance(v, cps.FilenameText):
                    control = self.make_filename_text_control(v, control)
                elif isinstance(v, cps.DirectoryPath):
                    control = self.make_directory_path_control(v, control_name,
                                                               control)
                elif isinstance(v, cps.Color):
                    control = self.make_color_control(v, control_name, control)
                else:
                    control = self.make_text_control(v, control_name, control)
                sizer.Add(control, 0, flag, border)
                self.__controls.append(control)
                help_name = help_ctrl_name(v)
                help_control = self.module_panel.FindWindowByName(help_name)
                    
                if help_control is None:
                    if v.doc is None:
                        help_control = wx.StaticText(self.__module_panel, 
                                                     -1, "",
                                                     name = help_name)
                    else:
                        help_control = self.make_help_control(v.doc, v.text, 
                                                              name = help_name)
                else:
                    help_control.Show()
                sizer.Add(help_control, 0, wx.LEFT, 2)
            self.module_panel.Fit()
            self.top_panel.FitInside()
        finally:
            self.top_panel.Thaw()
            self.validate_module()
            self.top_panel.Refresh()
            self.__handle_change = True

    def make_notes_gui(self):
        '''Make the GUI elements that contain the module notes'''
        #
        # The notes sizer contains a static box that surrounds the notes
        # plus the notes text control.
        #
        notes_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.notes_panel, -1, "Module notes"),
            wx.VERTICAL)
        self.notes_panel.SetSizer(notes_sizer)
        self.module_notes_control = wx.TextCtrl(
            self.notes_panel, -1, style = wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
        notes_sizer.Add(self.module_notes_control, 1, wx.EXPAND)
        def on_notes_changed(event):
            if not self.__handle_change:
                return
            if self.__module is not None:
                notes = str(self.module_notes_control.Value)
                self.__module.notes = notes.split('\n')
        self.notes_panel.Bind(wx.EVT_TEXT, on_notes_changed,
                               self.module_notes_control)
        
    def make_binary_control(self,v,control_name, control):
        """Make a checkbox control for a Binary setting"""
        if not control:
            control = wx.CheckBox(self.__module_panel,-1,name=control_name)
            def callback(event, setting=v, control=control):
                self.__on_checkbox_change(event, setting, control)
                
            self.__module_panel.Bind(wx.EVT_CHECKBOX,
                                     callback,
                                     control)
        control.SetValue(v.is_yes)
        return control
    
    def make_choice_control(self,v,choices,control_name,style,control):
        """Make a combo-box that shows choices
        
        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles 
        """
        if v.value not in choices and style == wx.CB_READONLY:
            choices = choices + [v.value]
        if not control:
            control = wx.ComboBox(self.__module_panel,-1,v.value,
                                  choices=choices,
                                  style=style,
                                  name=control_name)
            def callback(event, setting=v, control = control):
                self.__on_combobox_change(event, setting,control)
            self.__module_panel.Bind(wx.EVT_COMBOBOX,callback,control)
            if style == wx.CB_DROPDOWN:
                def on_cell_change(event, setting=v, control=control):
                    self.__on_cell_change(event, setting, control)
                self.__module_panel.Bind(wx.EVT_TEXT,on_cell_change,control)
        else:
            old_choices = control.Items
            if len(choices)!=len(old_choices) or\
               not all([x==y for x,y in zip(choices,old_choices)]):
                if v.value in old_choices:
                    # For Mac, if you change the choices and the current
                    # combo-box value isn't in the choices, it throws
                    # an exception. Windows is much more forgiving.
                    # But the Mac has those buttons that look like little
                    # jellies, so it is better.
                    control.Value = v.value
                control.Items = choices
            try:
                # more desperate MAC cruft
                i_am_different = (control.Value != v.value)
            except:
                i_am_different = True
            if len(choices) > 0 and i_am_different:
                control.Value = v.value
        
        if (getattr(v,'has_tooltips',False) and 
            v.has_tooltips and v.tooltips.has_key(control.Value)):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control
    
    def make_measurement_multichoice_control(self, v, control_name, control):
        '''Make a button that, when pressed, launches the tree editor'''
        if control is None:
            control = wx.Button(self.module_panel, -1,
                                "Press to select measurements")
            def on_press(event):
                d = {}
                assert isinstance(v, cps.MeasurementMultiChoice)
                #
                # Populate the tree
                #
                for choice in v.choices:
                    object_name, feature = v.split_choice(choice)
                    pieces = [object_name] + feature.split('_')
                    d1 = d
                    for piece in pieces:
                        if not d1.has_key(piece):
                            d1[piece] = {}
                            d1[None] = 0
                        d1 = d1[piece]
                    d1[None] = False
                #
                # Mark selected leaf states as true
                #
                for selection in v.selections:
                    object_name, feature = v.split_choice(selection)
                    pieces = [object_name] + feature.split('_')
                    d1 = d
                    for piece in pieces:
                        if not d1.has_key(piece):
                            break
                        d1 = d1[piece]
                    d1[None] = True
                #
                # Backtrack recursively through tree to get branch states
                #
                def get_state(d):
                    leaf_state = d[None]
                    for subtree_key in [x for x in d.keys() if x is not None]:
                        subtree_state = get_state(d[subtree_key])
                        if leaf_state is 0:
                            leaf_state = subtree_state
                        elif leaf_state != subtree_state:
                            leaf_state = None
                    d[None] = leaf_state
                    return leaf_state
                get_state(d)
                dlg = TreeCheckboxDialog(self.module_panel, d, size=(320,480))
                choices = set(v.choices)
                dlg.Title = "Select measurements"
                if dlg.ShowModal() == wx.ID_OK:
                    def collect_state(object_name, prefix, d):
                        if d[None] is False:
                            return []
                        result = []
                        if d[None] is True and prefix is not None:
                            name = v.make_measurement_choice(object_name, prefix)
                            if name in choices:
                                result.append(name)
                        for key in [x for x in d.keys() if x is not None]:
                            if prefix is None:
                                sub_prefix = key
                            else:
                                sub_prefix = '_'.join((prefix, key))
                            result += collect_state(object_name, sub_prefix,
                                                    d[key])
                        return result
                    selections = []
                    for object_name in [x for x in d.keys() if x is not None]:
                        selections += collect_state(object_name, None, 
                                                    d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(v, self.__module, 
                                                              proposed_value, 
                                                              event)
                    self.notify(setting_edited_event)
                    self.reset_view()
            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control
    
    def make_subdirectory_filter_control(self, v, control_name, control):
        if control is None:
            control = wx.Button(self.module_panel, -1,
                                "Press to select folders")
            def on_press(event):
                assert isinstance(v, cps.SubdirectoryFilter)
                
                root = v.directory_path.get_absolute_path()
                self.module_panel.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
                try:
                    def fn_populate(root):
                        d = { None:True }
                        try:
                            for dirname in os.listdir(root):
                                dirpath = os.path.join(root, dirname)
                                dirstat = os.stat(dirpath)
                                if not stat.S_ISDIR(dirstat.st_mode):
                                    continue
                                if stat.S_ISLNK(dirstat.st_mode):
                                    continue
                                if (stat.S_IREAD & dirstat.st_mode) == 0:
                                    continue
                                d[dirname] = lambda dirpath=dirpath: fn_populate(dirpath)
                        except:
                            print "Warning: failed to list directory %s" %root
                        return d
                                
                    d = fn_populate(root)
                    selections = v.get_selections()
                    def populate_selection(d, selection, root):
                        s0 = selection[0]
                        if not d.has_key(s0):
                            d[s0] = fn_populate(os.path.join(root, s0))
                        elif hasattr(d[s0], "__call__"):
                            d[s0] = d[s0]()
                        if len(selection) == 1:
                            d[s0][None] = False
                        else:
                            if d[s0][None] is not False:
                                populate_selection(d[s0], selection[1:], 
                                                   os.path.join(root, s0))
                        if d[s0][None] is False:
                            # At best, the root is not all true
                            d[None] = None
                    
                    def split_all(x):
                        head, tail = os.path.split(x)
                        if (len(head) == 0) or (len(tail) == 0):
                            return [x]
                        else:
                            return split_all(head) + [tail]
                        
                    for selection in selections:
                        selection_parts = split_all(selection)
                        populate_selection(d, selection_parts, root)
                finally:
                    self.module_panel.SetCursor(wx.NullCursor)
                    
                dlg = TreeCheckboxDialog(self.module_panel, d, size=(320,480))
                dlg.set_parent_reflects_child(False)
                dlg.Title = "Select folders"
                if dlg.ShowModal() == wx.ID_OK:
                    def collect_state(prefix, d):
                        if d is None:
                            return []
                        if hasattr(d, "__call__") or d[None]:
                            return []
                        elif d[None] is False:
                            return [prefix]
                        result = []
                        for key in d.keys():
                            if key is None:
                                continue
                            result += collect_state(os.path.join(prefix, key),
                                                    d[key])
                        return result
                    selections = []
                    for object_name in [x for x in d.keys() if x is not None]:
                        selections += collect_state(object_name, d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(v, self.__module, 
                                                              proposed_value, 
                                                              event)
                    self.notify(setting_edited_event)
                    self.reset_view()
            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control
    
    def make_multichoice_control(self, v, control_name, control):
        selections = v.selections
        assert isinstance(v, cps.MultiChoice)
        if isinstance(v, cps.SubscriberMultiChoice):
            # Get the choices from the providers
            v.load_choices(self.__pipeline)
        choices = v.choices + [selection for selection in selections
                               if selection not in v.choices]
        if not control:
            control = wx.ListBox(self.__module_panel, -1, choices=choices,
                                 style = wx.LB_EXTENDED,
                                 name=control_name)
            for selection in selections:
                index = choices.index(selection)
                control.SetSelection(index)
                if selection not in v.choices:
                    control.SetItemForegroundColour(index, ERROR_COLOR)
            
            def callback(event, setting = v, control = control):
                self.__on_multichoice_change(event, setting, control)
            self.__module_panel.Bind(wx.EVT_LISTBOX, callback, control)
        else:
            old_choices = control.Items
            if (len(choices) != len(old_choices) or
                not all([x==y for x,y in zip(choices, old_choices)])):
                control.Items = choices
            for i in range(len(choices)):
                if control.IsSelected(i):
                    if choices[i] not in selections:
                        control.Deselect(i)
                elif choices[i] in selections:
                    control.Select(i)
                    if choices[i] not in v.choices:
                        control.SetItemForegroundColour(i, ERROR_COLOR)
        return control
    
    def make_colormap_control(self, v, control_name, control):
        """Make a combo-box that shows colormap choices
        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles 
        """
        try:
            if v.value == cps.DEFAULT:
                cmap_name = cpprefs.get_default_colormap()
            else:
                cmap_name = v.value
            cm = matplotlib.cm.get_cmap(cmap_name)
            sm = matplotlib.cm.ScalarMappable(cmap=cm)
            i,j = np.mgrid[0:12,0:128]
            if cm.N < 128:
                j = j * int((cm.N+128) / 128)
            image = (sm.to_rgba(j) * 255).astype(np.uint8)
            bitmap = wx.BitmapFromBufferRGBA(128,12,image.tostring())
        except:
            logger.warning("Failed to create the %s colorbar"%cmap_name)
            bitmap = None 
        if not control:
            control = wx.Panel(self.__module_panel,-1,
                               name = control_name)
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            colorbar = wx.StaticBitmap(control, -1,
                                       name=colorbar_ctrl_name(v))
            if not bitmap is None:
                colorbar.SetBitmap(bitmap)
            sizer.Add(colorbar,0,wx.EXPAND|wx.BOTTOM, 2)
            
            combo = wx.ComboBox(control,-1,v.value,
                                  choices=v.choices,
                                  style=wx.CB_READONLY,
                                  name=combobox_ctrl_name(v))
            sizer.Add(combo,1,wx.EXPAND)
            def callback(event, setting=v, control = combo):
                self.__on_combobox_change(event, setting,combo)
            self.__module_panel.Bind(wx.EVT_COMBOBOX,callback,combo)
        else:
            combo = control.FindWindowByName(combobox_ctrl_name(v))
            colorbar = control.FindWindowByName(colorbar_ctrl_name(v))
            old_choices = combo.Items
            if len(v.choices)!=len(old_choices) or\
               not all([x==y for x,y in zip(v.choices,old_choices)]):
                combo.Items = v.choices
            if combo.Value != v.value:
                combo.Value = v.value
            if not bitmap is None:
                colorbar.SetBitmap(bitmap)
        return control
    
    def make_color_control(self, v, control_name, control):
        if control is None:
            control = wx.Button(self.module_panel)
            
            def on_press(event, v=v, control=control):
                color = wx.Colour()
                color.SetFromName(v.value)
                data = wx.ColourData()
                data.SetColour(color)
                dlg = wx.ColourDialog(self.module_panel, data)
                dlg.Title = v.text
                if dlg.ShowModal() == wx.ID_OK:
                    proposed_value = dlg.GetColourData().GetColour().GetAsString(
                        wx.C2S_NAME | wx.C2S_HTML_SYNTAX)
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event)
                    self.notify(setting_edited_event)
                    self.reset_view()
            control.Bind(wx.EVT_BUTTON, on_press)
        control.SetBackgroundColour(v.value)
        return control
        
    def make_callback_control(self,v,control_name,control):
        """Make a control that calls back using the callback buried in the setting"""
        if not control:
            control = wx.Button(self.module_panel,-1,
                                v.label,name=control_name)
            def callback(event, setting=v):
                self.__on_do_something(event, setting)
                
            self.module_panel.Bind(wx.EVT_BUTTON, callback, control)
        return control
    
    def make_regexp_control(self, v, control):
        """Make a textbox control + regular expression button"""
        if not control:
            panel = wx.Panel(self.__module_panel,
                             -1,
                             name=edit_control_name(v))
            control = panel
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            text_ctrl = wx.TextCtrl(panel, -1, str(v.value),
                                    name=text_control_name(v))
            sizer.Add(text_ctrl,1,wx.EXPAND|wx.RIGHT,1)
            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FIND,wx.ART_TOOLBAR,(16,16))
            bitmap_button = wx.BitmapButton(panel, bitmap=bitmap,
                                            name=button_control_name(v))
            sizer.Add(bitmap_button,0,wx.EXPAND)
            def on_cell_change(event, setting = v, control=text_ctrl):
                self.__on_cell_change(event, setting,control)
                
            def on_button_pressed(event, setting = v, control = text_ctrl):
                #
                # Find a file in the image directory
                #
                file = "plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
                try:
                    if setting.get_example_fn is None:
                        path = cpprefs.get_default_image_directory()
                        filenames = [x for x in os.listdir(path)
                                     if x.find('.') != -1 and
                                     os.path.splitext(x)[1].upper() in
                                     ('.TIF','.JPG','.PNG','.BMP')]
                        if len(filenames):
                            file = filenames[0]
                    else:
                        file = setting.get_example_fn()
                except:
                    pass
                new_value = edit_regexp(panel, control.Value, file)
                if new_value:
                    control.Value = new_value
                    self.__on_cell_change(event, setting,control)
            
            def on_kill_focus(event, setting = v, control = text_ctrl):
                if self.__module is not None:
                    self.set_selection(self.__module.module_num)
            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, text_ctrl)
            self.__module_panel.Bind(wx.EVT_BUTTON, on_button_pressed, bitmap_button)
            #
            # http://www.velocityreviews.com/forums/t359823-textctrl-focus-events-in-wxwidgets.html
            # explains why bind is to control itself
            #
            text_ctrl.Bind(wx.EVT_KILL_FOCUS, on_kill_focus)
        else:
            text_control = control.FindWindowByName(text_control_name(v))
            if v.value != text_control.Value:
                text_control.Value = v.value
        return control
     
    def make_filename_text_control(self, v, control):
        """Make a filename text control"""
        edit_name = subedit_control_name(v)
        control_name = edit_control_name(v)
        button_name = button_control_name(v)
        if control is None:
            control = wx.Panel(self.module_panel, -1, 
                               name = control_name)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            control.SetSizer(sizer)
            if v.metadata_display:
                edit_control = MetadataControl(
                    self.__pipeline,
                    self.__module,
                    control,
                    value = v.value,
                    name = edit_name)
            else:
                edit_control = wx.TextCtrl(control, -1, str(v), 
                                           name = edit_name)
            sizer.Add(edit_control, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP)
            def on_cell_change(event, setting = v, control=edit_control):
                self.__on_cell_change(event, setting, control)
            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, edit_control)

            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN,
                                              wx.ART_BUTTON, (16,16))
            button_control = wx.BitmapButton(control, bitmap=bitmap,
                                             name = button_name)
            def on_press(event):
                '''Open a file browser'''
                dlg = wx.FileDialog(control, v.browse_msg)
                if v.get_directory_fn is not None:
                    dlg.Directory = v.get_directory_fn()
                if v.exts is not None:
                    dlg.Wildcard = "|".join(["|".join(tuple(x)) for x in v.exts])
                if dlg.ShowModal() == wx.ID_OK:
                    if v.set_directory_fn is not None:
                        v.set_directory_fn(dlg.Directory)
                    v.value = dlg.Filename
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, v.value, event)
                    self.notify(setting_edited_event)
                    self.reset_view()
                    
            button_control.Bind(wx.EVT_BUTTON, on_press)
            sizer.Add(button_control, 0, wx.EXPAND | wx.LEFT, 2)
        else:
            edit_control = self.module_panel.FindWindowByName(edit_name)
            button_control = self.module_panel.FindWindowByName(button_name)
            if edit_control.Value != v.value:
                edit_control.Value = v.value
            button_control.Show(v.browsable)
        return control
    
    def make_directory_path_control(self, v, control_name, control):
        assert isinstance(v, cps.DirectoryPath)
        dir_ctrl_name = combobox_ctrl_name(v)
        custom_ctrl_name = subedit_control_name(v)
        custom_ctrl_label_name = custom_label_name(v)
        browse_ctrl_name = button_control_name(v)
        if control is None:
            control = wx.Panel(self.module_panel, 
                               style = wx.TAB_TRAVERSAL,
                               name=control_name)
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            dir_ctrl = wx.Choice(control, choices = v.dir_choices, 
                                 name= dir_ctrl_name)
            sizer.Add(dir_ctrl, 0, wx.ALIGN_LEFT | wx.BOTTOM, 2)
            custom_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(custom_sizer, 1, wx.EXPAND)
            custom_label = wx.StaticText(control, name = custom_ctrl_label_name)
            custom_sizer.Add(custom_label, 0, wx.ALIGN_CENTER_VERTICAL)
            if v.allow_metadata:
                custom_ctrl = MetadataControl(self.__pipeline,
                                              self.__module,
                                              control, value = v.custom_path,
                                              name = custom_ctrl_name)
            else:
                custom_ctrl = wx.TextCtrl(control, -1, v.custom_path,
                                          name = custom_ctrl_name)
            custom_sizer.Add(custom_ctrl, 1, wx.ALIGN_CENTER_VERTICAL)
            browse_bitmap = wx.ArtProvider.GetBitmap(wx.ART_FOLDER,
                                                     wx.ART_CMN_DIALOG,
                                                     (16,16))
            browse_ctrl = wx.BitmapButton(control, bitmap=browse_bitmap,
                                          name = browse_ctrl_name)
            custom_sizer.Add(browse_ctrl, 0, wx.ALIGN_CENTER | wx.LEFT, 2)

            def on_dir_choice_change(event, v=v, dir_ctrl = dir_ctrl):
                '''Handle a change to the directory choice combobox'''
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(dir_choice = dir_ctrl.StringSelection)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event)
                self.notify(setting_edited_event)
                self.reset_view()
                
            def on_custom_path_change(event, v=v, custom_ctrl=custom_ctrl):
                '''Handle a change to the custom path'''
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(custom_path = custom_ctrl.Value)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event)
                self.notify(setting_edited_event)
                self.reset_view()
                
            def on_browse_pressed(event, v=v, dir_ctrl = dir_ctrl, 
                                  custom_ctrl=custom_ctrl):
                '''Handle browse button pressed'''
                dlg = wx.DirDialog(self.module_panel,
                                   v.text,
                                   v.get_absolute_path())
                if dlg.ShowModal() == wx.ID_OK:
                    dir_choice, custom_path = v.get_parts_from_path(dlg.Path)
                    proposed_value = v.join_string(dir_choice, custom_path)
                    if v.allow_metadata:
                        # Do escapes on backslashes
                        proposed_value = proposed_value.replace('\\','\\\\')
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event)
                    self.notify(setting_edited_event)
                    self.reset_view()
            
            dir_ctrl.Bind(wx.EVT_CHOICE, on_dir_choice_change)
            custom_ctrl.Bind(wx.EVT_TEXT, on_custom_path_change)
            browse_ctrl.Bind(wx.EVT_BUTTON, on_browse_pressed)
        else:
            dir_ctrl = self.module_panel.FindWindowByName(dir_ctrl_name)
            custom_ctrl = self.module_panel.FindWindowByName(custom_ctrl_name)
            custom_label = self.module_panel.FindWindowByName(custom_ctrl_label_name)
            browse_ctrl = self.module_panel.FindWindowByName(browse_ctrl_name)
        if dir_ctrl.StringSelection != v.dir_choice:
            dir_ctrl.StringSelection = v.dir_choice
        if v.is_custom_choice:
            if not custom_ctrl.IsShown():
                custom_ctrl.Show()
            if not custom_label.IsShown():
                custom_label.Show()
            if not browse_ctrl.IsShown():
                browse_ctrl.Show()
            if v.dir_choice in (cps.DEFAULT_INPUT_SUBFOLDER_NAME,
                                cps.DEFAULT_OUTPUT_SUBFOLDER_NAME):
                custom_label.Label = "Sub-folder:"
            elif v.dir_choice == cps.URL_FOLDER_NAME:
                custom_label.Hide()
                custom_ctrl.Hide()
                browse_ctrl.Hide()
            if custom_ctrl.Value != v.custom_path:
                custom_ctrl.Value = v.custom_path
        else:
            custom_label.Hide()
            custom_ctrl.Hide()
            browse_ctrl.Hide()
        return control
    
    def make_text_control(self, v, control_name, control):
        """Make a textbox control"""
        if not control:
            if v.metadata_display:
                control = MetadataControl(
                    self.__pipeline,
                    self.__module,
                    self.__module_panel,
                    value = v.value,
                    name = control_name
                )
            else:
                style = 0
                if getattr(v, "multiline_display", False):
                    style = wx.TE_MULTILINE|wx.TE_PROCESS_ENTER
    
                text = v.value
                if not isinstance(text, (unicode, str)):
                    text = str(text)
                control = wx.TextCtrl(self.__module_panel,
                                      -1,
                                      text,
                                      name=control_name,
                                      style = style)
            def on_cell_change(event, setting = v, control=control):
                self.__on_cell_change(event, setting,control)
            self.__module_panel.Bind(wx.EVT_TEXT,on_cell_change,control)
        elif not (v == control.Value):
            text = v.value
            if not isinstance(text, (unicode, str)):
                text = str(text)
            control.Value = text
        return control
    
    def make_range_control(self, v, panel):
        """Make a "control" composed of a panel and two edit boxes representing a range"""
        if not panel:
            panel = wx.Panel(self.__module_panel,-1,name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            min_ctrl = wx.TextCtrl(panel,-1,str(v.min),
                                   name=min_control_name(v))
            sizer.Add(min_ctrl,0,wx.EXPAND|wx.RIGHT,1)
            max_ctrl = wx.TextCtrl(panel,-1,str(v.max),
                                   name=max_control_name(v))
            #max_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(max_ctrl,0,wx.EXPAND)
            def on_min_change(event, setting = v, control=min_ctrl):
                self.__on_min_change(event, setting,control)
            self.__module_panel.Bind(wx.EVT_TEXT,on_min_change,min_ctrl)
            def on_max_change(event, setting = v, control=max_ctrl):
                self.__on_max_change(event, setting,control)
            self.__module_panel.Bind(wx.EVT_TEXT,on_max_change,max_ctrl)
        else:
            min_ctrl = panel.FindWindowByName(min_control_name(v))
            if min_ctrl.Value != str(v.min):
                min_ctrl.Value = str(v.min)
            max_ctrl = panel.FindWindowByName(max_control_name(v))
            if max_ctrl.Value != str(v.max):
                max_ctrl.Value = str(v.max)
        
        for ctrl in (min_ctrl, max_ctrl):
            self.fit_ctrl(ctrl)
        return panel
    
    def make_unbounded_range_control(self, v, panel):
        """Make a "control" composed of a panel and two combo-boxes representing a range
        
        v - an IntegerOrUnboundedRange setting
        panel - put it in this panel
        
        The combo box has the word to use to indicate that the range is unbounded
        and the text portion is the value
        """
        if not panel:
            panel = wx.Panel(self.__module_panel,-1,name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            min_ctrl = wx.TextCtrl(panel,-1,value=str(v.min),
                                   name=min_control_name(v))
            best_width = min_ctrl.GetCharWidth()*5
            min_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(min_ctrl,0,wx.EXPAND|wx.RIGHT,1)
            max_ctrl = wx.TextCtrl(panel,-1,value=v.display_max,
                                   name=max_control_name(v))
            max_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(max_ctrl,0,wx.EXPAND)
            if v.unbounded_max or v.max < 0:
                value = FROM_EDGE
            else:
                value = ABSOLUTE 
            absrel_ctrl = wx.ComboBox(panel,-1,value,
                                      choices = [ABSOLUTE,FROM_EDGE],
                                      name = absrel_control_name(v),
                                      style = wx.CB_DROPDOWN|wx.CB_READONLY)
            sizer.Add(absrel_ctrl,0,wx.EXPAND|wx.RIGHT,1)
            def on_min_change(event, setting = v, control=min_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                if setting.unbounded_max:
                    max_value = cps.END
                else:
                    max_value = str(setting.max)
                proposed_value="%s,%s"%(str(control.Value),max_value)
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value,event)
                self.notify(setting_edited_event)
                self.fit_ctrl(control)
                
            self.__module_panel.Bind(wx.EVT_TEXT,on_min_change,min_ctrl)
            def on_max_change(event, setting = v, control=max_ctrl, 
                              absrel_ctrl=absrel_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                if (absrel_ctrl.Value == ABSOLUTE):
                    max_value = str(control.Value)
                elif control.Value == '0':
                    max_value = cps.END
                else:
                    max_value = "-"+str(control.Value)
                proposed_value="%s,%s"%(setting.display_min,max_value)
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value,event)
                self.notify(setting_edited_event)
                self.fit_ctrl(control)
                
            self.__module_panel.Bind(wx.EVT_TEXT,on_max_change,max_ctrl)
            def on_absrel_change(event, setting = v, control=absrel_ctrl):
                if not self.__handle_change:
                    return
                
                if not v.unbounded_max:
                    old_value = str(setting)
                    
                    if control.Value == ABSOLUTE:
                        proposed_value="%s,%s"%(setting.display_min,
                                                abs(setting.max))
                    else:
                        setting_max = setting.max
                        if setting_max is not None:
                            proposed_value="%s,%d"%(setting.display_min,
                                                    -abs(setting.max))
                        else:
                            proposed_value = None
                else:
                    proposed_value="%s,%s"%(setting.display_min,
                                            cps.END)
                if proposed_value is not None:
                    setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                              proposed_value,event)
                    self.notify(setting_edited_event)
            self.__module_panel.Bind(wx.EVT_COMBOBOX,
                                     on_absrel_change,absrel_ctrl)
        else:
            min_ctrl = panel.FindWindowByName(min_control_name(v))
            if min_ctrl.Value != v.display_min:
                min_ctrl.Value = v.display_min
            max_ctrl = panel.FindWindowByName(max_control_name(v))
            if max_ctrl.Value != v.display_max:
                min_ctrl.Value = v.display_max
            absrel_ctrl = panel.FindWindowByName(absrel_control_name(v))
            absrel_value = ABSOLUTE
            if v.unbounded_max or v.max < 0:
                absrel_value = FROM_EDGE
            if absrel_ctrl.Value != absrel_value:
                absrel_ctrl.Value = absrel_value
            
        return panel
    
    def make_coordinates_control(self, v, panel):
        """Make a "control" composed of a panel and two edit boxes representing X and Y"""
        if not panel:
            panel = wx.Panel(self.__module_panel,-1,name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            sizer.Add(wx.StaticText(panel,-1,"X:"),0,wx.EXPAND|wx.RIGHT,1)
            x_ctrl = wx.TextCtrl(panel,-1,str(v.x),
                                   name=x_control_name(v))
            best_width = x_ctrl.GetCharWidth()*5
            x_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(x_ctrl,0,wx.EXPAND|wx.RIGHT,1)
            sizer.Add(wx.StaticText(panel,-1,"Y:"),0,wx.EXPAND|wx.RIGHT,1)
            y_ctrl = wx.TextCtrl(panel,-1,str(v.y),
                                 name=y_control_name(v))
            y_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(y_ctrl,0,wx.EXPAND)
            def on_x_change(event, setting = v, control=x_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value="%s,%s"%(str(control.Value),str(setting.y))
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value,event)
                self.notify(setting_edited_event)
            self.__module_panel.Bind(wx.EVT_TEXT,on_x_change,x_ctrl)
            def on_y_change(event, setting = v, control=y_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value="%s,%s"%(str(setting.x),str(control.Value))
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value,event)
                self.notify(setting_edited_event)
            self.__module_panel.Bind(wx.EVT_TEXT,on_y_change,y_ctrl)
        else:
            x_ctrl = panel.FindWindowByName(x_control_name(v))
            if x_ctrl.Value != str(v.x):
                x_ctrl.Value = str(v.x)
            y_ctrl = panel.FindWindowByName(y_control_name(v))
            if y_ctrl.Value != str(v.y):
                y_ctrl.Value = str(v.y)
            
        return panel
    
    def make_measurement_control(self, v, panel):
        '''Make a composite measurement control
        
        The measurement control has the following parts:
        Category - a measurement category like AreaShape or Intensity
        Feature name - the feature being measured or algorithm applied
        Image name - an optional image that was used to compute the measurement
        Object name - an optional set of objects used to compute the measurement
        Scale - an optional scale, generally in pixels, that controls the size
                of the measured features.
        '''
        #
        # We either come in here with:
        # * panel = None - create the controls
        # * panel != None - find the controls
        #
        if not panel:
            panel = wx.Panel(self.__module_panel,-1,name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.VERTICAL)
            panel.SetSizer(sizer)
            #
            # The category combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer,0,wx.ALIGN_LEFT)
            category_text_ctrl = wx.StaticText(panel,label='Category:',
                                               name = category_text_control_name(v))
            sub_sizer.Add(category_text_ctrl,0, wx.EXPAND|wx.ALL,2)
            category_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                        name=category_control_name(v))
            sub_sizer.Add(category_ctrl, 0, wx.EXPAND|wx.ALL, 2)
            #
            # The measurement / feature combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0 , wx.ALIGN_LEFT)
            feature_text_ctrl = wx.StaticText(panel, label='Measurement:',
                                              name= feature_text_control_name(v))
            sub_sizer.Add(feature_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            feature_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                       name=feature_control_name(v))
            sub_sizer.Add(feature_ctrl, 0, wx.EXPAND|wx.ALL, 2)
            #
            # The object combo-box which sometimes doubles as an image combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            object_text_ctrl = wx.StaticText(panel, label='Object:',
                                            name = object_text_control_name(v))
            sub_sizer.Add(object_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            object_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                      name=object_control_name(v))
            sub_sizer.Add(object_ctrl, 0, wx.EXPAND|wx.ALL, 2)
            #
            # The scale combo-box
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            scale_text_ctrl = wx.StaticText(panel, label='Scale:',
                                            name = scale_text_ctrl_name(v))
            sub_sizer.Add(scale_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            scale_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                     name=scale_control_name(v))
            sub_sizer.Add(scale_ctrl, 0, wx.EXPAND|wx.ALL, 2)
            max_width = 0
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[0].Window
                max_width = max(max_width,static.Size.width)
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[0].Window
                static.Size = wx.Size(max_width,static.Size.height)
                static.SetSizeHints(max_width, -1, max_width)
            #
            # Bind all controls to the function that constructs a value
            # out of the parts
            #
            def on_change(event, v=v, category_ctrl = category_ctrl,
                          feature_ctrl = feature_ctrl,
                          object_ctrl = object_ctrl,
                          scale_ctrl = scale_ctrl):
                '''Reconstruct the measurement value if anything changes'''
                if not self.__handle_change:
                    return
                
                def value_of(ctrl):
                    return ctrl.Value if ctrl.Selection != -1 else None
                value = v.construct_value(value_of(category_ctrl),
                                          value_of(feature_ctrl),
                                          value_of(object_ctrl),
                                          value_of(scale_ctrl))
                setting_edited_event = SettingEditedEvent(v,
                                                          self.__module,
                                                          value,
                                                          event)
                self.notify(setting_edited_event)
                self.reset_view()
            
            for ctrl in (category_ctrl, feature_ctrl, object_ctrl, scale_ctrl):
                panel.Bind(wx.EVT_COMBOBOX, on_change, ctrl)
        else:
            #
            # Find the controls from inside the panel
            #
            category_ctrl = panel.FindWindowByName(category_control_name(v))
            category_text_ctrl = panel.FindWindowByName(category_text_control_name(v))
            feature_ctrl = panel.FindWindowByName(feature_control_name(v))
            feature_text_ctrl = panel.FindWindowByName(feature_text_control_name(v))
            object_ctrl = panel.FindWindowByName(object_control_name(v))
            object_text_ctrl = panel.FindWindowByName(object_text_control_name(v))
            scale_ctrl = panel.FindWindowByName(scale_control_name(v))
            scale_text_ctrl = panel.FindWindowByName(scale_text_ctrl_name(v))
        category = v.get_category(self.__pipeline)
        categories = v.get_category_choices(self.__pipeline)
        feature_name = v.get_feature_name(self.__pipeline)
        feature_names = v.get_feature_name_choices(self.__pipeline)
        image_name = v.get_image_name(self.__pipeline)
        image_names = v.get_image_name_choices(self.__pipeline)
        object_name = v.get_object_name(self.__pipeline)
        object_names = v.get_object_name_choices(self.__pipeline)
        scale = v.get_scale(self.__pipeline)
        scales = v.get_scale_choices(self.__pipeline)
        def set_up_combobox(ctrl, text_ctrl, choices, value, always_show=False):
            if len(choices):
                if not (len(ctrl.Strings) == len(choices) and
                        all([x==y for x,y in zip(ctrl.Strings,choices)])):
                    ctrl.Clear()
                    ctrl.AppendItems(choices)
                if (not value is None):
                    try:
                        if ctrl.Value != value:
                            ctrl.Value = value
                    except:
                        # Crashes on the Mac sometimes
                        ctrl.Value = value
                ctrl.Show()
                text_ctrl.Show()
            elif always_show:
                ctrl.Clear()
                ctrl.Value = "No measurements available"
            else:
                ctrl.Hide()
                ctrl.Clear()
                text_ctrl.Hide()
        set_up_combobox(category_ctrl, category_text_ctrl, categories, 
                        category, True)
        set_up_combobox(feature_ctrl, feature_text_ctrl, 
                        feature_names, feature_name)
        #
        # The object combo-box might have image choices
        #
        if len(object_names) > 0:
            if len(image_names) > 0:
                object_text_ctrl.Label = "Image or Objects:"
                object_names += image_names
            else:
                object_text_ctrl.Label = "Objects:"
        else:
            object_text_ctrl.Label = "Image:"
            object_names = image_names
        if object_name is None:
            object_name = image_name
        set_up_combobox(object_ctrl, object_text_ctrl, object_names, object_name)
        set_up_combobox(scale_ctrl, scale_text_ctrl, scales, scale)
        return panel
    
    def make_help_control(self, content, title="Help", 
                          name = wx.ButtonNameStr):
        control = wx.Button(self.__module_panel, -1, '?', (0, 0), (30, -1), 
                            name = name)
        def callback(event):
            dialog = HTMLDialog(self.__module_panel, title, content)
            dialog.CentreOnParent()
            dialog.Show()
        control.Bind(wx.EVT_BUTTON, callback, control)
        return control

    def add_listener(self,listener):
        self.__listeners.append(listener)
    
    def remove_listener(self,listener):
        self.__listeners.remove(listener)
    
    def notify(self,event):
        self.__inside_notify = True
        try:
            for listener in self.__listeners:
                listener(self,event)
        finally:
            self.__inside_notify = False
            
    def __on_column_sized(self,event):
        self.__module_panel.GetTopLevelParent().Layout()
    
    def __on_checkbox_change(self,event,setting,control):
        if not self.__handle_change:
            return
        self.__on_cell_change(event, setting, control)
        self.reset_view()
    
    def __on_combobox_change(self,event,setting,control):
        if not self.__handle_change:
            return
        self.__on_cell_change(event, setting, control)
        self.reset_view()
    
    def __on_multichoice_change(self, event, setting, control):
        if not self.__handle_change:
            return
        
        old_value = str(setting)
        proposed_value = str(','.join([control.Items[i]
                                       for i in control.Selections]))
        setting_edited_event = SettingEditedEvent(setting, self.__module, 
                                                  proposed_value, 
                                                  event)
        self.notify(setting_edited_event)
        self.reset_view()
        
    def __on_cell_change(self,event,setting,control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        if isinstance(control,wx.CheckBox):
            proposed_value = (control.GetValue() and 'Yes') or 'No'
        else:
            proposed_value = str(control.GetValue())
        setting_edited_event = SettingEditedEvent(setting,
                                                  self.__module, 
                                                  proposed_value,
                                                  event)
        self.notify(setting_edited_event)
        if setting.reset_view:
            self.reset_view()
    
    def fit_ctrl(self, ctrl):
        '''Fit the control to its text size'''
        width , height = ctrl.GetTextExtent(ctrl.Value + "MM")
        ctrl.SetSizeHintsSz(wx.Size(width, -1))
        ctrl.Parent.Fit()
        
    def __on_min_change(self,event,setting,control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value="%s,%s"%(str(control.Value),str(setting.max))
        setting_edited_event = SettingEditedEvent(setting,self.__module, 
                                                  proposed_value,event)
        self.notify(setting_edited_event)
        self.fit_ctrl(control)
        
    def __on_max_change(self,event,setting,control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value="%s,%s"%(str(setting.min),str(control.Value))
        setting_edited_event = SettingEditedEvent(setting,self.__module, 
                                                  proposed_value,event)
        self.notify(setting_edited_event)
        self.fit_ctrl(control)
        
    def __on_pipeline_event(self,pipeline,event):
        if (isinstance(event,cpp.PipelineClearedEvent)):
            self.clear_selection()
        elif (isinstance(event, cpp.PipelineLoadedEvent)):
            if len(self.__pipeline.modules()) == 0:
                self.clear_selection()
        elif isinstance(event, cpp.ModuleEditedPipelineEvent):
            if (not self.__inside_notify and self.__module is not None
                and self.__module.module_num == event.module_num):
                self.reset_view()
        elif isinstance(event, cpp.ModuleRemovedPipelineEvent):
            if (self.__module is not None and 
                event.module_num == self.__module.module_num):
                self.clear_selection()
    
    def __on_do_something(self, event, setting):
        setting.on_event_fired()
        setting_edited_event = SettingEditedEvent(setting,self.__module, 
                                                  None,event)
        self.notify(setting_edited_event)
        self.reset_view()
    

    def on_size(self, evt):
        if self.__startup_blurb:
            self.__startup_blurb.Size = self.__module_panel.ClientSize

    def on_idle(self,event):
        """Check to see if the selected module is valid"""
        last_idle_time = getattr(self, "last_idle_time", 0)
        running_time = getattr(self, "running_time", 0)
        timeout = max(CHECK_TIMEOUT_SEC, running_time * 4)
        if time.time() - last_idle_time > timeout:
            self.last_idle_time = time.time()
        else:
            return
        if self.__module:
            try:
                self.validate_module()
            finally:
                self.running_time = time.time() - self.last_idle_time
            
    def validate_module(self):
        validation_error = None
        signal_error = True
        signal_warning = False
        default_fg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        default_bg_color = cpprefs.get_background_color()
        try:
            self.__module.test_valid(self.__pipeline)
            signal_error = False
            signal_warning = True
            self.__module.validate_module_warnings(self.__pipeline)
            signal_warning = False
        except cps.ValidationError, instance:
            validation_error = instance
        try:
            for idx, setting in enumerate(self.__module.visible_settings()):
                static_text_name = text_control_name(setting)
                error_message = None
                if (validation_error is not None and 
                    validation_error.setting.key() == setting.key()):
                    error_message = validation_error.message
                else:
                    try:
                        setting.test_valid(self.__pipeline)
                    except cps.ValidationError, instance:
                        error_message = instance.message
                
                if error_message is not None:    
                    # always update the tooltip, in case the value changes to something that's still bad.
                    control_name = edit_control_name(setting)
                    control = self.__module_panel.FindWindowByName(
                        control_name)
                    if control is not None:
                        control.SetToolTipString(error_message)
                        for child in control.GetChildren():
                            child.SetToolTipString(error_message)
                    static_text = self.__module_panel.FindWindowByName(static_text_name)
                    if static_text is not None:
                        static_text.SetToolTipString(error_message)
                        if signal_error:
                            if static_text.GetForegroundColour() != ERROR_COLOR:
                                static_text.SetForegroundColour(ERROR_COLOR)
                                static_text.SetBackgroundColour(default_bg_color)
                                static_text.Refresh()
                        elif signal_warning:
                            if static_text.GetBackgroundColour() != WARNING_COLOR:
                                static_text.SetForegroundColour(default_fg_color)
                                static_text.SetBackgroundColour(WARNING_COLOR)
                                static_text.Refresh()
                    continue
                static_text_name = text_control_name(setting)
                static_text = self.__module_panel.FindWindowByName(
                    static_text_name)
                if (static_text is not None and
                    ((static_text.GetForegroundColour() == ERROR_COLOR) or
                     (static_text.GetBackgroundColour() == WARNING_COLOR))):
                    control_name = edit_control_name(setting)
                    control = self.__module_panel.FindWindowByName(
                        control_name)
                    if control is not None:
                        control.SetToolTipString('OK')
                        for child in control.GetChildren():
                            child.SetToolTipString('OK')
                    static_text.SetForegroundColour(default_fg_color)
                    static_text.SetBackgroundColour(default_bg_color)
                    static_text.SetToolTip(None)
                    static_text.Refresh()
        except:
            pass

    def reset_view(self):
        """Redo all of the controls after something has changed
        
        TO_DO: optimize this so that only things that have changed IRL change in the GUI
        """
        if self.__module is None:
            return
        focus_control = wx.Window.FindFocus()
        if not focus_control is None:
            focus_name = focus_control.GetName()
        else:
            focus_name = None
        self.validate_module()
        self.set_selection(self.__module.module_num)
        if focus_name:
            focus_control = self.module_panel.FindWindowByName(focus_name)
            if focus_control:
                focus_control.SetFocus()

    def disable(self):
        self.__module_panel.Disable()

    def enable(self):
        self.__module_panel.Enable()
        
    def get_max_width(self):
        sizer = self.__sizer
        return sizer.calc_max_text_width() + sizer.calc_edit_size()[0] + sizer.calc_help_size()[0]

class ModuleSizer(wx.PySizer):
    """The module sizer uses the maximum best width of the setting
    edit controls to compute the column widths, then it sets the text
    controls to wrap within the remaining space, then it uses the best
    height of each text control to lay out the rows.
    """
    
    def __init__(self,rows,cols=2):
        wx.PySizer.__init__(self)
        self.__rows = rows
        self.__cols = cols
        self.__min_text_width = 150
        self.__printed_exception = False
    
    def Reset(self, rows, cols=3, destroy_windows=True):
        if destroy_windows:
            windows = []
            for j in range(self.__rows):
                for i in range(self.__cols):
                    item = self.GetItem(self.idx(i,j))
                    if item is None:
                        print "Missing item"
                    if item.IsWindow():
                        window = item.GetWindow()
                        if isinstance(window, wx.Window):
                            windows.append(window)
            for window in windows:
                window.Hide()
                window.Destroy()
        self.Clear(False)    
        self.__rows = rows
        self.__cols = cols
    
    def CalcMin(self):
        """Calculate the minimum from the edit controls.  Returns a
        wx.Size where the height is the total height of the grid and
        the width is self.__min_text_width plus the widths of the edit
        controls and help controls.
        """
        try:
            if (self.__rows * self.__cols == 0 or 
                self.Children is None or
                len(self.Children) == 0):
                return wx.Size(0,0)
            height = 0
            for j in range(0,self.__rows):
                height_border = max([self.GetItem(col,j).GetBorder() 
                                     for col in range(2)])
                height += self.get_row_height(j) + 2*height_border
            self.__printed_exception = False
            return wx.Size(self.calc_edit_size()[0] + self.__min_text_width + 
                           self.calc_help_size()[0],
                           height)
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                logger.error("WX internal error detected", exc_info=True)
                self.__printed_exception = True
                return wx.Size(0,0)

    def get_row_height(self, j):
        height = 0
        for i in range(self.__cols):
            item = self.GetItem(self.idx(i,j))
            if item is None:
                continue
            control = item.GetWindow()
            if (isinstance(control, wx.StaticLine)):
                height = max(height, item.CalcMin()[1] * 1.25)
            else:
                height = max(height, item.CalcMin()[1])
        return height
        
    def calc_column_size(self, j):
        """Return a wx.Size with the total height of the controls in
        column j and the maximum of their widths.
        """
        height = 0
        width = 0
        for i in range(self.__rows):
            if len(self.Children) <= self.idx(j, i):
                break
            item = self.GetItem(self.idx(j, i))
            size = item.CalcMin()
            height += size[1]
            width = max(width, size[0])
        return wx.Size(width, height)
        
    def calc_help_size(self):
        return self.calc_column_size(2)

    def calc_edit_size(self):
        return self.calc_column_size(1)
    
    def calc_max_text_width(self):
        width = self.__min_text_width
        for i in range(self.__rows):
            if len(self.Children) <= self.idx(0, i):
                break
            item = self.GetItem(self.idx(0, i))
            control = item.GetWindow()
            assert isinstance(control, wx.StaticText), 'Control at column 0, '\
                '%d of grid is not StaticText: %s'%(i, str(control))
            text = control.GetLabel().replace('\n', ' ')
            ctrl_width = control.GetTextExtent(text)[0] + 2 * item.GetBorder()
            width = max(width, ctrl_width)
        return width

    
    def RecalcSizes(self):
        """Recalculate the sizes of our items, resizing the text boxes
        as we go.
        """
        if self.__rows * self.__cols == 0:
            return
        try:
            size = self.GetSize()
            width = size[0] - 20
            edit_width = self.calc_edit_size()[0]
            help_width = self.calc_help_size()[0]
            max_text_width = self.calc_max_text_width()
            if edit_width + help_width + max_text_width < width:
                edit_width = width - max_text_width - help_width
            elif edit_width * 4 < width:
                edit_width = width / 4
            text_width = max([width - edit_width - help_width, 
                              self.__min_text_width])
            widths = [text_width, edit_width, help_width]
            #
            # Change all static text controls to wrap at the text width. Then
            # ask the items how high they are and do the layout of the line.
            #
            height = 0
            panel = self.GetContainingWindow()
            for i in range(self.__rows):
                text_item = self.GetItem(self.idx(0, i))
                edit_item = self.GetItem(self.idx(1, i))
                inner_text_width = text_width - 2 * text_item.GetBorder() 
                control = text_item.GetWindow()
                assert isinstance(control, wx.StaticText), 'Control at column 0, %d of grid is not StaticText: %s'%(i,str(control))
                text = control.GetLabel()
                edit_control = edit_item.GetWindow()
                height_border = max([x.GetBorder() for x in (edit_item, text_item)])
                if (isinstance(edit_control, wx.StaticLine) and
                    len(text) == 0):
                    #
                    # A line spans both columns
                    #
                    text_item.Show(False)
                    # make the divider height the same as a text row plus some
                    item_height = self.get_row_height(i)
                    assert isinstance(edit_item, wx.SizerItem)
                    border = edit_item.GetBorder()
                    third_width = (text_width + edit_width - 2*border) / 3
                    item_location = wx.Point(text_width - third_width / 2, 
                                             height + border + item_height / 2)
                    item_size = wx.Size(third_width, edit_item.Size[1])
                    #item_location = panel.CalcScrolledPosition(item_location)
                    edit_item.SetDimension(item_location, item_size)
                else:
                    text_item.Show(True)
                    if (text_width > self.__min_text_width and
                        (text.find('\n') != -1 or
                         control.GetTextExtent(text)[0] > inner_text_width)):
                        text = text.replace('\n',' ')
                        control.SetLabel(text)
                        control.Wrap(inner_text_width)
                    for j in range(self.__cols):
                        item = self.GetItem(self.idx(j, i))
                        if (item.Flag & wx.EXPAND) == 0:
                            item_size = item.CalcMin()
                        else:
                            item_size = wx.Size(widths[j], item.CalcMin()[1])
                        item_location = wx.Point(sum(widths[0:j]), height)
                        #item_location = panel.CalcScrolledPosition(item_location)
                        item.SetDimension(item_location, item_size)
                height += self.get_row_height(i) + 2*height_border
            panel.SetVirtualSizeWH(width, height+20)
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                logger.warning("Detected WX error", exc_info=True)
                self.__printed_exception = True

    def coords(self,idx):
        """Return the column/row coordinates of an indexed item
        
        """
        (col,row) = divmod(idx,self.__cols)
        return (col,row)

    def idx(self,col,row):
        """Return the index of the given grid cell
        
        """
        return row*self.__cols + col
