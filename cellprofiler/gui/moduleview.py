"""ModuleView.py - implements a view on a module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import logging
import matplotlib.cm
import numpy as np
import os
import stat
import threading
import heapq
import time
import wx
import wx.grid

logger = logging.getLogger(__name__)
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.html import HtmlClickableWindow
from regexp_editor import edit_regexp
from htmldialog import HTMLDialog
from treecheckboxdialog import TreeCheckboxDialog
from metadatactrl import MetadataControl
from namesubscriber import NameSubcriberComboBox

ERROR_COLOR = wx.RED
WARNING_COLOR = wx.Colour(224,224,0,255)
RANGE_TEXT_WIDTH = 40 # number of pixels in a range text box TO_DO - calculate it
ABSOLUTE = "Absolute"
FROM_EDGE = "From edge"

CHECK_TIMEOUT_SEC = 2
EDIT_TIMEOUT_SEC = 5

# validation queue priorities, to allow faster updates for the displayed module
PRI_VALIDATE_DISPLAY = 0
PRI_VALIDATE_BACKGROUND = 1

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
        self.refresh_pending = False
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
        
    def check_settings(self, module_name, settings):
        try:
            assert len(settings) > 0
        except:
            wx.MessageBox("Module %s.visible_settings() did not return a list!\n  value: %s"%(module_name, settings),
                          "Pipeline Error", wx.ICON_ERROR, self.__module_panel)
            settings = []
        try:
            assert all([isinstance(s, cps.Setting) for s in settings])
        except:
            wx.MessageBox("Module %s.visible_settings() returned something other than a list of Settings!\n  value: %s"%(module_name, settings),
                          "Pipeline Error", wx.ICON_ERROR, self.__module_panel)
            settings = []
        return settings


    def set_selection(self, module_num):
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
                try:
                    # Need to initialize some controls.
                    new_module.test_valid(self.__pipeline)
                except:
                    pass
            if not self.__as_datatool:
                self.notes_panel.Show()
            self.__module       = new_module
            self.__controls     = []
            self.__static_texts = []
            data                = []
            settings            = self.check_settings(self.__module.module_name, self.__module.visible_settings())
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
                    control = self.make_name_subscriber_control(v, choices,
                                                                control_name,
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
                elif isinstance(v, cps.TreeChoice):
                    control = self.make_tree_choice_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cps.Filter):
                    if control is not None:
                        control.filter_panel_controller.update()
                    else:
                        fc = FilterPanelController(self, v, control)
                        control = fc.panel
                        control.filter_panel_controller = fc
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

    def make_name_subscriber_control(self, v, choices, control_name, control):
        """Make a read-only combobox with extra feedback about source modules,
        and a context menu with choices navigable by module name.

        v            - the setting
        choices      - a list of (name, module_name, module_number)
        control_name - assign this name to the control
        """
        if v.value not in [c[0] for c in choices]:
            choices = choices + [(v.value, "", 0)]
        if not control:
            control = NameSubcriberComboBox(self.__module_panel,
                                            value=v.value,
                                            choices=choices,
                                            name=control_name)
            def callback(event, setting=v, control=control):
                # the NameSubcriberComboBox behaves like a combobox
                self.__on_combobox_change(event, setting, control)
            control.add_callback(callback)
        else:
            if list(choices) != list(control.Items):
                control.Items = choices
        if (getattr(v, 'has_tooltips', False) and
            v.has_tooltips and (control.Value in v.tooltips)):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control


    def make_choice_control(self,v,choices,control_name,style,control):
        """Make a combo-box that shows choices
        
        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles 
        """
        assert isinstance(v, cps.Choice)
        try:
            v.test_valid(self.__pipeline)
        except:
            pass
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
                if len(v.choices) == 0:
                    v.populate_choices(self.__pipeline)
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
        
    def make_tree_choice_control(self, v, control_name, control):
        if control is None:
            control = wx.Button(self.module_panel)
            def on_press(event, v=v, control=control):
                id_dict = {}
                def make_menu(tree, id_dict = id_dict, path = []):
                    menu = wx.Menu()
                    for node in tree:
                        text, subtree = node[:2]
                        subpath = path + [text]
                        if subtree is None:
                            item = menu.Append(-1, text)
                            id_dict[item.GetId()] = subpath
                        else:
                            submenu = make_menu(subtree, path = subpath)
                            menu.AppendMenu(-1, text, submenu)
                    return menu
                
                menu = make_menu(v.get_tree())
                assert isinstance(control, wx.Window)
                def on_event(event, v = v, control = control, id_dict = id_dict):
                    new_path = id_dict[event.GetId()]
                    self.on_value_change(v, control, new_path, event)
                    
                menu.Bind(wx.EVT_MENU, on_event)
                control.PopupMenuXY(menu, 0, control.GetSize()[1])
                menu.Destroy()
            control.Bind(wx.EVT_BUTTON, on_press)
        control.SetLabel(">".join(v.get_value()))
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
                self.reset_view(1000)
                
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
                if v.support_urls == cps.SUPPORT_URLS_SHOW_DIR:
                    custom_label.Label = "URL:"
                    custom_label.Show()
                    custom_ctrl.Show()
                else:
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
        proposed_value = (control.GetValue() and 'Yes') or 'No'
        self.on_value_change(setting, control, proposed_value, event)
    
    def __on_combobox_change(self,event,setting,control):
        if not self.__handle_change:
            return
        self.on_value_change(setting, control, control.GetValue(), event)
    
    def __on_multichoice_change(self, event, setting, control):
        if not self.__handle_change:
            return
        
        proposed_value = u','.join([control.Items[i]
                                    for i in control.Selections])
        self.on_value_change(setting, control, proposed_value, event)
        
    def __on_cell_change(self,event,setting,control):
        if not self.__handle_change:
            return
        proposed_value = unicode(control.GetValue())
        self.on_value_change(setting, control, proposed_value, event,
                             EDIT_TIMEOUT_SEC * 1000)
        
    def on_value_change(self, setting, control, proposed_value, event, 
                        timeout = None):
        setting_edited_event = SettingEditedEvent(setting,
                                                  self.__module, 
                                                  proposed_value,
                                                  event)
        self.notify(setting_edited_event)
        if timeout is None:
            self.reset_view() # use the default timeout
        else:
            self.reset_view(timeout)
    
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
            # clear validation cache, since settings might not have changed,
            # but pipeline itself may have (due to a module source reload)
            clear_validation_cache()
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
            request_module_validation(self.__pipeline, self.__module,
                                      self.on_validation, PRI_VALIDATE_DISPLAY)

    def on_validation(self, setting_idx, message, level):
        self.running_time = time.time() - self.last_idle_time
        default_fg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        default_bg_color = cpprefs.get_background_color()

        if not self.__module:  # defensive coding, in case the module was deleted
            return

        visible_settings = self.__module.visible_settings()
        bad_setting = None
        if setting_idx is not None:
            # an error was detected by the validation thread.  The pipeline may
            # have changed in the meantime, so we revalidate here to make sure
            # what we display is up to date.
            if setting_idx >= len(visible_settings):
                return  # obviously changed, don't update display
            try:
                # fast-path: check the reported setting first
                level = logging.ERROR
                visible_settings[setting_idx].test_valid(self.__pipeline)
                self.__module.test_valid(self.__pipeline)
                level = logging.WARNING
                self.__module.validate_module_warnings(self.__pipeline)
            except cps.ValidationError, instance:
                message = instance.message
                bad_setting = instance.get_setting()

        # update settings' foreground/background
        try:
            for setting in visible_settings:
                self.set_tool_tip(setting, message if (setting is bad_setting) else None)
                static_text_name = text_control_name(setting)
                static_text = self.__module_panel.FindWindowByName(static_text_name)
                if static_text is not None:
                    desired_fg, desired_bg = default_fg_color, default_bg_color
                    if setting is bad_setting:
                        if level == logging.ERROR:
                            desired_fg = ERROR_COLOR
                        elif level == logging.WARNING:
                            desired_bg = WARNING_COLOR
                if (static_text.SetForegroundColour(desired_fg) or
                    static_text.SetBackgroundColour(desired_bg)):
                    static_text.Refresh()
        except Exception:
            logger.debug("Caught bare exception in ModuleView.on_validate()", exc_info=True)
            pass

    def set_tool_tip(self, setting, message):
        '''Set the tool tip for a setting to display a message
        
        setting - set the tooltip for this setting
        
        message - message to display or None for no tool tip
        '''
        control_name = edit_control_name(setting)
        control = self.__module_panel.FindWindowByName(
            control_name)
        if message is None:
            def set_tool_tip(ctrl):
                ctrl.SetToolTip(None)
        else:
            def set_tool_tip(ctrl, message = message):
                ctrl.SetToolTipString(message)
        if control is not None:
            set_tool_tip(control)
            for child in control.GetChildren():
                set_tool_tip(child)
        static_text_name = text_control_name(setting)
        static_text = self.__module_panel.FindWindowByName(static_text_name)
        if static_text is not None:
            set_tool_tip(static_text)
        
    def reset_view(self, refresh_delay = 250):
        """Redo all of the controls after something has changed
        
        refresh_delay - wait this many ms before refreshing the display
        """
        if self.__module is None:
            return
        if self.refresh_pending:
            return
        refresh_pending = True
        wx.CallLater(refresh_delay, self.do_reset)
        
    def do_reset(self):
        self.refresh_pending = False
        focus_control = wx.Window.FindFocus()
        if not focus_control is None:
            focus_name = focus_control.GetName()
        else:
            focus_name = None
        if self.__module is None:
            return
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
    

class FilterPanelController(object):
    '''Handle representation of the filter panel
    
    The code for handling the filter UI is moderately massive, so it gets
    its own class, if for no other reason than to organize the code.
    '''
    def __init__(self, module_view, v, panel):
        assert isinstance(module_view, ModuleView)
        assert isinstance(v, cps.Filter)
        self.module_view = module_view
        self.v = v
        self.panel = wx.Panel(self.module_view.module_panel,
                              style = wx.TAB_TRAVERSAL,
                              name = edit_control_name(self.v))
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer_dict = {}
        self.sizer_item_dict = {}
        self.stretch_spacer_dict = {}
        self.hide_show_dict = {}
        self.update()
       
    def get_sizer(self, address):
        '''Find or create the sizer that's associated with a particular address'''
        key = tuple(address)
        line_name = self.line_name(address)
        self.hide_show_dict[line_name] = True
        if self.sizer_dict.has_key(key):
            if len(address) > 0:
                self.hide_show_dict[self.remove_button_name(address)] = True
                self.hide_show_dict[self.add_button_name(address)] = True
                self.hide_show_dict[self.add_group_button_name(address)] = True
            return self.sizer_dict[key]
        #
        # Four possibilities:
        #
        # * The sizer is the top level one
        # * There is a sizer at the same level whose last address is one more.
        # * There are sizers at the same level whose next to last to address is
        #   one more than the next to last address of the address and whose
        #   last address is zero.
        # * None of the above which means the sizer can be added at the end.
        #
        line_style = wx.LI_HORIZONTAL | wx.BORDER_SUNKEN
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.indent(sizer, address)
        self.stretch_spacer_dict[key] = sizer.AddStretchSpacer()
        line = wx.StaticLine(self.panel, -1, style = line_style,
                             name = self.line_name(address))
        
        if len(address) == 0:
            key = None
        else:
            sizer.Add(self.make_delete_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.make_add_rule_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.make_add_rules_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            key = tuple(address[:-1] + [address[-1] + 1])
            if not self.sizer_dict.has_key(key):
                if len(address) == 1:
                    key = None
                else:
                    key = tuple(address[:-2] + [address[-2] + 1])
                    if not self.sizer_dict.has_key(key):
                        key = None
        if key is not None:
            next_sizer = self.sizer_dict[key]
            idx = self.get_sizer_index(self.panel.Sizer, next_sizer)
            self.panel.Sizer.Insert(idx, sizer, 0, wx.EXPAND)
            self.panel.Sizer.Insert(idx+1, line, 0, wx.EXPAND)
        else:
            self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
            self.panel.Sizer.Add(line, 0, wx.EXPAND)
        self.sizer_dict[tuple(address)] = sizer
        return sizer
    
    def get_tokens(self):
        try:
            tokens = self.v.parse()
        except Exception, e:
            logger.debug("Failed to parse filter (value=%s): %s",
                         self.v.text, str(e))
            tokens = self.v.default()
        #
        # Always require an "and" or "or" clause
        #
        if (len(tokens) == 0 or 
            (tokens[0] not in 
             (cps.Filter.AND_PREDICATE, cps.Filter.OR_PREDICATE))):
            tokens = [cps.Filter.AND_PREDICATE, tokens]
        return tokens
        
    def update(self):
        self.inside_update = True
        try:
            structure = self.get_tokens()
            for key in self.hide_show_dict:
                self.hide_show_dict[key] = False
            self.populate_subpanel(structure, [])
            for key, value in self.hide_show_dict.iteritems():
                self.panel.FindWindowByName(key).Show(value)
        except Exception:
            logger.exception("Threw exception while updating filter")
        finally:
            self.inside_update = False
    
    ANY_ALL_PREDICATES = [cps.Filter.AND_PREDICATE,
                          cps.Filter.OR_PREDICATE]

    def any_all_choices(self):
        return [x.display_name for x in self.ANY_ALL_PREDICATES]
        
    def indent(self, sizer, address):
        assert isinstance(sizer, wx.Sizer)
        if len(address) == 0:
            return
        sizer.AddSpacer((len(address)*20, 0))
        
    def find_and_mark(self, name):
        '''Find a control and mark it to be shown'''
        ctrl = self.panel.FindWindowByName(name)
        self.hide_show_dict[name] = True
        return ctrl
    
    def get_sizer_index(self, sizer, item):
        if isinstance(item, wx.Sizer):
            indexes = [i for i, s in enumerate(sizer.GetChildren())
                       if s.IsSizer() and s.GetSizer() is item]
        elif isinstance(item, wx.Window):
            indexes = [i for i, s in enumerate(sizer.GetChildren())
                       if s.IsWindow() and s.GetWindow() is item]
        elif isinstance(item, wx.SizerItem):
            return sizer.GetChildren().index(item)
        if len(indexes) > 0:
            return indexes[0]
        return None
    
    def on_value_change(self, event, new_text):
        if not self.inside_update:
            self.module_view.on_value_change(
                self.v, self.panel, new_text, event)
            
    def make_delete_button(self, address):
        name = self.remove_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button
            
        button = wx.Button(self.panel, -1, "-",
                           name = name,
                           style = wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, 
                    lambda event: self.on_delete_rule(event, address))
        return button
        
    def on_delete_rule(self, event, address):
        logger.debug("Delete row at " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        del sequence[address[-1] + 1]
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_add_rule_button(self, address):
        name = self.add_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button
        
        button = wx.Button(self.panel, -1, "+", 
                           name=name,
                           style = wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, 
                    lambda event: self.on_add_rule(event, address))
        return button
        
    def on_add_rule(self, event, address):
        logger.debug("Add rule after " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        new_rule = self.v.default()
        sequence.insert(address[-1]+2, new_rule)
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)
    
    def make_add_rules_button(self, address):
        name = self.add_group_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button
        button = wx.Button(self.panel, -1, "...",
                           name = name,
                           style = wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, 
                    lambda event: self.on_add_rules(event, address))
        return button
        
    def on_add_rules(self, event, address):
        logger.debug("Add rules after " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        new_rule = [cps.Filter.OR_PREDICATE, self.v.default()]
        sequence.insert(address[-1]+2, new_rule)
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)
        
    def make_predicate_choice(self, predicates, index, address, sizer):
        name = self.choice_name(index, address)
        choice_ctrl = self.find_and_mark(name)
        choices = [x.display_name for x in predicates]
        if choice_ctrl is not None:
            items = choice_ctrl.GetItems()
            if (len(items) != len(choices) or
                any([choice not in items for choice in choices])):
                choice_ctrl.SetItems(choices)
            return choice_ctrl
        choice_ctrl = wx.Choice(self.panel, -1, choices = choices, 
                                name=name)
        choice_ctrl.Bind(wx.EVT_CHOICE,
                         lambda event: self.on_predicate_changed(event, index, address))
        self.add_to_sizer(sizer, choice_ctrl, index, address)
        return choice_ctrl
        
    def on_predicate_changed(self, event, index,  address):
        logger.debug("Predicate choice at %d / %s changed" % (index, self.saddress(address)))
        structure = self.v.parse()
        sequence = self.find_address(structure, address)
        if index == 0:
            predicates = self.v.predicates
        else:
            predicates = sequence[index-1].subpredicates
        new_predicate = predicates[event.GetSelection()]
        sequence[index] = new_predicate
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def add_to_sizer(self, sizer, item, index, address):
        '''Insert the item in the sizer at the right location
        
        sizer - sizer for the line
        
        item - the control to be added
        
        index - index of the control within the sizer
        
        address - address of the sizer
        '''
        key = tuple(address + [index])
        next_key = tuple(address + [index + 1])
        if self.sizer_item_dict.has_key(next_key):
            next_ctrl = self.sizer_item_dict[next_key]
        else:
            next_ctrl = self.stretch_spacer_dict[tuple(address)]
        index = self.get_sizer_index(sizer, next_ctrl)
        sizer.Insert(index, item, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_HORIZONTAL)
        if not self.sizer_item_dict.has_key(key):
            self.sizer_item_dict[key] = item
            
    def make_literal(self, token, index, address, sizer):
        name = self.literal_name(index, address)
        literal_ctrl = self.find_and_mark(name)
        if literal_ctrl is not None:
            if literal_ctrl.GetValue() != token:
                literal_ctrl.SetValue(token)
            return literal_ctrl
        literal_ctrl = wx.TextCtrl(self.panel, -1, token, name=name)
        literal_ctrl.Bind(wx.EVT_TEXT, 
                          lambda event: self.on_literal_changed(event, index, address))
        self.add_to_sizer(sizer, literal_ctrl, index, address)
        return literal_ctrl
        
    def on_literal_changed(self, event, index, address):
        logger.debug("Literal at %d / %s changed" % (index, self.saddress(address)))
        structure = self.v.parse()
        sequence = self.find_address(structure, address)
        sequence[index] = event.GetString()
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_anyall_ctrl(self, address):
        anyall = wx.Choice(self.panel, -1, choices = self.any_all_choices(),
                           name = self.anyall_choice_name(address))
        anyall.Bind(wx.EVT_CHOICE, 
                    lambda event: self.on_anyall_changed(event, address))
        return anyall
    
    def on_anyall_changed(self, event, address):
        logger.debug("Any / all choice at %s changed" % self.saddress(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address)
        predicate = self.ANY_ALL_PREDICATES[event.GetSelection()]
        sequence[0] = predicate
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)
        
    def find_address(self, sequence, address):
        '''Find the sequence with the given address'''
        if len(address) == 0:
            return sequence
        subsequence = sequence[address[0] + 1]
        return self.find_address(subsequence, address[1:])
    
    def populate_subpanel(self, structure, address):
        parent_sizer = self.panel.Sizer
        any_all_name = self.anyall_choice_name(address)
        anyall = self.find_and_mark(any_all_name)
        self.hide_show_dict[self.static_text_name(0, address)] = True
        if len(address) == 0:
            self.hide_show_dict[self.static_text_name(1, address)] = True
        if anyall is None:
            anyall = self.make_anyall_ctrl(address)
            sizer = self.get_sizer(address)
            idx = self.get_sizer_index(sizer,
                                       self.stretch_spacer_dict[tuple(address)])
            if len(address) == 0:
                text = wx.StaticText(self.panel, -1, "Match", 
                                     name = self.static_text_name(0, address))
                sizer.Insert(idx, text, 0,
                             wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                sizer.InsertSpacer(idx+1, (3,0))
                sizer.Insert(idx+2, anyall, 0, 
                             wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                sizer.InsertSpacer(idx+3, (3,0))
                text = wx.StaticText(self.panel, -1, "of the following rules", 
                                     name = self.static_text_name(1, address))
                sizer.Insert(idx+4, text,
                          0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
            else:
                sizer.Insert(idx, anyall, 0,
                          wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.RIGHT)
                sizer.InsertSpacer(idx+1, (3,0))
                text = wx.StaticText(self.panel, -1, "of the following are true",
                                     name = self.static_text_name(0, address))
                sizer.Insert(idx+2, text,
                          0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        else:
            self.hide_show_dict[self.line_name(address)] = True
            if len(address) > 0:
                #
                # Show the buttons for the anyall if not top level
                #
                self.hide_show_dict[self.remove_button_name(address)] = True
                self.hide_show_dict[self.add_button_name(address)] = True
                self.hide_show_dict[self.add_group_button_name(address)] = True
            
        if anyall.GetStringSelection() != structure[0].display_name:
            anyall.SetStringSelection(structure[0].display_name)
        #
        # Now each subelement should be a list.
        #
        for subindex, substructure in enumerate(structure[1:]):
            subaddress = address + [subindex]
            if substructure[0].subpredicates is list:
                # A sublist
                self.populate_subpanel(substructure, subaddress)
            else:
                # A list of predicates
                sizer = self.get_sizer(subaddress)
                predicates = self.v.predicates
                for i, token in enumerate(substructure):
                    if isinstance(token, basestring):
                        literal_ctrl = self.make_literal(
                            token, i, subaddress, sizer)
                    else:
                        choice_ctrl = self.make_predicate_choice(
                            predicates, i, subaddress, sizer)
                        if choice_ctrl.GetStringSelection() != token.display_name:
                            choice_ctrl.SetStringSelection(token.display_name)
                        predicates = token.subpredicates
        #
        # Don't allow delete of only rule
        #
        name = self.remove_button_name(address + [0])
        delete_button = self.panel.FindWindowByName(name)
        delete_button.Enable(len(structure) > 2)
    
    @property
    def key(self):
        return str(self.v.key())

    def saddress(self, address):
        return "_".join([str(x) for x in address])
    
    def anyall_choice_name(self, address):
        return "%s_filter_anyall_%s" % (self.key, self.saddress(address))
    
    def choice_name(self, index, address):
        return "%s_choice_%d_%s" % (self.key, index, self.saddress(address))
    
    def literal_name(self, index, address):
        return "%s_literal_%d_%s" % (self.key, index, self.saddress(address))
    
    def remove_button_name(self, address):
        return "%s_remove_%s" % (self.key, self.saddress(address))
    
    def add_button_name(self, address):
        return "%s_add_%s" % (self.key, self.saddress(address))
    
    def add_group_button_name(self, address):
        return "%s_group_%s" % (self.key, self.saddress(address))
    def line_name(self, address):
        return "%s_line_%s" % (self.key, self.saddress(address))
    def static_text_name(self, index, address):
        return "%s_static_text_%d_%s" % (self.key, index, self.saddress(address))
    
        
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
        self.__items = []
    
    def get_item(self, i, j):
        if len(self.__items) <= j or len(self.__items[j]) <= i:
            return None
        return self.__items[j][i]
    
    def Reset(self, rows, cols=3, destroy_windows=True):
        if destroy_windows:
            windows = []
            for j in range(self.__rows):
                for i in range(self.__cols):
                    item = self.get_item(i,j)
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
        self.__items = []
        
    def Add(self, control, *args, **kwargs):
        if len(self.__items) == 0 or len(self.__items[-1]) == self.__cols:
            self.__items.append([])
        item = super(ModuleSizer, self).Add(control, *args, **kwargs)
        self.__items[-1].append(item)
        return item
    
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
                border_heights = [self.get_item(col,j).GetBorder() 
                                  for col in range(2)
                                  if self.get_item(col,j) is not None]
                if len(border_heights) == 0:
                    continue
                height_border = max(border_heights)
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
            item = self.get_item(i, j)
            if item is None:
                continue
            if item.IsWindow() and isinstance(item.GetWindow(), wx.StaticLine):
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
            item = self.get_item(j, i)
            if item is None:
                continue
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
            item = self.get_item(0, i)
            if item is None:
                continue
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
                text_item = self.get_item(0, i)
                edit_item = self.get_item(1, i)
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
                        item = self.get_item(j, i)
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

validation_queue_lock = threading.RLock()
validation_queue = []  # heapq, protected by above lock.  Can change to Queue.PriorityQueue() when we no longer support 2.5
pipeline_queue_thread = None  # global, protected by above lock
validation_queue_semaphore = threading.Semaphore(0)
request_pipeline_cache = threading.local()  # used to cache the last requested pipeline

def validate_module(pipeline, module_num, callback):
    '''Validate a module and execute the callback on error on the main thread
    
    pipeline - a pipeline to be validated
    module_num - the module number of the module to be validated
    callback - a callback with the signature, "fn(setting, message, pipeline_data)"
    where setting is the setting that is in error and message is the message to
    display.
    '''
    modules = [m for m in pipeline.modules() if m.module_num == module_num]
    if len(modules) != 1:
        return
    module = modules[0]
    level = logging.INFO
    setting_idx = None
    message = None
    try:
        level = logging.ERROR
        module.test_valid(pipeline)  # this method validates each visible
                                     # setting first, then the module itself.
        level = logging.WARNING
        module.validate_module_warnings(pipeline)
        level = logging.INFO
    except cps.ValidationError, instance:
        message = instance.message
        setting_idx = [m.key() for m in module.visible_settings()].index(instance.get_setting().key())
    wx.CallAfter(callback, setting_idx, message, level)

def validation_queue_handler():
    from cellprofiler.utilities.jutil import attach, detach
    attach()
    try:
        while True:
            validation_queue_semaphore.acquire()  # wait for work
            with validation_queue_lock:
                if len(validation_queue) == 0:
                    continue
                priority, module_num, pipeline, callback = heapq.heappop(validation_queue)
            try:
                validate_module(pipeline, module_num, callback)
            except:
                pass
    finally:
        detach()

def request_module_validation(pipeline, module, callback, priority=PRI_VALIDATE_BACKGROUND):
    '''Request that a module be validated

    pipeline - pipeline in question
    module - module in question
    callback - call this callback if there is an error. Do it on the GUI thread
    '''
    global pipeline_queue_thread, validation_queue

    # start validation queue handler thread if not already started
    with validation_queue_lock:
        if pipeline_queue_thread is None:
            pipeline_queue_thread = threading.Thread(target=validation_queue_handler)
            pipeline_queue_thread.setDaemon(True)
            pipeline_queue_thread.start()

    # minimize copies of pipelines
    pipeline_hash = pipeline.settings_hash()
    if pipeline_hash != getattr(request_pipeline_cache, "pipeline_hash", None):
        request_pipeline_cache.pipeline_hash = pipeline_hash
        request_pipeline_cache.pipeline = pipeline.copy()

    pipeline_copy = request_pipeline_cache.pipeline
    if pipeline_copy.settings_hash() != pipeline_hash:
        logger.warning("Pipeline and pipeline.copy() have different values for settings_hash()")
        # compare pipelines, try to find the changed setting
        orig_modules = pipeline.modules()
        copy_modules = pipeline_copy.modules()
        # If module names are changed by the copy operation, that's too much to continue from.
        assert [m.module_name for m in orig_modules] == [m.module_name for m in copy_modules], \
            "Module names do not match from original and copy, giving up!\nOrig: %s\nCopy: %s" % \
            ([m.module_name for m in orig_modules], [m.module_name for m in copy_modules])
        for midx, (om, cm) in enumerate(zip(orig_modules, copy_modules)):
            orig_settings = [s.unicode_value.encode('utf-8') for s in om.settings()]
            copy_settings = [s.unicode_value.encode('utf-8') for s in cm.settings()]
            differences = [oset != cset for oset, cset in zip(orig_settings, copy_settings)]
            if True in differences:
                logger.warning("  Differences in module #%d %s:" % (midx, om.module_name))
                for sidx, (diff, oset, cset) in enumerate(zip(differences, orig_settings, copy_settings)):
                    if diff:
                        logger.warning("    Setting #%d: was %s now %s" % (sidx, repr(oset), repr(cset)))

    with validation_queue_lock:
        # walk heap (as a list) removing any same-or-lower priority occurrences
        # of this module_num, to prevent the heap from growing indefinitely.
        mnum = module.module_num
        validation_queue = [req for req in validation_queue \
                                if ((req[0] >= priority) and (req[1] == mnum))]
        heapq.heapify(validation_queue)
        # order heap by priority, then module_number.
        heapq.heappush(validation_queue, (priority, module.module_num, pipeline_copy, callback))
    validation_queue_semaphore.release()  # notify handler of work

def clear_validation_cache():
    '''clear the cache when a new pipeline is loaded.'''
    global request_pipeline_cache
    setattr(request_pipeline_cache, "pipeline_hash", None)
