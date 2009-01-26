"""ModuleView.py - implements a view on a module
"""
__version__="$Revision$"
import wx
import wx.grid
import cellprofiler.pipeline
import cellprofiler.settings

ERROR_COLOR = wx.RED
RANGE_TEXT_WIDTH = 40 # number of pixels in a range text box TO_DO - calculate it

class SettingEditedEvent:
    """Represents an attempt by the user to edit a setting
    
    """
    def __init__(self,setting,proposed_value,event):
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
    
    def __init__(self,module_panel,pipeline):
        self.__module_panel = module_panel
        self.__pipeline = pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        self.__listeners = []
        self.__value_listeners = []
        self.__module = None
        self.__module_panel.SetupScrolling()
        wx.EVT_IDLE(module_panel,self.on_idle)

    def __set_columns(self):
        self.__grid.SetColLabelValue(0,'Setting description')
        self.__grid.SetColLabelValue(1,'Value')
        self.__grid.SetColSize(1,70)
    
    def get_module_panel(self):
        """The panel that hosts the module controls
        
        This is exposed for testing purposes.
        """
        return self.__module_panel
    
    module_panel = property(get_module_panel)
    
    def clear_selection(self):
        if self.__module:
            for listener in self.__value_listeners:
                listener['notifier'].remove_listener(listener['listener'])
            self.__value_listeners = []
            self.__module_panel.DestroyChildren()
            self.__module = None
    
    def hide_settings(self):
        for child in self.__module_panel.Children:
            child.Hide()
        
    def set_selection(self,module_num):
        """Initialize the controls in the view to the settings of the module"""
        self.module_panel.Freeze()
        try:
            if self.__module and self.__module.module_num == module_num:
                self.hide_settings()
            else:
                self.clear_selection()
            self.__module       = self.__pipeline.module(module_num)
            self.__controls     = []
            self.__static_texts = []
            data                = []
            settings           = self.__module.visible_settings()
            sizer               = ModuleSizer(len(settings),2)
            
            for v,i in zip(settings, range(0,len(settings))):
                control_name = edit_control_name(v)
                text_name    = text_control_name(v)
                static_text  = self.__module_panel.FindWindowByName(text_name)
                control      = self.__module_panel.FindWindowByName(control_name)
                if static_text:
                    static_text.Show()
                else:
                    static_text = wx.StaticText(self.__module_panel,
                                                -1,
                                                encode_label(v.text),
                                                style=wx.ALIGN_RIGHT,
                                                name=text_name)
                if control:
                    control.Show()
                sizer.Add(static_text,1,wx.EXPAND|wx.ALL,2)
                self.__static_texts.append(static_text)
                if isinstance(v,cellprofiler.settings.Binary):
                    control = self.make_binary_control(v,control_name,control)
                elif isinstance(v,cellprofiler.settings.Choice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name, 
                                                       wx.CB_READONLY,
                                                       control)
                elif isinstance(v,cellprofiler.settings.CustomChoice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name, 
                                                       wx.CB_DROPDOWN,
                                                       control)
                elif isinstance(v,cellprofiler.settings.NameSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_choice_control(v, choices,
                                                       control_name, 
                                                       wx.CB_DROPDOWN,
                                                       control)
                elif isinstance(v, cellprofiler.settings.DoSomething):
                    control = self.make_callback_control(v, control_name,
                                                         control)
                elif isinstance(v, cellprofiler.settings.IntegerRange) or\
                     isinstance(v, cellprofiler.settings.FloatRange):
                    control = self.make_range_control(v, control)
                else:
                    control = self.make_text_control(v, control_name, control)
                sizer.Add(control,0,wx.EXPAND|wx.ALL,2)
                self.__controls.append(control)
            self.__module_panel.SetSizer(sizer)
            self.__module_panel.Layout()
        finally:
            self.module_panel.Thaw()
    
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
                control.Items = choices
            if control.Value != v.value:
                control.Value = v.value
            
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
    
    def make_text_control(self, v, control_name, control):
        """Make a textbox control"""
        if not control:
            control = wx.TextCtrl(self.__module_panel,
                                  -1,
                                  str(v),
                                  name=control_name)
            def on_cell_change(event, setting = v, control=control):
                self.__on_cell_change(event, setting,control)
            self.__module_panel.Bind(wx.EVT_TEXT,on_cell_change,control)
        elif control.Value != v.value:
            control.Value = str(v.value)
        return control
    
    def make_range_control(self, v, panel):
        """Make a "control" composed of a panel and two edit boxes representing a range"""
        if not panel:
            panel = wx.Panel(self.__module_panel,-1,name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            min_ctrl = wx.TextCtrl(panel,-1,str(v.min),
                                   name=min_control_name(v))
            best_width = min_ctrl.GetCharWidth()*5
            min_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(min_ctrl,0,wx.EXPAND|wx.RIGHT,1)
            max_ctrl = wx.TextCtrl(panel,-1,str(v.max),
                                   name=max_control_name(v))
            max_ctrl.SetInitialSize(wx.Size(best_width,-1))
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
                min_ctrl.Value = str(v.max)
            
        return panel
        
    def add_listener(self,listener):
        self.__listeners.append(listener)
    
    def remove_listener(self,listener):
        self.__listeners.remove(listener)
    
    def notify(self,event):
        for listener in self.__listeners:
            listener(self,event)
            
    def __on_column_sized(self,event):
        self.__module_panel.GetTopLevelParent().Layout()
    
    def __on_checkbox_change(self,event,setting,control):
        self.__on_cell_change(event, setting, control)
        self.reset_view()
    
    def __on_combobox_change(self,event,setting,control):
        self.__on_cell_change(event, setting, control)
        self.reset_view()
        
    def __on_cell_change(self,event,setting,control):
        old_value = str(setting)
        if isinstance(control,wx.CheckBox):
            proposed_value = (control.GetValue() and 'Yes') or 'No'
        else:
            proposed_value = str(control.GetValue())
        setting_edited_event = SettingEditedEvent(setting,proposed_value,event)
        self.notify(setting_edited_event)
    
    def __on_min_change(self,event,setting,control):
        old_value = str(setting)
        proposed_value="%s,%s"%(str(control.Value),str(setting.max))
        setting_edited_event = SettingEditedEvent(setting,proposed_value,event)
        self.notify(setting_edited_event)
        
    def __on_max_change(self,event,setting,control):
        old_value = str(setting)
        proposed_value="%s,%s"%(str(setting.min),str(control.Value))
        setting_edited_event = SettingEditedEvent(setting,proposed_value,event)
        self.notify(setting_edited_event)
        
    def __on_pipeline_event(self,pipeline,event):
        if (isinstance(event,cellprofiler.pipeline.PipelineLoadedEvent) or
            isinstance(event,cellprofiler.pipeline.PipelineClearedEvent)):
            self.clear_selection()
    
    def __on_do_something(self, event, setting):
        setting.on_event_fired()
        self.reset_view()
    
    def on_idle(self,event):
        """Check to see if the selected module is valid"""
        if self.__module:
            validation_error = None
            try:
                self.__module.test_valid(self.__pipeline)
            except cellprofiler.settings.ValidationError, instance:
                validation_error = instance
            for idx, setting in zip(range(len(self.__module.visible_settings())),self.__module.visible_settings()):
                try:
                    if validation_error and validation_error.setting.key() == setting.key():
                        raise validation_error
                    setting.test_valid(self.__pipeline)
                    if self.__static_texts[idx].GetForegroundColour() == ERROR_COLOR:
                        self.__controls[idx].SetToolTipString('')
                        self.__static_texts[idx].SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
                        self.__static_texts[idx].Refresh()
                except cellprofiler.settings.ValidationError, instance:
                    if self.__static_texts[idx].GetForegroundColour() != ERROR_COLOR:
                        self.__controls[idx].SetToolTipString(instance.message)
                        self.__static_texts[idx].SetForegroundColour(ERROR_COLOR)
                        self.__static_texts[idx].Refresh()
    
    def reset_view(self):
        """Redo all of the controls after something has changed
        
        TO_DO: optimize this so that only things that have changed IRL change in the GUI
        """
        focus_control = wx.Window.FindFocus()
        focus_name = focus_control.GetName()
        self.set_selection(self.__module.module_num)
        focus_control = self.module_panel.FindWindowByName(focus_name)
        if focus_control:
            focus_control.SetFocus()
        
class ModuleSizer(wx.PySizer):
    """The module sizer uses the maximum best width of the setting edit controls
    to compute the column widths, then it sets the text controls to wrap within
    the remaining space, then it uses the best height of each text control to lay
    out the rows.
    """
    
    def __init__(self,rows,cols=2):
        wx.PySizer.__init__(self)
        self.__rows = rows
        self.__cols = cols
        self.__min_text_width = 150

    def CalcMin(self):
        """Calculate the minimum from the edit controls
        """
        size = self.calc_edit_size()
        height = 0
        for j in range(0,self.__rows):
            row_height = 0
            for i in range(0,self.__cols):
                item = self.GetItem(self.idx(i,j))
                row_height = max([row_height,item.CalcMin()[1]])
            height += row_height;
        return wx.Size(size[0]+self.__min_text_width,height)
        
    def calc_edit_size(self):
        height = 0
        width  = 0
        for i in range(0,self.__rows):
            item = self.GetItem(self.idx(1,i))
            size = item.CalcMin()
            height += size[1]
            width = max(width,size[0])
        return wx.Size(width,height)
    
    def RecalcSizes(self):
        """Recalculate the sizes of our items, resizing the text boxes as we go  
        """
        size = self.GetSize()
        width = size[0]
        edit_size = self.calc_edit_size()
        edit_width = edit_size[0] # the width of the edit controls portion
        text_width = max([width-edit_width,self.__min_text_width])
        #
        # Change all static text controls to wrap at the text width. Then
        # ask the items how high they are and do the layout of the line.
        #
        height = 0
        widths = [text_width, edit_width]
        panel = self.GetContainingWindow()
        for i in range(0,self.__rows):
            text_item = self.GetItem(self.idx(0,i))
            edit_item = self.GetItem(self.idx(1,i))
            inner_text_width = text_width - 2*text_item.GetBorder() 
            items = [text_item, edit_item]
            control = text_item.GetWindow()
            assert isinstance(control,wx.StaticText), 'Control at column 0, %d of grid is not StaticText: %s'%(i,str(control))
            text = control.GetLabel()
            text = text.replace('\n',' ')
            control.SetLabel(text)
            control.Wrap(inner_text_width)
            item_heights = [text_item.CalcMin()[1],edit_item.CalcMin()[1]]
            for j in range(0,self.__cols):
                item_size = wx.Size(widths[j],item_heights[j])
                item_location = wx.Point(sum(widths[0:j]),height)
                item_location = panel.CalcScrolledPosition(item_location)
                items[j].SetDimension(item_location, item_size)
            height += max(item_heights)
        if height > panel.GetVirtualSize()[1]:
            panel.SetVirtualSizeWH(panel.GetVirtualSize()[0],height+20)

    def coords(self,idx):
        """Return the column/row coordinates of an indexed item
        
        """
        (col,row) = divmod(idx,self.__cols)
        return (col,row)

    def idx(self,col,row):
        """Return the index of the given grid cell
        
        """
        return row*self.__cols + col
