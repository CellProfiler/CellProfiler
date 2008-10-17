"""ModuleView.py - implements a view on a module
    $Revision$
"""
import wx
import wx.grid
import CellProfiler.Pipeline
import CellProfiler.Variable

class VariableEditedEvent:
    """Represents an attempt by the user to edit a variable
    
    """
    def __init__(self,variable,proposed_value,event):
        self.__variable = variable
        self.__proposed_value = proposed_value
        self.__event = event
        self.__accept_change = True
    
    def GetVariable(self):
        """Return the variable being edited
        
        """
        return self.__variable
    
    def GetProposedValue(self):
        """Return the value proposed by the user
        
        """
        return self.__proposed_value
    
    def Cancel(self):
        self.__accept_change = False
        
    def AcceptChange(self):
        return self.__accept_change
    def UIEvent(self):
        """The event from the UI that triggered the edit
        
        """
        return self.__event

class ModuleView:
    """The module view implements a view on CellProfiler.Module
    
    The module view implements a view on CellProfiler.Module. The view consists
    of a table composed of one row per variable. The first column of the table
    has the explanatory text and the second has a VariableView which
    gives the ui for editing the variable.
    """
    
    def __init__(self,module_panel,pipeline):
        self.__module_panel = module_panel
        self.__pipeline = pipeline
        pipeline.AddListener(self.__OnPipelineEvent)
        self.__listeners = []
        self.__value_listeners = []
        self.__module = None
        self.__module_panel.SetupScrolling()   

    def __set_columns(self):
        self.__grid.SetColLabelValue(0,'Variable description')
        self.__grid.SetColLabelValue(1,'Value')
        self.__grid.SetColSize(1,70)
        
    def ClearSelection(self):
        if self.__module:
            for listener in self.__value_listeners:
                listener['notifier'].RemoveListener(listener['listener'])
            self.__value_listeners = []
            self.__module_panel.DestroyChildren()
            self.__module = None
        
    def SetSelection(self,ModuleNum):
        self.ClearSelection()
        self.__module = self.__pipeline.Module(ModuleNum)
        self.__controls = []
        data = []
        annotations = CellProfiler.Variable.GetAnnotationsAsDictionary(self.__module.Annotations())
        variables = self.__module.Variables()
        sizer = ModuleSizer(len(variables),2)
        for i in range(0,len(self.__module.Variables())):
            variable = self.__module.Variables()[i]
            vn = variable.VariableNumber()
            assert annotations.has_key(vn), 'There are no annotations for variable # %d'%(vn)
            if annotations[vn].has_key('text'):
                text = annotations[vn]['text'][0].Value
            elif annotations[vn].has_key('pathnametext'):
                text = annotations[vn]['pathnametext'][0].Value
            elif annotations[vn].has_key('filenametext'):
                text = annotations[vn]['filenametext'][0].Value
            else:
                text = ''
            static_text = wx.StaticText(self.__module_panel,-1,text,style=wx.ALIGN_RIGHT)
            sizer.Add(static_text,1,wx.EXPAND|wx.ALL,2)
            variable_choices = self.__pipeline.GetVariableChoices(variable)
            if variable_choices:
                choices = variable_choices.GetChoices(variable)
                if (not variable_choices.CanChange() and not variable_choices.CanAcceptOther()
                    and all([x in ['Yes','No'] for x in choices])):
                    control = wx.CheckBox(self.__module_panel,-1)
                    control.SetValue(variable.Value()=='Yes')
                    self.__module_panel.Bind(wx.EVT_CHECKBOX,lambda event,variable=variable,control=control: self.__OnCellChange(event, variable, control),control)
                else:
                    style = (variable_choices.CanAcceptOther() and wx.CB_DROPDOWN) or wx.CB_READONLY
                    if (len(choices)==0 and variable_choices.CanAcceptOther()):
                        choices=['None']
                    control = wx.ComboBox(self.__module_panel,-1,variable.Value(),
                                          choices=choices,
                                          style=style)
                    self.__module_panel.Bind(wx.EVT_COMBOBOX,lambda event,variable=variable,control=control: self.__OnCellChange(event, variable,control),control)
                    if variable_choices.CanChange():
                        listener = lambda sender,event,vn=vn: self.__OnVariableChoicesChanged(sender,event,vn)
                        listener_dict = {'notifier':variable_choices,
                                         'listener':listener }
                        variable_choices.AddListener(listener)
                        self.__value_listeners.append(listener_dict)
                    if variable_choices.CanAcceptOther():
                        self.__module_panel.Bind(wx.EVT_TEXT,lambda event,variable=variable,control=control: self.__OnCellChange(event, variable, control),control)
            else:
                 control = wx.TextCtrl(self.__module_panel,-1,variable.Value())
                 self.__module_panel.Bind(wx.EVT_TEXT,lambda event,variable=variable,control=control: self.__OnCellChange(event, variable,control),control)
            sizer.Add(control,0,wx.EXPAND|wx.ALL,2)
            self.__controls.append(control)
        self.__module_panel.SetSizer(sizer)
        self.__module_panel.Layout()
    
    def AddListener(self,listener):
        self.__listeners.append(listener)
    
    def RemoveListener(self,listener):
        self.__listeners.remove(listener)
    
    def Notify(self,event):
        for listener in self.__listeners:
            listener(self,event)
            
    def __OnColumnSized(self,event):
        self.__module_panel.GetTopLevelParent().Layout()
    
    def __OnCellChange(self,event,variable,control):
        old_value = variable.Value()
        if isinstance(control,wx.CheckBox):
            proposed_value = (control.GetValue() and 'Yes') or 'No'
        else:
            proposed_value = control.GetValue()
        variable_edited_event = VariableEditedEvent(variable,proposed_value,event)
        self.Notify(variable_edited_event)
        if not variable_edited_event.AcceptChange():
            control.SetValue(old_value)
    
    def __OnPipelineEvent(self,pipeline,event):
        if (isinstance(event,CellProfiler.Pipeline.PipelineLoadedEvent) or
            isinstance(event,CellProfiler.Pipeline.PipelineClearedEvent)):
            self.ClearSelection()
    
    def __OnVariableChoicesChanged(self,sender,event,VariableNum):
        idx = VariableNum-1
        variable = self.__module.Variables()[idx]
        control = self.__controls[idx]
        assert isinstance(control,wx.ComboBox)
        control.SetItems(sender.GetChoices(variable))
        control.SetValue(variable.Value())
    
class ModuleSizer(wx.PySizer):
    """The module sizer uses the maximum best width of the variable edit controls
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
        size = self.CalcEditSize()
        height = 0
        for j in range(0,self.__rows):
            row_height = 0
            for i in range(0,self.__cols):
                item = self.GetItem(self.Idx(i,j))
                row_height = max([row_height,item.CalcMin()[1]])
            height += row_height;
        return wx.Size(size[0]+self.__min_text_width,height)
        
    def CalcEditSize(self):
        height = 0
        width  = 0
        for i in range(0,self.__rows):
            item = self.GetItem(self.Idx(1,i))
            size = item.CalcMin()
            height += size[1]
            width = max(width,size[0])
        return wx.Size(width,height)
    
    def RecalcSizes(self):
        """Recalculate the sizes of our items, resizing the text boxes as we go  
        """
        size = self.GetSize()
        width = size[0]
        edit_size = self.CalcEditSize()
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
            text_item = self.GetItem(self.Idx(0,i))
            edit_item = self.GetItem(self.Idx(1,i))
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

    def Coords(self,idx):
        """Return the column/row coordinates of an indexed item
        
        """
        (col,row) = divmod(idx,self.__cols)
        return (col,row)

    def Idx(self,col,row):
        """Return the index of the given grid cell
        
        """
        return row*self.__cols + col