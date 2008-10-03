"""ModuleView.py - implements a view on a module
    $Revision$
"""
import wx
import wx.grid
import CellProfiler.Pipeline
import CellProfiler.Variable

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
        self.__sizer = wx.GridSizer(1,1,0,0)
        self.__grid = wx.grid.Grid(module_panel,-1)
        self.__grid.CreateGrid(0,2)
        self.__grid.SetRowLabelSize(20)
        self.__sizer.Add(self.__grid,0,wx.EXPAND|wx.ALL,1)
        module_panel.SetSizer(self.__sizer)
        self.__set_columns()
        module_panel.Bind(wx.grid.EVT_GRID_COL_SIZE,self.__OnColumnSized,self.__grid)
        module_panel.Bind(wx.grid.EVT_GRID_CELL_CHANGE,self.__OnCellChange,self.__grid)
        pipeline.AddListener(self.__OnPipelineEvent)
        self.__grid.AutoSize()
        self.__listeners = []
        self.__value_listeners = []

    def __set_columns(self):
        self.__grid.SetColLabelValue(0,'Variable description')
        self.__grid.SetColLabelValue(1,'Value')
        self.__grid.SetColSize(1,70)
        
    def ClearSelection(self):
        if self.__grid.GetNumberRows()>0:
            for listener in self.__value_listeners:
                listener['notifier'].RemoveListener(listener['listener'])
            self.__value_listeners = []
            self.__grid.DeleteRows(0,self.__grid.GetNumberRows())
            self.__module = None
        
    def SetSelection(self,ModuleNum):
        if self.__grid.GetNumberRows():
            self.__grid.DeleteRows(0,self.__grid.GetNumberRows())
        self.__module = self.__pipeline.Module(ModuleNum)
        data = []
        annotations = CellProfiler.Variable.GetAnnotationsAsDictionary(self.__module.Annotations())
        self.__grid.AppendRows(len(self.__module.Variables()))
        text_renderer = wx.grid.GridCellAutoWrapStringRenderer()
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
            self.__grid.Table.SetRowLabelValue(i,str(vn))
            self.__grid.Table.SetValue(i,0,text)
            self.__grid.SetReadOnly(i,0,True)
            self.__grid.SetCellRenderer(i,0,text_renderer)
            self.__grid.Table.SetValue(i,1,variable.Value())
            variable_choices = self.__pipeline.GetVariableChoices(variable)
            if variable_choices:
                editor = wx.grid.GridCellChoiceEditor(variable_choices.GetChoices(variable),
                                                      variable_choices.CanAcceptOther())
                self.__grid.SetCellEditor(i,1,editor)
                if variable_choices.CanChange():
                    listener = lambda sender,event: __OnVariableChoicesChanged(sender,event,vn)
                    listener_dict = {'notifier':variable_choices,
                                     'listener':listener }
                    variable_choices.AddListener(listener)
        self.__grid.AutoSize()
    
    def AddListener(self,listener):
        self.__listeners.append(listener)
    
    def RemoveListener(self,listener):
        self.__listeners.remove(listener)
    
    def Notify(self,event):
        for listener in self.__listeners:
            listener(self,event)
            
    def __OnColumnSized(self,event):
        self.__module_panel.GetTopLevelParent().Layout()
    
    def __OnCellChange(self,event):
        variable = self.__module.Variables()[event.Row]
        old_value = variable.Value()
        proposed_value = self.__grid.GetCellValue(event.Row,1)
        variable_edited_event = VariableEditedEvent(variable,proposed_value,event)
        self.Notify(variable_edited_event)
        if not variable_edited_event.AcceptChange():
            self.__grid.SetCellValue(event.Row,1,old_value)
    
    def __OnPipelineEvent(self,pipeline,event):
        if (isinstance(event,CellProfiler.Pipeline.PipelineLoadedEvent) or
            isinstance(event,CellProfiler.Pipeline.PipelineClearedEvent)):
            self.ClearSelection()
    
    def __OnVariableChoicesChanged(self,sender,event,VariableNum):
        idx = VariableNum-1
        variable = self.__module.Variables()[idx]
        editor = wx.grid.GridCellChoiceEditor(sender.GetChoices(variable),
                                              sender.CanAcceptOther())
        self.__grid.SetCellEditor(idx,1,editor)
    
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