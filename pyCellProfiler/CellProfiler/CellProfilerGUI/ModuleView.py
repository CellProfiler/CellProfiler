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

    def __set_columns(self):
        self.__grid.SetColLabelValue(0,'Variable description')
        self.__grid.SetColLabelValue(1,'Value')
        self.__grid.SetColSize(1,70)
        
    def ClearSelection(self):
        self.__grid.ClearGrid()
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
            self.__grid.SetCellRenderer(i,0,text_renderer)
            self.__grid.Table.SetValue(i,1,variable.Value())
        self.__grid.AutoSize()
            
        