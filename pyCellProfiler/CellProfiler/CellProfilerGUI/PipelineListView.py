"""PipelineListView.py
    $Revision$
    """
import wx
import CellProfiler.Pipeline

class PipelineListView:
    """View on a set of modules
    
    """
    def __init__(self,panel):
        self.__panel=panel
        self.__sizer=wx.BoxSizer(wx.VERTICAL)
        self.__list_box=wx.ListBox(self.__panel,-1,
                                   choices=['No pipeline loaded'],
                                   style=wx.LB_EXTENDED)
        self.__sizer.Add(self.__list_box,1,wx.EXPAND|wx.ALL,1)
        self.__panel.SetSizer(self.__sizer)

    def AttachToPipeline(self,pipeline,controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        pipeline.AddListener(self)
        controller.AttachToPipelineListView(self)
        
    def AttachToModuleView(self, module_view):
        self.__module_view = module_view
        self.__panel.Bind(wx.EVT_LISTBOX,self.__OnItemSelected,self.__list_box)
        
    def Notify(self,pipeline,event):
        """Pipeline event notifications come through here
        
        """
        if isinstance(event,CellProfiler.Pipeline.PipelineLoadedEvent):
            self.__OnPipelineLoaded(pipeline,event)
        elif isinstance(event,CellProfiler.Pipeline.ModuleAddedPipelineEvent):
            self.__OnModuleAdded(pipeline,event)
        elif isinstance(event,CellProfiler.Pipeline.ModuleMovedPipelineEvent):
            self.__OnModuleMoved(pipeline,event)
        elif isinstance(event,CellProfiler.Pipeline.ModuleRemovedPipelineEvent):
            self.__OnModuleRemoved(pipeline,event)
    
    def GetSelectedModules(self):
        return [self.__list_box.GetClientData(i) for i in self.__list_box.GetSelections()]
    
    def __OnPipelineLoaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        self.__list_box.Clear()
        for module in pipeline.Modules():
            self.__list_box.Append(module.ModuleName(),module)
            
    def __OnModuleAdded(self,pipeline,event):
        module=pipeline.Modules()[event.ModuleNum-1]
        self.__list_box.Insert(module.ModuleName(),event.ModuleNum-1,module)
    
    def __OnModuleRemoved(self,pipeline,event):
        self.__list_box.Delete(event.ModuleNum-1)
        
    def __OnModuleMoved(self,pipeline,event):
        module=pipeline.Modules()[event.ModuleNum-1]
        selected = False;
        for i in self.__list_box.GetSelections():
            if module == self.__list_box.GetClientData(i):
                selected = True
                break
        if event.Direction == CellProfiler.Pipeline.DIRECTION_UP:
            self.__list_box.Delete(event.ModuleNum)
        else:
            self.__list_box.Delete(event.ModuleNum-2)
        self.__list_box.Insert(module.ModuleName(),event.ModuleNum-1,module)
        if selected:
            self.__list_box.Select(event.ModuleNum-1)
            
    def __OnItemSelected(self,event):
        if self.__module_view:
            selections = self.__list_box.GetSelections()
            if len(selections):
                self.__module_view.SetSelection(self.__list_box.GetClientData(selections[0]).ModuleNum())

