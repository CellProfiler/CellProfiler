"""PipelineListView.py
    $Revision$
    """
import wx
import CellProfiler.Pipeline

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

class PipelineListView:
    """View on a set of modules
    
    """
    def __init__(self,panel):
        self.__panel=panel
        self.__sizer=wx.BoxSizer(wx.VERTICAL)
        self.__list_box=wx.ListBox(self.__panel,-1,
                                   choices=[NO_PIPELINE_LOADED],
                                   style=wx.LB_EXTENDED)
        self.__sizer.Add(self.__list_box,1,wx.EXPAND|wx.LEFT|wx.RIGHT,PADDING)
        self.__panel.SetSizer(self.__sizer)
        self.__set_min_width()
        
    def __set_min_width(self):
        """Make the minimum width of the panel be the best width
           of the list box + the padding
        """
        width = self.__list_box.GetBestSize()[0]+2*PADDING
        self.__panel.SetMinSize(wx.Size(width,self.__panel.GetMinSize()[1]))

    def AttachToPipeline(self,pipeline,controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        pipeline.AddListener(self.Notify)
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
        elif isinstance(event,CellProfiler.Pipeline.PipelineClearedEvent):
            self.__OnPipelineCleared(pipeline, event)
    
    def GetSelectedModules(self):
        return [self.__list_box.GetClientData(i) for i in self.__list_box.GetSelections()]
    
    def __OnPipelineLoaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        self.__list_box.Clear()
        for module in pipeline.Modules():
            self.__list_box.Append(module.ModuleName(),module)
        self.__set_min_width()
    
    def __OnPipelineCleared(self,pipeline,event):
        self.__list_box.Clear()
            
    def __OnModuleAdded(self,pipeline,event):
        module=pipeline.Modules()[event.ModuleNum-1]
        if len(self.__list_box.GetItems()) == 1 and self.__list_box.GetItems()[0]==NO_PIPELINE_LOADED:
            self.__list_box.Clear()
        self.__list_box.Insert(module.ModuleName(),event.ModuleNum-1,module)
        self.__set_min_width()
    
    def __OnModuleRemoved(self,pipeline,event):
        self.__list_box.Delete(event.ModuleNum-1)
        self.__set_min_width()
        
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

