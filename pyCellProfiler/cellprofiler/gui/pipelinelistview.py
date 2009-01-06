"""PipelineListView.py
    $Revision$
    """
import wx
import cellprofiler.pipeline

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
        wx.EVT_IDLE(panel,self.on_idle)
        
    def __set_min_width(self):
        """Make the minimum width of the panel be the best width
           of the list box + the padding
        """
        width = self.__list_box.GetBestSize()[0]+2*PADDING
        self.__panel.SetMinSize(wx.Size(width,self.__panel.GetMinSize()[1]))

    def attach_to_pipeline(self,pipeline,controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        pipeline.add_listener(self.notify)
        controller.attach_to_pipeline_list_view(self)
        
    def attach_to_module_view(self, module_view):
        self.__module_view = module_view
        self.__panel.Bind(wx.EVT_LISTBOX,self.__on_item_selected,self.__list_box)
        
    def notify(self,pipeline,event):
        """Pipeline event notifications come through here
        
        """
        if isinstance(event,cellprofiler.pipeline.PipelineLoadedEvent):
            self.__on_pipeline_loaded(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleAddedPipelineEvent):
            self.__on_module_added(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleMovedPipelineEvent):
            self.__on_module_moved(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleRemovedPipelineEvent):
            self.__on_module_removed(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.PipelineClearedEvent):
            self.__on_pipeline_cleared(pipeline, event)
    
    def select_module(self,module_num,selected=True):
        """Select the given one-based module number in the list
        This is mostly for testing
        """
        self.__list_box.SetSelection(module_num-1,selected)
        
    def get_selected_modules(self):
        return [self.__list_box.GetClientData(i)\
                for i in self.__list_box.GetSelections() \
                if self.__list_box.Items[i] != NO_PIPELINE_LOADED]
    
    def __on_pipeline_loaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        self.__list_box.Clear()
        for module in pipeline.modules():
            self.__list_box.Append(module.module_name,module)
        self.__set_min_width()
    
    def __on_pipeline_cleared(self,pipeline,event):
        self.__list_box.SetItems([NO_PIPELINE_LOADED])
            
    def __on_module_added(self,pipeline,event):
        module=pipeline.modules()[event.module_num-1]
        if len(self.__list_box.GetItems()) == 1 and self.__list_box.GetItems()[0]==NO_PIPELINE_LOADED:
            self.__list_box.Clear()
        self.__list_box.Insert(module.module_name,event.module_num-1,module)
        self.__set_min_width()
    
    def __on_module_removed(self,pipeline,event):
        self.__list_box.Delete(event.module_num-1)
        self.__set_min_width()
        self.__module_view.clear_selection()
        
    def __on_module_moved(self,pipeline,event):
        module=pipeline.modules()[event.module_num-1]
        selected = False;
        for i in self.__list_box.GetSelections():
            if module == self.__list_box.GetClientData(i):
                selected = True
                break
        if event.direction == cellprofiler.pipeline.DIRECTION_UP:
            self.__list_box.Delete(event.module_num)
        else:
            self.__list_box.Delete(event.module_num-2)
        self.__list_box.Insert(module.module_name,event.module_num-1,module)
        if selected:
            self.__list_box.Select(event.module_num-1)
            
    def __on_item_selected(self,event):
        if self.__module_view:
            selections = self.__list_box.GetSelections()
            if len(selections) and not (len(selections)==1 and self.__list_box.GetItems()[0] == NO_PIPELINE_LOADED):
                self.__module_view.set_selection(self.__list_box.GetClientData(selections[0]).module_num)

    def on_idle(self,event):
        modules = self.__pipeline.modules()
        for idx,module in zip(range(len(modules)),modules):
            try:
                module.test_valid(self.__pipeline)
                target_name = module.module_name
            except:
                target_name = '*%s*'%(module.module_name)
            if self.__list_box.GetString(idx) != target_name:
                self.__list_box.SetString(idx,target_name)

