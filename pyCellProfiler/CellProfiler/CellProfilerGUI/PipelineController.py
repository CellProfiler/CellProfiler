"""PipelineController.py - controls (modifies) a pipeline

   $Revision$
"""
import CPFrame
import CellProfiler.Pipeline
import CellProfiler.Preferences
from CellProfiler.CellProfilerGUI.AddModuleFrame import AddModuleFrame
import CellProfiler.CellProfilerGUI.ModuleView
import math
import wx
import os
import re
import scipy.io.matlab.mio
import CellProfiler.Matlab.Utils

class PipelineController:
    """Controls the pipeline through the UI
    
    """
    def __init__(self,pipeline,frame):
        self.__pipeline =pipeline
        self.__frame = frame
        self.__add_module_frame = AddModuleFrame(frame,-1,"Add modules")
        self.__add_module_frame.AddListener(self.__OnAddToPipeline) 
        wx.EVT_MENU(frame,CPFrame.ID_FILE_LOAD_PIPELINE,self.__OnLoadPipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_SAVE_PIPELINE,self.__OnSavePipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_CLEAR_PIPELINE,self.__OnClearPipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_ANALYZE_IMAGES,self.OnAnalyzeImages)
    
    def AttachToPipelineListView(self,pipeline_list_view):
        """Glom onto events from the list box with all of the module names in it
        
        """
        self.__pipeline_list_view = pipeline_list_view
        
    def AttachToModuleView(self,module_view):
        """Listen for variable changes from the module view
        
        """
        self.__module_view = module_view
        module_view.AddListener(self.__OnModuleViewEvent)
    
    def AttachToModuleControlsPanel(self,module_controls_panel):
        """Attach the pipeline controller to the module controls panel
        
        Attach the pipeline controller to the module controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        self.__module_controls_panel = module_controls_panel
        self.__mcp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__help_button = wx.Button(self.__module_controls_panel,-1,"?",(0,0),(15,15))
        self.__mcp_text = wx.StaticText(self.__module_controls_panel,-1,"Adjust modules:")
        self.__mcp_add_module_button = wx.Button(self.__module_controls_panel,-1,"+",(0,0),(15,15))
        self.__mcp_remove_module_button = wx.Button(self.__module_controls_panel,-1,"-",(0,0),(15,15))
        self.__mcp_module_up_button = wx.Button(self.__module_controls_panel,-1,"^",(0,0),(15,15))
        self.__mcp_module_down_button = wx.Button(self.__module_controls_panel,-1,"v",(0,0),(15,15))
        self.__mcp_sizer.AddMany([(self.__help_button,0,wx.EXPAND),
                                  (self.__mcp_text,0,wx.EXPAND),
                                  (self.__mcp_add_module_button,0,wx.EXPAND),
                                  (self.__mcp_remove_module_button,0,wx.EXPAND),
                                  (self.__mcp_module_up_button,0,wx.EXPAND),
                                  (self.__mcp_module_down_button,0,wx.EXPAND)])
        self.__module_controls_panel.SetSizer(self.__mcp_sizer)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__OnHelp, self.__help_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__OnAddModule,self.__mcp_add_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__OnRemoveModule,self.__mcp_remove_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__OnModuleUp,self.__mcp_module_up_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__OnModuleDown,self.__mcp_module_down_button)
        
    def __OnLoadPipeline(self,event):
        dlg = wx.FileDialog(self.__frame,"Choose a pipeline file to open",wildcard="*.mat")
        if dlg.ShowModal()==wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            try:
                handles=scipy.io.matlab.mio.loadmat(pathname, struct_as_record=True)
            except Exception,instance:
                self.__frame.DisplayError('Failed to open %s'%(pathname),instance)
                return
            try:
                if handles.has_key('handles'):
                    self.__pipeline.CreateFromHandles(handles['handles'][0,0])
                else:
                    self.__pipeline.CreateFromHandles(handles)
            except Exception,instance:
                self.__frame.DisplayError('Failed during loading of %s'%(pathname),instance)

    def __OnSavePipeline(self,event):
        dlg = wx.FileDialog(self.__frame,"Save pipeline",wildcard="*.mat",style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            handles = self.__pipeline.SaveToHandles()
            scipy.io.matlab.mio.savemat(pathname,handles,format='5')
            
    def __OnClearPipeline(self,event):
        if wx.MessageBox("Do you really want to remove all modules from the pipeline?",
                         "Clearing pipeline",
                         wx.YES_NO | wx.ICON_QUESTION, self.__frame) == wx.YES:
            self.__pipeline.Clear()
            
    def __OnHelp(self,event):
        print "No help yet"
        
    def __OnAddModule(self,event):
        self.__add_module_frame.Show()
    
    def __GetSelectedModules(self):
        return self.__pipeline_list_view.GetSelectedModules()
    
    def __OnRemoveModule(self,event):
        selected_modules = self.__GetSelectedModules()
        for module in selected_modules:
            self.__pipeline.RemoveModule(module.ModuleNum())
            
    def __OnModuleUp(self,event):
        selected_modules = self.__GetSelectedModules()
        for module in selected_modules:
            self.__pipeline.MoveModule(module.ModuleNum(),CellProfiler.Pipeline.DIRECTION_UP);
        
    def __OnModuleDown(self,event):
        selected_modules = self.__GetSelectedModules()
        selected_modules.reverse()
        for module in selected_modules:
            self.__pipeline.MoveModule(module.ModuleNum(),CellProfiler.Pipeline.DIRECTION_DOWN);
    
    def __OnAddToPipeline(self,caller,event):
        selected_modules = self.__GetSelectedModules()
        ModuleNum = 1
        if len(selected_modules):
            ModuleNum=selected_modules[-1].ModuleNum()+1
        self.__pipeline.AddModule(event.ModulePath,ModuleNum)
        
    def __OnModuleViewEvent(self,caller,event):
        assert isinstance(event,CellProfiler.CellProfilerGUI.ModuleView.VariableEditedEvent), '%s is not an instance of CellProfiler.CellProfilerGUI.ModuleView.VariableEditedEvent'%(str(event))
        variable = event.GetVariable()
        proposed_value = event.GetProposedValue()
        if not variable.SetValue(proposed_value):
            event.Cancel()
            
    def OnAnalyzeImages(self,event):
        output_path = self.GetOutputFilePath()
        if output_path:
            handles = self.__pipeline.LoadPipelineIntoMatlab()
            handles = self.RunPipeline(handles)
            matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
            matlab.output_struct = matlab.struct('handles',handles)
            #
            # Here, foo._name is the undocumented variable name in the
            # matlab instance.
            #
            matlab.save(output_path,'-struct',str(matlab.output_struct._name))

    def GetOutputFilePath(self):
        path = os.path.join(CellProfiler.Preferences.GetDefaultOutputDirectory(),
                            CellProfiler.Preferences.GetOutputFileName())
        if os.path.exists(path):
            (first_part,ext)=os.path.splitext(path)
            start = 1
            match = re.match('^(.+)__([0-9]+)$',first_part)
            if match:
                first_part = match.groups()[0]
                start = int(match.groups()[1])
            for i in range(start,1000):
                alternate_name = '%(first_part)s__%(i)d%(ext)s'%(locals())
                if not os.path.exists(alternate_name):
                    break
            result = wx.MessageDialog(parent=self.__frame,
                                message='%s already exists. Would you like to create %s instead?'%(path, alternate_name),
                                caption='Output file exists',
                                style = wx.YES_NO+wx.ICON_QUESTION)
            user_choice = result.ShowModal()
            if user_choice & wx.YES:
                path = alternate_name
                CellProfiler.Preferences.SetOutputFileName(os.path.split(alternate_name)[1])
            else:
                return None
        return path
        
    def RunPipeline(self,handles):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        self.SetMatlabPath()
        DisplaySize = wx.GetDisplaySize()
        while handles.Current.SetBeingAnalyzed <= handles.Current.NumberOfImageSets:
            NumberofWindows = 0;
            SlotNumber = 0
            for module in self.__pipeline.Modules():
                handles.Current.CurrentModuleNumber = str(module.ModuleNum())
                if handles.Current.SetBeingAnalyzed == 1:
                    figure_field = 'FigureNumberForModule%d'%(module.ModuleNum())
                    if handles.Preferences.DisplayWindows[SlotNumber] == 0:
                        # Make up a fake figure for the module if we're not displaying its window
                        unused_figure_handle = math.ceil(max(matlab.findobj()))+1 
                        handles.Current = matlab.setfield(handles.Current,figure_field,unused_figure_handle)
                        figure = unused_figure_handle
                    else:
                        NumberofWindows = NumberofWindows+1;
                        LeftPos = DisplaySize.width * ((NumberofWindows-1)%12)/12;
                        figure = matlab.CPfigure(handles,'',
                                                 'Name','%s Display, cycle # '%(module.ModuleName()),
                                                 'Position',[LeftPos,DisplaySize.height-522, 560, 442])
                        handles.Current = matlab.setfield(handles.Current, figure_field, figure)
                module_error_measurement = 'ModuleError_%02d%s'%(module.ModuleNum(),module.ModuleName())
                failure = 1
                try:
                    handles = module.Run(handles)
                    failure = 0
                except Exception,instance:
                    self.__frame.DisplayError('Failed during run of module %s (module # %d)'%(module.ModuleName(),module.ModuleNum()),instance)
                if module.ModuleName() != 'Restart':
                    handles = matlab.CPaddmeasurements(handles,'Image',module_error_measurement,failure);
                SlotNumber+=1
            handles.Current.SetBeingAnalyzed += 1
        return handles
    
    def SetMatlabPath(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'DataTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'ImageTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'CPsubfunctions'),matlab.path())
        matlab.path(CellProfiler.Preferences.ModuleDirectory(),matlab.path())
        matlab.path(CellProfiler.Preferences.CellProfilerRootDirectory(),matlab.path())