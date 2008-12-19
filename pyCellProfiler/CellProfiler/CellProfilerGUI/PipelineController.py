"""PipelineController.py - controls (modifies) a pipeline

   $Revision$
"""
import CPFrame
import CellProfiler.Pipeline
import CellProfiler.Preferences
from CellProfiler.CellProfilerGUI.AddModuleFrame import AddModuleFrame
import CellProfiler.CellProfilerGUI.ModuleView
import math
import numpy
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
        pipeline.AddListener(self.__OnPipelineEvent)
        self.__frame = frame
        self.__add_module_frame = AddModuleFrame(frame,-1,"Add modules")
        self.__add_module_frame.AddListener(self.__OnAddToPipeline)
        self.__variable_errors = {}
        self.__running_pipeline = None 
        self.__pipeline_measurements = None
        wx.EVT_MENU(frame,CPFrame.ID_FILE_LOAD_PIPELINE,self.__OnLoadPipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_SAVE_PIPELINE,self.__OnSavePipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_CLEAR_PIPELINE,self.__OnClearPipeline)
        wx.EVT_MENU(frame,CPFrame.ID_FILE_ANALYZE_IMAGES,self.OnAnalyzeImages)
        wx.EVT_IDLE(frame,self.OnIdle)
    
    def AttachToPipelineListView(self,pipeline_list_view):
        """Glom onto events from the list box with all of the module names in it
        
        """
        self.__pipeline_list_view = pipeline_list_view
        
    def AttachToModuleView(self,module_view):
        """Listen for variable changes from the module view
        
        """
        self.__module_view = module_view
        module_view.AddListener(self.__OnModuleViewEvent)
    
    def AttachToDirectoryView(self,directory_view):
        """Listen for requests to load pipelines
        
        """
        self.__directory_view = directory_view
        directory_view.AddPipelineListener(self.__OnDirLoadPipeline)
    
    def AttachToModuleControlsPanel(self,module_controls_panel):
        """Attach the pipeline controller to the module controls panel
        
        Attach the pipeline controller to the module controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        self.__module_controls_panel = module_controls_panel
        self.__mcp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__help_button = wx.Button(self.__module_controls_panel,-1,"?",(0,0),(25,25))
        self.__mcp_text = wx.StaticText(self.__module_controls_panel,-1,"Adjust modules:")
        self.__mcp_add_module_button = wx.Button(self.__module_controls_panel,-1,"+",(0,0),(25,25))
        self.__mcp_remove_module_button = wx.Button(self.__module_controls_panel,-1,"-",(0,0),(25,25))
        self.__mcp_module_up_button = wx.Button(self.__module_controls_panel,-1,"^",(0,0),(25,25))
        self.__mcp_module_down_button = wx.Button(self.__module_controls_panel,-1,"v",(0,0),(25,25))
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
            self.__do_load_pipeline(pathname)
    
    def __OnDirLoadPipeline(self,caller,event):
        if wx.MessageBox('Do you want to load the pipeline, "%s"?'%(os.path.split(event.Path)[1]),
                         'Load path', wx.YES_NO|wx.ICON_QUESTION ,self.__frame) & wx.YES:
            self.__do_load_pipeline(event.Path)
    
    def __do_load_pipeline(self,pathname):
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
            self.__ClearErrors()
        except Exception,instance:
            self.__frame.DisplayError('Failed during loading of %s'%(pathname),instance)

    def __ClearErrors(self):
        for key,error in self.__variable_errors.iteritems():
            self.__frame.PreferencesView.PopErrorText(error)
        self.__variable_errors = {}
        
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
            self.__ClearErrors()
    
    def __OnPipelineEvent(self,caller,event):
        if isinstance(event,CellProfiler.Pipeline.RunExceptionEvent):
            message = "Error while processing %s: %s\nDo you want to stop processing?"%(event.Module.ModuleName(),event.Error.message)
            if wx.MessageBox(message,"Pipeline error",wx.YES_NO | wx.ICON_ERROR,self.__frame) == wx.NO:
                event.CancelRun = False
            
    def __OnHelp(self,event):
        print "No help yet"
        
    def __OnAddModule(self,event):
        self.__add_module_frame.Show()
    
    def __GetSelectedModules(self):
        return self.__pipeline_list_view.GetSelectedModules()
    
    def __OnRemoveModule(self,event):
        selected_modules = self.__GetSelectedModules()
        for module in selected_modules:
            for variable in module.Variables():
                if self.__variable_errors.has_key(variable.Key()):
                    self.__frame.PreferencesView.PopErrorText(self.__variable_errors.pop(variable.Key()))                    
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
        self.__pipeline.AddModule(event.ModuleLoader(ModuleNum))
        
    def __OnModuleViewEvent(self,caller,event):
        assert isinstance(event,CellProfiler.CellProfilerGUI.ModuleView.VariableEditedEvent), '%s is not an instance of CellProfiler.CellProfilerGUI.ModuleView.VariableEditedEvent'%(str(event))
        variable = event.GetVariable()
        proposed_value = event.GetProposedValue()
        
        try:
            variable.SetValue(proposed_value)
            if self.__variable_errors.has_key(variable.Key()):
                self.__frame.PreferencesView.PopErrorText(self.__variable_errors.pop(variable.Key()))
            
        except ValueError, instance:
            if self.__variable_errors.has_key(variable.Key()):
                self.__frame.PreferencesView.PopErrorText(self.__variable_errors.pop(variable.Key()))
            message = "%s(%d): %s"%(variable.Module().ModuleName(),variable.VariableNumber(),instance.message)
            self.__frame.PreferencesView.SetErrorText(message)
            self.__variable_errors[variable.Key()] = message
            event.Cancel()
            
    def OnAnalyzeImages(self,event):
        if len(self.__variable_errors):
            wx.MessageBox("Please correct the errors in your pipeline before running.","Can't run pipeline",self.__frame)
            return
        output_path = self.GetOutputFilePath()
        if output_path:
            if self.__running_pipeline:
                self.__running_pipeline.close()
            self.__output_path = output_path
            self.__running_pipeline = self.__pipeline.ExperimentalRun(self.__frame)
            
    def OnIdle(self,event):
        if self.__running_pipeline:
            try:
                self.__pipeline_measurements = self.__running_pipeline.next()
                event.RequestMore()
            except StopIteration:
                self.__running_pipeline = None
                if self.__pipeline_measurements != None:
                    self.__pipeline.SaveMeasurements(self.__output_path,self.__pipeline_measurements)
                    self.__pipeline_measurements = None
                    self.__output_path = None


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
        
    def RunPipeline(self):
        """Run the current pipeline, returning the measurements
        """
        return self.__pipeline.Run(self.__frame)
    
    def SetMatlabPath(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'DataTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'ImageTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'CPsubfunctions'),matlab.path())
        matlab.path(CellProfiler.Preferences.ModuleDirectory(),matlab.path())
        matlab.path(CellProfiler.Preferences.CellProfilerRootDirectory(),matlab.path())
