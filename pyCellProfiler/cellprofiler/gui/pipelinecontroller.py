"""PipelineController.py - controls (modifies) a pipeline

"""
__version__="$Revision$"
import math
import numpy
import wx
import os
import re
import scipy.io.matlab.mio
import cpframe
import cellprofiler.pipeline
import cellprofiler.preferences
from cellprofiler.gui.addmoduleframe import AddModuleFrame
import cellprofiler.gui.moduleview
import cellprofiler.matlab.utils

class PipelineController:
    """Controls the pipeline through the UI
    
    """
    def __init__(self,pipeline,frame):
        self.__pipeline =pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        self.__frame = frame
        self.__add_module_frame = AddModuleFrame(frame,-1,"Add modules")
        self.__add_module_frame.add_listener(self.__on_add_to_pipeline)
        self.__variable_errors = {}
        self.__running_pipeline = None 
        self.__pipeline_measurements = None
        wx.EVT_MENU(frame,cpframe.ID_FILE_LOAD_PIPELINE,self.__on_load_pipeline)
        wx.EVT_MENU(frame,cpframe.ID_FILE_SAVE_PIPELINE,self.__on_save_pipeline)
        wx.EVT_MENU(frame,cpframe.ID_FILE_CLEAR_PIPELINE,self.__on_clear_pipeline)
        wx.EVT_MENU(frame,cpframe.ID_FILE_ANALYZE_IMAGES,self.on_analyze_images)
        wx.EVT_IDLE(frame,self.on_idle)
    
    def attach_to_pipeline_list_view(self,pipeline_list_view):
        """Glom onto events from the list box with all of the module names in it
        
        """
        self.__pipeline_list_view = pipeline_list_view
        
    def attach_to_module_view(self,module_view):
        """Listen for variable changes from the module view
        
        """
        self.__module_view = module_view
        module_view.add_listener(self.__on_module_view_event)
    
    def attach_to_directory_view(self,directory_view):
        """Listen for requests to load pipelines
        
        """
        self.__directory_view = directory_view
        directory_view.add_pipeline_listener(self.__on_dir_load_pipeline)
    
    def attach_to_module_controls_panel(self,module_controls_panel):
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
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_help, self.__help_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_add_module,self.__mcp_add_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_remove_module,self.__mcp_remove_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_module_up,self.__mcp_module_up_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_module_down,self.__mcp_module_down_button)
        
    def __on_load_pipeline(self,event):
        dlg = wx.FileDialog(self.__frame,"Choose a pipeline file to open",wildcard="*.mat")
        if dlg.ShowModal()==wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            self.__do_load_pipeline(pathname)
    
    def __on_dir_load_pipeline(self,caller,event):
        if wx.MessageBox('Do you want to load the pipeline, "%s"?'%(os.path.split(event.Path)[1]),
                         'Load path', wx.YES_NO|wx.ICON_QUESTION ,self.__frame) & wx.YES:
            self.__do_load_pipeline(event.Path)
    
    def __do_load_pipeline(self,pathname):
        try:
            handles=scipy.io.matlab.mio.loadmat(pathname, struct_as_record=True)
        except Exception,instance:
            self.__frame.display_error('Failed to open %s'%(pathname),instance)
            return
        try:
            if handles.has_key('handles'):
                self.__pipeline.create_from_handles(handles['handles'][0,0])
            else:
                self.__pipeline.create_from_handles(handles)
            self.__clear_errors()
        except Exception,instance:
            self.__frame.display_error('Failed during loading of %s'%(pathname),instance)

    def __clear_errors(self):
        for key,error in self.__variable_errors.iteritems():
            self.__frame.preferences_view.pop_error_text(error)
        self.__variable_errors = {}
        
    def __on_save_pipeline(self,event):
        dlg = wx.FileDialog(self.__frame,"Save pipeline",wildcard="*.mat",style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            handles = self.__pipeline.save_to_handles()
            scipy.io.matlab.mio.savemat(pathname,handles,format='5')
            
    def __on_clear_pipeline(self,event):
        if wx.MessageBox("Do you really want to remove all modules from the pipeline?",
                         "Clearing pipeline",
                         wx.YES_NO | wx.ICON_QUESTION, self.__frame) == wx.YES:
            self.__pipeline.Clear()
            self.__clear_errors()
    
    def __on_pipeline_event(self,caller,event):
        if isinstance(event,cellprofiler.pipeline.RunExceptionEvent):
            message = "Error while processing %s: %s\nDo you want to stop processing?"%(event.module.module_name,event.error.message)
            if wx.MessageBox(message,"Pipeline error",wx.YES_NO | wx.ICON_ERROR,self.__frame) == wx.NO:
                event.cancel_run = False
            
    def __on_help(self,event):
        print "No help yet"
        
    def __on_add_module(self,event):
        self.__add_module_frame.Show()
    
    def __get_selected_modules(self):
        return self.__pipeline_list_view.get_selected_modules()
    
    def __on_remove_module(self,event):
        selected_modules = self.__get_selected_modules()
        for module in selected_modules:
            for variable in module.variables():
                if self.__variable_errors.has_key(variable.key()):
                    self.__frame.preferencesview.pop_error_text(self.__variable_errors.pop(variable.key()))                    
            self.__pipeline.remove_module(module.module_num)
            
    def __on_module_up(self,event):
        selected_modules = self.__get_selected_modules()
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num,cellprofiler.pipeline.DIRECTION_UP);
        
    def __on_module_down(self,event):
        selected_modules = self.__get_selected_modules()
        selected_modules.reverse()
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num,cellprofiler.pipeline.DIRECTION_DOWN);
    
    def __on_add_to_pipeline(self,caller,event):
        selected_modules = self.__get_selected_modules()
        module_num = 1
        if len(selected_modules):
            module_num=selected_modules[-1].module_num+1
        self.__pipeline.add_module(event.module_loader(module_num))
        
    def __on_module_view_event(self,caller,event):
        assert isinstance(event,cellprofiler.gui.moduleview.VariableEditedEvent), '%s is not an instance of CellProfiler.CellProfilerGUI.ModuleView.VariableEditedEvent'%(str(event))
        variable = event.get_variable()
        proposed_value = event.get_proposed_value()
        
        try:
            variable.value = proposed_value
            if self.__variable_errors.has_key(variable.key()):
                self.__frame.preferences_view.pop_error_text(self.__variable_errors.pop(variable.key()))
            
        except ValueError, instance:
            if self.__variable_errors.has_key(variable.key()):
                self.__frame.preferences_view.pop_error_text(self.__variable_errors.pop(variable.key()))
            message = "%s(%d): %s"%(variable.module().module_name,variable.variable_number(),instance.message)
            self.__frame.preferences_view.set_error_text(message)
            self.__variable_errors[variable.key()] = message
            event.cancel()
            
    def on_analyze_images(self,event):
        if len(self.__variable_errors):
            wx.MessageBox("Please correct the errors in your pipeline before running.","Can't run pipeline",self.__frame)
            return
        output_path = self.get_output_file_path()
        if output_path:
            if self.__running_pipeline:
                self.__running_pipeline.close()
            self.__output_path = output_path
            self.__running_pipeline = self.__pipeline.experimental_run(self.__frame)
            
    def on_idle(self,event):
        if self.__running_pipeline:
            try:
                self.__pipeline_measurements = self.__running_pipeline.next()
                event.RequestMore()
            except StopIteration:
                self.__running_pipeline = None
                if self.__pipeline_measurements != None:
                    self.__pipeline.save_measurements(self.__output_path,self.__pipeline_measurements)
                    self.__pipeline_measurements = None
                    self.__output_path = None


    def get_output_file_path(self):
        path = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                            cellprofiler.preferences.get_output_file_name())
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
                cellprofiler.preferences.set_output_file_name(os.path.split(alternate_name)[1])
            else:
                return None
        return path
        
    def run_pipeline(self):
        """Run the current pipeline, returning the measurements
        """
        return self.__pipeline.Run(self.__frame)
    
    def set_matlab_path(self):
        matlab = cellprofiler.matlab.utils.get_matlab_instance()
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'DataTools'),matlab.path())
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'ImageTools'),matlab.path())
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'CPsubfunctions'),matlab.path())
        matlab.path(cellprofiler.preferences.module_directory(),matlab.path())
        matlab.path(cellprofiler.preferences.cell_profiler_root_directory(),matlab.path())
