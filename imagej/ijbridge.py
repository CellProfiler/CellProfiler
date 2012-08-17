"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

'''
ijbridge provides a high-level interface to ImageJ so in-process and inter-
process execution of ImageJ can be abstracted and branched cleanly.
'''
import sys
import logging
import numpy as np
from subimager.client import make_imagej_request
import subimager.imagejrequest as ijrq
from cellprofiler.utilities.singleton import Singleton

logger = logging.getLogger(__name__)

'''The image ID for image value parameters for single-image calls'''
IMAGE_ID = "Image1"

'''The ImageJ plugin that sets the active display for us'''
SET_ACTIVE_DISPLAY_CLASS = "org.cellprofiler.subimager.imagej.SetActiveDisplay"

'''The ImageJ plugin that gets the active display'''
GET_ACTIVE_DISPLAY_CLASS = "org.cellprofiler.subimager.imagej.GetActiveDisplay"

def get_ij_bridge():
    '''Returns an an ijbridge that will work given the platform and preferences
    '''
    return inter_proc_ij_bridge.getInstance()


class ij_bridge(object):
    '''This class provides a high-level interface for running ImageJ from
    CellProfiler. It is intended to abstract away whether IJ is being run within
    the same process or in a separate process.
    '''   
    def inject_image(self, pixels, name=None):
        '''inject an image into ImageJ for processing'''
        raise NotImplementedError

    def get_current_image(self):
        '''returns the WindowManager's current image as a numpy float array'''
        raise NotImplementedError

    def get_commands(self):
        '''returns a list of the available command strings'''
        raise NotImplementedError

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        raise NotImplementedError

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ

        macro_text - the macro program to be run
        '''
        raise NotImplementedError

    def show_imagej(self):
        '''show the ImageJ user interface'''
        raise NotImplementedError

class inter_proc_ij_bridge(ij_bridge, Singleton):
    '''Interface for running ImageJ in a separate process from CellProfiler.
    '''
    def __init__(self):
        self.context_id = None
        self.modules = None
        self.start_ij()

    def __del__(self):
        '''call del on this object to close ImageJ and the TCPClient.'''
        self.quit()
        
    def start_ij(self):
        request = ijrq.RequestType(
            CreateContext = ijrq.CreateContextRequestType())
        response, _ = make_imagej_request(request, {})
        response = ijrq.parseString(response)
        if not isinstance(response, ijrq.ResponseType):
            raise ValueError("Failed to get a valid ImageJ context from subimager")
        self.context_id = response.CreateContextResponse.ContextID.ContextID
        
    def inject_image(self, pixels, name=None):
        '''inject an image into ImageJ for processing'''
        axis = [ "X", "Y" ]
        if pixels.ndim == 3:
            axis.append("CHANNEL")
        image_value = ijrq.ImageDisplayParameterValueType(
            ImageName = "Untitled" if name is None else name,
            ImageID = IMAGE_ID,
            Axis = axis)
        modules = [m for m in self.get_commands()
                   if m.Name == "Set Active Display"]
        assert len(modules) == 1
        
        rmr = ijrq.RunModuleRequestType(
            ContextID = self.context_id,
            ModuleID = modules[0].ModuleID)
        rmr.add_Parameter(ijrq.ParameterValueType(
            Name = "display",
            ImageValue = image_value))
        request = ijrq.RequestType(RunModule = rmr)
        response, image_dict = make_imagej_request(
            request, { IMAGE_ID: pixels })
        response = ijrq.parseString(response)
        exception = response.RunModuleResponse.Exception
        if (exception != None):
            raise ValueError(exception.Message)

    def get_current_image(self):
        '''Returns the active image or None if no active image'''
        modules = [m for m in self.get_commands()
                   if m.Name == "Get Active Display"]
        assert len(modules) == 1
        request = ijrq.RequestType(
            RunModule = ijrq.RunModuleRequestType(
                ContextID = self.context_id,
                ModuleID = modules[0].ModuleID))
        response, image_dict = make_imagej_request(request, {})
        response = ijrq.parseString(response)
        exception = response.RunModuleResponse.Exception
        if (exception != None):
            raise ValueError(exception.Message)
        rmr_response = response.RunModuleResponse
        assert isinstance(rmr_response, ijrq.RunModuleResponseType)
        active_display_params = [p for p in rmr_response.Parameter
                                 if p.Name == "activeDisplay"]
        if len(active_display_params) == 0:
            raise ValueError("No active display parameter found in response to GetActiveDisplay call")
        active_display_param = active_display_params[0]
        assert isinstance(active_display_param, ijrq.ParameterValueType)
        if active_display_param.ImageValue is None:
            return None
        image_id = active_display_param.ImageValue.ImageID
        return image_dict[image_id]

    def get_commands(self):
        '''returns the modules that can be run'''
        if self.modules is None:
            request = ijrq.RequestType(
                GetModules = ijrq.GetModulesRequestType(self.context_id))
            response, _ = make_imagej_request(request, {})
            response = ijrq.parseString(response)
            assert isinstance(response, ijrq.ResponseType)
            response = response.GetModulesResponse
            assert isinstance(response, ijrq.GetModulesResponseType)
            self.modules = response.Module
        return self.modules

    def execute_command(self, command, options=None):
        '''execute the named command within ImageJ'''
        raise NotImplementedError("There is no macro facility yet in ImageJ 2.0")

    def execute_macro(self, macro_text):
        '''execute a macro in ImageJ
        macro_text - the macro program to be run
        '''
        raise NotImplementedError("There is no macro facility yet in ImageJ 2.0")
    
    def show_imagej(self):
        '''show the ImageJ user interface'''
        pass

    def quit(self):
        '''close the java process'''
        if self.context_id is not None:
            request = ijrq.RequestType(
                DestroyContext = ijrq.DestroyContextRequestType(
                    ContextID = self.context_id))
            make_imagej_request(request, {})
            self.context_id = None

if __name__ == '__main__':
    import wx
    from time import time
    from subimager.client import start_subimager, stop_subimager

    start_subimager()
    app = wx.PySimpleApp()
    PIXELS = np.tile(np.linspace(0,200,200), 300).reshape((300,200)).T
    PIXELS[50:150,100:200] = 0.0
    PIXELS[100, 150] = 1.0
    ipb = inter_proc_ij_bridge.getInstance()

    f = wx.Frame(None)
    b1 = wx.Button(f, -1, 'inject')
    b2 = wx.Button(f, -1, 'get')
    b3 = wx.Button(f, -1, 'get cmds')
    b4 = wx.Button(f, -1, 'cmd:add noise')
    b5 = wx.Button(f, -1, 'macro:invert')
    b6 = wx.Button(f, -1, 'quit')
    f.SetSizer(wx.BoxSizer(wx.VERTICAL))
    f.Sizer.Add(b1)
    f.Sizer.Add(b2)
    f.Sizer.Add(b3)
    f.Sizer.Add(b4)
    f.Sizer.Add(b5)
    f.Sizer.Add(b6)

    def on_getcmds(evt):
        print ipb.get_commands()

    b1.Bind(wx.EVT_BUTTON, lambda(x): ipb.inject_image(PIXELS, 'name ignored'))
    b2.Bind(wx.EVT_BUTTON, lambda(x): ipb.get_current_image())
    b3.Bind(wx.EVT_BUTTON, on_getcmds)
    b4.Bind(wx.EVT_BUTTON, lambda(x): ipb.execute_command('Add Noise'))
    b5.Bind(wx.EVT_BUTTON, lambda(x): ipb.execute_macro('run("Invert");'))
    b6.Bind(wx.EVT_BUTTON, lambda(x): ipb.quit())
    f.Show()

    app.MainLoop()
    
    ipb.quit()
    stop_subimager()

