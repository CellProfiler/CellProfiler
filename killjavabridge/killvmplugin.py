'''killvmplugin.py - a nose plugin that kills the javabridge on finalize

'''

__version__="$Revision$"

from nose.plugins import Plugin
import os
import numpy as np
np.seterr(all='ignore')
import sys

class KillVMPlugin(Plugin):
    '''CellProfiler Javabridge nose test plugin
    
    This plugin is meant to be run with the CellProfiler unit tests using
    Nose. Its purpose is to control the Javabridge environment during tests,
    starting and stopping it as necessary.
    '''
    enabled = False
    name = "kill-vm"

    def options(self, parser, env=os.environ):
        self.env = env
        Plugin.options(self, parser, env)
        
    def configure(self, options, conf):
        Plugin.configure(self, options, conf)
        if self.env.has_key("NOSE_WITH_KILL_VM"):
            self.enabled = True
    
    def begin(self):
        if self.enabled:
            import wx
            self.app = wx.GetApp()
            if self.app is None:
                self.app = wx.PySimpleApp(False)
        
    def prepareTestRunner(self, testRunner):
        '''Need to make the test runner call finalize if in Wing
        
        Wing IDE's XML test runner fails to call finalize, so we
        wrap it and add that function here
        '''
        if (getattr(testRunner, "__module__","unknown") == 
            "wingtest_common"):
            outer_self = self
            class TestRunnerProxy(object):
                def run(self, test):
                    result = testRunner.run(test)
                    outer_self.finalize(testRunner.result)
                    return result
                
                @property
                def result(self):
                    return testRunner.result
            return TestRunnerProxy()
            
    def startContext(self, context):
        try:
            import cellprofiler.utilities
            import cellprofiler.cpmath
    
            if context in (cellprofiler.utilities, cellprofiler.cpmath):
                del context.setup
        except:
            pass
        
    def finalize(self, result):
        import wx
        if self.enabled:
            from cellprofiler.utilities.jutil import kill_vm
            kill_vm()
            os._exit(0)
