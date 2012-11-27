'''killvmplugin.py - a nose plugin that kills the javabridge on finalize

'''

__version__="$Revision$"

from nose.plugins import Plugin
import os
import numpy as np
np.seterr(all='ignore')
import sys

class KillVMPlugin(Plugin):
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
        self.ran_jvm_hook = False
        self.has_jvm = False
        
    def beforeImport(self, filename, module):
        if module == "bioformats" and not self.ran_jvm_hook:
            sys.stderr.write("Preparing to import bioformats.\n")
            import wx
            self.app = wx.GetApp()
            if self.app is None:
                self.app = wx.PySimpleApp(False)
            self.ran_jvm_hook = True
        
    def afterImport(self, filename, module):
        if module == "bioformats":
            self.has_jvm = True
        
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
        if self.has_jvm:
            from cellprofiler.utilities.jutil import kill_vm
            import sys
            kill_vm()
            import wx
            self.app.Exit()
            self.app.Destroy()
            del self.app
