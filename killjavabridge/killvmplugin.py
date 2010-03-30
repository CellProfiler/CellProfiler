'''killvmplugin.py - a nose plugin that kills the javabridge on finalize

'''

__version__="$Revision$"

from nose.plugins import Plugin
import os

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
        try:
            from cellprofiler.utilities.jutil import kill_vm
            kill_vm()
        except:
            pass
