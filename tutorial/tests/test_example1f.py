from cStringIO import StringIO
import unittest

from cellprofiler.modules import instantiate_module
import cellprofiler.settings as cps
import cellprofiler.pipeline as cpp

class TestExample1f(unittest.TestCase):
    def setUp(self):
        #
        # Get the module and class using CellProfiler's loader
        #
        self.module = instantiate_module("Example1f")
        self.E = __import__(self.module.__class__.__module__)
        
    def test_00_00(self):
        self.assertEqual(self.E.S_GAUSSIAN, "Gaussian")
        
    def test_00_01(self):
        self.assertTrue(hasattr(self.module, "sigma"), 
                        "Example1f does not have a sigma setting")
        self.assertTrue(isinstance(self.module.sigma, cps.Float),
                        "The sigma setting should be cps.Float")
        
    def test_01_01_load_v1(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20130304190157
ModuleCount:1

Example1f:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Filter choice:Gassian
    Input image:CombinedGray
    Output image:Filtered
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.load(StringIO(data))
        module = pipeline.modules()[-1]
        self.assertEqual(module.filter_choice.value, self.E.S_GAUSSIAN)
        
    def test_01_02_load_v2(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20130304190157
ModuleCount:1

Example1f:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Filter choice:Gaussian
    Input image:CombinedGray
    Output image:Filtered
    Sigma:3
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.load(StringIO(data))
        module = pipeline.modules()[-1]
        self.assertEqual(module.filter_choice.value, self.E.S_GAUSSIAN)
        self.assertEqual(module.sigma, 3)
