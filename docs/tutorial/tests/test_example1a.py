from cStringIO import StringIO
import re
import sys
import unittest

import cellprofiler.pipeline as cpp
from cellprofiler.modules import instantiate_module

class TestExample1a(unittest.TestCase):
    def test_01_01_load(self):
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(
                isinstance(event, cpp.LoadExceptionEvent),
                "Either you have settings in your module or, more likely, you need to set your plugins directory to the tutorial.")
        pipeline.add_listener(callback)
        pipeline.load(StringIO(PIPELINE))
        module = pipeline.modules()[-1]
        self.assertTrue(module.module_name, "Example1a")
        self.assertEqual(len(module.settings()), 5)

    def make_instance(self):
        '''Return an instance of example1 this way because it's not on classpath'''
        return instantiate_module("Example1a")
    
    def test_02_01_create_settings(self):
        module = self.make_instance()
        self.assertTrue(hasattr(module, "text_setting"), 
                        "Couldn't find self.text_setting")
        self.assertEqual(module.text_setting, "suggested value")
        self.assertEqual(module.choice_setting, "Choice 1")
        self.assertFalse(module.binary_setting)
        self.assertEqual(module.integer_setting, 15)
        self.assertEqual(module.float_setting, 1.5)
        
    def test_02_02_settings(self):
        module = self.make_instance()
        settings = module.settings()
        self.assertEqual(len(settings), 5)
        self.assertEqual(id(settings[0]), id(module.text_setting))
        self.assertEqual(id(settings[1]), id(module.choice_setting))
        self.assertEqual(id(settings[2]), id(module.binary_setting))
        self.assertEqual(id(settings[3]), id(module.integer_setting))
        self.assertEqual(id(settings[4]), id(module.float_setting))
        
    def test_02_03_run(self):
        module = self.make_instance()
        module.integer_setting.value = 3
        module.float_setting.value = 4.5
        fd = StringIO()
        old_stdout = sys.stdout
        sys.stdout = fd
        try:
            module.run(None)
        finally:
            sys.stdout = old_stdout
        result = fd.getvalue()
        pattern = r"([0-9]+)\s*\+\s*([0-9.]+)\s*=\s*([0-9.]+)"
        match = re.search(pattern, result)
        self.assertFalse(match is None)
        fields = tuple([float(x) for x in match.groups()])
        self.assertEqual(fields, (3, 4.5, 7.5))
        

PIPELINE = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

Example1a:[module_num:1|svn_version:\'10581\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
"""