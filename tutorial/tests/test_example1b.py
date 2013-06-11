from cStringIO import StringIO
import unittest

import cellprofiler.pipeline as cpp
from cellprofiler.modules import instantiate_module

class TestExample1b(unittest.TestCase):
    def make_instance(self):
        '''Return an instance of example1 this way because it's not on classpath'''
        return instantiate_module("Example1b")
    
    def test_01_01_off(self):
        module = self.make_instance()
        module.binary_setting.value = False
        settings = module.visible_settings()
        self.assertEqual(len(settings), 3)
        self.assertEqual(id(module.text_setting), id(settings[0]))
        self.assertEqual(id(module.choice_setting), id(settings[1]))
        self.assertEqual(id(module.binary_setting), id(settings[2]))
        
    def test_01_02_on(self):
        module = self.make_instance()
        module.binary_setting.value = True
        settings = module.visible_settings()
        self.assertEqual(len(settings), 5)
        self.assertEqual(id(module.text_setting), id(settings[0]))
        self.assertEqual(id(module.choice_setting), id(settings[1]))
        self.assertEqual(id(module.binary_setting), id(settings[2]))
        self.assertEqual(id(module.integer_setting), id(settings[3]))
        self.assertEqual(id(module.float_setting), id(settings[4]))

PIPELINE = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

Example1b:[module_num:1|svn_version:\'10581\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
"""