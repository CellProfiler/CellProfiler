import unittest
from cStringIO import StringIO

import cellprofiler.pipeline as cpp


class TestExample1(unittest.TestCase):
    def test_01_01_load(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

Example1:[module_num:1|svn_version:\'10581\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(
                isinstance(event, cpp.LoadExceptionEvent),
                "Either you have settings in your module or, more likely, you need to set your plugins directory to the tutorial.")
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(module.module_name, "Example1")
        self.assertEqual(len(module.settings()), 0)

