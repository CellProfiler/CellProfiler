"""test_nowx.py - ensure that there's no dependency on wx when headless
"""
import builtins
import sys
import tempfile
import unittest
from urllib.request import urlopen

from tests.frontend.modules import example_images_directory, get_test_resources_directory


def import_all_but_wx(
    name,
    globals=builtins.globals(),
    locals=builtins.locals(),
    fromlist=[],
    level=0,
    default_import=builtins.__import__,
):
    if name == "wx" or name.startswith("wx."):
        raise ImportError("Not allowed to import wx!")
    return default_import(name, globals, locals, fromlist, level)


@unittest.skipIf(sys.platform.startswith("linux"), "Do not test under Linux")
class TestNoWX(unittest.TestCase):
    def setUp(self):
        from cellprofiler_core.preferences import set_temporary_directory

        set_temporary_directory(tempfile.gettempdir())
        self.old_import = builtins.__import__
        builtins.__import__ = import_all_but_wx

    def tearDown(self):
        builtins.__import__ = self.old_import

    def example_dir(self):
        return example_images_directory()

    def test_01_01_can_import(self):
        import os

        self.assertTrue(hasattr(os, "environ"))

    # FIXME: wxPython 4 PR
    # def test_01_02_throws_on_wx_import(self):
    #     def import_wx():
    #         pass
    #
    #     self.assertRaises(ImportError, import_wx)

    def test_01_03_import_modules(self):
        """Import cellprofiler.modules and make sure it doesn't import wx"""

    # def test_01_04_instantiate_all(self):
    #     '''Instantiate each module and make sure none import wx'''
    #     import cellprofiler.modules as M
    #     for name in M.get_module_names():
    #         try:
    #             M.instantiate_module(name)
    #         except:
    #             print "Module %s probably imports wx" % name
    #             traceback.print_exc()

    fly_url = "http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cppipe"

    def test_01_05_load_pipeline(self):
        import cellprofiler_core.pipeline as cpp
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.event.LoadException))

        pipeline = cpp.Pipeline()
        pipeline.add_listener(callback)
        fly_pipe = get_test_resources_directory("../ExampleFlyURL.cppipe")
        pipeline.load(fly_pipe)

    def test_01_06_run_pipeline(self):
        import cellprofiler_core.pipeline as cpp
        import cellprofiler_core.module as cpm

        def callback(caller, event):
            self.assertFalse(isinstance(event, (cpp.event.LoadException,
                                                cpp.event.RunException)))
        pipeline = cpp.Pipeline()
        pipeline.add_listener(callback)
        fly_pipe = get_test_resources_directory("../ExampleFlyURL.cppipe")
        pipeline.load(fly_pipe)
        while True:
            removed_something = False
            for module in reversed(pipeline.modules()):
                self.assertTrue(isinstance(module, cpm.Module))
                if module.module_name in ("SaveImages",
                                          "CalculateStatistics",
                                          "ExportToSpreadsheet",
                                          "ExportToDatabase"):
                    pipeline.remove_module(module.module_num)
                    removed_something = True
                    break
            if not removed_something:
                break
        for module in pipeline.modules():
            module.show_window = False
        m = pipeline.run(image_set_end=1)
