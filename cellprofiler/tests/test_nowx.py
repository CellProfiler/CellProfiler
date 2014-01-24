"""test_nowx.py - ensure that there's no dependency on wx when headless

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import tempfile
import traceback
import unittest
from cellprofiler.modules.tests import example_images_directory

import __builtin__

def import_all_but_wx(name, 
                      globals = __builtin__.globals(), 
                      locals = __builtin__.locals(),
                      fromlist=[], level=-1,
                      default_import = __builtin__.__import__):
    if name == "wx" or name.startswith("wx."):
        raise ImportError("Not allowed to import wx!")
    return default_import(name, globals, locals, fromlist, level)

class TestNoWX(unittest.TestCase):
    def setUp(self):
        from cellprofiler.preferences import set_headless, set_temporary_directory
        set_headless()
        set_temporary_directory(tempfile.gettempdir())
        self.old_import = __builtin__.__import__
        __builtin__.__import__ = import_all_but_wx

    def tearDown(self):
        __builtin__.__import__ = self.old_import

    def example_dir(self):
        return example_images_directory()
    
    def test_01_01_can_import(self):
        import os
        self.assertTrue(hasattr(os, "environ"))
        
    def test_01_02_throws_on_wx_import(self):
        def import_wx():
            import wx
        self.assertRaises(ImportError, import_wx)
        
    def test_01_03_import_modules(self):
        '''Import cellprofiler.modules and make sure it doesn't import wx'''
        import cellprofiler.modules
        
    def test_01_04_instantiate_all(self):
        '''Instantiate each module and make sure none import wx'''
        import cellprofiler.modules as M
        for name in M.get_module_names():
            try:
                M.instantiate_module(name)
            except:
                print "Module %s probably imports wx" % name
                traceback.print_exc()
                
    def test_01_05_load_pipeline(self):
        import cellprofiler.pipeline as cpp
        import os
        pipeline_file = os.path.join(self.example_dir(), 
                                     "ExampleSBSImages", "ExampleSBS.cp")
        self.assertTrue(os.path.isfile(pipeline_file), "No ExampleSBS.cp file")
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(pipeline_file)
        
    def test_01_06_run_pipeline(self):
        import cellprofiler.pipeline as cpp
        import cellprofiler.cpmodule as cpm
        import os
        from cellprofiler.preferences import set_default_image_directory, set_default_output_directory
        example_fly_dir = os.path.join(self.example_dir(), 
                                       "ExampleFlyImages")
        set_default_image_directory(example_fly_dir)
        output_dir = tempfile.mkdtemp()
        set_default_output_directory(output_dir)
        try:
            pipeline_file = os.path.join(example_fly_dir, "ExampleFlyURL.cppipe")
            if not os.path.exists(pipeline_file):
                pipeline_file = os.path.join(example_fly_dir, "ExampleFly.cp")
            pipeline = cpp.Pipeline()
            def callback(caller, event):
                self.assertFalse(isinstance(event, (cpp.LoadExceptionEvent,
                                                    cpp.RunExceptionEvent)))
            pipeline.add_listener(callback)
            pipeline.load(pipeline_file)
            while True:
                removed_something = False
                for module in reversed(pipeline.modules()):
                    self.assertTrue(isinstance(module, cpm.CPModule))
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
            m = pipeline.run(image_set_end = 1)
            del m
        finally:
            for file_name in os.listdir(output_dir):
                try:
                    os.remove(os.path.join(output_dir, file_name))
                except Exception, e:
                    print "Failed to remove %s" % os.path.join(output_dir, file_name), e
                    traceback.print_exc()
            try:
                os.rmdir(output_dir)
            except:
                print "Failed to remove %s" % output_dir
                traceback.print_exc()
        
