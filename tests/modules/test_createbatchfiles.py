'''test_createbatchfiles - test the CreateBatchFiles module
'''

import base64
import os
import sys
import tempfile
import unittest
import zlib
from StringIO import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps

import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.createbatchfiles as C
import tests.modules as T


class TestCreateBatchFiles(unittest.TestCase):
    def test_01_00_test_load_version_9_please(self):
        self.assertEqual(C.CreateBatchFiles.variable_revision_number, 8)

    def test_01_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150713184605
GitHash:2f7b3b9
ModuleCount:1
HasImagePlaneDetails:False

CreateBatchFiles:[module_num:19|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Store batch files in default output folder?:Yes
    Output folder path:C\x3A\\\\foo\\\\bar
    Are the cluster computers running Windows?:No
    Hidden\x3A in batch mode:No
    Hidden\x3A in distributed mode:No
    Hidden\x3A default input folder at time of save:C\x3A\\\\bar\\\\baz
    Hidden\x3A revision number:0
    Hidden\x3A from old matlab:No
    Launch BatchProfiler:Yes
    Local root path:\\\\\\\\argon-cifs\\\\imaging_docs
    Cluster root path:/imaging/docs
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, C.CreateBatchFiles)
        self.assertTrue(module.wants_default_output_directory)
        self.assertEqual(module.custom_output_directory, r"C:\foo\bar")
        self.assertFalse(module.remote_host_is_windows)
        self.assertFalse(module.distributed_mode)
        self.assertEqual(module.default_image_directory, r"C:\bar\baz")
        self.assertEqual(module.revision, 0)
        self.assertFalse(module.from_old_matlab)
        self.assertEqual(len(module.mappings), 1)
        mapping = module.mappings[0]
        self.assertEqual(mapping.local_directory, r"\\argon-cifs\imaging_docs")
        self.assertEqual(mapping.remote_directory, r"/imaging/docs")

    def test_01_08_load_v8(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150713184605
GitHash:2f7b3b9
ModuleCount:1
HasImagePlaneDetails:False

CreateBatchFiles:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Store batch files in default output folder?:Yes
    Output folder path:/Users/cellprofiler
    Are the cluster computers running Windows?:No
    Hidden\x3A in batch mode:No
    Hidden\x3A in distributed mode:No
    Hidden\x3A default input folder at time of save:/Users/mcquin/Pictures
    Hidden\x3A revision number:0
    Hidden\x3A from old matlab:No
    Local root path:/Users/cellprofiler/Pictures
    Cluster root path:/Remote/cellprofiler/Pictures
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert module.wants_default_output_directory.value
        assert module.custom_output_directory.value == "/Users/cellprofiler"
        assert not module.remote_host_is_windows.value
        assert module.mappings[0].local_directory.value == "/Users/cellprofiler/Pictures"
        assert module.mappings[0].remote_directory.value == "/Remote/cellprofiler/Pictures"

    def test_02_01_module_must_be_last(self):
        '''Make sure that the pipeline is invalid if CreateBatchFiles is not last'''
        #
        # First, make sure that a naked CPModule tests valid
        #
        pipeline = cpp.Pipeline()
        module = cpm.Module()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        pipeline.test_valid()
        #
        # Make sure that CreateBatchFiles on its own tests valid
        #
        pipeline = cpp.Pipeline()
        module = C.CreateBatchFiles()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        pipeline.test_valid()

        module = cpm.Module()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        self.assertRaises(cps.ValidationError, pipeline.test_valid)

    def test_03_01_save_and_load(self):
        '''Save a pipeline to batch data, open it to check and load it'''
        data = ('eJztWW1PGkEQXhC1WtPYTzb9tB+llROoGiWNgi9NSYUSIbZGbbvCApvu7ZJ7'
                'UWlj0o/9Wf1J/QndxTs4tsoBRS3JHbkcMzfPPDOzs8uxl8uU9jPbcFWLw1ym'
                'FKsSimGBIqvKDT0FmbUEdwyMLFyBnKVgycYwY9dgIgET8dTqRmolCZPx+AYY'
                '7ghlc0/EJf4cgClxfSTOsHNr0pFDnlPKRWxZhNXMSRABzxz9L3EeIoOgM4oP'
                'EbWx2aFw9VlW5aVmo30rxys2xXmke43Fkbf1M2yY76su0LldIJeYFsk3rKTg'
                'mh3gc2ISzhy841/Vtnm5pfDKOhTmOnUIKXWQdVnw6KX9W9Cxj9xQt6ce+3lH'
                'JqxCzknFRhQSHdXaUbTGwcffRJe/CXAk0BKX9sHNK3HIs4QvrdjeJSpbUEdW'
                'uS79rPv4mVb8SLmcOrGw3ugr/lAXPgRe9Zl3uAsXBnkO+sp7VolXyrscMm5B'
                '28T91/02/lHgphSce7i4GdCJ0y/fOSVfKe9RE1/UsYE1TXP91H38rCh+pCzG'
                'uYwpbRhcLlHGiWXY7OuJaCC9ISZ3q5NdqbhdzLZb+yH4hpmXy3I2ioVtGTFE'
                'myYZxbwcdpzvu68eku8+cGmf/GZAdz9IeaeOGMM0ERtx3P30z24+c6d86jqc'
                'uOP8Il18EdE/DP8L3w8fvnegezyl/Glxq/BaPljhTe1l9LOUPoj15YBfbB5n'
                'YoXTqKvZ4dTW2eZxPLZx+j2xlLy6Ni4SgWwpozfmPUj8fuvhuhK/lGUMRxgZ'
                'TmArV9GYVOU4s+qOLunodlGzo3mgeZMcxbwZir9p8QZFpj4C/kHnUfKO+YJ5'
                'NJ7z6OPsYP8rxuV3NcAFuAAX4P43XNqD63c/pLUZUzO43YCEVXBjnPINcOON'
                'S4OgXwPc8DipvO35Ut2/kfZfQO9+ewG6+03K3s04TW9topsa5ahyvYut7Yuv'
                'Wc+Gdj/P52sKz9ptPOXWK5AzuU8tb5ja9TuRbal4Q1rvCNT6zdzA561DWHwW'
                'pnvXXa13Zxx+bw3DFwn9zffYBxdxKidxP8Fg47zYw97NbVj7P/nFW+E=')

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        T.maybe_download_sbs()
        for windows_mode in ((False, True) if sys.platform.startswith("win")
                             else (False,)):
            pipeline = cpp.Pipeline()
            pipeline.add_listener(callback)
            pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
            ipath = os.path.join(T.example_images_directory(), 'ExampleSBSImages')
            bpath = tempfile.mkdtemp()
            bfile = os.path.join(bpath, C.F_BATCH_DATA)
            hfile = os.path.join(bpath, C.F_BATCH_DATA_H5)
            try:
                li = pipeline.modules()[0]
                self.assertTrue(isinstance(li, LI.LoadImages))
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                li.location.dir_choice = cps.ABSOLUTE_FOLDER_NAME
                li.location.custom_path = ipath
                module.wants_default_output_directory.value = False
                module.custom_output_directory.value = bpath
                module.remote_host_is_windows.value = windows_mode
                self.assertEqual(len(module.mappings), 1)
                mapping = module.mappings[0]
                mapping.local_directory.value = ipath
                self.assertFalse(pipeline.in_batch_mode())
                measurements = cpmeas.Measurements(mode="memory")
                image_set_list = cpi.ImageSetList()
                result = pipeline.prepare_run(
                        cpw.Workspace(pipeline, None, None, None,
                                      measurements, image_set_list))
                self.assertFalse(pipeline.in_batch_mode())
                self.assertFalse(result)
                self.assertFalse(module.batch_mode.value)
                self.assertTrue(measurements.has_feature(
                        cpmeas.EXPERIMENT, cpp.M_PIPELINE))
                pipeline = cpp.Pipeline()
                pipeline.add_listener(callback)
                image_set_list = cpi.ImageSetList()
                measurements = cpmeas.Measurements(mode="memory")
                workspace = cpw.Workspace(pipeline, None, None, None,
                                          cpmeas.Measurements(),
                                          image_set_list)
                workspace.load(hfile, True)
                measurements = workspace.measurements
                self.assertTrue(pipeline.in_batch_mode())
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                self.assertTrue(module.batch_mode.value)
                image_numbers = measurements.get_image_numbers()
                self.assertTrue([x == i + 1 for i, x in enumerate(image_numbers)])
                pipeline.prepare_run(workspace)
                pipeline.prepare_group(workspace, {}, range(1, 97))
                for i in range(96):
                    image_set = image_set_list.get_image_set(i)
                    for image_name in ('DNA', 'Cytoplasm'):
                        pathname = measurements.get_measurement(
                                cpmeas.IMAGE, "PathName_" + image_name, i + 1)
                        self.assertEqual(pathname,
                                         '\\imaging\\analysis' if windows_mode
                                         else '/imaging/analysis')
                measurements.close()
            finally:
                if os.path.exists(bfile):
                    os.unlink(bfile)
                if os.path.exists(hfile):
                    os.unlink(hfile)
                os.rmdir(bpath)

    def test_04_01_alter_path(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        self.assertEqual(module.alter_path("foo/bar"), "bar/bar")
        self.assertEqual(module.alter_path("baz/bar"), "baz/bar")

    def test_04_02_alter_path_regexp(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        self.assertEqual(
                module.alter_path("foo/bar", regexp_substitution=True), "bar/bar")
        self.assertEqual(
                module.alter_path("baz/bar", regexp_substitution=True), "baz/bar")

        module.mappings[0].local_directory.value = r"\foo\baz"
        module.remote_host_is_windows.value = True
        self.assertEqual(
                module.alter_path(r"\\foo\\baz\\bar", regexp_substitution=True),
                r"bar\\bar")

    if sys.platform == 'win32':
        def test_04_03_alter_path_windows(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "\\foo"
            module.mappings[0].remote_directory.value = "\\bar"

            self.assertEqual(
                    module.alter_path("\\foo\\bar"), "/bar/bar")
            self.assertEqual(
                    module.alter_path("\\FOO\\bar"), "/bar/bar")
            self.assertEqual(
                    module.alter_path("\\baz\\bar"), "/baz/bar")

        def test_04_04_alter_path_windows_regexp(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "foo"
            module.mappings[0].remote_directory.value = "bar"

            self.assertEqual(
                    module.alter_path("\\\\foo\\\\bar", regexp_substitution=True),
                    "/foo/bar")
            self.assertEqual(
                    module.alter_path("\\\\foo\\g<bar>", regexp_substitution=True),
                    "/foo\\g<bar>")
