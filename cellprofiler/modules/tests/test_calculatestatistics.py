"""test_calculatestatistics.py - test the CalculateStatistics module
"""

import base64
import os
import tempfile
import unittest
import zlib
from StringIO import StringIO

import numpy

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.measurement
import cellprofiler.image
import cellprofiler.pipeline
import cellprofiler.region
import cellprofiler.workspace
import cellprofiler.preferences

import cellprofiler.modules.calculatestatistics

INPUT_OBJECTS = "my_object"
TEST_FTR = "my_measurement"
FIGURE_NAME = "figname"


class TestCalculateStatistics(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwyy9TMDBTMLCwMjKzMrZQMDIwsFQgGTAwevryMzAwZDEy'
                'MFTM2Rp00Ouwg8DcJZmhE7Y9WMru2D+v654Nl5ZGhtayL/t81VIVz057tyzE'
                'X+6vSn9d2/zH6/IunY3yjSqYljjz5vd7PMa32Rr61Ry87+2+FPtj27I+2cOh'
                'MgsETOZ+UKu4cPln+Utlo+Upcx50qEgWaeicv7/MUNrubNfyarFUzl8pWU1L'
                'f94s6Sm26yheVFe3wNuH6VdPwDaZbx3cpzS9TeNf3765SE847O62L+LppuzH'
                'ZzXP8j58QvuP5ZoP92btfCvxY11f8rFfivE1H/21f8R1vj8o+vjHfJvtR4oP'
                'pges9l9+pHgB/9cK+18yLNe7rt8u4P9ZGxiYz3/qd0ChC8vMM5Fmtx2WJ78s'
                '8ZmbPMvNQvtraPj27Jday4ONHJ0jXu45+P7zxu8Rz+fPsoqVvW8oxFuoKfJ/'
                'WVN9wsfQGs9JBt4dBxwEHHcuUNT/ZfjfdfWi+vB5bsm76yu2pdqW++7VmqfO'
                'L/dWXuueds62+ZPtP+zM7f7Pstg39TAAo3nSDA==')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.calculatestatistics.CalculateStatistics))
        self.assertEqual(module.grouping_values.value, "Dose")
        self.assertEqual(len(module.dose_values), 1)
        self.assertEqual(module.dose_values[0].measurement, "Dose")
        self.assertFalse(module.dose_values[0].log_transform)
        self.assertTrue(module.dose_values[0].wants_save_figure)
        self.assertEqual(module.dose_values[0].figure_name, "DOSE")
        self.assertEqual(module.dose_values[0].pathname.dir_choice,
                         cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME)

    def test_01_02_load_v1(self):
        data = ('eJztWNFu2jAUdWhA7SYmtJfx6KdpD12Wom5qUaUN6KohAasW1G5PlZuYNpKD'
                'Uewwui/Y5+1xn9FPmN0mJHGhIQX2RFCU3Ot7zrn32kpiuo1+p9GE7w0Tdhv9'
                'twOXYHhKEB9Q36vDId+FLR8jjh1Ih3V44ruwR8fQ/ADNg/q+Wa8dwJppHoKn'
                'HVq7+0Jc/lQAKInrtjgL4VAxtLXEKW0Lc+4Or1gR6KAa+m/FeYZ8F10SfIZI'
                'gFksEfnbwwHt34ymQ13qBAT3kJcMFkcv8C6xz74OImA4fOpOMLHcX1gpIQr7'
                'hscuc+kwxIf8qneqS7mia13Tnye+SEfhbyJuX1tczEDaL/s2KcV905S+yT6W'
                'E34Z/wXE8fqMPr9KxFdC+xgPUEA4bHvoCsNj18c2p/7NQnwvFT5pdzFHDuLo'
                'wmpaFw5lURskn5nBp6X4NGAsWFfePA4y+LYVPmnvmfuHhs3Gi9RRSOELoEcX'
                'q38rhdsCP0TSy+itej0k8ygpfNER8e2AuN/rmvcqSOdfTeRPAz4KOHSiAtbZ'
                '/3n559Xb263d4T5l4MogXbe0p+v9HBOyoP461+kq18f/1tNTejo4/9zpLKOX'
                '9bwpgfR8SrsVME69h/muW9d4J6u9x/8t5nv/rGI+NrgNboPb4Da49eOkc97z'
                'Xf0+kPHfEzqz3ievE/GV0LbF58jIp3Lf6Rve3eaIGYQih+MJNzripi9u7vlH'
                'GfxHCv/RPH4bETsQm1zMxMbKZdy1mdGKfNbUN6t/OzN0k30oiN+z8uN9V/sd'
                'z8Ptx6foFbSHes8zcHrYQYn7DfLN85tH4qPalonPW7+mLV9HrKNPcwIg/j8i'
                'b/w/eRPZpA==')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        #
        # The CalculateStatistics module has the following settings:
        #
        # grouping measurement: Metadata_SBS_doses
        # dose_values[0]:
        #    measurement: Metadata_SBS_doses
        #    log transform: False
        #    wants_save_figure: False
        #
        # dose_values[1]:
        #    measurement: Metadata_Well
        #    log transform: True
        #    wants_save_figure: True
        #    pathname_choice: PC_DEFAULT
        #    pathname: ./WELL
        #
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.calculatestatistics.CalculateStatistics))
        self.assertEqual(module.grouping_values, "Metadata_SBS_doses")
        self.assertEqual(len(module.dose_values), 2)
        dose_value = module.dose_values[0]
        self.assertEqual(dose_value.measurement, "Metadata_SBS_doses")
        self.assertFalse(dose_value.log_transform)
        self.assertFalse(dose_value.wants_save_figure)

        dose_value = module.dose_values[1]
        self.assertEqual(dose_value.measurement, "Metadata_Well")
        self.assertTrue(dose_value.log_transform)
        self.assertTrue(dose_value.wants_save_figure)
        self.assertEqual(dose_value.pathname.dir_choice,
                         cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(dose_value.pathname.custom_path, './WELL')

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9525

CalculateStatistics:[module_num:1|svn_version:\'9495\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Where is information about the positive and negative control status of each image?:Metadata_Controls
    Where is information about the treatment dose for each image?:Metadata_SBS_Doses
    Log-transform dose values?:No
    Create dose/response plots?:Yes
    Figure prefix:DoseResponsePlot
    File output location:Default Output Folder\x7CTest
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.calculatestatistics.CalculateStatistics))
        self.assertEqual(module.grouping_values, "Metadata_Controls")
        self.assertEqual(len(module.dose_values), 1)
        dv = module.dose_values[0]
        self.assertEqual(dv.measurement, "Metadata_SBS_Doses")
        self.assertFalse(dv.log_transform)
        self.assertTrue(dv.wants_save_figure)
        self.assertEqual(dv.figure_name, "DoseResponsePlot")
        self.assertEqual(dv.pathname.dir_choice, cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(dv.pathname.custom_path, "Test")

    def make_workspace(self, mdict, controls_measurement, dose_measurements=[]):
        """Make a workspace and module for running CalculateStatistics

        mdict - a two-level dictionary that mimics the measurements structure
                for instance:
                mdict = { cpmeas.Image: { "M1": [ 1,2,3] }}
                for the measurement M1 with values for 3 image sets
        controls_measurement - the name of the controls measurement
        """
        module = cellprofiler.modules.calculatestatistics.CalculateStatistics()
        module.module_num = 1
        module.grouping_values.value = controls_measurement

        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)

        m = cellprofiler.measurement.Measurements()
        nimages = None
        for object_name in mdict.keys():
            odict = mdict[object_name]
            for feature in odict.keys():
                m.add_all_measurements(object_name, feature, odict[feature])
                if nimages is None:
                    nimages = len(odict[feature])
                else:
                    self.assertEqual(nimages, len(odict[feature]))
                if object_name == cellprofiler.measurement.IMAGE and feature in dose_measurements:
                    if len(module.dose_values) > 1:
                        module.add_dose_value()
                    dv = module.dose_values[-1]
                    dv.measurement.value = feature
        m.image_set_number = nimages
        image_set_list = cellprofiler.image.ImageSetList()
        for i in range(nimages):
            image_set = image_set_list.get_image_set(i)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     cellprofiler.region.Set(), m, image_set_list)
        return workspace, module

    def test_02_02_NAN(self):
        """Regression test of IMG-762

        If objects have NAN values, the means are NAN and the
        z-factors are NAN too.
        """
        mdict = {
            cellprofiler.measurement.IMAGE: {
                "Metadata_Controls": [1, 0, -1],
                "Metadata_Doses": [0, .5, 1]},
            INPUT_OBJECTS: {
                TEST_FTR: [numpy.array([1.0, numpy.NaN, 2.3, 3.4, 2.9]),
                           numpy.array([5.3, 2.4, numpy.NaN, 3.2]),
                           numpy.array([numpy.NaN, 3.1, 4.3, 2.2, 1.1, 0.1])]
            }}
        workspace, module = self.make_workspace(mdict,
                                                "Metadata_Controls",
                                                ["Metadata_Doses"])
        module.post_run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for category in ("Zfactor", "OneTailedZfactor", "Vfactor"):
            feature = '_'.join((category, INPUT_OBJECTS, TEST_FTR))
            value = m.get_experiment_measurement(feature)
            self.assertFalse(numpy.isnan(value))

    def test_02_03_make_path(self):
        # regression test of issue #1478
        # If the figure directory doesn't exist, it should be created
        #
        mdict = {
            cellprofiler.measurement.IMAGE: {
                "Metadata_Controls": [1, 0, -1],
                "Metadata_Doses": [0, .5, 1]},
            INPUT_OBJECTS: {
                TEST_FTR: [numpy.array([1.0, 2.3, 3.4, 2.9]),
                           numpy.array([5.3, 2.4, 3.2]),
                           numpy.array([3.1, 4.3, 2.2, 1.1, 0.1])]
            }}
        workspace, module = self.make_workspace(mdict,
                                                "Metadata_Controls",
                                                ["Metadata_Doses"])
        assert isinstance(module, cellprofiler.modules.calculatestatistics.CalculateStatistics)
        my_dir = tempfile.mkdtemp()
        my_subdir = os.path.join(my_dir, "foo")
        fnfilename = FIGURE_NAME + INPUT_OBJECTS + "_" + TEST_FTR + ".pdf"
        fnpath = os.path.join(my_subdir, fnfilename)
        try:
            dose_group = module.dose_values[0]
            dose_group.wants_save_figure.value = True
            dose_group.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            dose_group.pathname.custom_path = my_subdir
            dose_group.figure_name.value = FIGURE_NAME
            module.post_run(workspace)
            self.assertTrue(os.path.isfile(fnpath))
        finally:
            if os.path.exists(fnpath):
                os.remove(fnpath)
            if os.path.exists(my_subdir):
                os.rmdir(my_subdir)
            os.rmdir(my_dir)
