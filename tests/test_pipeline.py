"""test_Pipeline.py - test the CellProfiler.Pipeline module"""
from __future__ import print_function

import base64
import cProfile
import cStringIO
import os
import pstats
import sys
import tempfile
import traceback
import unittest
import zlib

import six

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.module as cpm
import cellprofiler.modules
import cellprofiler.modules.loadimages as LI
import cellprofiler.pipeline as cpp
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
import numpy as np
import numpy.lib.index_tricks
from cellprofiler.modules.injectimage import InjectImage
from tests.modules import example_images_directory

IMAGE_NAME = "myimage"
ALT_IMAGE_NAME = "altimage"
OBJECT_NAME = "myobject"
CATEGORY = "category"
FEATURE_NAME = "category_myfeature"


def module_directory():
    d = cpp.__file__
    d = os.path.split(d)[0]  # ./CellProfiler/pyCellProfiler/cellProfiler
    d = os.path.split(d)[0]  # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0]  # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d, 'Modules')


def image_with_one_cell(size=(100, 100)):
    img = np.zeros(size)
    mgrid = np.lib.index_tricks.nd_grid()
    g = mgrid[0:size[0], 0:size[1]] - 50
    dist = g[0, :, :] * g[0, :, :] + g[1, :, :] * g[1, :, :]  # squared Euclidean distance.
    img[dist < 25] = (25.0 - dist.astype(float)[dist < 25]) / 25  # A circle centered at (50, 50)
    return img


def get_empty_pipeline():
    pipeline = cpp.Pipeline()
    while len(pipeline.modules()) > 0:
        pipeline.remove_module(pipeline.modules()[-1].module_num)
    return pipeline


def exploding_pipeline(test):
    """Return a pipeline that fails if the run exception callback is called during a run
    """
    x = get_empty_pipeline()

    def fn(pipeline, event):
        if isinstance(event, cpp.RunExceptionEvent):
            import traceback
            test.assertFalse(
                    isinstance(event, cpp.RunExceptionEvent),
                    "\n".join([event.error.message] + traceback.format_tb(event.tb)))

    x.add_listener(fn)
    return x


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Change the default output directory to a temporary file
        cpprefs.set_headless()
        self.new_output_directory = os.path.normcase(tempfile.mkdtemp())
        cpprefs.set_default_output_directory(self.new_output_directory)

    def tearDown(self):
        subdir = self.new_output_directory
        for filename in os.listdir(subdir):
            try:
                os.remove(os.path.join(subdir, filename))
            except:
                sys.stderr.write("Failed to remove %s" % filename)
                traceback.print_exc()
        try:
            os.rmdir(subdir)
        except:
            sys.stderr.write("Failed to remove temporary %s directory" % subdir)
            traceback.print_exc()

    def test_00_00_init(self):
        x = cpp.Pipeline()

    def test_01_02_is_txt_fd_sorry_for_your_proofpoint(self):
        # Regression test of issue #1318
        sensible = r"""CellProfiler Pipeline: http://www.cellprofiler.org
    Version:3
    DateRevision:20140723174500
    GitHash:6c2d896
    ModuleCount:17
    HasImagePlaneDetails:False"""
        proofpoint = r"""CellProfiler Pipeline: https://urldefense.proofpoint.com/v2/url?u=http-3A__www.cellprofiler.org&d=AwIGAg&c=4R1YgkJNMyVWjMjneTwN5tJRn8m8VqTSNCjYLg1wNX4&r=ZlgBKM1XjsDOEFy5b6o_Y9E076K1Jlt5FonpX_9mB-M&m=mjjreN4DEr49dWksH8OkXbV51OsYqIX18TSsFFmPurA&s=tQ-7XP8ph9RHRlzicZb6N-OxPxQNMXYLqkucuJS9Hys&e=
Version:3
DateRevision:20140723174500
GitHash:6c2d896
ModuleCount:17
HasImagePlaneDetails:False"""
        not_txt = r"""not CellProfiler Pipeline: http://www.cellprofiler.org"""
        for text, expected in ((sensible, True),
                               (proofpoint, True),
                               (not_txt, False)):
            fd = cStringIO.StringIO(text)
            self.assertEqual(cpp.Pipeline.is_pipeline_txt_fd(fd), expected)

    def test_02_01_copy_nothing(self):
        # Regression test of issue #565
        #
        # Can't copy an empty pipeline
        #
        pipeline = cpp.Pipeline()
        p2 = pipeline.copy()

    def test_06_01_run_pipeline(self):
        x = exploding_pipeline(self)
        module = InjectImage('OneCell', image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        x.run()

    def test_09_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = get_empty_pipeline()
        module = TestModuleWithMeasurement()
        module.module_num = 1
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 9)
        self.assertTrue(any([column[0] == 'Image' and
                             column[1] == 'Group_Number' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and
                             column[1] == 'Group_Index' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and
                             column[1] == 'ModuleError_01TestModuleWithMeasurement'
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and
                             column[1] == 'ExecutionTime_01TestModuleWithMeasurement'
                             for column in columns]))
        self.assertTrue(any([column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_PIPELINE
                             for column in columns]))
        self.assertTrue(any([column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_VERSION
                             for column in columns]))
        self.assertTrue(any([column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_TIMESTAMP
                             for column in columns]))
        self.assertTrue(any([len(columns) > 3 and
                             column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_MODIFICATION_TIMESTAMP and
                             column[3][cpmeas.MCA_AVAILABLE_POST_RUN]
                             for column in columns]))

        self.assertTrue(any([column[1] == "foo" for column in columns]))
        module.my_variable.value = "bar"
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 9)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        module = TestModuleWithMeasurement()
        module.module_num = 2
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 12)
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        columns = x.get_measurement_columns(module)
        self.assertEqual(len(columns), 9)
        self.assertTrue(any([column[1] == "bar" for column in columns]))

    def test_10_01_all_groups(self):
        '''Test running a pipeline on all groups'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun', 0]
        keys = ('foo', 'bar')
        groupings = (({'foo': 'foo-A', 'bar': 'bar-A'}, (1, 2)),
                     ({'foo': 'foo-B', 'bar': 'bar-B'}, (3, 4)))

        def prepare_run(workspace):
            image_set_list = workspace.image_set_list
            self.assertEqual(expects[0], 'PrepareRun')
            for group_number_idx, (grouping, image_numbers) in enumerate(groupings):
                for group_idx, image_number in enumerate(image_numbers):
                    workspace.measurements[cpmeas.IMAGE,
                                           cpmeas.GROUP_NUMBER,
                                           image_number] = group_number_idx + 1
                    workspace.measurements[cpmeas.IMAGE,
                                           cpmeas.GROUP_INDEX,
                                           image_number] = group_idx + 1
            expects[0], expects[1] = ('PrepareGroup', 0)
            return True

        def prepare_group(workspace, grouping, image_numbers):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            if expects_grouping == 0:
                expects[0], expects[1] = ('Run', 1)
                self.assertSequenceEqual(image_numbers, (1, 2))
            else:
                expects[0], expects[1] = ('Run', 3)
                self.assertSequenceEqual(image_numbers, (3, 4))
            return True

        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            if image_number == 1:
                expects[0], expects[1] = ('Run', 2)
            elif image_number == 3:
                expects[0], expects[1] = ('Run', 4)
            elif image_number == 2:
                expects[0], expects[1] = ('PostGroup', 0)
            else:
                expects[0], expects[1] = ('PostGroup', 1)
            workspace.measurements.add_image_measurement("mymeasurement", image_number)

        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(key in grouping)
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            if expects_grouping == 0:
                self.assertEqual(workspace.measurements.image_set_number, 2)
                expects[0], expects[1] = ('PrepareGroup', 1)
            else:
                self.assertEqual(workspace.measurements.image_set_number, 4)
                expects[0], expects[1] = ('PostRun', 0)

        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0], expects[1] = ('Done', 0)

        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement",
                     cpmeas.COLTYPE_INTEGER)]

        module = GroupModule()
        module.setup((keys, groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run()
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image", "mymeasurement")
        self.assertEqual(len(image_numbers), 4)
        self.assertTrue(np.all(image_numbers == np.array([1, 2, 3, 4])))
        group_numbers = measurements.get_all_measurements("Image", "Group_Number")
        self.assertTrue(np.all(group_numbers == np.array([1, 1, 2, 2])))
        group_indexes = measurements.get_all_measurements("Image", "Group_Index")
        self.assertTrue(np.all(group_indexes == np.array([1, 2, 1, 2])))

    def test_10_02_one_group(self):
        '''Test running a pipeline on one group'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun', 0]
        keys = ('foo', 'bar')
        groupings = (({'foo': 'foo-A', 'bar': 'bar-A'}, (1, 2)),
                     ({'foo': 'foo-B', 'bar': 'bar-B'}, (3, 4)),
                     ({'foo': 'foo-C', 'bar': 'bar-C'}, (5, 6)))

        def prepare_run(workspace):
            self.assertEqual(expects[0], 'PrepareRun')
            for group_number_idx, (grouping, image_numbers) in enumerate(groupings):
                for group_idx, image_number in enumerate(image_numbers):
                    workspace.measurements[cpmeas.IMAGE,
                                           cpmeas.GROUP_NUMBER,
                                           image_number] = group_number_idx + 1
                    workspace.measurements[cpmeas.IMAGE,
                                           cpmeas.GROUP_INDEX,
                                           image_number] = group_idx + 1
            expects[0], expects[1] = ('PrepareGroup', 1)
            return True

        def prepare_group(workspace, grouping, *args):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for key in keys:
                self.assertTrue(key in grouping)
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            self.assertEqual(expects_grouping, 1)
            expects[0], expects[1] = ('Run', 3)
            return True

        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            if image_number == 3:
                expects[0], expects[1] = ('Run', 4)
            elif image_number == 4:
                expects[0], expects[1] = ('PostGroup', 1)

            workspace.measurements.add_image_measurement("mymeasurement", image_number)

        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(key in grouping)
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            expects[0], expects[1] = ('PostRun', 0)

        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0], expects[1] = ('Done', 0)

        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement",
                     cpmeas.COLTYPE_INTEGER)]

        module = GroupModule()
        module.setup((keys, groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run(grouping={'foo': 'foo-B', 'bar': 'bar-B'})
        self.assertEqual(expects[0], 'Done')

    def test_10_03_display(self):
        # Test that the individual pipeline methods do appropriate display.

        pipeline = exploding_pipeline(self)
        module = GroupModule()
        module.show_window = True
        callbacks_called = set()

        def prepare_run(workspace):
            workspace.measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1] = 1
            workspace.measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX, 1] = 1
            return True

        def prepare_group(workspace, grouping, *args):
            return True

        def run(workspace):
            workspace.display_data.foo = "Bar"

        def display_handler(module1, display_data, image_set_number):
            self.assertIs(module1, module)
            self.assertEqual(display_data.foo, "Bar")
            self.assertEqual(image_set_number, 1)
            callbacks_called.add("display_handler")

        def post_group(workspace, grouping):
            workspace.display_data.bar = "Baz"

        def post_group_display_handler(module1, display_data, image_set_number):
            self.assertIs(module1, module)
            self.assertEqual(display_data.bar, "Baz")
            self.assertEqual(image_set_number, 1)
            callbacks_called.add("post_group_display_handler")

        def post_run(workspace):
            workspace.display_data.baz = "Foo"

        def post_run_display_handler(workspace, module1):
            self.assertIs(module1, module)
            self.assertEqual(workspace.display_data.baz, "Foo")
            callbacks_called.add("post_run_display_handler")

        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement",
                     cpmeas.COLTYPE_INTEGER)]

        module.setup(((), ({}, (1,))),
                     prepare_run_callback=prepare_run,
                     prepare_group_callback=prepare_group,
                     run_callback=run,
                     post_group_callback=post_group,
                     post_run_callback=post_run)
        module.module_num = 1
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, m, None, m,
                                  cpi.ImageSetList)
        workspace.post_group_display_handler = post_group_display_handler
        workspace.post_run_display_handler = post_run_display_handler
        self.assertTrue(pipeline.prepare_run(workspace))
        pipeline.prepare_group(workspace, {}, (1,))
        pipeline.run_image_set(m, 1, None, display_handler, None)
        self.assertIn("display_handler", callbacks_called)
        pipeline.post_group(workspace, {})
        self.assertIn("post_group_display_handler", callbacks_called)
        pipeline.post_run(workspace)
        self.assertIn("post_run_display_handler", callbacks_called)

    def test_11_01_catch_operational_error(self):
        '''Make sure that a pipeline can catch an operational error

        This is a regression test of IMG-277
        '''
        module = MyClassForTest1101()
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        should_be_true = [False]

        def callback(caller, event):
            if isinstance(event, cpp.RunExceptionEvent):
                should_be_true[0] = True

        pipeline.add_listener(callback)
        pipeline.run()
        self.assertTrue(should_be_true[0])

    def test_11_02_catch_prepare_run_error(self):
        pipeline = exploding_pipeline(self)
        module = GroupModule()
        keys = ('foo', 'bar')
        groupings = (({'foo': 'foo-A', 'bar': 'bar-A'}, (1, 2)),
                     ({'foo': 'foo-B', 'bar': 'bar-B'}, (3, 4)),
                     ({'foo': 'foo-C', 'bar': 'bar-C'}, (5, 6)))

        def prepare_run(workspace):
            m = workspace.measurements
            for i in range(1, 7):
                m[cpmeas.IMAGE, cpmeas.C_PATH_NAME + "_DNA", i] = \
                    "/imaging/analysis"
                m[cpmeas.IMAGE, cpmeas.C_FILE_NAME + "_DNA", i] = "img%d.tif" % i
            workspace.pipeline.report_prepare_run_error(
                    module, "I am configured incorrectly")
            return True

        module.setup(groupings,
                     prepare_run_callback=prepare_run)
        module.module_num = 1
        pipeline.add_module(module)
        workspace = cpw.Workspace(
                pipeline, None, None, None, cpmeas.Measurements(),
                cpi.ImageSetList())
        self.assertFalse(pipeline.prepare_run(workspace))
        self.assertEqual(workspace.measurements.image_set_count, 0)

    def test_12_01_img_286(self):
        '''Regression test for img-286: module name in class'''
        cellprofiler.modules.fill_modules()
        success = True
        all_keys = list(cellprofiler.modules.all_modules.keys())
        all_keys.sort()
        for k in all_keys:
            v = cellprofiler.modules.all_modules[k]
            try:
                v.module_name
            except:
                print("%s needs to define module_name as a class variable" % k)
                success = False
        self.assertTrue(success)

    def test_13_01_save_pipeline(self):
        pipeline = get_empty_pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.save(fd)
        fd.seek(0)

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)

    def test_13_02_save_measurements(self):
        pipeline = get_empty_pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        my_measurement = [np.random.uniform(size=np.random.randint(3, 25))
                          for i in range(20)]
        my_image_measurement = [np.random.uniform() for i in range(20)]
        my_experiment_measurement = np.random.uniform()
        measurements.add_experiment_measurement("expt", my_experiment_measurement)
        for i in range(20):
            if i > 0:
                measurements.next_image_set()
            measurements.add_measurement("Foo", "Bar", my_measurement[i])
            measurements.add_image_measurement(
                    "img", my_image_measurement[i])
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cpmeas.load_measurements(fd)
        my_measurement_out = measurements.get_all_measurements("Foo", "Bar")
        self.assertEqual(len(my_measurement), len(my_measurement_out))
        for m_in, m_out in zip(my_measurement, my_measurement_out):
            self.assertEqual(len(m_in), len(m_out))
            self.assertTrue(np.all(m_in == m_out))
        my_image_measurement_out = measurements.get_all_measurements(
                "Image", "img")
        self.assertEqual(len(my_image_measurement), len(my_image_measurement_out))
        for m_in, m_out in zip(my_image_measurement, my_image_measurement_out):
            self.assertTrue(m_in == m_out)
        my_experiment_measurement_out = \
            measurements.get_experiment_measurement("expt")
        self.assertAlmostEqual(my_experiment_measurement, my_experiment_measurement_out)

        fd.seek(0)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)

    def test_13_03_save_long_measurements(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        # m2 and m3 should go into panic mode because they differ by a cap
        m1_name = "dalkzfsrqoiualkjfrqealkjfqroupifaaalfdskquyalkhfaafdsafdsqteqteqtew"
        m2_name = "lkjxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ"
        m3_name = "druxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ"
        my_measurement = [np.random.uniform(size=np.random.randint(3, 25))
                          for i in range(20)]
        my_other_measurement = [np.random.uniform(size=my_measurement[i].size)
                                for i in range(20)]
        my_final_measurement = [np.random.uniform(size=my_measurement[i].size)
                                for i in range(20)]
        measurements.add_all_measurements("Foo", m1_name, my_measurement)
        measurements.add_all_measurements("Foo", m2_name, my_other_measurement)
        measurements.add_all_measurements("Foo", m3_name, my_final_measurement)
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cpmeas.load_measurements(fd)
        reverse_mapping = cpp.map_feature_names([m1_name, m2_name, m3_name])
        mapping = {}
        for key in reverse_mapping.keys():
            mapping[reverse_mapping[key]] = key
        for name, expected in ((m1_name, my_measurement),
                               (m2_name, my_other_measurement),
                               (m3_name, my_final_measurement)):
            map_name = mapping[name]
            my_measurement_out = measurements.get_all_measurements("Foo", map_name)
            for m_in, m_out in zip(expected, my_measurement_out):
                self.assertEqual(len(m_in), len(m_out))
                self.assertTrue(np.all(m_in == m_out))

                #     def test_13_04_pipeline_measurement(self):
                #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
                # Version:3
                # DateRevision:20120709180131
                # ModuleCount:1
                # HasImagePlaneDetails:False
                #
                # LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
                #     File type to be loaded:individual images
                #     File selection method:Text-Exact match
                #     Number of images in each group?:3
                #     Type the text that the excluded images have in common:Do not use
                #     Analyze all subfolders within the selected folder?:None
                #     Input image file location:Elsewhere...\x7Cc\x3A\\\\trunk\\\\ExampleImages\\\\ExampleSBSImages
                #     Check image sets for unmatched or duplicate files?:Yes
                #     Group images by metadata?:No
                #     Exclude certain files?:No
                #     Specify metadata fields to group by:
                #     Select subfolders to analyze:
                #     Image count:2
                #     Text that these images have in common (case-sensitive):Channel1-01
                #     Position of this image in each group:1
                #     Extract metadata from where?:File name
                #     Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
                #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
                #     Channel count:1
                #     Group the movie frames?:No
                #     Grouping method:Interleaved
                #     Number of channels per group:2
                #     Load the input as images or objects?:Images
                #     Name this loaded image:rawGFP
                #     Name this loaded object:Nuclei
                #     Retain outlines of loaded objects?:No
                #     Name the outline image:NucleiOutlines
                #     Channel number:1
                #     Rescale intensities?:Yes
                #     Text that these images have in common (case-sensitive):Channel2-01
                #     Position of this image in each group:2
                #     Extract metadata from where?:File name
                #     Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
                #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
                #     Channel count:1
                #     Group the movie frames?:No
                #     Grouping method:Interleaved
                #     Number of channels per group:2
                #     Load the input as images or objects?:Images
                #     Name this loaded image:rawDNA
                #     Name this loaded object:Nuclei
                #     Retain outlines of loaded objects?:No
                #     Name the outline image:NucleiOutlines
                #     Channel number:1
                #     Rescale intensities?:Yes
                # """
                #         maybe_download_sbs()
                #         path = os.path.join(example_images_directory(), "ExampleSBSImages")
                #         pipeline = cpp.Pipeline()
                #         pipeline.load(cStringIO.StringIO(data))
                #         module = pipeline.modules()[0]
                #         self.assertTrue(isinstance(module, LI.LoadImages))
                #         module.location.custom_path = path
                #         m = cpmeas.Measurements()
                #         image_set_list = cpi.ImageSetList()
                #         self.assertTrue(pipeline.prepare_run(cpw.Workspace(
                #             pipeline, module, None, None, m, image_set_list)))
                #         pipeline_text = m.get_experiment_measurement(cpp.M_PIPELINE)
                #         pipeline_text = pipeline_text.encode("us-ascii")
                #         pipeline = cpp.Pipeline()
                #         pipeline.loadtxt(cStringIO.StringIO(pipeline_text))
                #         self.assertEqual(len(pipeline.modules()), 1)
                #         module_out = pipeline.modules()[0]
                #         self.assertTrue(isinstance(module_out, module.__class__))
                #         self.assertEqual(len(module_out.settings()), len(module.settings()))
                #         for m1setting, m2setting in zip(module.settings(), module_out.settings()):
                #             self.assertTrue(isinstance(m1setting, cps.Setting))
                #             self.assertTrue(isinstance(m2setting, cps.Setting))
                #             self.assertEqual(m1setting.value, m2setting.value)

    def test_14_01_unicode_save(self):
        pipeline = get_empty_pipeline()
        module = TestModuleWithMeasurement()
        # Little endian utf-16 encoding
        module.my_variable.value = u"\\\u2211"
        module.other_variable.value = u"\u2222\u0038"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd, save_image_plane_details=False)
        result = fd.getvalue()
        lines = result.split("\n")
        self.assertEqual(len(lines), 11)
        text, value = lines[-3].split(":")
        #
        # unicode encoding:
        #     backslash: \\ (BOM encoding)
        #     unicode character: \u2211 (n-ary summation)
        #
        # escape encoding:
        #     utf-16 to byte: \xff\xfe\\\x00\x11"
        #
        # result = \\xff\\xfe\\\\\\x00\\x11"
        self.assertEqual(value, '\\xff\\xfe\\\\\\x00\\x11"')
        text, value = lines[-2].split(":")
        #
        # unicode encoding:
        #     unicode character: \u
        #
        # escape encoding:
        #     utf-16 to byte: \xff\xfe""8\x00
        #
        # result = \\xff\\xfe""8\\x00
        self.assertEqual(value, '\\xff\\xfe""8\\x00')
        mline = lines[7]
        idx0 = mline.find("notes:")
        mline = mline[(idx0 + 6):]
        idx1 = mline.find("|")
        value = eval(mline[:idx1].decode('string_escape'))
        self.assertEqual(value, module.notes)

    def test_14_02_unicode_save_and_load(self):
        #
        # Put "TestModuleWithMeasurement" into the module list
        #
        cellprofiler.modules.fill_modules()
        cellprofiler.modules.all_modules[TestModuleWithMeasurement.module_name] = \
            TestModuleWithMeasurement
        #
        # Continue with test
        #
        pipeline = get_empty_pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        module = TestModuleWithMeasurement()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd)
        fd.seek(0)
        pipeline.loadtxt(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        result_module = pipeline.modules()[0]
        self.assertTrue(isinstance(result_module, TestModuleWithMeasurement))
        self.assertEqual(module.notes, result_module.notes)
        self.assertEqual(module.my_variable.value, result_module.my_variable.value)

    def test_14_03_deprecated_unicode_load(self):
        pipeline = get_empty_pipeline()
        deprecated_pipeline_file = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                 'resources/pipelineV3.cppipe'))
        pipeline.loadtxt(deprecated_pipeline_file)
        module = TestModuleWithMeasurement()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        self.assertEqual(len(pipeline.modules()), 1)
        result_module = pipeline.modules()[0]
        self.assertTrue(isinstance(result_module, TestModuleWithMeasurement))
        self.assertEqual(module.notes, result_module.notes)
        self.assertEqual(module.my_variable.value, result_module.my_variable.value)

    # Sorry Ray, Python 2.6 and below doesn't have @skip
    if False:
        @unittest.skip("skipping profiling AllModules - too slow")
        @np.testing.decorators.slow
        def test_15_01_profile_example_all(self):
            """
            Profile ExampleAllModulesPipeline

            Dependencies:
            User must have ExampleImages on their machine,
            in a location which can be found by example_images_directory().
            This directory should contain the pipeline ExampleAllModulesPipeline
            """
            example_dir = example_images_directory()
            if not example_dir:
                import warnings
                warnings.warn('example_images_directory not found, skipping profiling of ExampleAllModulesPipeline')
                return
            pipeline_dir = os.path.join(example_dir, 'ExampleAllModulesPipeline')
            pipeline_filename = os.path.join(pipeline_dir, 'ExampleAllModulesPipeline.cp')
            image_dir = os.path.join(pipeline_dir, 'Images')

            # Might be better to write these paths into the pipeline
            old_image_dir = cpprefs.get_default_image_directory()
            cpprefs.set_default_image_directory(image_dir)
            profile_pipeline(pipeline_filename)
            cpprefs.set_default_image_directory(old_image_dir)

    # def test_15_02_profile_example_fly(self):
    #     """
    #     Profile ExampleFlyImages pipeline
    #
    #     """
    #     maybe_download_fly()
    #     example_dir = example_images_directory()
    #     pipeline_dir = os.path.join(example_dir, 'ExampleFlyImages')
    #     pipeline_filename = os.path.join(pipeline_dir, 'ExampleFly.cppipe')
    #
    #     #Might be better to write these paths into the pipeline
    #     old_image_dir = cpprefs.get_default_image_directory()
    #     cpprefs.set_default_image_directory(pipeline_dir)
    #     fd = urlopen(
    #         "http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cppipe")
    #     build_dir = os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    #         "build")
    #     if not os.path.isdir(build_dir):
    #         os.makedirs(build_dir)
    #     profile_pipeline(fd, output_filename=os.path.join(build_dir, "profile.txt"))
    #     cpprefs.set_default_image_directory(old_image_dir)

    def test_16_00_get_provider_dictionary_nothing(self):
        for module in (ATestModule(),
                       ATestModule([cps.Choice("foo", ["Hello", "World"])])):
            pipeline = get_empty_pipeline()
            module.module_num = 1
            pipeline.add_module(module)
            for groupname in (cps.IMAGE_GROUP, cps.OBJECT_GROUP, cps.MEASUREMENTS_GROUP):
                d = pipeline.get_provider_dictionary(groupname)
                self.assertEqual(len(d), 0)

    def test_16_01_get_provider_dictionary_image(self):
        pipeline = get_empty_pipeline()
        my_setting = cps.ImageNameProvider("foo", IMAGE_NAME)
        module = ATestModule([my_setting])
        module.module_num = 1
        pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.keys()[0], IMAGE_NAME)
        providers = d[IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        self.assertEqual(provider[1], my_setting)
        for group in (cps.OBJECT_GROUP, cps.MEASUREMENTS_GROUP):
            self.assertEqual(len(pipeline.get_provider_dictionary(group)), 0)

    def test_16_02_get_provider_dictionary_object(self):
        pipeline = get_empty_pipeline()
        my_setting = cps.ObjectNameProvider("foo", OBJECT_NAME)
        module = ATestModule([my_setting])
        module.module_num = 1
        pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.OBJECT_GROUP)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.keys()[0], OBJECT_NAME)
        providers = d[OBJECT_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        self.assertEqual(provider[1], my_setting)
        for group in (cps.IMAGE_GROUP, cps.MEASUREMENTS_GROUP):
            self.assertEqual(len(pipeline.get_provider_dictionary(group)), 0)

    def test_16_03_get_provider_dictionary_measurement(self):
        pipeline = get_empty_pipeline()
        module = ATestModule(
                measurement_columns=[(OBJECT_NAME, FEATURE_NAME, cpmeas.COLTYPE_FLOAT)])
        module.module_num = 1
        pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.MEASUREMENTS_GROUP)
        self.assertEqual(len(d), 1)
        key = d.keys()[0]
        self.assertEqual(len(key), 2)
        self.assertEqual(key[0], OBJECT_NAME)
        self.assertEqual(key[1], FEATURE_NAME)
        providers = d[key]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        for group in (cps.OBJECT_GROUP, cps.IMAGE_GROUP):
            self.assertEqual(len(pipeline.get_provider_dictionary(group)), 0)

    def test_16_04_get_provider_dictionary_other(self):
        pipeline = get_empty_pipeline()
        module = ATestModule(other_providers={cps.IMAGE_GROUP: [IMAGE_NAME]})
        module.module_num = 1
        pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.keys()[0], IMAGE_NAME)
        providers = d[IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        for group in (cps.OBJECT_GROUP, cps.MEASUREMENTS_GROUP):
            self.assertEqual(len(pipeline.get_provider_dictionary(group)), 0)

    def test_16_05_get_provider_dictionary_combo(self):
        pipeline = get_empty_pipeline()
        image_setting = cps.ImageNameProvider("foo", IMAGE_NAME)
        object_setting = cps.ObjectNameProvider("foo", OBJECT_NAME)
        measurement_columns = [(OBJECT_NAME, FEATURE_NAME, cpmeas.COLTYPE_FLOAT)]
        other_providers = {cps.IMAGE_GROUP: [ALT_IMAGE_NAME]}
        module = ATestModule(settings=[image_setting, object_setting],
                             measurement_columns=measurement_columns,
                             other_providers=other_providers)
        module.module_num = 1
        pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP)
        self.assertEqual(len(d), 2)
        self.assertTrue(IMAGE_NAME in d)
        providers = d[IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        self.assertEqual(provider[1], image_setting)
        self.assertTrue(ALT_IMAGE_NAME in d)
        providers = d[ALT_IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(len(provider), 2)
        self.assertEqual(provider[0], module)

        d = pipeline.get_provider_dictionary(cps.OBJECT_GROUP)
        self.assertEqual(len(d), 1)
        self.assertTrue(OBJECT_NAME in d)
        providers = d[OBJECT_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(len(provider), 2)
        self.assertEqual(provider[0], module)
        self.assertEqual(provider[1], object_setting)

        d = pipeline.get_provider_dictionary(cps.MEASUREMENTS_GROUP)
        self.assertEqual(len(d), 1)
        key = d.keys()[0]
        self.assertEqual(len(key), 2)
        self.assertEqual(key[0], OBJECT_NAME)
        self.assertEqual(key[1], FEATURE_NAME)
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)

    def test_16_06_get_provider_module(self):
        #
        # Module 1 provides IMAGE_NAME
        # Module 2 provides OBJECT_NAME
        # Module 3 provides IMAGE_NAME again
        # Module 4 might be a consumer
        #
        # Test disambiguation of the sources
        #
        pipeline = get_empty_pipeline()
        my_image_setting_1 = cps.ImageNameProvider("foo", IMAGE_NAME)
        my_image_setting_2 = cps.ImageNameProvider("foo", IMAGE_NAME)
        my_object_setting = cps.ObjectNameProvider("foo", OBJECT_NAME)
        module1 = ATestModule(settings=[my_image_setting_1])
        module2 = ATestModule(settings=[my_object_setting])
        module3 = ATestModule(settings=[my_image_setting_2])
        module4 = ATestModule()

        for i, module in enumerate((module1, module2, module3, module4)):
            module.module_num = i + 1
            pipeline.add_module(module)
        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP)
        self.assertEqual(len(d), 1)
        self.assertTrue(IMAGE_NAME in d)
        self.assertEqual(len(d[IMAGE_NAME]), 2)
        for module in (module1, module3):
            self.assertTrue(any([x[0] == module for x in d[IMAGE_NAME]]))

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module1)
        self.assertEqual(len(d), 0)

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module2)
        self.assertEqual(len(d), 1)
        self.assertTrue(IMAGE_NAME in d)
        self.assertEqual(d[IMAGE_NAME][0][0], module1)

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module4)
        self.assertEqual(len(d), 1)
        self.assertTrue(IMAGE_NAME in d)
        self.assertEqual(len(d[IMAGE_NAME]), 1)
        self.assertEqual(d[IMAGE_NAME][0][0], module3)

    def test_17_00_get_dependency_graph_empty(self):
        for module in (ATestModule(),
                       ATestModule([cps.Choice("foo", ["Hello", "World"])]),
                       ATestModule([cps.ImageNameProvider("foo", IMAGE_NAME)]),
                       ATestModule([cps.ImageNameSubscriber("foo", IMAGE_NAME)])):
            pipeline = cpp.Pipeline()
            module.module_num = 1
            pipeline.add_module(module)
            result = pipeline.get_dependency_graph()
            self.assertEqual(len(result), 0)

    def test_17_01_get_dependency_graph_image(self):
        pipeline = cpp.Pipeline()
        for i, module in enumerate((
                ATestModule([cps.ImageNameProvider("foo", IMAGE_NAME)]),
                ATestModule([cps.ImageNameProvider("foo", ALT_IMAGE_NAME)]),
                ATestModule([cps.ImageNameSubscriber("foo", IMAGE_NAME)]))):
            module.module_num = i + 1
            pipeline.add_module(module)
        g = pipeline.get_dependency_graph()
        self.assertEqual(len(g), 1)
        edge = g[0]
        self.assertTrue(isinstance(edge, cpp.ImageDependency))
        self.assertEqual(edge.source, pipeline.modules()[0])
        self.assertEqual(edge.source_setting, pipeline.modules()[0].settings()[0])
        self.assertEqual(edge.image_name, IMAGE_NAME)
        self.assertEqual(edge.destination, pipeline.modules()[2])
        self.assertEqual(edge.destination_setting, pipeline.modules()[2].settings()[0])

    def test_17_02_get_dependency_graph_object(self):
        pipeline = cpp.Pipeline()
        for i, module in enumerate((
                ATestModule([cps.ObjectNameProvider("foo", OBJECT_NAME)]),
                ATestModule([cps.ImageNameProvider("foo", IMAGE_NAME)]),
                ATestModule([cps.ObjectNameSubscriber("foo", OBJECT_NAME)]))):
            module.module_num = i + 1
            pipeline.add_module(module)
        g = pipeline.get_dependency_graph()
        self.assertEqual(len(g), 1)
        edge = g[0]
        self.assertTrue(isinstance(edge, cpp.ObjectDependency))
        self.assertEqual(edge.source, pipeline.modules()[0])
        self.assertEqual(edge.source_setting, pipeline.modules()[0].settings()[0])
        self.assertEqual(edge.object_name, OBJECT_NAME)
        self.assertEqual(edge.destination, pipeline.modules()[2])
        self.assertEqual(edge.destination_setting, pipeline.modules()[2].settings()[0])

    def test_17_03_get_dependency_graph_measurement(self):
        pipeline = cpp.Pipeline()
        measurement_columns = [
            (OBJECT_NAME, FEATURE_NAME, cpmeas.COLTYPE_FLOAT)]
        measurement_setting = cps.Measurement("text", lambda: OBJECT_NAME, FEATURE_NAME)
        for i, module in enumerate((
                ATestModule(measurement_columns=measurement_columns),
                ATestModule([cps.ImageNameProvider("foo", ALT_IMAGE_NAME)]),
                ATestModule([measurement_setting]))):
            module.module_num = i + 1
            pipeline.add_module(module)
        g = pipeline.get_dependency_graph()
        self.assertEqual(len(g), 1)
        edge = g[0]
        self.assertTrue(isinstance(edge, cpp.MeasurementDependency))
        self.assertEqual(edge.source, pipeline.modules()[0])
        self.assertEqual(edge.object_name, OBJECT_NAME)
        self.assertEqual(edge.feature, FEATURE_NAME)
        self.assertEqual(edge.destination, pipeline.modules()[2])
        self.assertEqual(edge.destination_setting, pipeline.modules()[2].settings()[0])

    def test_18_01_read_image_plane_details(self):
        test_data = (
            ([], ['"foo","1","2","3"', '"bar","4","5","6"'],
             (("foo", 1, 2, 3, {}), ("bar", 4, 5, 6, {}))),
            (["Well", "Plate"],
             ['"foo","1","2",,"A01","P-12345"',
              '"bar","4",,"6",,"P-67890"',
              '"baz","7","8",,"A03",'],
             (("foo", 1, 2, None, {"Well": "A01", "Plate": "P-12345"}),
              ("bar", 4, None, 6, {"Plate": "P-67890"}),
              ("baz", 7, 8, None, {"Well": "A03"}))),
            (["Well"],
             ['"foo","1","2","3","\\xce\\xb1\\xce\\xb2"'],
             [("foo", 1, 2, 3, {"Well": u"\u03b1\u03b2"})]),
            ([],
             [r'"\\foo\"bar","4","5","6"'],
             [(r'\foo"bar', 4, 5, 6)]))
        for metadata_columns, body_lines, expected in test_data:
            s = '"%s":"%d","%s":"%d"\n' % (
                cpp.H_VERSION, cpp.IMAGE_PLANE_DESCRIPTOR_VERSION,
                cpp.H_PLANE_COUNT, len(body_lines))
            s += '"' + '","'.join([
                                      cpp.H_URL, cpp.H_SERIES, cpp.H_INDEX, cpp.H_CHANNEL] +
                                  metadata_columns) + '"\n'
            s += "\n".join(body_lines) + "\n"
            fd = cStringIO.StringIO(s)
            result = cpp.read_file_list(fd)
            self.assertEqual(len(result), len(expected))
            for r, e in zip(result, expected):
                self.assertEqual(r, e[0])

    def test_18_02_write_image_plane_details(self):
        test_data = (
            "foo", u"\u03b1\u03b2",
            "".join([chr(i) for i in range(128)]))
        fd = cStringIO.StringIO()
        cpp.write_file_list(fd, test_data)
        fd.seek(0)
        result = cpp.read_file_list(fd)
        for rr, tt in zip(result, test_data):
            if isinstance(tt, six.text_type):
                tt = tt.encode("utf-8")
            self.assertEquals(rr, tt)

    def test_19_01_read_file_list_pathnames(self):
        root = os.path.split(__file__)[0]
        paths = [os.path.join(root, x) for x in ("foo.tif", "bar.tif")]
        fd = cStringIO.StringIO("\n".join([
            paths[0], "", paths[1]]))
        p = cpp.Pipeline()
        p.read_file_list(fd)
        self.assertEqual(len(p.file_list), 2)
        for path in paths:
            self.assertIn(LI.pathname2url(path), p.file_list)

    def test_19_02_read_file_list_urls(self):
        root = os.path.split(__file__)[0]
        file_url = LI.pathname2url(os.path.join(root, "foo.tif"))
        urls = ["http://cellprofiler.org/foo.tif",
                file_url,
                "https://github.com/foo.tif",
                "ftp://example.com/foo.tif"]
        fd = cStringIO.StringIO("\n".join(urls))
        p = cpp.Pipeline()
        p.read_file_list(fd)
        self.assertEqual(len(p.file_list), len(urls))
        for url in urls:
            self.assertIn(url, p.file_list)

    def test_19_03_read_file_list_file(self):
        urls = ["http://cellprofiler.org/foo.tif",
                "https://github.com/foo.tif",
                "ftp://example.com/foo.tif"]
        fd, path = tempfile.mkstemp(".txt", text=True)
        try:
            os.write(fd, "\n".join(urls))
            p = cpp.Pipeline()
            p.read_file_list(path)
        finally:
            os.close(fd)
            os.remove(path)
        self.assertEqual(len(p.file_list), len(urls))
        for url in urls:
            self.assertIn(url, p.file_list)

    def test_19_04_read_http_file_list(self):
        url = "https://gist.githubusercontent.com/mcquin/67438dc4e8481c5b1d3881df56e1c4c4/raw/274835d9d3fef990d8bf34c4ee5f991b3880d74f/gistfile1.txt"
        urls = ["http://cellprofiler.org/foo.tif", "https://github.com/foo.tif", "ftp://example.com/foo.tif"]
        p = cpp.Pipeline()
        p.read_file_list(url)
        self.assertEqual(len(p.file_list), len(urls))
        for url in urls:
            self.assertIn(url, p.file_list)


class TestImagePlaneDetails(unittest.TestCase):
    def get_ipd(self,
                url="http://cellprofiler.org",
                series=0, index=0, channel=0,
                metadata={}):
        d = cpp.J.make_map(**metadata)
        jipd = cpp.J.run_script(
                """
            var uri = new java.net.URI(url);
            var f = new Packages.org.cellprofiler.imageset.ImageFile(uri);
            var fd = new Packages.org.cellprofiler.imageset.ImageFileDetails(f);
            var s = new Packages.org.cellprofiler.imageset.ImageSeries(f, series);
            var sd = new Packages.org.cellprofiler.imageset.ImageSeriesDetails(s, fd);
            var p = new Packages.org.cellprofiler.imageset.ImagePlane(s, index, channel);
            var ipd = new Packages.org.cellprofiler.imageset.ImagePlaneDetails(p, sd);
            ipd.putAll(d);
            ipd;
            """, dict(url=url, series=series, index=index, channel=channel, d=d))
        return cpp.ImagePlaneDetails(jipd)

        # def test_01_01_init(self):
        #     self.get_ipd();

        # def test_02_01_path_url(self):
        #     url = "http://google.com"
        #     ipd = self.get_ipd(url=url)
        #     self.assertEquals(ipd.path, url)

        # def test_02_02_path_file(self):
        #     path = "file:" + cpp.urllib.pathname2url(__file__)
        #     ipd = self.get_ipd(url=path)
        #     if sys.platform == 'win32':
        #         self.assertEquals(ipd.path.lower(), __file__.lower())
        #     else:
        #         self.assertEquals(ipd.path, __file__)

        # def test_03_01_url(self):
        #     url = "http://google.com"
        #     ipd = self.get_ipd(url=url)
        #     self.assertEquals(ipd.url, url)

        # def test_04_01_series(self):
        #     ipd = self.get_ipd(series = 4)
        #     self.assertEquals(ipd.series, 4)

        # def test_05_01_index(self):
        #     ipd = self.get_ipd(index = 2)
        #     self.assertEquals(ipd.index, 2)

        # def test_06_01_channel(self):
        #     ipd = self.get_ipd(channel=3)
        #     self.assertEquals(ipd.channel, 3)

        # def test_07_01_metadata(self):
        #     ipd = self.get_ipd(metadata = dict(foo="Bar", baz="Blech"))
        #     self.assertEquals(ipd.metadata["foo"], "Bar")
        #     self.assertEquals(ipd.metadata["baz"], "Blech")

        # def test_08_01_save_pipeline_notes(self):
        #     fd = cStringIO.StringIO()
        #     pipeline = cpp.Pipeline()
        #     module = ATestModule()
        #     module.module_num = 1
        #     module.notes.append("Hello")
        #     module.notes.append("World")
        #     pipeline.add_module(module)
        #     module = ATestModule()
        #     module.module_num = 2
        #     module.enabled = False
        #     pipeline.add_module(module)
        #     expected = "\n".join([
        #         "[   1] [ATestModule]",
        #         "  Hello",
        #         "  World",
        #         "",
        #         "[   2] [ATestModule] (disabled)",
        #         ""])
        #
        #     pipeline.save_pipeline_notes(fd)
        #     self.assertEqual(fd.getvalue(), expected)


def profile_pipeline(pipeline_filename,
                     output_filename=None,
                     always_run=True):
    """
    Run the provided pipeline, output the profiled results to a file.
    Pipeline is run each time by default, if canskip_rerun = True
    the pipeline is only run if the profile results filename does not exist

    Parameters
    --------------
    pipeline_filename: str
        Absolute path to pipeline
    output_filename: str, optional
        Output file for profiled results. Default is cpprefs default output directory.
    always_run: Bool, optional
        By default, only runs if output_filename does not exist
        If always_run = True, then always runs
    """

    # helper function
    def run_pipeline(pipeline_filename,
                     image_set_start=None, image_set_end=None,
                     groups=None, measurements_filename=None):
        pipeline = cpp.Pipeline()
        measurements = None
        pipeline.load(pipeline_filename)
        measurements = pipeline.run(
                image_set_start=image_set_start,
                image_set_end=image_set_end,
                grouping=groups,
                measurements_filename=measurements_filename,
                initial_measurements=measurements)

    if not output_filename:
        pipeline_name = os.path.basename(pipeline_filename).split('.')[0]
        output_filename = os.path.join(cpprefs.get_default_output_directory(), pipeline_name + '_profile')

    if not os.path.exists(output_filename) or always_run:
        print('Profiling %s' % pipeline_filename)
        cProfile.runctx('run_pipeline(pipeline_filename)', globals(), locals(), output_filename)

    p = pstats.Stats(output_filename)
    # sort by cumulative time spent, optionally strip directory names
    to_print = p.sort_stats('cumulative')
    to_print.print_stats(20)


class ATestModule(cpm.Module):
    module_name = "ATestModule"
    variable_revision_number = 1

    def __init__(self, settings=[], measurement_columns=[], other_providers={}):
        super(type(self), self).__init__()
        self.__settings = settings
        self.__measurement_columns = measurement_columns
        self.__other_providers = other_providers

    def settings(self):
        return self.__settings

    def get_measurement_columns(self, pipeline):
        return self.__measurement_columns

    def other_providers(self, group):
        if group not in self.__other_providers.keys():
            return []
        return self.__other_providers[group]

    def get_categories(self, pipeline, object_name):
        categories = set()
        for cobject_name, cfeature_name, ctype \
                in self.get_measurement_columns(pipeline):
            if cobject_name == object_name:
                categories.add(cfeature_name.split("_")[0])
        return list(categories)

    def get_measurements(self, pipeline, object_name, category):
        measurements = set()
        for cobject_name, cfeature_name, ctype \
                in self.get_measurement_columns(pipeline):
            ccategory, measurement = cfeature_name.split("_", 1)
            if cobject_name == object_name and category == category:
                measurements.add(measurement)
        return list(measurements)


class TestModuleWithMeasurement(cpm.Module):
    module_name = "Test0801"
    category = "Test"
    variable_revision_number = 1

    def create_settings(self):
        self.my_variable = cps.Text('', '')
        self.other_variable = cps.Text('', '')

    def settings(self):
        return [self.my_variable, self.other_variable]

    module_name = "TestModuleWithMeasurement"
    variable_revision_number = 1

    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.TestModuleWithMeasurement"

    def get_measurement_columns(self, pipeline):
        return [(cpmeas.IMAGE,
                 self.my_variable.value,
                 "varchar(255)")]


class MyClassForTest1101(cpm.Module):
    def create_settings(self):
        self.my_variable = cps.Text('', '')

    def settings(self):
        return [self.my_variable]

    module_name = "MyClassForTest1101"
    variable_revision_number = 1

    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest1101"

    def prepare_run(self, workspace, *args):
        image_set = workspace.image_set_list.get_image_set(0)
        workspace.measurements.add_measurement("Image", "Foo", 1)
        return True

    def prepare_group(self, workspace, *args):
        image_set = workspace.image_set_list.get_image_set(0)
        image = cpi.Image(np.zeros((5, 5)))
        image_set.add("dummy", image)
        return True

    def run(self, *args):
        import MySQLdb
        raise MySQLdb.OperationalError("Bogus error")


class GroupModule(cpm.Module):
    module_name = "Group"
    variable_revision_number = 1

    def setup(self, groupings,
              prepare_run_callback=None,
              prepare_group_callback=None,
              run_callback=None,
              post_group_callback=None,
              post_run_callback=None,
              get_measurement_columns_callback=None,
              display_callback=None,
              display_post_group_callback=None,
              display_post_run_callback=None):
        self.prepare_run_callback = prepare_run_callback
        self.prepare_group_callback = prepare_group_callback
        self.run_callback = run_callback
        self.post_group_callback = post_group_callback
        self.post_run_callback = post_run_callback
        self.groupings = groupings
        self.get_measurement_columns_callback = get_measurement_columns_callback
        self.display_callback = None
        self.display_post_group_callback = None
        self.display_post_run_callback = None

    def settings(self):
        return []

    def get_groupings(self, workspace):
        return self.groupings

    def prepare_run(self, *args):
        if self.prepare_run_callback is not None:
            return self.prepare_run_callback(*args)
        return True

    def prepare_group(self, *args):
        if self.prepare_group_callback is not None:
            return self.prepare_group_callback(*args)
        return True

    def run(self, *args):
        if self.run_callback is not None:
            return self.run_callback(*args)

    def post_run(self, *args):
        if self.post_run_callback is not None:
            return self.post_run_callback(*args)

    def post_group(self, *args):
        if self.post_group_callback is not None:
            return self.post_group_callback(*args)

    def get_measurement_columns(self, *args):
        if self.get_measurement_columns_callback is not None:
            return self.get_measurement_columns_callback(*args)
        return []

    def display(self, workspace, figure):
        if self.display_callback is not None:
            return self.display_callback(workspace, figure)
        return super(GroupModule, self).display(workspace, figure)

    def display_post_group(self, workspace, figure):
        if self.display_post_group_callback is not None:
            return self.display_post_group_callback(workspace, figure)
        return super(GroupModule, self).display_post_group(workspace, figure)

    def display_post_run(self, workspace, figure):
        if self.display_post_run is not None:
            return self.display_post_run_callback(workspace, figure)
        return super(GroupModule, self).display_post_run(workspace, figure)


if __name__ == "__main__":
    unittest.main()


class TestUtils(unittest.TestCase):
    def test_02_001_EncapsulateString(self):
        a = cellprofiler.pipeline.encapsulate_string('Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'S')
        self.assertTrue(a[0] == 'Hello')

    def test_02_001_EncapsulateUnicode(self):
        a = cellprofiler.pipeline.encapsulate_string(u'Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'U')
        self.assertTrue(a[0] == u'Hello')

    def test_02_01_EncapsulateCell(self):
        cell = numpy.ndarray((1, 1), dtype=object)
        cell[0, 0] = u'Hello, world'
        cellprofiler.pipeline.encapsulate_strings_in_arrays(cell)
        self.assertTrue(isinstance(cell[0, 0], numpy.ndarray))
        self.assertTrue(cell[0, 0][0] == u'Hello, world')

    def test_02_02_EncapsulateStruct(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])
        struct['foo'][0, 0] = u'Hello, world'
        cellprofiler.pipeline.encapsulate_strings_in_arrays(struct)
        self.assertTrue(isinstance(struct['foo'][0, 0], numpy.ndarray))
        self.assertTrue(struct['foo'][0, 0][0] == u'Hello, world')

    def test_02_03_EncapsulateCellInStruct(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])
        cell = numpy.ndarray((1, 1), dtype=object)
        cell[0, 0] = u'Hello, world'
        struct['foo'][0, 0] = cell
        cellprofiler.pipeline.encapsulate_strings_in_arrays(struct)
        self.assertTrue(isinstance(cell[0, 0], numpy.ndarray))
        self.assertTrue(cell[0, 0][0] == u'Hello, world')

    def test_02_04_EncapsulateStructInCell(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])
        cell = numpy.ndarray((1, 1), dtype=object)
        cell[0, 0] = struct
        struct['foo'][0, 0] = u'Hello, world'
        cellprofiler.pipeline.encapsulate_strings_in_arrays(cell)
        self.assertTrue(isinstance(struct['foo'][0, 0], numpy.ndarray))
        self.assertTrue(struct['foo'][0, 0][0] == u'Hello, world')
