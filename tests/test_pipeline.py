"""test_Pipeline.py - test the CellProfiler.Pipeline module"""

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

    def test_01_01_load_mat(self):
        '''Regression test of img-942, load a batch data pipeline with notes'''

        global img_942_data  # see bottom of this file

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(zlib.decompress(base64.b64decode(img_942_data))))
        module = pipeline.modules()[0]
        self.assertEqual(len(module.notes), 1)
        self.assertEqual(
                module.notes[0],
                """Excluding "_E12f03d" since it has an incomplete set of channels (and is the only one as such).""")

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
        module = MyClassForTest0801()
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
                             column[1] == 'ModuleError_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and
                             column[1] == 'ExecutionTime_01MyClassForTest0801'
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
        module = MyClassForTest0801()
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
                self.assertTrue(grouping.has_key(key))
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
                self.assertTrue(grouping.has_key(key))
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
                self.assertTrue(grouping.has_key(key))
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
                print "%s needs to define module_name as a class variable" % k
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
        module = MyClassForTest0801()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd, save_image_plane_details=False)
        result = fd.getvalue()
        lines = result.split("\n")
        self.assertEqual(len(lines), 10)
        text, value = lines[-2].split(":")
        #
        # unicode encoding:
        #     backslash: \\
        #     unicode character: \u
        #
        # escape encoding:
        #     backslash * 2: \\\\
        #     unicode character: \\
        #
        # result = \\\\\\u2211
        self.assertEqual(value, r"\\\\\\u2211")
        mline = lines[7]
        idx0 = mline.find("notes:")
        mline = mline[(idx0 + 6):]
        idx1 = mline.find("|")
        value = eval(mline[:idx1].decode('string_escape'))
        self.assertEqual(value, module.notes)

    def test_14_02_unicode_save_and_load(self):
        #
        # Put "MyClassForTest0801" into the module list
        #
        cellprofiler.modules.fill_modules()
        cellprofiler.modules.all_modules[MyClassForTest0801.module_name] = \
            MyClassForTest0801
        #
        # Continue with test
        #
        pipeline = get_empty_pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        module = MyClassForTest0801()
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
        self.assertTrue(isinstance(result_module, MyClassForTest0801))
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
        self.assertTrue(d.has_key(IMAGE_NAME))
        providers = d[IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider[0], module)
        self.assertEqual(provider[1], image_setting)
        self.assertTrue(d.has_key(ALT_IMAGE_NAME))
        providers = d[ALT_IMAGE_NAME]
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(len(provider), 2)
        self.assertEqual(provider[0], module)

        d = pipeline.get_provider_dictionary(cps.OBJECT_GROUP)
        self.assertEqual(len(d), 1)
        self.assertTrue(d.has_key(OBJECT_NAME))
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
        self.assertTrue(d.has_key(IMAGE_NAME))
        self.assertEqual(len(d[IMAGE_NAME]), 2)
        for module in (module1, module3):
            self.assertTrue(any([x[0] == module for x in d[IMAGE_NAME]]))

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module1)
        self.assertEqual(len(d), 0)

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module2)
        self.assertEqual(len(d), 1)
        self.assertTrue(d.has_key(IMAGE_NAME))
        self.assertEqual(d[IMAGE_NAME][0][0], module1)

        d = pipeline.get_provider_dictionary(cps.IMAGE_GROUP, module4)
        self.assertEqual(len(d), 1)
        self.assertTrue(d.has_key(IMAGE_NAME))
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
            if isinstance(tt, unicode):
                tt = tt.encode("utf-8")
            self.assertEquals(rr, tt)

    def test_19_01_read_file_list_pathnames(self):
        root = os.path.split(__file__)[0]
        paths = [os.path.join(root, x) for x in "foo.tif", "bar.tif"]
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
        print 'Profiling %s' % pipeline_filename
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


class MyClassForTest0801(cpm.Module):
    module_name = "Test0801"
    category = "Test"
    variable_revision_number = 1

    def create_settings(self):
        self.my_variable = cps.Text('', '')

    def settings(self):
        return [self.my_variable]

    module_name = "MyClassForTest0801"
    variable_revision_number = 1

    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest0801"

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


# Global declarations

img_942_data = ('eJyVd3VQFN7bLwiKSAtIs7T00p1SUgLSXZIqtdTCLogoISUgHYuggkgo3Q1L'
                '7oJ0dzcrLLAs+/r7ft97Z9577z/3M2fmxJzz1HliHn1VEz3VJ2ySwiJs+qom'
                'Qq4er1wE2QxfOfq7evu9lmMzVDPXfibIpubn4ujv8oLN20uOzfzvrO8IZhOV'
                'ZRORlRMTlZOQYRMTERVh+/8GHr62PgUeHt5HZjy84NxhS3vvcRHKILjUPb1x'
                'SiAbF80h8Ux7DQGh92HTM1J36SmP3/lv5xvy2H8VyTAIwd4kwTkBRM36pVM6'
                'mxPaUhyWk423i7YINHz7FR7e7StghOltcx/8Ib2yAJVtNuNbryK2ji9Isw6x'
                'YMYR3Ob6w2v6QGZ/4jxKURHw2hX17yI6Ne8z89S5z5FWo3l2wzCzncCPyGeh'
                'oiLXGFKKj+N97ZSZ0DWX7Z174nra75Iuh6qpGyU+btRV7iKdYT84T5dXiprs'
                'smeKuWa3ZQHzxignrJbS2vkQ8QVEoTZMvFD01ylZndoO/cx7bGzgW9j3lXqF'
                'bm+GiRxz2N2j1jv1HLvsO9TsCl8XQsUvQwtc551D+XxyUmxE9yYOHd6WuR6W'
                'OU0+mwOfGc2utvVojRYrHQFqPb8fGGZ3ZqjIVsF0jkq5zEfLf+6z5ZK386xO'
                'c7T2Nu+vzTs+PnGU7lcLT1tMdz3RLdBQ5ceBeQdo/buPOOUDl6yrwlOWmdk5'
                '6lSsleDS9/LbIOy5za1SbUfGQJIlVDLtOh0CwOeLuvm2A5EGIn48WODUrigA'
                '5/t9PvV76bY/6W5njtv6iT3X1cG6hNqf8hfluAFH2WZekwZ/E7Gh969EnZin'
                'sWr72heTn39tAHoCxrdYaGEBsUkF1rx/mDYYU4Srxs/AbKdU/iO+azRSY49f'
                'sgzoJmkgRV/vdQlivgtUuodybxFEV3loXQP2DIwumvebiqyVTfziEOHa4Ehn'
                'IcZ75MqoRKsohT+j5ZzLNA3TKTaq5nqoSJe2ar0wK4ChUx4ls8OOxeuJICri'
                'MoNPEp8TGLidcz+irNm3n6dw7kDHTE/7AgAPd0gZXDG73KivLuNiy1abFGYJ'
                'NzIy3pMnogGvNvYncyI/Yb4Misf1ucU9NHmhlLdNJ3dLP/nkEGTUWAU6ebRh'
                'rWUKSVmrjmUjgXIp8AdAeNAurAaS8bTgNu3fuPvVKq/e/XrJ7PobNh1VA/X4'
                'HRaoAR2+UmzjuzzQhhCOxHFy1anKLxtJO2i6uatEgkviQuln7zEsijPUuAWv'
                'tX3YxdhD6Q2REcMJW7Woawuj8pBb12KnwQZTYjplUeE25WFxO/9LrsnewjOX'
                'JWiZV77Q3kIqDdKsyGY3iyiQFf0SQHv5+KHdKFwpAC79VZNCPqZHul7uakwv'
                'NvXQ68i0ZqNVmw+YEC74MEkO20F3sPZmd9e64sV8UumdevZdgR2jvE6uF0V9'
                'Njg7F//Kj51e5x/vToyglRyM5M3e3KJjtLGyeTqFbENQuvPXeg43i9yVCkE5'
                't7DJpA/DoLgZVenjXD4fIBf27uK7TAsGTHHLvq1phG4oMtgcGmi0XJSy19Kv'
                'lkl0aDshsIiyGjuW9bY3OT3VSxHYAc2tpq1RBob/gTKeIQpd1YAd0u6D5KZ3'
                'l3tAEswxUoRwVvnj1lt8PDwc7sBywGtchV5qkfeOjizew2IFFZck4zsjQl4e'
                'upSWMdxcDfHMoT76cPHkjN86zMGEicxgEwXtBSljrXW9Qy4NC0bu9uxj+63l'
                'xQANI6lU0PGcIzLwTN9AqT4FJSzLtL5cnoALY0E73ZEuDK0BRI+1ucCa8BNi'
                'jWX7arxX2ZXZVurFOr01PIy56bwC7tSXQjRXKesfd5SDrsWvT0jv0ukoNYL/'
                'LK747KaumfXBwBvl+v3WuhtUPT5k+EdUbo+LrKfA78PKg9s2C03adtqOQBD6'
                'RWKFaffsr3NCTHQ30CyAskoUhq+jVkaeoZysiwSjzslLXvJoDv8JoZzZC497'
                'souHtWTHMU5Ed+fwuzKv81nMsg881cpf1iwHhtybwC/ZlSshwPt9c809uk5S'
                'TFRE4RvTBivA+156XeT+1e/a2aqVVyxsEAFKvaBCeAABWteJ5Om/lGjSHoMP'
                'c7aSLH1Vypf5j7ZMvSOazK3fLbusMR2wrW3UgJZ8+ZmkqwLcjLNaAJrVTdVJ'
                'zjX35cNJfKSKgUWP95VezrP6SsZZXHL8eVL+QIXl7TSlyWffZ8SvHgLPBEn1'
                'N65j1E8UuYWGTp/opMD28N4yTuz7VOlv1MUh40KwI0ccN8IClewdr0FoJAL/'
                'HMOq+tPFP1XAK1vsjAwCIs/nXlIPUd744+r/+MG6/z2Dq0/cOOLHsDEt1sSU'
                'kgA5rHd/otd189Mr61yIZs5EvRVrbLcaXONZW+sWHzx6TOVut5sWIDxLtbLs'
                'Kmd3MpYzt8ck8yy5NaPwkOo0FHHqXAN+2LNBkxz4ynnD1rKuv22acqV6zSUT'
                'aEsTv1a7ABkYnJawn9GzkYBx7KHEhn+M/6HpPRQ/yqcvhxbKXKSmuTfodI7x'
                'm1bdjl9bqr0ZKq0c/Xb7JsruHZu+Zrb1iFpZ/eJ7sJH5j/qUILqLIzE0/VTo'
                'xgWgYOjyR4bE0tWRh/T5pzTzF+2DITgjXSsfPy6UCm4b5ZB+6XnfJ0k2lH7E'
                'Ffy08t4vjT/yWQ2nkOldAtu5I+LpzQn7C+x8TRgI8sLfZYM8xnpvqX+9lKFB'
                'A7HtMlJEIaNIz9JD/97Tn5WWLdzutpSuxC0LNOyluWivqWifcH4DgPuhM0wr'
                'XZPrD45JXK4zFH8yL3OqQczfhA+sMkAb6xP0IzO+uWC+H8geXeN+3d4/ISBX'
                'zkxqlxiXUyNnTnAI/zcELV7+rdc0Ac0V8dGUJxFIdxjv7+fFX53TDD0M5Y2+'
                '1I29JXTPFkwNztLYM9GWK3mD6zLgPlKT8OBBp0nkPeIrffZpHASBVlbmwfHe'
                'UOMQGHRE+HbH5RxsJT58OefmbH+AMIjFk1+IzEeJ5gR950bxJK0B1AKJkW79'
                '1nKwEUTVaqMCJCog3JQ5F3MwAxJs3F8SCwmIWCKgzKVIzInSjjejyHlRLmMq'
                'RVucuS2eptQNtnvr8/istsbV5vbBn2u2zEpmaaGMmwYO8mSu3aDk61jdLMse'
                'Umu1jCYNZX9TwiQqQfbN4hHDOKMkJxKPngOqrc9JUi+dI5ukUPZp7G9pucyT'
                'u9oykHIPFGuFImU0yOypCRI+dCWI5892XFYVOCeBCwMkAR9I4E1yPkOITJxq'
                'R/ijNZACYLAXToWIm3b45bswAWBkceFeTL2LjdrXT+cXOC3ggjfZLVP0Rq8/'
                'Qt0HfZfOvYJ92WPhKZM+jC6UUXzQG+s8crR61FHNi5ImY/rTSmpgEm2g1iY7'
                'FeMlTiDfbYmyPmvfu659tbw3E7+wpZBolZE+Rl1eRMSWqcVcHxtNdrBqul+s'
                'KnzaccZ4zL1uPdEgvatcCmMwvJ6Qs4XetCTV0NbWHII0H7Y0OUB80cnAO9ju'
                'j7erZUqe1JFOFwADNy+ovfbhNBaKzkNKQpYmtPbu2C2AhfaWkWrXzALknXO0'
                'R9QTEwG4nlVD6AueAfCza3ltE9yzSKRGW+UrP8A1JSEqYn1L0jcK+OezV/LT'
                'ZY8jgaGpctIgaWRqRZI8LOcczISqFh7V9WuNBdkVIhI8Tho9cTEgiEcJ/Auy'
                'HsOWuc2MCZ314/56OpumVCugJG3a7v3wgvjr3gI6mguhPz+cwQW5uBsiSWkD'
                'lDFLe1YU+q0g43iivvS94vNUO1rNHfFfZXONfl/scruL9norf8O6uMMrjHDc'
                'MiFmMegWKgTTeGf36+nPMWiKf10bqdstQoMIkynumiiqZw0cZdnbGoh0j3Ph'
                'dZf2L1hTf8LUpPNac+4Zenn+p/7TvMDxcu39O58h7zPQm6HA468jkWMl76pZ'
                'Ec9lhlYFE5crE3yFc69OUxPzKHcvf9jyOUkzUMsb3Yw4F/ecLDgFJ/psk+lG'
                'EShSgMMv2tdGzpZiTr5dQnbWw/ox50y48Sk+kX2aDrWG7ZotlES4Rp89lQUh'
                'ObhH3MP2yPnGFD9HmWaoFu6FPvHAlUELVQF5g7WRSDrpCF3fgCnsRuY5qKBT'
                'm2gMcSjlEFDcoH9g/jSqvCPoVdDAtnd6A9fncQ1S5GLMM5pl9vm9q/5P9z7O'
                'jEqO3gm+Mb/9cfOo2ktA9FYK/wewlzHd5kA3Ivry1M5MOebCm6U5Bn2r3gAU'
                'RC3xVOXd2VIG6Ot91PBZyRZYshxZDoxvV+zY0/2lQ9ly2KkU0hYFzCw5rgt6'
                'bzN0zP0znArHydWJkulblnNYV/eNm0n/9mMBJSMUaHkcbTnNLaSgcURk2a7c'
                'Ozkm5rDP25s9Z+TSzXSjCZj0bQgHxDBTjCGM0/c0pgIGnKAPy5e01pdKUH9f'
                'hQhXAdPo8zlfsQ7kDfGE2447Seve7PvlySO5DmhOqoVpO7aYevXnT6lUv7HS'
                'k1fhX+L9882Zy7pdhjRw5ARHTGFxQfHFitqDJpd23q5PHCH3WWhNSgZqdCu8'
                'jBdoY3fCvetHZqJKfs0elcvTBy2/BiAal/PyEpof4JW2AucwKfK/X3dOFun7'
                'wd/JR63qsArTrnCh86nqt/ZABNQTWklpqkycnDY3Mo2+KP3rrRxCyUqEpF0X'
                'HRA/zeum7ecJ89u2g2OEdD7hUvBDGvqT93ReuTC+k+usqv05dd/tgbdAwbbj'
                'lo0BXSjit30dw8mmtAwZD6YK386e2VYyWe/PI00/ivNNr32G2li9pRDeNXBM'
                'G60QehrTb9BAknkwJHe5n++JCw8IUPnpyKG7Ady7wvmOtzHlrFQql8q/+5NI'
                '0xs8CRulrM0l0JWdHMyIunJ6P1gszg2LVFEyUX6V5w5nnCLm36rO4jGtv7/u'
                'VJ47zigpGTQTDIS8yH7U6vN6JgiaN+RzntuDGWCFD4WTAcGasu0r178VH0xn'
                'EMI1YQ05D4GHcds7kDqzSgGyuVvkQlIPOvZHhHZA9l1c8KqrNXIxVtGgE2vg'
                'VXe9EPd3pTEUe95xYkPMg2kktVNmgYofDL4us35k+27lVdvb9RyBzo0QW/zw'
                '1udY4n8+MQvp2aVC2dfeY9ITOSN9QvwhnBIU/+CrYk0IZY/z03QuXqanE9ve'
                'j+p+8nkwfsi0zX3vmNlN0ZZ19UTqBZ1bfMLbVNi2IJEIr5ONen1M/vCawsA1'
                'xkJsXoy6mGKbPxcVWf1JEGE4lSMYq5Q+/10bupQTqoa1gRnl/yhBC8auiZNo'
                'mSvrQlhFCDA8iFeJEN0Fb5s0gGBJ2E+RMZViRG70NfjCKKxNvfUs0vnTVzrW'
                '7GmgryuItb+yB4vNG9WYJSokdwM2o82rrycOsZ5qeT8SkDXQvB5FHvNNJLlJ'
                'eKCPA3hfLrZZaHpjSXlcpZjPJhXcxLjDdSTayHOb+7IgtWHezzQ0qOBoke8I'
                'lltbbux2pIwznuJTfnzzBMSdCNFZsG2gP8fYF7SEG/131Z46JulUoSc6cRtO'
                'Mj/TCnb0FNcx4uDeJCPRGHlqaSU0+Pe0fX52jHs1IRNjZzsNmi8+Ompc9Er3'
                'tgoYsj9YjDIa3RJkbG5b/mndtDnzzQ6UuJkqtfyLSA0YvP+72CgqZriAJzJ7'
                'S0iAmNU+rmue3tsFtJfnfAy/m0FW+MEjDJrr4yd3YABeeJe/rajltNQ9wzsz'
                '7jHmCA/PwXF5sUR2Z/c2lpR3V0K9+WP3K557Vwe2Kz6TkWW5KM49MlqifYiq'
                'MNwl6iMm+gt0248HR2OyMI7TXIr12ynr2qbupyPU5P8BOkRGRqYhPLDy6Fz7'
                'IfJKxC493qieuhzcBUXk/WuFGVWStyI0RCcMPHmFFKufibM1KFXpVbc/ytGU'
                'aBN9VjkwzCs8XkovSo+PEebHEiUdjmAdDk5Y3T9CJCxr+l+6O3nFNI50uTjG'
                'gHx8fDZlW1ck+C53fk7+vIN+F/kXmJHvey3PNuQqU3tFdQfbHjP/xQLaCyhr'
                'mgegakRsX9XJnz8Oa5J/nO88lum2z6Xf6uCKMVqZW9+Raxi77FEuIz+YcE6a'
                '2rNd+YvbxddcxXsUPOBjCoWgMjbSUCogDjTGlc80pNsHqvY6xvvHGI0AGhwO'
                'LeBj8FfxvSb5f/NbxsiDLhHSN9Y4SiIxpF1L6FgOPMTI+/rBbbyoqnRUW3HS'
                'xl6pv0ad2FwFC/G7lM4MCXbdOvJBl9xdjnHVb2qBz3fFMQKZeR2762bM09Qy'
                'w3kmENfjcd4FrymijDNCcpmVNwQVkCXnHmRofPqjBer/7MllwEeTvcrkJAsS'
                '47Loha8te37v1m8PW6ngV8OJXnCR++OXWTP7xV6KgbSnj59APOHFL33CIMIi'
                'cUcZbKs3knkGphCDmGNa1Birval9/9ZqNmB+SXm2xaRdq2J4vA5EODtZ2XR2'
                '8v4yjD1sKZ8fO9MUVHpTkg2qCvCctzM3pZu8yqrI2FWqclHOKo17vUzVtN0s'
                'wKxqWMyz/wODg9MmFpzJ7S4hv8labylvLv13DXj9HxtF2Wto0LzQOrnPym5+'
                'WhRP3MQeEk733YKoQtlf6r005ZMCZMo39poc/Mz6UjOXPWxxfAtLaLNnwPo0'
                'Li71TY6/jK09oc1qrdxrXXO7PevqCgTRNClD+GxsZVirfBP2PLljGoN8T/1n'
                '2iMCLfcdJ1d8c6GbiHVmP7a4+bzNKevHdy15wfIDxUH3bNTefim033KN0AMj'
                '+QiGimGW9dvoT67u1u44xd5kUX8pTz2/NDsO+tUyu3wDVD1C576BOiRTn9j0'
                'Jxs3Evyjz6TRg3dslG8c4xZjWB8l/dTauQ9KK4hhDZR+1BSuozkVaaBqietg'
                'ewvgQ1gUfnu6eClmmT2st+VVc+qPknUKXGuE982ev9+uyQmClb6ew3nvrMsD'
                'yaO7FxdjtxOD0Um3Somhnas+f5S9lB/z2yhble/Dt54rHx9ZSyti+lYRm+rv'
                'MZG4O+TRzTnUyFPhNZ/pHYoLFi3oqk9oI9U/4k0b/eOSq7zGLwhmqRTeavWb'
                'Lgg6v52nbG0XIpyKvZQxx5KwE14rEk6V0VoLDL0cCgVoAawB8kOOr8v6VtsO'
                'Jfxm40gwqP40s/pp4/Vsaxz973266O7V20WPjaUR7JNvIOUJaIDvOZT5hu88'
                'rlto9jtEKemrHtI55pQ10L25mun5Hl1GnfFq6CZOwb69Rc1Xhtkr6xknHIr3'
                'l4iPzEJl8+N//WJK90GkCM37VV6rnogd8qZ3wp2f50+o1DuJvi62x2rGr5UU'
                '3bslyfBhB3BRkp/2msXuZexF7/WXNT0RP97Mov0PsWgBQTqb3CWYpbAkyuS8'
                '6rbmeFdeuflXsvg2fNWHvGnBL+iKKg4UttPINGhWcbq6BPX1arY5sfOVWQjf'
                '0OdHgIsw/TmHoSONf0Uk/0c4ZnJ0nPW/0iE7id+IkL5f5efP/8Ip85az4YMD'
                'n2KJhk84U4yhQnBaFyBY4c4V+W68KPRDjd4PjHhaYLniF72QpH90hHkhbFB+'
                'ft7ibpKO1i2obd9e0YMfcdzYuH+ZRP+PCb3p9T9Z8v6fLK1jDBcK/lAosEZA'
                'okk5xPq+PBvfv9UvWvQ0EI1bWv+/KFYI1rtWYBr9vcXcJG5ncgPQ5+9mYc1k'
                '/5u9eWrovzl6qJMYj42UoIOfty23F1ts/PloZf0OO02IIkff7mJOOlp3oLwR'
                '5b+pDPdl/n+N8Aayf5Mg8j9tKjzH/zWRxijxZ3Z2/alnBN9KOmrGycfr0vWd'
                '9r4cdRBvcVfqsyVwzueZPbshvpFlvhr/SV0/oiceZBwR8Vwydjmu/XbjCq5r'
                'w5emM/R1F/nc3NoW8KGIL7G01Nz09on5zgPbzwL+rotemvVfdl9gNTWnDbVE'
                'OqA7xmLxRpRgSpzu6aOm/3XrfcUobdMTtPL+aOR3xG//32Hpp0PXbgyIMmxH'
                'nB42qsr2fndmih3lnYB7PUxRHbkfHYCzvCLbebys3L1XsjmMbuJJ48XBPbG6'
                'T57k1RdCdC5cwKunIhB8/vwK3r3TFmkfKQmEJutjR1IRdB6hr6AOqmfV/VAe'
                'fKi2sd5/9+QFV/mEO1hbMAXkscLsmatGEG0oFe/8IP4cD8We90S+5sdRzWQe'
                '+DKUMRxYD1CMCR5l6jX5w8uSjDJY8dti5kNr7x19OdqVEY5XxhtWN+STjpFR'
                'F1zh3JYJKF4o+hU6KJ0AMl6XRl12ranAHeupdTVC6fqugirpjCGvy7F1Hqyi'
                'PtobI5nai6cJAzYpkOWO5d+X1mCIH75ED6fYhiMlcXPqHItL0vZIbES0BOvg'
                'Otn9BOi2Rf7xE+W+e7cVfcsjecUZookKovpQRnUY/MRsjdZY4nTJAcrsvkBg'
                'vOB0deqgOVM8CBtQR8VvZ/fZD603f+pBtMkIybQvfr8ZfAvsgDB3T978bVUr'
                '/A2u3O7dFp7tw+xw3xoeneeh6peTvkF+Fp0GnV1yACgHZR3XJt8pbhH4ZJG9'
                'O/7kX9wmOi2RP88YNN9zLgLfzuwkDClgJtlq5KAh385sbmvpVBf+mKi4664L'
                'r7zaWiK1C4Q5CGA2B3Va7TQ2NWvbmmwsy01Z2TEDZzpXp+OY1NMmr2Unl8Qa'
                'WwG+5QtwaWBv1aFVo5P1mbdloD4N/6aVs8eZwmxYOdOc7XNsyDxsA9q+eruv'
                'tXt9nW6Gs969Bp999g/If9ocsBulBoy5dsc670/u26ES6hMt2NomDTuSkxi6'
                'qB7yMPSw+6iQynAobnxANt5/EiXVI0B9l9QXQBnblSyD/5Bfyh+7/ezaxt6b'
                'Hl6GXjT4J8wPTTu9xlUo9SnZGmgS2YQ5hYWeM7L3sJR+i6ExLb//lLn5Fxzx'
                '4bxi01Z6oHytJ5kT1plssCIu47dBnWk5Xz6bbmEkG/hVX+zixdbIcE6Wdavd'
                'Mfmi0mL0uTQMU9tib0UUy8eWLmXuRAy23OgrLD+X2vEaa12s6i7+NZ8bpotr'
                'wHljmc6r68KWqsLFR91a6fC90oP0pdIYHUm+5jjiEzmxV4sRfHjxOBGmIkxX'
                'sqJe8lDoKRmRvBZSZYc+sC+V/Q0a09p9XNwBHKL+zllWpk3C41vCF8AwTEx4'
                'R9WR6MbTqt9/695tsjkgiYeegSXOiRjk2CmXuv28eCU+4qkRYpfOIjU2gihb'
                '++WVpOKpFIne17HA4Px6b4G938e2wygPrRaMaotgez195MPE53o7D2oqPiZj'
                'BOhf8j5dVZGLIbNq3cWjk412kMBqOVs+nab84Cs41YNtIkBH4kkPhYbwNDyv'
                'DWL7WUE59asVJnBj34fMYLmVvoMTNBAZMZQfdfliaIZ0ZQnMJwq0yB9WT5zn'
                '9MijiI4FXbrijad4FsRC5m+pt75dPn2fXVikI9kwfTi/jJMFot8GW4Rx04IG'
                'KG9NtH0jKDJmZ8kOfh6AZUtdp57K+JbAs556sIocl2yXCBh9VmO829Op/tsY'
                '/7voTYqFgji/Utz60IcG4Pq6L9HSIR5wyFtU/PECQl+elN580qJu6wFa6Tze'
                'aiosrNwyaZzcZh2PbijVTweqyvI6hfBi7VMVP5Qjp8IrKGR8ycNxi62wy4Kp'
                'ph5+6n83cavGVwKH+eXK0hbYk0MYZrH0g4faYM5syoOT+fnS5xdmY/I26qXx'
                'Y6iouC23povqluz9gPKU6ljAWMNYjS9EI5q2DfmyD/WZMx+uqiA4LWTIvq2o'
                '5Ohg8lP9UZZPjSjiED21MGR7pU+gIGhOUxqSXMfG3SsoG2kcT+j4Chirzj/5'
                'QjLVt+yntvsIX+lgMwPD7MbhU02piHSKcpBblPnabBPo1ApnheO/1hLCNkco'
                '7St67i67b+xJLI94h1/d4HsGhvD8F7MQ7RE=')

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