'''test_straightenworms - test the StraightenWorms module'''

import itertools
import unittest
from StringIO import StringIO

import centrosome.cpmorphology as morph
import numpy as np

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.identify as I
import cellprofiler.modules.straightenworms as S
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw

OBJECTS_NAME = "worms"
STRAIGHTENED_OBJECTS_NAME = "straightenedworms"
IMAGE_NAME = "wormimage"
STRAIGHTENED_IMAGE_NAME = "straightenedimage"
AUX_IMAGE_NAME = "auximage"
AUX_STRAIGHTENED_IMAGE_NAME = "auxstraightenedimage"


class TestStraightenWorms(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10732

StraightenWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Untangled worms name?:OverlappingWorms
    Straightened worms name?:StraightenedWorms
    Worm width\x3A:20
    Training set file location:Default Output Folder\x7CNone
    Training set file name:TrainingSet.xml
    Image count:2
    Image name:Brightfield
    Straightened image name:StraightenedBrightfield
    Image name:Fluorescence
    Straightened image name:StraightenedFluorescence
'''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, S.StraightenWorms))
        self.assertEqual(module.objects_name, "OverlappingWorms")
        self.assertEqual(module.straightened_objects_name, "StraightenedWorms")
        self.assertEqual(module.width, 20)
        self.assertEqual(module.training_set_directory.dir_choice,
                         cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.training_set_file_name, "TrainingSet.xml")
        self.assertEqual(module.image_count.value, 2)
        for group, input_name, output_name in (
                (module.images[0], "Brightfield", "StraightenedBrightfield"),
                (module.images[1], "Fluorescence", "StraightenedFluorescence")):
            self.assertEqual(group.image_name, input_name)
            self.assertEqual(group.straightened_image_name, output_name)

    def test_01_02_load_v2(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10891

StraightenWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input untangled worm objects:OverlappingWorms
    Name the output straightened worm objects:StraightenedWorms
    Worm width:20
    Training set file location:Default Input Folder\x7CNone
    Training set file name:TrainingSet.mat
    Image count:1
    Measure intensity distribution?:Yes
    Number of segments:4
    Align worms?:Top brightest
    Alignment image:Brightfield
    Select an input image to straighten:Brightfield
    Name the output straightened image:StraightenedImage

StraightenWorms:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input untangled worm objects:OverlappingWorms
    Name the output straightened worm objects:StraightenedWorms
    Worm width:20
    Training set file location:Default Input Folder\x7CNone
    Training set file name:TrainingSet.mat
    Image count:1
    Measure intensity distribution?:Yes
    Number of segments:4
    Align worms?:Bottom brightest
    Alignment image:Brightfield
    Select an input image to straighten:Brightfield
    Name the output straightened image:StraightenedImage

StraightenWorms:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input untangled worm objects:OverlappingWorms
    Name the output straightened worm objects:StraightenedWorms
    Worm width:20
    Training set file location:Default Input Folder\x7CNone
    Training set file name:TrainingSet.mat
    Image count:1
    Measure intensity distribution?:Yes
    Number of segments:4
    Align worms?:Do not align
    Alignment image:Brightfield
    Select an input image to straighten:Brightfield
    Name the output straightened image:StraightenedImage
'''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        for alignment, module in zip((S.FLIP_TOP, S.FLIP_BOTTOM, S.FLIP_NONE),
                                     pipeline.modules()):
            self.assertTrue(isinstance(module, S.StraightenWorms))
            self.assertEqual(module.objects_name, "OverlappingWorms")
            self.assertEqual(module.straightened_objects_name, "StraightenedWorms")
            self.assertEqual(module.width, 20)
            self.assertEqual(module.training_set_directory.dir_choice,
                             cps.DEFAULT_INPUT_FOLDER_NAME)
            self.assertEqual(module.training_set_file_name, "TrainingSet.mat")
            self.assertEqual(len(module.images), 1)
            self.assertTrue(module.wants_measurements)
            self.assertEqual(module.number_of_segments, 4)
            self.assertEqual(module.number_of_stripes, 1)
            self.assertEqual(module.flip_worms, alignment)
            self.assertEqual(module.flip_image, "Brightfield")
            self.assertEqual(module.images[0].image_name, "Brightfield")
            self.assertEqual(module.images[0].straightened_image_name, "StraightenedImage")

    def make_workspace(self, control_points, lengths, radii, image,
                       mask=None, auximage=None):
        '''Create a workspace containing the control point measurements

        control_points - an n x 2 x m array where n is the # of control points,
                         and m is the number of objects.
        lengths - the length of each object
        radii - the radii_from_training defining the radius at each control pt
        image - the image to be straightened
        mask - the mask associated with the image (default = no mask)
        auximage - a second image to be straightnened (default = no second image)
        '''
        module = S.StraightenWorms()
        module.objects_name.value = OBJECTS_NAME
        module.straightened_objects_name.value = STRAIGHTENED_OBJECTS_NAME
        module.images[0].image_name.value = IMAGE_NAME
        module.images[0].straightened_image_name.value = STRAIGHTENED_IMAGE_NAME
        module.flip_image.value = IMAGE_NAME
        module.module_num = 1

        # Trick the module into thinking it's read the data file

        class P:
            def __init__(self):
                self.radii_from_training = radii

        module.training_set_directory.dir_choice = cps.URL_FOLDER_NAME
        module.training_set_directory.custom_path = "http://www.cellprofiler.org"
        module.training_set_file_name.value = "TrainingSet.xml"
        module.training_params = {"TrainingSet.xml": (P(), "URL")}

        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)

        m = cpmeas.Measurements()
        for i, (y, x) in enumerate(control_points):
            for v, f in ((x, S.F_CONTROL_POINT_X),
                         (y, S.F_CONTROL_POINT_Y)):
                feature = "_".join((S.C_WORM, f, str(i + 1)))
                m.add_measurement(OBJECTS_NAME, feature, v)
        feature = "_".join((S.C_WORM, S.F_LENGTH))
        m.add_measurement(OBJECTS_NAME, feature, lengths)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cpi.Image(image, mask))

        if auximage is not None:
            image_set.add(AUX_IMAGE_NAME, cpi.Image(auximage))
            module.add_image()
            module.images[1].image_name.value = AUX_IMAGE_NAME
            module.images[1].straightened_image_name.value = AUX_STRAIGHTENED_IMAGE_NAME

        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        labels = np.zeros(image.shape, int)
        for i in range(control_points.shape[2]):
            if lengths[i] == 0:
                continue
            self.rebuild_worm_from_control_points_approx(
                    control_points[:, :, i], radii, labels, i + 1)
        objects.segmented = labels

        object_set.add_objects(objects, OBJECTS_NAME)

        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, m, image_set_list)
        return workspace, module

    def rebuild_worm_from_control_points_approx(self, control_coords,
                                                worm_radii, labels, idx):
        '''Rebuild a worm from its control coordinates

        Given a worm specified by some control points along its spline,
        reconstructs an approximate binary image representing the worm.

        Specifically, this function generates an image where successive control
        points have been joined by line segments, and then dilates that by a
        certain (specified) radius.

        Inputs:

        control_coords: A N x 2 double array, where each column contains the x
        and y coordinates for a control point.

        worm_radius: Scalar double. Approximate radius of a typical worm; the
        radius by which the reconstructed worm spline is dilated to form the
        final worm.

        Outputs:
        The coordinates of all pixels in the worm in an N x 2 array'''
        index, count, i, j = morph.get_line_pts(control_coords[:-1, 0],
                                                control_coords[:-1, 1],
                                                control_coords[1:, 0],
                                                control_coords[1:, 1])
        #
        # Get rid of the last point for the middle elements - these are
        # duplicated by the first point in the next line
        #
        i = np.delete(i, index[1:])
        j = np.delete(j, index[1:])
        index = index - np.arange(len(index))
        count -= 1
        #
        # Find the control point and within-control-point index of each point
        #
        label = np.zeros(len(i), int)
        label[index[1:]] = 1
        label = np.cumsum(label)
        order = np.arange(len(i)) - index[label]
        frac = order.astype(float) / count[label].astype(float)
        radius = (worm_radii[label] * (1 - frac) +
                  worm_radii[label + 1] * frac)
        iworm_radius = int(np.max(np.ceil(radius)))
        #
        # Get dilation coordinates
        #
        ii, jj = np.mgrid[-iworm_radius:iworm_radius + 1,
                 -iworm_radius:iworm_radius + 1]
        dd = np.sqrt((ii * ii + jj * jj).astype(float))
        mask = ii * ii + jj * jj <= iworm_radius * iworm_radius
        ii = ii[mask]
        jj = jj[mask]
        dd = dd[mask]
        #
        # All points (with repeats)
        #
        i = (i[:, np.newaxis] + ii[np.newaxis, :]).flatten()
        j = (j[:, np.newaxis] + jj[np.newaxis, :]).flatten()
        #
        # We further mask out any dilation coordinates outside of
        # the radius at our point in question
        #
        m = (radius[:, np.newaxis] >= dd[np.newaxis, :]).flatten()
        i = i[m]
        j = j[m]
        #
        # Find repeats by sorting and comparing against next
        #
        order = np.lexsort((i, j))
        i = i[order]
        j = j[order]
        mask = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
        i = i[mask]
        j = j[mask]
        mask = (i >= 0) & (j >= 0) & (i < labels.shape[0]) & (j < labels.shape[1])
        labels[i[mask], j[mask]] = idx

    def test_02_01_straighten_nothing(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)))
        module.run(workspace)

        image = workspace.image_set.get_image(STRAIGHTENED_IMAGE_NAME)
        self.assertFalse(np.any(image.mask))
        objectset = workspace.object_set
        self.assertTrue(isinstance(objectset, cpo.ObjectSet))
        labels = objectset.get_objects(STRAIGHTENED_OBJECTS_NAME).segmented
        self.assertTrue(np.all(labels == 0))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.get_current_image_measurement(
                "_".join((I.C_COUNT, STRAIGHTENED_OBJECTS_NAME))), 0)
        self.assertEqual(len(m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X)), 0)
        self.assertEqual(len(m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y)), 0)

    def test_02_02_straighten_straight_worm(self):
        '''Do a "straightening" that is a 1-1 mapping'''
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21], [15]],
                                   [[23], [15]],
                                   [[25], [15]],
                                   [[27], [15]],
                                   [[29], [15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = False
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        np.testing.assert_almost_equal(pixels, image[16:35, 10:21])

        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.get_current_image_measurement(
                "_".join((I.C_COUNT, STRAIGHTENED_OBJECTS_NAME))), 1)
        v = m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X)
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 5)
        v = m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y)
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 9)
        object_set = workspace.object_set
        objects = object_set.get_objects(STRAIGHTENED_OBJECTS_NAME)
        orig_objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(objects.segmented ==
                               orig_objects.segmented[16:35, 10:21]))

    def test_02_03_straighten_diagonal_worm(self):
        '''Do a straightening on a worm on the 3x4x5 diagonal'''
        r = np.random.RandomState()
        r.seed(23)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[10], [10]],
                                   [[13], [14]],
                                   [[16], [18]],
                                   [[19], [22]],
                                   [[22], [26]]])
        lengths = np.array([20])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 31)
        expected = image[control_points[:, 0, 0], control_points[:, 1, 0]]
        samples = pixels[5:26:5, 5]
        np.testing.assert_almost_equal(expected, samples)

    def test_02_04_straighten_two_worms(self):
        '''Straighten the worms from tests 02_02 and 02_03 together'''
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21, 10], [15, 10]],
                                   [[23, 13], [15, 14]],
                                   [[25, 16], [15, 18]],
                                   [[27, 19], [15, 22]],
                                   [[29, 22], [15, 26]]])
        lengths = np.array([8, 20])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 22)
        self.assertEqual(pixels.shape[0], 31)
        np.testing.assert_almost_equal(pixels[0:19, 0:11], image[16:35, 10:21])
        expected = image[control_points[:, 0, 1], control_points[:, 1, 1]]
        samples = pixels[5:26:5, 16]
        np.testing.assert_almost_equal(expected, samples)

    def test_02_05_straighten_missing_worm(self):
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21, np.nan, 10], [15, np.nan, 10]],
                                   [[23, np.nan, 13], [15, np.nan, 14]],
                                   [[25, np.nan, 16], [15, np.nan, 18]],
                                   [[27, np.nan, 19], [15, np.nan, 22]],
                                   [[29, np.nan, 22], [15, np.nan, 26]]])
        lengths = np.array([8, 0, 20])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 33)
        self.assertEqual(pixels.shape[0], 31)
        np.testing.assert_almost_equal(pixels[0:19, 0:11], image[16:35, 10:21])
        expected = image[control_points[:, 0, 2].astype(int),
                         control_points[:, 1, 2].astype(int)]
        samples = pixels[5:26:5, 27]
        np.testing.assert_almost_equal(expected, samples)

    def test_03_01_get_measurement_columns(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.wants_measurements.value = False
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name, feature_name in (
                (cpmeas.IMAGE, "_".join((I.C_COUNT, STRAIGHTENED_OBJECTS_NAME))),
                (STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X),
                (STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y),
                (STRAIGHTENED_OBJECTS_NAME, I.M_NUMBER_OBJECT_NUMBER)):
            self.assertTrue(any([(o == object_name) and (f == feature_name)
                                 for o, f, t in columns]))

        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], I.C_COUNT)

        categories = module.get_categories(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME)
        self.assertEqual(len(categories), 2)
        self.assertTrue(I.C_LOCATION in categories)
        self.assertTrue(I.C_NUMBER in categories)

        f = module.get_measurements(workspace.pipeline, cpmeas.IMAGE, I.C_COUNT)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], STRAIGHTENED_OBJECTS_NAME)

        f = module.get_measurements(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME, I.C_NUMBER)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], I.FTR_OBJECT_NUMBER)

        f = module.get_measurements(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME, I.C_LOCATION)
        self.assertEqual(len(f), 2)
        self.assertTrue(I.FTR_CENTER_X in f)
        self.assertTrue(I.FTR_CENTER_Y in f)

    def test_03_02_get_measurement_columns_wants_images_vertical(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)),
                                                auximage=np.zeros((20, 10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.wants_measurements.value = True
        module.number_of_segments.value = 5
        module.number_of_stripes.value = 1

        expected_columns = (
            [(OBJECTS_NAME,
              "_".join((S.C_WORM, ftr, image, module.get_scale_name(None, segno))),
              cpmeas.COLTYPE_FLOAT)
             for ftr, image, segno in zip(
                    [S.FTR_MEAN_INTENSITY] * 10 + [S.FTR_STD_INTENSITY] * 10,
                    ([STRAIGHTENED_IMAGE_NAME] * 5 + [AUX_STRAIGHTENED_IMAGE_NAME] * 5) * 2,
                    list(range(5)) * 4)])

        columns = module.get_measurement_columns(workspace.pipeline)
        columns = [column for column in columns if column[0] == OBJECTS_NAME]
        for expected_column in expected_columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for column in columns]))
        for column in columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for expected_column in expected_columns]))

        categories = module.get_categories(workspace.pipeline, OBJECTS_NAME)
        self.assertTrue(S.C_WORM in categories)

        features = module.get_measurements(workspace.pipeline,
                                           OBJECTS_NAME, S.C_WORM)
        self.assertEqual(len(features), 2)
        self.assertTrue(S.FTR_MEAN_INTENSITY in features)
        self.assertTrue(S.FTR_STD_INTENSITY in features)

        for ftr in (S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY):
            images = module.get_measurement_images(workspace.pipeline,
                                                   OBJECTS_NAME,
                                                   S.C_WORM, ftr)
            self.assertEqual(len(images), 2)
            self.assertTrue(STRAIGHTENED_IMAGE_NAME in images)
            self.assertTrue(AUX_STRAIGHTENED_IMAGE_NAME in images)

        for ftr, image in zip([S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY] * 2,
                              [STRAIGHTENED_IMAGE_NAME] * 2 +
                                              [AUX_STRAIGHTENED_IMAGE_NAME] * 2):
            scales = module.get_measurement_scales(workspace.pipeline,
                                                   OBJECTS_NAME,
                                                   S.C_WORM,
                                                   ftr, image)
            self.assertEqual(len(scales), 5)
            for expected_scale in range(5):
                self.assertTrue(module.get_scale_name(None, expected_scale) in scales)

    def test_03_03_get_measurement_columns_horizontal(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)),
                                                auximage=np.zeros((20, 10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.wants_measurements.value = True
        module.number_of_segments.value = 1
        module.number_of_stripes.value = 5
        expected_columns = (
            [(OBJECTS_NAME,
              "_".join((S.C_WORM, ftr, image, module.get_scale_name(segno, None))),
              cpmeas.COLTYPE_FLOAT)
             for ftr, image, segno in zip(
                    [S.FTR_MEAN_INTENSITY] * 10 + [S.FTR_STD_INTENSITY] * 10,
                    ([STRAIGHTENED_IMAGE_NAME] * 5 + [AUX_STRAIGHTENED_IMAGE_NAME] * 5) * 2,
                    list(range(5)) * 4)])

        columns = module.get_measurement_columns(workspace.pipeline)
        columns = [column for column in columns if column[0] == OBJECTS_NAME]
        for expected_column in expected_columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for column in columns]))
        for column in columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for expected_column in expected_columns]))

        categories = module.get_categories(workspace.pipeline,
                                           OBJECTS_NAME)
        self.assertTrue(S.C_WORM in categories)

        features = module.get_measurements(workspace.pipeline,
                                           OBJECTS_NAME, S.C_WORM)
        self.assertEqual(len(features), 2)
        self.assertTrue(S.FTR_MEAN_INTENSITY in features)
        self.assertTrue(S.FTR_STD_INTENSITY in features)

        for ftr in (S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY):
            images = module.get_measurement_images(workspace.pipeline,
                                                   OBJECTS_NAME,
                                                   S.C_WORM, ftr)
            self.assertEqual(len(images), 2)
            self.assertTrue(STRAIGHTENED_IMAGE_NAME in images)
            self.assertTrue(AUX_STRAIGHTENED_IMAGE_NAME in images)

        for ftr, image in zip([S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY] * 2,
                              [STRAIGHTENED_IMAGE_NAME] * 2 +
                                              [AUX_STRAIGHTENED_IMAGE_NAME] * 2):
            scales = module.get_measurement_scales(workspace.pipeline,
                                                   OBJECTS_NAME,
                                                   S.C_WORM,
                                                   ftr, image)
            self.assertEqual(len(scales), 5)
            for expected_scale in range(5):
                self.assertTrue(module.get_scale_name(expected_scale, None) in scales)

    def test_03_04_get_measurement_columns_both(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)),
                                                auximage=np.zeros((20, 10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.wants_measurements.value = True
        module.number_of_segments.value = 2
        module.number_of_stripes.value = 3
        expected_columns = []
        vscales = (None, 0, 1)
        hscales = (None, 0, 1, 2)
        for image in (STRAIGHTENED_IMAGE_NAME, AUX_STRAIGHTENED_IMAGE_NAME):
            for ftr in (S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY):
                for vscale in vscales:
                    for hscale in hscales:
                        if vscale is None and hscale is None:
                            continue
                        meas = "_".join((S.C_WORM, ftr, image,
                                         module.get_scale_name(hscale, vscale)))
                        expected_columns.append(
                                (OBJECTS_NAME, meas, cpmeas.COLTYPE_FLOAT))
        columns = module.get_measurement_columns(workspace.pipeline)
        columns = [column for column in columns if column[0] == OBJECTS_NAME]
        for expected_column in expected_columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for column in columns]))
        for column in columns:
            self.assertTrue(any([all([x == y
                                      for x, y in zip(column, expected_column)])
                                 for expected_column in expected_columns]))
        for ftr in (S.FTR_MEAN_INTENSITY, S.FTR_STD_INTENSITY):
            for image in (STRAIGHTENED_IMAGE_NAME, AUX_STRAIGHTENED_IMAGE_NAME):
                scales = module.get_measurement_scales(
                        workspace.pipeline, OBJECTS_NAME, S.C_WORM, ftr, image)
                self.assertEqual(len(scales), 11)
                for vscale in vscales:
                    for hscale in hscales:
                        if vscale is None and hscale is None:
                            continue
                        self.assertTrue(module.get_scale_name(hscale, vscale) in scales)

    def test_04_00_measure_no_worms(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)))
        module.wants_measurements.value = True
        module.number_of_segments.value = 5
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for i in range(5):
            for ftr, function in (
                    (S.FTR_MEAN_INTENSITY, np.mean),
                    (S.FTR_STD_INTENSITY, np.std)):
                mname = "_".join((S.C_WORM, ftr,
                                  STRAIGHTENED_IMAGE_NAME,
                                  module.get_scale_name(None, i)))
                v = m.get_current_measurement(OBJECTS_NAME, mname)
                self.assertEqual(len(v), 0)

    def test_04_01_measure_one_worm(self):
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21], [15]],
                                   [[24], [15]],
                                   [[27], [15]],
                                   [[30], [15]],
                                   [[33], [15]]])
        lengths = np.array([12])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = True
        module.number_of_segments.value = 4
        module.number_of_stripes.value = 3
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        oo = workspace.object_set.get_objects(OBJECTS_NAME)
        #
        # The worm goes from 20 to 34. Each segment is 15 / 4 = 3 3/4 long
        #
        # 20, 21, and 22 are in 1 as is 3/4 of 23
        # 1/4 of 23 is in 2 as is 24, 25 and 26 and 1/2 of 27
        # 1/2 of 27 is in 3 as is 28, 29, 30 and 1/4 of 31
        # 3/4 of 31 is in 4 as is 32, 33, and 34
        #
        segments = [
            [(20, 1.0), (21, 1.0), (22, 1.0), (23, .75)],
            [(23, .25), (24, 1.0), (25, 1.0), (26, 1.0), (27, .5)],
            [(27, .50), (28, 1.0), (29, 1.0), (30, 1.0), (31, .25)],
            [(31, .75), (32, 1.0), (33, 1.0), (34, 1.0)]]

        def weighted_mean(img, segments, mask):
            accumulator = 0.0
            weight_accumulator = 0.0
            for i, w in segments:
                piece = img[i, mask[i, :]]
                accumulator += np.sum(piece) * w
                weight_accumulator += w * np.sum(mask[i, :])
            return accumulator / weight_accumulator

        def weighted_std(img, segments, mask):
            mean = weighted_mean(img, segments, mask)
            accumulator = 0.0
            weight_accumulator = 0.0
            pixel_count = 0.0
            for i, w in segments:
                piece = img[i, mask[i, :]]
                accumulator += np.sum((piece - mean) ** 2) * w
                weight_accumulator += w * np.sum(mask[i, :])
                pixel_count += np.sum(mask[i, :])
            return np.sqrt(accumulator / weight_accumulator /
                           (pixel_count - 1) * pixel_count)

        for i, segment in enumerate(segments):
            for ftr, function in (
                    (S.FTR_MEAN_INTENSITY, weighted_mean),
                    (S.FTR_STD_INTENSITY, weighted_std)):
                mname = "_".join((S.C_WORM, ftr,
                                  STRAIGHTENED_IMAGE_NAME,
                                  module.get_scale_name(None, i)))
                v = m.get_current_measurement(OBJECTS_NAME, mname)
                expected = function(image, segment, oo.segmented == 1)
                self.assertEqual(len(v), 1)
                self.assertAlmostEqual(v[0], expected)

    def test_04_02_measure_checkerboarded_worm(self):
        r = np.random.RandomState()
        r.seed(42)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21], [15]],
                                   [[24], [15]],
                                   [[27], [15]],
                                   [[30], [15]],
                                   [[33], [15]]])
        lengths = np.array([12])
        radii = np.array([1, 4, 7, 4, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 21
        module.wants_measurements.value = True
        module.number_of_segments.value = 4
        module.number_of_stripes.value = 3
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        oo = workspace.object_set.get_objects(OBJECTS_NAME)
        image = workspace.image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        f1 = 1.0 / 3.0
        f2 = 2.0 / 3.0
        stripes = [
            (9, ((10, 11, 2, 1),)),
            (10, ((8, 9, 1, 1), (9, 10, 1, f2), (9, 10, 2, f1), (10, 11, 2, 1), (11, 12, 2, f1), (11, 12, 3, f2),
                  (12, 13, 3, 1))),
            (11, ((7, 9, 1, 1), (9, 10, 1, f1), (9, 10, 2, f2), (10, 11, 2, 1), (11, 12, 2, f2), (11, 12, 3, f1),
                  (12, 14, 3, 1))),
            (12, ((6, 9, 1, 1), (9, 12, 2, 1), (12, 15, 3, 1))),
            (13, ((5, 8, 1, 1), (8, 9, 1, f2), (8, 9, 2, f1), (9, 12, 2, 1), (12, 13, 2, f1), (12, 13, 3, f2),
                  (13, 16, 3, 1))),
            (14, ((4, 8, 1, 1), (8, 9, 1, f1), (8, 9, 2, f2), (9, 12, 2, 1), (12, 13, 2, f2), (12, 13, 3, f1),
                  (13, 17, 3, 1))),
            (15, ((4, 8, 1, 1), (8, 13, 2, 1), (13, 17, 3, 1))),
            (16, ((3, 8, 1, 1), (8, 13, 2, 1), (13, 18, 3, 1))),
            (17, ((4, 8, 1, 1), (8, 13, 2, 1), (13, 17, 3, 1))),
            (18, ((4, 8, 1, 1), (8, 9, 1, f1), (8, 9, 2, f2), (9, 12, 2, 1), (12, 13, 2, f2), (12, 13, 3, f1),
                  (13, 17, 3, 1))),
            (19, ((5, 8, 1, 1), (8, 9, 1, f2), (8, 9, 2, f1), (9, 12, 2, 1), (12, 13, 2, f1), (12, 13, 3, f2),
                  (13, 16, 3, 1))),
            (20, ((6, 9, 1, 1), (9, 12, 2, 1), (12, 15, 3, 1))),
            (21, ((7, 9, 1, 1), (9, 10, 1, f1), (9, 10, 2, f2), (10, 11, 2, 1), (11, 12, 2, f2), (11, 12, 3, f1),
                  (12, 14, 3, 1))),
            (22, ((8, 9, 1, 1), (9, 10, 1, f2), (9, 10, 2, f1), (10, 11, 2, 1), (11, 12, 2, f1), (11, 12, 3, f2),
                  (12, 13, 3, 1))),
            (23, ((10, 11, 2, 1),))]
        segments = [
            [(9, 1.0), (10, 1.0), (11, 1.0), (12, .75)],
            [(12, .25), (13, 1.0), (14, 1.0), (15, 1.0), (16, .5)],
            [(16, .50), (17, 1.0), (18, 1.0), (19, 1.0), (20, .25)],
            [(20, .75), (21, 1.0), (22, 1.0), (23, 1.0)]]

        i_w = np.zeros((image.shape[0], image.shape[1], 4))
        j_w = np.zeros((image.shape[0], image.shape[1], 3))
        mask = np.zeros(image.shape, bool)
        for i, sstripes in stripes:
            for jstart, jend, idx, w in sstripes:
                for j in range(jstart, jend):
                    j_w[i, j, idx - 1] = w
                    mask[i, j] = True

        for idx, segment in enumerate(segments):
            for i, w in segment:
                i_w[i, mask[i, :], idx] = w

        s2 = lambda x: np.sum(np.sum(x, 0), 0)
        weights = s2(i_w)
        expected_means = s2(image[:, :, np.newaxis] * i_w) / weights
        counts = s2(i_w > 0)
        expected_sds = np.sqrt(
                s2(i_w * (image[:, :, np.newaxis] - expected_means[np.newaxis, np.newaxis, :]) ** 2) /
                weights * counts / (counts - 1))
        for i in range(4):
            for ftr, expected in ((S.FTR_MEAN_INTENSITY, expected_means),
                                  (S.FTR_STD_INTENSITY, expected_sds)):
                value = m.get_current_measurement(
                        OBJECTS_NAME,
                        "_".join((S.C_WORM, ftr, STRAIGHTENED_IMAGE_NAME,
                                  module.get_scale_name(None, i))))
                self.assertEqual(len(value), 1)
                self.assertAlmostEqual(value[0], expected[i])
        weights = s2(j_w)
        expected_means = s2(image[:, :, np.newaxis] * j_w) / weights
        counts = s2(j_w > 0)
        expected_sds = np.sqrt(
                s2(j_w * (image[:, :, np.newaxis] -
                          expected_means[np.newaxis, np.newaxis, :]) ** 2) /
                weights * counts / (counts - 1))
        for i in range(3):
            for ftr, expected in ((S.FTR_MEAN_INTENSITY, expected_means),
                                  (S.FTR_STD_INTENSITY, expected_sds)):
                value = m.get_current_measurement(
                        OBJECTS_NAME,
                        "_".join((S.C_WORM, ftr, STRAIGHTENED_IMAGE_NAME,
                                  module.get_scale_name(i, None))))
                self.assertEqual(len(value), 1)
                self.assertAlmostEqual(value[0], expected[i])

        ww = i_w[:, :, :, np.newaxis] * j_w[:, :, np.newaxis, :]
        weights = s2(ww)
        expected_means = s2(image[:, :, np.newaxis, np.newaxis] * ww) / weights
        counts = s2(ww > 0)
        expected_sds = np.sqrt(
                s2(ww * (image[:, :, np.newaxis, np.newaxis] -
                         expected_means[np.newaxis, np.newaxis, :, :]) ** 2) /
                weights * counts / (counts - 1))
        for stripe in range(3):
            for segment in range(4):
                for ftr, expected in ((S.FTR_MEAN_INTENSITY, expected_means),
                                      (S.FTR_STD_INTENSITY, expected_sds)):
                    mname = "_".join((S.C_WORM, ftr, STRAIGHTENED_IMAGE_NAME,
                                      module.get_scale_name(stripe, segment)))
                    value = m.get_current_measurement(OBJECTS_NAME, mname)
                    self.assertEqual(len(value), 1)
                    self.assertAlmostEqual(value[0], expected[segment, stripe])

    def test_05_01_flip_no_worms(self):
        workspace, module = self.make_workspace(np.zeros((5, 2, 0)),
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20, 10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.wants_measurements.value = True
        module.number_of_segments.value = 5
        module.flip_worms.value = S.FLIP_TOP
        module.run(workspace)

    def test_05_02_flip_dont_flip_top(self):
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        image[25:] /= 5.0
        control_points = np.array([[[21], [15]],
                                   [[23], [15]],
                                   [[25], [15]],
                                   [[27], [15]],
                                   [[29], [15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = True
        module.number_of_segments.value = 3
        module.flip_worms.value = S.FLIP_TOP
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        np.testing.assert_almost_equal(pixels, image[16:35, 10:21])

    def test_05_03_flip_top(self):
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        image[:25] /= 5.0
        control_points = np.array([[[21], [15]],
                                   [[23], [15]],
                                   [[25], [15]],
                                   [[27], [15]],
                                   [[29], [15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = True
        module.number_of_segments.value = 3
        module.flip_worms.value = S.FLIP_TOP
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        i, j = np.mgrid[34:15:-1, 20:9:-1]
        np.testing.assert_almost_equal(pixels, image[i, j])

    def test_05_04_flip_dont_flip_bottom(self):
        r = np.random.RandomState()
        r.seed(53)
        image = r.uniform(size=(60, 30))
        image[:25] /= 5.0
        control_points = np.array([[[21], [15]],
                                   [[23], [15]],
                                   [[25], [15]],
                                   [[27], [15]],
                                   [[29], [15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = True
        module.number_of_segments.value = 3
        module.flip_worms.value = S.FLIP_BOTTOM
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        np.testing.assert_almost_equal(pixels, image[16:35, 10:21])

    def test_05_05_flip_bottom(self):
        r = np.random.RandomState()
        r.seed(54)
        image = r.uniform(size=(60, 30))
        image[25:] /= 5.0
        control_points = np.array([[[21], [15]],
                                   [[23], [15]],
                                   [[25], [15]],
                                   [[27], [15]],
                                   [[29], [15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.wants_measurements.value = True
        module.number_of_segments.value = 3
        module.flip_worms.value = S.FLIP_BOTTOM
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        i, j = np.mgrid[34:15:-1, 20:9:-1]
        np.testing.assert_almost_equal(pixels, image[i, j])
