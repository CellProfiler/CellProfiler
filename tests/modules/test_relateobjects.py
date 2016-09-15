'''test_relateobjects.py - test the RelateObjects module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np
from scipy.ndimage import distance_transform_edt

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.relateobjects as R

PARENT_OBJECTS = 'parentobjects'
CHILD_OBJECTS = 'childobjects'
MEASUREMENT = 'Measurement'
IGNORED_MEASUREMENT = '%s_Foo' % R.C_PARENT


class TestRelateObjects(unittest.TestCase):
    def make_workspace(self, parents, children, fake_measurement=False):
        '''Make a workspace for testing Relate'''
        pipeline = cpp.Pipeline()
        if fake_measurement:
            class FakeModule(cpm.Module):
                def get_measurement_columns(self, pipeline):
                    return [(CHILD_OBJECTS, MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                            (CHILD_OBJECTS, IGNORED_MEASUREMENT, cpmeas.COLTYPE_INTEGER)]

            module = FakeModule()
            module.module_num = 1
            pipeline.add_module(module)
        module = R.Relate()
        module.parent_name.value = PARENT_OBJECTS
        module.sub_object_name.value = CHILD_OBJECTS
        module.find_parent_child_distances.value = R.D_NONE
        module.module_num = 2 if fake_measurement else 1
        pipeline.add_module(module)
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        m = cpmeas.Measurements()
        m.add_image_measurement(cpmeas.GROUP_NUMBER, 1)
        m.add_image_measurement(cpmeas.GROUP_INDEX, 1)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        o = cpo.Objects()
        if parents.shape[1] == 3:
            # IJV format
            o.ijv = parents
        else:
            o.segmented = parents
        object_set.add_objects(o, PARENT_OBJECTS)
        o = cpo.Objects()
        if children.shape[1] == 3:
            o.ijv = children
        else:
            o.segmented = children
        object_set.add_objects(o, CHILD_OBJECTS)
        return workspace, module

    def features_and_columns_match(self, workspace):
        module = workspace.module
        pipeline = workspace.pipeline
        measurements = workspace.measurements
        object_names = [x for x in measurements.get_object_names()
                        if x != cpmeas.IMAGE]
        features = [[feature
                     for feature in measurements.get_feature_names(object_name)
                     if feature not in (MEASUREMENT, IGNORED_MEASUREMENT)]
                    for object_name in object_names]
        columns = module.get_measurement_columns(pipeline)
        self.assertEqual(sum([len(f) for f in features]), len(columns))
        for column in columns:
            index = object_names.index(column[0])
            self.assertTrue(column[1] in features[index])

    def test_01_01_load_matlab_v4(self):
        '''Load a Matlab pipeline with a version 4 relate module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUggpTVXwKs1TMDRTMLCwMjGwMjBWMDIwsFQgGTAwevryMzAwHG'
                'ViYKiY8zbidt5lA4my3UtuLZBKFhXyUHg6qdsw6uBRve2GEhMXrL2WJSRkn'
                'vdKrN9knVB6VqWSXLM3CDHPVl37uNXS+w4zh/u0r9b/9v1/cPh9ijTDh7+C'
                'DKLX8jZFsYUf7XkyKWDnHccpSfkSMw4w8C/+l79mnrP84Xnd5xZKL2I443U'
                'nv5r50ty+9PX3Jte9WKQm09kcmdm2WO5T8zUFNrf1tyua7vD9a3LYdOTwuQ'
                'uqfAYflWqO6H8/90VI6+XdtK/H91bKeAXJ8IRlO9nL13M8e9w7Zcdbjsfli'
                '265Cv9685XF6oqU46uaRzbsXPN8O9tP68rz90eu2Hxjo+rD7Up/VvZPUP8x'
                '+XjfOQnL+Td1CmtiG82FKpjnyGx9q1OlPSlOz9z5uGtE/93VF/TPzefK96r'
                'QM39VrV+5p/QBv+gPa6F5p9WPfOLYekjh1oTKZ/9779ZOZd0+8UBuA3/Oe7'
                'GDLz+FH51cen/lHw31ilMLPMW6o0vuTLO9vzEu9tvqindeNjF5FxW2+MVea'
                'l5XJnf338pHMV4fpjgsVsi7+e306yX7rmu0nJbeLi5Z+Gln5P315l0PSo5H'
                'H51cxndLzjWu4eUXsW8pTztvb+fa91tIfvrG68en75B3un5I2WLOXaHrN8/'
                'vqVTol3VXVH3zqGtu078lH1O+ut9z/z3r7YW3/nL7FvIvb2SOVfmwRit/3a'
                '99C7rVV9tXfj+965fy+f7vMutKbFO3Rn/fa37t99U9P63jz3C7fn76XsXoi'
                '6dFnJ3/rb8Tv+hNcH7u4vp56YH0jXc6j3E/ufx0fZf62/1nbf6Fpy1dmx76'
                'e/vt//H/919d9e25nsexiadVluh9u7CVdxfHvh250//VxJ2+b1l5/no5y7f'
                'ch4fmH53PLsQ//a3dJw05n9t2cp3lh1w/PP2ZX9kW43Hxq01U4JGtZ5bcjT'
                'p75Wu9Qrfd5ijTFPtQC+VX/3/sODrvlXDff8687XXz7hnY6b22+ifOIxZ/F'
                'wAIq5Eo')
        #
        # The Relate module is the fourth in the pipeline:
        # Children = Speckles
        # Parents = Nuclei
        # Calculate per-parent means = yes
        #
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, R.Relate))
        self.assertEqual(module.sub_object_name.value, "Speckles")
        self.assertEqual(module.parent_name.value, "Nuclei")
        self.assertTrue(module.wants_per_parent_means.value)

    def test_01_02_load_v1(self):
        '''Load a pipeline with a version 1 relate module'''
        data = ('eJztm1tP2zAUx90LCIY0sUnTmHixeJomiNINNuBlKTCgEzdBxba3palbvLl'
                '2lTiM7hPsI+1xH2WPe9xHWAwpSb1AQkjbFBIpSo/r3/n7HF+aS7Nbru6U1+'
                'CSosLdcnWhgQmCB0TnDWa2ViHl83DdRDpHdcjoKqzaCL63KSy9hury6uLK6'
                'qIKX6rqCoi35Sq7D53D9BsAxp3jhLPn3a/GXDvn24V9hDjHtGmNgSKYcct/'
                'OfuxbmK9RtCxTmxkeRLd8gptsGqnffnVLqvbBO3pLX9lZ9uzWzVkWvuNLuh'
                '+fYDPEDnC35EUQrfaITrFFmbU5V3/cumlLuOSrsjDj2deHnJSHorOPusrF/'
                'W3gVe/GJC3R776066NaR2f4rqtE4hbevOyFcLfcoi/CcmfsPdN3FxzUi54L'
                'YSflPhJl98yEaK3bMdRGxlfnd6K0o5piRd7FZ3xhXdnusFhS+fGSZR2jEl+'
                'hL2OCLESiuO2+moIn+vhc+BVxHYH6ZbU+UU1IV4L4R9IvLA3GKSMQ9tyJ2i'
                'SfsLyWOjxUwCfnFl1G25Q7c73+MmDPTZaXNj691TKk7A3UEO3CYcVsfjBDW'
                'wigzOzM5R+9nPjEtfdutyke4wzrxUQbT5MSfkS9j63bLhFWE0nl/pJ+elX/'
                'HHHmZy3UkS9KFwa4pPHoTpfSoxLMr5ij14RbJQPKnE4VVFLSXE3iW8Y51EV'
                'yhG1MO/42pGUn0H3832Nf49RFGf+l9R460YQl4b45PV0KaJeFC4N8Y3KeYQ'
                'c32alut7PvEThbhKfFqIX9Xr0vvfnqMZ31+dj3Pic85t4530B3CDi00K4Ub'
                'xuScP1QBQuDfGlcX72s/+UpeS4QcSnhbQzaF5VvzFoEN2y3DvQSfoZdPzbI'
                'e0Oug//AeHmiXiscioeIFAD9dFf2sZD0P3LTWaipslsWk/eT9w4fs54XE7i'
                'gp7HDDJf5w9vRKDtwfsJmoes9gUZ3HM06Pnk04eY1lG7j/6GnZ+MG/7vV9a'
                'fGZdxGZdxyXGaj4v6vw3v/OXiZ3qU4s24jMu40ePCriceg951StjM5gRT9N'
                '8FRT/8ZetvxmXc/eU0H5eG+0mjkreMy7iMy7i7wv3OeZz8/ELY/uftov5nn'
                '07Qev8C9K73wjYQIW2TifeGTKV1/nKLpRCm1y/eLlF2nI8V34smQqcdoqNJ'
                'OtpVOriOKMeNTtt01GzOWjrHhlJxSw+c0nK3dJi6H0N05yTduat0TUR0jpT'
                'D84OXT//4mAzw7+/nvGM9mZ0du25cAdA7nrxx9vdtHL18oZATnP9/sVMhXB'
                'H0jm/B/wE3G8/Pr6nfjTGt9f8Be1L4ug==')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, R.Relate))
        self.assertEqual(module.sub_object_name.value, "Speckles")
        self.assertEqual(module.parent_name.value, "Cells")
        self.assertFalse(module.wants_per_parent_means.value)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925

LoadImages:[module_num:1|svn_version:\'8913\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:
    Do you want to check image sets for missing or duplicate files?:Yes
    Do you want to group image sets by metadata?:No
    Do you want to exclude certain files?:No
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):hoe
    What do you want to call this image in CellProfiler?:Cytoplasm
    What is the position of this image in each group?:1
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path\x3A:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Type the text that these images have in common (case-sensitive):ax2
    What do you want to call this image in CellProfiler?:Speckles
    What is the position of this image in each group?:2
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path\x3A:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

Smooth:[module_num:2|svn_version:\'8664\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Select the input image:Cytoplasm
    Name the output image:SmoothedCytoplasm
    Select smoothing method\x3A:Smooth Keeping Edges
    Calculate object size automatically?:No
    Size of objects\x3A:100.0
    Edge intensity difference\x3A:0.1

Morph:[module_num:3|svn_version:\'8780\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Select input image\x3A:SmoothedCytoplasm
    Name the output image\x3A:DNA
    Select operation to perform\x3A:erode
    Repeat operation\x3A:Custom
    Custom # of repeats:20

IdentifyPrimAutomatic:[module_num:4|svn_version:\'8826\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Select the input image:DNA
    Name the identified primary objects:Nuclei
    Typical diameter of objects, in pixel units (Min,Max)\x3A:50,400
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:Otsu Global
    Threshold correction factor:1.0
    Lower and upper bounds on threshold\x3A:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter\x3A:10
    Suppress local maxima within this distance\x3A:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum size of local maxima?:Yes
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Calculate the Laplacian of Gaussian threshold automatically?:Yes
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter\x3A :5

IdentifySecondary:[module_num:5|svn_version:\'8826\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Select the input objects:Nuclei
    Name the identified objects:Cells
    Select the method to identify the secondary objects:Propagation
    Select the input image:Cytoplasm
    Select the thresholding method:Otsu Global
    Threshold correction factor:1.0
    Lower and upper bounds on threshold\x3A:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Number of pixels by which to expand the primary objects\x3A:10
    Regularization factor\x3A:0.05
    Name the outline image:SecondaryOutlines
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Do you want to discard objects that touch the edge of the image?:No
    Do you want to discard associated primary objects?:No
    New primary objects name\x3A:FilteredNuclei

IdentifyTertiarySubregion:[module_num:6|svn_version:\'8886\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Select the larger identified objects:Cells
    Select the smaller identified objects:Nuclei
    Name the identified subregion objects:Cytoplasm
    Name the outline image:CytoplasmOutlines
    Retain the outlines for use later in the pipeline (for example, in SaveImages)?:No

IdentifyPrimAutomatic:[module_num:7|svn_version:\'8826\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Select the input image:Speckles
    Name the identified primary objects:Speckles
    Typical diameter of objects, in pixel units (Min,Max)\x3A:4,30
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:Otsu Global
    Threshold correction factor:1.0
    Lower and upper bounds on threshold\x3A:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter\x3A:10
    Suppress local maxima within this distance\x3A:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum size of local maxima?:Yes
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Calculate the Laplacian of Gaussian threshold automatically?:Yes
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter\x3A :5

Relate:[module_num:8|svn_version:\'8866\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input child objects:Speckles
    Select the input parent objects:Cells
    Find distances?:Both
    Calculate per-parent means for all child measurements?:No
    Find distances to other parents?:Yes
    Parent name\x3A:Cytoplasm
    Parent name\x3A:Nuclei
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 8)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, R.Relate))
        self.assertEqual(module.sub_object_name, "Speckles")
        self.assertEqual(module.parent_name, "Cells")
        self.assertEqual(module.find_parent_child_distances, R.D_BOTH)
        self.assertFalse(module.wants_per_parent_means)
        self.assertTrue(module.wants_step_parent_distances)
        self.assertEqual(len(module.step_parent_names), 2)
        for group, expected in zip(module.step_parent_names,
                                   ("Cytoplasm", "Nuclei")):
            self.assertEqual(group.step_parent_name, expected)

    def test_02_01_relate_zeros(self):
        '''Relate a field of empty parents to empty children'''
        labels = np.zeros((10, 10), int)
        workspace, module = self.make_workspace(labels, labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 0)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 0)
        self.features_and_columns_match(workspace)

    def test_02_01_relate_one(self):
        '''Relate one parent to one child'''
        parent_labels = np.ones((10, 10), int)
        child_labels = np.zeros((10, 10), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_02_relate_wrong_size(self):
        '''Regression test of IMG-961

        Perhaps someone is trying to relate cells to wells and the grid
        doesn't completely cover the labels matrix.
        '''
        parent_labels = np.ones((20, 10), int)
        parent_labels[10:, :] = 0
        child_labels = np.zeros((10, 20), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_03_relate_ijv(self):
        '''Regression test of IMG-1317: relating objects in ijv form'''

        child_ijv = np.array([[5, 5, 1], [5, 5, 2], [20, 15, 3]])
        parent_ijv = np.array([[5, 5, 1], [20, 15, 2]])
        workspace, module = self.make_workspace(parent_ijv, child_ijv)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 3)
        self.assertTrue(parents_of[0], 1)
        self.assertEqual(parents_of[1], 1)
        self.assertEqual(parents_of[2], 2)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 2)
        self.assertEqual(child_count[0], 2)
        self.assertEqual(child_count[1], 1)

    def test_03_01_mean(self):
        '''Compute the mean for two parents and four children'''
        i, j = np.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = (i / 10).astype(int) + (j / 10).astype(int) * 2 + 1
        workspace, module = self.make_workspace(parent_labels, child_labels,
                                                fake_measurement=True)
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT,
                          np.array([1.0, 2.0, 3.0, 4.0]))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT,
                          np.array([1, 2, 3, 4]))
        expected = np.array([2.0, 3.0])
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(np.all(data == expected))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_03_02_empty_mean(self):
        # Regression test - if there are no children, the per-parent means
        #                   should still be populated
        i, j = np.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = np.zeros(parent_labels.shape, int)
        workspace, module = self.make_workspace(parent_labels, child_labels,
                                                fake_measurement=True)
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT, np.zeros(0))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT, np.zeros(0, int))
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(np.all(np.isnan(data)))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_04_00_distance_empty(self):
        '''Make sure we can handle labels matrices that are all zero'''
        empty_labels = np.zeros((10, 20), int)
        some_labels = np.zeros((10, 20), int)
        some_labels[2:7, 3:8] = 1
        some_labels[3:8, 12:17] = 2
        for parent_labels, child_labels, n in ((empty_labels, empty_labels, 0),
                                               (some_labels, empty_labels, 0),
                                               (empty_labels, some_labels, 2)):
            workspace, module = self.make_workspace(parent_labels, child_labels)
            self.assertTrue(isinstance(module, R.Relate))
            module.find_parent_child_distances.value = R.D_BOTH
            module.run(workspace)
            self.features_and_columns_match(workspace)
            meas = workspace.measurements
            for feature in (R.FF_CENTROID, R.FF_MINIMUM):
                m = feature % PARENT_OBJECTS
                v = meas.get_current_measurement(CHILD_OBJECTS, m)
                self.assertEqual(len(v), n)
                if n > 0:
                    self.assertTrue(np.all(np.isnan(v)))

    def test_04_01_distance_centroids(self):
        '''Check centroid-centroid distance calculation'''
        i, j = np.mgrid[0:14, 0:30]
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        # Centers should be at i=3 and j=7
        parent_centers = np.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        parent_indexes = parent_labels[child_centers[0],
                                       child_centers[1]] - 1
        expected = np.sqrt(np.sum((parent_centers[parent_indexes, :] -
                                   child_centers.transpose()) ** 2, 1))

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_CENTROID
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_CENTROID % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))

    def test_04_02_distance_minima(self):
        '''Check centroid-perimeter distance calculation'''
        i, j = np.mgrid[0:14, 0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i >= 6) * 1 + (j >= 14) * 2 + 1
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        #
        # Measure the distance from the child to the edge of its parent.
        # We do this using the distance transform with a background that's
        # the edges of the labels
        #
        background = ((i != 0) & (i != 5) & (i != 6) & (i != 13) &
                      (j != 0) & (j != 13) & (j != 14) & (j != 29))
        d = distance_transform_edt(background)
        expected = d[child_centers[0], child_centers[1]]

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_MINIMUM
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_MINIMUM % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))

    def test_04_03_means_of_distances(self):
        #
        # Regression test of issue #1409
        #
        # Make sure means of minimum and mean distances of children
        # are recorded properly
        #
        i, j = np.mgrid[0:14, 0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        parent_centers = np.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        parent_indexes = parent_labels[child_centers[0],
                                       child_centers[1]] - 1
        expected = np.sqrt(np.sum((parent_centers[parent_indexes, :] -
                                   child_centers.transpose()) ** 2, 1))

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_CENTROID
        module.wants_per_parent_means.value = True
        mnames = module.get_measurements(workspace.pipeline,
                                         PARENT_OBJECTS,
                                         "_".join((R.C_MEAN, CHILD_OBJECTS)))
        self.assertTrue(R.FF_CENTROID % PARENT_OBJECTS in mnames)
        feat_mean = R.FF_MEAN % (CHILD_OBJECTS, R.FF_CENTROID % PARENT_OBJECTS)
        mcolumns = module.get_measurement_columns(workspace.pipeline)
        self.assertTrue(any([c[0] == PARENT_OBJECTS and c[1] == feat_mean
                             for c in mcolumns]))
        m = workspace.measurements
        m[CHILD_OBJECTS, R.M_LOCATION_CENTER_X, 1] = child_centers[1]
        m[CHILD_OBJECTS, R.M_LOCATION_CENTER_Y, 1] = child_centers[0]
        module.run(workspace)

        v = m[PARENT_OBJECTS, feat_mean, 1]

        plabel = m[CHILD_OBJECTS, "_".join((R.C_PARENT, PARENT_OBJECTS)), 1]

        self.assertEqual(len(v), 4)
        for idx in range(4):
            if np.any(plabel == idx + 1):
                self.assertAlmostEqual(
                        v[idx], np.mean(expected[plabel == idx + 1]), 4)
