'''test_measurecorrelation - test the MeasureCorrelation module'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.measurecorrelation as M

IMAGE1_NAME = 'image1'
IMAGE2_NAME = 'image2'
OBJECTS_NAME = 'objects'


class TestMeasureCorrelation(unittest.TestCase):
    def make_workspace(self, image1, image2, objects=None):
        '''Make a workspace for testing ApplyThreshold'''
        module = M.MeasureCorrelation()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        for image_group, name, image in zip(module.image_groups,
                                            (IMAGE1_NAME, IMAGE2_NAME),
                                            (image1, image2)):
            image_group.image_name.value = name
            image_set.add(name, image)
        object_set = cpo.ObjectSet()
        if objects is None:
            module.images_or_objects.value = M.M_IMAGES
        else:
            module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
            module.object_groups[0].object_name.value = OBJECTS_NAME
            object_set.add_objects(objects, OBJECTS_NAME)
        pipeline = cpp.Pipeline()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline with a MeasureCorrelation module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUgjJKFXwKs1TMDRUMLC0MjS1MjZTMDIwsFQgGTAwevryMzAwSD'
                'EzMFTMeRtxN++WgYiZQMT10NjrWZeWFB58qhfBfre/bGefn/Qlroy1Wr63j'
                'voYXJmcvPjIyoT8zbttj8ml92+UEOLQSzjO0rYsctOaxz9qns95f+O45zdG'
                'hr/HGUSaCtVfRBrk/Qw4d2TXBH2B5zO+7bHZcGBxwb/wC+sXdCfOXqDtUfO'
                'k0bP72estp/Xa/55cXHztI2eWg/o+iwze+5nK17PLlI9+KOTmF9h54L7gjN'
                'yLm/xfsFXPuBhgf2R6unqZ5PQ379hrC+pPObs9Osj1KzAnbk0987LXay33y'
                'B65Jt/yV4j3x+Vy5ozPk5h31uoWaHHNucPjxzfH1M/9W7z2F65Jfw91xM9W'
                'sPlWH1tWfmBNrOL1dYpLO+Ysebjdy17/JffONg5Byesnb8d6xp/Yqphey2S'
                'rPie+1X7qPNPF6Y3hT8VFZfe/96zj3vl0k90bRzmZps77WnNiOd1PpK9xkd'
                'T/KTfr2P5DditmffhuabF5WdlsvXd7eb2CZ31cXXDplvtNyTk6GQa2vz9/c'
                '55cXffSNjHXQ+2b77ngu422S042681f9ttx6fSZF083rEq7s8pqStgdLZm7'
                'NmpP1wety1syb9nU/66VPlH+2senvlL7vH3dX/mn+pl/V4T9y7s3L/uV7qn'
                'ZjfdO557ZzpV+72/gcplA/s+KHvfnZ/xZcKBa4SGn5GW+fQdl5ScJiYU3ix'
                '/21va7XF0UWiv837/2XJN4Tc7xN6eTLxUX72Gab/VvxoNvMRZfcjwCeI5/k'
                'fM/dV5JJlTG83zzOvtlv9Nj/534cyHd4OqfNae6ez5/Pvfp/fuKu56b8p9/'
                'tNYKsp3+s7Z2TuX7xDV/eW/WcE1ZZn8xI6g/uW5q062eX9lHvLXjRM/t+a8'
                'ZPGdjtFqpbqadmez3bZ+XV+v5VPq8vvO3///9vni50NJ3AYdEjT5bHhS5q1'
                'Vy+s/9a4V3P9af1dm3/ul7g8tW3yce9LP95rhQ+LozMMS/8i2SKW/XFbf+W'
                'J54+bfBod0OZotlXxzbH+xTYe9+4XNsSPK67hNfyt+u/b66qel5joKs3Nu3'
                'y77aux6Pq9OfPvm/8sovawr+eW+M/BOoI976L39Cjr1t5Ca7rNu7/vE73Wj'
                '5BgDjormq')
        #
        # 4 modules, MeasureCorrelation is last
        #
        # Images:
        #    DNA
        #    Cytoplasm
        # Measure images and objects
        # Objects:
        #    Nuclei
        #    Cells
        #
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    def test_01_02_load_v1(self):
        '''Load a version-1 MeasureCorrelation module'''
        data = ('eJztW++O2kYQXzju2muk6FKpatR82Y+59kCG5NoLqi5Q6B/ag6AcTRRFabs'
                'HC2y17CJ7fTlaRcoj9RH6GH2Efswj1As2NlsfNmBzENmSBTPsb2ZndnZmbO'
                'N6uXVW/gYe5zRYL7eyXUIxbFIkulwfFCETR7CiYyRwB3JWhK2+CX80Gczno'
                'XZSfHBc1AqwoGmPwHJHqla/bX30TwDYsz4/tM60/dOuTac8p6TPsRCE9Yxd'
                'kAF3bf7f1vkM6QRdUPwMURMbrgqHX2Nd3hoNpz/VecekuIEG3sHW0TAHF1g'
                '3nnQdoP1zk1xhek7+wIoJzrCn+JIYhDMbb8tXuVO9XCh6pR/++sz1Q0rxQ8'
                'Y673n4cvwPwB2f8fHbHc/4A5smrEMuScdEFJIB6k1nIeVpAfJ2ZuTtgGqjP'
                'MadBOD2lHnsjf3cppiE05uawadAwZ5vKQB3oOiVZwtfiey3V6gt4ACJdj+K'
                '+QfhdxW8pCuYUiOk36+zf1Hcg5C4zAwuA748eqiF8fctxU5JN3U+RD0krE0'
                'w5scZZ35+rp2d/VxfUu8La3eEsXtf0SvpykjwIUXGwOZHKSfIjvSMnDRo8N'
                'VwQfP2W/cnwjDh95RfIDqdd1C++liRI+naOElBxKwCdPE7bgsDhJb3qSJP0'
                'lXcRSYVcCwXVoluieT6aKX4WBSXz2mR5809Be8cDn4fuH4rBeiNaj396o+W'
                '08bHUd7+4plX3Hk06vVbNI9atudXWbdl7Yt6nRbNI3lttXgNigtvv3Rg05U'
                '+YgzTQlTrFBJ3vM71depiKQDnV1dqTGBmEDHy6A+S85EiR9JVDhkX0DSwvx'
                '1R2q/2M/mQ8w5r/6L+13zyQJT2qvHV4AxHtf+W1beIfW8D9P0EZtdF0r/cf'
                '9z8Wl6A4tPcF4e/Suq5leKf8tenL8vZ5qtDh1Ph1Byw05da9tGrP/NHhTeT'
                'wefEQo6Zh6H9rMbVVyFxYfqsRfzVD9B3ovhL0tLmFxjptiMevjnMSladM9G'
                '3eQWbV0UjlxNnP+zXP7Rec9i2+ljDvuK8oXqQjzLPLVNXn2PS68vbKZfyxg'
                'FrO/cT4uznorquWDYe/Pz5HddxT+cm66xuf1zXz2H6xWXnuQl1YlPsexswz'
                '02pE3HW/22uE1H3DZu2v6Ku85tm36bkgXXZp+WOb3yecd+vibLfWjduU/qq'
                'TVvndfZRi+D+ueviUgrO77nVOv0zfsglHTQML8dvP01uPruCtmVfeOYNCev'
                'gYYzytmWf3XT8beu8t9X+BBddX7EJeTCxN8EluOtxJTA/zg/AbJzL083Tk7'
                'IeZ5/i95yfm4IShv/XWGyT3993XAnMX9ek/ie4OHDbnk9KAfOPOx8nuASX4'
                'FbHlTy4pP4luASX4BLc+497l3Jx6vMNSXuff8vxv3n0+NWJz8FsnZB0G1M6'
                '1Ll8/0rPDcYvCRk5ylFn8pZO7sz6WvO8sCP1DAP0lBQ9pev0kA5mgnRHQ93'
                'SZgo+QIK0czWb27S4ZYcr9fYD9Pr9/2CuXgO3OesgfTTVee5wwthZVPQVr9'
                'M3wMgwdUu0rmM6ficlV5+wKi7L9a83XvZ99HrXPW1Rn9y788G8OANgNr7cu'
                'Hv3eBl96Z106jaY/T/crQBcBszGu8T/CxaL7/tzxjs2bur4/wDdHk0H')
        #
        # 4 modules, MeasureCorrelation is last
        #
        # Images:
        #    DNA
        #    Cytoplasm
        # Measure images and objects
        # Objects:
        #    Nuclei
        #    Cells
        #
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8905

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:ILLUM
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:
    Do you want to check image sets for missing or duplicate files?:Yes
    Do you want to group image sets by metadata?:Yes
    Do you want to exclude certain files?:Yes
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):Channel2
    What do you want to call this image in CellProfiler?:DNA
    What is the position of this image in each group?:1
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:None
    Type the regular expression that finds metadata in the subfolder path\x3A:None
    Type the text that these images have in common (case-sensitive):Channel1
    What do you want to call this image in CellProfiler?:Cytoplasm
    What is the position of this image in each group?:2
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:None
    Type the regular expression that finds metadata in the subfolder path\x3A:None

IdentifyPrimAutomatic:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the identified primary objects:Nuclei
    Typical diameter of objects, in pixel units (Min,Max)\x3A:6,40
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
    Name the outline image:Do not use
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum size of local maxima?:Yes
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Calculate the Laplacian of Gaussian threshold automatically?:Yes
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter\x3A :5

IdentifySecondary:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
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
    Name the outline image:Do not use
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Do you want to discard objects that touch the edge of the image?:No
    Do you want to discard associated primary objects?:No
    New primary objects name\x3A:FilteredNuclei

MeasureCorrelation:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Hidden:2
    Hidden:2
    Select an image to measure:DNA
    Select an image to measure:Cytoplasm
    Select where to measure correlation:Both
    Select an object to measure:Nuclei
    Select an object to measure:Cells
"""
        fd = StringIO(data)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.thr, 15.0)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160216135025
GitHash:e55aeba
ModuleCount:1
HasImagePlaneDetails:False

MeasureCorrelation:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:2
    Hidden:2
    Select an image to measure:DNA
    Select an image to measure:Cytoplasm
    Set threshold as percentage of maximum intensity for the images:25.0
    Select where to measure correlation:Both
    Select an object to measure:Nuclei
    Select an object to measure:Cells
"""
        fd = StringIO(data)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.thr, 25.0)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    all_object_measurement_formats = [
        M.F_CORRELATION_FORMAT, M.F_COSTES_FORMAT, M.F_K_FORMAT,
        M.F_MANDERS_FORMAT, M.F_OVERLAP_FORMAT, M.F_RWC_FORMAT]
    all_image_measurement_formats = all_object_measurement_formats + [
        M.F_SLOPE_FORMAT]
    asymmetrical_measurement_formats = [
        M.F_COSTES_FORMAT, M.F_K_FORMAT, M.F_MANDERS_FORMAT, M.F_RWC_FORMAT]

    def test_02_01_get_categories(self):
        '''Test the get_categories function for some different cases'''
        module = M.MeasureCorrelation()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES

        def cat(name):
            return module.get_categories(None, name) == ["Correlation"]

        self.assertTrue(cat("Image"))
        self.assertFalse(cat(OBJECTS_NAME))
        module.images_or_objects.value = M.M_OBJECTS
        self.assertFalse(cat("Image"))
        self.assertTrue(cat(OBJECTS_NAME))
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        self.assertTrue(cat("Image"))
        self.assertTrue(cat(OBJECTS_NAME))

    def test_02_02_get_measurements(self):
        '''Test the get_measurements function for some different cases'''
        module = M.MeasureCorrelation()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES

        def meas(name):
            ans = list(module.get_measurements(None, name, "Correlation"))
            ans.sort()
            if name == "Image":
                mf = self.all_image_measurement_formats
            else:
                mf = self.all_object_measurement_formats
            expected = sorted([_.split("_")[1] for _ in mf])
            return ans == expected

        self.assertTrue(meas("Image"))
        self.assertFalse(meas(OBJECTS_NAME))
        module.images_or_objects.value = M.M_OBJECTS
        self.assertFalse(meas("Image"))
        self.assertTrue(meas(OBJECTS_NAME))
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        self.assertTrue(meas("Image"))
        self.assertTrue(meas(OBJECTS_NAME))

    def test_02_03_get_measurement_images(self):
        '''Test the get_measurment_images function for some different cases'''
        for iocase, names in (
                (M.M_IMAGES, [cpmeas.IMAGE]),
                (M.M_OBJECTS, [OBJECTS_NAME]),
                (M.M_IMAGES_AND_OBJECTS, [cpmeas.IMAGE, OBJECTS_NAME])):
            module = M.MeasureCorrelation()
            module.image_groups[0].image_name.value = IMAGE1_NAME
            module.image_groups[1].image_name.value = IMAGE2_NAME
            module.object_groups[0].object_name.value = OBJECTS_NAME
            module.images_or_objects.value = iocase
            for name, mfs in ((cpmeas.IMAGE, self.all_image_measurement_formats),
                              (OBJECTS_NAME, self.all_object_measurement_formats)):
                if name not in names:
                    continue
                for mf in mfs:
                    ftr = mf.split("_")[1]
                    ans = module.get_measurement_images(
                            None, name, "Correlation", ftr)
                    expected = ["%s_%s" % (i1, i2) for i1, i2 in
                                ((IMAGE1_NAME, IMAGE2_NAME),
                                 (IMAGE2_NAME, IMAGE1_NAME))]
                    if mf in self.asymmetrical_measurement_formats:
                        self.assertTrue(all([e in ans for e in expected]))
                    else:
                        self.assertTrue(any([e in ans for e in expected]))

    def test_02_04_01_get_measurement_columns_images(self):
        module = M.MeasureCorrelation()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES
        columns = module.get_measurement_columns(None)
        expected = [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_image_measurement_formats] + [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_02_04_02_get_measurement_columns_objects(self):
        module = M.MeasureCorrelation()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_OBJECTS
        columns = module.get_measurement_columns(None)
        expected = [
                       (OBJECTS_NAME,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_object_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_02_04_03_get_measurement_columns_both(self):
        module = M.MeasureCorrelation()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        columns = module.get_measurement_columns(None)
        expected = [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_image_measurement_formats] + [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_object_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]

        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_03_01_correlated(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        i1 = cpi.Image(image)
        i2 = cpi.Image(image)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

        self.assertEqual(len(m.get_object_names()), 1)
        self.assertEqual(m.get_object_names()[0], cpmeas.IMAGE)
        columns = module.get_measurement_columns(None)
        features = m.get_feature_names(cpmeas.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_03_02_anticorrelated(self):
        '''Test two anticorrelated images'''
        #
        # Make a checkerboard pattern and reverse it for one image
        #
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = 1 - image1
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, -1)

    def test_04_01_slope(self):
        '''Test the slope measurement'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(10, 10)).astype(np.float32)
        image2 = image1 * .5
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Slope")
        slope = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Slope_%s" % mi[0])
        if mi[0] == "%s_%s" % (IMAGE1_NAME, IMAGE2_NAME):
            self.assertAlmostEqual(slope, .5, 5)
        else:
            self.assertAlmostEqual(slope, 2)

    def test_05_01_crop(self):
        '''Test similarly cropping one image to another'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        crop_mask = np.zeros((20, 20), bool)
        crop_mask[5:16, 5:16] = True
        i2 = cpi.Image(image1[5:16, 5:16], crop_mask=crop_mask)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

    def test_05_02_mask(self):
        '''Test images with two different masks'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        mask1 = np.ones((20, 20), bool)
        mask1[5:8, 8:12] = False
        mask2 = np.ones((20, 20), bool)
        mask2[14:18, 2:5] = False
        mask = mask1 & mask2
        image2 = image1.copy()
        #
        # Try to confound the module by making masked points anti-correlated
        #
        image2[~mask] = 1 - image1[~mask]
        i1 = cpi.Image(image1, mask=mask1)
        i2 = cpi.Image(image2, mask=mask2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

    def test_06_01_objects(self):
        '''Test images with two objects'''
        labels = np.zeros((10, 10), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = image1.copy()
        #
        # Anti-correlate the second object
        #
        image2[labels == 2] = 1 - image1[labels == 2]
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME,
                                           "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertEqual(len(corr), 2)
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], -1)

        self.assertEqual(len(m.get_object_names()), 2)
        self.assertTrue(OBJECTS_NAME in m.get_object_names())
        columns = module.get_measurement_columns(None)
        image_features = m.get_feature_names(cpmeas.IMAGE)
        object_features = m.get_feature_names(OBJECTS_NAME)
        self.assertEqual(len(columns), len(image_features) + len(object_features))
        for column in columns:
            if column[0] == cpmeas.IMAGE:
                self.assertTrue(column[1] in image_features)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in object_features)

    def test_06_02_cropped_objects(self):
        '''Test images and objects with a cropping mask'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        crop_mask = np.zeros((20, 20), bool)
        crop_mask[5:15, 5:15] = True
        i2 = cpi.Image(image1[5:15, 5:15], crop_mask=crop_mask)
        labels = np.zeros((10, 10), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        o = cpo.Objects()
        o.segmented = labels
        #
        # Make the objects have the cropped image as a parent
        #
        o.parent_image = i2
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], 1)

    def test_06_03_no_objects(self):
        '''Test images with no objects'''
        labels = np.zeros((10, 10), int)
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = image1.copy()
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME,
                                           "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertEqual(len(corr), 0)
        self.assertEqual(len(m.get_object_names()), 2)
        self.assertTrue(OBJECTS_NAME in m.get_object_names())
        columns = module.get_measurement_columns(None)
        image_features = m.get_feature_names(cpmeas.IMAGE)
        object_features = m.get_feature_names(OBJECTS_NAME)
        self.assertEqual(len(columns), len(image_features) + len(object_features))
        for column in columns:
            if column[0] == cpmeas.IMAGE:
                self.assertTrue(column[1] in image_features)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in object_features)

    def test_06_04_wrong_size(self):
        '''Regression test of IMG-961 - objects and images of different sizes'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        labels = np.zeros((10, 30), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i1, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], 1)

    def test_06_05_last_object_masked(self):
        # Regression test of issue #1553
        # MeasureCorrelation was truncating the measurements
        # if the last had no pixels or all pixels masked.
        #
        r = np.random.RandomState()
        r.seed(65)
        image1 = r.uniform(size=(20, 20))
        image2 = r.uniform(size=(20, 20))
        labels = np.zeros((20, 20), int)
        labels[3:8, 3:8] = 1
        labels[13:18, 13:18] = 2
        mask = labels != 2
        objects = cpo.Objects()
        objects.segmented = labels

        for mask1, mask2 in ((mask, None), (None, mask), (mask, mask)):
            workspace, module = self.make_workspace(
                    cpi.Image(image1, mask=mask1),
                    cpi.Image(image2, mask=mask2),
                    objects)
            module.run(workspace)
            m = workspace.measurements
            feature = M.F_CORRELATION_FORMAT % (IMAGE1_NAME, IMAGE2_NAME)
            values = m[OBJECTS_NAME, feature]
            self.assertEqual(len(values), 2)
            self.assertTrue(np.isnan(values[1]))
