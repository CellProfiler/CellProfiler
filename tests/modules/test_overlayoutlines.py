import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.overlayoutlines
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

INPUT_IMAGE_NAME = 'inputimage'
OUTPUT_IMAGE_NAME = 'outputimage'
OUTLINE_NAME = 'outlineimage'
OBJECTS_NAME = 'objectsname'


class TestOverlayOutlines(unittest.TestCase):
    def make_workspace(self, image, outline=None, labels=None):
        '''Make a workspace for testing ApplyThreshold'''
        m = cellprofiler.measurement.Measurements()
        object_set = cellprofiler.region.Set()
        module = cellprofiler.modules.overlayoutlines.OverlayOutlines()
        module.blank_image.value = False
        module.image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        if outline is not None:
            module.outlines[0].outline_name.value = OUTLINE_NAME
            m.add(OUTLINE_NAME, cellprofiler.image.Image(outline))
            module.outlines[0].outline_choice.value = cellprofiler.modules.overlayoutlines.FROM_IMAGES
        if labels is not None:
            objects = cellprofiler.region.Region()
            if len(labels) > 1:
                ijv = numpy.vstack(
                        [numpy.column_stack(list(numpy.where(l > 0)) + [l[l > 0]])
                         for l in labels])
                objects.set_ijv(ijv, shape=labels[0].shape)
            else:
                objects.segmented = labels[0]
            object_set.add_objects(objects, OBJECTS_NAME)
            module.outlines[0].outline_choice.value = cellprofiler.modules.overlayoutlines.FROM_OBJECTS
            module.outlines[0].objects_name.value = OBJECTS_NAME

        pipeline = cellprofiler.pipeline.Pipeline()
        workspace = cellprofiler.workspace.Workspace(pipeline, module, m, object_set, m, None)
        m.add(INPUT_IMAGE_NAME, cellprofiler.image.Image(image))
        return workspace, module

    def test_01_01_load_v1(self):
        data = ('eJztWl1v0zAUdbtubAyNMR5A2osfAW1ROlY09kK7DUbR+iFWDfEEXu'
                'p2Rq5dJc5o+WU88sjP4ScQt0mbmrTpWtamkEhRdx0fn+vjazu5cyFX'
                'OcsdwYymw0KuslsjFMMyRaLGzcYhZGIHHpsYCVyFnB3CAmfwnc2gfg'
                'DTLw71/cP0HtzT9ZdgsiuRL2w4Pz+fArDi/K46d9J9tOzaCd8t7XMs'
                'BGF1axmkwGO3/IdzXyCToEuKLxC1sdWn8MrzrMYr7WbvUYFXbYqLqO'
                'Gv7FxFu3GJTatU84Du4zJpYXpOvmGlC1619/iaWIQzF++2r5b2eLlQ'
                'eKUO8GFfh4Siw5Jzb/vKZf23oF8/FaDbA1/9TdcmrEquSdVGFJIGqv'
                'e8kO0dhLS3qrQn7ZJJ6keO5BKvh+CTA/gkKPIubzYEt6nwyruCW2L3'
                'dQsZAjaQMK7G8X9FaUfaRdugmMym/4kBfAI8H1P3ZYVX2ml9Z18H4+'
                'l3T8FLu3SNTYraeRkDYMx27irtSPuEQ8YFtC13Yoyjw9JAO0vgoxOF'
                'k+pwzCk3x+QdFn+3FbdBuhdQC/Jad+55eoXN40dKO9I+wTVkUwE7Aw'
                'hPiIkNwc32VPqH9WdD8UPa3flTsgUlTC4mk8wDDYyn57rCL+2SsGx4'
                'SvklosDTc5I4OjUxZuB240jtdzqAb0XBeZeHW3N/b9NPNT70nfSt+p'
                'ka4EuBk1w5PwlO1/T0NH5Ou/5nQ/BrCl7aeSYws4hoD9FrFjovmt9F'
                'ziZ630jr0fRTXRcyAXxR8HPYvvE3/LwJLhvi5yT71Cz8XhR9Yz/n66'
                'ezj0VyXo37/hV1fRclDqLqp7qvapn5jHs2xM+geK185dCgyLLczMc8'
                '/A773gvK23zApH4l03DXMuHEDOxrL2q6B+UJ3nAT101us+r0/M2tm+'
                'XJZtnPzoe97Ghzev5Zxhe//IIN0XEcElbFzX9AhzC/g/KKfb+7Msx7'
                '/Y1x0+OyYHQcBOXpePdLoRfBi9TfGPd/4rJgdJzH612Mi3HReQ/bAo'
                'PzUdr+fcf/IrZI/Y5xMS4K+11Uv0tiXIyLcYuHayX6ODXvpOZvZf3P'
                'Pp6g9ekZGFyfpG1gSpsml+fwTK3ROSxmaZSjave0lnbm/Jn3Hdzq5M'
                'NCeLIKT3YYD6liJkit3TQdNlvwBhLE0PJuadkpzXmlkvcqhDej8GaG'
                '8fLuYaRP7suPpbmnk/r/rvtj3NYC+Pz6Jx3r/vadkeOtjnN//H+9mo'
                'QvuZTo8PnPEayH4FI+n4Dbz+/gZnH2ZER9r4+zqv8b5il/ZA==')
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.overlayoutlines.OverlayOutlines))
        self.assertFalse(module.blank_image.value)
        self.assertEqual(module.image_name.value, "OrigBlue")
        self.assertEqual(module.wants_color.value, cellprofiler.modules.overlayoutlines.WANTS_COLOR)
        self.assertEqual(len(module.outlines), 1)
        self.assertEqual(module.outlines[0].outline_name.value, "NucleiOutlines")
        self.assertEqual(module.outlines[0].color.value, "Green")
        self.assertEqual(module.max_type.value, cellprofiler.modules.overlayoutlines.MAX_IMAGE)

    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9063

OverlayOutlines:[module_num:5|svn_version:\'9000\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Display outlines on a blank image?:No
    Select image on which to display outlines:DNA
    Name the output image\x3A:PrimaryOverlay
    Select outline display mode\x3A:Color
    Select method to determine brightness of outlines\x3A:Max of image
    Line width\x3A:1.5
    Select outlines to display\x3A:PrimaryOutlines
    Select outline color\x3A:Red
    Select outlines to display\x3A:SecondaryOutlines
    Select outline color\x3A:Green
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.overlayoutlines.OverlayOutlines))
        self.assertFalse(module.blank_image)
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.output_image_name, "PrimaryOverlay")
        self.assertEqual(module.wants_color, "Color")
        self.assertEqual(module.max_type, cellprofiler.modules.overlayoutlines.MAX_IMAGE)
        self.assertAlmostEqual(module.line_width.value, 1.5)
        self.assertEqual(len(module.outlines), 2)
        for outline, name, color in zip(module.outlines,
                                        ("PrimaryOutlines", "SecondaryOutlines"),
                                        ("Red", "Green")):
            self.assertEqual(outline.outline_name, name)
            self.assertEqual(outline.color, color)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140505183007
GitHash:c675ec6
ModuleCount:1
HasImagePlaneDetails:False

OverlayOutlines:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Display outlines on a blank image?:No
    Select image on which to display outlines:DNA
    Name the output image:PrimaryOverlay
    Outline display mode:Color
    Select method to determine brightness of outlines:Max of image
    Width of outlines:1.5
    Select outlines to display:PrimaryOutlines
    Select outline color:Red
    Load outlines from an image or objects?:Image
    Select objects to display:Nuclei
    Select outlines to display\x3A:SecondaryOutlines
    Select outline color\x3A:Green
    Load outlines from an image or objects?:Objects
    Select objects to display:Cells

"""
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.overlayoutlines.OverlayOutlines))
        self.assertFalse(module.blank_image)
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.output_image_name, "PrimaryOverlay")
        self.assertEqual(module.wants_color, "Color")
        self.assertEqual(module.max_type, cellprofiler.modules.overlayoutlines.MAX_IMAGE)
        self.assertAlmostEqual(module.line_width.value, 1.5)
        self.assertEqual(len(module.outlines), 2)
        for outline, name, color, choice, objects_name in (
                (module.outlines[0], "PrimaryOutlines", "Red",
                 cellprofiler.modules.overlayoutlines.FROM_IMAGES, "Nuclei"),
                (module.outlines[1], "SecondaryOutlines", "Green",
                 cellprofiler.modules.overlayoutlines.FROM_OBJECTS, "Cells")):
            self.assertEqual(outline.outline_name, name)
            self.assertEqual(outline.color, color)
            self.assertEqual(outline.outline_choice, choice)
            self.assertEqual(outline.objects_name, objects_name)

    def test_02_01_gray_to_color_outlines(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32)
        image[0, 0] = 1
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = numpy.dstack((image, image, image))
        expected[:, :, 0][outline.astype(bool)] = 1
        expected[:, :, 1][outline.astype(bool)] = 0
        expected[:, :, 2][outline.astype(bool)] = 0
        for i in range(2):
            if i == 0:
                workspace, module = self.make_workspace(image, outline)
            else:
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])

            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
            module.outlines[0].color.value = "Red"
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_02_02_color_to_color_outlines(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
        image[0, 0] = 1
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = image.copy()
        expected[:, :, 0][outline.astype(bool)] = 1
        expected[:, :, 1][outline.astype(bool)] = 0
        expected[:, :, 2][outline.astype(bool)] = 0
        for i in range(2):
            if i == 0:
                outline[21:30, 21:30] = 0
                workspace, module = self.make_workspace(image, outline)
            else:
                outline[21:30, 21:30] = 1
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])
            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
            module.outlines[0].color.value = "Red"
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_02_03_blank_to_color_outlines(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50, 3))
        image[0, 0] = 1
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = numpy.zeros((50, 50, 3))
        expected[:, :, 0][outline.astype(bool)] = 1
        expected[:, :, 1][outline.astype(bool)] = 0
        expected[:, :, 2][outline.astype(bool)] = 0
        for i in range(2):
            if i == 0:
                workspace, module = self.make_workspace(image, outline)
            else:
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])
            workspace, module = self.make_workspace(image, outline)
            module.blank_image.value = True
            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
            module.outlines[0].color.value = "Red"
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_02_04_wrong_size_gray_to_color(self):
        '''Regression test of img-961'''
        numpy.random.seed(24)
        image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32)
        image[0, 0] = 1
        outline = numpy.zeros((60, 40), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = numpy.dstack((image, image, image))
        sub_expected = expected[:50, :40]
        sub_expected[:, :, 0][outline[:50, :40].astype(bool)] = 1
        sub_expected[:, :, 1][outline[:50, :40].astype(bool)] = 0
        sub_expected[:, :, 2][outline[:50, :40].astype(bool)] = 0
        for i in range(2):
            if i == 0:
                workspace, module = self.make_workspace(image, outline)
            else:
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])
            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
            module.outlines[0].color.value = "Red"
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_02_05_wrong_size_color_to_color(self):
        numpy.random.seed(25)
        image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
        image[0, 0] = 1
        outline = numpy.zeros((60, 40), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = image.copy()
        sub_expected = expected[:50, :40]
        sub_expected[:, :, 0][outline[:50, :40].astype(bool)] = 1
        sub_expected[:, :, 1][outline[:50, :40].astype(bool)] = 0
        sub_expected[:, :, 2][outline[:50, :40].astype(bool)] = 0
        for i in range(2):
            if i == 0:
                workspace, module = self.make_workspace(image, outline)
            else:
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])
            workspace, module = self.make_workspace(image, outline)
            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
            module.outlines[0].color.value = "Red"
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_03_01_blank_to_gray(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50))
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = numpy.zeros_like(image)
        expected[outline.astype(bool)] = 1
        for i in range(2):
            if i == 0:
                workspace, module = self.make_workspace(image, outline)
            else:
                workspace, module = self.make_workspace(
                        image, labels=[outline.astype(int)])
            workspace, module = self.make_workspace(image, outline)
            module.blank_image.value = True
            module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
            module.line_width.value = 0.0
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_03_02_gray_max_image(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * .5
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = image.copy()
        expected[outline.astype(bool)] = numpy.max(image)
        workspace, module = self.make_workspace(image, outline)
        module.blank_image.value = False
        module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
        module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_IMAGE
        module.line_width.value = 0.0
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_03_02_gray_max_possible(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * .5
        outline = numpy.zeros((50, 50), bool)
        outline[20:31, 20:31] = 1
        outline[21:30, 21:30] = 0
        expected = image.copy()
        expected[outline.astype(bool)] = 1
        workspace, module = self.make_workspace(image, outline)
        module.blank_image.value = False
        module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
        module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_POSSIBLE
        module.line_width.value = 0.0
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_03_03_wrong_size_gray(self):
        '''Regression test of IMG-961 - image and outline size differ'''
        numpy.random.seed(41)
        image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * .5
        outline = numpy.zeros((60, 40), bool)
        outline[20:31, 20:31] = True
        outline[21:30, 21:30] = False
        expected = image.copy()
        expected[:50, :40][outline[:50, :40]] = 1
        workspace, module = self.make_workspace(image, outline)
        module.blank_image.value = False
        module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
        module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_POSSIBLE
        module.line_width.value = 0.0
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output_image.pixel_data == expected))

    def test_04_01_ijv(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
        image[0, 0] = 1
        labels0 = numpy.zeros(image.shape[:2], int)
        labels0[20:30, 20:30] = 1
        labels1 = numpy.zeros(image.shape[:2], int)
        labels1[25:35, 25:35] = 2
        labels = [labels0, labels1]
        expected = image.copy()
        mask = numpy.zeros(image.shape[:2], bool)
        mask[20:30, 20] = True
        mask[20:30, 29] = True
        mask[20, 20:30] = True
        mask[29, 20:30] = True
        mask[25:35, 25] = True
        mask[25:35, 34] = True
        mask[25, 25:35] = True
        mask[34, 25:35] = True
        expected[mask, 0] = 1
        expected[mask, 1:] = 0
        workspace, module = self.make_workspace(image, labels=labels)
        module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
        module.outlines[0].color.value = "Red"
        module.line_width.value = 0.0
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output_image.pixel_data == expected))
