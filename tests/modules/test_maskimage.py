'''test_maskimage - Test the MaskImage module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.maskimage as M

MASKING_IMAGE_NAME = "maskingimage"
MASKED_IMAGE_NAME = "maskedimage"
IMAGE_NAME = "image"
OBJECTS_NAME = "objects"


class TestMaskImage(unittest.TestCase):
    def test_02_01_mask_with_objects(self):
        labels = np.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[labels > 0] ==
                               pixel_data[labels > 0]))
        self.assertTrue(np.all(masked_image.pixel_data[labels == 0] == 0))
        self.assertTrue(np.all(masked_image.mask == (labels > 0)))
        self.assertTrue(np.all(masked_image.masking_objects.segmented == labels))

    def test_02_02_mask_invert(self):
        labels = np.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = True
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[labels == 0] ==
                               pixel_data[labels == 0]))
        self.assertTrue(np.all(masked_image.pixel_data[labels > 0] == 0))
        self.assertTrue(np.all(masked_image.mask == (labels == 0)))
        self.assertTrue(np.all(masked_image.masking_objects.segmented == labels))

    def test_02_03_double_mask(self):
        labels = np.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
        mask = np.random.uniform(size=(10, 15)) > .5
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data, mask))

        expected_mask = (mask & (labels > 0))

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[expected_mask] ==
                               pixel_data[expected_mask]))
        self.assertTrue(np.all(masked_image.pixel_data[~ expected_mask] == 0))
        self.assertTrue(np.all(masked_image.mask == expected_mask))
        self.assertTrue(np.all(masked_image.masking_objects.segmented == labels))

    def test_03_01_binary_mask(self):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

        masking_image = np.random.uniform(size=(10, 15)) > .5
        image_set.add(MASKING_IMAGE_NAME, cpi.Image(masking_image))

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[masking_image] ==
                               pixel_data[masking_image]))
        self.assertTrue(np.all(masked_image.pixel_data[~masking_image] == 0))
        self.assertTrue(np.all(masked_image.mask == masking_image))
        self.assertFalse(masked_image.has_masking_objects)

    def test_03_02_gray_mask(self):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

        masking_image = np.random.uniform(size=(10, 15))
        image_set.add(MASKING_IMAGE_NAME, cpi.Image(masking_image))
        masking_image = masking_image > .5

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[masking_image] ==
                               pixel_data[masking_image]))
        self.assertTrue(np.all(masked_image.pixel_data[~masking_image] == 0))
        self.assertTrue(np.all(masked_image.mask == masking_image))
        self.assertFalse(masked_image.has_masking_objects)

    def test_03_03_color_mask(self):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        np.random.seed(0)
        pixel_data = np.random.uniform(size=(10, 15, 3)).astype(np.float32)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

        masking_image = np.random.uniform(size=(10, 15))

        image_set.add(MASKING_IMAGE_NAME, cpi.Image(masking_image))
        expected_mask = masking_image > .5

        pipeline = cpp.Pipeline()
        module = M.MaskImage()
        module.source_choice.value = M.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cpi.Image))
        self.assertTrue(np.all(masked_image.pixel_data[expected_mask, :] ==
                               pixel_data[expected_mask, :]))
        self.assertTrue(np.all(masked_image.pixel_data[~expected_mask, :] == 0))
        self.assertTrue(np.all(masked_image.mask == expected_mask))
        self.assertFalse(masked_image.has_masking_objects)
