import numpy as np
import os
import unittest
import tempfile

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw

from cellprofiler.modules import instantiate_module

IMAGE_1_NAME = "Image1.tif"
IMAGE_2_NAME = "Image2.tif"

MODULE_NAME = "Example6b"
IMAGE_NAME = "imagename"
class TestExample6b(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        cls.module = instantiate_module(MODULE_NAME)
        cls.module.image_name.value = IMAGE_NAME
        cls.module.folder.set_dir_choice(cps.ABSOLUTE_FOLDER_NAME)
        cls.module.folder.set_custom_path(cls.tempdir)
        cls.module.module_num = 1
        cls.pipeline = cpp.Pipeline()
        cls.pipeline.add_module(cls.module)

        for name, data in ((IMAGE_1_NAME, TIF_1),
                           (IMAGE_2_NAME, TIF_2)):
            fd = open(os.path.join(cls.tempdir, name), "wb")
            fd.write(data)
            fd.close()
        
    @classmethod
    def tearDownClass(cls):
        for name in (IMAGE_1_NAME, IMAGE_2_NAME):
            os.remove(os.path.join(cls.tempdir, name))
        os.rmdir(cls.tempdir)
        
    def test_01_01_prepare_run(self):
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(
            self.pipeline,
            self.module,
            None, None,
            measurements,
            cpi.ImageSetList())
        self.assertTrue(self.module.prepare_run(workspace))
        self.assertTrue(measurements.has_feature(cpmeas.EXPERIMENT, 
                                                 "Example6b_FirstTime"))
        first_time = measurements.get_experiment_measurement("Example6b_FirstTime")
        #
        # Breaks if you put CP in a time machine
        #
        self.assertTrue(int(first_time[:4]) >= 2013)
        self.assertEqual(measurements.image_set_count, 2, 
                         "You should have two image sets")
        M_FILE_NAME = cpmeas.C_FILE_NAME + "_" + IMAGE_NAME
        M_PATH_NAME = cpmeas.C_PATH_NAME + "_" + IMAGE_NAME
        self.assertTrue(measurements.has_feature(
            cpmeas.IMAGE, M_FILE_NAME))
        self.assertTrue(measurements.has_feature(
            cpmeas.IMAGE, M_PATH_NAME))
        for i, filename in enumerate((IMAGE_1_NAME, IMAGE_2_NAME)):
            image_number = i+1
            self.assertEqual(measurements.get_measurement(
                cpmeas.IMAGE, M_FILE_NAME, image_set_number=image_number),
                             filename)
            self.assertEqual(measurements.get_measurement(
                cpmeas.IMAGE, M_PATH_NAME, image_set_number=image_number),
                             self.tempdir)

    def test_01_02_run(self):
        measurements = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(
            self.pipeline,
            self.module,
            None, None,
            measurements,
            image_set_list)
        self.assertTrue(self.module.prepare_run(workspace))
        for i, expected in enumerate((IMAGE_1, IMAGE_2)):
            image_number = i+1
            if hasattr(measurements, "get_image"):
                measurements.next_image_set(image_number)
                image_set = measurements
            else:
                image_set = image_set_list.get_image_set(image_number)
            workspace = cpw.Workspace(
                self.pipeline,
                self.module,
                image_set, None,
                measurements,
                image_set_list)
            self.module.run(workspace)
            image = image_set.get_image(IMAGE_NAME)
            pixel_data = (image.pixel_data * 255).astype(int)
            np.testing.assert_array_equal(pixel_data, expected)
            
        
                                                 
        
IMAGE_1 = np.array([
    [206, 115,  84,  69, 239, 197, 189, 170,  54,   7],
    [152,   5, 131, 196, 209, 133, 162,  10,  54,   3],
    [119, 173, 199,  47,  14, 214,  96, 223,  26, 199],
    [254,  83, 233, 222, 144, 187, 245, 114,  93, 199],
    [236, 192, 125, 122,  15, 198, 122,  91,  36, 222],
    [158, 161,  50, 131, 240, 197,  66, 151,  15,  41],
    [ 13,  94,  36,  84, 236, 113,  13, 107,  60, 180],
    [ 23, 219,  97, 233, 159,   2, 145,  97, 237, 156],
    [ 19, 200, 228, 243, 159,  84, 239,  95, 253,  57],
    [160, 248,  69, 203,  97, 124, 164, 167, 118, 250]])

IMAGE_2 = np.array([[ 11, 168, 236, 176, 252,  42,   5, 241, 146, 134],
       [230, 247, 236, 184,  43, 115,  33,  90, 178, 127],
       [131,  65, 229,  18, 208, 205, 199, 209, 107, 112],
       [ 90, 118, 236,  81,  20,  86,  89,  91, 155, 227],
       [177, 156, 218, 240,  94,  42, 137, 130, 158,   4],
       [ 26,  50,  28, 145, 145, 178, 151,  24,  10,  63],
       [ 24, 111, 254,  11, 161, 182, 104,  79, 129,  79],
       [201, 247, 139,  64,  32,  33, 178, 128,  39,  32],
       [187, 239, 167,  76, 222,  40, 195, 139, 137, 250],
       [104, 245, 203,  41, 108,  82, 191, 108, 168, 186]])

TIF_1 = np.array([
    73,  73,  42,   0,   8,   0,   0,   0,  16,   0,   0,   1,   4,
    0,   1,   0,   0,   0,  10,   0,   0,   0,   1,   1,   4,   0,
    1,   0,   0,   0,  10,   0,   0,   0,   2,   1,   3,   0,   1,
    0,   0,   0,   8,   0,   0,   0,   3,   1,   3,   0,   1,   0,
    0,   0,   1,   0,   0,   0,   6,   1,   3,   0,   1,   0,   0,
    0,   1,   0,   0,   0,  14,   1,   2,   0,   1,   0,   0,   0,
    0,   0,   0,   0,  17,   1,   4,   0,  10,   0,   0,   0, 206,
    0,   0,   0,  21,   1,   3,   0,   1,   0,   0,   0,   1,   0,
    0,   0,  22,   1,   4,   0,   1,   0,   0,   0,   1,   0,   0,
    0,  23,   1,   4,   0,  10,   0,   0,   0, 246,   0,   0,   0,
    26,   1,   5,   0,   1,   0,   0,   0,  30,   1,   0,   0,  27,
    1,   5,   0,   1,   0,   0,   0,  38,   1,   0,   0,  28,   1,
    3,   0,   1,   0,   0,   0,   1,   0,   0,   0,  40,   1,   3,
    0,   1,   0,   0,   0,   3,   0,   0,   0,  49,   1,   2,   0,
    16,   0,   0,   0,  46,   1,   0,   0,  83,   1,   3,   0,   1,
    0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,  62,   1,
    0,   0,  72,   1,   0,   0,  82,   1,   0,   0,  92,   1,   0,
    0, 102,   1,   0,   0, 112,   1,   0,   0, 122,   1,   0,   0,
    132,   1,   0,   0, 142,   1,   0,   0, 152,   1,   0,   0,  10,
    0,   0,   0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,
    0,   0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,   0,
    0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,   0,   0,
    0,   0,   0,   0, 232,   3,   0,   0,   0,   0,   0,   0, 232,
    3,   0,   0,  79,  77,  69,  32,  66, 105, 111,  45,  70, 111,
    114, 109,  97, 116, 115,   0, 206, 115,  84,  69, 239, 197, 189,
    170,  54,   7, 152,   5, 131, 196, 209, 133, 162,  10,  54,   3,
    119, 173, 199,  47,  14, 214,  96, 223,  26, 199, 254,  83, 233,
    222, 144, 187, 245, 114,  93, 199, 236, 192, 125, 122,  15, 198,
    122,  91,  36, 222, 158, 161,  50, 131, 240, 197,  66, 151,  15,
    41,  13,  94,  36,  84, 236, 113,  13, 107,  60, 180,  23, 219,
    97, 233, 159,   2, 145,  97, 237, 156,  19, 200, 228, 243, 159,
    84, 239,  95, 253,  57, 160, 248,  69, 203,  97, 124, 164, 167,
    118, 250], dtype=np.uint8)

TIF_2 = np.array([ 73,  73,  42,   0,   8,   0,   0,   0,  16,   0,   0,   1,   4,
         0,   1,   0,   0,   0,  10,   0,   0,   0,   1,   1,   4,   0,
         1,   0,   0,   0,  10,   0,   0,   0,   2,   1,   3,   0,   1,
         0,   0,   0,   8,   0,   0,   0,   3,   1,   3,   0,   1,   0,
         0,   0,   1,   0,   0,   0,   6,   1,   3,   0,   1,   0,   0,
         0,   1,   0,   0,   0,  14,   1,   2,   0,   1,   0,   0,   0,
         0,   0,   0,   0,  17,   1,   4,   0,  10,   0,   0,   0, 206,
         0,   0,   0,  21,   1,   3,   0,   1,   0,   0,   0,   1,   0,
         0,   0,  22,   1,   4,   0,   1,   0,   0,   0,   1,   0,   0,
         0,  23,   1,   4,   0,  10,   0,   0,   0, 246,   0,   0,   0,
        26,   1,   5,   0,   1,   0,   0,   0,  30,   1,   0,   0,  27,
         1,   5,   0,   1,   0,   0,   0,  38,   1,   0,   0,  28,   1,
         3,   0,   1,   0,   0,   0,   1,   0,   0,   0,  40,   1,   3,
         0,   1,   0,   0,   0,   3,   0,   0,   0,  49,   1,   2,   0,
        16,   0,   0,   0,  46,   1,   0,   0,  83,   1,   3,   0,   1,
         0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,  62,   1,
         0,   0,  72,   1,   0,   0,  82,   1,   0,   0,  92,   1,   0,
         0, 102,   1,   0,   0, 112,   1,   0,   0, 122,   1,   0,   0,
       132,   1,   0,   0, 142,   1,   0,   0, 152,   1,   0,   0,  10,
         0,   0,   0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,
         0,   0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,   0,
         0,  10,   0,   0,   0,  10,   0,   0,   0,  10,   0,   0,   0,
         0,   0,   0,   0, 232,   3,   0,   0,   0,   0,   0,   0, 232,
         3,   0,   0,  79,  77,  69,  32,  66, 105, 111,  45,  70, 111,
       114, 109,  97, 116, 115,   0,  11, 168, 236, 176, 252,  42,   5,
       241, 146, 134, 230, 247, 236, 184,  43, 115,  33,  90, 178, 127,
       131,  65, 229,  18, 208, 205, 199, 209, 107, 112,  90, 118, 236,
        81,  20,  86,  89,  91, 155, 227, 177, 156, 218, 240,  94,  42,
       137, 130, 158,   4,  26,  50,  28, 145, 145, 178, 151,  24,  10,
        63,  24, 111, 254,  11, 161, 182, 104,  79, 129,  79, 201, 247,
       139,  64,  32,  33, 178, 128,  39,  32, 187, 239, 167,  76, 222,
        40, 195, 139, 137, 250, 104, 245, 203,  41, 108,  82, 191, 108,
       168, 186], dtype=np.uint8)
