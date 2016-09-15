'''test_loaddata - Test the LoadData (formerly LoadText) module'''

import base64
import hashlib
import os
import re
import tempfile
import unittest
import zlib
from StringIO import StringIO

import numpy as np
from bioformats import write_image, PT_UINT8
from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.setting as cps
import cellprofiler.modules.loaddata as L
from cellprofiler.modules.loadimages import pathname2url
from tests.modules import \
    example_images_directory, testimages_directory, maybe_download_sbs, \
    maybe_download_example_image, maybe_download_tesst_image, \
    make_12_bit_image, cp_logo_url, cp_logo_url_filename, cp_logo_url_folder, \
    cp_logo_url_shape

from bioformats.formatreader import clear_image_reader_cache

OBJECTS_NAME = "objects"


class TestLoadData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        maybe_download_sbs()
        cls.test_folder = "loaddata"
        cls.test_path = os.path.join(
                example_images_directory(), cls.test_folder)
        cls.test_filename = "image.tif"
        cls.test_shape = (13, 15)
        path = maybe_download_example_image([cls.test_folder],
                                            cls.test_filename,
                                            shape=cls.test_shape)
        with open(path, "rb") as fd:
            cls.test_md5 = hashlib.md5(fd.read()).hexdigest()

    def make_pipeline(self, csv_text, name=None):
        if name is None:
            handle, name = tempfile.mkstemp(".csv")
            fd = os.fdopen(handle, 'w')
        else:
            fd = open(name, "w")
        fd.write(csv_text)
        fd.close()
        csv_path, csv_file = os.path.split(name)
        module = L.LoadText()
        module.csv_directory.dir_choice = L.ABSOLUTE_FOLDER_NAME
        module.csv_directory.custom_path = csv_path
        module.csv_file_name.value = csv_file
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def error_callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(error_callback)
        return pipeline, module, name

    def test_01_00_revision(self):
        '''Remember to update this and write another test on new revision'''
        self.assertEqual(L.LoadData().variable_revision_number, 6)

    def test_01_01_load_v1(self):
        data = ('eJztV01v2jAYdvgabBPith59mnrootANqeWyMtAEU6EVRdV2qlwwzJITR46'
                'DYL9gP2nHHfdz9hNm06QkHiUQsXXSsGQlr/0+z/tlJ3a3MThvvIM104Ldxu'
                'DVmFAMLykSY8btOnTEEWxyjAQeQebU4eCzDz/4DqyewGqt/vq4btXgsWWdg'
                'nTN6HTL8nFYAaAgn0XZM8FUPpCNSFfyFRaCOBMvD3LgIBj/Lvs14gTdUnyN'
                'qI+9pYlwvOOM2WDu3k912cinuIfsqLJsPd++xdy7GIfAYPqSzDC9Il+wFkK'
                'o1sdT4hHmBPiAXx+9t8uEZlfl4aCwzIOh5SGrjSv9Nljq51bk7UVEvxLILT'
                'xGPhWwY6MJhi3C8VAwPl/wWQl82RhfFrR6jQXuLAFX0vxQcnMumEuRZ0fiS'
                'bJvxHgMYAa4kwRcEcTtK/m9rOdIBrCJ/2UNXw7xkSDS+lG13pyaQ2+6Cf6J'
                'hldyk3G+aR1WxdG86PdvtDi2XQef5FrexP+CZl/JLTIlI7xZ/pLwu94PSfl'
                '8rvEpuccgx94QUfmRAkGOd8WTdn9EcQUNF7YQVwqeu8Kt8jMT8zMjY30cP1'
                'Pt16Oqpdpj+Psjv93/Ia2ds4S8PNXyomSi9s+EM9/dPc+fiqOi2a/E7EPij'
                'LC7S57/vR573B63xy1xX42Hv+f6OUvpfwTr9+9LEN+/Sh5iSl3O1L2Om/bi'
                '8uGZlKGRwDNhnsuXgXy5458l8Lc1/vZD/EN5KJXnKEKpbxMHCXkDukGuS+d'
                'm826mE51pqJl/wX60jqUV9qP1yEipmM+vrb9e9+V6+Pk2jT3DMH47tzxLwO'
                'UiPqmm8N/AduvucI1+GOPf0v8FsdkNYQ==')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadText))
        self.assertEqual(module.csv_directory.dir_choice,
                         L.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.csv_file_name, "1049.csv")
        self.assertTrue(module.wants_images.value)
        self.assertFalse(module.wants_image_groupings.value)
        self.assertEqual(module.image_directory.dir_choice,
                         L.DEFAULT_INPUT_FOLDER_NAME)
        self.assertFalse(module.wants_rows.value)

    def test_01_02_load_v2(self):
        data = ('eJztVd1KwzAUTuv82QTxTi9zJV5oycTf3eimiAM3xQ3RK4lbOgJpM9J0bD6B'
                'j+Jj+Cg+go9gMtKtrWOdeiPogZCc5PvOyfk4bWrl5mW5AvccBGvl5rZLGYHX'
                'DEuXC68EfbkFTwXBkrQh90vwXFBYDjsQHcDifqmISmgX7iB0BL5nVrW2oqa3'
                'RQAW1Lykhm2O5o1vxYb2G0RK6neCeZAD62b/VY1bLCh+ZOQWs5AE4xTRftV3'
                'eXPQHR3VeDtkpI69OFhZPfQeiQiu3Ihojq9pn7AGfSKpEiLYDenRgHLf8E38'
                '9O4oL5epvFqHF3usgzVBh0JsX+MvwBifm4Bfi+FXjX9GXBwyCase7hB4RgVp'
                'SS4Gw3goI56ViGcBx9zjMIO3BJL30H4R7R45raA3S965BH8O3CvtflP9WTw7'
                'wbNBnf9At60i0vYT3U4yePlUXu03Ko2HNg9GXavj3H2xX+P3XUjhI4vw+X/e'
                'n+c9g+n9Ff8eh/0Ipvf1Bkj2tfZbhLGu4PrdE443/DkHDuO4LUlfOpdq0VSL'
                'z3XkJ8SP38dWq0JG/em6x3q8H38nnz0h33IGL2de3LR+s+i9OQUPUvgPhsiG'
                'ig==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadText))
        self.assertEqual(module.csv_file_name, "1049.csv")
        self.assertTrue(module.wants_images.value)
        self.assertTrue(module.wants_image_groupings.value)
        self.assertEqual(len(module.metadata_fields.selections), 1)
        self.assertEqual(module.metadata_fields.selections[0], "SBS_doses")
        self.assertEqual(module.image_directory.dir_choice,
                         L.DEFAULT_INPUT_FOLDER_NAME)
        self.assertFalse(module.wants_rows.value)

    def test_01_03_load_v3(self):
        data = ('eJztVUtPg0AQXihtfCTGgwePezIelICv2F5MqzE2kWqE+Dhu28WSLGwDS338'
                'An+CP9Of4FKX8pAUbT2Y6AQyO7PzzTczLGA0rfNmC+6rGjSa1rbtEAwvCWI2'
                '9d0G9NgWPPYxYrgPqdeAN1yf4B7UD/jV0Hcbe/twR9PqYDaR2sYKV68LANS4'
                '5grIYqsqbCl1R7aJGXO8+6AKFLAu/G/8vka+g7oEXyMS4iChiP1tz6bW03Cy'
                'ZdB+SHAHuelgLp3Q7WI/uLBjoNi+dB4xMZ1nnGshDrvCIydwqCfwIn/eO+Gl'
                'LMdrDujDqc/LyeVvIdYbmIw/gax/PDc5mZtUMLellD+KPwNJvFIQv5aKXxX2'
                'CbZRSBhse8OQwVNK+tiP82kl+aRMPgmoAndYglsA2ToiW9f26movGH2Ft5LB'
                'V8Adn/Vv6r8MJ2dwMujQOea2pWuRzMObxtVyuFhi3KLQEe72m+dzVp5/3N/E'
                'vYDp5yv9/o3PI5h+/jdA9v2J7B4mZOjT6L/oq+744x2ohKI+w49MPecLiy8+'
                '97FYkD9dj8xXSyX95/tO5vF2NAtfpYBvuQSniD9yfn5fmffmlHhQEP/dfqQf'
                'qCvhUSY1feT/kHfzN4ip')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadText))
        self.assertEqual(module.csv_file_name, "1049.csv")
        self.assertTrue(module.wants_images.value)
        self.assertFalse(module.wants_image_groupings.value)

    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9722

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Input data file location:Default Input Folder\x7C.
    Name of the file:1049_Metadata.csv
    Load images based on this data?:Yes
    Base image location:Default Input Folder\x7C.
    Process just a range of rows?:No
    Rows to process:10,36
    Group images by metadata?:No
    Select metadata fields for grouping:Well
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadData))
        self.assertEqual(module.csv_file_name, "1049_Metadata.csv")
        self.assertEqual(module.csv_directory.dir_choice,
                         cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.wants_images)
        self.assertTrue(module.rescale)
        self.assertFalse(module.wants_image_groupings)
        self.assertFalse(module.wants_rows)
        self.assertEqual(module.row_range.min, 10)
        self.assertEqual(module.row_range.max, 36)
        self.assertEqual(len(module.metadata_fields.selections), 1)
        self.assertEqual(module.metadata_fields.selections[0], "Well")

    def test_01_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10534

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Input data file location:Elsewhere...\x7Cx\x3A\\projects\\NightlyBuild\\trunk\\ExampleImages\\ExampleSBSImages
    Name of the file:1049_Metadata.csv
    Load images based on this data?:Yes
    Base image location:Default Input Folder\x7C.
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:Yes
    Select metadata fields for grouping:Column,Row
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadData))
        self.assertEqual(module.csv_file_name, "1049_Metadata.csv")
        self.assertEqual(module.csv_directory.dir_choice,
                         cps.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.csv_directory.custom_path,
                         r"x:\projects\NightlyBuild\trunk\ExampleImages\ExampleSBSImages")
        self.assertTrue(module.wants_images)
        self.assertEqual(module.image_directory.dir_choice,
                         cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.rescale)
        self.assertTrue(module.wants_image_groupings)
        self.assertFalse(module.wants_rows)
        self.assertEqual(module.row_range.min, 1)
        self.assertEqual(module.row_range.max, 100000)
        self.assertEqual(len(module.metadata_fields.selections), 2)
        self.assertEqual(module.metadata_fields.selections[0], "Column")
        self.assertEqual(module.metadata_fields.selections[1], "Row")

    def test_01_06_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10536

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D]
    Input data file location:Elsewhere...\x7Cx\x3A\\projects\\NightlyBuild\\trunk\\ExampleImages\\ExampleSBSImages
    Name of the file:1049_Metadata.csv
    Load images based on this data?:Yes
    Base image location:Default Input Folder\x7C.
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:Yes
    Select metadata fields for grouping:Column,Row
    Rescale intensities?:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadData))
        self.assertEqual(module.csv_file_name, "1049_Metadata.csv")
        self.assertEqual(module.csv_directory.dir_choice,
                         cps.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.csv_directory.custom_path,
                         r"x:\projects\NightlyBuild\trunk\ExampleImages\ExampleSBSImages")
        self.assertTrue(module.wants_images)
        self.assertEqual(module.image_directory.dir_choice,
                         cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.rescale)
        self.assertTrue(module.wants_image_groupings)
        self.assertFalse(module.wants_rows)
        self.assertEqual(module.row_range.min, 1)
        self.assertEqual(module.row_range.max, 100000)
        self.assertEqual(len(module.metadata_fields.selections), 2)
        self.assertEqual(module.metadata_fields.selections[0], "Column")
        self.assertEqual(module.metadata_fields.selections[1], "Row")

    def test_02_01_string_image_measurement(self):
        csv_text = '''"Test_Measurement"
"Hello, world"
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run()
        data = m.get_current_image_measurement("Test_Measurement")
        self.assertFalse(np.isreal(data))
        self.assertEqual(data, "Hello, world")
        os.remove(filename)

    def test_02_02_float_image_measurement(self):
        csv_text = '''"Test_Measurement"
1.5
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run()
        data = m.get_current_image_measurement("Test_Measurement")
        self.assertTrue(np.isreal(data))
        self.assertAlmostEqual(data, 1.5)
        os.remove(filename)

    def test_02_03_int_image_measurement(self):
        csv_text = '''"Test_Measurement"
1
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run()
        data = m.get_current_image_measurement("Test_Measurement")
        self.assertTrue(isinstance(data, np.int), "data is type %s, not np.int" % (type(data)))
        self.assertEqual(data, 1)
        os.remove(filename)

    def test_02_04_long_int_image_measurement(self):
        csv_text = '''"Test_Measurement"
1234567890123
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run()
        data = m.get_current_image_measurement("Test_Measurement")
        self.assertTrue(isinstance(data, unicode), "Expected <type 'unicode'> got %s" % type(data))
        self.assertEqual(data, "1234567890123")
        os.remove(filename)

    def test_03_01_metadata(self):
        csv_text = '''"Metadata_Plate"
"P-12345"
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run()
        data = m.get_current_image_measurement("Metadata_Plate")
        self.assertEqual(data, "P-12345")
        os.remove(filename)

    def test_03_02_metadata_row_and_column(self):
        csv_text = '''"Metadata_Row","Metadata_Column"
"C","03"
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        columns = module.get_measurement_columns(pipeline)
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Row" and
                             c[2] == "varchar(1)" for c in columns]))
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Column" and
                             c[2] == "varchar(2)" for c in columns]))
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Well" and
                             c[2] == "varchar(3)" for c in columns]))
        m = pipeline.run()
        features = module.get_measurements(pipeline, cpmeas.IMAGE,
                                           cpmeas.C_METADATA)
        for feature, expected in (("Row", "C"),
                                  ("Column", "03"),
                                  ("Well", "C03")):
            self.assertTrue(feature in features)
            value = m.get_current_image_measurement('_'.join((cpmeas.C_METADATA, feature)))
            self.assertEqual(value, expected)

    def test_03_03_metadata_row_and_column_and_well(self):
        csv_text = '''"Metadata_Row","Metadata_Column","Metadata_Well"
"C","03","B14"
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        columns = module.get_measurement_columns(pipeline)
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Row" and
                             c[2] == "varchar(1)" for c in columns]))
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Column" and
                             c[2] == "varchar(2)" for c in columns]))
        self.assertTrue(any([c[0] == cpmeas.IMAGE and
                             c[1] == "Metadata_Well" and
                             c[2] == "varchar(3)" for c in columns]))
        m = pipeline.run()
        features = module.get_measurements(pipeline, cpmeas.IMAGE,
                                           cpmeas.C_METADATA)
        for feature, expected in (("Row", "C"),
                                  ("Column", "03"),
                                  ("Well", "B14")):
            self.assertTrue(feature in features)
            value = m.get_current_image_measurement('_'.join((cpmeas.C_METADATA, feature)))
            self.assertEqual(value, expected)

    def test_04_01_load_file(self):
        csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"
''' % (self.test_filename, self.test_path)
        pipeline, module, filename = self.make_pipeline(csv_text)
        c0_ran = [False]

        def callback(workspace):
            imgset = workspace.image_set
            image = imgset.get_image("DNA")
            pixels = image.pixel_data
            self.assertEqual(pixels.shape[0], self.test_shape[0])
            c0_ran[0] = True

        c0 = C0()
        c0.callback = callback
        c0.module_num = 2
        pipeline.add_module(c0)

        try:
            m = pipeline.run()
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(c0_ran[0])
            hexdigest = m.get_current_image_measurement('MD5Digest_DNA')
            self.assertEqual(hexdigest, self.test_md5)
            self.assertTrue('PathName_DNA' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement('PathName_DNA'),
                             self.test_path)
            self.assertTrue('FileName_DNA' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement('FileName_DNA'),
                             self.test_filename)
        finally:
            os.remove(filename)

    def test_04_02_dont_load_file(self):
        csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"
''' % (self.test_filename, self.test_path)
        pipeline, module, filename = self.make_pipeline(csv_text)
        c0_ran = [False]

        def callback(workspace):
            imgset = workspace.image_set
            self.assertEqual(len(imgset.names), 0)
            c0_ran[0] = True

        c0 = C0()
        c0.callback = callback
        c0.module_num = 1
        pipeline.add_module(c0)
        try:
            module.wants_images.value = False
            pipeline.run()
            self.assertTrue(c0_ran[0])
        finally:
            os.remove(filename)

    def test_04_03_load_planes(self):
        file_name = "RLM1 SSN3 300308 008015000.flex"
        maybe_download_tesst_image(file_name)
        path = testimages_directory()
        pathname = os.path.join(path, file_name)
        url = pathname2url(pathname)
        ftrs = (cpmeas.C_URL, cpmeas.C_SERIES, cpmeas.C_FRAME)
        channels = ("Channel1", "Channel2")
        header = ",".join([",".join(["_".join((ftr, channel)) for ftr in ftrs])
                           for channel in channels])

        csv_lines = [header]
        for series in range(4):
            csv_lines.append(",".join(['"%s","%d","%d"' % (url, series, frame)
                                       for frame in range(2)]))
        csv_text = "\n".join(csv_lines)
        pipeline, module, filename = self.make_pipeline(csv_text)
        assert isinstance(module, L.LoadData)
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        try:
            workspace = cpw.Workspace(pipeline, module, m, None, m,
                                      image_set_list)
            self.assertTrue(module.prepare_run(workspace))
            pixel_hashes = []
            for i in range(4):
                m.next_image_set(i + 1)
                module.run(workspace)
                chashes = []
                for channel in channels:
                    pixel_data = m.get_image(channel).pixel_data
                    h = hashlib.md5()
                    h.update(pixel_data)
                    chashes.append(h.digest())
                self.assertNotEqual(chashes[0], chashes[1])
                for j, ph in enumerate(pixel_hashes):
                    for k, phh in enumerate(ph):
                        for l, phd in enumerate(chashes):
                            self.assertNotEqual(phh, phd)
                pixel_hashes.append(chashes)
        finally:
            os.remove(filename)

    def test_05_01_some_rows(self):
        csv_text = '''"Test_Measurement"
1
2
3
4
5
6
7
8
9
10
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        module.wants_rows.value = True
        module.row_range.min = 4
        module.row_range.max = 6
        m = pipeline.run()
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        data = m.get_all_measurements(cpmeas.IMAGE, "Test_Measurement")
        self.assertTrue(np.all(data == np.arange(4, 7)))
        os.remove(filename)

    def test_05_02_img_717(self):
        '''Regression test of img-717, column without underbar'''
        csv_text = '''"Image","Test_Measurement"
"foo",1
"foo",2
"foo",3
"foo",4
"foo",5
"foo",6
"foo",7
"foo",8
"foo",9
"foo",10
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        module.wants_rows.value = True
        module.row_range.min = 4
        module.row_range.max = 6
        m = pipeline.run()
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        data = m.get_all_measurements(cpmeas.IMAGE, "Test_Measurement")
        self.assertTrue(np.all(data == np.arange(4, 7)))
        os.remove(filename)

    def test_06_01_alternate_image_start(self):
        csv_text = '''"Metadata_Measurement"
1
2
3
4
5
6
7
8
9
10
'''
        pipeline, module, filename = self.make_pipeline(csv_text)
        m = pipeline.run(image_set_start=2)
        data = m.get_all_measurements(cpmeas.IMAGE, "Metadata_Measurement")
        self.assertTrue(all([data[i - 2] == i for i in range(2, 11)]))
        os.remove(filename)

    def test_07_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        colnames = ('Integer_Measurement', 'Float_Measurement', 'String_Measurement')
        coltypes = [cpmeas.COLTYPE_INTEGER, cpmeas.COLTYPE_FLOAT,
                    cpmeas.COLTYPE_VARCHAR_FORMAT % 9]
        csv_text = '''"%s","%s","%s"
1,1,1
2,1.5,"Hi"
3,1,"Hello"
4,1.7,"Hola"
5,1.2,"Bonjour"
6,1.5,"Gutentag"
7,1.1,"Hej"
8,2.3,"Bevakasha"
''' % colnames
        pipeline, module, filename = self.make_pipeline(csv_text)
        columns = module.get_measurement_columns(pipeline)
        for colname, coltype in zip(colnames, coltypes):
            self.assertTrue(any([(column[0] == cpmeas.IMAGE and
                                  column[1] == colname and
                                  column[2] == coltype) for column in columns]),
                            'Failed to find %s' % colname)
        os.remove(filename)

    def test_07_02_file_name_measurement_columns(self):
        '''Regression test bug IMG-315

        A csv header of Image_FileName_Foo or Image_PathName_Foo should
        yield column names of FileName_Foo and PathName_Foo
        '''
        colnames = ('Image_FileName_Foo', 'Image_PathName_Foo')
        csv_text = '''"%s","%s"
"Channel1-01.tif","/imaging/analysis/2500_01_01_Jones"
"Channel1-02.tif","/imaging/analysis/2500_01_01_Jones"
''' % colnames
        pipeline, module, filename = self.make_pipeline(csv_text)
        try:
            columns = module.get_measurement_columns(pipeline)
            self.assertTrue('FileName_Foo' in [c[1] for c in columns])
            self.assertTrue('PathName_Foo' in [c[1] for c in columns])
        finally:
            os.remove(filename)

    def test_07_03_long_integer_column(self):
        '''This is a regression test of IMG-644 where a 13-digit number got turned into an int'''
        colnames = ('Long_Integer_Measurement', 'Float_Measurement', 'String_Measurement')
        coltypes = [cpmeas.COLTYPE_VARCHAR_FORMAT % 13, cpmeas.COLTYPE_FLOAT,
                    cpmeas.COLTYPE_VARCHAR_FORMAT % 9]
        csv_text = '''"%s","%s","%s"
1,1,1
2,1.5,"Hi"
3,1,"Hello"
4,1.7,"Hola"
5,1.2,"Bonjour"
6,1.5,"Gutentag"
7,1.1,"Hej"
1234567890123,2.3,"Bevakasha"
''' % colnames
        pipeline, module, filename = self.make_pipeline(csv_text)
        columns = module.get_measurement_columns(pipeline)
        fmt = "%15s %30s %20s"
        print fmt % ("Object", "Feature", "Type")
        for object_name, feature, coltype in columns:
            print fmt % (object_name, feature, coltype)
        for colname, coltype in zip(colnames, coltypes):
            self.assertTrue(any([(column[0] == cpmeas.IMAGE and
                                  column[1] == colname and
                                  column[2] == coltype) for column in columns]),
                            'Failed to find %s' % colname)
        os.remove(filename)

    def test_07_04_objects_measurement_columns(self):
        csv_text = """%s_%s,%s_%s
Channel1-01-A-01.tif,/imaging/analysis/trunk/ExampleImages/ExampleSBSImages
""" % (L.C_OBJECTS_FILE_NAME, OBJECTS_NAME,
       L.C_OBJECTS_PATH_NAME, OBJECTS_NAME)
        pipeline, module, filename = self.make_pipeline(csv_text)
        columns = module.get_measurement_columns(pipeline)
        expected_columns = (
            (cpmeas.IMAGE, L.C_OBJECTS_URL + "_" + OBJECTS_NAME),
            (cpmeas.IMAGE, L.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME),
            (cpmeas.IMAGE, L.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME),
            (cpmeas.IMAGE, L.I.C_COUNT + "_" + OBJECTS_NAME),
            (OBJECTS_NAME, L.I.M_LOCATION_CENTER_X),
            (OBJECTS_NAME, L.I.M_LOCATION_CENTER_Y),
            (OBJECTS_NAME, L.I.M_NUMBER_OBJECT_NUMBER))
        for column in columns:
            self.assertTrue(any([
                                    True for object_name, feature in expected_columns
                                    if object_name == column[0] and feature == column[1]]))
        for object_name, feature in expected_columns:
            self.assertTrue(any([
                                    True for column in columns
                                    if object_name == column[0] and feature == column[1]]))

    def test_08_01_get_groupings(self):
        '''Test the get_groupings method'''
        dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        pattern = 'Channel1-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})\\.tif'
        csv_text = '"Image_FileName_Cytoplasm","Image_PathName_Cytoplasm","Metadata_ROW","Metadata_COL"\n'
        for filename in os.listdir(dir):
            match = re.match(pattern, filename)
            if match:
                csv_text += ('"%s","%s","%s","%s"\n' %
                             (filename, dir, match.group("ROW"),
                              match.group("COL")))
        pipeline, module, filename = self.make_pipeline(csv_text)
        self.assertTrue(isinstance(module, L.LoadText))
        module.wants_images.value = True
        module.wants_image_groupings.value = True
        module.metadata_fields.value = "ROW"
        image_set_list = cpi.ImageSetList()
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, None,
                                  None, measurements, image_set_list)
        module.prepare_run(workspace)
        keys, groupings = module.get_groupings(workspace)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], "Metadata_ROW")
        self.assertEqual(len(groupings), 8)
        my_rows = [g[0]["Metadata_ROW"] for g in groupings]
        my_rows.sort()
        self.assertEqual(''.join(my_rows), 'ABCDEFGH')
        for grouping in groupings:
            row = grouping[0]["Metadata_ROW"]
            module.prepare_group(cpw.Workspace(
                    pipeline, module, None, None, measurements, image_set_list),
                    grouping[0], grouping[1])
            for image_number in grouping[1]:
                image_set = image_set_list.get_image_set(image_number - 1)
                measurements.next_image_set(image_number)
                workspace = cpw.Workspace(pipeline, module, image_set,
                                          cpo.ObjectSet(), measurements,
                                          image_set_list)
                module.run(workspace)
                provider = image_set.get_image_provider("Cytoplasm")
                match = re.search(pattern, provider.get_filename())
                self.assertTrue(match)
                self.assertEqual(row, match.group("ROW"))

    def test_09_01_load_bcb_file(self):

        csv_text = '''ELN_RUN_ID,CBIP_RUN_ID,ASSAY_PLATE_BARCODE,\
MX_PLATE_ID,ASSAY_WELL_POSITION,ASSAY_WELL_ROLE,SITE_X,SITE_Y,\
MICROSCOPE,SOURCE_DESCRIPTION,DATE_CREATED,FILE_PATH,FILE_NAME,\
CPD_PLATE_MAP_NAME,CPD_WELL_POSITION,BROAD_ID,\
CPD_MMOL_CONC,SOURCE_NAME,SOURCE_COMPOUND_NAME,CPD_SMILES
"4012-10-W01-01-02","4254","BR00021547","20777","N01","COMPOUND",\
"2","2","GS IX Micro","DAPI","2010/03/19 06:01:12","%s",\
"%s","C-4012-00-D80-001_Rev3","N01",\
"BRD-K71194192-001-01-6","2.132352941","ChemBridge","",\
"Oc1ccnc(SCC(=O)Nc2ccc(Oc3ccccc3)cc2)n1"
''' % (self.test_path, self.test_filename)
        pipeline, module, filename = self.make_pipeline(csv_text)
        c0_ran = [False]

        def callback(workspace):
            imgset = workspace.image_set
            image = imgset.get_image("DAPI")
            pixels = image.pixel_data
            self.assertEqual(pixels.shape[0], self.test_shape[0])
            c0_ran[0] = True

        c0 = C0()
        c0.callback = callback
        c0.module_num = 2
        pipeline.add_module(c0)

        try:
            m = pipeline.run()
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(c0_ran[0])
            hexdigest = m.get_current_image_measurement('MD5Digest_DAPI')
            self.assertEqual(hexdigest, self.test_md5)
            self.assertTrue('PathName_DAPI' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement('PathName_DAPI'),
                             self.test_path)
            self.assertTrue('FileName_DAPI' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement('FileName_DAPI'),
                             self.test_filename)
        finally:
            os.remove(filename)

    def test_10_01_scaling(self):
        '''Test loading an image scaled and unscaled'''
        folder = "loaddata"
        file_name = "1-162hrh2ax2.tif"
        path = make_12_bit_image(folder, file_name, (22, 18))
        csv_text = ("Image_PathName_MyFile,Image_FileName_MyFile\n"
                    "%s,%s\n" % os.path.split(path))
        c0_image = []
        for rescale in (False, True):
            pipeline, module, filename = self.make_pipeline(csv_text)
            try:
                module.rescale.value = rescale

                def callback(workspace):
                    imgset = workspace.image_set
                    image = imgset.get_image("MyFile")
                    pixels = image.pixel_data
                    c0_image.append(pixels.copy())

                c0 = C0()
                c0.callback = callback
                c0.module_num = 2
                pipeline.add_module(c0)
                pipeline.run()
            finally:
                os.remove(filename)
        unscaled, scaled = c0_image
        np.testing.assert_almost_equal(unscaled * 65535. / 4095., scaled)

    def test_11_01_load_objects(self):
        r = np.random.RandomState()
        r.seed(1101)
        labels = r.randint(0, 10, size=(30, 20)).astype(np.uint8)
        handle, name = tempfile.mkstemp(".png")
        write_image(name, labels, PT_UINT8)
        os.close(handle)
        png_path, png_file = os.path.split(name)
        sbs_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        csv_text = """%s_%s,%s_%s,%s_DNA,%s_DNA
%s,%s,Channel2-01-A-01.tif,%s
""" % (L.C_OBJECTS_FILE_NAME, OBJECTS_NAME,
       L.C_OBJECTS_PATH_NAME, OBJECTS_NAME,
       L.C_FILE_NAME, L.C_PATH_NAME,
       png_file, png_path, sbs_dir)
        pipeline, module, csv_name = self.make_pipeline(csv_text)
        assert isinstance(pipeline, cpp.Pipeline)
        assert isinstance(module, L.LoadData)
        module.wants_images.value = True
        try:
            image_set_list = cpi.ImageSetList()
            measurements = cpmeas.Measurements()
            workspace = cpw.Workspace(
                    pipeline, module, None, None, measurements, image_set_list)
            pipeline.prepare_run(workspace)
            key_names, g = pipeline.get_groupings(workspace)
            self.assertEqual(len(g), 1)
            module.prepare_group(workspace, g[0][0], g[0][1])
            image_set = image_set_list.get_image_set(g[0][1][0] - 1)
            object_set = cpo.ObjectSet()
            workspace = cpw.Workspace(pipeline, module, image_set,
                                      object_set, measurements, image_set_list)
            module.run(workspace)
            objects = object_set.get_objects(OBJECTS_NAME)
            self.assertTrue(np.all(objects.segmented == labels))
            self.assertEqual(measurements.get_current_image_measurement(
                    L.I.FF_COUNT % OBJECTS_NAME), 9)
            for feature in (L.I.M_LOCATION_CENTER_X,
                            L.I.M_LOCATION_CENTER_Y,
                            L.I.M_NUMBER_OBJECT_NUMBER):
                value = measurements.get_current_measurement(
                        OBJECTS_NAME, feature)
                self.assertEqual(len(value), 9)
        finally:
            clear_image_reader_cache()
            os.remove(name)
            os.remove(csv_name)

    # def test_12_01_load_unicode(self):
    #     base_directory = tempfile.mkdtemp()
    #     directory = u"\u2211\u03B1"
    #     filename = u"\u03B2.jpg"
    #     base_path = os.path.join(base_directory, directory)
    #     os.mkdir(base_path)
    #     path = os.path.join(base_path, filename)
    #     csv_filename = u"\u03b3.csv"
    #     csv_path = os.path.join(base_path, csv_filename)
    #     unicode_value = u"\u03b4.csv"
    #     try:
    #         r = np.random.RandomState()
    #         r.seed(1101)
    #         labels = r.randint(0, 10, size=(30, 20)).astype(np.uint8)
    #         write_image(path, labels, PT_UINT8)
    #         csv_text = ("Image_FileName_MyFile,Image_PathName_MyFile,Metadata_Unicode\n"
    #                     "%s,%s,%s\n" %
    #                     (filename.encode('utf8'), base_path.encode('utf8'),
    #                      unicode_value.encode('utf8')))
    #         pipeline, module, _ = self.make_pipeline(csv_text, csv_path)
    #         image_set_list = cpi.ImageSetList()
    #         m = cpmeas.Measurements()
    #         workspace = cpw.Workspace(pipeline, module, None, None,
    #                                   m, image_set_list)
    #         self.assertTrue(module.prepare_run(workspace))
    #         self.assertEqual(len(m.get_image_numbers()), 1)
    #         key_names, group_list = pipeline.get_groupings(workspace)
    #         self.assertEqual(len(group_list), 1)
    #         group_keys, image_numbers = group_list[0]
    #         self.assertEqual(len(image_numbers), 1)
    #         module.prepare_group(workspace, group_keys, image_numbers)
    #         image_set = image_set_list.get_image_set(image_numbers[0] - 1)
    #         workspace = cpw.Workspace(pipeline, module, image_set,
    #                                   cpo.ObjectSet(), m, image_set_list)
    #         module.run(workspace)
    #         pixel_data = image_set.get_image("MyFile").pixel_data
    #         self.assertEqual(pixel_data.shape[0], 30)
    #         self.assertEqual(pixel_data.shape[1], 20)
    #         value = m.get_current_image_measurement("Metadata_Unicode")
    #         self.assertEqual(value, unicode_value)
    #     finally:
    #         if os.path.exists(path):
    #             try:
    #                 os.unlink(path)
    #             except:
    #                 pass
    #
    #         if os.path.exists(csv_path):
    #             try:
    #                 os.unlink(csv_path)
    #             except:
    #                 pass
    #         if os.path.exists(base_path):
    #             try:
    #                 os.rmdir(base_path)
    #             except:
    #                 pass
    #         if os.path.exists(base_directory):
    #             try:
    #                 os.rmdir(base_directory)
    #             except:
    #                 pass

    def test_13_01_load_filename(self):
        #
        # Load a file, only specifying the FileName in the CSV
        #
        csv_text = '''"Image_FileName_DNA"
"%s"
''' % self.test_filename
        pipeline, module, filename = self.make_pipeline(csv_text)
        assert isinstance(module, L.LoadData)
        module.image_directory.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.image_directory.custom_path = self.test_path
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                  m, cpi.ImageSetList())
        self.assertTrue(module.prepare_run(workspace))
        self.assertEqual(m.get_measurement(cpmeas.IMAGE, "FileName_DNA", 1),
                         self.test_filename)
        path = m.get_measurement(cpmeas.IMAGE, "PathName_DNA", 1)
        self.assertEqual(path, self.test_path)
        self.assertEqual(
                m.get_measurement(cpmeas.IMAGE, "URL_DNA", 1),
                L.pathname2url(os.path.join(self.test_path, self.test_filename)))
        module.prepare_group(workspace, {}, [1])
        module.run(workspace)
        img = workspace.image_set.get_image("DNA", must_be_grayscale=True)
        self.assertEqual(tuple(img.pixel_data.shape), self.test_shape)

    def test_13_02_load_url(self):
        #
        # Load, only specifying URL
        #
        csv_text = '''"Image_URL_DNA"
"%(cp_logo_url)s"
"http:%(cp_logo_url_filename)s"
"bogusurl.png"
''' % globals()
        pipeline, module, filename = self.make_pipeline(csv_text)
        assert isinstance(module, L.LoadData)
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                  m, cpi.ImageSetList())
        self.assertTrue(module.prepare_run(workspace))
        self.assertEqual(m.get_measurement(cpmeas.IMAGE, "FileName_DNA", 1),
                         cp_logo_url_filename)
        path = m.get_measurement(cpmeas.IMAGE, "PathName_DNA", 1)
        self.assertEqual(path, cp_logo_url_folder)
        self.assertEqual(m[cpmeas.IMAGE, "URL_DNA", 1], cp_logo_url)
        self.assertEqual(m[cpmeas.IMAGE, "FileName_DNA", 2], cp_logo_url_filename)
        self.assertEqual(m[cpmeas.IMAGE, "PathName_DNA", 2], "http:")
        self.assertEqual(m[cpmeas.IMAGE, "FileName_DNA", 3], "bogusurl.png")
        self.assertEqual(m[cpmeas.IMAGE, "PathName_DNA", 3], "")
        module.prepare_group(workspace, {}, [1])
        module.run(workspace)
        img = workspace.image_set.get_image("DNA", must_be_color=True)
        self.assertEqual(tuple(img.pixel_data.shape), cp_logo_url_shape)

    def test_13_03_extra_fields(self):
        #
        # Regression test of issue #853, extra fields
        #
        csv_text = '''"Image_URL_DNA"
"%(cp_logo_url)s", "foo"
"http:%(cp_logo_url_filename)s"
"bogusurl.png"
''' % globals()
        pipeline, module, filename = self.make_pipeline(csv_text)
        assert isinstance(module, L.LoadData)
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                  m, cpi.ImageSetList())
        self.assertTrue(module.prepare_run(workspace))
        self.assertEqual(m.get_measurement(cpmeas.IMAGE, "FileName_DNA", 1),
                         cp_logo_url_filename)
        path = m.get_measurement(cpmeas.IMAGE, "PathName_DNA", 1)
        self.assertEqual(path, cp_logo_url_folder)
        self.assertEqual(m.get_measurement(cpmeas.IMAGE, "URL_DNA", 1),
                         cp_logo_url)
        self.assertEqual(m[cpmeas.IMAGE, "FileName_DNA", 2], cp_logo_url_filename)
        self.assertEqual(m[cpmeas.IMAGE, "PathName_DNA", 2], "http:")
        self.assertEqual(m[cpmeas.IMAGE, "FileName_DNA", 3], "bogusurl.png")
        self.assertEqual(m[cpmeas.IMAGE, "PathName_DNA", 3], "")
        module.prepare_group(workspace, {}, [1])
        module.run(workspace)
        img = workspace.image_set.get_image("DNA", must_be_color=True)
        self.assertEqual(tuple(img.pixel_data.shape), cp_logo_url_shape)

    def test_13_04_extra_lines(self):
        #
        # Regression test of issue #1211 - extra line at end / blank lines
        #
        dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        file_name = 'Channel2-01-A-01.tif'

        csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"

''' % (file_name, dir)
        pipeline, module, filename = self.make_pipeline(csv_text)
        try:
            assert isinstance(module, L.LoadData)
            m = cpmeas.Measurements()
            workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                      m, cpi.ImageSetList())
            self.assertTrue(module.prepare_run(workspace))
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertEqual(m.image_set_count, 1)
            self.assertTrue('FileName_DNA' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m[cpmeas.IMAGE, 'FileName_DNA', 1], file_name)
        finally:
            os.remove(filename)

    def test_13_05_extra_lines_skip_rows(self):
        #
        # Regression test of issue #1211 - extra line at end / blank lines
        # Different code path from 13_04
        #
        path = os.path.join(example_images_directory(), "ExampleSBSImages")
        file_names = ['Channel2-01-A-01.tif',
                      'Channel2-02-A-02.tif']

        csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"

"%s","%s"

"%s","%s"

''' % (file_names[0], path, file_names[1], path)
        pipeline, module, filename = self.make_pipeline(csv_text)
        try:
            assert isinstance(module, L.LoadData)
            m = cpmeas.Measurements()
            workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                      m, cpi.ImageSetList())
            module.wants_rows.value = True
            module.row_range.min = 2
            module.row_range.max = 3
            self.assertTrue(module.prepare_run(workspace))
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertEqual(m.image_set_count, 1)
            self.assertTrue('FileName_DNA' in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m[cpmeas.IMAGE, 'FileName_DNA', 1], file_names[0])
        finally:
            os.remove(filename)

    def test_13_06_load_default_input_folder(self):
        # Regression test of issue #1365 - load a file from the default
        # input folder and check that PathName_xxx is absolute
        csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"\n"%s","%s"''' \
                   % (self.test_filename, self.test_path)
        pipeline, module, filename = self.make_pipeline(csv_text)
        try:
            assert isinstance(module, L.LoadData)
            module.image_directory.dir_choice = cps.ABSOLUTE_FOLDER_NAME
            module.image_directory.custom_path = self.test_path
            m = cpmeas.Measurements()
            workspace = cpw.Workspace(pipeline, module, m, cpo.ObjectSet(),
                                      m, cpi.ImageSetList())
            self.assertTrue(module.prepare_run(workspace))
            self.assertEqual(m.get_measurement(cpmeas.IMAGE, "FileName_DNA", 1),
                             self.test_filename)
            path_out = m.get_measurement(cpmeas.IMAGE, "PathName_DNA", 1)
            self.assertEqual(self.test_path, path_out)
            self.assertEqual(
                    m.get_measurement(cpmeas.IMAGE, "URL_DNA", 1),
                    L.pathname2url(os.path.join(self.test_path, self.test_filename)))
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            img = workspace.image_set.get_image("DNA", must_be_grayscale=True)
            self.assertEqual(tuple(img.pixel_data.shape), self.test_shape)
        finally:
            os.remove(filename)


class C0(cpm.Module):
    module_name = 'C0'
    variable_revision_number = 1

    def create_settings(self):
        self.callback = None

    def settings(self):
        return []

    def run(self, workspace):
        self.callback(workspace)
