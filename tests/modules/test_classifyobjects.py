import StringIO
import base64
import unittest
import zlib

import cellprofiler.grid
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.classifyobjects
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

OBJECTS_NAME = "myobjects"
MEASUREMENT_NAME_1 = "Measurement1"
MEASUREMENT_NAME_2 = "Measurement2"
IMAGE_NAME = "image"


class TestClassifyObjects(unittest.TestCase):
    def test_01_03_load_v1(self):
        data = ('eJztXNtu2zYYphwna1pgSLeLFRiG6SoYCkdQTluWmymHZjEQJ8EcdO1NM0Wi'
                'bQ4yaUhUEu9J9lh7lD7CRFu2ZEaObFmWLJdCFJs0v/8jP/48/JKg2tHNxdGx'
                'vK+ocu3oZquBLChfWzptELt9KGNakU9sqFNoygQfyjWC5VNoyOovsnpwuPvz'
                'obov76jqryDZIVVrX3sf/34CYM37fOGdJf+nVT8thU6WrkNKEW46q6AM3vj5'
                'n73zvW4j/c6C73XLhU5AMciv4ga56XaGP9WI6VrwUm+HC3vHpdu+g7Zz1RgA'
                '/Z+v0SO06ugfyDVhUOwPeI8cRLCP9+3zuUNeQjneeos8nNledTj7xzo1WnXq'
                '9cBofk+3SqCbxOnGPjdD+az8OQjKlyN0fh0qv+GnETbRPTJd3ZJRW28Oa83s'
                'HcTYe8HZY+krGzWPvS5KAx/Xnm85PEvXPefxXLwNdce1YRti6gzbo8XY2+Ds'
                'sfPmgYwYS2wHPtKtd4+6QeU26/JJ9Fnj7LD0pWtYEE3WnpccnqVPiYwJlV3H'
                '9zdmR42xI43YkcBOQtx2QtwumMwfV7n2svS2WtlTJ8TH6T0rPq7d5RF8GVwS'
                'DLPs55UROyvgozcb8Lg1Djc4Brh1EPB9iOELz18bfrqKKcQOot1b9q1ps6Up'
                'yBvMDrPpGYcrjeBKHm4yvklw0+gXN/99w+nH0u/uIba6stPRDW9Nv0PYyc/e'
                'd5w9lj6FDd21qFxli418imxoUGJ3c/HPaeeh/YS43YQ4BUw2z7/idGbpK+q4'
                '8u8WudOtyPamqRPfL6qyk6i9g3l6XuNzkvVovroczHXdTKrLk3pWtrPVJYIv'
                'TVxSXfj1Q1XU7Sx1iZrfslh3tBi+9RG+fnq4Ro9p7zT81zH8P3L8LH2GbIdW'
                '6tAg2KzctJBtVs6Ia9NW5Qw1aGum/UKSOOakpWMMrZ1F0DNrv5s17puWd1ud'
                'r79HxXMnlu44qIGg2d/hp2lnVn9LY18zjX5Fjauy3m8mmdeC2KemP0YEQkmv'
                '+1wQQ6eI4NsT6Bm1bz+E6jkvPSfBpRkfRV0fOnEdStpbJmwgHA5okvphDZpI'
                'x8Xyw3nFL0/3u/u5r1vz3c+rqY2/j2PqsUx6bft6pTnOsujfea2LUeu4slNR'
                '9ivKQR7jTYvhy/s6w6Jcl1s0vqRxy1dgtD9Z2iIPt965VPNQmnEK06eFmq0s'
                '6+sN64WIb5LoxbRiDpX1vMTu2xmsDTDY3+UV38TVPyr+7+nG/oXqMa99ctT+'
                '5E/ocbPHAu7ZDXBswBTqkef9mjT9Lep+2xmxYdMmLjaz16kGdZxH/LXo8/ys'
                '82ae95WyHldZ67II/TDJOCpK+5Z9HC2iLmnu6xcZl2T/c0EevL9itXNe+kTF'
                'Wefe9uoixThrkXFJ9PG0YRIVqp1ZXidi4pynGIcWBSfm5QCnged1mTSuT8r/'
                '39sAJ3E49rkZys86Hus9zMwCsk76doriH+cx7Y2K+8nd39CgvQbLCJuwk2P9'
                'i4LTwPM6R12fCum8sHaKor/ACZzACZzACZzACZzAPY/TQrhJ48YgDuqHBUWu'
                'd1HbvyzxSVF0KwpOA1+WPwucwAmcwC0aTgvhvoTrywIncMuI00I4sZ8aj9PA'
                '8zqJeEDg5oHTgBifAidwAidwAidwAidwAidwy4LTQrg89vetUoCTOJzkf5dC'
                '5f+Kqe9brr4sbUDL6tiEvY/ZVtq9lwY7ikV0s/8WXuXC+1oNvZCX8XRieDSO'
                'RxvHg0yIKWp0O7bH5lLS1ikylKqfe+3lHg1yGe9jDO8xx3s8jtd/kW7/GgAa'
                'vMxGqfWzr3rZ3Iuvev0Rw7/H8e+N4zf6Dz93+xVwFP9h6G6f2cmLL/z8/3oE'
                'X9jfSn769Q8rm9+D58cZAKP+Hfj959+S8pbLJUkCT8fpqxg80/EleHowO2+k'
                '6cbbT2B8+UGbl6l8kn6S2AFm1zfgKw/rNuBZhvL/A/z19oA=')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
        self.assertEqual(module.contrast_choice.value, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT)
        self.assertEqual(len(module.single_measurements), 2)

        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_IntegratedIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, cellprofiler.modules.classifyobjects.BC_EVEN)
        self.assertEqual(group.bin_count, 3)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertTrue(group.wants_low_bin)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertTrue(group.wants_high_bin)
        self.assertTrue(group.wants_custom_names)
        for name, expected in zip(group.bin_names.value.split(','),
                                  ('First', 'Second', 'Third', 'Fourth', 'Fifth')):
            self.assertEqual(name, expected)
        self.assertTrue(group.wants_images)
        self.assertEqual(group.image_name, "ClassifiedNuclei")

        group = module.single_measurements[1]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_MaxIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, cellprofiler.modules.classifyobjects.BC_CUSTOM)
        self.assertEqual(group.custom_thresholds, ".2,.5,.8")
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
        self.assertEqual(module.contrast_choice, cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Location_Center_X")
        self.assertEqual(module.first_threshold_method, cellprofiler.modules.classifyobjects.TM_MEDIAN)
        self.assertEqual(module.second_measurement, "Location_Center_Y")
        self.assertEqual(module.second_threshold_method, cellprofiler.modules.classifyobjects.TM_MEDIAN)

    def test_01_04_load_v2(self):
        data = ('eJztne9z2jYYxwUhXdPe7dJtL/qmN7/a7XrEA5bs0rwZ+VEW7kLCDa5rX6WO'
                'LUA7I3G2nIT91fkTZoEdG8XExtgGU3EhIOHP80hfP5L1iBBax92L4xPpQK5I'
                'rePuXg/pUGrrCu0RY3gkYVqWTg2oUKhJBB9JLYKlM6hKtapUPTiq/XG0/0Gq'
                'VSofQLxbodn63n5o9wF4YT++tO9F56Vtp1zw3Vm5AylFuG9ugxJ469Q/2PdP'
                'ioGUGx1+UnQLmp4Lt76Je6Q7Hj2+1CKapcNLZeg/2L5dWsMbaJhXPRd0Xm6j'
                'e6h30H+Q64J72N/wFpmIYId37PO1j34J5fx2BuSuYdjN4eyfKFQddKh9Bmbr'
                'mW4PFU+3Aqcbe3zvq2fHnwPv+FKAzm98x+86ZYQ1dIs0S9ElNFT6j61m9g5D'
                '7L3k7LHylYH6J/YpYnwlhC/M8AVQjdiPHzi/rNyxg8YO7SFUTMuAQ4hpdF1+'
                '4uyxcltBhkR6foOma68eYm+Xs8fuXXhP9z7eKyqVhuyUR9H3BWeHlS8tVYdo'
                'Neen5vQ/7nldlPsdROvnNtdPVq5WyvuViPyyOofxYf0uzfAlcEkwjBJnrzi/'
                'rHxGJEyoZJnOfBLF/9aMnS3wxZ4FeO4Fx7k3l9sBnr/PIf5+4drNyk1MITYR'
                'HV+zZ32DXZK8Ojdql9MzjCvOcEWbi+YvCreIfnHmv4+3EOtjyRwpqn0tv0HY'
                'XJ29oPn0DPYUS6dSk11kpAbRNWi49rKOz7jzUFbzngyiXWdeczqz8hU1Lekv'
                'ndwoemB/k9SJPy8VuRarv+48ndb4jHJe0tQlKF6T5OLq8uT8lasZx8thquMq'
                'ri789aMiV6p5jZdF2lkP8bcz429afrxGz+lvqvETEK+LcHHWz6cDBWOo1/Ko'
                '17LzSdr5Bu+3WllOn3aIv5+59rJyAxkmLXegSrBW7g6QoZUbxDLooNxAPTpY'
                'av25bLwtOk8egHzFV1rr+qznyaD8/1RXTBP1ENSmmWGSdtLSLQq3iG55zcOz'
                'jq8485aXK7eU+4DEOe7+4AVRFYoIvj6FtlHj+rOvnaucV5LMp3/k+s3Kp5ZJ'
                'yXBPgz2E/Qlw3DhsQQ0pOF+6pbUuf5oHHGScd1RW4i+J8fdlTjuSvH6t276C'
                'G8dxxt10HKfb3qT3/xbVpxoQz9Hicn+j5qM016Npxos9/S3lL05eIdfK8kFZ'
                'PozCf8fxrKyTu2v7nvk81L0jksrWwNC7Hmc9ruPozfQaoP4gzXVM0PXjH2g7'
                'ZW/v37I3srEK5/Q/Sd2SzH+ZZizQ0oyzoPfRGsSAfYNYWEtfr3lxFtbuoP2l'
                'iV7s1xz/6zy/h/U3qXw46/k+6X2ZdciD4+qTNbfp/UtTlxZU8KrbuQ66JL1e'
                'yyMnxlEwJ8ZRMCfGUTCX9Xpknbk465gLcmf/5KufaekTlJef20v/iyXz8rxw'
                'cfSxtWES5aqfWe4LMXHOl9ynyCMn5mWPq4PndYmThy/U7t88rsBx7JH/3EOW'
                '+0KTD0mwjaHR8v7zwp2D53UK2nckN/9ClU6EkhDW4Mhnrx5iT+gef3/cp/va'
                '2smL/oITnOAEJzjBCU5wgnueq/u4qHmjt66fpgl56q/gvk2uDpKJ86TsrLof'
                'm5Lf5iX+BCc4wX07XN3Hif1SwQlOcIITnOAEt075XF64Onhep6TyTqG34AQn'
                'OMEJTnCCE5zgBCc4wW0mV/dxq8gHB0WPK3BcwXle8B3/NaS9/r+z33XKKtT1'
                'kUHY90gY8nDyZQemrBNFm357gHxhP236vkiA+RmF+Klzfurz/CANYop645Fh'
                'e7MoGSoUqXLTqW3btcduLfN7H+L3hPN7Ms+v8yUA0z0A5P5zNbk1rb6aVHP/'
                '2HNyPkL873P+9+f5V6cfrhhPG2DKzoctxlPP5qr8+T9ftBPgzx9vRaf85t3W'
                '+3fg+XEGwGx8e3H/8Gdcv6XSVrEIno7T1yE80/EVeHpjdt4WFhtvv4L5x7t9'
                '3qTj45ynAruB5fX1/JUe2+b62YTj/wfn4gog')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
        self.assertEqual(module.contrast_choice.value, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT)
        self.assertEqual(len(module.single_measurements), 2)

        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_IntegratedIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, cellprofiler.modules.classifyobjects.BC_EVEN)
        self.assertEqual(group.bin_count, 3)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertTrue(group.wants_low_bin)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertTrue(group.wants_high_bin)
        self.assertTrue(group.wants_custom_names)
        for name, expected in zip(group.bin_names.value.split(','),
                                  ('First', 'Second', 'Third', 'Fourth', 'Fifth')):
            self.assertEqual(name, expected)
        self.assertTrue(group.wants_images)
        self.assertEqual(group.image_name, "ClassifiedNuclei")

        group = module.single_measurements[1]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_MaxIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, cellprofiler.modules.classifyobjects.BC_CUSTOM)
        self.assertEqual(group.custom_thresholds, ".2,.5,.8")
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
        self.assertEqual(module.contrast_choice, cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Location_Center_X")
        self.assertEqual(module.first_threshold_method, cellprofiler.modules.classifyobjects.TM_MEDIAN)
        self.assertEqual(module.second_measurement, "Location_Center_Y")
        self.assertEqual(module.second_threshold_method, cellprofiler.modules.classifyobjects.TM_CUSTOM)
        self.assertAlmostEqual(module.second_threshold.value, .4)

    def make_workspace(self, labels, contrast_choice, measurement1=None, measurement2=None):
        object_set = cellprofiler.region.Set()
        objects = cellprofiler.region.Region()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        measurements = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.classifyobjects.ClassifyObjects()
        m_names = []
        if measurement1 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_1,
                                         measurement1)
            m_names.append(MEASUREMENT_NAME_1)
        if measurement2 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_2,
                                         measurement2)
            module.add_single_measurement()
            m_names.append(MEASUREMENT_NAME_2)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        module.contrast_choice.value = contrast_choice
        if module.contrast_choice == cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT:
            for i, m in enumerate(m_names):
                group = module.single_measurements[i]
                group.object_name.value = OBJECTS_NAME
                group.measurement.value = m
                group.image_name.value = IMAGE_NAME
        else:
            module.object_name.value = OBJECTS_NAME
            module.image_name.value = IMAGE_NAME
            module.first_measurement.value = MEASUREMENT_NAME_1
            module.second_measurement.value = MEASUREMENT_NAME_2
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     object_set, measurements,
                                                     image_set_list)
        return workspace, module

    def test_02_01_classify_single_none(self):
        """Make sure the single measurement mode can handle no objects"""
        workspace, module = self.make_workspace(
                numpy.zeros((10, 10), int),
                cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT,
                numpy.zeros((0,), float))
        module.run(workspace)
        for m_name in ("Classify_Measurement1_Bin_1",
                       "Classify_Measurement1_Bin_2",
                       "Classify_Measurement1_Bin_3"):
            m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                               m_name)
            self.assertEqual(len(m), 0)

    def test_02_02_classify_single_even(self):
        m = numpy.array((.5, 0, 1, .1))
        labels = numpy.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = cellprofiler.modules.classifyobjects.BC_EVEN
        module.single_measurements[0].low_threshold.value = .2
        module.single_measurements[0].high_threshold.value = .7
        module.single_measurements[0].bin_count.value = 1
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].wants_images.value = True

        expected_obj = dict(Classify_Measurement1_Bin_1=(0, 1, 0, 1),
                            Classify_Measurement1_Bin_2=(1, 0, 0, 0),
                            Classify_Measurement1_Bin_3=(0, 0, 1, 0))
        expected_img = dict(Classify_Measurement1_Bin_1_NumObjectsPerBin=2,
                            Classify_Measurement1_Bin_2_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_3_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_1_PctObjectsPerBin=50.0,
                            Classify_Measurement1_Bin_2_PctObjectsPerBin=25.0,
                            Classify_Measurement1_Bin_3_PctObjectsPerBin=25.0)
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(numpy.all(values == numpy.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)

        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(numpy.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(numpy.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cellprofiler.measurement.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER if column[1].endswith(
                        cellprofiler.modules.classifyobjects.F_NUM_PER_BIN) else cellprofiler.measurement.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER)

        categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], cellprofiler.modules.classifyobjects.M_CATEGORY)
        names = module.get_measurements(None, cellprofiler.measurement.IMAGE, "foo")
        self.assertEqual(len(names), 0)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], cellprofiler.modules.classifyobjects.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((cellprofiler.modules.classifyobjects.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cellprofiler.measurement.IMAGE, cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((cellprofiler.modules.classifyobjects.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_03_classify_single_custom(self):
        m = numpy.array((.5, 0, 1, .1))
        labels = numpy.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = cellprofiler.modules.classifyobjects.BC_CUSTOM
        module.single_measurements[0].custom_thresholds.value = ".2,.7"
        module.single_measurements[0].bin_count.value = 14  # should ignore
        module.single_measurements[0].wants_custom_names.value = True
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
        module.single_measurements[0].wants_images.value = True

        expected_img = dict(Classify_Three_NumObjectsPerBin=2,
                            Classify_Three_PctObjectsPerBin=50.0,
                            Classify_Blind_NumObjectsPerBin=1,
                            Classify_Blind_PctObjectsPerBin=25.0,
                            Classify_Mice_NumObjectsPerBin=1,
                            Classify_Mice_PctObjectsPerBin=25.0)
        expected_obj = dict(Classify_Three=(0, 1, 0, 1),
                            Classify_Blind=(1, 0, 0, 0),
                            Classify_Mice=(0, 0, 1, 0))
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(numpy.all(values == numpy.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(numpy.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(numpy.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cellprofiler.measurement.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER if column[1].endswith(
                        cellprofiler.modules.classifyobjects.F_NUM_PER_BIN) else cellprofiler.measurement.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER)

        categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 1)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], cellprofiler.modules.classifyobjects.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((cellprofiler.modules.classifyobjects.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cellprofiler.measurement.IMAGE, cellprofiler.modules.classifyobjects.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((cellprofiler.modules.classifyobjects.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_04_last_is_nan(self):
        # regression test for issue #1553
        #
        # Test that classify objects classifies an object whose measurement
        # is NaN as none of the categories. Test for no exception thrown
        # if showing the figure and last object has a measurement of NaN
        #
        for leave_last_out in (False, True):
            m = numpy.array((.5, 0, 1, numpy.NaN))
            if leave_last_out:
                m = m[:-1]
            labels = numpy.zeros((20, 10), int)
            labels[2:5, 3:7] = 1
            labels[12:15, 1:4] = 2
            labels[6:11, 5:9] = 3
            labels[16:19, 5:9] = 4
            workspace, module = self.make_workspace(
                    labels, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m)
            module.single_measurements[0].bin_choice.value = cellprofiler.modules.classifyobjects.BC_CUSTOM
            module.single_measurements[0].custom_thresholds.value = ".2,.7"
            module.single_measurements[0].bin_count.value = 14  # should ignore
            module.single_measurements[0].wants_custom_names.value = True
            module.single_measurements[0].wants_low_bin.value = True
            module.single_measurements[0].wants_high_bin.value = True
            module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
            module.single_measurements[0].wants_images.value = True

            expected_img = dict(Classify_Three_NumObjectsPerBin=1,
                                Classify_Three_PctObjectsPerBin=25.0,
                                Classify_Blind_NumObjectsPerBin=1,
                                Classify_Blind_PctObjectsPerBin=25.0,
                                Classify_Mice_NumObjectsPerBin=1,
                                Classify_Mice_PctObjectsPerBin=25.0)
            expected_obj = dict(Classify_Three=(0, 1, 0, 0),
                                Classify_Blind=(1, 0, 0, 0),
                                Classify_Mice=(0, 0, 1, 0))
            module.run(workspace)
            for measurement, expected_values in expected_obj.iteritems():
                values = workspace.measurements.get_current_measurement(
                        OBJECTS_NAME, measurement)
                self.assertEqual(len(values), 4)
                self.assertTrue(numpy.all(values == numpy.array(expected_values)))
            for measurement, expected_values in expected_img.iteritems():
                values = workspace.measurements.get_current_measurement(
                        cellprofiler.measurement.IMAGE, measurement)
                self.assertTrue(values == expected_values)
            image = workspace.image_set.get_image(IMAGE_NAME)
            pixel_data = image.pixel_data
            self.assertTrue(numpy.all(pixel_data[labels == 0, :] == 0))
            colors = [pixel_data[x, y, :] for x, y in
                      ((2, 3), (12, 1), (6, 5), (16, 5))]
            for i, color in enumerate(colors + [colors[1]]):
                self.assertTrue(numpy.all(pixel_data[labels == i + 1, :] == color))

    def test_03_01_two_none(self):
        workspace, module = self.make_workspace(
                numpy.zeros((10, 10), int),
                cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
                numpy.zeros((0,), float), numpy.zeros((0,), float))
        module.run(workspace)
        for lh1 in ("low", "high"):
            for lh2 in ("low", "high"):
                m_name = ("Classify_Measurement1_%s_Measurement2_%s" %
                          (lh1, lh2))
                m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                   m_name)
                self.assertEqual(len(m), 0)

    def test_03_02_two(self):
        numpy.random.seed(0)
        labels = numpy.zeros((10, 20), int)
        index = 1
        for i_min, i_max in ((1, 4), (6, 9)):
            for j_min, j_max in ((2, 6), (8, 11), (13, 18)):
                labels[i_min:i_max, j_min:j_max] = index
                index += 1
        num_labels = index - 1
        exps = numpy.exp(numpy.arange(numpy.max(labels)))
        m1 = numpy.random.permutation(exps)
        m2 = numpy.random.permutation(exps)
        for wants_custom_names in (False, True):
            for tm1 in (cellprofiler.modules.classifyobjects.TM_MEAN, cellprofiler.modules.classifyobjects.TM_MEDIAN, cellprofiler.modules.classifyobjects.TM_CUSTOM):
                for tm2 in (cellprofiler.modules.classifyobjects.TM_MEAN, cellprofiler.modules.classifyobjects.TM_MEDIAN, cellprofiler.modules.classifyobjects.TM_CUSTOM):
                    workspace, module = self.make_workspace(labels,
                                                            cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
                                                            m1, m2)
                    self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
                    module.first_threshold_method.value = tm1
                    module.first_threshold.value = 8
                    module.second_threshold_method.value = tm2
                    module.second_threshold.value = 70
                    module.wants_image.value = True

                    def cutoff(method, custom_cutoff):
                        if method == cellprofiler.modules.classifyobjects.TM_MEAN:
                            return numpy.mean(exps)
                        elif method == cellprofiler.modules.classifyobjects.TM_MEDIAN:
                            return numpy.median(exps)
                        else:
                            return custom_cutoff

                    c1 = cutoff(tm1, module.first_threshold.value)
                    c2 = cutoff(tm2, module.second_threshold.value)
                    m1_over = m1 >= c1
                    m2_over = m2 >= c2
                    if wants_custom_names:
                        f_names = ("TL", "TR", "BL", "BR")
                        module.wants_custom_names.value = True
                        module.low_low_custom_name.value = f_names[0]
                        module.low_high_custom_name.value = f_names[1]
                        module.high_low_custom_name.value = f_names[2]
                        module.high_high_custom_name.value = f_names[3]
                    else:
                        f_names = ("Measurement1_low_Measurement2_low",
                                   "Measurement1_low_Measurement2_high",
                                   "Measurement1_high_Measurement2_low",
                                   "Measurement1_high_Measurement2_high")
                    m_names = ["_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
                               for name in f_names]

                    module.run(workspace)
                    columns = module.get_measurement_columns(None)
                    for column in columns:
                        if column[0] != OBJECTS_NAME:  # Must be image
                            self.assertEqual(column[0], cellprofiler.measurement.IMAGE)
                            self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER if column[1].endswith(
                                    cellprofiler.modules.classifyobjects.F_NUM_PER_BIN) else cellprofiler.measurement.COLTYPE_FLOAT)
                        else:
                            self.assertEqual(column[0], OBJECTS_NAME)
                            self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_INTEGER)

                    self.assertEqual(len(columns), 12)
                    self.assertEqual(len(set([column[1] for column in columns])), 12)  # no duplicates

                    categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
                    self.assertEqual(len(categories), 1)
                    categories = module.get_categories(None, OBJECTS_NAME)
                    self.assertEqual(len(categories), 1)
                    self.assertEqual(categories[0], cellprofiler.modules.classifyobjects.M_CATEGORY)
                    names = module.get_measurements(None, OBJECTS_NAME, "foo")
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY)
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY)
                    self.assertEqual(len(names), 4)

                    for m_name, expected in zip(m_names,
                                                ((~m1_over) & (~m2_over),
                                                 (~m1_over) & m2_over,
                                                 m1_over & ~m2_over,
                                                 m1_over & m2_over)):
                        m = workspace.measurements.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                                           '_'.join((m_name, cellprofiler.modules.classifyobjects.F_NUM_PER_BIN)))
                        self.assertTrue(m == expected.astype(int).sum())
                        m = workspace.measurements.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                                           '_'.join((m_name, cellprofiler.modules.classifyobjects.F_PCT_PER_BIN)))
                        self.assertTrue(m == 100.0 * float(expected.astype(int).sum()) / num_labels)
                        m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                           m_name)
                        self.assertTrue(numpy.all(m == expected.astype(int)))
                        self.assertTrue(m_name in [column[1] for column in columns])
                        self.assertTrue(m_name in ["_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
                                                   for name in names])
                    image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
                    self.assertTrue(numpy.all(image[labels == 0, :] == 0))
                    colors = image[(labels > 0) & (m[labels - 1] == 1), :]
                    if colors.shape[0] > 0:
                        self.assertTrue(all([numpy.all(colors[:, i] == colors[0, i])
                                             for i in range(3)]))

    def test_03_04_nans(self):
        # Test for NaN values in two measurements.
        #
        labels = numpy.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 3
        labels[3:5, 6:8] = 4
        labels[6:8, 6:8] = 5
        labels[3:5, 10:12] = 2

        m1 = numpy.array((1, 2, numpy.NaN, 1, numpy.NaN))
        m2 = numpy.array((1, 2, 1, numpy.NaN, numpy.NaN))
        for leave_last_out in (False, True):
            end = numpy.max(labels) - 1 if leave_last_out else numpy.max(labels)
            workspace, module = self.make_workspace(
                    labels,
                    cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
                    m1[:end], m2[:end])
            self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
            module.first_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
            module.first_threshold.value = 2
            module.second_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
            module.second_threshold.value = 2
            module.wants_image.value = True
            module.wants_custom_names.value = False
            module.run(workspace)
            f_names = ("Measurement1_low_Measurement2_low",
                       "Measurement1_low_Measurement2_high",
                       "Measurement1_high_Measurement2_low",
                       "Measurement1_high_Measurement2_high")
            m_names = ["_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
                       for name in f_names]
            m = workspace.measurements
            for m_name, expected in zip(
                    m_names,
                    [numpy.array((1, 0, 0, 0, 0)),
                     numpy.array((0, 0, 0, 0, 0)),
                     numpy.array((0, 0, 0, 0, 0)),
                     numpy.array((0, 1, 0, 0, 0))]):
                values = m[OBJECTS_NAME, m_name]
                numpy.testing.assert_array_equal(values, expected)

    def test_03_05_nan_offset_by_1(self):
        # Regression test of 1636
        labels = numpy.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 2

        m1 = numpy.array((4, numpy.NaN))
        m2 = numpy.array((4, 4))
        workspace, module = self.make_workspace(
                labels,
                cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
                m1, m2)
        self.assertTrue(isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects))
        module.first_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
        module.first_threshold.value = 2
        module.second_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
        module.second_threshold.value = 2
        module.wants_image.value = True
        module.wants_custom_names.value = False
        module.run(workspace)
        image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        colors = module.get_colors(4)
        reverse = numpy.zeros(image.shape[:2], int)
        for idx, color in enumerate(colors):
            reverse[
                numpy.all(image == color[numpy.newaxis, numpy.newaxis, :3], 2)] = idx
        self.assertTrue(numpy.all(reverse[labels == 1] == 4))
