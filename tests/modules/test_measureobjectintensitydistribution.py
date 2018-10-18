import StringIO
import unittest

import centrosome.propagate
import numpy
import scipy.stats

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureobjectintensitydistribution
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

OBJECT_NAME = 'objectname'
CENTER_NAME = 'centername'
IMAGE_NAME = 'imagename'
HEAT_MAP_NAME = 'heatmapname'


def feature_frac_at_d(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join([
            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D,
            image_name,
            cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW
        ])

    return cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY + \
           "_" + \
           cellprofiler.modules.measureobjectintensitydistribution.FF_FRAC_AT_D % (image_name, bin, bin_count)


def feature_mean_frac(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join([
            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC,
            image_name,
            cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW
        ])

    return cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY + \
           "_" + \
           cellprofiler.modules.measureobjectintensitydistribution.FF_MEAN_FRAC % (image_name, bin, bin_count)


def feature_radial_cv(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join([
            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV,
            image_name,
            cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW
        ])

    return cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY +\
           "_" + \
           cellprofiler.modules.measureobjectintensitydistribution.FF_RADIAL_CV % (image_name, bin, bin_count)


class TestMeasureObjectIntensityDistribution(unittest.TestCase):
    def test_01_00_please_implement_a_test_of_the_new_version(self):
        self.assertEqual(cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution.variable_revision_number, 5)

    def test_01_03_load_v2(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120126174947

MeasureObjectRadialDistribution:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:2
    Hidden:2
    Select an image to measure:EnhancedGreen
    Select an image to measure:OrigBlue
    Select objects to measure:Nuclei
    Object to use as center?:These objects
    Select objects to use as centers:Cells
    Select objects to measure:Nuclei
    Object to use as center?:Other objects
    Select objects to use as centers:Cells
    Scale bins?:No
    Number of bins:4
    Maximum radius:200
    Scale bins?:Yes
    Number of bins:5
    Maximum radius:50
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution))
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.object_count.value, 2)
        self.assertEqual(module.bin_counts_count.value, 2)
        self.assertEqual(module.images[0].image_name, "EnhancedGreen")
        self.assertEqual(module.images[1].image_name, "OrigBlue")
        self.assertEqual(module.objects[0].object_name, "Nuclei")
        self.assertEqual(module.objects[0].center_choice, cellprofiler.modules.measureobjectintensitydistribution.C_SELF)
        self.assertEqual(module.objects[0].center_object_name, "Cells")
        self.assertEqual(module.objects[1].center_choice, cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER)
        self.assertEqual(module.objects[1].center_object_name, "Cells")
        self.assertEqual(module.bin_counts[0].bin_count, 4)
        self.assertFalse(module.bin_counts[0].wants_scaled)
        self.assertEqual(module.bin_counts[0].maximum_radius, 200)
        self.assertEqual(module.bin_counts[1].bin_count, 5)
        self.assertTrue(module.bin_counts[1].wants_scaled)
        self.assertEqual(module.bin_counts[1].maximum_radius, 50)

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120126174947

MeasureObjectRadialDistribution:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:3
    Hidden:2
    Select an image to measure:EnhancedGreen
    Select an image to measure:OrigBlue
    Select objects to measure:Nuclei
    Object to use as center?:These objects
    Select objects to use as centers:Cells
    Select objects to measure:Nuclei
    Object to use as center?:Centers of other objects
    Select objects to use as centers:Cells
    Select objects to measure:Nuclei
    Object to use as center?:Edges of other objects
    Select objects to use as centers:Cells
    Scale bins?:No
    Number of bins:4
    Maximum radius:200
    Scale bins?:Yes
    Number of bins:5
    Maximum radius:50
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution))
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.object_count.value, 3)
        self.assertEqual(module.bin_counts_count.value, 2)
        self.assertEqual(module.images[0].image_name, "EnhancedGreen")
        self.assertEqual(module.images[1].image_name, "OrigBlue")
        self.assertEqual(module.objects[0].object_name, "Nuclei")
        self.assertEqual(module.objects[0].center_choice, cellprofiler.modules.measureobjectintensitydistribution.C_SELF)
        self.assertEqual(module.objects[0].center_object_name, "Cells")
        self.assertEqual(module.objects[1].center_choice, cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER)
        self.assertEqual(module.objects[1].center_object_name, "Cells")
        self.assertEqual(module.objects[2].center_choice, cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER)
        self.assertEqual(module.objects[2].center_object_name, "Cells")
        self.assertEqual(module.bin_counts[0].bin_count, 4)
        self.assertFalse(module.bin_counts[0].wants_scaled)
        self.assertEqual(module.bin_counts[0].maximum_radius, 200)
        self.assertEqual(module.bin_counts[1].bin_count, 5)
        self.assertTrue(module.bin_counts[1].wants_scaled)
        self.assertEqual(module.bin_counts[1].maximum_radius, 50)
        self.assertEqual(len(module.heatmaps), 0)

    def test_01_05_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150603122126
GitHash:200cfc0
ModuleCount:1
HasImagePlaneDetails:False

MeasureObjectRadialDistribution:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:2
    Hidden:2
    Hidden:2
    Hidden:3
    Select an image to measure:CropGreen
    Select an image to measure:CropRed
    Select objects to measure:Nuclei
    Object to use as center?:These objects
    Select objects to use as centers:Ichthyosaurs
    Select objects to measure:Cells
    Object to use as center?:Edges of other objects
    Select objects to use as centers:Nuclei
    Scale the bins?:Yes
    Number of bins:5
    Maximum radius:100
    Scale the bins?:No
    Number of bins:4
    Maximum radius:100
    Image:CropRed
    Objects to display:Cells
    Number of bins:5
    Measurement:Fraction at Distance
    Color map:Default
    Save display as image?:Yes
    Output image name:Heat
    Image:CropGreen
    Objects to display:Nuclei
    Number of bins:4
    Measurement:Mean Fraction
    Color map:Spectral
    Save display as image?:No
    Output image name:A
    Image:CropRed
    Objects to display:Nuclei
    Number of bins:5
    Measurement:Radial CV
    Color map:Default
    Save display as image?:No
    Output image name:B
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution))
        self.assertEqual(module.wants_zernikes, cellprofiler.modules.measureobjectintensitydistribution.Z_NONE)
        self.assertEqual(module.zernike_degree, 9)
        self.assertEqual(len(module.images), 2)
        for group, image_name in zip(module.images, ("CropGreen", "CropRed")):
            self.assertEqual(group.image_name.value, image_name)
        self.assertEqual(len(module.objects), 2)
        for group, (object_name, center_choice, center_object_name) in zip(
                module.objects, (("Nuclei", cellprofiler.modules.measureobjectintensitydistribution.C_SELF, "Ichthyosaurs"),
                                 ("Cells", cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER, "Nuclei"))):
            self.assertEqual(group.object_name.value, object_name)
            self.assertEqual(group.center_choice.value, center_choice)
            self.assertEqual(group.center_object_name, center_object_name)
        self.assertEqual(len(module.bin_counts), 2)
        for group, (bin_count, scale, max_radius) in zip(
                module.bin_counts, ((5, True, 100), (4, False, 100))):
            self.assertEqual(group.wants_scaled, scale)
            self.assertEqual(group.bin_count, bin_count)
            self.assertEqual(group.maximum_radius, max_radius)
        for group, (image_name, object_name, bin_count, measurement,
                    colormap, wants_to_save, output_image_name) in zip(
                module.heatmaps,
                (("CropRed", "Cells", 5, cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D, cellprofiler.setting.DEFAULT, True, "Heat"),
                 ("CropGreen", "Nuclei", 4, cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC, "Spectral", False, "A"),
                 ("CropRed", "Nuclei", 5, cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV, cellprofiler.setting.DEFAULT, False, "B"))):
            self.assertEqual(group.image_name.value, image_name)
            self.assertEqual(group.object_name.value, object_name)
            self.assertEqual(int(group.bin_count.value), bin_count)
            self.assertEqual(group.measurement, measurement)
            self.assertEqual(group.colormap, colormap)
            self.assertEqual(group.wants_to_save_display, wants_to_save)
            self.assertEqual(group.display_name, output_image_name)

    def test_01_06_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160301131517
GitHash:bd768bc
ModuleCount:2
HasImagePlaneDetails:False

MeasureObjectIntensityDistribution:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:2
    Hidden:2
    Hidden:2
    Hidden:3
    Calculate intensity Zernikes?:Magnitudes only
    Maximum zernike moment:7
    Select an image to measure:CropGreen
    Select an image to measure:CropRed
    Select objects to measure:Nuclei
    Object to use as center?:These objects
    Select objects to use as centers:Ichthyosaurs
    Select objects to measure:Cells
    Object to use as center?:Edges of other objects
    Select objects to use as centers:Nuclei
    Scale the bins?:Yes
    Number of bins:5
    Maximum radius:100
    Scale the bins?:No
    Number of bins:4
    Maximum radius:100
    Image:CropRed
    Objects to display:Cells
    Number of bins:5
    Measurement:Fraction at Distance
    Color map:Default
    Save display as image?:Yes
    Output image name:Heat
    Image:CropGreen
    Objects to display:Nuclei
    Number of bins:4
    Measurement:Mean Fraction
    Color map:Spectral
    Save display as image?:No
    Output image name:A
    Image:CropRed
    Objects to display:Nuclei
    Number of bins:5
    Measurement:Radial CV
    Color map:Default
    Save display as image?:No
    Output image name:B

    MeasureObjectRadialDistribution:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
        Hidden:2
        Hidden:1
        Hidden:1
        Hidden:0
        Calculate intensity Zernikes?:Magnitudes and phase
        Maximum zernike moment:9
        Select an image to measure:CorrBlue
        Select an image to measure:CorrGreen
        Select objects to measure:PropCells
        Object to use as center?:These objects
        Select objects to use as centers:None
        Scale the bins?:Yes
        Number of bins:4
        Maximum radius:100

"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution))
        self.assertEqual(module.wants_zernikes, cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES)
        self.assertEqual(module.zernike_degree, 7)
        self.assertEqual(len(module.images), 2)
        for group, image_name in zip(module.images, ("CropGreen", "CropRed")):
            self.assertEqual(group.image_name.value, image_name)
        self.assertEqual(len(module.objects), 2)
        for group, (object_name, center_choice, center_object_name) in zip(
                module.objects, (("Nuclei", cellprofiler.modules.measureobjectintensitydistribution.C_SELF, "Ichthyosaurs"),
                                 ("Cells", cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER, "Nuclei"))):
            self.assertEqual(group.object_name.value, object_name)
            self.assertEqual(group.center_choice.value, center_choice)
            self.assertEqual(group.center_object_name, center_object_name)
        self.assertEqual(len(module.bin_counts), 2)
        for group, (bin_count, scale, max_radius) in zip(
                module.bin_counts, ((5, True, 100), (4, False, 100))):
            self.assertEqual(group.wants_scaled, scale)
            self.assertEqual(group.bin_count, bin_count)
            self.assertEqual(group.maximum_radius, max_radius)
        for group, (image_name, object_name, bin_count, measurement,
                    colormap, wants_to_save, output_image_name) in zip(
                module.heatmaps,
                (("CropRed", "Cells", 5, cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D, cellprofiler.setting.DEFAULT, True, "Heat"),
                 ("CropGreen", "Nuclei", 4, cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC, "Spectral", False, "A"),
                 ("CropRed", "Nuclei", 5, cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV, cellprofiler.setting.DEFAULT, False, "B"))):
            self.assertEqual(group.image_name.value, image_name)
            self.assertEqual(group.object_name.value, object_name)
            self.assertEqual(int(group.bin_count.value), bin_count)
            self.assertEqual(group.measurement, measurement)
            self.assertEqual(group.colormap, colormap)
            self.assertEqual(group.wants_to_save_display, wants_to_save)
            self.assertEqual(group.display_name, output_image_name)

        module = pipeline.modules()[1]
        self.assertEqual(module.wants_zernikes, cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE)

    def test_02_01_01_get_measurement_columns(self):
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        for i, image_name in ((0, "DNA"), (1, "Cytoplasm"), (2, "Actin")):
            if i:
                module.add_image()
            module.images[i].image_name.value = image_name
        for i, object_name, center_name in ((0, "Nucleii", None),
                                            (1, "Cells", "Nucleii"),
                                            (2, "Cytoplasm", "Nucleii")):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
            if center_name is None:
                module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_SELF
            else:
                module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
                module.objects[i].center_object_name.value = center_name
        for i, bin_count in enumerate((4, 5, 6)):
            if i:
                module.add_bin_count()
            module.bin_counts[i].bin_count.value = bin_count
        module.bin_counts[2].wants_scaled.value = False

        columns = module.get_measurement_columns(None)
        column_dictionary = {}
        for object_name, feature, coltype in columns:
            key = (object_name, feature)
            self.assertFalse(key in column_dictionary)
            self.assertEqual(coltype, cellprofiler.measurement.COLTYPE_FLOAT)
            column_dictionary[key] = (object_name, feature, coltype)

        for object_name in [x.object_name.value for x in module.objects]:
            for image_name in [x.image_name.value for x in module.images]:
                for bin_count, wants_scaled in [
                    (x.bin_count.value, x.wants_scaled.value)
                    for x in module.bin_counts]:
                    for bin in range(1, bin_count + (1 if wants_scaled else 2)):
                        for feature_fn in (feature_frac_at_d,
                                           feature_mean_frac,
                                           feature_radial_cv):
                            measurement = feature_fn(bin, bin_count, image_name)
                            key = (object_name, measurement)
                            self.assertTrue(key in column_dictionary)
                            del column_dictionary[key]
        self.assertEqual(len(column_dictionary), 0)

    def test_02_01_02_get_zernike_columns(self):
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        for wants_zernikes, ftrs in (
                (cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES, (cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,)),
                (cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
                 (cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE, cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE))):
            module.wants_zernikes.value = wants_zernikes
            module.zernike_degree.value = 2
            for i, image_name in ((0, "DNA"), (1, "Cytoplasm"), (2, "Actin")):
                if i:
                    module.add_image()
                module.images[i].image_name.value = image_name
            for i, object_name, center_name in ((0, "Nucleii", None),
                                                (1, "Cells", "Nucleii"),
                                                (2, "Cytoplasm", "Nucleii")):
                if i:
                    module.add_object()
                module.objects[i].object_name.value = object_name
            columns = module.get_measurement_columns(None)
            for image_name in "DNA", "Cytoplasm", "Actin":
                for object_name in "Nucleii", "Cells", "Cytoplasm":
                    for n, m in ((0, 0), (1, 1), (2, 0), (2, 2)):
                        for ftr in ftrs:
                            name = "_".join(
                                    (cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, ftr, image_name, str(n), str(m)))
                            col = (object_name, name, cellprofiler.measurement.COLTYPE_FLOAT)
                            self.assertIn(col, columns)

    def test_02_02_01_get_measurements(self):
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        for i, image_name in ((0, "DNA"), (1, "Cytoplasm"), (2, "Actin")):
            if i:
                module.add_image()
            module.images[i].image_name.value = image_name
        for i, object_name, center_name in ((0, "Nucleii", None),
                                            (1, "Cells", "Nucleii"),
                                            (2, "Cytoplasm", "Nucleii")):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
            if center_name is None:
                module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_SELF
            else:
                module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
                module.objects[i].center_object_name.value = center_name
        for i, bin_count in ((0, 4), (0, 5), (0, 6)):
            if i:
                module.add_bin_count()
            module.bin_counts[i].bin_count.value = bin_count

        for object_name in [x.object_name.value for x in module.objects]:
            self.assertEqual(tuple(module.get_categories(None, object_name)),
                             (cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,))
            for feature in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
                self.assertTrue(feature in module.get_measurements(
                        None, object_name, cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY))
            for image_name in [x.image_name.value for x in module.images]:
                for feature in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
                    self.assertTrue(image_name in
                                    module.get_measurement_images(None,
                                                                  object_name,
                                                                  cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                                                                  feature))
                for bin_count in [x.bin_count.value for x in module.bin_counts]:
                    for bin in range(1, bin_count + 1):
                        for feature in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
                            self.assertTrue("%dof%d" % (bin, bin_count) in
                                            module.get_measurement_scales(
                                                    None, object_name,
                                                    cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, feature,
                                                    image_name))

    def test_02_02_02_get_zernike_measurements(self):
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        for wants_zernikes, ftrs in (
                (cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES, (cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,)),
                (cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
                 (cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE, cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE))):
            module.wants_zernikes.value = wants_zernikes
            module.zernike_degree.value = 2

            for i, image_name in ((0, "DNA"), (1, "Cytoplasm"), (2, "Actin")):
                if i:
                    module.add_image()
                module.images[i].image_name.value = image_name
            for i, object_name, center_name in ((0, "Nucleii", None),
                                                (1, "Cells", "Nucleii"),
                                                (2, "Cytoplasm", "Nucleii")):
                if i:
                    module.add_object()
                module.objects[i].object_name.value = object_name
                if center_name is None:
                    module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_SELF
                else:
                    module.objects[i].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
                    module.objects[i].center_object_name.value = center_name

            for object_name in "Nucleii", "Cells", "Cytoplasm":
                result = module.get_measurements(
                        None, object_name, cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY)
                for ftr in ftrs:
                    self.assertIn(ftr, result)
                    iresult = module.get_measurement_images(
                            None, object_name, cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, ftr)
                    for image in "DNA", "Cytoplasm", "Actin":
                        self.assertIn(image, iresult)
                        sresult = module.get_measurement_scales(
                                None, object_name, cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, ftr, image)
                        for n, m in ((0, 0), (1, 1), (2, 0), (2, 2)):
                            self.assertIn("%d_%d" % (n, m), sresult)

    def test_02_03_default_heatmap_values(self):
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        module.add_heatmap()
        module.heatmaps[0].image_name.value = IMAGE_NAME
        module.heatmaps[0].object_name.value = OBJECT_NAME
        module.heatmaps[0].bin_count.value = 10
        module.images[0].image_name.value = "Bar"
        module.objects[0].object_name.value = "Foo"
        module.bin_counts[0].bin_count.value = 2
        self.assertEqual(module.heatmaps[0].image_name.get_image_name(), "Bar")
        self.assertFalse(module.heatmaps[0].image_name.is_visible())
        self.assertEqual(
                module.heatmaps[0].object_name.get_objects_name(), "Foo")
        self.assertFalse(module.heatmaps[0].object_name.is_visible())
        self.assertEqual(module.heatmaps[0].get_number_of_bins(), 2)
        module.add_image()
        self.assertTrue(module.heatmaps[0].image_name.is_visible())
        self.assertEqual(
                module.heatmaps[0].image_name.get_image_name(), IMAGE_NAME)
        module.add_object()
        self.assertTrue(module.heatmaps[0].object_name.is_visible())
        self.assertEqual(
                module.heatmaps[0].object_name.get_objects_name(), OBJECT_NAME)
        module.add_bin_count()
        self.assertEqual(
                module.heatmaps[0].get_number_of_bins(), 10)

    def run_module(self, image, labels, center_labels=None,
                   center_choice=cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER,
                   bin_count=4,
                   maximum_radius=100, wants_scaled=True,
                   wants_workspace=False,
                   wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_NONE,
                   zernike_degree=2):
        '''Run the module, returning the measurements

        image - matrix representing the image to be analyzed
        labels - labels matrix of objects to be analyzed
        center_labels - labels matrix of alternate centers or None for self
                        centers
        bin_count - # of radial bins
        '''
        module = cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
        module.wants_zernikes.value = wants_zernikes
        module.zernike_degree.value = zernike_degree
        module.images[0].image_name.value = IMAGE_NAME
        module.objects[0].object_name.value = OBJECT_NAME
        object_set = cellprofiler.object.ObjectSet()
        main_objects = cellprofiler.object.Objects()
        main_objects.segmented = labels
        object_set.add_objects(main_objects, OBJECT_NAME)
        if center_labels is None:
            module.objects[0].center_choice.value = cellprofiler.modules.measureobjectintensitydistribution.C_SELF
        else:
            module.objects[0].center_choice.value = center_choice
            module.objects[0].center_object_name.value = CENTER_NAME
            center_objects = cellprofiler.object.Objects()
            center_objects.segmented = center_labels
            object_set.add_objects(center_objects, CENTER_NAME)
        module.bin_counts[0].bin_count.value = bin_count
        module.bin_counts[0].wants_scaled.value = wants_scaled
        module.bin_counts[0].maximum_radius.value = maximum_radius
        module.add_heatmap()
        module.add_heatmap()
        module.add_heatmap()
        for i, (a, f) in enumerate(
                ((cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D, cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D),
                 (cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC, cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC),
                 (cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV, cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV))):
            module.heatmaps[i].image_name.value = IMAGE_NAME
            module.heatmaps[i].object_name.value = OBJECT_NAME
            module.heatmaps[i].bin_count.value = str(bin_count)
            module.heatmaps[i].wants_to_save_display.value = True
            display_name = HEAT_MAP_NAME + f
            module.heatmaps[i].display_name.value = display_name
            module.heatmaps[i].colormap.value = "gray"
            module.heatmaps[i].measurement.value = a
        pipeline = cellprofiler.pipeline.Pipeline()
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = measurements
        img = cellprofiler.image.Image(image)
        image_set.add(IMAGE_NAME, img)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set,
                                                     measurements, image_set_list)
        module.run(workspace)
        if wants_workspace:
            return measurements, workspace
        return measurements

    def test_03_01_zeros_self(self):
        '''Test the module on an empty labels matrix, self-labeled'''
        m = self.run_module(numpy.zeros((10, 10)), numpy.zeros((10, 10), int),
                            wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
                            zernike_degree=2)
        for bin in range(1, 5):
            for feature in (feature_frac_at_d(bin, 4),
                            feature_mean_frac(bin, 4),
                            feature_radial_cv(bin, 4)):
                self.assertTrue(feature in m.get_feature_names(OBJECT_NAME))
                data = m.get_current_measurement(OBJECT_NAME, feature)
                self.assertEqual(len(data), 0)
        for ftr in cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE, cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE:
            for n_, m_ in ((0, 0), (1, 1), (2, 0), (2, 2)):
                feature = "_".join(
                        (cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, ftr, IMAGE_NAME, str(n_), str(m_)))
                self.assertIn(feature, m.get_feature_names(OBJECT_NAME))
                self.assertEqual(len(m[OBJECT_NAME, feature]), 0)

    def test_03_02_circle(self):
        '''Test the module on a uniform circle'''
        i, j = numpy.mgrid[-50:51, -50:51]
        labels = (numpy.sqrt(i * i + j * j) <= 40).astype(int)
        m, workspace = self.run_module(
                numpy.ones(labels.shape), labels,
                wants_workspace=True, wants_zernikes=True, zernike_degree=2)
        assert isinstance(workspace, cellprofiler.workspace.Workspace)
        bins = labels * (1 + (numpy.sqrt(i * i + j * j) / 10).astype(int))
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            area = (float(bin) * 2.0 - 1.0) / 16.0
            self.assertTrue(data[0] > area - .1)
            self.assertTrue(data[0] < area + .1)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D).pixel_data
            data = data.astype(heatmap.dtype)
            self.assertEqual(scipy.stats.mode(heatmap[bins == bin])[0][0], data[0])
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0], 1, 2)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC).pixel_data
            data = data.astype(heatmap.dtype)
            self.assertEqual(scipy.stats.mode(heatmap[bins == bin])[0][0], data[0])
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0], 0, 2)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV).pixel_data
            data = data.astype(heatmap.dtype)
            self.assertEqual(scipy.stats.mode(heatmap[bins == bin])[0][0], data[0])
        module = workspace.module
        assert isinstance(module, cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution)
        data = m[OBJECT_NAME, module.get_zernike_magnitude_name(
                IMAGE_NAME, 0, 0)]
        self.assertEqual(len(data), 1)
        self.assertAlmostEqual(data[0], 1, delta=.001)
        for n_, m_ in ((1, 1), (2, 0), (2, 2)):
            data = m[OBJECT_NAME, module.get_zernike_magnitude_name(
                    IMAGE_NAME, n_, m_)]
            self.assertAlmostEqual(data[0], 0, delta=.001)

    def test_03_03_01_half_circle(self):
        '''Test the module on a circle and an image that's 1/2 zeros

        The measurements here are somewhat considerably off because
        the propagate function uses a Manhattan distance with jaywalking
        allowed instead of the Euclidean distance.
        '''
        i, j = numpy.mgrid[-50:51, -50:51]
        labels = (numpy.sqrt(i * i + j * j) <= 40).astype(int)
        image = numpy.zeros(labels.shape)
        image[i > 0] = (numpy.sqrt(i * i + j * j) / 100)[i > 0]
        image[j == 0] = 0
        image[i == j] = 0
        image[i == -j] = 0
        # 1/2 of the octants should be pretty much all zero and 1/2
        # should be all one
        x = [0, 0, 0, 0, 1, 1, 1, 1]
        expected_cv = numpy.std(x) / numpy.mean(x)
        m = self.run_module(image, labels)
        bin_labels = (numpy.sqrt(i * i + j * j) * 4 / 40.001).astype(int)
        mask = i * i + j * j <= 40 * 40
        total_intensity = numpy.sum(image[mask])
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            bin_count = numpy.sum(bin_labels[mask] == bin - 1)
            frac_in_bin = float(bin_count) / numpy.sum(mask)
            bin_intensity = numpy.sum(image[mask & (bin_labels == bin - 1)])
            expected = bin_intensity / total_intensity
            self.assertTrue(numpy.abs(expected - data[0]) < .2 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            self.assertEqual(len(data), 1)
            expected = expected / frac_in_bin
            self.assertTrue(numpy.abs(data[0] - expected) < .2 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertTrue(numpy.abs(data[0] - expected_cv) < .2 * expected_cv)

    def test_03_03_02_half_circle_zernike(self):
        i, j = numpy.mgrid[-50:50, -50:50]
        ii, jj = [_.astype(float) + .5 for _ in (i, j)]
        labels = (numpy.sqrt(ii * ii + jj * jj) <= 40).astype(int)
        image = numpy.zeros(labels.shape)
        image[ii > 0] = 1
        m = self.run_module(image, labels,
                            wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
                            zernike_degree=2)
        for n_, m_, expected, delta in (
                (0, 0, .5, .001),
                (1, 1, .225, .1),
                (2, 0, 0, .01),
                (2, 2, 0, .01)):
            ftr = "_".join(
                    (cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE, IMAGE_NAME,
                     str(n_), str(m_)))
            self.assertAlmostEqual(m[OBJECT_NAME, ftr][0], expected, delta=delta)
        ftr = "_".join(
                (cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY, cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE, IMAGE_NAME, "1", "1"))
        phase_i_1_1 = m[OBJECT_NAME, ftr][0]
        image = numpy.zeros(labels.shape)
        image[jj > 0] = 1
        m = self.run_module(image, labels,
                            wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
                            zernike_degree=1)
        phase_j_1_1 = m[OBJECT_NAME, ftr][0]
        self.assertAlmostEqual(abs(phase_i_1_1 - phase_j_1_1), numpy.pi / 2, .1)

    def test_03_04_line(self):
        '''Test the alternate centers with a line'''
        labels = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        centers = numpy.zeros(labels.shape, int)
        centers[2, 1] = 1
        distance_to_center = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 1, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0],
                                          [0, 0, 1, 2, 3, 4, 5, 6, 7, 0],
                                          [0, 1, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        distance_to_edge = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        numpy.random.seed(0)
        image = numpy.random.uniform(size=labels.shape)
        m = self.run_module(image, labels, centers)
        total_intensity = numpy.sum(image[labels == 1])
        normalized_distance = distance_to_center / (distance_to_center + distance_to_edge + .001)
        bin_labels = (normalized_distance * 4).astype(int)
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            bin_intensity = numpy.sum(image[(labels == 1) &
                                            (bin_labels == bin - 1)])
            expected = bin_intensity / total_intensity
            self.assertTrue(numpy.abs(expected - data[0]) < .1 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            expected = expected * numpy.sum(labels == 1) / numpy.sum((labels == 1) &
                                                                     (bin_labels == bin - 1))
            self.assertTrue(numpy.abs(data[0] - expected) < .1 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)

    def test_03_05_no_scaling(self):
        i, j = numpy.mgrid[-40:40, -40:40]
        #
        # I'll try to calculate the distance the same way as propagate
        # jaywalk min(i,j) times and go straight abs(i - j) times
        #
        jaywalks = numpy.minimum(numpy.abs(i), numpy.abs(j))
        straights = numpy.abs(numpy.abs(i) - numpy.abs(j))
        distance = jaywalks * numpy.sqrt(2) + straights
        labels = (distance <= 35).astype(int)
        r = numpy.random.RandomState()
        r.seed(35)
        image = r.uniform(size=i.shape)
        total_intensity = numpy.sum(image[labels == 1])
        bin_labels = (distance / 5).astype(int)
        bin_labels[bin_labels > 4] = 4
        m = self.run_module(image, labels, bin_count=4,
                            maximum_radius=20, wants_scaled=False)
        for bin in range(1, 6):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            bin_intensity = numpy.sum(image[(labels == 1) &
                                            (bin_labels == bin - 1)])
            expected = bin_intensity / total_intensity
            self.assertAlmostEqual(expected, data[0], 4)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            expected = expected * numpy.sum(labels == 1) / numpy.sum((labels == 1) &
                                                                     (bin_labels == bin - 1))
            self.assertAlmostEqual(data[0], expected, 4)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)

    def test_03_06_edges_of_objects(self):
        r = numpy.random.RandomState()
        r.seed(36)
        i, j = numpy.mgrid[-20:21, -20:21]
        labels = ((i > -19) & (i < 19) & (j > -19) & (j < 19)).astype(int)
        centers = numpy.zeros(labels.shape, int)
        centers[(i > -5) * (i < 5) & (j > -5) & (j < 5)] = 1
        image = r.uniform(size=labels.shape)
        m = self.run_module(image, labels,
                            center_labels=centers,
                            center_choice=cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER,
                            bin_count=4,
                            maximum_radius=8,
                            wants_scaled=False)

        _, d_from_center = centrosome.propagate.propagate(numpy.zeros(labels.shape),
                                                          centers,
                                                          (labels > 0), 1)
        good_mask = (labels > 0) & (centers == 0)
        d_from_center = d_from_center[good_mask]
        bins = (d_from_center / 2).astype(int)
        bins[bins > 4] = 4
        bin_counts = numpy.bincount(bins)
        image_sums = numpy.bincount(bins, image[good_mask])
        frac_at_d = image_sums / numpy.sum(image_sums)
        for i in range(1, 6):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(i, 4))
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0], frac_at_d[i - 1])

    def test_03_07_two_circles(self):
        i, j = numpy.mgrid[-50:51, -50:51]
        i, j = [numpy.hstack((x, x)) for x in (i, j)]
        d = numpy.sqrt(i * i + j * j)
        labels = (d <= 40).astype(int)
        labels[:, (j.shape[1] / 2):] *= 2
        img = numpy.zeros(labels.shape)
        img[labels == 1] = 1
        img[labels == 2] = d[labels == 2] / 40
        m, workspace = self.run_module(
                img, labels, wants_workspace=True)
        assert isinstance(workspace, cellprofiler.workspace.Workspace)
        bins = (labels != 0) * (1 + (numpy.sqrt(i * i + j * j) / 10).astype(int))
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 2)
            area = (float(bin) * 2.0 - 1.0) / 16.0
            bin_d = (float(bin) - .5) * 8 / 21
            self.assertLess(numpy.abs(data[0] - area), .1)
            self.assertLess(numpy.abs(data[1] - area * bin_d), .1)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D).pixel_data
            data = data.astype(heatmap.dtype)
            for label in 1, 2:
                mask = (bins == bin) & (labels == label)
                self.assertEqual(scipy.stats.mode(heatmap[mask])[0][0], data[label - 1])
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            self.assertEqual(len(data), 2)
            self.assertAlmostEqual(data[0], 1, 2)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC).pixel_data
            data = data.astype(heatmap.dtype)
            for label in 1, 2:
                mask = (bins == bin) & (labels == label)
                self.assertEqual(scipy.stats.mode(heatmap[mask])[0][0], data[label - 1])
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 2)
            self.assertAlmostEqual(data[0], 0, 2)
            heatmap = workspace.image_set.get_image(
                HEAT_MAP_NAME + cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV).pixel_data
            data = data.astype(heatmap.dtype)
            for label in 1, 2:
                mask = (bins == bin) & (labels == label)
                self.assertEqual(scipy.stats.mode(heatmap[mask])[0][0], data[label - 1])

    def test_04_01_img_607(self):
        '''Regression test of bug IMG-607

        MeasureObjectRadialDistribution fails if there are no pixels for
        some of the objects.
        '''
        numpy.random.seed(41)
        labels = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                              [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                              [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        image = numpy.random.uniform(size=labels.shape)
        for center_labels in (labels, None):
            m = self.run_module(image, labels,
                                center_labels=center_labels,
                                bin_count=4)
            for bin in range(1, 5):
                data = m.get_current_measurement(OBJECT_NAME,
                                                 feature_frac_at_d(bin, 4))
                self.assertEqual(len(data), 3)
                self.assertTrue(numpy.isnan(data[1]))

    def test_04_02_center_outside_of_object(self):
        '''Make sure MeasureObjectRadialDistribution can handle oddly shaped objects'''
        numpy.random.seed(42)
        labels = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        center_labels = numpy.zeros(labels.shape, int)
        center_labels[int(center_labels.shape[0] / 2),
                      int(center_labels.shape[1] / 2)] = 1

        image = numpy.random.uniform(size=labels.shape)
        for center_choice in (cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER, cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER):
            m = self.run_module(image, labels,
                                center_labels=center_labels,
                                center_choice=center_choice,
                                bin_count=4)
            for bin in range(1, 5):
                data = m.get_current_measurement(OBJECT_NAME,
                                                 feature_frac_at_d(bin, 4))
                self.assertEqual(len(data), 1)

        m = self.run_module(image, labels, bin_count=4)
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)

    def test_04_03_wrong_size(self):
        '''Regression test for IMG-961: objects & image of different sizes

        Make sure that the module executes without exception with and
        without centers and with similarly and differently shaped centers
        '''
        numpy.random.seed(43)
        labels = numpy.ones((30, 40), int)
        image = numpy.random.uniform(size=(20, 50))
        m = self.run_module(image, labels)
        centers = numpy.zeros(labels.shape)
        centers[15, 20] = 1
        m = self.run_module(image, labels, centers)
        centers = numpy.zeros((35, 35), int)
        centers[15, 20] = 1
        m = self.run_module(image, labels, centers)

    def test_05_01_more_labels_than_centers(self):
        '''Regression test of img-1463'''
        numpy.random.seed(51)
        i, j = numpy.mgrid[0:100, 0:100]
        ir = (i % 10) - 5
        jr = (j % 10) - 5
        il = (i / 10).astype(int)
        jl = (j / 10).astype(int)
        ll = il + jl * 10 + 1

        center_labels = numpy.zeros((100, 100), int)
        center_labels[ir ** 2 + jr ** 2 < 25] = ll[ir ** 2 + jr ** 2 < 25]

        labels = numpy.zeros((100, 100), int)
        i = numpy.random.randint(1, 98, 2000)
        j = numpy.random.randint(1, 98, 2000)
        order = numpy.lexsort((i, j))
        i = i[order]
        j = j[order]
        duplicate = numpy.hstack([[False], (i[:-1] == i[1:]) & (j[:-1] == j[1:])])
        i = i[~duplicate]
        j = j[~duplicate]
        labels[i, j] = numpy.arange(1, len(i) + 1)
        image = numpy.random.uniform(size=(100, 100))
        #
        # Crash here prior to fix
        #
        m = self.run_module(image, labels, center_labels)

    def test_05_02_more_centers_than_labels(self):
        '''Regression test of img-1463'''
        numpy.random.seed(51)
        i, j = numpy.mgrid[0:100, 0:100]
        ir = (i % 10) - 5
        jr = (j % 10) - 5
        il = (i / 10).astype(int)
        jl = (j / 10).astype(int)
        ll = il + jl * 10 + 1

        labels = numpy.zeros((100, 100), int)
        labels[ir ** 2 + jr ** 2 < 25] = ll[ir ** 2 + jr ** 2 < 25]

        center_labels = numpy.zeros((100, 100), int)
        i = numpy.random.randint(1, 98, 2000)
        j = numpy.random.randint(1, 98, 2000)
        order = numpy.lexsort((i, j))
        i = i[order]
        j = j[order]
        duplicate = numpy.hstack([[False], (i[:-1] == i[1:]) & (j[:-1] == j[1:])])
        i = i[~duplicate]
        j = j[~duplicate]
        center_labels[i, j] = numpy.arange(1, len(i) + 1)
        image = numpy.random.uniform(size=(100, 100))
        #
        # Crash here prior to fix
        #
        m = self.run_module(image, labels, center_labels)
