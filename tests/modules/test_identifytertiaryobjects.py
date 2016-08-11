import StringIO
import base64
import unittest
import zlib

import cellprofiler.identify
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.identifytertiaryobjects
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

PRIMARY = "primary"
SECONDARY = "secondary"
TERTIARY = "tertiary"
OUTLINES = "Outlines"


class TestIdentifyTertiaryObjects(unittest.TestCase):
    def on_pipeline_event(self, caller, event):
        self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

    def make_workspace(self, primary_labels, secondary_labels):
        """Make a workspace that has objects for the input labels

        returns a workspace with the following
            object_set - has object with name "primary" containing
                         the primary labels
                         has object with name "secondary" containing
                         the secondary labels
        """
        isl = cellprofiler.image.ImageSetList()
        module = cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion()
        module.module_num = 1
        module.primary_objects_name.value = PRIMARY
        module.secondary_objects_name.value = SECONDARY
        module.subregion_objects_name.value = TERTIARY
        workspace = cellprofiler.workspace.Workspace(cellprofiler.pipeline.Pipeline(),
                                                     module,
                                                     isl.get_image_set(0),
                                                     cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(),
                                                     isl)
        workspace.pipeline.add_module(module)

        for labels, name in ((primary_labels, PRIMARY),
                             (secondary_labels, SECONDARY)):
            objects = cellprofiler.region.Region()
            objects.segmented = labels
            workspace.object_set.add_objects(objects, name)
        return workspace

    def test_00_00_zeros(self):
        """Test IdentifyTertiarySubregion on an empty image"""
        primary_labels = numpy.zeros((10, 10), int)
        secondary_labels = numpy.zeros((10, 10), int)
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s" % TERTIARY
        self.assertTrue(count_feature in
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(numpy.product(value.shape), 1)
        self.assertEqual(value, 0)
        self.assertTrue(TERTIARY in workspace.object_set.object_names)
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(numpy.all(output_objects.segmented == primary_labels))
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name in (cellprofiler.measurement.IMAGE, PRIMARY, SECONDARY, TERTIARY):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_01_01_one_object(self):
        """Test creation of a single tertiary object"""
        primary_labels = numpy.zeros((10, 10), int)
        secondary_labels = numpy.zeros((10, 10), int)
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels = numpy.zeros((10, 10), int)
        expected_labels[2:7, 3:8] = 1
        expected_labels[4, 5] = 0
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s" % TERTIARY
        self.assertTrue(count_feature in
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(numpy.product(value.shape), 1)
        self.assertEqual(value, 1)

        self.assertTrue(TERTIARY in measurements.get_object_names())
        child_count_feature = "Children_%s_Count" % TERTIARY
        for parent_name in (PRIMARY, SECONDARY):
            parents_of_feature = ("Parent_%s" % parent_name)
            self.assertTrue(parents_of_feature in
                            measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY,
                                                         parents_of_feature)
            self.assertTrue(numpy.product(value.shape), 1)
            self.assertTrue(value[0], 1)
            self.assertTrue(child_count_feature in
                            measurements.get_feature_names(parent_name))
            value = measurements.get_current_measurement(parent_name,
                                                         child_count_feature)
            self.assertTrue(numpy.product(value.shape), 1)
            self.assertTrue(value[0], 1)

        for axis, expected in (("X", 5), ("Y", 4)):
            feature = "Location_Center_%s" % axis
            self.assertTrue(feature in measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY, feature)
            self.assertTrue(numpy.product(value.shape), 1)
            self.assertEqual(value[0], expected)

        self.assertTrue(TERTIARY in workspace.object_set.object_names)
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(numpy.all(output_objects.segmented == expected_labels))

    def test_01_02_two_objects(self):
        """Test creation of two tertiary objects"""
        primary_labels = numpy.zeros((10, 20), int)
        secondary_labels = numpy.zeros((10, 20), int)
        expected_primary_parents = numpy.zeros((10, 20), int)
        expected_secondary_parents = numpy.zeros((10, 20), int)
        centers = ((4, 5, 1, 2), (4, 15, 2, 1))
        for x, y, primary_label, secondary_label in centers:
            primary_labels[x - 1:x + 2, y - 1:y + 2] = primary_label
            secondary_labels[x - 2:x + 3, y - 2:y + 3] = secondary_label
            expected_primary_parents[x - 2:x + 3, y - 2:y + 3] = primary_label
            expected_primary_parents[x, y] = 0
            expected_secondary_parents[x - 2:x + 3, y - 2:y + 3] = secondary_label
            expected_secondary_parents[x, y] = 0

        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        count_feature = "Count_%s" % TERTIARY
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(value, 2)

        child_count_feature = "Children_%s_Count" % TERTIARY
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        for parent_name, idx, parent_labels in ((PRIMARY, 2, expected_primary_parents),
                                                (SECONDARY, 3, expected_secondary_parents)):
            parents_of_feature = ("Parent_%s" % parent_name)
            cvalue = measurements.get_current_measurement(parent_name,
                                                          child_count_feature)
            self.assertTrue(numpy.all(cvalue == 1))
            pvalue = measurements.get_current_measurement(TERTIARY,
                                                          parents_of_feature)
            for value in (pvalue, cvalue):
                self.assertTrue(numpy.product(value.shape), 2)
            #
            # Make an array that maps the parent label index to the
            # corresponding child label index
            #
            label_map = numpy.zeros((len(centers) + 1,), int)
            for center in centers:
                label = center[idx]
                label_map[label] = pvalue[center[idx] - 1]
            expected_labels = label_map[parent_labels]
            self.assertTrue(numpy.all(expected_labels == output_labels))

    def test_01_03_overlapping_secondary(self):
        """Make sure that an overlapping tertiary is assigned to the larger parent"""
        expected_primary_parents = numpy.zeros((10, 20), int)
        expected_secondary_parents = numpy.zeros((10, 20), int)
        primary_labels = numpy.zeros((10, 20), int)
        secondary_labels = numpy.zeros((10, 20), int)
        primary_labels[3:6, 3:10] = 2
        primary_labels[3:6, 10:17] = 1
        secondary_labels[2:7, 2:12] = 1
        expected_primary_parents[2:7, 2:12] = 2
        expected_primary_parents[4, 4:12] = 0  # the middle of the primary
        expected_primary_parents[4, 9] = 2  # the outline of primary # 2
        expected_primary_parents[4, 10] = 2  # the outline of primary # 1
        expected_secondary_parents[expected_primary_parents > 0] = 1
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        self.assertTrue(isinstance(module, cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion))
        module.use_outlines.value = True
        module.outlines_name.value = OUTLINES
        module.run(workspace)
        measurements = workspace.measurements
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        output_outlines = workspace.image_set.get_image(OUTLINES,
                                                        must_be_binary=True)
        self.assertTrue(numpy.all(output_labels[output_outlines.pixel_data] > 0))
        for parent_name, parent_labels in ((PRIMARY, expected_primary_parents),
                                           (SECONDARY, expected_secondary_parents)):
            parents_of_feature = ("Parent_%s" % parent_name)
            pvalue = measurements.get_current_measurement(TERTIARY,
                                                          parents_of_feature)
            label_map = numpy.zeros((numpy.product(pvalue.shape) + 1,), int)
            label_map[1:] = pvalue.flatten()
            mapped_labels = label_map[output_labels]
            self.assertTrue(numpy.all(parent_labels == mapped_labels))

    def test_01_04_wrong_size(self):
        """Regression test of img-961, what if objects have different sizes?

        Slightly bizarre use case: maybe if user wants to measure background
        outside of cells in a plate of wells???
        """
        expected_primary_parents = numpy.zeros((20, 20), int)
        expected_secondary_parents = numpy.zeros((20, 20), int)
        primary_labels = numpy.zeros((10, 30), int)
        secondary_labels = numpy.zeros((20, 20), int)
        primary_labels[3:6, 3:10] = 2
        primary_labels[3:6, 10:17] = 1
        secondary_labels[2:7, 2:12] = 1
        expected_primary_parents[2:7, 2:12] = 2
        expected_primary_parents[4, 4:12] = 0  # the middle of the primary
        expected_primary_parents[4, 9] = 2  # the outline of primary # 2
        expected_primary_parents[4, 10] = 2  # the outline of primary # 1
        expected_secondary_parents[expected_primary_parents > 0] = 1
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        self.assertTrue(isinstance(module, cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion))
        module.use_outlines.value = True
        module.outlines_name.value = OUTLINES
        module.run(workspace)
        measurements = workspace.measurements
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        output_outlines = workspace.image_set.get_image(OUTLINES,
                                                        must_be_binary=True)
        self.assertTrue(numpy.all(output_labels[output_outlines.pixel_data] > 0))

    def test_02_02_load_v1(self):
        data = ('eJztW89v2zYUlh0nWFagSHfYivXCY7PFhuwmaBoMqT17W7zVjlEbLYqi3Wi'
                'ZtjnQoiFRabyhwI77k3bcn7PjjjtOdCRLZlSLln9ETiVAcN4Tv/fxPZLvUV'
                'JUK7Welb4FRzkV1EqtbBcTBBoEsi41BidAZwegbCDIUAdQ/QS0LAR+tAhQH'
                '4P84Un+6KTwCBRU9YkS7UhVa3ftnz+PFWXH/v3EPtPOpW1HTvlOLjcRY1jv'
                'mdtKRrnv6P+2zxfQwLBN0AtILGR6FK6+qndpazScXKrRjkVQHQ78je2jbg3'
                'ayDDPuy7QudzAl4g08W9IcMFt9hxdYBNT3cE79kXthJcygZfH4a8vvTikhD'
                'hk7POBT8/bnyle+0xA3O752u85MtY7+AJ3LEgAHsDepBfcnhpib2vK3pZSq'
                'ZfGuOMQ3I7Qj51xnDWCsCKF3xbwXC4jQkyn38UQ/J6A52cLXbLsd5dQY2AA'
                'mdaPgx+L8oeNX2oKn1IeSfIG9TuvHhyqkvG/I+C53DDoEPYgsxfHWC9jZ1e'
                'ww+XyiNEhgebA0cvY+VSww+UKBTplwDKRZ2fe9fDKXk3r9CNoXk/snFuMYB'
                '3J+JGespNW6lTO/w/hwvodNB/OmWmBHwhtQ7Jw/MPy4hcCP5crqAstwkCVJ'
                '0VQwQbSGDVGC/VjXlw+p17D7Qg493Bxu87vIrhiSD9lxytKPVJz6vg4yDt/'
                'LMGfmxyvIFxmCpfhPudvwr9Vjo9Mfsirqx3XqHlMZnwkcUeL+BdWh/37wj1'
                'HLvehriNSiFp3qjpDuonZyNePMDtR6ucq57e7D12W//Pyq0vM2zL7trwkTm'
                'b9yczrOr2+j1imfzJ8MnF5HDEuQXliHv/+COH7SZmed1x++/Bp4xt+o41Oc'
                '1/v/8yll/atwHP67vR1Kdt4s+9qypRYA/30tZp98ub3/EHh/VXjJraRY+W+'
                'dJyj7leD6n/rHQWavc80nTvIReLXD+E/Fvi5zGPwCkHDCczh+/0sV9Wozvq'
                'OruDoKnDkaZaZ36LU05cI9/r8ccoFf3Cga+7zhBuqG1J1Luq8CYrj99RAPY'
                'Naemdxv8P4Ze+zlrFfXMc+fd58WLhl/kWtE6us17exTqyy/se5Lqx73xC39'
                'RXHPLBK/9Tc0Y33c9XPaZa5T1s3Li77q7iN86r3VVFx/9z3cCkBF/T+ap3x'
                'Gb/s4gEaytsJWk+0/SvSmGdomXbWuU58/ADrHTRcob2bjs8m5K1Z8duE/BO'
                '0vsLi8Jlgh8v06rXVtUBs0vgnuASX4BLcx46LS/4vhvRjT+gHP716dtWLTY'
                'r7pu4XNsXfBBcPXFzyS4JL8m6C+3hxRWX2PE/qX4JLcAkuwd0u3H8pDye+7'
                '+Cy/z06b/+LjyeoTnylTNcJLmuIkKFB+XdZRm4w/njIzBEKO1df7+Se2X9W'
                'fR/ycJ5hCE9R4Cl+iAd3kM5wdzQ0bDaL0QFkWMtVHW3D1pZcLefth/AGva+'
                'fyWsijeodaIwmnE1Xw/kuQ/jOBL6zMD6GDIZt46bVNlAPU33C23KuNN0r4r'
                'zZDeD3j3/alj5/cG971nxTlOl55s2/f59G4dvaSqfuKtP/H3YnBJdRpuf9+'
                'L2eMt88fzijvetjXNv/D2IFXD8=')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_pipeline_event)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion))
        self.assertEqual(module.primary_objects_name.value, "Nuclei")
        self.assertEqual(module.secondary_objects_name.value, "Cells")
        self.assertEqual(module.subregion_objects_name.value, "Cytoplasm")
        self.assertTrue(module.use_outlines.value)
        self.assertEqual(module.outlines_name.value, "CytoplasmOutline")

    def test_03_01_get_measurement_columns(self):
        """Test the get_measurement_columns method"""
        module = cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion()
        module.primary_objects_name.value = PRIMARY
        module.secondary_objects_name.value = SECONDARY
        module.subregion_objects_name.value = TERTIARY
        columns = module.get_measurement_columns(None)
        expected = ((cellprofiler.measurement.IMAGE, cellprofiler.identify.FF_COUNT % TERTIARY, cellprofiler.measurement.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.identify.M_LOCATION_CENTER_X, cellprofiler.measurement.COLTYPE_FLOAT),
                    (TERTIARY, cellprofiler.identify.M_LOCATION_CENTER_Y, cellprofiler.measurement.COLTYPE_FLOAT),
                    (TERTIARY, cellprofiler.identify.M_NUMBER_OBJECT_NUMBER, cellprofiler.measurement.COLTYPE_INTEGER),
                    (PRIMARY, cellprofiler.identify.FF_CHILDREN_COUNT % TERTIARY, cellprofiler.measurement.COLTYPE_INTEGER),
                    (SECONDARY, cellprofiler.identify.FF_CHILDREN_COUNT % TERTIARY, cellprofiler.measurement.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.identify.FF_PARENT % PRIMARY, cellprofiler.measurement.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.identify.FF_PARENT % SECONDARY, cellprofiler.measurement.COLTYPE_INTEGER))
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cv == ev for cv, ev in zip(column, ec)])
                                 for ec in expected]))

    def test_04_01_do_not_shrink(self):
        """Test the option to not shrink the smaller objects"""
        primary_labels = numpy.zeros((10, 10), int)
        secondary_labels = numpy.zeros((10, 10), int)
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels = numpy.zeros((10, 10), int)
        expected_labels[2:7, 3:8] = 1
        expected_labels[3:6, 4:7] = 0

        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.shrink_primary.value = False
        module.run(workspace)
        measurements = workspace.measurements

        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(numpy.all(output_objects.segmented == expected_labels))

    def test_04_02_do_not_shrink_identical(self):
        """Test a case where the primary and secondary objects are identical"""
        primary_labels = numpy.zeros((20, 20), int)
        secondary_labels = numpy.zeros((20, 20), int)
        expected_labels = numpy.zeros((20, 20), int)

        # first and third objects have different sizes
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels[2:7, 3:8] = 1
        expected_labels[3:6, 4:7] = 0

        primary_labels[13:16, 4:7] = 3
        secondary_labels[12:17, 3:8] = 3
        expected_labels[12:17, 3:8] = 3
        expected_labels[13:16, 4:7] = 0

        # second object and fourth have same size

        primary_labels[3:6, 14:17] = 2
        secondary_labels[3:6, 14:17] = 2
        primary_labels[13:16, 14:17] = 4
        secondary_labels[13:16, 14:17] = 4
        workspace = self.make_workspace(primary_labels, secondary_labels)

        module = workspace.module
        module.shrink_primary.value = False
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(numpy.all(output_objects.segmented == expected_labels))

        measurements = workspace.measurements
        count_feature = "Count_%s" % TERTIARY
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(value, 3)

        child_count_feature = "Children_%s_Count" % TERTIARY
        for parent_name in PRIMARY, SECONDARY:
            parent_of_feature = "Parent_%s" % parent_name
            parent_of = measurements.get_current_measurement(
                    TERTIARY, parent_of_feature)
            child_count = measurements.get_current_measurement(
                    parent_name, child_count_feature)
            for parent, expected_child_count in ((1, 1), (2, 0), (3, 1), (4, 0)):
                self.assertEqual(child_count[parent - 1], expected_child_count)
            for child in (1, 3):
                self.assertEqual(parent_of[child - 1], child)

        for location_feature in (
                cellprofiler.identify.M_LOCATION_CENTER_X, cellprofiler.identify.M_LOCATION_CENTER_Y):
            values = measurements.get_current_measurement(
                    TERTIARY, location_feature)
            self.assertTrue(numpy.all(numpy.isnan(values) == [False, True, False]))

    def test_04_03_do_not_shrink_missing(self):
        # Regression test of 705

        for missing in range(1, 3):
            for missing_primary in False, True:
                primary_labels = numpy.zeros((20, 20), int)
                secondary_labels = numpy.zeros((20, 20), int)
                expected_labels = numpy.zeros((20, 20), int)
                centers = ((5, 5), (15, 5), (5, 15))
                pidx = 1
                sidx = 1
                for idx, (i, j) in enumerate(centers):
                    if (idx + 1 != missing) or not missing_primary:
                        primary_labels[(i - 1):(i + 2), (j - 1):(j + 2)] = pidx
                        pidx += 1
                    if (idx + 1 != missing) or missing_primary:
                        secondary_labels[(i - 2):(i + 3), (j - 2):(j + 3)] = sidx
                        sidx += 1
                expected_labels = secondary_labels * (primary_labels == 0)
                workspace = self.make_workspace(primary_labels, secondary_labels)

                module = workspace.module
                module.shrink_primary.value = False
                module.run(workspace)
                output_objects = workspace.object_set.get_objects(TERTIARY)
                self.assertTrue(numpy.all(output_objects.segmented == expected_labels))

                m = workspace.measurements

                child_name = module.subregion_objects_name.value
                primary_name = module.primary_objects_name.value
                ftr = cellprofiler.identify.FF_PARENT % primary_name
                pparents = m[child_name, ftr]
                self.assertEqual(len(pparents), 3 if missing_primary else 2)
                if missing_primary:
                    self.assertEqual(pparents[missing - 1], 0)

                secondary_name = module.secondary_objects_name.value
                ftr = cellprofiler.identify.FF_PARENT % secondary_name
                pparents = m[child_name, ftr]
                self.assertEqual(len(pparents), 3 if missing_primary else 2)
                if not missing_primary:
                    self.assertTrue(all([x in pparents for x in range(1, 3)]))

                ftr = cellprofiler.identify.FF_CHILDREN_COUNT % child_name
                children = m[primary_name, ftr]
                self.assertEqual(len(children), 2 if missing_primary else 3)
                if not missing_primary:
                    self.assertEqual(children[missing - 1], 0)
                    self.assertTrue(numpy.all(numpy.delete(children, missing - 1) == 1))
                else:
                    self.assertTrue(numpy.all(children == 1))

                children = m[secondary_name, ftr]
                self.assertEqual(len(children), 3 if missing_primary else 2)
                self.assertTrue(numpy.all(children == 1))

    def test_05_00_no_relationships(self):
        workspace = self.make_workspace(numpy.zeros((10, 10), int),
                                        numpy.zeros((10, 10), int))
        workspace.module.run(workspace)
        m = workspace.measurements
        for parent, relationship in (
                (PRIMARY, cellprofiler.modules.identifytertiaryobjects.R_REMOVED),
                (SECONDARY, cellprofiler.modules.identifytertiaryobjects.R_PARENT)):
            result = m.get_relationships(
                    workspace.module.module_num, relationship,
                    parent, TERTIARY)
            self.assertEqual(len(result), 0)

    def test_05_01_relationships(self):
        primary = numpy.zeros((10, 30), int)
        secondary = numpy.zeros((10, 30), int)
        for i in range(3):
            center_j = 5 + i * 10
            primary[3:6, (center_j - 1):(center_j + 2)] = i + 1
            secondary[2:7, (center_j - 2):(center_j + 3)] = i + 1
        workspace = self.make_workspace(primary, secondary)
        workspace.module.run(workspace)
        m = workspace.measurements
        for parent, relationship in (
                (PRIMARY, cellprofiler.modules.identifytertiaryobjects.R_REMOVED),
                (SECONDARY, cellprofiler.modules.identifytertiaryobjects.R_PARENT)):
            result = m.get_relationships(
                    workspace.module.module_num, relationship,
                    parent, TERTIARY)
            self.assertEqual(len(result), 3)
            for i in range(3):
                self.assertEqual(result[cellprofiler.measurement.R_FIRST_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cellprofiler.measurement.R_SECOND_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cellprofiler.measurement.R_FIRST_OBJECT_NUMBER][i], i + 1)
                self.assertEqual(result[cellprofiler.measurement.R_SECOND_OBJECT_NUMBER][i], i + 1)
