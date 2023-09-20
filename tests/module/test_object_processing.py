import numpy

import cellprofiler_core.constants.measurement
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.module.image_segmentation._image_segmentation
import cellprofiler_core.module.image_segmentation._object_processing
import cellprofiler_core.object
import cellprofiler_core.workspace


class TestObjectProcessing:
    def test_get_categories_image(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_categories(
            None, cellprofiler_core.constants.measurement.IMAGE
        )

        expected = [cellprofiler_core.constants.measurement.C_COUNT]

        assert actual == expected

    def test_get_categories_input_object(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "Objects")

        expected = [cellprofiler_core.constants.measurement.C_CHILDREN]

        assert actual == expected

    def test_get_categories_output_object(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "ObjectProcessing")

        expected = [
            cellprofiler_core.constants.measurement.C_LOCATION,
            cellprofiler_core.constants.measurement.C_NUMBER,
            cellprofiler_core.constants.measurement.C_PARENT,
        ]

        assert actual == expected

    def test_get_categories_other(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "foo")

        expected = []

        assert actual == expected

    def test_get_measurement_columns(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(None)

        expected = [
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "Objects",
                cellprofiler_core.constants.measurement.FF_CHILDREN_COUNT
                % "ObjectProcessing",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.FF_PARENT % "Objects",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
        ]

        assert actual == expected

    def test_get_measurement_columns_additional_objects(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(
            None, additional_objects=[("additional_input", "additional_output")]
        )

        expected = [
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "Objects",
                cellprofiler_core.constants.measurement.FF_CHILDREN_COUNT
                % "ObjectProcessing",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.constants.measurement.FF_PARENT % "Objects",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_output",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.constants.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.constants.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.FF_COUNT % "additional_output",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_input",
                cellprofiler_core.constants.measurement.FF_CHILDREN_COUNT
                % "additional_output",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_output",
                cellprofiler_core.constants.measurement.FF_PARENT % "additional_input",
                cellprofiler_core.constants.measurement.COLTYPE_INTEGER,
            ),
        ]

        assert actual == expected

    def test_get_measurements_image_count(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None,
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.C_COUNT,
        )

        expected = ["ObjectProcessing"]

        assert actual == expected

    def test_get_measurements_image_other(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, cellprofiler_core.constants.measurement.IMAGE, "foo"
        )

        expected = []

        assert actual == expected

    def test_get_measurements_input_object_children(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "Objects", cellprofiler_core.constants.measurement.C_CHILDREN
        )

        expected = [
            "%s_Count" % "ObjectProcessing"
        ]

        assert actual == expected

    def test_get_measurements_input_object_other(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "Objects", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_output_object_location(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.constants.measurement.C_LOCATION
        )

        expected = [
            cellprofiler_core.constants.measurement.FTR_CENTER_X,
            cellprofiler_core.constants.measurement.FTR_CENTER_Y,
            cellprofiler_core.constants.measurement.FTR_CENTER_Z,
        ]

        assert actual == expected

    def test_get_measurements_output_object_number(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.constants.measurement.C_NUMBER
        )

        expected = [cellprofiler_core.constants.measurement.FTR_OBJECT_NUMBER]

        assert actual == expected

    def test_get_measurements_output_object_parent(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.constants.measurement.C_PARENT
        )

        expected = ["Objects"]

        assert actual == expected

    def test_get_measurements_output_object_other(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_other_other(self):
        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "foo", "bar")

        expected = []

        assert actual == expected

    def test_add_measurements(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        object_set = cellprofiler_core.object.ObjectSet()

        labels = numpy.zeros((30, 30), dtype=numpy.uint8)

        i, j = numpy.mgrid[-15:15, -7:23]
        labels[i ** 2 + j ** 2 <= 25] = 1

        i, j = numpy.mgrid[-15:15, -22:8]
        labels[i ** 2 + j ** 2 <= 16] = 2

        objects = cellprofiler_core.object.Objects()

        objects.segmented = labels

        object_set.add_objects(objects, "ObjectProcessing")

        parent_labels = numpy.zeros((30, 30), dtype=numpy.uint8)

        i, j = numpy.mgrid[-15:15, -15:15]
        parent_labels[i ** 2 + j ** 2 <= 196] = 1

        parent_objects = cellprofiler_core.object.Objects()

        parent_objects.segmented = parent_labels

        object_set.add_objects(parent_objects, "Objects")

        workspace = cellprofiler_core.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=None,
            object_set=object_set,
            measurements=measurements,
            image_set_list=None,
        )

        module.add_measurements(workspace)

        expected_center_x = [7.0, 22.0]

        actual_center_x = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_X,
        )

        numpy.testing.assert_array_equal(actual_center_x, expected_center_x)

        expected_center_y = [15.0, 15.0]

        actual_center_y = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Y,
        )

        numpy.testing.assert_array_equal(actual_center_y, expected_center_y)

        expected_center_z = [0.0, 0.0]

        actual_center_z = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Z,
        )

        numpy.testing.assert_array_equal(actual_center_z, expected_center_z)

        expected_object_number = [1.0, 2.0]

        actual_object_number = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_NUMBER_OBJECT_NUMBER,
        )

        numpy.testing.assert_array_equal(actual_object_number, expected_object_number)

        expected_count = [2.0]

        actual_count = measurements.get_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.FF_COUNT % "ObjectProcessing",
        )

        numpy.testing.assert_array_equal(actual_count, expected_count)

        expected_children_count = [2]

        actual_children_count = measurements.get_measurement(
            "Objects",
            cellprofiler_core.constants.measurement.FF_CHILDREN_COUNT
            % "ObjectProcessing",
        )

        numpy.testing.assert_array_equal(actual_children_count, expected_children_count)

        expected_parents = [1, 1]

        actual_parents = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.FF_PARENT % "Objects",
        )

        numpy.testing.assert_array_equal(actual_parents, expected_parents)

    def test_run(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = (
            cellprofiler_core.module.image_segmentation._object_processing.ObjectProcessing()
        )

        module.x_name.value = "Objects"

        object_set = cellprofiler_core.object.ObjectSet()

        parent_objects = cellprofiler_core.object.Objects()

        parent_objects.segmented = numpy.ones((10, 10), dtype=numpy.uint8)

        object_set.add_objects(parent_objects, "Objects")

        workspace = cellprofiler_core.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=None,
            object_set=object_set,
            measurements=measurements,
            image_set_list=None,
        )

        module.function = lambda x: x

        module.run(workspace)

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_X,
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Y,
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_LOCATION_CENTER_Z,
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.M_NUMBER_OBJECT_NUMBER,
        )

        assert measurements.has_feature(
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.FF_COUNT % "ObjectProcessing",
        )

        assert measurements.has_feature(
            "Objects",
            cellprofiler_core.constants.measurement.FF_CHILDREN_COUNT
            % "ObjectProcessing",
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler_core.constants.measurement.FF_PARENT % "Objects",
        )
